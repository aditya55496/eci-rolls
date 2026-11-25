"""lang-pan round 2 (2026-11-25): CTC fit of the 2027 hint value on the CRNN output of the same line (ECI_PAN_HINTCTC).

With 2027 hints, a field is accepted only when an engine reads exactly the 2027 value; else the field is flagged and the
engine value is kept. On the Punjabi gold most such fields are fine-mark misses of the CRNNs (u-sign of 'ਕੁਮਾਰ' read 'ਕਮਾਰ',
subjoined 'ਰ' of 'ਮਹੇਂਦ੍ਰੂ', halant of 'ਫ੍ਲ'): the image holds the mark, the greedy path drops it.
margin = log P(line with the 2027 value | x) - log P(line with the engine value | x), exact CTC forward on the softmax output.
A margin near 0 means the image fits the 2027 value about as well as the greedy reading.
"""
import re
import unicodedata

import numpy as np


def gur_norm(t):
    """= train/pan_crnn.py gur_norm (the CRNN label form) + two 2027 text-layer repairs"""
    t = unicodedata.normalize('NFC', t).replace('‌', '').replace('‍', '')
    t = re.sub(r'([਼੍ੰੱਂਾ-ੌ])\1+', r'\1', t)
    t = re.sub('ਰ੍(?=[ਕ-ਮਲਲ਼ਸ਼ਸਖ਼-ਫ਼])', 'ਰ', t)
    # 2027 text-layer artefacts that print as one sign (held-out check 2026-11-25: 'ਸੰਜੁੂ' prints 'ਸੰਜੂ', 'ਡੇੈਨੀਅਲ' prints 'ਡੈਨੀਅਲ')
    t = t.replace('ੁੂ', 'ੂ').replace('ੇੈ', 'ੈ')
    return re.sub(r'\s+', ' ', t).strip()


def ctc_loglik(logP, labels):
    """logP: (T, C) log-probs (blank 0); labels: list of label index lists -> np.array of log-likelihoods (batched forward)."""
    T = logP.shape[0]
    V = len(labels)
    L = max(len(l) for l in labels) * 2 + 1
    ext = np.zeros((V, L), np.int64)
    lens = np.zeros(V, np.int64)
    skip = np.zeros((V, L), bool)
    for v, l in enumerate(labels):
        e = [0]
        for c in l: e += [c, 0]
        ext[v, :len(e)] = e; lens[v] = len(e)
        for s in range(2, len(e)):
            skip[v, s] = e[s] != 0 and e[s] != e[s - 2]
    NEG = -1e30
    a = np.full((V, L), NEG)
    a[:, 0] = logP[0, 0]
    a[:, 1] = logP[0, ext[:, 1]]
    a[lens < 2, 1] = NEG
    for t in range(1, T):
        e = logP[t, ext]
        a1 = np.concatenate([np.full((V, 1), NEG), a[:, :-1]], 1)
        a2 = np.concatenate([np.full((V, 2), NEG), a[:, :-2]], 1)
        a2 = np.where(skip, a2, NEG)
        m = np.maximum(np.maximum(a, a1), a2)
        a = m + np.log(np.exp(a - m) + np.exp(a1 - m) + np.exp(a2 - m)) + e
    idx = np.arange(V)
    last = a[idx, lens - 1]
    prev = np.where(lens >= 2, a[idx, np.maximum(lens - 2, 0)], NEG)
    m = np.maximum(last, prev)
    return m + np.log(np.exp(last - m) + np.exp(prev - m))


def hint_line(raw, value, hint):
    """raw = CRNN line text, value = the field value parsed from it, hint = 2027 value (label form).
    -> (reference line, hint line) or None. The value must be the end of the raw line (name / relative / house lines)."""
    r = raw.rstrip()
    if value and r.endswith(value):
        pre = r[:len(r) - len(value)]
    elif ':' in r:
        pre = r[:r.index(':') + 1] + ' '
        r = pre + value
    else:
        return None
    return r, pre + hint


def margin(P, CH, raw, value, hint):
    """P: (T, C) softmax of the line (blank 0); CH: charset without blank. -> margin (float) or None (no comparison)."""
    hl = hint_line(raw, value, hint)
    if hl is None: return None
    idx = {c: i + 1 for i, c in enumerate(CH)}
    try:
        labs = [[idx[c] for c in s] for s in hl]
    except KeyError:
        return None
    if not labs[0] or not labs[1]: return None
    ll = ctc_loglik(np.log(np.maximum(P, 1e-12)), labs)
    return float(ll[1] - ll[0])
