"""lang-ben r4 (2026-11-25): CTC mark-variant margin of a Bengali CRNN line reading.

A CRNN with a strong language prior writes the common spelling ('চ্যাটার্জী', 'মুন্না') where the print has a rare one
('চ্যাটাজী', 'মন্না'); two such models agree, so the no-hint vote does not flag it. The CTC output still holds the evidence:
margin = max over one-mark variants v of the value part [log P(v | x) - log P(greedy | x)] (exact CTC forward, all variants of
a line in one batched pass). A margin near 0 means the image fits a reading with one mark more / less / swapped about as well.

Variants (value part = text after the first ':' if the line has one, else the whole line), per Bengali token:
  parse_ben.mark_variants (u/uu/ri-kar insert or swap, reph insert, chandrabindu insert, nukta insert) plus deletions of
  u/uu/ri-kar, chandrabindu, nukta, reph, hasanta; i-kar <-> ii-kar; mid-word ত্C <-> ৎC and word-final ত <-> ত্ <-> ৎ.
"""
import re
import unicodedata

import numpy as np

import parse_ben

_DEL = set('ুূৃঁ়্')
_KT_NEXT = set('পকফলশসখগঘমবনদ')


def variants(tok):
    out = set(parse_ben.mark_variants(tok))
    for i, c in enumerate(tok):
        if c in _DEL and i > 0:
            out.add(tok[:i] + tok[i + 1:])
        if c == 'র' and i + 2 < len(tok) and tok[i + 1] == '্' and tok[i + 2] in parse_ben._CONS:
            out.add(tok[:i] + tok[i + 2:])
        if c in 'িী':
            out.add(tok[:i] + ('ী' if c == 'ি' else 'ি') + tok[i + 1:])
        if c == 'ত' and tok[i + 1:i + 2] == '্' and (i + 2 == len(tok) or tok[i + 2] in _KT_NEXT):
            out.add(tok[:i] + 'ৎ' + tok[i + 2:])
        if c == 'ৎ':
            out.add(tok[:i] + 'ত্' + tok[i + 1:])
            if i + 1 == len(tok): out.add(tok[:i] + 'ত' + tok[i + 1:])
        if c == 'ত' and i + 1 == len(tok) and i > 0:
            out.add(tok + '্'); out.add(tok[:i] + 'ৎ')
    out = {unicodedata.normalize('NFC', v) for v in out}
    out.discard(tok)
    return out


def line_variants(text):
    """-> list of full-line variant strings (one edit in one token of the value part)"""
    k = text.find(':')
    start = k + 1 if k >= 0 else 0
    out = []
    for m in re.finditer(r'[ঀ-৿]+', text):
        if m.start() < start: continue
        for v in variants(m.group(0)):
            out.append(text[:m.start()] + v + text[m.end():])
    return out


def ctc_loglik(logP, labels):
    """logP: (T, C) log-probs (blank 0); labels: list of label index lists. -> np.array of log-likelihoods (batched forward)."""
    T = logP.shape[0]
    V = len(labels)
    L = max(len(l) for l in labels) * 2 + 1
    ext = np.zeros((V, L), np.int64)          # blank-interleaved label sequences, padded with blank
    lens = np.zeros(V, np.int64)
    skip = np.zeros((V, L), bool)             # transition s-2 -> s allowed
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


def margin(P, CH, text, cap=60):
    """P: (T, C) softmax output of one line (blank 0, CH = charset without blank); text = greedy reading.
    -> (margin, best variant) ; margin = -inf when there is no variant. cap: at most this many variants (nearest first)."""
    vs = line_variants(text)
    if not vs: return float('-inf'), None
    idx = {c: i + 1 for i, c in enumerate(CH)}
    labs, keep = [], []
    for s in [text] + vs[:cap]:
        try:
            labs.append([idx[c] for c in s]); keep.append(s)
        except KeyError:
            continue
    if len(keep) < 2 or keep[0] != text: return float('-inf'), None
    ll = ctc_loglik(np.log(np.maximum(P, 1e-12)), labs)
    j = int(np.argmax(ll[1:])) + 1
    return float(ll[j] - ll[0]), keep[j]
