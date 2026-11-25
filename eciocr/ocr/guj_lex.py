"""Gujarati name-token correction by word evidence (lang-guj round 3, 2026-11-25).
The Gujarati CRNNs share SYSTEMATIC errors (a vote does not catch them): missing anusvara, an e-/o-matra read as a reph one
cluster early ('અલ્પેશ' -> 'અર્લ્પશ', 'ડામોર' -> 'ડાર્માર'), ફ as ક, ો as ા, ઘ for ધ, ધ્ન for દ, ડ્ત / ત્ત for જ્ઞ, ૃ for ુ.
For each token, candidates are made with up to 2 of these edits; a candidate replaces the token when its word count is >= 5 and
>= 3x the token's count. Word count = guj_lex.json: Gujarati name tokens of class-A pseudo-labels (Tesseract and all CRNNs agree,
non-gold ACs of S06) + Devanagari words of Hindi 2027 text layers (Rajasthan, MP) mapped to the Gujarati block; a token also
counts as its stem without a common name suffix (ભાઈ, બેન, કુમાર ...). Gold r5 (569 boxes): name + relation errors 137 -> 105,
32 fixed, 0 broken (parameters set on gold: recheck on new gold). Caller keeps the field flagged unless a Tesseract read equals
the result. ECI_GUJ_LEX=0 = off."""
import json, os, re

D = os.path.dirname(os.path.abspath(__file__))
CONS = set(chr(c) for c in range(0xA95, 0xAB9 + 1)) - {'઩', '઱', '઴'}
VS = set('ાિીુૂૃેૈોૌ')
ANU, VIR, REPH = 'ં', '્', 'ર્'
SUBS = [('ક', 'ફ'), ('ા', 'ો'), ('ઘ', 'ધ'), ('ધ્નિ', 'દિ'), ('ધ્ન', 'દ'), ('ડ્ત', 'જ્ઞ'), ('ત્ત', 'જ્ઞ'), ('ૃ', 'ુ'), ('ઔ', 'ઓ'),
        ('ય', 'ચ'), ('ધ્', 'દ')]
SUF = ['ભાઈ', 'ભાઇ', 'બેન', 'કુમાર', 'કુમારી', 'લાલ', 'સિંહ', 'સિહ', 'ચંદ્ર', 'જી', 'બા', 'દાસ', 'ગીરી', 'ગિરી', 'રામ', 'દેવી']
WR, WMIN, KEEP = 3, 5, 3
_W = None


def wcount(t):
    global _W
    if _W is None: _W = json.load(open(os.path.join(D, 'guj_lex.json'), encoding='utf-8'))
    c = _W.get(t, 0)
    for s in SUF:
        if t.endswith(s) and len(t) - len(s) >= 2: c = max(c, _W.get(t[:-len(s)], 0))
    return c


def _clusters(t):
    out, i = [], 0
    while i < len(t):
        if t[i] in CONS:
            j = i + 1
            while j + 1 < len(t) and t[j] == VIR and t[j + 1] in CONS: j += 2
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def edits(t):
    out = set()
    for i in range(1, len(t) + 1):                      # insert anusvara
        a = t[i - 1]; b = t[i] if i < len(t) else ''
        if (a in CONS and b not in VS and b != VIR) or a in VS:
            if b not in (ANU, 'ઁ', 'ઃ', VIR): out.add(t[:i] + ANU + t[i:])
    for k in range(len(t) - 1):                         # reph one cluster early -> e / o matra on that cluster
        if t[k:k + 2] == REPH and k + 2 < len(t) and t[k + 2] in CONS:
            i = k + 2; j = i + 1
            while j + 1 < len(t) and t[j] == VIR and t[j + 1] in CONS: j += 2
            rest = t[j:]
            if rest[:1] == 'ા': out.add(t[:k] + t[i:j] + 'ો' + rest[1:])
            elif rest[:1] in VS: out.add(t[:k] + t[i:])
            else: out.add(t[:k] + t[i:j] + 'ે' + rest)
    for a, b in SUBS:
        for m in re.finditer(re.escape(a), t):
            out.add(t[:m.start()] + b + t[m.end():])
    out.discard(t)
    return out


def correct_token(t):
    if len(t) < 2 or not re.fullmatch(r'[઀-૿]+', t): return t
    c0 = wcount(t)
    cands = edits(t)
    if len(t) <= 14: cands |= {c2 for c in list(cands) for c2 in edits(c)}
    cands.discard(t)
    if not cands: return t
    wb = max(sorted(cands), key=wcount); cb = wcount(wb)
    return wb if cb >= WMIN and cb >= WR * max(c0, 1) else t


def correct(v):
    return ' '.join(correct_token(t) for t in v.split(' ')) if v else v
