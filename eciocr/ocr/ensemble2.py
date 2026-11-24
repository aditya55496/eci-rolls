"""Ensemble v2: engines read the 2026 image; the 2027 draft text of the same EPIC only CONFIRMS.

Per field:
  - candidates = readings from each engine (values always come from an engine)
  - if a 2027 record exists for the EPIC: accept the first engine reading that equals the 2027 value
    (age: equal or 1 less); if none equals, flag the box
  - if no 2027 record: accept when engines agree (>= MIN_AGREE) and validators pass; else flag
  - code: presence from the pixel test; text from Tesseract (eng whitelist)
Flagged boxes go to a fallback reader (Haiku). Reports accuracy with and without fallback.
Usage: ensemble2.py ENGINE1 ENGINE2 ... [--fallback claude-haiku-5-5]
"""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from score import load_gold, preds, visual, FIELDS, CORE
from ensemble import valid as valid_hin, EPIC_RE
import parse_eng

B = os.path.expanduser('~/projects/eci-data/bench')
args = [a for a in sys.argv[1:] if not a.startswith('--') and not a.endswith('.json') and not a.replace('.', '').isdigit()]
FB = sys.argv[sys.argv.index('--fallback') + 1] if '--fallback' in sys.argv else None
if FB in args: args.remove(FB)
FB2 = sys.argv[sys.argv.index('--escalate') + 1] if '--escalate' in sys.argv else None
if FB2 in args: args.remove(FB2)
HINT_FIELDS = ['name', 'relation_label', 'relation_name', 'house_no', 'gender']
PREF = {'epic': ['tess'], 'serial_no': ['crnn', 'tess'], 'part_no': ['crnn'], 'house_no': ['crnn', 'surya'], 'age': ['crnn', 'surya'],
        'name': ['surya', 'crnn'], 'relation_name': ['surya', 'crnn'], 'relation_label': ['crnn'], 'gender': ['crnn']}


def load_hints():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    H = {}
    for l in open(f'{B}/hint_2027_v2.jsonl' if os.path.exists(f'{B}/hint_2027_v2.jsonl') else f'{B}/hint_2027.jsonl'):
        r = json.loads(l); H[r['epic']] = r
    return H


LINE_OF = {'name': 'line0', 'relation_label': 'line1', 'relation_name': 'line1', 'house_no': 'line2', 'age': 'line3', 'gender': 'line3'}
CONFW = None  # set by --confvote
# 2026-11-22: an EPIC read by only one engine (CRNNs do not know prefixes such as UWY, DQH outside the training ACs) is
# confirmed when it is in the 2027 roll and that record's name or relative's name is read by an engine. ECI_EPIC_CONFIRM=0 = off.
# years between the roll read and the 2027 roll it is confirmed with (2026 rolls: 1 -> age gap 0 or 1, as benchmarked;
# 2024 roll: 3 -> gap 2 or 3, measured on AC 174 part 1: 735 of 772 voters 3, 22 voters 2). Set by eci_ocr --roll-year.
AGE_GAP = 1
EPIC_CONFIRM = os.environ.get('ECI_EPIC_CONFIRM', '1') != '0'
EPIC_PAIR = os.environ.get('ECI_EPIC_PAIR', '1') != '0'    # rule 1 (2026-11-23), see decide()
CODE_CRNN = os.environ.get('ECI_CODE_CRNN', '1') != '0'    # rule 2 (2026-11-23), see decide()
EPIC_TESS_NOHINT = os.environ.get('ECI_EPIC_TESS_NOHINT', '1') != '0'   # rule 3 (2026-11-25), see decide()
LEX = None  # set by --lexicon: word set from 2027 text layer (non-gold voters only)
# 2026-11-25 (engpipe): language of the roll being read. 'ENG': English validators (Latin names, Male/Female/Third Gender,
# printed relative labels) and the relative label compared with the 2027 hint by kind (Father/Husband/Mother/Other), because the
# 2027 roll of a state can print another form of the label ("Father's Name" vs 'Fathers Name'). Set by eci_ocr per page.
LANG = 'HIN'


def valid(f, v):
    return parse_eng.valid_eng(f, v, EPIC_RE) if LANG == 'ENG' else valid_hin(f, v)


def hvis(f, v):
    """visual form used to compare an engine value with the 2027 hint"""
    if LANG == 'ENG' and f == 'relation_label': return parse_eng.rel_canon(v)
    return visual(v)


# 2026-11-25 (job3): English rolls, hinted box, a hint field (or age) that no engine reads as in the 2027 roll: when every engine
# reads the SAME valid value and the engines include at least one CRNN and Tesseract (independent readers), take it unflagged:
# the print differs from 2027 (new spelling, house number, age) - the same evidence the no-hint rule accepts. ECI_ENG_AGREE=1.
# ECI_ENG_AGREE=2 (held-out check with Opus reads, 12 non-gold S05/S14 parts): agreeing engines were wrong in 12 of 117 boxes, all
# where the value is CLOSE to the 2027 value (1-2 letters: clipped descenders j/y/g -> i/v/q, I/l) or a truncated part of it
# (a wrap line lost); ages and clearly different values were right (82 of 83). Mode 2 takes only ages and values whose
# normalised edit distance to the 2027 value is >= 0.34 and that are not a prefix / suffix of it (or it of them).
ENG_AGREE = int(os.environ.get('ECI_ENG_AGREE', '0') or 0)
ENG_CODE = os.environ.get('ECI_ENG_CODE', '0') != '0'


def agree_eng(f, vals, names, hv='', full=False):
    if not (ENG_AGREE and LANG == 'ENG' and names and len(vals) >= 2): return False
    if not any(n.startswith('crnn') for n in names) or not any(n.startswith('tess') for n in names): return False
    v = [visual(x) for x in vals]
    if not (v[0] and all(x == v[0] for x in v) and valid(f, vals[0])): return False
    if ENG_AGREE >= 2 and f != 'age':
        from rapidfuzz.distance import Levenshtein
        a, b = v[0].lower(), visual(str(hv or '')).lower()
        if not b: return False
        far = not (a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a)) and Levenshtein.normalized_distance(a, b) >= 0.34
        if far: return True
        # ECI_ENG_AGREE=3 (job3 round 2, 2026-11-25): a CLOSE value (or one that contains / is contained in the 2027 value) is
        # taken only when at least TWO CRNNs and Tesseract read it the same (in the cascade the 2nd CRNN reads only the boxes
        # stage 1 flagged), and the reading is not a shorter part of the 2027 value (a lost wrap line or cut word).
        # Round-1 errors of the plain rule: 7 clipped descenders (now fixed by ECI_ENG_DESC), 3 I/l, 2 lost wrap lines.
        if ENG_AGREE < 3 or sum(n.startswith('crnn') for n in names) < 2: return False
        if len(a) < len(b) and (b.startswith(a) or b.endswith(a)):
            # ECI_ENG_AGREE=4 (job3 round 2): the reading is the 2027 value without whole leading / trailing words or tokens
            # ('Soibam Punyakumari' / 2027 'SOIBAM PUNYAKUMARI DEVI', '180/2' / 'H.No-180/2') and the box is complete (every
            # engine read a valid age and gender: the two lost wrap lines of round 1 had lost the age line too)
            cut = b[len(a)] if b.startswith(a) else b[len(b) - len(a) - 1]
            if not (ENG_AGREE >= 4 and full and cut in ' -/,.'): return False
        # I / l / 1 / | look the same in the roll font: a reading that differs from 2027 only there is not evidence (round 2
        # held-out Opus check: 'Flat no. 1/El-2' read by all three engines, 2027 and Opus 'EI-2')
        x, y = v[0], visual(str(hv or ''))
        if any(op == 'replace' and x[i] in 'Il1|' and y[j] in 'Il1|' for op, i, j in Levenshtein.editops(x, y)): return False
    return True


# ECI_ENG_TIL=1 (job3 round 2, 2026-11-25): English field without 2027 confirmation: >= 2 CRNNs read the same value and every
# Tesseract reading differs from it only in I / l / 1 / | (substituted, added or dropped): Tesseract eng reads a capital I as l
# ('lrungbam', 'MaqbooI', 'Alli' for Ali) - the CRNNs agreeing is the evidence.
ENG_TIL = os.environ.get('ECI_ENG_TIL', '0') != '0'


def til_ok(vals, names):
    from rapidfuzz.distance import Levenshtein
    cr = [visual(v) for v, nm in zip(vals, names) if nm.startswith('crnn')]
    ts = [visual(v) for v, nm in zip(vals, names) if not nm.startswith('crnn')]
    if len(cr) < 2 or not cr[0] or any(c != cr[0] for c in cr) or not ts: return False
    for t in ts:
        for op, i, j in Levenshtein.editops(cr[0], t):
            ch = (cr[0][i] if op != 'insert' else '') + (t[j] if op != 'delete' else '')
            if not ch or any(c not in 'Il1|' for c in ch): return False
    return True


NOHINT_SLACK = int(sys.argv[sys.argv.index('--slack') + 1]) if '--slack' in sys.argv else 0
# job2 2026-11-25 (no 2027 roll): house_no and age are not flagged only because Tesseract disagrees when ALL CRNNs read the same
# valid value (house_no: with a digit and without . , ; : - the CRNNs confuse those marks). Tesseract misreads these lines (it
# caused ~11,000 of ~11,400 house_no flags on the no-hint gold runs). Set on a non-gold validation set (14 UP ACs outside every
# gold set). ECI_NOHINT_DIGITS=1 = on (default off).
NOHINT_DIGITS = os.environ.get('ECI_NOHINT_DIGITS', '0') != '0'
AGE_HINT_MAJ = os.environ.get('ECI_AGE_HINT_MAJ', '0') == '1'   # 2026-11-25 ocrv5, see decide() (default off: UP output unchanged)
# job2 2026-11-25 (ECI_EPIC_FMT=1, default off): (a) the CRNNs have no backslash and read the old-format separator '\\' as '1':
# a reading 'UP' [+ one separator] + 14 digits with '1' at digit 3 and digit 7 is rewritten to UP\\dd\\ddd\\ddddddd (field flagged);
# (b) the majority EPIC is not valid but the two first CRNNs agree on a valid one -> take it (field flagged). With the relaxed
# EPIC_RE of ensemble.py. Found on GE gold v3 (3 EPIC errors, all flagged); no EPIC error of the 2026 gold / HC is of these kinds.
EPIC_FMT = os.environ.get('ECI_EPIC_FMT', '0') == '1'
# job2 2026-11-25 (ECI_NOHINT_NAMES=K, default 0 = off): no 2027 roll: name / relation_name not flagged only because Tesseract
# disagrees when at least K CRNNs run and ALL of them read the same valid value (meant for K = 5 with an independent 5th CRNN;
# with the 4 KD students the CRNN name errors are correlated: round 1 rule N, rejected).
NOHINT_NAMES = int(os.environ.get('ECI_NOHINT_NAMES', '0') or 0)
# job2 r3 2026-11-25 (ECI_NOHINT_CLOSED=1, default off): no 2027 roll: gender and relation_label (closed value sets) are not
# flagged only because Tesseract disagrees when ALL CRNNs read the same valid value. Flag study of the no-hint runs: GE gold 267
# gender + 183 relation_label flags were of this kind (Tesseract misreads the line). Checked first on the non-gold val set.
NOHINT_CLOSED = os.environ.get('ECI_NOHINT_CLOSED', '0') != '0'
# job2 r3 2026-11-25 (ECI_NOHINT_TESSCLEAN=bits, default 0 = off): no 2027 roll, name / relation_name: when ALL CRNNs (at least
# 4) read the same valid value and Tesseract differs from it ONLY by Tesseract noise, the field is not flagged. Bit 1 (always on
# when set): marks and junk that never belong to a printed name (stress signs U+0951 / U+0952, ! |, Devanagari digits), a leaked
# label prefix ('पति का नाम! ...'), a doubled virama ('नन््' = 'न्'), extra spaces. Bit 2: Tesseract dropped u-matra signs only.
# Bit 4: Tesseract inserted anusvara signs only. Bit 8: Tesseract inserted nukta signs only (REJECTED in sim: +2 G26 silent
# errors, printed nukta in फ़ूल...). Bit 16: Tesseract inserted subscript ra only. Bit 32: Tesseract inserted virama only. Flag study r3 (j2Is outputs): single-edit Tesseract differences on GE: anusvara inserted 637,
# virama 194, nukta 184, u-matra dropped 91; val / G26: the 'नन््' artifact, u-matra dropped.
NOHINT_TESSCLEAN = int(os.environ.get('ECI_NOHINT_TESSCLEAN', '0') or 0)
_TJUNK = re.compile('[\u0951\u0952!|\u0966-\u096f]')
_TLABEL = re.compile(r'^(?:.{0,8}का\s*नाम\s*[:;।|.]?|नाम\s*[:;।|.]+)\s*')


def _tess_clean(t, level):
    t = _TJUNK.sub('', t)
    t = _TLABEL.sub('', t) if 'नाम' in t else t
    t = re.sub('(.)\\1\u094d\u094d', '\\1\u094d', t).replace('\u094d\u094d', '\u094d')  # 'नन््' -> 'न्' (half-letter glyph artifact)
    return ' '.join(t.split())


def _only_ops(a, b, drop, ins):
    """True when b = a with only chars of `drop` deleted and chars of `ins` inserted (DP over both strings)."""
    m = len(b)
    row = [True] + [False] * m
    for j in range(1, m + 1): row[j] = row[j - 1] and b[j - 1] in ins
    for ca in a:
        new = [row[0] and ca in drop] + [False] * m
        for j in range(1, m + 1):
            new[j] = (row[j - 1] and ca == b[j - 1]) or (row[j] and ca in drop) or (new[j - 1] and b[j - 1] in ins)
        row = new
    return row[m]


def _tess_noise_only(crnn, tess, level):
    """level bits: 1 = junk / label / glyph-artifact cleaning (always on), 2 = Tesseract dropped u-matra, 4 = Tesseract
    inserted anusvara, 8 = Tesseract inserted nukta"""
    a, b = visual(crnn), visual(_tess_clean(tess or '', level))
    if a == b: return True
    drop = '\u0941' if level & 2 else ''
    ins = ('\u0902' if level & 4 else '') + ('\u093c' if level & 8 else '') + ('\u094d' if level & 32 else '')
    if level & 16 and '\u094dर' in b:  # bit 16: Tesseract inserted a subscript ra ('्र') only
        b2 = b.replace('\u094dर', '')
        if a.replace('\u094dर', '') == b2 and b.count('\u094dर') > a.count('\u094dर'): return True
    return bool(drop or ins) and _only_ops(a, b, drop, ins)


# job5 2026-11-25 (ECI_PAREN_GUARD=1, default off): the CRNN charsets have no '(' ')'. Himachal prints house numbers such as
# 13(1), 56(2)2, 57(2)क; all CRNNs read ')' as '1' with one voice (13/11, 56/212) and the record was not flagged (6 silent
# errors on the job5 SSR gold). When any engine (Tesseract) reads a bracket in house_no and the chosen value has none, the
# field is flagged.
PAREN_GUARD = os.environ.get('ECI_PAREN_GUARD', '0') == '1'
_EPIC_BS = re.compile(r'^UP[/\\]?([0-9]{2})1([0-9]{3})1([0-9]{7})$')


# job5 2026-11-25 (ECI_EPIC_STATE=1, default off): old-format EPICs ('BR/02/007/123010') start with the state's 2-letter code.
# The CRNNs never saw BR / HR / HP and read them as 8R, HP, HA, HF. Each engine's reading of an old-format EPIC whose 2-letter
# part differs from the state's code in at most 1 character gets the state's code (set per PDF by eci_ocr from the file name,
# EPIC_STATE). Codes checked on job5 gold prints: S04 BR, S07 HR, S08 HP, S24 UP; S12 MP (own code) and S26 Chhattisgarh MP
# (MP/021/163/642044, ocrv5 local test). Other states: no change (code unknown).
EPIC_STATE_ON = os.environ.get('ECI_EPIC_STATE', '0') == '1'
STATE_EPIC_CODE = {'S04': 'BR', 'S07': 'HR', 'S08': 'HP', 'S24': 'UP', 'S12': 'MP', 'S26': 'MP'}
EPIC_STATE = None
_EPIC_OLD = re.compile(r'^(..)([/\\])([0-9]{1,3})\2([0-9]{2,3})\2([0-9]{6,7})$')   # Haryana: HR/01/07/0279513 (2-digit part)
_EPIC_CONF = {frozenset(p) for p in ('8B', '0D', '1I', '5S', '2Z', 'PR')}   # digit/letter shapes the CRNNs mix up


def _epic_state(v):
    m = _EPIC_OLD.match(v or '')
    if not m or not EPIC_STATE or m.group(1) == EPIC_STATE: return v
    if sum(a != b and frozenset(a + b) not in _EPIC_CONF for a, b in zip(m.group(1), EPIC_STATE)) > 1: return v
    return EPIC_STATE + v[2:]


def _epic_bs(v):
    m = _EPIC_BS.match(v or '')
    return 'UP\\%s\\%s\\%s' % m.groups() if m else v


# job2 r3 (ECI_NOHINT_HOUSE_PUNCT=1, default off, with NOHINT_DIGITS): house_no with . , ; : (excluded by the DIGITS rule because
# the CRNNs confuse those marks) is not flagged when all CRNNs agree AND every other engine (Tesseract) has the same sequence of
# these marks (Tesseract drops digits such as the 1 in 'बी.15' -> 'बी.5', but reads the marks).
# job2 r3 (ECI_NOHINT_SLACK1=field,field, default empty): no 2027 roll: in these fields one dissenting CRNN is allowed when all
# other engines (Tesseract included) read the same valid value (>= 5 engines).
# job2 r3 (ECI_NOHINT_AMP=1, default off): no 2027 roll, flagged house_no: when Tesseract read an '&' and at least one CRNN read
# a value with '&', the value is the most common CRNN reading with '&' (the CRNNs trained with hin_ms_2027 lines read the printed
# '&' as '8': round 2 finding, GE '10190&' read '101908'). The field stays flagged.
NOHINT_AMP = os.environ.get('ECI_NOHINT_AMP', '0') != '0'
# job2 r4 2026-11-25 (ECI_CONFW_SKIP=field,field, default empty): no 2027 roll: the confidence-weighted vote (--confvote) does not
# re-choose the value of these flagged fields (plain majority of decide() stays). Sim on j2Z: house_no without CONFW GE house
# 99.22 -> 99.29 (the CRNN confidences of house lines are 0.5-0.7 for right reads too, round 1).
CONFW_SKIP = set(x for x in os.environ.get('ECI_CONFW_SKIP', '').split(',') if x)
# job2 r4 2026-11-25 (ECI_NOHINT_LEX=path.npz, default off): word lexicon of the 2027 UP text layers of 281 ACs outside every gold
# set and the job2 val ACs (name + relation_name words, visual form), stored as sorted 64-bit blake2b hashes: key kN = words seen
# >= N times. ECI_NOHINT_LEXG=N (no 2027 roll): an UNFLAGGED name / relation_name with a word seen < N times gets a flag (all
# engines can read a rare name the same wrong way: HC जाफ़्रीन read जाफ़ीन by all 5). ECI_NOHINT_NUNF=K (eci_ocr._valset): a
# flagged name / relation_name is cleared when every CRNN (voters + VALSET extras) read the same value, every word of it is seen
# >= K times, and Tesseract differs by more than marks (nukta, anusvara, candrabindu, virama, reph, candra vowels).
NOHINT_LEXG = int(os.environ.get('ECI_NOHINT_LEXG', '0') or 0)
NOHINT_NUNF = int(os.environ.get('ECI_NOHINT_NUNF', '0') or 0)
_LEXH = {}


def _whash(w):
    import hashlib
    return int.from_bytes(hashlib.blake2b(w.encode('utf-8'), digest_size=8).digest(), 'little')


def lex_count_ge(word, n):
    """True when the (visual-form) word is seen >= n times in the ECI_NOHINT_LEX lexicon (key 'k<n>')."""
    if n not in _LEXH:
        import numpy as np
        _LEXH[n] = np.load(os.environ['ECI_NOHINT_LEX'])[f'k{n}']
    a = _LEXH[n]; h = _whash(word)
    import numpy as np
    i = int(np.searchsorted(a, np.uint64(h)))
    return i < len(a) and int(a[i]) == h


# job5 r3 2026-11-25 (ECI_NOHINT_MARKVAR=FILE.npz, default unset = off; no 2027 roll): the CRNNs drop printed marks with one
# voice (nukta: अरोड़ा read अरोडा by all 7 CRNNs of config j5r3G, Tesseract wrong elsewhere -> NUNF un-flagged it; anusvara:
# दहनीं read दहनी). The name un-flag paths (TESSCLEAN, NUNF) are not taken when a word of the value has a mark variant (nukta,
# anusvara or candrabindu added / removed / swapped) that the lexicon saw MORE often than the word. FILE: arrays h (sorted
# blake2b-64 of the visual word) and c (count) of the 2027 UP text words (job2 lexicon, non-gold ACs) that have such a variant
# in the lexicon (job5 r3/build_markvar.py). Gold sim (j5r3G): 4 more flagged records of 1,283, 1 silent error caught.
_MV = {}
_MV_NUK = 'डढजफकखग'
_MV_VS = set('\u093e\u093f\u0940\u0941\u0942\u0947\u0948\u094b\u094c')


def mark_variants(w):
    out = set()
    for i, ch in enumerate(w):
        if ch in ('\u093c', '\u0902', '\u0901'): out.add(w[:i] + w[i + 1:])
        if ch == '\u0902': out.add(w[:i] + '\u0901' + w[i + 1:])
        if ch == '\u0901': out.add(w[:i] + '\u0902' + w[i + 1:])
        nxt = w[i + 1] if i + 1 < len(w) else ''
        if ch in _MV_NUK and nxt != '\u093c': out.add(w[:i + 1] + '\u093c' + w[i + 1:])
        if (ch in _MV_VS or '\u0915' <= ch <= '\u0939') and nxt not in ('\u0902', '\u0901', '\u093c', '\u094d') and nxt not in _MV_VS:
            out.add(w[:i + 1] + '\u0902' + w[i + 1:])
    out.discard(w)
    return {visual(v) for v in out}


def _mv_count(w):
    import numpy as np
    if 'h' not in _MV:
        z = np.load(os.environ['ECI_NOHINT_MARKVAR']); _MV['h'], _MV['c'] = z['h'], z['c']
    h = _whash(w); i = int(np.searchsorted(_MV['h'], np.uint64(h)))
    return int(_MV['c'][i]) if i < len(_MV['h']) and int(_MV['h'][i]) == h else 0


def markvar_suspect(val):
    """True when ECI_NOHINT_MARKVAR is set and a word of val has a mark variant seen more often than the word."""
    if not os.environ.get('ECI_NOHINT_MARKVAR'): return False
    for w in visual(val or '').split():
        c = _mv_count(w)
        if any(_mv_count(v) > max(c, 1) for v in mark_variants(w)): return True
    return False


def marks_strip(x):
    return re.sub('[\u093c\u0902\u0901\u094d\u0945]', '', visual(x or '').replace('र्', '').replace('\u0949', '\u093e')).replace(' ', '')


def amp_pick(vals, names):
    if not names or not any('&' in (v or '') for v, nm in zip(vals, names) if not nm.startswith('crnn')): return None
    amp = [v for v, nm in zip(vals, names) if nm.startswith('crnn') and '&' in (v or '')]
    return collections.Counter(amp).most_common(1)[0][0] if amp else None


SLACK1_F = set(filter(None, os.environ.get('ECI_NOHINT_SLACK1', '').split(',')))
HOUSE_PUNCT = os.environ.get('ECI_NOHINT_HOUSE_PUNCT', '0') != '0'


def _house_punct_agree(vals, vis, names):
    cr = [i for i, nm in enumerate(names or []) if nm.startswith('crnn')]
    if len(cr) < 2 or len({vis[i] for i in cr}) != 1: return None
    v = vals[cr[0]]
    if not v or not any(c.isdigit() for c in v): return None
    pv = [c for c in v if c in '.,;:']
    for i, nm in enumerate(names):
        if i not in cr and [c for c in (vals[i] or '') if c in '.,;:'] != pv: return None
    return v


def _crnn_all_agree(f, vals, vis, names):
    cr = [i for i, nm in enumerate(names or []) if nm.startswith('crnn')]
    if len(cr) < 2 or len({vis[i] for i in cr}) != 1: return None
    v = vals[cr[0]]
    if not v or not valid(f, v): return None
    if f == 'house_no' and (not any(c.isdigit() for c in v) or any(c in v for c in '.,;:')): return None
    return v


def decide(cands, hint, pix, names=None, confs=None, stamp=None):
    out, flags = {}, []
    full = LANG == 'ENG' and all(valid('age', c['age']) and valid('gender', c['gender']) for c in cands)
    for f in FIELDS:
        vals = [c[f] for c in cands]
        vis = [visual(v) for v in vals]
        if f == 'code':
            tv = [v for v, nm in zip(vals, names or [''] * len(vals)) if nm.startswith('tess')] or vals
            out[f] = (next((v for v in tv if v), '#') if pix else '')
            if pix and not any(tv): flags.append(f)
            # 2026-11-23 (rule 2): Tesseract misses letter codes under the DELETED stamp ('' -> default '#'). When the pixel test
            # says a code is printed, Tesseract gives nothing or only '#', and the first CRNN reads a letter code, take that;
            # the field stays flagged. gold_v3 (GE 2024): 20 of 82 stamped boxes had S/E/R read as '#'. Only for boxes with a
            # DELETED stamp (stamp=True from segment.LAST_STAMP): on unstamped boxes the CRNN reads real '#' codes as R/E
            # (gold_v3 measured run: 4 such errors without this condition).
            # job3 (ECI_ENG_CODE=1): English rolls have no stamp test; when the pixel test says a code is printed and NO Tesseract
            # reading gave anything (not even '#'), the first CRNN's letter code is taken (the field stays flagged)
            eng_code = ENG_CODE and LANG == 'ENG' and pix and all(t == '' for t in tv)
            if ((CODE_CRNN and stamp) or eng_code) and pix and all(t in ('', '#') for t in tv) and names:
                j = next((i for i, nm in enumerate(names) if nm.startswith('crnn')), None)
                if j is not None and re.fullmatch(r'[SERQM][0-9]?', vals[j] or ''):
                    out[f] = vals[j]
                    if f not in flags: flags.append(f)
            continue
        if f in ('serial_no', 'part_no', 'epic'):
            fmt_flag = False
            if f == 'epic' and EPIC_STATE_ON and EPIC_STATE:
                vals = [_epic_state(v) for v in vals]; vis = [visual(v) for v in vals]
            if f == 'epic' and EPIC_FMT:
                v2 = [_epic_bs(v) for v in vals]
                if v2 != vals:
                    fmt_flag = True; vals = v2; vis = [visual(v) for v in vals]
            c = collections.Counter(vis); top, n = c.most_common(1)[0]
            out[f] = vals[vis.index(top)]
            if n * 2 <= len(vals) and names:
                cr = [i for i, nm in enumerate(names) if nm.startswith('crnn')][:2]
                if (f == 'epic' and EPIC_PAIR and len(cr) == 2 and vals[cr[0]] and visual(vals[cr[0]]) == visual(vals[cr[1]])
                        and valid('epic', vals[cr[0]])):
                    # 2026-11-23 (rule 1): no majority, but the two strongest CRNNs (c11, c9) agree on a valid EPIC: take it
                    # instead of the Tesseract fallback (c4/c3 cannot read most letters, so a right c11+c9 pair lost to one
                    # Tesseract misread such as SISA.../ZWL00...; gold_v3: 22 boxes)
                    out[f] = vals[cr[0]]
                else:
                    for p in PREF.get(f, []):
                        j = next((i for i, nm in enumerate(names) if nm.startswith(p)), None)
                        if j is not None and vals[j]: out[f] = vals[j]; break
            if f == 'epic' and EPIC_CONFIRM and hint and hint.get('epic') and visual(hint['epic']) in vis:
                # an engine read exactly this voter's EPIC from the 2027 roll (that is how the record was found): take it, even
                # when more engines agree on another prefix (CRNNs turn prefixes they never saw into training prefixes;
                # flag check 2026-11-23: 14 of 28 unflagged EPIC errors in 500 flagged records were of this kind)
                out[f] = hint['epic']
                continue
            if f == 'epic' and EPIC_FMT and not valid(f, out[f]) and names:
                cr = [i for i, nm in enumerate(names) if nm.startswith('crnn')][:2]
                if len(cr) == 2 and vals[cr[0]] and visual(vals[cr[0]]) == visual(vals[cr[1]]) and valid(f, vals[cr[0]]):
                    out[f] = vals[cr[0]]; fmt_flag = True
            if f in CORE and (n < 2 or not valid(f, out[f]) or fmt_flag): flags.append(f)
            elif f == 'epic' and hint is None and EPIC_TESS_NOHINT and names and any(nm.startswith('tess') for nm in names) \
                    and visual(vals[next(i for i, nm in enumerate(names) if nm.startswith('tess'))]) != visual(out[f]):
                # 2026-11-25 (rule 3): without a 2027 record nothing confirms the EPIC. The CRNNs know only the EPIC prefixes of
                # their training ACs and turn others into them with one voice (eye check of 4 boxes per part: Jharkhand KDF/KKD ->
                # XGF in 5 of 8, Delhi ZIP -> ZSY in 3 of 4, West Bengal IBN -> BLN in 1 of 4, all unflagged; Tesseract had
                # all 9 right). Tesseract reads letters as printed:
                # when it disagrees with the chosen EPIC, flag (value unchanged: taking Tesseract's value when only the prefix
                # differs fixed 9 of 9 such boxes of other states but cost UP bench gold without hints 95.19 -> 95.15).
                # Bench gold (UP, 4,266 boxes): with hints unchanged (97.80 / 99.632 / 4.57 %); without hints flagged
                # 60.97 -> 61.18 %, unflagged 99.700 -> 99.698 %, record 95.19 unchanged.
                flags.append(f)
            continue
        if hint and f in HINT_FIELDS and hint.get(f):
            h = hvis(f, hint[f])
            m = next((v for v in vals if hvis(f, v) == h), None)
            if m is not None: out[f] = m; continue
            if agree_eng(f, vals, names, hint[f], full): out[f] = vals[0]; continue
            out[f] = collections.Counter(vals).most_common(1)[0][0]
            if names and collections.Counter(vals).most_common(1)[0][1] * 2 <= len(vals):
                for p in PREF.get(f, []):
                    j = next((i for i, nm in enumerate(names) if nm.startswith(p)), None)
                    if j is not None and vals[j]: out[f] = vals[j]; break
            flags.append(f); continue  # no engine matches 2027: flag
        if hint and f == 'age' and (hint.get('age') or '').isdigit():   # a hint without age has None
            ok = [v for v in vals if v.isdigit() and int(hint['age']) - int(v) in (AGE_GAP - 1, AGE_GAP)]
            if ok and AGE_HINT_MAJ and names:
                # 2026-11-25 (ocrv5 review, ECI_AGE_HINT_MAJ=1): an age in the 2027 window was taken even when only Tesseract
                # read it (S20 AC 14 part 70 final roll, page 7 box 14: printed 21, c11 c9 c4 c3 all '21', Tesseract '24',
                # 2027 age 24 -> 24 without a flag). Now the window value is taken without a flag only when a CRNN read it AND
                # no strict majority of the readings is outside the window; else the field is flagged (value: the outside
                # majority when there is one, else the window value). In the cascade's stage 1 (c11 + Tesseract) a
                # Tesseract-only match is flagged, so the box gets all engines.
                maj, m = collections.Counter(v for v in vals if v).most_common(1)[0] if any(vals) else ('', 0)
                crnn_ok = any(v in ok for v, nm in zip(vals, names) if nm.startswith('crnn'))
                if m * 2 > len(vals) and maj not in ok:
                    out[f] = maj; flags.append(f); continue
                out[f] = collections.Counter(ok).most_common(1)[0][0]
                if not crnn_ok: flags.append(f)
                continue
            if ok: out[f] = collections.Counter(ok).most_common(1)[0][0]; continue
            if agree_eng(f, vals, names, hint['age']): out[f] = vals[0]; continue
            j = next((i for i, nm in enumerate(names or []) if nm.startswith('crnn')), 0)
            out[f] = vals[j]; flags.append(f); continue
        c = collections.Counter(vis); top, n = c.most_common(1)[0]
        out[f] = vals[vis.index(top)]
        if n * 2 <= len(vals) and names:  # no majority: engine chosen per field type (fixed rule, not tuned on gold)
            pref = PREF.get(f, [])
            for p in pref:
                j = next((i for i, nm in enumerate(names) if nm.startswith(p)), None)
                if j is not None and vals[j]: out[f] = vals[j]; break
        if PAREN_GUARD and f == 'house_no' and any(b in v for v in vals for b in '()') and not any(b in out[f] for b in '()'):
            flags.append(f); continue
        need = max(2, len(vals) - NOHINT_SLACK) if hint is None else len(vals)  # no confirmation: (almost) all engines must agree
        if f in CORE and n < need and ENG_TIL and LANG == 'ENG' and names and til_ok(vals, names):
            out[f] = vals[next(i for i, nm in enumerate(names) if nm.startswith('crnn'))]; n = need
        if hint is None and f in SLACK1_F and n == len(vals) - 1 and len(vals) >= 5 and names and valid(f, out[f]):
            odd = [nm for nm, x in zip(names, vis) if x != top]
            if odd and odd[0].startswith('crnn'): need = n  # one CRNN alone dissents, Tesseract with the majority
        if hint is None and ((NOHINT_DIGITS and f in ('house_no', 'age')) or (NOHINT_CLOSED and f in ('gender', 'relation_label'))) \
                and n < need:
            v = _crnn_all_agree(f, vals, vis, names)
            if v is None and HOUSE_PUNCT and f == 'house_no':
                v = _house_punct_agree(vals, vis, names)
            if v is not None:
                out[f] = v; continue
        if hint is None and NOHINT_NAMES and f in ('name', 'relation_name') and n < need \
                and sum(nm.startswith('crnn') for nm in (names or [])) >= NOHINT_NAMES:
            v = _crnn_all_agree(f, vals, vis, names)
            if v is not None:
                out[f] = v; continue
        if hint is None and NOHINT_TESSCLEAN and f in ('name', 'relation_name') and n < need and names \
                and sum(nm.startswith('crnn') for nm in names) >= 4:
            v = _crnn_all_agree(f, vals, vis, names)
            others = [vals[i] for i, nm in enumerate(names) if not nm.startswith('crnn')]
            if v is not None and all(_tess_noise_only(v, t, NOHINT_TESSCLEAN) for t in others) and not markvar_suspect(v):
                out[f] = v; continue
        if f in CORE and (n < need or not valid(f, out[f])): flags.append(f)
    if LEX is not None:
        for f in flags:
            if f not in ('name', 'relation_name'): continue
            good = [c[f] for c in cands if c[f] and all(w in LEX for w in visual(c[f]).split())]
            if good:
                out[f] = collections.Counter(good).most_common(1)[0][0]
        return out, flags
    if CONFW is not None and confs:
        for f in flags:
            if f not in LINE_OF: continue
            if hint is None and f in CONFW_SKIP: continue
            w = collections.Counter()
            for c, cf, nm in zip(cands, confs, names or []):
                v = c[f]
                if not v: continue
                if ENG_CODE and LANG == 'ENG' and not valid(f, v): continue   # job3: an invalid English value (gender '33 Gee..Female') never wins the vote
                w[visual(v)] += (cf.get(LINE_OF[f], CONFW) if nm.startswith('crnn') else CONFW)
            if w:
                top = w.most_common(1)[0][0]
                out[f] = next(c[f] for c in cands if visual(c[f]) == top)
    if hint is None and NOHINT_AMP and 'house_no' in flags:  # after the CONFW vote (production default 0.6)
        v = amp_pick([c['house_no'] for c in cands], names)
        if v is not None: out['house_no'] = v
    if hint is None and NOHINT_LEXG:
        for f in ('name', 'relation_name'):
            if f not in flags and out.get(f) and not all(lex_count_ge(w, NOHINT_LEXG) for w in visual(out[f]).split()):
                flags.append(f)
    return out, flags


def fallback_fix(out, flags, fb, cands, hint):
    """Replace only flagged fields. Prefer the fallback value when it agrees with an engine or 2027;
    if it agrees with neither, still take it (it is the only other reading of the image)."""
    fixed = dict(out)
    for f in flags:
        if f == 'code':
            fixed['code'] = fb.get('code') or out['code']; continue
        v = fb.get(f, out[f])
        fixed[f] = v
    return fixed


if __name__ == '__main__':
    G = load_gold(); Ps = [preds(c) for c in args]; H = load_hints()
    if '--lexicon' in sys.argv:
        gold_e = {g['fields']['epic'] for g in G.values()}
        LEX = collections.Counter()
        for l in open(f'{B}/hint_2027_v2.jsonl'):
            r = json.loads(l)
            if r['epic'] in gold_e: continue
            for f in ('name', 'relation_name'): LEX.update(visual(r[f]).split())
        LEX = {w for w, n in LEX.items() if n >= int(sys.argv[sys.argv.index('--lexicon') + 1])}
    if '--nohint' in sys.argv: H = {}
    if '--confvote' in sys.argv:
        CONFW = float(sys.argv[sys.argv.index('--confvote') + 1])
    RAW = [{(r['page'], r['box']): r.get('scores', {}) for r in map(json.loads, open(f'{B}/ocr/{c}.jsonl'))} for c in args]
    PIX = json.load(open(f'{B}/code_pixel.json'))
    F = {}; F2 = {}
    if FB:
        for l in open(f'{B}/fallback_reads.jsonl'):
            r = json.loads(l)
            p = r['cid'].split('__')
            if r['model'] == FB and 'out' in r: F[(p[1], int(p[2][1:]))] = r['out']
            if r['model'] == FB2 and 'out' in r: F2[(p[1], int(p[2][1:]))] = r['out']
    keys = [k for k in G if all(k in P for P in Ps)]
    if '--sample' in sys.argv: keys = [k for k in keys if k in F]
    DUMP = []
    st = collections.Counter(); fe = collections.Counter(); bysub = collections.defaultdict(collections.Counter)
    for k in keys:
        cands = [P[k] for P in Ps]
        epic = collections.Counter(c['epic'] for c in cands).most_common(1)[0][0]
        out, flags = decide(cands, H.get(epic), PIX[f'{k[0]}|{k[1]}'], args, [R.get(k, {}) for R in RAW])
        g = G[k]['fields']
        good = lambda o: all(visual(o[f]) == visual(g[f]) for f in CORE + ['code'])
        ok = good(out)
        sub = 'SUP' if G[k]['kind'] == 'SUP' else ('115' if k[0].startswith('SIR_86-SIR-FinalRoll') else '144')
        st['n'] += 1; st['ok'] += ok; st['flag'] += bool(flags); bysub[sub]['n'] += 1; bysub[sub]['ok'] += ok; bysub[sub]['flag'] += bool(flags)
        if not flags: st['unflag'] += 1; st['unflag_ok'] += ok
        else:
            for f in flags: fe[f] += 1
        if not ok and not flags: st['silent_err'] += 1
        if '--dump' in sys.argv: DUMP.append(dict(page=k[0], box=k[1], ok=bool(ok), flagged=bool(flags), sub=sub))
        # fallback: flagged boxes re-read (only boxes in the 1,000 sample have a fallback read)
        st['final_ok'] += ok if not flags else 0
        if flags:
            st['flag_in_fb'] += k in F
            fb = F.get(k)
            if fb is not None:
                fixed = fallback_fix(out, flags, fb, cands, H.get(epic))
                if FB2:
                    esc = [f for f in flags if f != 'code' and visual(fb.get(f, '')) not in {visual(c[f]) for c in cands} | ({visual(H[epic].get(f, ''))} if H.get(epic) and f in HINT_FIELDS + ['age'] else set())]
                    if esc and k in F2:
                        st['escalated'] += 1
                        for f in esc: fixed[f] = F2[k].get(f, fixed[f])
                st['fb_ok'] += good(fixed); st['final_ok'] += good(fixed)
    n = st['n']
    res = dict(engines=args, boxes=n, record_no_fallback=round(100 * st['ok'] / n, 2), flagged_pct=round(100 * st['flag'] / n, 2),
               unflagged_acc=round(100 * st['unflag_ok'] / max(1, st['unflag']), 3), silent_errors=st['silent_err'],
               perfect_fallback_acc=round(100 * (st['unflag_ok'] + st['flag']) / n, 3), flag_fields=dict(fe.most_common()),
               subsets={s: dict(acc=round(100 * c['ok'] / c['n'], 2), flagged=round(100 * c['flag'] / c['n'], 2)) for s, c in bysub.items()})
    if FB and '--sample' in sys.argv: res['final_with_fallback'] = round(100 * st['final_ok'] / n, 2)
    if FB: res['fallback'] = dict(model=FB, escalate=FB2, escalated=st['escalated'], flagged_with_read=st['flag_in_fb'], fixed_ok=st['fb_ok'],
                                  fix_rate=round(100 * st['fb_ok'] / max(1, st['flag_in_fb']), 1))
    if '--dump' in sys.argv: json.dump(DUMP, open(sys.argv[sys.argv.index('--dump') + 1], 'w'))
    print(json.dumps(res, ensure_ascii=False))
