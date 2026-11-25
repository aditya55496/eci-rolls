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
import parse_ben

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
LEX = None  # set by --lexicon: word set from 2027 text layer (non-gold voters only)
# 2026-11-25 (engpipe): language of the roll being read. 'ENG': English validators (Latin names, Male/Female/Third Gender,
# printed relative labels) and the relative label compared with the 2027 hint by kind (Father/Husband/Mother/Other), because the
# 2027 roll of a state can print another form of the label ("Father's Name" vs 'Fathers Name'). Set by eci_ocr per page.
LANG = 'HIN'


def valid(f, v):
    if LANG == 'BEN': return parse_ben.valid_ben(f, v, EPIC_RE)
    return parse_eng.valid_eng(f, v, EPIC_RE) if LANG == 'ENG' else valid_hin(f, v)


def hvis(f, v):
    """visual form used to compare an engine value with the 2027 hint"""
    if LANG == 'ENG' and f == 'relation_label': return parse_eng.rel_canon(v)
    if LANG == 'BEN' and f == 'relation_label': return parse_ben.rel_canon(v)
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


def agree_eng(f, vals, names, hv=''):
    if not (ENG_AGREE and LANG == 'ENG' and names and len(vals) >= 2): return False
    if not any(n.startswith('crnn') for n in names) or not any(n.startswith('tess') for n in names): return False
    v = [visual(x) for x in vals]
    if not (v[0] and all(x == v[0] for x in v) and valid(f, vals[0])): return False
    if ENG_AGREE >= 2 and f != 'age':
        from rapidfuzz.distance import Levenshtein
        a, b = v[0].lower(), visual(str(hv or '')).lower()
        if not b or a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a): return False
        if Levenshtein.normalized_distance(a, b) < 0.34: return False
    return True


NOHINT_SLACK = int(sys.argv[sys.argv.index('--slack') + 1]) if '--slack' in sys.argv else 0


def decide(cands, hint, pix, names=None, confs=None, stamp=None):
    out, flags = {}, []
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
            if f in CORE and (n < 2 or not valid(f, out[f])): flags.append(f)
            continue
        if hint and f in HINT_FIELDS and hint.get(f):
            h = hvis(f, hint[f])
            m = next((v for v in vals if hvis(f, v) == h), None)
            if m is not None: out[f] = m; continue
            if agree_eng(f, vals, names, hint[f]): out[f] = vals[0]; continue
            out[f] = collections.Counter(vals).most_common(1)[0][0]
            if names and collections.Counter(vals).most_common(1)[0][1] * 2 <= len(vals):
                for p in PREF.get(f, []):
                    j = next((i for i, nm in enumerate(names) if nm.startswith(p)), None)
                    if j is not None and vals[j]: out[f] = vals[j]; break
            flags.append(f); continue  # no engine matches 2027: flag
        if hint and f == 'age' and hint.get('age', '').isdigit():
            ok = [v for v in vals if v.isdigit() and int(hint['age']) - int(v) in (AGE_GAP - 1, AGE_GAP)]
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
        need = max(2, len(vals) - NOHINT_SLACK) if hint is None else len(vals)  # no confirmation: (almost) all engines must agree
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
            w = collections.Counter()
            for c, cf, nm in zip(cands, confs, names or []):
                v = c[f]
                if not v: continue
                if ENG_CODE and LANG == 'ENG' and not valid(f, v): continue   # job3: an invalid English value (gender '33 Gee..Female') never wins the vote
                w[visual(v)] += (cf.get(LINE_OF[f], CONFW) if nm.startswith('crnn') else CONFW)
            if w:
                top = w.most_common(1)[0][0]
                out[f] = next(c[f] for c in cands if visual(c[f]) == top)
    return out, flags


# ---------------------------------------------------------------- job4 (2026-11-25): English rolls WITHOUT a 2027 roll
# Measured on the job4 dev gold (440 boxes of 11 states, non-gold; rules chosen there, then tested once on the English gold):
# requiring ALL engines to agree (decide() without a hint) flags 24-38 % of English boxes, because Tesseract misreads names /
# houses that the English CRNNs read right. decide_eng_nohint():
#   stage 1 (stage1=True, the first 2 CRNNs, trained on different data): a field is accepted when both CRNNs read the same
#            valid value. A box with every field accepted is done; otherwise stage 2.
#   stage 2: accepted when all CRNNs agree on a valid value and a Tesseract reading equals it; or when more than half of the
#            CRNNs and a Tesseract reading agree on a valid value. Else flagged (value: majority of all engines, CRNN order on
#            ties). An EPIC / serial flagged only for missing Tesseract support is settled per PDF (eci_ocr: EPIC prefix
#            seen in >= 3 other records of the PDF; serial: sequence_confirm).
#   plausibility (both stages): gender Male with a Husband label / Female with a Wife label, age outside 18..120 -> flag.
NOHINT_SUPPORT = {}   # last decide_eng_nohint call: {field: 'crnn_unanimous'} for fields flagged only for missing Tesseract support


def decide_eng_nohint(cands, names, pix, stamp=None, stage1=False):
    global NOHINT_SUPPORT
    NOHINT_SUPPORT = {}
    out, flags = {}, []
    ci = [i for i, nm in enumerate(names) if nm.startswith('crnn')]
    ti = [i for i, nm in enumerate(names) if nm.startswith('tess')]
    if stage1: ci = ci[:int(os.environ.get('ECI_STAGE1_N', '2'))]
    for f in FIELDS:
        vals = [c[f] for c in cands]
        if f == 'code':
            tv = [vals[i] for i in ti] or vals
            out[f] = (next((v for v in tv if v), '#') if pix else '')
            if pix and not any(tv): flags.append(f)
            continue
        cv = [visual(vals[i]) for i in ci]; tvv = [visual(vals[i]) for i in ti]
        allc = collections.Counter(v for v in cv + tvv if v)
        if allc:
            top_all = max(allc, key=lambda v: (allc[v], -(cv.index(v) if v in cv else 99)))
            out[f] = next(vals[i] for i in ci + ti if visual(vals[i]) == top_all)
        else:
            out[f] = vals[ci[0]] if ci else ''
        c = collections.Counter(cv); top, n = c.most_common(1)[0] if cv else ('', 0)
        if f == 'part_no' and not top: continue
        ok = False
        if n == len(cv) and cv and valid(f, next(vals[i] for i in ci if visual(vals[i]) == top)):
            if stage1: ok = len(cv) >= 2
            elif top in tvv: ok = True
            else: NOHINT_SUPPORT[f] = 'crnn_unanimous'
            if ok or f in NOHINT_SUPPORT: out[f] = next(vals[i] for i in ci if visual(vals[i]) == top)
        elif not stage1 and n * 2 > len(cv) and top in tvv and valid(f, next(vals[i] for i in ci if visual(vals[i]) == top)):
            ok = True; out[f] = next(vals[i] for i in ci if visual(vals[i]) == top)
        if not ok and f in CORE: flags.append(f)
    lab = (out.get('relation_label') or '').lower()
    if LANG == 'BEN':   # lang-ben: Bengali labels / genders -> the English words of the rule
        lab = parse_ben.rel_canon(out.get('relation_label') or '')
        g = {'পুরুষ': 'Male', 'মহিলা': 'Female'}.get(out.get('gender'), '')
    else:
        g = out.get('gender')
    if (lab.startswith('husband') and g == 'Male') or (lab.startswith('wife') and g == 'Female'):
        if 'gender' not in flags: flags.append('gender')
    if (out.get('age') or '').isdigit() and not 18 <= int(out['age']) <= 120 and 'age' not in flags: flags.append('age')
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
