"""Combine engines per field, run validators, flag boxes for fallback.

Usage: ensemble.py CFG1 CFG2 [CFG3 ...]   (first config = tie-break priority)
Reports: accuracy of the vote, share of boxes flagged, accuracy of unflagged boxes,
and accuracy if flagged boxes were fixed perfectly by a fallback reader.
"""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from score import load_gold, preds, visual, FIELDS, CORE

EPIC_RE = re.compile(r'^([A-Z]{3}[0-9]{7}|[A-Z]{2}/[0-9]{2}/[0-9]{3}/[0-9]{6,7})$')
# job2 2026-11-25 (ECI_EPIC_FMT=1, default off): old-format EPICs also with a 1-digit district (UP/9/027/0132288: 37 right
# boxes in GE gold v3, all flagged before only for the format) and with backslashes (UP\\58\\283\\0723298: printed in 2 GE boxes)
if os.environ.get('ECI_EPIC_FMT', '0') == '1':
    EPIC_RE = re.compile(r'^([A-Z]{3}[0-9]{7}|[A-Z]{2}/[0-9]{1,2}/[0-9]{3}/[0-9]{6,7}|[A-Z]{2}\\[0-9]{1,2}\\[0-9]{3}\\[0-9]{6,7})$')
# ocrv5 2026-11-25 (ECI_EPIC_OLDFMT=1, default off): old-format EPICs of other states, 2 letters / 1-3 digits / 2-3 digits /
# 6-7 digits (Haryana HR/01/01/0000369: 184 of 918 boxes of S07 AC 1 part 1, all flagged only for the format; Chhattisgarh
# MP/021/163/642044). A superset of the UP old format; off in the adopted sets v4a and j2Ps (not gold-checked).
if os.environ.get('ECI_EPIC_OLDFMT', '0') == '1':
    EPIC_RE = re.compile(EPIC_RE.pattern[:-2] + r'|[A-Z]{2}/[0-9]{1,3}/[0-9]{2,3}/[0-9]{6,7})$')
LABELS = {'पिता का नाम', 'पति का नाम', 'माता का नाम', 'अन्य'}
GENDERS = {'पुरुष', 'महिला', 'तृतीय लिंग'}


def valid(f, v):
    if f == 'epic': return bool(EPIC_RE.match(v))
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in GENDERS
    if f == 'relation_label': return v in LABELS
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'): return bool(v) and not re.search(r'[A-Za-z|;]', v)
    return True


def combine(cands):
    """cands: list of per-engine field dicts (priority order). Returns fields, flags."""
    out, flags = {}, []
    for f in FIELDS:
        vals = [c[f] for c in cands if c is not None]
        vis = collections.Counter(visual(v) for v in vals)
        top, n = vis.most_common(1)[0] if vis else ('', 0)
        pick = next((v for v in vals if visual(v) == top), '')
        if n < len(vals) and n <= len(vals) / 2:  # no majority: take priority engine
            pick = vals[0]
        if f in CORE and (n < len(vals) or not valid(f, pick)): flags.append(f)
        out[f] = pick
    return out, flags


if __name__ == '__main__':
    cfgs = sys.argv[1:]
    G = load_gold(); Ps = [preds(c) for c in cfgs]
    PIX = json.load(open(os.path.expanduser('~/projects/eci-data/bench/code_pixel.json')))
    keys = [k for k in G if all(k in P for P in Ps)]
    # serial order check inside each page
    stats = collections.Counter()
    res = {}
    for k in keys:
        out, flags = combine([P[k] for P in Ps])
        if not PIX[f'{k[0]}|{k[1]}']: out['code'] = ''           # no ink left of the serial: no code
        elif not out['code']: out['code'] = '#'; flags.append('code')  # ink but no code read: most common code, flag
        res[k] = (out, flags)
    by_page = collections.defaultdict(list)
    for k in keys: by_page[k[0]].append(k)
    for page, ks in by_page.items():
        ks.sort(key=lambda k: k[1])
        for a, b in zip(ks, ks[1:]):
            sa, sb = res[a][0]['serial_no'], res[b][0]['serial_no']
            if sa.isdigit() and sb.isdigit() and int(sb) != int(sa) + 1 and G[b]['kind'] == 'SIR':
                if 'serial_no' not in res[b][1]: res[b][1].append('serial_no')
    for k in keys:
        out, flags = res[k]; g = G[k]['fields']
        ok = all(visual(out[f]) == visual(g[f]) for f in CORE + ['code'])
        stats['n'] += 1; stats['ok'] += ok; stats['flagged'] += bool(flags)
        if not flags: stats['unflagged'] += 1; stats['unflagged_ok'] += ok
        else: stats['flagged_wrong'] += not ok
        for f in CORE: stats[f'f_{f}'] += visual(out[f]) == visual(g[f])
    n = stats['n']
    print(json.dumps({'configs': cfgs, 'boxes': n,
                      'vote_record_acc': round(100 * stats['ok'] / n, 2),
                      'flagged_pct': round(100 * stats['flagged'] / n, 2),
                      'unflagged_record_acc': round(100 * stats['unflagged_ok'] / max(1, stats['unflagged']), 3),
                      'errors_caught_pct': round(100 * stats['flagged_wrong'] / max(1, n - stats['ok']), 1),
                      'acc_if_flagged_fixed': round(100 * (stats['unflagged_ok'] + stats['flagged']) / n, 3),
                      **{f: round(100 * stats[f'f_{f}'] / n, 2) for f in CORE}}, ensure_ascii=False))
