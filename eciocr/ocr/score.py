"""Score engine outputs against gold. Usage: score.py CONFIG [CONFIG...] (or 'all')."""
import collections, glob, json, os, re, sys, unicodedata
sys.path.insert(0, os.path.dirname(__file__))
from parse import from_lines, from_box_lines
from rapidfuzz.distance import Levenshtein

B = os.path.expanduser('~/projects/eci-data/bench')
FIELDS = ['code', 'serial_no', 'part_no', 'epic', 'name', 'relation_label', 'relation_name', 'house_no', 'age', 'gender']
CORE = ['serial_no', 'epic', 'name', 'relation_label', 'relation_name', 'house_no', 'age', 'gender']


def strict(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', s or '')).strip()


def visual(s):  # sequences that render the same
    s = strict(s).replace('‌', '').replace('‍', '')
    s = s.replace('ॉं', 'ाँ').replace('ऎ', 'ऐ').replace('ोे', 'ो')
    s = re.sub('्(?=[ा-ौ])', '', s)
    return s


def load_gold():
    return {(g['page'], g['box']): g for g in map(json.loads, open(f'{B}/gold.jsonl'))}


def preds(cfg, pp=True):
    out = {}
    for l in open(f'{B}/ocr/{cfg}.jsonl'):
        r = json.loads(l)
        out[(r['page'], r['box'])] = from_lines(r['fields'], pp) if 'fields' in r else from_box_lines(r['lines'], pp=pp)
    return out


def score(P, G, keys=None):
    if keys is None: keys = list(G)
    if not keys: return {"record": None, "n": 0}
    c = collections.Counter(); ed = collections.Counter(); ln = collections.Counter()
    for k in keys:
        g = G[k]; p = P.get(k)
        rec_ok = True
        for f in FIELDS:
            gv = g['fields'][f]; pv = p[f] if p else ''
            s_ok = strict(pv) == strict(gv); v_ok = visual(pv) == visual(gv)
            c[(f, 'strict')] += s_ok; c[(f, 'visual')] += v_ok
            if f in CORE and not v_ok: rec_ok = False
            if f in ('name', 'relation_name', 'house_no'):
                ed[f] += Levenshtein.distance(visual(pv), visual(gv)); ln[f] += max(1, len(visual(gv)))
        c['record_visual'] += rec_ok
    n = len(keys)
    res = {f: round(100 * c[(f, 'visual')] / n, 2) for f in FIELDS}
    res.update({f'{f}_strict': round(100 * c[(f, 'strict')] / n, 2) for f in ('name', 'relation_name', 'house_no')})
    res.update({f'CER_{f}': round(100 * ed[f] / ln[f], 3) for f in ed})
    res['record'] = round(100 * c['record_visual'] / n, 2); res['n'] = n
    return res


def subsets(G):
    s = {'all': list(G)}
    s['SIR_144dpi'] = [k for k, g in G.items() if g['kind'] == 'SIR' and not k[0].startswith('SIR_86-SIR-FinalRoll')]
    s['SIR_115dpi'] = [k for k, g in G.items() if k[0].startswith('SIR_86-SIR-FinalRoll')]
    s['SUP'] = [k for k, g in G.items() if g['kind'] == 'SUP']
    s['coded'] = [k for k, g in G.items() if g['fields']['code']]
    return s


if __name__ == '__main__':
    G = load_gold(); S = subsets(G)
    cfgs = sys.argv[1:]
    if cfgs == ['all']: cfgs = sorted(os.path.basename(f)[:-6] for f in glob.glob(f'{B}/ocr/*.jsonl'))
    rows = []
    for cfg in cfgs:
        P = preds(cfg)
        tel = json.load(open(f'{B}/ocr/{cfg}.telemetry.json')) if os.path.exists(f'{B}/ocr/{cfg}.telemetry.json') else {}
        r = dict(config=cfg, boxes=len(P), **{k: score(P, G, [x for x in v if x in P])['record'] for k, v in S.items()},
                 **{k: v for k, v in score(P, G, [x for x in G if x in P]).items() if k not in ('record', 'n')},
                 s_per_page=tel.get('s_per_page_30'), peak_mb=tel.get('peak_rss_mb'))
        rows.append(r); print(json.dumps(r, ensure_ascii=False))
    json.dump(rows, open(f'{B}/scores_latest.json', 'w'), ensure_ascii=False, indent=1)
