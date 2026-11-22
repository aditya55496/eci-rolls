"""EPIC repair from the 2027 roll (2026-11-22).

The CRNNs were trained on 3 ACs and turn EPIC prefixes they never saw into familiar ones (ZCQ1412691 -> ZSY1412691, XFT3242062 ->
XGF3242062); with 3-4 CRNNs agreeing, the vote takes the wrong prefix without a flag. Found by the hand check of 402 boxes from
112 other ACs: 8 EPIC errors, 5 of them unflagged. In all 8 the 2027 text-layer roll of the AC has the voter with the right EPIC,
the same 7 digits and the same name.

Rule: a record whose EPIC is not in the AC's 2027 roll gets the 2027 EPIC with the same last 7 characters when exactly one such
2027 record has the same name or relative's name (visual comparison as in the benchmark). The OCR value is kept in
'epic_ocr' and the record gets 'epic_from_2027': true. Nothing else changes.
"""
import re
import unicodedata


def _visual(s):
    s = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(s or ''))).strip().replace('‌', '').replace('‍', '')
    s = s.replace('ॉं', 'ाँ').replace('ऎ', 'ऐ').replace('ोे', 'ो')
    return re.sub('्(?=[ा-ौ])', '', s)


def index(hints):
    """hints: iterable of 2027 records (keys epic, name, relation_name). -> (set of EPICs, {last 7 chars: [records]})."""
    epics, by = set(), {}
    for h in hints:
        e = h.get('epic')
        if not e:
            continue
        epics.add(e)
        by.setdefault(e[-7:], []).append(h)
    return epics, by


def fix(recs, idx):
    """Repair EPICs in place. Returns the number of records changed."""
    epics, by = idx
    n = 0
    for r in recs:
        e = r.get('epic') or ''
        if len(e) < 7 or e in epics:
            continue
        cands = by.get(e[-7:], [])
        nm, rn = _visual(r.get('name')), _visual(r.get('relation_name'))
        m = [h for h in cands if (nm and _visual(h.get('name')) == nm) or (rn and _visual(h.get('relation_name')) == rn)]
        if len({h['epic'] for h in m}) == 1:
            r['epic_ocr'], r['epic'], r['epic_from_2027'] = e, m[0]['epic'], True
            if 'epic' in (r.get('flags') or []):
                r['flags'] = [f for f in r['flags'] if f != 'epic']
            n += 1
    return n
