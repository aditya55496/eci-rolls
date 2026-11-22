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


def _label(h):
    if h.get('relation_label') is not None:
        return h.get('relation_label')
    rt = h.get('relation_type') or ''
    return (rt + ' का नाम') if rt and rt != 'अन्य' else rt


def fix2(recs, idx, age_gap=1):
    """Second pass (flag check 2026-11-23): in flagged records the names are often the misread fields, so the name rule above
    misses. A record whose EPIC is still not in the 2027 roll gets the 2027 EPIC when exactly ONE 2027 voter has the same last
    7 characters AND the same gender AND an age gap of (age_gap - 1 .. age_gap) years AND the same house number or relation
    label. On 902 hand-checked boxes: 6 changed, 6 right, 0 wrong."""
    epics, by = idx
    n = 0
    for r in recs:
        e = r.get('epic') or ''
        if len(e) < 7 or e in epics or r.get('epic_from_2027'):
            continue
        cands = by.get(e[-7:], [])
        if len(cands) != 1:
            continue
        h = cands[0]
        try:
            gap = int(h.get('age')) - int(r.get('age'))
        except (TypeError, ValueError):
            continue
        same_house = _visual(h.get('house_no')) == _visual(r.get('house_no'))
        same_label = bool(_label(h)) and _visual(_label(h)) == _visual(r.get('relation_label'))
        if _visual(h.get('gender')) == _visual(r.get('gender')) and gap in (age_gap - 1, age_gap) and (same_house or same_label):
            r['epic_ocr'], r['epic'], r['epic_from_2027'] = e, h['epic'], True
            if 'epic' in (r.get('flags') or []):
                r['flags'] = [f for f in r['flags'] if f != 'epic']
            n += 1
    return n


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
