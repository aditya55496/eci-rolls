"""EPIC repair from the 2027 roll (2026-11-22).

The CRNNs were trained on 3 ACs and turn EPIC prefixes they never saw into familiar ones (ZCQ1412691 -> ZSY1412691, XFT3242062 ->
XGF3242062); with 3-4 CRNNs agreeing, the vote takes the wrong prefix without a flag. Found by the hand check of 402 boxes from
112 other ACs: 8 EPIC errors, 5 of them unflagged. In all 8 the 2027 text-layer roll of the AC has the voter with the right EPIC,
the same 7 digits and the same name.

Rule: a record whose EPIC is not in the AC's 2027 roll gets the 2027 EPIC with the same last 7 characters when exactly one such
2027 record has the same name or relative's name (visual comparison as in the benchmark). The OCR value is kept in
'epic_ocr' and the record gets 'epic_from_2027': true. Nothing else changes.
"""
import collections
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


_DIGITS = str.maketrans({'I': '1', 'l': '1', 'O': '0', 'o': '0'})
DRAFT_FIELDS = ('epic', 'name', 'relation_label', 'relation_name', 'house_no', 'age', 'gender')


def final_deleted(recs):
    """SIR final roll (2026-11-23): voters deleted after the draft keep their box in the final roll, with status code Q and a
    "DELETED" stamp across the text. The SIR models never saw the stamp: they misread the box, it gets >= 6 flags and the
    not_voter_box rule takes it for page noise, so it falls out of the counts (AC 37 part 344: 30 of 727 voters).
    The same voter, same serial, is in the draft roll of the part without a stamp. A final-roll record marked not_voter_box
    whose serial is a draft voter's serial AND whose EPIC ends in the same 6 digits (misread I/O mapped to 1/0) becomes a voter
    record with the draft's fields: 'deleted_in_final': True, 'from_draft': True, the final roll's own readings in 'final_ocr'.
    recs: all records of one part file (draft and final). Returns the number of records changed."""
    draft = {}
    for r in recs:
        if 'DraftRoll' in (r.get('source') or '') and not r.get('not_voter_box') and str(r.get('serial_no') or '').isdigit():
            draft[r['serial_no']] = r
    n = 0
    for r in recs:
        if 'FinalRoll' not in (r.get('source') or '') or not r.get('not_voter_box'):
            continue
        d = draft.get(str(r.get('serial_no') or ''))
        e = (r.get('epic') or '').translate(_DIGITS)
        if not d or len(e) < 6 or not d.get('epic') or e[-6:] != d['epic'][-6:]:
            continue
        r['final_ocr'] = {f: r.get(f) for f in DRAFT_FIELDS + ('flags',)}
        for f in DRAFT_FIELDS:
            r[f] = d.get(f)
        r['flags'] = list(d.get('flags') or [])
        # the status code (deletion reason, e.g. Q / S / E) is only in the final roll; under the stamp the vote is often
        # undecided ('#'): take the letter when c11 and c9 agree (as CODE_CRNN for GE 2024), the field stays flagged
        rc = (r.get('readings') or {}).get('code') or {}
        a, b = rc.get('crnn_c11q'), rc.get('crnn_c9q')
        if r.get('code') in ('#', '', None) and a and a == b and a.isalpha():
            r['code'] = a
            r['flags'].append('code')
        r['deleted_in_final'] = r['from_draft'] = True
        r.pop('not_voter_box', None)
        n += 1
    return n


def serial_gaps(recs):
    """A box marked not_voter_box whose serial fills a gap in its PDF's serial sequence is a voter box (2026-11-23).
    The not_voter_box rule (EPIC not valid AND >= 6 flags) is meant for garbage on cover / map / photo pages, but it also took
    real voter boxes: deleted voters under a DELETED stamp, often with old-format EPICs such as UP/9/041/0303522 (GE 2024:
    AC 111 part 192, all 10 missing serials). Per PDF (record 'source'): voters' serials 1..last, where last is the highest
    voter serial not above the number of boxes + 10 (one far-out misread such as 11001 must not open a huge range). A marked box
    whose serial is in 1..last and not yet taken becomes a voter: 'voter_by_serial': True, flags kept. Returns the number changed."""
    by = {}
    for r in recs:
        by.setdefault(r.get('source'), []).append(r)
    n = 0
    for rs in by.values():
        cap = len(rs) + 10
        have = {int(r['serial_no']) for r in rs if not r.get('not_voter_box') and str(r.get('serial_no') or '').isdigit()
                and int(r['serial_no']) <= cap}
        if not have:
            continue
        last = max(have)
        for r in rs:
            s = str(r.get('serial_no') or '')
            if r.get('not_voter_box') and s.isdigit() and 1 <= int(s) <= last and int(s) not in have:
                r.pop('not_voter_box', None)
                r['voter_by_serial'] = True
                have.add(int(s))
                n += 1
    return n


PFX_RE = re.compile(r'^([A-Z]{3})([0-9]{7})$')


def prefix_check(recs, min_frequent=10, max_rare=2):
    """No 2027 roll (job2, 2026-11-25): EPIC prefixes of one roll come from a few series. An EPIC whose prefix is rare in this
    PDF (<= max_rare boxes) and one letter away from a frequent prefix (>= max(min_frequent, 2 % of boxes)) is suspect
    (NQU4279394 read for NOU4279394). If an engine read the frequent-prefix form (flagged boxes keep readings), take it; the
    field is flagged either way. Returns (taken, flagged)."""
    C = collections.Counter(m.group(1) for r in recs for m in [PFX_RE.match(r.get('epic') or '')] if m)
    freq = {p for p, n in C.items() if n >= max(min_frequent, 0.02 * len(recs))}
    max_rare = max(max_rare, int(0.005 * len(recs)))
    taken = flagged = 0
    for r in recs:
        m = PFX_RE.match(r.get('epic') or '')
        if not m or m.group(1) in freq or C[m.group(1)] > max_rare: continue
        near = [p for p in freq if sum(a != b for a, b in zip(p, m.group(1))) == 1]
        if not near: continue
        rd = (r.get('readings') or {}).get('epic') or {}
        alt = [v for k, v in rd.items() if isinstance(v, str) and (PFX_RE.match(v) or [None])[0] and v[:3] in near and v[3:] == m.group(2)]
        if alt:
            r['epic'] = alt[0]; taken += 1; r['epic_by_prefix'] = True
        if 'epic' not in r['flags']:
            r['flags'].append('epic'); flagged += 1
            r.setdefault('readings', {}).setdefault('epic', {})['prefix_check'] = m.group(1)
    return taken, flagged
