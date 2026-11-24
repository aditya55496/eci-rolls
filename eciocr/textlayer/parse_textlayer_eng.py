#!/usr/bin/env python3
"""Parse ENGLISH "SSR DraftRoll 2027" electoral roll PDFs (text layer) to JSONL, and to --hint-2027 records for eci_ocr.

States with an English 2027 draft roll (checked 2026-11-25): Goa S05, Manipur S14, Tamil Nadu S22 (English edition),
Lakshadweep U06, Puducherry U07 (and Andaman U01, not in the local sample). One JSON record per voter box. No OCR.

Box layout (same 186 x 75 pt drawing as the Hindi roll, parse_textlayer.find_boxes):
    <serial>                                   <EPIC>
    Name : <name>                              Photo
    <Fathers|Husbands|Mothers> Name: <rel>     Available
    (or) Others: <rel>
    House Number : <house>
    Age : <age> Gender : <Male|Female|Third Gender>
A long name or relative name wraps to an unlabelled next row: it is joined to the field before it (warning wrapped_line).
The text layer has none of the Devanagari /ActualText artifacts; each labelled row is still compared with the glyph-level
text (page.get_texttrace()) and a difference is reported in 'text_warnings'.

Output record (like parse_textlayer.py): ac, part, page, section_no, section_name, serial_no, epic, name, relation_type
(Father / Husband / Mother / Others / ... the word before 'Name'), relation_label (the printed label without the colon, e.g.
'Fathers Name'), relation_name, house_no, age, gender, status_code, extra_text, box_marks, text_warnings.

Validation per PDF: record count and gender counts against the cover totals ("NUMBER OF ELECTORS": start / end serial, male,
female, third gender, total) and against the last row of the summary page; serials 1..N; EPIC, age and gender formats.

Usage
  parse_textlayer_eng.py PDF [PDF ...] [-o OUTDIR] [--summary FILE] [--hints FILE]
  Writes OUTDIR/<pdf stem>.jsonl; --hints writes all records of all PDFs in the eci_ocr --hint-2027 form (hint_of_eng).
"""
import argparse, collections, json, os, re, sys, time, unicodedata

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_textlayer import (find_boxes, page_chars, assign, rows_of, row_text, page_lines, box_marks,   # noqa: E402
                             EPIC_NEW, EPIC_OLD, EPIC_ANY)

GENDERS = {'Male', 'Female', 'Third Gender'}
PHOTO_TEXT = 'Photo'
NAME_RE = re.compile(r'^Name\s*:\s*(.*)$')
# 'Fathers Name:', "Father's Name:", 'Husbands Name:', 'Mothers Name:', 'Others:' (and any other '<word> Name:' a state prints)
REL_RE = re.compile(r"^(?:([A-Za-z][A-Za-z'’]*)\s+Name|(Others?))\s*:\s*(.*)$")
HOUSE_RE = re.compile(r'^House\s+(?:Number|No\.?)\s*:\s*(.*)$')
AGE_RE = re.compile(r'^Age\s*:\s*(\S*)\s*Gender\s*:\s*(.*)$')
EPIC_LOOSE = re.compile(r'^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9/]{8,}$')   # printed so: 'A140254078', 'FQ11664630', 'gdw0042770'
SECTION_RE = re.compile(r'Section No\.? and Name\s*:\s*(\d+)\s*-\s*(.*)')


def clean(s):
    s = unicodedata.normalize('NFC', s or '').replace(' ', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def letters(s):
    return collections.Counter(c for c in unicodedata.normalize('NFC', s) if not c.isspace())


# --------------------------------------------------------------------------- cover / summary

def parse_cover(page):
    txt = page.get_text()
    meta = {}
    m = re.search(r'Assembly Constituency\s*:[\s:]*(\d+)\s*-\s*([^\n(]+?)\s*\(([^)\n]*)\)', txt)
    if m: meta.update(ac_no=int(m.group(1)), ac_name=clean(m.group(2)), ac_reservation=clean(m.group(3)))
    m = re.search(r'Parliamentary Constituency.*?:\s*(\d+)\s*-\s*([^(]+?)\s*\(', txt, re.S)
    if m: meta.update(pc_no=int(m.group(1)), pc_name=clean(m.group(2)))
    m = re.search(r'ELECTORAL ROLL\s+\d{4}\s+([SU]\d\d)\s+([^\n]+)', txt)
    if m: meta.update(state_code=m.group(1), state=clean(m.group(2)))
    m = re.search(r'Part No\.?\s*:[\s:]*(\d+)', txt)
    if m: meta['part_no'] = int(m.group(1))
    # totals: the first row of 6 integers with male + female + third = total (as parse_textlayer.parse_cover). Some covers
    # (Goa) lay the table out as one word per line: then the 6 numbers after 'Total' in reading order.
    rows = collections.defaultdict(list)
    for w in page.get_text('words'):
        if re.fullmatch(r'\d+', w[4]):
            rows[round((w[1] + w[3]) / 6)].append(w)
    for k in sorted(rows):
        r = sorted(rows[k], key=lambda w: w[0])
        v = [int(w[4]) for w in r]
        if len(v) == 6 and v[0] <= v[1] and v[2] + v[3] + v[4] == v[5]:
            meta['cover'] = dict(serial_from=v[0], serial_to=v[1], male=v[2], female=v[3], third_gender=v[4], total=v[5])
            break
    if 'cover' not in meta:
        m = re.search(r'NUMBER OF ELECTORS(.*)', txt, re.S)
        if m:
            t = m.group(1)
            t = t[t.find('Total') + 5:] if 'Total' in t else t
            v = [int(x) for x in re.findall(r'(?<![\d-])\d+(?![\d-])', t)[:6]]
            if len(v) == 6 and v[0] <= v[1] and v[2] + v[3] + v[4] == v[5]:
                meta['cover'] = dict(serial_from=v[0], serial_to=v[1], male=v[2], female=v[3], third_gender=v[4], total=v[5])
    return meta


def parse_summary(page):
    """Last page 'SUMMARY OF ELECTORS': the last 4 integers m, f, t, total with m + f + t == total (net electors)."""
    txt = page.get_text()
    if 'SUMMARY OF ELECTORS' not in txt.upper():
        return None
    body = txt.split('Signature of Electoral Registration Officer')[0]
    nums = [int(x) for x in re.findall(r'(?<![\d-])\d+(?![\d-])', body) if len(x) < 7]
    res = [nums[i:i + 4] for i in range(len(nums) - 3) if nums[i] + nums[i + 1] + nums[i + 2] == nums[i + 3] and nums[i + 3] > 0]
    if not res:
        return None
    v = res[-1]
    return dict(male=v[0], female=v[1], third_gender=v[2], total=v[3])


# --------------------------------------------------------------------------- box

def parse_box(box, chars, tchars):
    warn, rec, extra = [], {}, []
    photo_sids = {c[3] for c in chars if c[4].strip() in ('Photo', 'Available', 'Photo Available', 'Photo Not Available')
                  or (c[4].lstrip().startswith(PHOTO_TEXT) and c[1] > box.x0 + 0.55 * box.width)}
    chars = [c for c in chars if c[3] not in photo_sids]
    rows = rows_of(chars, key_y=lambda c: c[2])
    top_y = box.y0 + 0.25 * box.height
    fields, last = {}, None
    for r in rows:
        if r['y'] < top_y:
            for tok in row_text(r['c'], gap=0.5).split():
                if re.fullmatch(r'\d+', tok) and 'serial_no' not in rec: rec['serial_no'] = int(tok)
                elif (EPIC_ANY.match(tok) or EPIC_LOOSE.match(tok)) and 'epic' not in rec: rec['epic'] = tok   # odd prints too
                else: extra.append(('top', tok))
            continue
        t = clean(row_text(r['c']))
        if not t: continue
        lab = ('name' if NAME_RE.match(t) else 'rel' if REL_RE.match(t) else 'house' if HOUSE_RE.match(t)
               else 'age' if AGE_RE.match(t) else None)
        if lab is None or lab in fields:
            if last and lab is None:
                fields[last] = (fields[last][0] + ' ' + t, fields[last][1] + r['c'])
                warn.append('wrapped_line:' + last)
            else:
                extra.append(('row', t))
            continue
        fields[lab] = (t, r['c']); last = lab
    if 'name' in fields: rec['name'] = clean(NAME_RE.match(fields['name'][0]).group(1))
    if 'rel' in fields:
        m = REL_RE.match(fields['rel'][0])
        if m.group(1):
            rec['relation_type'] = m.group(1).replace('’', "'").rstrip('s').rstrip("'") if m.group(1).lower() != 'others' else 'Others'
            rec['relation_label'] = m.group(1) + ' Name'
        else:
            rec['relation_type'] = rec['relation_label'] = m.group(2)
        rec['relation_name'] = clean(m.group(3))
    if 'house' in fields: rec['house_no'] = clean(HOUSE_RE.match(fields['house'][0]).group(1))
    if 'age' in fields:
        m = AGE_RE.match(fields['age'][0])
        rec['age'] = int(m.group(1)) if m.group(1).isdigit() else m.group(1)
        rec['gender'] = clean(m.group(2))
    for k in ('name', 'relation_type', 'relation_name', 'house_no', 'age', 'gender', 'serial_no', 'epic'):
        if k not in rec or rec[k] in ('', None):
            warn.append('missing:' + k)
    tops = [tok for w, tok in extra if w == 'top']
    if tops: rec['status_code'] = ' '.join(tops)
    rx = [tok for w, tok in extra if w == 'row']
    if rx: rec['extra_text'] = rx; warn.append('extra_text')
    for lab, (t, cs) in fields.items():   # glyph cross-check
        ys = [c[2] for c in cs]; xs = [c[1] for c in cs]
        lo, hi = min(ys) - 2.5, max(ys) + 2.5
        gl = ''.join(tc[0] for tc in tchars if lo <= tc[2] <= hi and min(xs) - 2 <= tc[1] <= max(xs) + 8)
        a, b = letters(t), letters(gl)
        if a != b:
            warn.append('glyph_mismatch:%s:+%s:-%s' % (lab, ''.join(sorted((a - b).elements())), ''.join(sorted((b - a).elements()))))
    return rec, warn


# --------------------------------------------------------------------------- document

def parse_pdf(path):
    t0 = time.time()
    doc = pymupdf.open(path)
    fn = os.path.basename(path)
    m = re.search(r'-([SU]\d+)-(\d+)-.*-(\d+)-WI\.pdf$', fn)
    st_f, ac_f, part_f = (m.group(1), int(m.group(2)), int(m.group(3))) if m else (None, None, None)
    meta = parse_cover(doc[0]); meta['file'] = fn
    if 'cover' not in meta and len(doc) > 1:   # a long section list can push the totals table to page 2
        c2 = parse_cover(doc[1]).get('cover')
        if c2: meta['cover'] = c2
    meta.setdefault('state_code', st_f)
    summ = None
    for k in range(len(doc) - 1, max(-1, len(doc) - 4), -1):   # the summary is the last page, or close to it
        summ = parse_summary(doc[k])
        if summ: break
    if summ: meta['summary'] = summ
    records, section = [], (None, None)
    for pno, page in enumerate(doc):
        drawings = page.get_drawings()
        boxes = find_boxes(page, drawings)
        if not boxes: continue
        chars, plines = page_chars(page)
        sec_marks = []
        for y, x, t in page_lines(plines, boxes):
            sm = SECTION_RE.search(t)
            if sm: sec_marks.append((y, int(sm.group(1)), clean(sm.group(2))))
        trace = page.get_texttrace()
        tch = [(chr(cp), org[0], org[1], gid) for sp in trace for cp, gid, org, bb in sp['chars']]
        own = assign([(c[1], c[2], c[3]) for c in chars], boxes)
        town = assign([(c[1], c[2], -1 - i) for i, c in enumerate(tch)], boxes)
        per_box, per_tbox = collections.defaultdict(list), collections.defaultdict(list)
        for c, k in zip(chars, own):
            if k is not None: per_box[k].append(c)
        for c, k in zip(tch, town):
            if k is not None: per_tbox[k].append(c)
        marks = box_marks(page, boxes, drawings)
        for i, b in enumerate(boxes):
            for y, no, name in sec_marks:
                if y < b.y0: section = (no, name)
            rec, warn = parse_box(b, per_box[i], per_tbox[i])
            if 'epic' not in rec and 'serial_no' not in rec and 'name' not in rec:
                continue
            if marks.get(i):
                rec['box_marks'] = marks[i]; warn.append('box_marks')
            out = dict(state=meta.get('state_code'), ac=meta.get('ac_no', ac_f), part=meta.get('part_no', part_f), page=pno + 1,
                       section_no=section[0], section_name=section[1])
            for k in ('serial_no', 'epic', 'name', 'relation_type', 'relation_label', 'relation_name', 'house_no', 'age', 'gender',
                      'status_code', 'extra_text', 'box_marks'):
                if k in rec: out[k] = rec[k]
            if warn: out['text_warnings'] = warn
            records.append(out)
    meta['pages'] = len(doc)
    meta['seconds'] = round(time.time() - t0, 3)
    return records, meta


def validate(records, meta):
    v = {}
    cov = meta.get('cover', {})
    n = len(records)
    v['records'] = n; v['cover_total'] = cov.get('total')
    g = collections.Counter(r.get('gender') for r in records)
    v['by_gender'] = {'male': g['Male'], 'female': g['Female'], 'third_gender': g['Third Gender']}
    v['count_match'] = (n == cov.get('total') and g['Male'] == cov.get('male') and g['Female'] == cov.get('female')
                        and g['Third Gender'] == cov.get('third_gender'))
    if meta.get('summary'):
        s = meta['summary']
        v['summary_total'] = s['total']
        v['summary_match'] = (s['total'] == n and s['male'] == g['Male'] and s['female'] == g['Female'])
    ser = [r.get('serial_no') for r in records]
    first = cov.get('serial_from', 1) or 1
    v['serial_continuous'] = ser == list(range(first, first + n))
    if not v['serial_continuous']:
        s = set(x for x in ser if isinstance(x, int))
        v['serial_missing'] = sorted(set(range(first, first + n)) - s)[:20]
        v['serial_dups'] = [k for k, c in collections.Counter(ser).items() if c > 1][:20]
    bad = [r.get('epic') for r in records if not (EPIC_NEW.match(r.get('epic') or '') or EPIC_OLD.match(r.get('epic') or ''))]
    v['epic_bad'] = bad[:20]; v['n_epic_bad'] = len(bad)
    v['epic_old_format'] = sum(1 for r in records if EPIC_OLD.match(r.get('epic') or ''))
    v['epic_dups'] = [k for k, c in collections.Counter(r.get('epic') for r in records).items() if c > 1][:20]
    v['age_bad'] = [(r.get('serial_no'), r.get('age')) for r in records if not (isinstance(r.get('age'), int) and 18 <= r['age'] <= 120)][:20]
    v['gender_bad'] = [(r.get('serial_no'), r.get('gender')) for r in records if r.get('gender') not in GENDERS][:20]
    v['with_warnings'] = sum(1 for r in records if r.get('text_warnings'))
    v['warning_kinds'] = dict(collections.Counter(w.split(':')[0] for r in records for w in r.get('text_warnings', [])))
    v['labels'] = dict(collections.Counter(r.get('relation_label') for r in records))
    v['ok'] = bool(v['count_match'] and v['serial_continuous'] and not bad and not v['age_bad'] and not v['gender_bad'])
    return v


def hint_of_eng(r):
    """A parse_textlayer_eng record -> the --hint-2027 record eci_ocr expects (relation_label = printed label)."""
    return {"epic": r["epic"].upper(), "name": r.get("name"), "relation_label": r.get("relation_label"), "relation_name": r.get("relation_name"),
            "house_no": r.get("house_no"), "age": str(r.get("age")), "gender": r.get("gender"), "serial_no": str(r.get("serial_no")),
            "state": r.get("state"), "ac": str(r["ac"]), "part": str(r["part"]), **({"epic_printed": r["epic"]} if r["epic"] != r["epic"].upper() else {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pdf', nargs='+')
    ap.add_argument('-o', '--outdir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    ap.add_argument('--summary', help='write per-PDF meta + validation JSON here')
    ap.add_argument('--hints', help='write the --hint-2027 records of all PDFs to this JSONL')
    ap.add_argument('--no-records', action='store_true', help='do not write OUTDIR/<stem>.jsonl')
    a = ap.parse_args()
    if not a.no_records: os.makedirs(a.outdir, exist_ok=True)
    allsum = []
    hf = open(a.hints, 'w', encoding='utf-8') if a.hints else None
    for p in a.pdf:
        recs, meta = parse_pdf(p)
        val = validate(recs, meta)
        if not a.no_records:
            with open(os.path.join(a.outdir, os.path.basename(p)[:-4] + '.jsonl'), 'w', encoding='utf-8') as fh:
                for r in recs: fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        if hf:
            for r in recs:
                if r.get('epic'): hf.write(json.dumps(hint_of_eng(r), ensure_ascii=False) + '\n')
        meta['validation'] = val; allsum.append(meta)
        print('%s  records=%d cover=%s summary=%s ok=%s  %.2fs (%d pages)  warnings=%s' % (
            meta['file'], val['records'], val['cover_total'], val.get('summary_total'), val['ok'], meta['seconds'], meta['pages'],
            val['warning_kinds']))
        if not val['ok']:
            print('   ', json.dumps({k: val[k] for k in val if k not in ('ok', 'labels')}, ensure_ascii=False))
    if hf: hf.close()
    if a.summary:
        with open(a.summary, 'w') as fh: json.dump(allsum, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
