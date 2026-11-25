#!/usr/bin/env python3
"""Parse Hindi "SSR DraftRoll 2027" electoral roll PDFs (iText pdfCalligraph text layer) to JSONL.

States (2026-11-25, branch hinvar): UP S24, Uttarakhand S28, Bihar S04, MP S12, Rajasthan S20, Chhattisgarh S26. The states
print different labels (see LABEL VARIANTS below); every variant gives the same output fields.

One JSON record per voter box. No OCR: every field comes from the PDF text layer.

How it works
- Boxes come from the vector drawing: each voter box is a 186 x 75 pt rectangle; its top and
  bottom edges are horizontal strokes. (Fallback when a page has no such strokes: anchor a
  box on each EPIC word, as bench/hint_2027.py does.)
- Characters come from page.get_text('rawdict') (logical order, from the PDF /ActualText).
  Each character goes to the box that contains its origin point. A character that overflows
  the box edge stays with the box of the preceding character in the same span.
- Inside a box, characters are grouped into rows by baseline, and each row is identified by its
  label ('नाम :', '<पिता|पति|माता> का नाम:', 'अन्य:', 'मकान संख्या :', 'आयु : .. लिंग :').
  LABEL VARIANTS: name 'निर्वाचक का नाम :' (Bihar, MP, Rajasthan, Chhattisgarh); house 'गृह संख्या :' (MP, Rajasthan);
  age 'उम्र :' (Bihar, MP, Rajasthan, Chhattisgarh); Bihar doubles colons ('पिता का नाम::', 'लिंग: :').
  'निर्वाचक का नाम' is the voter's name, not a relative (before 2026-11-25 it was read as relation_type 'निर्वाचक').
  The PDF meta gets 'label_sets': boxes per printed label set ('नाम/मकान/आयु', ...), for checks per state.
  The 'फोटो उपलब्ध है' placeholder is its own span and is removed by span, so it can not
  bleed into the house number.
- Repairs of known pdfCalligraph /ActualText artifacts (each one counted per record):
    doubled_mark   'लिंंग' -> 'लिंग'   (a combining mark repeated)
    space_before_mark  'भ ा' -> 'भा' (space before a vowel sign; drawn in the PDF, i.e. a source typo)
    phantom_space  'अश् विनी' -> 'अश्विनी' (a space char with no space glyph drawn there)
    zwj_repeat     'खन्‍नाना' -> 'खन्ना' (after virama+ZWJ the next syllable is repeated)
    zwj_removed    a virama+ZWJ without repetition: the ZWJ is removed ('स्‍वर्ण' -> 'स्वर्ण')
    glyph_fix      the character code disagrees with the embedded font's own cmap for that glyph.
                   In these files this is glyph uni090B (ऋ), which the text layer maps to 'र'
                   ('रषभ' -> 'ऋषभ'). Found from page.get_texttrace() glyph ids.
- Cross-check: for every field, the multiset of letters after repair is compared with the
  glyph-level text from page.get_texttrace() (which has none of the /ActualText artifacts but
  is in visual order). A difference is reported in 'text_warnings'.

Usage
  parse_textlayer.py PDF [PDF ...] [-o OUTDIR] [--summary FILE]
  Writes OUTDIR/<pdf stem>.jsonl (default OUTDIR: ./out) and prints validation per PDF.
"""
import argparse, collections, io, json, os, re, sys, time, unicodedata

import pymupdf

EPIC_NEW = re.compile(r'^[A-Z]{3}[0-9]{7}$')
EPIC_OLD = re.compile(r'^[A-Z]{2}/[0-9]{1,3}/[0-9]{1,4}/[0-9]{5,7}$')
EPIC_ANY = re.compile(r'^([A-Z]{2,3}[0-9]{6,8}|[A-Z]{2}/[0-9]+/[0-9]+/[0-9]+)$')
# EPIC printed in a non-standard form (2026-11-25, Bihar/Rajasthan 2027: 'MVK/1234567A', 'XNJ//1234567', 'BR/06/035//762069',
# 'RJ/21/160/022098B'): kept as printed in 'epic' (validate() lists it under epic_bad), warning 'epic_nonstandard'
EPIC_ODD = re.compile(r'^[A-Z]{2,3}[A-Z0-9/]*[0-9][A-Z0-9/]*$')
GENDERS = {'पुरुष', 'महिला', 'तृतीय लिंग'}
VIRAMA, ZWJ, ZWNJ = '्', '‍', '‌'
DEP_SIGNS = '[ा-ौॢॣँ-ः]'   # matras, candrabindu, anusvara, visarga
PHOTO_TEXT = 'फोटो'
DEP_SIGNS_NO_I = '[\u093e\u0940-\u094c\u0962\u0963\u0901-\u0903]'
DOUBLE_RE = re.compile(r'([\u0900-\u0903\u093c-\u094d])\1+')
RE_REPH_I = re.compile('\u0930\u094d(?:[\u0915-\u0939\u0958-\u095f]\u093c?\u094d)*[\u0915-\u0939\u0958-\u095f]\u093c?\u093f')


# --------------------------------------------------------------------------- repairs

def repair(s, counts):
    """Fix /ActualText artifacts. `counts` (Counter) gets +1 per repair kind that changed s."""
    s = unicodedata.normalize('NFD', s)        # NFD so that 'ऩ' + '़' is seen as a repeated nukta
    t = DOUBLE_RE.sub(r'\1', s)
    if t != s: counts['doubled_mark'] += 1
    s = unicodedata.normalize('NFC', t)
    t = re.sub(' +(?=' + DEP_SIGNS + ')', '', s)
    if t != s: counts['space_before_mark'] += 1
    s = t
    if VIRAMA + ZWJ in s:
        out, i, rep = [], 0, False
        while i < len(s):
            if s.startswith(VIRAMA + ZWJ, i):
                out.append(VIRAMA); i += 2
                for n in (4, 3, 2, 1):
                    p = s[i:i + n]
                    if len(p) == n and s[i + n:i + 2 * n] == p and p[0] not in ' ' + VIRAMA:
                        i += n; rep = True; break
                continue
            out.append(s[i]); i += 1
        s = ''.join(out)
        counts['zwj_repeat' if rep else 'zwj_removed'] += 1
    if ZWJ in s:
        s = s.replace(ZWJ, '')
        counts['zwj_removed'] += 1
    return re.sub(r'\s+', ' ', s).strip()


def letters(s):
    """Multiset used for the glyph cross-check (order-free, no spaces/joiners/colons: Bihar prints label colons twice)."""
    return collections.Counter(c for c in unicodedata.normalize('NFD', s)
                               if not c.isspace() and c not in (ZWJ, ZWNJ, ':'))


# --------------------------------------------------------------------------- fonts

def font_reverse_cmaps(doc):
    """font base name (without subset tag) -> list of {gid: set(codepoints)} from embedded TrueType cmaps."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:  # glyph fix is then off; cross-check still flags the records
        return {}
    rev = collections.defaultdict(list)
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, 'Subtype')[1] != '/CIDFontType2':
                continue
            name = doc.xref_get_key(xref, 'BaseFont')[1].lstrip('/')
            fd = doc.xref_get_key(xref, 'FontDescriptor')
            if fd[0] != 'xref':
                continue
            ff = doc.xref_get_key(int(fd[1].split()[0]), 'FontFile2')
            if ff[0] != 'xref':
                continue
            tt = TTFont(io.BytesIO(doc.xref_stream(int(ff[1].split()[0]))), lazy=True)
            r = collections.defaultdict(set)
            for cp, gname in tt.getBestCmap().items():
                r[tt.getGlyphID(gname)].add(cp)
            rev[name.split('+', 1)[-1]].append(r)
        except Exception:
            continue
    return rev


def glyph_fixes(trace, rev):
    """From get_texttrace(): {(x, y) origin: (wrong_char, right_char)} where the char code disagrees
    with the font's own cmap. A base letter followed by a nukta is not a disagreement (the font has
    a precomposed glyph for क़ etc. and the text layer spells it क + ़)."""
    fixes = {}
    for sp in trace:
        maps = rev.get(sp['font'])
        if not maps:
            continue
        ch = sp['chars']
        for i, (cp, gid, org, _bb) in enumerate(ch):
            if gid < 0:
                continue
            cps = set().union(*[m.get(gid, ()) for m in maps])
            if len(cps) != 1 or cp in cps:
                continue
            right = chr(next(iter(cps)))
            nxt = chr(ch[i + 1][0]) if i + 1 < len(ch) else ''
            if unicodedata.normalize('NFD', right) == chr(cp) + '़' and nxt == '़':
                continue
            if unicodedata.category(right)[0] != 'L':
                continue
            fixes[(round(org[0], 1), round(org[1], 1))] = (chr(cp), right)
    return fixes


# --------------------------------------------------------------------------- page geometry

def find_boxes(page, drawings):
    """Voter boxes from the drawing: pairs of 180-192 pt horizontal strokes 60-90 pt apart."""
    cols = collections.defaultdict(list)
    for d in drawings:
        r = d['rect']
        if r.height < 1.5 and 180 <= r.width <= 192:
            cols[(round(r.x0), round(r.x1))].append(r.y0)
    boxes = []
    for (x0, x1), ys in cols.items():
        ys = sorted(set(round(y, 1) for y in ys))
        i = 0
        while i < len(ys) - 1:
            if 60 <= ys[i + 1] - ys[i] <= 90:
                boxes.append(pymupdf.Rect(x0, ys[i], x1, ys[i + 1])); i += 2
            else:
                i += 1
    if not boxes:  # fallback: EPIC anchor (EPIC at the top-right of the box)
        for x0, y0, x1, y1, w, *_ in page.get_text('words'):
            if EPIC_ANY.match(w):
                boxes.append(pymupdf.Rect(x1 - 188, y0 - 4, x1 + 3, y0 + 72))
    return sorted(boxes, key=lambda r: (round(r.y0 / 10), r.x0))


def box_marks(page, boxes, drawings):
    """Drawings inside a box that are not part of the normal box template (possible deletion
    stamp / strike-out). Normal: 186-wide edges, 75-high edges, serial-cell strokes and shadow."""
    marks = collections.Counter()
    for d in drawings:
        r = d['rect']
        w, h = r.width, r.height
        normal = ((h < 1.5 and (180 <= w <= 192 or 40 <= w <= 65)) or
                  (w < 1.5 and (10 <= h <= 14 or 50 <= h <= 80)))
        if normal:
            continue
        for i, b in enumerate(boxes):
            if b.contains(r.tl) and b.contains(r.br):
                marks[i] += 1
    for img in page.get_image_info():
        r = pymupdf.Rect(img['bbox'])
        for i, b in enumerate(boxes):
            if b.intersects(r):
                marks[i] += 1
    return marks


def page_chars(page):
    """rawdict chars as (char, ox, oy, span_id, span_text, bbox) in stream order, plus the lines
    as (y0, x0, x1, text)."""
    out, lines, sid = [], [], 0
    for b in page.get_text('rawdict')['blocks']:
        for l in b.get('lines', []):
            lt = ''
            for s in l['spans']:
                sid += 1
                txt = ''.join(c['c'] for c in s['chars'])
                lt += txt
                for c in s['chars']:
                    out.append((c['c'], c['origin'][0], c['origin'][1], sid, txt, c['bbox']))
            lines.append((l['bbox'][1], l['bbox'][0], lt))
    return out, lines


def assign(points, boxes, tol=1.0):
    """Index of the box containing each (x, y, span_id); overflow chars follow the previous char of the span."""
    res, last = [], {}
    bx = [(b.x0 - tol, b.y0 - tol, b.x1 + tol, b.y1 + tol) for b in boxes]
    for x, y, sid in points:
        k = None
        for i, (a, b_, c, d) in enumerate(bx):
            if a <= x <= c and b_ <= y <= d:
                k = i; break
        if k is None and sid in last:
            k = last[sid]
            b = boxes[k]
            if not (b.y0 - tol <= y <= b.y1 + tol):
                k = None
        if k is not None:
            last[sid] = k
        res.append(k)
    return res


def rows_of(chars, key_y, tol=2.5):
    """Group chars into rows by baseline. Within a row: spans sorted by x, chars kept in stream order."""
    rows = []
    for c in sorted(chars, key=key_y):
        y = key_y(c)
        if rows and abs(y - rows[-1]['y']) <= tol:
            rows[-1]['c'].append(c)
        else:
            rows.append({'y': y, 'c': [c]})
    return rows


def row_text(chars, gap=None):
    """chars: (ch, ox, oy, sid, ..., bbox). Join spans in x order. Spaces come only from real space
    characters: span bboxes of shaped clusters are too narrow to infer a gap ('चन्द्रि|का')."""
    spans = collections.OrderedDict()
    for c in chars:
        spans.setdefault(c[3], []).append(c)
    parts = sorted(spans.values(), key=lambda cs: min(c[1] for c in cs))
    out, prev_x1 = '', None
    for cs in parts:
        t = ''.join(c[0] for c in cs)
        x0 = min(c[5][0] for c in cs)
        if gap is not None and prev_x1 is not None and x0 - prev_x1 > gap and out and not out.endswith(' ') and not t.startswith(' '):
            out += ' '
        out += t
        prev_x1 = max(c[5][2] for c in cs)
    return out


# --------------------------------------------------------------------------- header / cover

def page_lines(lines, boxes):
    """Text lines outside the voter boxes, top to bottom."""
    bx = [(b.x0, b.y0, b.x1, b.y1) for b in boxes]
    out = []
    for y0, x0, t in lines:
        x, y = x0 + 1, y0 + 1
        if any(a <= x <= c and b <= y <= d for a, b, c, d in bx):
            continue
        out.append((y0, x0, t))
    return sorted(out)


# section line: UP 'अनुभाग संख्या और नाम :', Bihar 'प्रभाग संख्या एवं नाम :', MP/Raj 'अनुभाग की संख्या व नाम :',
# CG 'अनुभाग संख्या एवं नाम :', Uttarakhand 'अनुभाग संख्या व नाम' (no colon)
SECTION_RE = re.compile(r'(?:अनुभाग|प्रभाग)\s+(?:की\s+)?संख्या\s+(?:और|एवं|व)\s+नाम\s*:?\s*(\d+)\s*-\s*(.*)')
# 'कुल पृष्ठ 33 - का पृष्ठ 3' (printed page total); some Bihar files spell it 'पर्ष्ठ' in the text layer
FOOTER_RE = re.compile(r'कुल\s+प\S{0,2}ष्ठ\s+(\d+)\s*-\s*(?:का\s+)?प\S{0,2}ष्ठ\s+(\d+)')
SUMMARY_TITLES = ('निर्वाचकों का सार', 'निर्वाचकों का विवरण', 'मतदाताओं की संक्षिप्त विवरणी')


def parse_cover(page):
    """Cover page: AC, part, PC, polling station, and the totals table '4. निर्वाचकों की संख्या'."""
    txt = page.get_text()
    meta = {}
    rep = collections.Counter()
    # the cover text breaks lines inside words ('स्थि\nति'), so match '<no> - <name> (<reservation>)' in order:
    # the first is the assembly constituency, the second the parliamentary constituency
    ms = re.findall(r':\s*(\d+)\s*-\s*([^(\n:]+?)\s*\(([^)\n]*)\)', txt)
    if ms:
        meta.update(ac_no=int(ms[0][0]), ac_name=repair(ms[0][1], rep), ac_reservation=repair(ms[0][2], rep))
    if len(ms) > 1:
        meta.update(pc_no=int(ms[1][0]), pc_name=repair(ms[1][1], rep))
    m = re.search(r'S(\d\d)\s+(\S[^\n]*)', txt.split('\n', 1)[0])
    if m:
        meta.update(state_code='S' + m.group(1), state=repair(m.group(2), rep))
    m = re.search(r'भाग संख्या\s*:[\s:]*(\d+)', txt)
    if m: meta['part_no'] = int(m.group(1))
    # totals table: the first row of exactly 6 integers where male + female + third = total
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
    return meta


def parse_summary(page):
    """Last page (title 'निर्वाचकों का सार' / 'निर्वाचकों का विवरण' / 'मतदाताओं की संक्षिप्त विवरणी'): rows of male / female /
    third / total. Returns the last row whose 4 right-most numbers add up (net). Other numbers on the row (a year such as
    2027 at x 330-340 in MP, Rajasthan, Chhattisgarh; a row number) are ignored."""
    words = page.get_text('words')
    rows = collections.defaultdict(list)
    for w in words:
        if re.fullmatch(r'\d+', w[4]) and w[0] > 380:
            rows[round(w[1] / 4)].append(w)
    res = []
    for k in sorted(rows):
        r = sorted(rows[k], key=lambda w: w[0])
        if len(r) == 4:
            v = [int(w[4]) for w in r]
            if v[0] + v[1] + v[2] == v[3]:
                res.append(v)
    text = page.get_text()
    if not res or not any(t in text for t in SUMMARY_TITLES):
        return None
    v = res[-1]
    return dict(male=v[0], female=v[1], third_gender=v[2], total=v[3], rows=res)


# --------------------------------------------------------------------------- box parsing

REL_RE = re.compile(r'^(?:(?!निर्वाचक\s)(\S+)\s+का\s+नाम|(अन्य))\s*:[\s:]*(.*)$')
NAME_RE = re.compile(r'^(निर्वाचक\s+का\s+)?नाम\s*:[\s:]*(.*)$')
HOUSE_RE = re.compile(r'^(मकान|गृह)\s+संख्या\s*:[\s:]*(.*)$')
AGE_RE = re.compile(r'^(आयु|उम्र)\s*:[\s:]*(\S*)\s*लिंग\s*:[\s:]*(.*)$')


def parse_box(box, chars, tchars, fixes):
    """chars: rawdict chars in this box; tchars: texttrace chars (ch, ox, oy, gid) in this box."""
    rep = collections.Counter()
    warn = []
    rec = {}
    # photo placeholder: whole span removed by span id
    photo_sids = {c[3] for c in chars if c[4].lstrip().startswith(PHOTO_TEXT) and c[1] > box.x0 + 0.55 * box.width}
    if photo_sids:
        rec['photo'] = repair(next(c[4] for c in chars if c[3] in photo_sids), collections.Counter())
    chars = [c for c in chars if c[3] not in photo_sids]
    # glyph fixes (ऋ)
    fixed = []
    for c in chars:
        f = fixes.get((round(c[1], 1), round(c[2], 1)))
        if f and c[0] == f[0]:
            c = (f[1],) + c[1:]
            rep['glyph_fix'] += 1
        fixed.append(c)
    chars = fixed
    # phantom spaces: a space character with no space glyph drawn at that point ('अश्| |विनी')
    tsp = [(tc[1], tc[2]) for tc in tchars if tc[0] == ' ']
    keep = []
    for c in chars:
        if c[0] == ' ' and not any(abs(x - c[1]) < 0.6 and abs(y - c[2]) < 2 for x, y in tsp):
            rep['phantom_space'] += 1
            continue
        keep.append(c)
    chars = keep
    rows = rows_of(chars, key_y=lambda c: c[2])
    top_y = box.y0 + 0.25 * box.height
    fields = {}
    extra = []
    last_label = None
    for r in rows:
        raw = row_text(r['c'])
        if r['y'] < top_y:  # serial / EPIC cell row: separate spans, so split on span as well
            for tok in row_text(r['c'], gap=0.5).split():
                if re.fullmatch(r'\d+', tok) and 'serial_no' not in rec:
                    rec['serial_no'] = int(tok)
                elif EPIC_ANY.match(tok) and 'epic' not in rec:
                    rec['epic'] = tok
                elif len(tok) >= 6 and EPIC_ODD.match(tok) and 'epic' not in rec:
                    rec['epic'] = tok
                    warn.append('epic_nonstandard')
                else:
                    extra.append(('top', tok))
            continue
        t = repair(raw, collections.Counter())   # labels carry artifacts too ('लिंंग'): not counted
        lab = None
        if NAME_RE.match(t): lab = 'name'
        elif REL_RE.match(t): lab = 'rel'
        elif HOUSE_RE.match(t): lab = 'house'
        elif AGE_RE.match(t): lab = 'age'
        if lab is None or lab in fields:
            if last_label and lab is None:  # continuation line of a wrapped value
                fields[last_label] = (fields[last_label][0] + ' ' + t, fields[last_label][1] + r['c'])
                warn.append('wrapped_line:' + last_label)
            else:
                extra.append(('row', t))
            continue
        if lab != 'age' and ':' in raw:              # count repairs on the value only
            v = raw.split(':', 1)[1]
            if lab is not None and v.lstrip().startswith(':'):   # Bihar 'पिता का नाम:: X': the second colon is part of the label
                v = v.lstrip()[1:]
            t = t.split(':', 1)[0] + ': ' + repair(v, rep)
        fields[lab] = (t, r['c'])
        last_label = lab
    labs = []
    if 'name' in fields:
        m = NAME_RE.match(fields['name'][0])
        rec['name'] = m.group(2).strip(); labs.append('निर्वाचक का नाम' if m.group(1) else 'नाम')
    if 'rel' in fields:
        m = REL_RE.match(fields['rel'][0])
        rec['relation_type'] = m.group(1) or m.group(2)
        rec['relation_name'] = m.group(3).strip()
    if 'house' in fields:
        m = HOUSE_RE.match(fields['house'][0])
        rec['house_no'] = m.group(2).strip(); labs.append(m.group(1))
    if 'age' in fields:
        m = AGE_RE.match(fields['age'][0])
        rec['age'] = int(m.group(2)) if m.group(2).isdigit() else m.group(2)
        rec['gender'] = m.group(3).strip(); labs.append(m.group(1))
    if labs:
        rec['labels'] = '/'.join(labs)
    for k in ('name', 'relation_type', 'relation_name', 'house_no', 'age', 'gender', 'serial_no', 'epic'):
        if k not in rec or rec[k] in ('', None):
            warn.append('missing:' + k)
    tops = [tok for w, tok in extra if w == 'top']
    if tops:
        rec['status_code'] = ' '.join(tops)
    rows_extra = [tok for w, tok in extra if w == 'row']
    if rows_extra:
        rec['extra_text'] = rows_extra
        warn.append('extra_text')
    # cross-check every labelled row against glyph-level text (texttrace)
    for lab, (t, cs) in fields.items():
        ys = [c[2] for c in cs]
        lo, hi = min(ys) - 2.5, max(ys) + 2.5
        xs = [c[1] for c in cs]
        gl = ''.join(tc[0] for tc in tchars if lo <= tc[2] <= hi and min(xs) - 2 <= tc[1] <= max(xs) + 8)
        gl = unicodedata.normalize('NFD', gl)
        if DOUBLE_RE.search(gl):  # the source itself has a repeated mark (e.g. 'काॅॅजी'): kept collapsed
            warn.append('source_double_mark:' + lab)
            gl = DOUBLE_RE.sub(r'\1', gl)
        g2 = re.sub(' +(?=' + DEP_SIGNS_NO_I + ')', '', gl)  # texttrace is visual: ि may follow a space
        if g2 != gl:  # the PDF draws a space before a vowel sign ('भ ा'): a source typo; the space is removed
            warn.append('source_space_before_mark:' + lab)
            gl = g2
        a, b = letters(t), letters(gl)
        if a != b:
            plus, minus = a - b, b - a
            # known texttrace quirk: i-matra+reph ligature ('र्ति') carries the code of 'ं' in ToUnicode
            k = len(RE_REPH_I.findall(t))
            # and the reverse (Bihar 2027 fonts, 2026-11-25): the i-matra+anusvara glyph ('लिंग', 'सिंह') carries 'र्'
            k2 = t.count('\u093f\u0902')
            if not (k and plus == collections.Counter({'र': k, VIRAMA: k}) and minus == collections.Counter({'ं': k})) and \
                    not (k2 and plus == collections.Counter({'ं': k2}) and minus == collections.Counter({'र': k2, VIRAMA: k2})):
                warn.append('glyph_mismatch:%s:+%s:-%s' % (lab, ''.join(sorted(plus.elements())),
                                                             ''.join(sorted(minus.elements()))))
        sa, sb = len(t.split()), len(re.sub(r'\s+', ' ', gl).strip().split())
        if sa != sb:
            warn.append('space_mismatch:%s:%d:%d' % (lab, sa, sb))
    for k in ('name', 'relation_name', 'house_no'):
        if 'ऎ' in str(rec.get(k, '')):
            warn.append('rare_vowel_e_short:' + k)  # the PDF draws glyph ऎ (U+090E); likely ए or ऐ meant
    return rec, rep, warn


def texttrace_chars(trace, fixes):
    out = []
    for sp in trace:
        for cp, gid, org, bb in sp['chars']:
            ch = chr(cp)
            f = fixes.get((round(org[0], 1), round(org[1], 1)))
            if f and ch == f[0]:
                ch = f[1]
            out.append((ch, org[0], org[1], gid))
    return out


# --------------------------------------------------------------------------- document

def parse_pdf(path):
    t0 = time.time()
    doc = pymupdf.open(path)
    rev = font_reverse_cmaps(doc)
    fn = os.path.basename(path)
    m = re.search(r'S(\d+)-(\d+)-.*-(\d+)-WI\.pdf$', fn)
    ac_f, part_f = (int(m.group(2)), int(m.group(3))) if m else (None, None)
    meta = parse_cover(doc[0])
    meta['file'] = fn
    summ = None
    for pg in range(len(doc) - 1, max(0, len(doc) - 4), -1):   # last page; some Rajasthan files put a section list after it
        summ = parse_summary(doc[pg])
        if summ: break
    if summ:
        meta['summary'] = summ
    records = []
    printed_pages = []
    label_sets = collections.Counter()
    repairs = collections.Counter()      # records needing each repair
    section = (None, None)
    for pno, page in enumerate(doc):
        drawings = page.get_drawings()
        boxes = find_boxes(page, drawings)
        if not boxes:
            continue
        chars, plines = page_chars(page)
        lines = page_lines(plines, boxes)
        sec_marks = []
        for y, x, t in lines:
            fm = FOOTER_RE.search(t)
            if fm:
                printed_pages.append(int(fm.group(1)))
            sm = SECTION_RE.search(t)
            if sm:
                rr = collections.Counter()
                sec_marks.append((y, int(sm.group(1)), repair(sm.group(2), rr)))
        trace = page.get_texttrace()
        fixes = glyph_fixes(trace, rev)
        own = assign([(c[1], c[2], c[3]) for c in chars], boxes)
        tch = texttrace_chars(trace, fixes)
        town = assign([(c[1], c[2], -1 - i) for i, c in enumerate(tch)], boxes)
        per_box = collections.defaultdict(list)
        for c, k in zip(chars, own):
            if k is not None:
                per_box[k].append(c)
        per_tbox = collections.defaultdict(list)
        for c, k in zip(tch, town):
            if k is not None:
                per_tbox[k].append(c)
        marks = box_marks(page, boxes, drawings)
        for i, b in enumerate(boxes):
            for y, no, name in sec_marks:
                if y < b.y0:
                    section = (no, name)
            rec, rep, warn = parse_box(b, per_box[i], per_tbox[i], fixes)
            if 'epic' not in rec and 'serial_no' not in rec and 'name' not in rec:
                continue  # empty template box
            if marks.get(i):
                rec['box_marks'] = marks[i]
                warn.append('box_marks')
            out = dict(ac=meta.get('ac_no', ac_f), part=meta.get('part_no', part_f), page=pno + 1,
                       section_no=section[0], section_name=section[1])
            for k in ('serial_no', 'epic', 'name', 'relation_type', 'relation_name', 'house_no', 'age',
                      'gender', 'status_code', 'extra_text', 'box_marks'):
                if k in rec:
                    out[k] = rec[k]
            label_sets[rec.get('labels')] += 1   # per PDF only (not per record: the output columns stay as they were)
            if rep:
                out['repairs'] = sorted(rep)
            if warn:
                out['text_warnings'] = warn
            for k in rep:
                repairs[k] += 1
            records.append(out)
    meta['pages'] = len(doc)
    if printed_pages:
        meta['printed_pages'] = collections.Counter(printed_pages).most_common(1)[0][0]
    meta['seconds'] = round(time.time() - t0, 3)
    meta['repairs'] = dict(repairs)
    meta['label_sets'] = dict(label_sets)
    return records, meta


def validate(records, meta):
    v = {}
    cov = meta.get('cover', {})
    n = len(records)
    v['records'] = n
    v['cover_total'] = cov.get('total')
    g = collections.Counter(r.get('gender') for r in records)
    v['count_match'] = (n == cov.get('total') and g['पुरुष'] == cov.get('male') and g['महिला'] == cov.get('female')
                        and g['तृतीय लिंग'] == cov.get('third_gender'))
    v['by_gender'] = {'male': g['पुरुष'], 'female': g['महिला'], 'third_gender': g['तृतीय लिंग']}
    if meta.get('summary'):
        s = meta['summary']
        v['summary_match'] = (s['total'] == n and s['male'] == g['पुरुष'] and s['female'] == g['महिला'])
    if meta.get('printed_pages'):
        v['pages_match'] = meta['printed_pages'] == meta.get('pages')
    ser = [r.get('serial_no') for r in records]
    v['serial_continuous'] = ser == list(range(1, n + 1))
    if not v['serial_continuous']:
        s = set(x for x in ser if isinstance(x, int))
        v['serial_missing'] = sorted(set(range(1, n + 1)) - s)[:20]
        v['serial_dups'] = [k for k, c in collections.Counter(ser).items() if c > 1][:20]
    bad_epic = [r.get('epic') for r in records if not (EPIC_NEW.match(r.get('epic') or '') or EPIC_OLD.match(r.get('epic') or ''))]
    v['epic_bad'] = bad_epic[:20]
    v['epic_old_format'] = sum(1 for r in records if EPIC_OLD.match(r.get('epic') or ''))
    v['epic_dups'] = [k for k, c in collections.Counter(r.get('epic') for r in records).items() if c > 1]
    v['age_bad'] = [(r.get('serial_no'), r.get('age')) for r in records
                    if not (isinstance(r.get('age'), int) and 18 <= r['age'] <= 120)][:20]
    v['gender_bad'] = [(r.get('serial_no'), r.get('gender')) for r in records if r.get('gender') not in GENDERS][:20]
    v['with_warnings'] = sum(1 for r in records if r.get('text_warnings'))
    v['warning_kinds'] = dict(collections.Counter(w.split(':')[0] for r in records for w in r.get('text_warnings', [])))
    v['ok'] = (v['count_match'] and v['serial_continuous'] and not bad_epic and not v['age_bad'] and not v['gender_bad']
               and v.get('summary_match', True) and v.get('pages_match', True))
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pdf', nargs='+')
    ap.add_argument('-o', '--outdir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    ap.add_argument('--summary', help='write per-PDF meta + validation JSON here')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    allsum = []
    for p in a.pdf:
        recs, meta = parse_pdf(p)
        val = validate(recs, meta)
        out = os.path.join(a.outdir, os.path.basename(p)[:-4] + '.jsonl')
        with open(out, 'w') as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        meta['validation'] = val
        allsum.append(meta)
        print('%s  records=%d cover=%s ok=%s  %.2fs (%d pages, %.3f s/page)  repairs=%s warnings=%s' % (
            meta['file'], val['records'], val['cover_total'], val['ok'], meta['seconds'], meta['pages'],
            meta['seconds'] / meta['pages'], meta['repairs'], val['warning_kinds']))
        if not val['ok']:
            print('   ', json.dumps({k: val[k] for k in val if k not in ('ok',)}, ensure_ascii=False))
    if a.summary:
        with open(a.summary, 'w') as fh:
            json.dump(allsum, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
