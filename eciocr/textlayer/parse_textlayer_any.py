#!/usr/bin/env python3
"""Parse a "SSR DraftRoll 2027" electoral roll PDF of ANY language from its text layer (no OCR), by line position.

For the languages without their own parser (2026-11-25): Assamese (S03 ASM), Bengali (S03 BEN), Malayalam (S11 MAL),
Kannada (S11 KAN), Punjabi (S19 PAN). Hindi has parse_textlayer.py, English parse_textlayer_eng.py. The code knows no label
text: it uses the box layout, which is the same in every state (checked on the samples of all 5 languages):
    <serial>                                   <EPIC>
    <name label> : <name>                      <photo text>
    <relation label>: <relation name>          <photo text>
    <house label> : <house no>
    <age label> : <age> <gender label> : <gender>
- Boxes: the 186 x 75 pt rectangles of the drawing (parse_textlayer.find_boxes). Characters: rawdict, each to the box that
  holds its origin; rows by baseline (parse_textlayer.rows_of / row_text).
- Photo text: spans that start in the right 28 % of the box and whose text repeats in at least 30 % of the boxes of the page
  ('ਫੋਟੋ ਉਪਲਬਧ', 'ഫോട്ടോ', 'লভ্যমাণ্', ...). Any other span there is kept (warning right_span).
- Body rows: a row with ':' starts a field, a row without ':' continues the field before it (warning wrapped_line).
  Fields by position: 1st name, 2nd relation, 3rd house, 4th age + gender (the row with two ':' and a number after the
  first). Each row is split into label and value at the first ':' (more ':' right after it belong to the label).
- Labels -> relation_type (Father / Husband / Mother / Wife / Others / Guardian) and gender -> M / F / T: small tables per
  language below, made from the printed words of the samples. An unknown relation label gives relation_type None and warning
  relation_label_unknown; an unknown gender gives gender_std None (validate: gender_bad).
- Repairs of the pdfCalligraph /ActualText artifacts (same kind as in the Hindi rolls; each kind counted per record):
    doubled_mark       a combining mark repeated: 'ਸਿੰੰਘ' -> 'ਸਿੰਘ', 'পুরুুষ' -> 'পুরুষ', 'ശോാഭന' -> 'ശോഭന' (NFD: ോ = െ + ാ)
    space_before_mark  a space before a combining mark
    bengali_ra         Bengali-script rolls: the font maps the glyph of র to ব + ় ('পিতাব়' -> 'পিতার'; Assamese: ৰ); after a
                       virama it stands for ব-phala + র ('ঈশ্ব়' -> 'ঈশ্বর', as the glyph trace shows)
    ra_phala_restored  Bengali-script rolls: the text layer drops the ra-phala (or ya-phala) of some conjuncts ('শুক্া',
                       'হ্াপি'); restored after the virama when the glyph trace has exactly that many more র (or য) and nothing
                       else differs (kept as a warning)
    visual_order       an i-matra at the start of a word ('ਿਪਤਾ'): the row came in visual order (Urdu names in Punjabi rolls);
                       the matra is moved after the consonant ('ਪਿਤਾ')
    chillu_atomic      Malayalam: chillu written as consonant + virama + ZWJ ('ബാലന്‍') -> the atomic chillu ('ബാലൻ'); the
                       same file prints both forms
- Cross-check: per field, the multiset of letters (NFD, after the same repairs) against the glyph-level text of
  page.get_texttrace(); a difference is reported in text_warnings (glyph_mismatch:<field>:+<extra>:-<missing>).
- Totals for the check: cover table (start / end serial, male, female, third gender, total: the first row of 6 integers with
  m + f + t = total), summary page (the last row of 4 integers with m + f + t = total), page footer ('<total pages> - <page>'
  in any language: the most frequent first number = printed page count), header part number.

Every record keeps raw_lines: the row texts of the box as the text layer gives them (before repairs), top row first.

Usage
  parse_textlayer_any.py PDF [PDF ...] [--lang ASM] [-o OUTDIR] [--summary FILE]
"""
import argparse, collections, json, os, re, sys, time, unicodedata

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_textlayer import (find_boxes, page_chars, assign, rows_of, row_text,   # noqa: E402
                             EPIC_NEW, EPIC_OLD, EPIC_ANY, EPIC_ODD)

EPIC_LOOSE = re.compile(r'^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9/]{8,}$')
FILE_RE = re.compile(r'(\d{4})-EROLLGEN-([SU]\d\d)-(\d+)-.*-([A-Z]{3})-(\d+)-WI\.pdf$')
ZWJ, ZWNJ = '‍', '‌'

# --------------------------------------------------------------------------- language tables (from the samples, 2026-11-25)
# relation label (repaired, without ':') -> relation_type. Words as printed; keys are matched after whitespace collapse.
REL = {
    'ASM': {'পিতাৰ নাম': 'Father', 'স্বামীৰ নাম': 'Husband', 'মাতৃৰ নাম': 'Mother', 'মাকৰ নাম': 'Mother', 'পত্নীৰ নাম': 'Wife',
            'অন্যান্য': 'Others', 'অভিভাৱকৰ নাম': 'Guardian'},
    'BEN': {'পিতার নাম': 'Father', 'স্বামীর নাম': 'Husband', 'মাতার নাম': 'Mother', 'মায়ের নাম': 'Mother', 'স্ত্রীর নাম': 'Wife',
            'অন্যান্য': 'Others', 'অভিভাবকের নাম': 'Guardian'},
    'MAL': {'അച്ഛന്റെ പേര്': 'Father', 'ഭർത്താവിന്റെ പേര്': 'Husband', 'അമ്മയുടെ പേര്': 'Mother', 'ഭാര്യയുടെ പേര്': 'Wife',
            'മറ്റുള്ളവർ': 'Others', 'മറ്റുള്ളവ': 'Others', 'രക്ഷിതാവിന്റെ പേര്': 'Guardian'},
    'KAN': {'ತಂದೆಯ ಹೆಸರು': 'Father', 'ಗಂಡನ ಹೆಸರು': 'Husband', 'ತಾಯಿಯ ಹೆಸರು': 'Mother', 'ಹೆಂಡತಿಯ ಹೆಸರು': 'Wife',
            'ಇತರೆ': 'Others', 'ಇತರರು': 'Others', 'ಪೋಷಕರ ಹೆಸರು': 'Guardian'},
    'PAN': {'ਪਿਤਾ ਦਾ ਨਾਮ': 'Father', 'ਪਤੀ ਦਾ ਨਾਮ': 'Husband', 'ਮਾਤਾ ਦਾ ਨਾਮ': 'Mother', 'ਪਤਨੀ ਦਾ ਨਾਮ': 'Wife',
            'ਹੋਰ': 'Others', 'ਹੋਰ ਦਾ ਨਾਮ': 'Others', 'ਸਰਪ੍ਰਸਤ ਦਾ ਨਾਮ': 'Guardian'},
    # Tamil (lang-tam 2026-11-25): SIR 2026 prints 'தந்தையின் பெயர்', the 2027 rolls 'தந்தை பெயர்'
    'TAM': {'தந்தையின் பெயர்': 'Father', 'தந்தை பெயர்': 'Father', 'கணவர் பெயர்': 'Husband', 'கணவரின் பெயர்': 'Husband',
            'தாயின் பெயர்': 'Mother', 'தாய் பெயர்': 'Mother', 'மனைவி பெயர்': 'Wife', 'மனைவியின் பெயர்': 'Wife',
            'இதர': 'Others', 'இதரர்': 'Others', 'இதரர் பெயர்': 'Others', 'மற்றவர்': 'Others', 'மற்றவை': 'Others', 'பாதுகாவலர் பெயர்': 'Guardian'},
}
GENDER = {
    'ASM': {'পুৰুষ': 'M', 'মহিলা': 'F', 'তৃতীয় লিঙ্গ': 'T', '৩য় লিঙ্গ': 'T'},
    'BEN': {'পুরুষ': 'M', 'মহিলা': 'F', 'তৃতীয় লিঙ্গ': 'T', '৩য় লিঙ্গ': 'T'},
    'MAL': {'പുരുഷൻ': 'M', 'സ്ത്രീ': 'F', 'മൂന്നാം ലിംഗം': 'T', 'ട്രാൻസ്ജെൻഡർ': 'T'},
    'KAN': {'ಗಂಡು': 'M', 'ಹೆಣ್ಣು': 'F', 'ತೃತೀಯ ಲಿಂಗಿಗಳು': 'T', 'ತೃತೀಯ ಲಿಂಗ': 'T', 'ಇತರೆ': 'T'},
    'PAN': {'ਪੁਰਸ਼': 'M', 'ਇਸਤਰੀ': 'F', 'ਤੀਜਾ ਲਿੰਗ': 'T'},
    'TAM': {'ஆண்': 'M', 'பெண்': 'F', 'மூன்றாம் பாலினம்': 'T', 'திருநங்கை': 'T', 'மற்றவை': 'T'},
}
# a pre-base i-matra at the start of a word comes from text in visual order (a Punjabi row with Urdu text: 'ਿਪਤਾ ਦਾ ਨਾਮ')
VISUAL_I = re.compile('(^|\\s)([\u0a3f\u09bf\u093f])([\u0a15-\u0a39\u0a59-\u0a5e\u0995-\u09b9\u09dc-\u09df\u09f0\u09f1\u0915-\u0939])')
RA = {'ASM': '\u09f0', 'BEN': '\u09b0'}            # ৰ (Assamese), র (Bengali)
DANGLING = re.compile('\u09cd(?=[\u09be-\u09cc\u09d7])')   # a virama right before a vowel sign: a consonant is missing
CHILLU = {'ണ': 'ൺ', 'ന': 'ൻ', 'ര': 'ർ', 'ല': 'ൽ', 'ള': 'ൾ', 'ക': 'ൿ'}
CHILLU_RE = re.compile('([ണനരലളക])്‍')
TAM_RA_RE = re.compile('\u0bbe(?=[\u0bbf-\u0bcd])')   # NFD: an aa-sign right before another vowel sign / virama = ர
TAM_AARA_RE = re.compile('\u0bbe\u0bbe(?=[\u0bbf-\u0bcd])')   # NFD: 'ாா்' / 'ாாி' = ா + ர் / ா + ரி (not a doubled mark)
TAM_DANGLING = re.compile('(?<![\u0b95-\u0bb9])\u0bbe')   # a Tamil aa-sign not after a consonant: letters are missing   # Tamil aa-sign + virama = ர் (one glyph for ர and ா)


def is_mark(c):
    return unicodedata.category(c) in ('Mn', 'Mc')


MARK_RUN = re.compile(r'[\u0900-\u0dff\u0300-\u036f]{2,}')


def collapse_run(run):
    """In a run of characters, drop a combining-mark block (1-3 marks) that repeats the block right before it: in NFD,
    'ো' + 'া' = 'ে' 'া' 'া', 'গোঁাঁ' = গ 'ে' 'া' 'ঁ' 'া' 'ঁ', 'মোঃাঃ'. Blocks of letters are never touched."""
    c = list(run)
    i = 0
    while i < len(c):
        for n in (3, 2, 1):
            blk = c[i:i + n]
            if len(blk) == n and all(is_mark(x) for x in blk) and c[i + n:i + 2 * n] == blk:
                del c[i + n:i + 2 * n]
                break
        else:
            i += 1
    return ''.join(c)


def repair(s, lang, counts=None):
    """Fix the /ActualText artifacts. counts (Counter) gets +1 per repair kind that changed s."""
    counts = counts if counts is not None else collections.Counter()
    s = unicodedata.normalize('NFD', s or '')
    if lang == 'TAM' and TAM_AARA_RE.search(s):
        # lang-tam 2026-11-25: 'செல்வகுமாா்' is 'செல்வகுமார்' (2027 print checked by eye), 'முத்துமாாியப்பன்' is
        # 'முத்துமாரியப்பன்': the second aa-sign is the ர glyph. Done before the doubled-mark collapse, which made 'குமர்'.
        s = TAM_AARA_RE.sub('\u0bbe\u0bb0', s); counts['tamil_aara'] += 1
    t = MARK_RUN.sub(lambda m: collapse_run(m.group(0)), s)
    if t != s: counts['doubled_mark'] += 1
    s = unicodedata.normalize('NFC', t)
    t = re.sub(r' +(?=[̀-ͯऀ-෿])', lambda m: m.group(0) if not is_mark(s[m.end()]) else '', s)
    if t != s: counts['space_before_mark'] += 1
    s = t
    t = VISUAL_I.sub(r'\1\3\2', s)
    if t != s:
        s = t; counts['visual_order'] += 1
    if lang in RA and '\u09ac\u09bc' in s:
        # after a virama the glyph trace shows ব-phala + র ('ঈশ্ব়' is 'ঈশ্বর': 43 of 44 sample cases); elsewhere ব় is র
        s = s.replace('\u09cd\u09ac\u09bc', '\u09cd\u09ac' + RA[lang]).replace('\u09ac\u09bc', RA[lang])
        counts['bengali_ra'] += 1
    if lang == 'MAL' and CHILLU_RE.search(s):
        s = CHILLU_RE.sub(lambda m: CHILLU[m.group(1)], s); counts['chillu_atomic'] += 1
    if lang == 'TAM':
        # Tamil fonts draw ர and the aa-sign ா with one glyph; the /ActualText of 'ர்' is sometimes 'ா்' ('பெயா்'), of 'ரி'
        # 'ாி' ('மோி' = 'மேரி'). An aa-sign never takes a virama or another vowel sign, so that aa-sign is ர (on NFD).
        d = unicodedata.normalize('NFD', s)
        if TAM_RA_RE.search(d):
            s = unicodedata.normalize('NFC', TAM_RA_RE.sub('\u0bb0', d)); counts['tamil_ra'] += 1
    return re.sub(r'\s+', ' ', s).strip()


# glyph-trace letters that the Bengali-script fonts draw with one glyph: the trace can not tell them apart (ৰ / র, ৱ / ব)
FOLD = {'ASM': str.maketrans('\u09f0\u09f1', '\u09b0\u09ac'), 'BEN': str.maketrans('\u09f0\u09f1', '\u09b0\u09ac'),
        'TAM': str.maketrans('\u0bb0', '\u0bbe')}   # Tamil: ர and ா are one glyph


def letters(s, lang, trace=False):
    """Multiset for the glyph cross-check. The glyph trace is in visual order, so it gets only the order-free repairs."""
    if trace:
        s = unicodedata.normalize('NFD', s)
        if lang == 'TAM': s = TAM_AARA_RE.sub('\u0bbe\u0bb0', s)   # as repair(): 'ாா்' = ார் before the collapse
        s = unicodedata.normalize('NFC', MARK_RUN.sub(lambda m: collapse_run(m.group(0)), s))
        if lang in RA:
            s = s.replace('\u09ac\u09bc', '\u09b0')
        if lang == 'MAL':
            s = CHILLU_RE.sub(lambda m: CHILLU[m.group(1)], s)
    else:
        s = repair(s, lang)
    if lang in FOLD:
        s = s.translate(FOLD[lang])
    if lang == 'KAN':   # Kannada conjunct glyphs ('ಸ್ಟ') carry no virama in the glyph trace
        s = s.replace('\u0ccd', '')
    return collections.Counter(c for c in unicodedata.normalize('NFD', s) if not c.isspace() and c not in (ZWJ, ZWNJ, ':'))


def toint(s):
    """'१२', '১২', '12' -> 12; None when s is not all decimal digits."""
    s = (s or '').strip()
    if not s or not all(unicodedata.decimal(c, None) is not None for c in s):
        return None
    return int(''.join(str(unicodedata.decimal(c)) for c in s))


INT_TOK = re.compile(r'(?<![\w-])(\w+)(?![\w-])')


# --------------------------------------------------------------------------- cover / summary / footer

def _int_words(page, xmin=None):
    out = []
    for w in page.get_text('words'):
        v = toint(w[4])
        if v is not None and (xmin is None or w[0] > xmin):
            out.append((w, v))
    return out


def parse_cover(page):
    """Cover totals: the first row of 6 integers with v0 <= v1 and v2 + v3 + v4 == v5; else 6 integers in reading order."""
    rows = collections.defaultdict(list)
    for w, v in _int_words(page):
        rows[round((w[1] + w[3]) / 6)].append((w[0], v))
    for k in sorted(rows):
        v = [x for _, x in sorted(rows[k])]
        if len(v) == 6 and v[0] <= v[1] and v[2] + v[3] + v[4] == v[5]:
            return dict(serial_from=v[0], serial_to=v[1], male=v[2], female=v[3], third_gender=v[4], total=v[5])
    nums = [v for _, v in _int_words(page)]
    for i in range(len(nums) - 5):
        v = nums[i:i + 6]
        if v[0] <= v[1] and v[2] + v[3] + v[4] == v[5] and v[5] > 0 and v[5] == v[1] - v[0] + 1:
            return dict(serial_from=v[0], serial_to=v[1], male=v[2], female=v[3], third_gender=v[4], total=v[5], read='order')
    return None


def parse_summary(page):
    """Summary page: rows of 4 integers right of x 380 with m + f + t == total; the last one (net electors)."""
    rows = collections.defaultdict(list)
    for w, v in _int_words(page, xmin=380):
        rows[round(w[1] / 4)].append((w[0], v))
    res = []
    for k in sorted(rows):
        v = [x for _, x in sorted(rows[k])]
        if len(v) == 4 and v[0] + v[1] + v[2] == v[3] and v[3] > 0:
            res.append(v)
    if not res:
        return None
    v = res[-1]
    return dict(male=v[0], female=v[1], third_gender=v[2], total=v[3], rows=res)


FOOTER_RE = re.compile(r'(\d+)\s*-\s*\D{1,30}?\s(\d+)\s*$')
SECTION_RE = re.compile(r'(\d+)\s*-\s*(\S.*)$')
PART_RE = re.compile(r':[\s:]*(\d+)\s*$')


def merged_lines(lines, boxes, tol=3.0):
    """Text lines outside the voter boxes, pieces on the same baseline joined in x order: [(y, x, text)] top to bottom."""
    bx = [(b.x0, b.y0, b.x1, b.y1) for b in boxes]
    pieces = sorted((y0, x0, t) for y0, x0, t in lines
                    if not any(a <= x0 + 1 <= c and b <= y0 + 1 <= d for a, b, c, d in bx))
    out = []
    for y, x, t in pieces:
        if out and abs(out[-1][0] - y) <= tol and x - out[-1][3] < 150:
            out[-1][2].append((x, t)); out[-1][3] = max(out[-1][3], x + 5 * len(t))
        else:
            out.append([y, x, [(x, t)], x + 5 * len(t)])
    return [(y, x, ''.join(t for _, t in sorted(ps))) for y, x, ps, _ in out]


def printed_pages_of(doc):
    """The page count the footers print ('Total Pages 36 - Page 3', 'ਕੁੱਲ ਪੰਨੇ 36 - ਪੰਨਾ 3'): the most frequent first number of the
    bottom 40 pt of each page. For parsers that do not read it (English)."""
    nums = []
    for page in doc:
        r = page.rect
        for line in page.get_text(clip=pymupdf.Rect(0, r.height - 40, r.width, r.height)).splitlines():
            m = FOOTER_RE.search(line.strip())
            if m: nums.append(int(m.group(1)))
    return collections.Counter(nums).most_common(1)[0][0] if nums else None


# --------------------------------------------------------------------------- box

def split_label(t):
    """'label : value' -> (label, value); further ':' right after the first belong to the label."""
    if ':' not in t:
        return None, t
    a, b = t.split(':', 1)
    return a.strip(), re.sub(r'^[\s:]+', '', b).strip()


AGE_ROW = re.compile(r'^(.*?)\s*:[\s:]*(\S+)\s*(.*?)\s*:[\s:]*(.*)$')


def is_age_row(t):
    m = AGE_ROW.match(t)
    return bool(m and toint(m.group(2)) is not None and m.group(3))


def parse_box(box, chars, tchars, lang, photo_texts):
    rep, warn, rec = collections.Counter(), [], {}
    rx = box.x0 + 0.72 * box.width
    spans = collections.OrderedDict()
    for c in chars:
        spans.setdefault(c[3], []).append(c)
    drop = set()
    for sid, cs in spans.items():
        if min(c[1] for c in cs) >= rx and min(c[2] for c in cs) > box.y0 + 0.25 * box.height:
            if cs[0][4].strip() in photo_texts:
                drop.add(sid)
            else:
                warn.append('right_span')
    chars = [c for c in chars if c[3] not in drop]
    rows = rows_of(chars, key_y=lambda c: c[2])
    top_y = box.y0 + 0.25 * box.height
    raw_lines, body = [], []
    for r in rows:
        raw = row_text(r['c'])
        raw_lines.append(raw)
        if r['y'] < top_y:
            for tok in row_text(r['c'], gap=0.5).split():
                if re.fullmatch(r'\d+', tok) and 'serial_no' not in rec:
                    rec['serial_no'] = int(tok)
                elif EPIC_ANY.match(tok) and 'epic' not in rec:
                    rec['epic'] = tok
                elif (EPIC_ODD.match(tok) or EPIC_LOOSE.match(tok)) and len(tok) >= 6 and 'epic' not in rec:
                    rec['epic'] = tok; warn.append('epic_nonstandard')
                else:
                    rec.setdefault('_top', []).append(tok)
            continue
        t = repair(raw, lang)
        if t:
            body.append((t, raw, r['c']))
    # fields: a row with ':' starts one, a row without ':' continues the one before; the age row is found by its form
    fields = []      # [label_kind_or_None, text, raw, chars]
    for t, raw, cs in body:
        if ':' in t or not fields:
            fields.append([None, t, raw, list(cs)])
        elif fields and not is_age_row(fields[-1][1]):
            fields[-1][1] += ' ' + t; fields[-1][2] += ' ' + raw; fields[-1][3] += cs
            warn.append('wrapped_line:%d' % (len(fields) - 1))
        else:
            rec.setdefault('extra_text', []).append(t); warn.append('extra_text')
    ai = next((i for i, f in enumerate(fields) if is_age_row(f[1])), None)
    if ai is not None and ai >= 3:
        kinds = ['name', 'rel', 'house'] + [None] * (ai - 3) + ['age']
    elif ai is not None:
        kinds = (['name', 'rel', 'house'][:ai] + ['age'])
        warn.append('layout:%d_rows_before_age' % ai)
    else:
        kinds = ['name', 'rel', 'house', 'age'][:len(fields)]
        warn.append('layout:no_age_row')
    for i, f in enumerate(fields):
        k = kinds[i] if i < len(kinds) else None
        if k is None:
            rec.setdefault('extra_text', []).append(f[1]); warn.append('extra_text')
        f[0] = k
    labels = {}
    for k, t, raw, cs in fields:
        if k == 'age':
            m = AGE_ROW.match(t)
            if m:
                labels['age'], labels['gender'] = m.group(1).strip(), m.group(3).strip()
                rec['age'] = toint(m.group(2)) if toint(m.group(2)) is not None else m.group(2)
                rec['gender'] = m.group(4).strip()
            continue
        if k is None:
            continue
        lab, val = split_label(t)
        labels[k] = lab
        if k == 'name': rec['name'] = val
        elif k == 'rel': rec['relation_name'] = val
        elif k == 'house': rec['house_no'] = val
    # count the repairs on values only (labels carry artifacts too, 'ਲਿੰੰਗ': not counted); age row: the gender value
    for k, t, raw, cs in fields:
        if k is not None and ':' in raw:
            repair(raw.rsplit(':', 1)[1] if k == 'age' else raw.split(':', 1)[1], lang, rep)
    if 'rel' in labels:
        rec['relation_label'] = labels['rel'] or ''
        rt = REL.get(lang, {}).get(re.sub(r'\s+', ' ', labels['rel'] or ''))
        rec['relation_type'] = rt
        if rt is None:
            warn.append('relation_label_unknown')
    if 'gender' in rec:
        rec['gender_std'] = GENDER.get(lang, {}).get(rec['gender'])
    rec['labels'] = labels
    for k in ('name', 'relation_name', 'house_no', 'age', 'gender', 'serial_no', 'epic'):
        if k not in rec or rec[k] in ('', None):
            warn.append('missing:' + k)
    if rec.get('_top'):
        rec['status_code'] = ' '.join(rec.pop('_top'))
    rec.pop('_top', None)
    # glyph cross-check per field
    for k, t, raw, cs in fields:
        if k is None or not cs:
            continue
        ys = [c[2] for c in cs]; xs = [c[1] for c in cs]
        lo, hi = min(ys) - 2.5, max(ys) + 2.5
        gl = ''.join(tc[0] for tc in tchars if lo <= tc[2] <= hi and min(xs) - 2 <= tc[1] <= max(xs) + 8)
        a, b = letters(t, lang), letters(gl, lang, trace=True)
        fk = {'name': 'name', 'rel': 'relation_name', 'house': 'house_no'}.get(k)
        if a != b and lang in RA and fk:
            # the text layer drops the ra-phala of some conjuncts ('শুক্া' for 'শুক্রা', 'মেক্ী' for 'মেক্ৰী'): restored when the
            # glyph trace has exactly one more র per virama-before-vowel-sign and nothing else differs
            plus, minus, v = a - b, b - a, rec.get(fk) or ''
            n = len(DANGLING.findall(v))
            miss = next(iter(minus)) if len(minus) == 1 else None
            if n and not plus and miss in ('\u09b0', '\u09af') and minus[miss] == n:   # র (ASM: ৰ) or য
                rec[fk] = DANGLING.sub('\u09cd' + (RA[lang] if miss == '\u09b0' else miss), v)
                rep['ra_phala_restored'] += 1
                warn.append('ra_phala_restored:' + k)
                continue
        if a != b and lang == 'TAM' and fk:
            # the text layer drops consonant + e-sign of some o/oo syllables ('தேன்ாழி' for 'தேன்மொழி', 'பூங்ாடி' for
            # 'பூங்கொடி'): an aa-sign right after a virama, a space or the start. Restored when the glyph trace has exactly
            # one consonant and one e/ee sign more per such aa-sign (the same consonant each time) and nothing else differs.
            plus, minus, v = a - b, b - a, rec.get(fk) or ''
            n = len(TAM_DANGLING.findall(v))
            cons = [c for c in minus if '\u0b95' <= c <= '\u0bb9']
            es = [c for c in minus if c in '\u0bc6\u0bc7']
            if (n and not plus and len(cons) == 1 and len(es) == 1 and len(minus) == 2 and minus[cons[0]] == n
                    and minus[es[0]] == n):
                rec[fk] = unicodedata.normalize('NFC', TAM_DANGLING.sub(cons[0] + es[0] + '\u0bbe', v))
                rep['tamil_o_restored'] += 1
                warn.append('tamil_o_restored:' + k)
                continue
        if a != b:
            warn.append('glyph_mismatch:%s:+%s:-%s' % (k, ''.join(sorted((a - b).elements())), ''.join(sorted((b - a).elements()))))
    rec['raw_lines'] = raw_lines
    return rec, rep, warn


# --------------------------------------------------------------------------- document

def box_marks(page, boxes, drawings):
    """As parse_textlayer.box_marks (drawings and images inside a box that are not part of the box template: a possible
    deletion stamp), but the serial cell strokes may be up to 75 pt wide: 4-digit serials widen the cell (Malayalam 65.25 pt)."""
    marks = collections.Counter()
    for d in drawings:
        r = d['rect']
        w, h = r.width, r.height
        if (h < 1.5 and (180 <= w <= 192 or 30 <= w <= 75)) or (w < 1.5 and (8 <= h <= 16 or 50 <= h <= 80)):
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


def photo_texts_of(page, boxes, per_box):
    """Span texts in the right part of the boxes that repeat in >= 30 % of the boxes: the photo placeholder."""
    cnt = collections.Counter()
    for i, b in enumerate(boxes):
        rx = b.x0 + 0.72 * b.width
        seen = set()
        for c in per_box.get(i, []):
            if c[1] >= rx and c[2] > b.y0 + 0.25 * b.height:
                seen.add(c[4].strip())
        cnt.update(seen)
    n = max(1, len(boxes))
    return {t for t, k in cnt.items() if k >= max(2, 0.3 * n) and t}


def parse_pdf(path, lang=None):
    t0 = time.time()
    doc = pymupdf.open(path)
    fn = os.path.basename(path)
    m = FILE_RE.search(fn)
    st_f, ac_f, lang_f, part_f = (m.group(2), int(m.group(3)), m.group(4), int(m.group(5))) if m else (None, None, None, None)
    lang = lang or lang_f
    meta = {'file': fn, 'lang': lang, 'state_code': st_f, 'pages': len(doc)}
    cov = parse_cover(doc[0]) if len(doc) else None
    if not cov and len(doc) > 1:
        cov = parse_cover(doc[1])
    if cov: meta['cover'] = cov
    for k in range(len(doc) - 1, max(0, len(doc) - 4), -1):
        s = parse_summary(doc[k])
        if s and not find_boxes(doc[k], doc[k].get_drawings()):
            meta['summary'] = s; break
    records, section, footers, parts_hdr, acs_hdr = [], (None, None), [], [], []
    photo_doc = set()   # photo placeholder texts of the earlier pages (a last page with 1 box can not show the repeat)
    repairs = collections.Counter()
    for pno, page in enumerate(doc):
        chars, plines = page_chars(page)
        drawings = page.get_drawings()
        boxes = find_boxes(page, drawings)
        lines = merged_lines(plines, boxes)
        for y, x, t in lines:
            if y > page.rect.height - 40:
                fm = FOOTER_RE.search(t)
                if fm: footers.append(int(fm.group(1)))
        if not boxes:
            continue
        # header: the first line holds AC (left) and part (right); section lines are the other lines above the footer
        sec_marks = []
        for j, (y, x, t) in enumerate(lines):
            if y > page.rect.height - 40:
                continue
            if j == 0:
                pm = PART_RE.search(t)
                if pm: parts_hdr.append(int(pm.group(1)))
                am = SECTION_RE.search(t)
                if am: acs_hdr.append(int(am.group(1)))
                continue
            sm = SECTION_RE.search(t)
            if sm:
                sec_marks.append((y, int(sm.group(1)), repair(sm.group(2), lang)))
        trace = page.get_texttrace()
        tch = [(chr(cp), org[0], org[1], gid) for sp in trace for cp, gid, org, bb in sp['chars']]
        own = assign([(c[1], c[2], c[3]) for c in chars], boxes)
        town = assign([(c[1], c[2], -1 - i) for i, c in enumerate(tch)], boxes)
        per_box, per_tbox = collections.defaultdict(list), collections.defaultdict(list)
        for c, k in zip(chars, own):
            if k is not None: per_box[k].append(c)
        for c, k in zip(tch, town):
            if k is not None: per_tbox[k].append(c)
        photo = photo_texts_of(page, boxes, per_box) | photo_doc
        photo_doc |= photo
        marks = box_marks(page, boxes, drawings)
        for i, b in enumerate(boxes):
            for y, no, name in sec_marks:
                if y < b.y0: section = (no, name)
            rec, rep, warn = parse_box(b, per_box[i], per_tbox[i], lang, photo)
            if 'epic' not in rec and 'serial_no' not in rec and 'name' not in rec:
                continue
            if marks.get(i):
                rec['box_marks'] = marks[i]; warn.append('box_marks')
            out = dict(state=st_f, ac=ac_f, part=part_f, page=pno + 1, section_no=section[0], section_name=section[1])
            for k in ('serial_no', 'epic', 'name', 'relation_type', 'relation_label', 'relation_name', 'house_no', 'age',
                      'gender', 'gender_std', 'status_code', 'extra_text', 'box_marks', 'labels', 'raw_lines'):
                if k in rec: out[k] = rec[k]
            if rep: out['repairs'] = sorted(rep)
            if warn: out['text_warnings'] = warn
            for k in rep: repairs[k] += 1
            records.append(out)
    if footers: meta['printed_pages'] = collections.Counter(footers).most_common(1)[0][0]
    if parts_hdr: meta['part_no'] = collections.Counter(parts_hdr).most_common(1)[0][0]
    if acs_hdr: meta['ac_no'] = collections.Counter(acs_hdr).most_common(1)[0][0]
    meta['repairs'] = dict(repairs)
    meta['seconds'] = round(time.time() - t0, 3)
    return records, meta


def validate(records, meta):
    v = {}
    cov = meta.get('cover') or {}
    n = len(records)
    v['records'] = n; v['cover_total'] = cov.get('total')
    g = collections.Counter(r.get('gender_std') for r in records)
    v['by_gender'] = {'male': g['M'], 'female': g['F'], 'third_gender': g['T']}
    v['count_match'] = (n == cov.get('total') and g['M'] == cov.get('male') and g['F'] == cov.get('female')
                        and g['T'] == cov.get('third_gender'))
    if meta.get('summary'):
        s = meta['summary']
        v['summary_total'] = s['total']
        v['summary_match'] = (s['total'] == n and s['male'] == g['M'] and s['female'] == g['F'])
    if meta.get('printed_pages'):
        v['pages_match'] = meta['printed_pages'] == meta.get('pages')
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
    v['gender_bad'] = [(r.get('serial_no'), r.get('gender')) for r in records if not r.get('gender_std')][:20]
    v['with_warnings'] = sum(1 for r in records if r.get('text_warnings'))
    v['warning_kinds'] = dict(collections.Counter(w.split(':')[0] for r in records for w in r.get('text_warnings', [])))
    v['relation_labels'] = dict(collections.Counter(r.get('relation_label') for r in records))
    v['ok'] = bool(v['count_match'] and v['serial_continuous'] and not bad and not v['age_bad'] and not v['gender_bad']
                   and v.get('summary_match', True) and v.get('pages_match', True))
    return v


def hint_of_any(r):
    """A record of this parser -> the --hint-2027 record form (as hint_of_eng: relation_label = printed label)."""
    return {"epic": r["epic"].upper(), "name": r.get("name"), "relation_label": r.get("relation_label"),
            "relation_name": r.get("relation_name"), "house_no": r.get("house_no"), "age": str(r.get("age")),
            "gender": r.get("gender"), "serial_no": str(r.get("serial_no")), "state": r.get("state"), "ac": str(r["ac"]),
            "part": str(r["part"]), **({"epic_printed": r["epic"]} if r["epic"] != r["epic"].upper() else {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pdf', nargs='+')
    ap.add_argument('--lang')
    ap.add_argument('-o', '--outdir')
    ap.add_argument('--summary')
    a = ap.parse_args()
    if a.outdir: os.makedirs(a.outdir, exist_ok=True)
    allsum = []
    for p in a.pdf:
        recs, meta = parse_pdf(p, a.lang)
        val = validate(recs, meta)
        meta['validation'] = val; allsum.append(meta)
        if a.outdir:
            with open(os.path.join(a.outdir, os.path.basename(p)[:-4] + '.jsonl'), 'w', encoding='utf-8') as fh:
                for r in recs: fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('%s  records=%d cover=%s summary=%s ok=%s  %.2fs (%d pages)  repairs=%s warnings=%s' % (
            meta['file'], val['records'], val['cover_total'], val.get('summary_total'), val['ok'], meta['seconds'], meta['pages'],
            meta['repairs'], val['warning_kinds']))
        if not val['ok']:
            print('   ', json.dumps({k: val[k] for k in val if k not in ('ok', 'relation_labels', 'warning_kinds')}, ensure_ascii=False))
    if a.summary:
        with open(a.summary, 'w') as fh: json.dump(allsum, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
