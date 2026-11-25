"""Punjabi (Gurmukhi) SIR rolls (Punjab S19; 2026-11-25, job lang-pan): engine text of the 4 box lines -> the 10 fields.

The Punjabi box uses the Hindi box template (segment.segment_pan) with these printed labels (2027 text layer of all 117 ACs and
the SIR 2026 page images):
  name      'ਨਾਮ : X'
  relative  'ਪਿਤਾ ਦਾ ਨਾਮ: Y' | 'ਪਤੀ ਦਾ ਨਾਮ: Y' | 'ਮਾਤਾ ਦਾ ਨਾਮ: Y' | 'ਪਤਨੀ ਦਾ ਨਾਮ' | 'ਹੋਰ' | 'ਹੋਰ ਦਾ ਨਾਮ' | 'ਸਰਪ੍ਰਸਤ ਦਾ ਨਾਮ'
  house     'ਮਕਾਨ ਨੰ. : Z'
  age, sex  'ਉਮਰ : N ਲਿੰਗ : ਪੁਰਸ਼|ਇਸਤਰੀ|ਤੀਜਾ ਲਿੰਗ'
Values are kept in the printed Gurmukhi form (relation label and gender are snapped to the closest printed variant), so they
compare directly with the 2027 hints of parse_textlayer_any (relation_label = printed label, gender = printed word).
Digits are Latin in the print; Gurmukhi digits read by an engine are mapped to Latin for serial, part and age.
"""
import re, unicodedata
from rapidfuzz import process, fuzz

REL_PAN = ['ਪਿਤਾ ਦਾ ਨਾਮ', 'ਪਤੀ ਦਾ ਨਾਮ', 'ਮਾਤਾ ਦਾ ਨਾਮ', 'ਪਤਨੀ ਦਾ ਨਾਮ', 'ਹੋਰ', 'ਹੋਰ ਦਾ ਨਾਮ', 'ਸਰਪ੍ਰਸਤ ਦਾ ਨਾਮ']
GENDER_PAN = ['ਪੁਰਸ਼', 'ਇਸਤਰੀ', 'ਤੀਜਾ ਲਿੰਗ']
REL_PAN = [unicodedata.normalize('NFC', x) for x in REL_PAN]
GENDER_PAN = [unicodedata.normalize('NFC', x) for x in GENDER_PAN]
GUR2LAT = str.maketrans('੦੧੨੩੪੫੬੭੮੯', '0123456789')
CODE_RE = re.compile(r'^\s*([#A-Z][0-9A-Z]?)\s+([0-9੦-੯]+)\s*$')
NAME_LAB = re.compile(r'^\s*ਨਾ[ਮਸ]\s*[:;]?\s*')
HOUSE_LAB = re.compile(r'^\s*ਮਕਾਨ\s*ਨੰ?\s*\.?\s*[:;]?\s*')
REL_LAB = re.compile(r'^\s*((?:ਪਿਤਾ|ਪਤੀ|ਮਾਤਾ|ਪਤਨੀ|ਹੋਰ|ਸਰਪ੍ਰਸਤ)(?:\s*ਦਾ\s*ਨਾਮ)?)\s+(?=\S)')
AGE_LAB = r'(?:ਉਮਰ)'
LABEL_COLON = re.compile(r'(ਨਾਮ|ਨੰ\.|ਉਮਰ|ਲਿੰਗ|ਹੋਰ)\s*[;:]')
EPIC_RE = re.compile(r'^([A-Z]{3}[0-9]{7}|[A-Z]{2}/[0-9]{2}/[0-9]{3}/[0-9]{6,7})$')


def norm(s):
    s = unicodedata.normalize('NFC', s or '')
    s = s.replace('|', ' ').replace('‌', '').replace('‍', '')
    return re.sub(r'\s+', ' ', s).strip(' :;_')


def fix_raw(t):
    t = unicodedata.normalize('NFC', t or '')
    t = LABEL_COLON.sub(lambda m: m.group(1) + ':', t)
    return re.sub(r'([਼੍ੰੱਂ])\1+', r'\1', t)   # doubled nukta / virama / tippi / addak / bindi


def after_colon(s, lab=None):
    s = s.replace('：', ':')
    if ':' in s: return s.split(':', 1)[1]
    if lab is not None:
        m = lab.match(s)
        if m and m.end(): return s[m.end():]
    return re.sub(r'^\S+\s*', '', s)


def snap(v, choices, cutoff=60):
    m = process.extractOne(v, choices, scorer=fuzz.ratio)
    return m[0] if m and m[1] >= cutoff else v


def fix_epic(e):
    e = re.sub(r'\s+', '', e.upper())
    if '/' in e:
        return e.translate(str.maketrans('OIlSB', '01158'))
    m = re.match(r'^([A-Z0-9]{3})([A-Z0-9]{6,8})$', e)
    if m:
        a = m.group(1).translate(str.maketrans('0158', 'OISB')); b = m.group(2).translate(str.maketrans('OISBZ', '01582'))
        return a + b
    return e


def from_lines_pan(f, pp=True):
    """f: dict with keys serial, (code), (part), epic, line0..line3 (raw engine text)."""
    if pp: f = {k: fix_raw(v) if k.startswith('line') else v for k, v in f.items()}
    out = dict(code='', serial_no='', part_no='', epic='', name='', relation_label='', relation_name='', house_no='', age='', gender='')
    s = norm(f.get('serial', ''))
    if 'code' in f:
        out['code'] = re.sub(r'\s+', '', f['code']).upper().replace('$', 'S')
        d = re.findall(r'[0-9੦-੯]+', s); s = d[-1] if d else s
    m = CODE_RE.match(s)
    if m and not out['code']: out['code'], out['serial_no'] = m.group(1), m.group(2)
    else:
        d = re.findall(r'[0-9੦-੯]+', s); out['serial_no'] = d[-1] if d else s
        pre = s[: s.rfind(out['serial_no'])].strip() if out['serial_no'] in s else ''
        if pre and not out['code']: out['code'] = pre.replace(' ', '')
    out['part_no'] = norm(f.get('part', ''))
    out['epic'] = norm(f.get('epic', ''))
    out['name'] = norm(after_colon(f.get('line0', ''), NAME_LAB))
    l1 = f.get('line1', '')
    if ':' in l1:
        a, b = l1.split(':', 1); out['relation_label'], out['relation_name'] = norm(a), norm(b)
    else:
        m = REL_LAB.match(l1)
        if m: out['relation_label'], out['relation_name'] = norm(m.group(1)), norm(l1[m.end():])
        else: out['relation_name'] = norm(l1)
    out['house_no'] = norm(after_colon(f.get('line2', ''), HOUSE_LAB))
    l3 = norm(f.get('line3', ''))
    m = re.search(r':\s*([0-9੦-੯OoIl]+)', l3)
    if not m: m = re.search(AGE_LAB + r'\s*[;]?\s*([0-9੦-੯]+)(?=\s|$)', l3)
    if not m: m = re.match(r'([0-9੦-੯]+)(?=\s|$)', l3)
    if m: out['age'] = m.group(1)
    parts = l3.split(':')
    if len(parts) >= 3: out['gender'] = norm(parts[-1])
    elif parts: out['gender'] = norm(re.sub(r'.*ਲਿੰ?ਗ', '', parts[-1]))
    if pp:
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(GUR2LAT).translate(str.maketrans('OoIl', '0011')))
        out['epic'] = fix_epic(out['epic'])
        if out['gender']: out['gender'] = snap(out['gender'], GENDER_PAN)
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], REL_PAN)
    return out


def valid_pan(f, v):
    if f == 'epic': return bool(EPIC_RE.match(v))
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in GENDER_PAN
    if f == 'relation_label': return v in REL_PAN
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'): return bool(v) and not re.search(r'[A-Za-z|;ऀ-ॿ]', v)
    return True


SECTION_RE = re.compile(r'ਸੈਕ\S*\s+ਨੰ\S*\s+ਅਤੇ\s+ਨਾਮ\s*[:;.]?\s*([0-9]+)\s*-?\s*(.*)')


def parse_heading_pan(t):
    """'ਸੈਕਸ਼ਨ ਨੰ. ਅਤੇ ਨਾਮ 1-ਮਾਂਝੀ' -> '1-ਮਾਂਝੀ' (None if the line is not a section heading)."""
    m = SECTION_RE.search(unicodedata.normalize('NFC', t or ''))
    return f'{m.group(1)}-{m.group(2).strip()}' if m else None
