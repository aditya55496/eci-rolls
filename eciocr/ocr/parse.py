"""Turn raw engine text into the 10 gold fields. Same parser for every engine.

pp=True applies cheap, safe rules: Latin digits for serial/part/age, EPIC
character fixes, closed sets for gender and relation label.
"""
import re, unicodedata
from rapidfuzz import process, fuzz

LABELS = ['पिता का नाम', 'पति का नाम', 'माता का नाम', 'अन्य']
GENDERS = ['पुरुष', 'महिला', 'तृतीय लिंग']
DEV2LAT = str.maketrans('०१२३४५६७८९', '0123456789')
CODE_RE = re.compile(r'^\s*([#A-Z][0-9A-Z]?)\s+([0-9०-९]+)\s*$')


def norm(s):
    s = unicodedata.normalize('NFC', s or '')
    s = s.replace('|', ' ').replace('‌', '').replace('‍', '')
    return re.sub(r'\s+', ' ', s).strip(' :;_')


def after_colon(s):
    s = s.replace('：', ':')
    return s.split(':', 1)[1] if ':' in s else re.sub(r'^\S+\s*', '', s)


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


LABEL_COLON = re.compile(r'(नाम|संख्या|आयु|लिंग|लिग|अन्य)\s*[;ःः:]')


def fix_raw(t):
    t = unicodedata.normalize('NFC', t or '')
    t = LABEL_COLON.sub(lambda m: m.group(1) + ':', t)       # label colon read as ; or visarga
    return re.sub('\u094d{2,}', '\u094d', re.sub('\u093c{2,}', '\u093c', t))  # doubled halant / nukta


def from_lines(f, pp=True):
    """f: dict with keys serial, part, epic, line0..line3 (raw text)."""
    if pp: f = {k: fix_raw(v) if k.startswith('line') else v for k, v in f.items()}
    out = dict(code='', serial_no='', part_no='', epic='', name='', relation_label='', relation_name='', house_no='', age='', gender='')
    s = norm(f.get('serial', ''))
    if 'code' in f:  # code cut out by the segmenter
        out['code'] = re.sub(r'\s+', '', f['code']).upper().replace('$', 'S')
        d = re.findall(r'[0-9०-९]+', s); s = d[-1] if d else s
    m = CODE_RE.match(s)
    if m and not out['code']: out['code'], out['serial_no'] = m.group(1), m.group(2)
    else:
        d = re.findall(r'[0-9०-९]+', s); out['serial_no'] = d[-1] if d else s
        pre = s[: s.rfind(out['serial_no'])].strip() if out['serial_no'] in s else ''
        if pre and not out['code']: out['code'] = pre.replace(' ', '')
    out['part_no'] = norm(f.get('part', ''))
    out['epic'] = norm(f.get('epic', ''))
    out['name'] = norm(after_colon(f.get('line0', '')))
    l1 = f.get('line1', '')
    if ':' in l1:
        a, b = l1.split(':', 1); out['relation_label'], out['relation_name'] = norm(a), norm(b)
    else:
        out['relation_name'] = norm(l1)
    out['house_no'] = norm(after_colon(f.get('line2', '')))
    l3 = norm(f.get('line3', ''))
    m = re.search(r':\s*([0-9०-९OoIl]+)', l3)
    if m: out['age'] = m.group(1)
    parts = l3.split(':')
    if len(parts) >= 3: out['gender'] = norm(parts[-1])
    elif parts: out['gender'] = norm(re.sub(r'.*(लिंग|लिग)', '', parts[-1]))
    if pp:
        def dandas(v):  # Tesseract hin reads Latin 1 as danda
            v = re.sub(r'(?<=[0-9०-९/\-])॥|॥(?=[0-9०-९/\-])', '11', v)
            return re.sub(r'(?<=[0-9०-९/\-])।|।(?=[0-9०-९/\-])', '1', v)
        out['age'] = dandas(out['age'] + ('' if not m else l3[m.end(1):m.end(1) + 1] if l3[m.end(1):m.end(1) + 1] in '।॥' else ''))
        out['house_no'] = dandas(out['house_no'])
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(DEV2LAT).translate(str.maketrans('OoIl', '0011')))
        out['epic'] = fix_epic(out['epic'])
        if out['gender']: out['gender'] = snap(out['gender'], GENDERS)
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], LABELS)
    return out


def from_box_lines(lines, box_w=None, pp=True):
    """lines: [{text, x, y}] from a whole-box OCR. Assign by label words."""
    L = sorted(lines, key=lambda l: (round(l['y'] / 8), l['x']))
    f = {}
    rest = []
    for l in L:
        t = norm(l['text'])
        if not t or re.fullmatch(r'(फोटो|उपलब्ध|है|फोटो उपलब्ध|फोटो उपलब्ध है|उपलब्ध है)', t): continue
        if re.search(r'[A-Z]{2,3}\s*[0-9]{6,8}|[A-Z]{2}/[0-9]+/', t) and 'epic' not in f: f['epic'] = re.search(r'([A-Z]{2,3}\s*[0-9]{6,8}|[A-Z]{2}/[0-9/]+)', t).group(1); continue
        if 'आयु' in t or 'लिंग' in t: f['line3'] = t; continue
        if 'मकान' in t or 'संख्या' in t: f['line2'] = t; continue
        if 'का नाम' in t or t.startswith('अन्य'): f['line1'] = t; continue
        if t.startswith('नाम') or ('नाम' in t and ':' in t): f.setdefault('line0', t); continue
        rest.append(t)
    nums = [t for t in rest if re.fullmatch(r'([#A-Z][0-9A-Z]?\s+)?[0-9०-९]+', t)]
    if nums: f['serial'] = nums[0]
    if len(nums) > 1: f['part'] = nums[1]
    return from_lines(f, pp)
