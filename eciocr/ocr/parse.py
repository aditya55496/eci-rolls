"""Turn raw engine text into the 10 gold fields. Same parser for every engine.

pp=True applies cheap, safe rules: Latin digits for serial/part/age, EPIC
character fixes, closed sets for gender and relation label.

Printed label variants (2026-11-25, branch hinvar; collected from the 2027 text layers of 5 states and SIR page images of
11 Hindi-script states/UTs, see models2/hinvar/NOTES.md):
  name     'नाम :' (UP, Uttarakhand, Haryana, Chandigarh) | 'निर्वाचक का नाम :' (Bihar, MP, Rajasthan, Chhattisgarh,
           Jharkhand, Delhi) | 'नाँउ :' (West Bengal Darjeeling ACs, Nepali)
  relative '<पिता|पति|माता> का नाम:' | 'अन्य:' | 'पिता:' (Chandigarh) | 'पिताको नाँउ:' (Nepali); Bihar prints 'पिता का नाम::'
  house    'मकान संख्या :' | 'गृह संख्या :' (MP, Rajasthan, Jharkhand) | 'मकान नं. :' (Chandigarh) | 'घर संख्या :' (Nepali)
  age      'आयु :' | 'उम्र :' (Bihar, MP, Rajasthan, Chhattisgarh, Jharkhand) | 'उमेर :' (Nepali); Bihar prints 'लिंग: :'
  gender   'पुरुष' | 'महिला' | 'तृतीय लिंग' | 'स्त्री' (Nepali, = महिला)
The values always come after the label colon, so the colon split works for every variant. Without a readable colon the
known label words are cut off (before 2026-11-25: the first word only, which left 'का नाम ...' of 'निर्वाचक का नाम').
Relation labels and genders are given in the canonical (UP) form, so the 2027 hints and the validators work unchanged.
"""
import re, unicodedata
from rapidfuzz import process, fuzz

LABELS = ['पिता का नाम', 'पति का नाम', 'माता का नाम', 'अन्य']
GENDERS = ['पुरुष', 'महिला', 'तृतीय लिंग']
# printed relation labels of other states -> canonical label (snap() runs over the keys, then maps)
REL_VARIANTS = {**{l: l for l in LABELS}, 'पिता': 'पिता का नाम', 'पति': 'पति का नाम', 'माता': 'माता का नाम',
                'पिताको नाँउ': 'पिता का नाम', 'पतिको नाँउ': 'पति का नाम', 'आमाको नाँउ': 'माता का नाम', 'माताको नाँउ': 'माता का नाम',
                'अरु': 'अन्य'}
GENDER_VARIANTS = {**{g: g for g in GENDERS}, 'स्त्री': 'महिला'}
DEV2LAT = str.maketrans('०१२३४५६७८९', '0123456789')
CODE_RE = re.compile(r'^\s*([#A-Z][0-9A-Z]?)\s+([0-9०-९]+)\s*$')
# label words at the start of a line (used only when the line has no colon)
NAME_LAB = re.compile(r'^\s*(?:निर्वाचक\s*का\s*)?(?:नाम|नाँउ|नांउ|नाउ)\s*[:;ः]?\s*')
HOUSE_LAB = re.compile(r'^\s*(?:मकान|गृह|घर)\s*(?:संख्या|नं\.?)\s*[:;ः]?\s*')
AGE_LAB = r'(?:आयु|उम्र|उमेर)'
REL_LAB = re.compile(r'^\s*((?:पिता|पति|माता|आमा)(?:\s*का\s*नाम|\s*को\s*नाँउ)?|अन्य)\s+(?=\S)')


def norm(s):
    s = unicodedata.normalize('NFC', s or '')
    s = s.replace('|', ' ').replace('‌', '').replace('‍', '')
    return re.sub(r'\s+', ' ', s).strip(' :;_')


def after_colon(s, lab=None):
    """Value after the label colon. No colon: cut the label words `lab` (regex) when they match, else the first word."""
    s = s.replace('：', ':')
    if ':' in s: return s.split(':', 1)[1]
    if lab is not None:
        m = lab.match(s)
        if m and m.end(): return s[m.end():]
    return re.sub(r'^\S+\s*', '', s)


def canon(v, variants, cutoff=60):
    """Snap to the closest printed variant, return its canonical form (unchanged if nothing is close)."""
    m = process.extractOne(v, list(variants), scorer=fuzz.ratio)
    return variants[m[0]] if m and m[1] >= cutoff else v


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


LABEL_COLON = re.compile(r'(नाम|नाँउ|संख्या|नं\.|आयु|उम्र|उमेर|लिंग|लिग|अन्य|पिता|पति|माता)\s*[;ःः:]')


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
    out['name'] = norm(after_colon(f.get('line0', ''), NAME_LAB))
    l1 = f.get('line1', '')
    if ':' in l1:
        a, b = l1.split(':', 1); out['relation_label'], out['relation_name'] = norm(a), norm(b)
    else:
        m = REL_LAB.match(l1)   # colon not read (CRNNs drop it after Chandigarh's short label 'पिता:')
        if m: out['relation_label'], out['relation_name'] = norm(m.group(1)), norm(l1[m.end():])
        else: out['relation_name'] = norm(l1)
    out['house_no'] = norm(after_colon(f.get('line2', ''), HOUSE_LAB))
    l3 = norm(f.get('line3', ''))
    m = re.search(r':\s*([0-9०-९OoIl]+)', l3)
    if not m: m = re.search(AGE_LAB + r'\s*[;ः]?\s*([0-9०-९]+)(?=\s|$)', l3)   # colon not read
    if not m: m = re.match(r'([0-9०-९]+)(?=\s|$)', l3)   # label not read, colon stripped by norm(): ': 19 लिंग' (c3q on 'उम्र')
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
        if out['gender']: out['gender'] = canon(out['gender'], GENDER_VARIANTS)
        if out['relation_label']: out['relation_label'] = canon(out['relation_label'], REL_VARIANTS)
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
        if re.search(AGE_LAB, t) or 'लिंग' in t: f['line3'] = t; continue
        if HOUSE_LAB.match(t) or 'मकान' in t or 'संख्या' in t: f['line2'] = t; continue
        if t.startswith('निर्वाचक') or t.startswith('नाँउ'): f.setdefault('line0', t); continue
        if 'का नाम' in t or 'को नाँउ' in t or t.startswith('अन्य') or re.match(r'(पिता|पति|माता)\s*:', t): f['line1'] = t; continue
        if t.startswith('नाम') or ('नाम' in t and ':' in t): f.setdefault('line0', t); continue
        rest.append(t)
    nums = [t for t in rest if re.fullmatch(r'([#A-Z][0-9A-Z]?\s+)?[0-9०-९]+', t)]
    if nums: f['serial'] = nums[0]
    if len(nums) > 1: f['part'] = nums[1]
    return from_lines(f, pp)
