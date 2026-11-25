"""Tamil roll boxes (S22 Tamil Nadu + U07 Puducherry SIR 2026): engine text of the cut lines -> the 10 record fields
(2026-11-25, job lang-tam).

The Tamil SIR box is the English SIR template (segment.segment_eng gives line0..lineN) with Tamil labels:
    பெயர் : <name>
    தந்தையின் பெயர்: <relative>  (U07 and the 2027 rolls: தந்தை பெயர்; கணவர் பெயர் / தாயின் பெயர் / தாயார் பெயர் / இதரர் பெயர்)
    வீட்டு எண் : <house>
    வயது : <age> பாலினம் : <ஆண் | பெண் | மூன்றாம் பாலினம்>
A long name / relative name / house number wraps to an unlabelled next line. The role logic (vote over engines, order, wrap
lines) is the one of parse_eng; only the label tests differ. Output keys = parse.from_lines.
relation_label = the printed label without the colon (as the gold), e.g. 'தந்தையின் பெயர்'.
Text normalisation (norm): NFC, ZWJ / ZWNJ dropped, spaces collapsed. Visual folds for comparisons only (vis): ௌ = ெள
(one glyph sequence), ஂ = ் (both a dot above).
"""
import collections, re, unicodedata
from rapidfuzz import fuzz, process

NAME_LABEL = 'பெயர்'
LABELS = ['தந்தையின் பெயர்', 'தந்தை பெயர்', 'கணவர் பெயர்', 'கணவரின் பெயர்', 'தாயின் பெயர்', 'தாய் பெயர்', 'தாயார் பெயர்',
          'இதரர் பெயர்', 'இதரர்', 'மனைவி பெயர்', 'மனைவியின் பெயர்']
GENDERS = ['ஆண்', 'பெண்', 'மூன்றாம் பாலினம்']
HOUSE_LABEL = 'வீட்டு எண்'
AGE_LABEL = 'வயது'
GEN_LABEL = 'பாலினம்'
COLON = re.compile(r'\s*[:;：!|ः](?=\s|$|[஀-௿0-9A-Za-z(])')
ROLE_ORDER = ['name', 'rel', 'house', 'age']
TAM = '஀-௿'


def norm(s):
    s = unicodedata.normalize('NFC', s or '').replace('‌', '').replace('‍', '').replace(' ', ' ')
    return re.sub(r'\s+', ' ', s).strip(' ;_')


def vis(s):
    """visual form: letters the print draws the same are one letter (for comparisons with the gold / the 2027 hint)"""
    return norm(s).replace('ௌ', 'ெள').replace('ஂ', '்')


def _prefix(t):
    m = COLON.search(t)
    if m and m.start() <= 20: return t[:m.start()].strip()
    return ' '.join(t.split()[:2])


def line_role(text):
    t = norm(text)
    if not t: return None
    p = _prefix(t)
    if (fuzz.partial_ratio(AGE_LABEL, t[:7]) >= 75 and re.search(r'[0-9]', t)) or (fuzz.partial_ratio(GEN_LABEL, t) >= 85
                                                                                    and re.search(r'[0-9]', t)):
        return 'age'
    if fuzz.ratio(p, HOUSE_LABEL) >= 70 or t.startswith('வீட்') or (t.startswith('வீ') and 'எண்' in t[:12]): return 'house'
    m = process.extractOne(p, LABELS, scorer=fuzz.ratio)
    if m and m[1] >= 72: return 'rel'
    if re.match(r'^(தந்தை|கணவ|தாய|இதர|மனைவி)', t): return 'rel'
    if fuzz.ratio(p, NAME_LABEL) >= 70 or (t.startswith('பெய') and len(p) <= 7): return 'name'
    return None


def box_roles(per_engine_lines, n_lines):
    votes = []
    for i in range(n_lines):
        c = collections.Counter(line_role(L.get(f'line{i}', '')) for L in per_engine_lines)
        lab = [(r, n) for r, n in c.most_common() if r]
        none = c.get(None, 0)
        votes.append(lab[0][0] if lab and lab[0][1] * 2 >= len(per_engine_lines) and lab[0][1] >= none * 0.5 else None)
    roles, seen, last = [], set(), -1
    for r in votes:
        if r and r not in seen and ROLE_ORDER.index(r) > last:
            roles.append(r); seen.add(r); last = ROLE_ORDER.index(r)
        else:
            roles.append(None)
    if len(seen) < 4 and n_lines >= 4:
        for k, r in enumerate(ROLE_ORDER):
            if r in seen: continue
            prev = max([i for i, x in enumerate(roles) if x and ROLE_ORDER.index(x) < k], default=-1)
            nxt = min([i for i, x in enumerate(roles) if x and ROLE_ORDER.index(x) > k], default=n_lines)
            free = [i for i in range(prev + 1, nxt) if roles[i] is None]
            if r == 'age' and free: free = free[:1]
            if free: roles[free[0]] = r; seen.add(r)
    out, cur = [], None
    for r in roles:
        if r: cur = r; out.append(r)
        elif cur is None or cur == 'age': out.append('drop')
        else: out.append(None)
    return out


def canonical(raw, roles):
    out = {k: v for k, v in raw.items() if not k.startswith('line')}
    texts = collections.OrderedDict((r, []) for r in ROLE_ORDER)
    cur = None
    for i, r in enumerate(roles):
        t = norm(raw.get(f'line{i}', ''))
        if r == 'drop': continue
        if r: cur = r
        if cur and t: texts[cur].append(t)
    for j, r in enumerate(ROLE_ORDER):
        out[f'line{j}'] = ' '.join(texts[r])
    return out


def canonical_conf(conf, roles):
    out = {k: v for k, v in conf.items() if not k.startswith('line')}
    cur, acc = None, collections.defaultdict(list)
    for i, r in enumerate(roles):
        if r == 'drop': continue
        if r: cur = r
        if cur and f'line{i}' in conf: acc[cur].append(conf[f'line{i}'])
    for j, r in enumerate(ROLE_ORDER):
        if acc[r]: out[f'line{j}'] = min(acc[r])
    return out


def _value(t, nwords):
    m = COLON.search(t)
    if m and m.start() <= 22: return t[m.end():]
    return ' '.join(t.split()[nwords:])


# lang-tam 2026-11-25 (ECI_TAM_EPIC_L2D8=1, default off): Tamil Nadu has a few old EPICs of 2 letters + 8 digits (XG10861989,
# SX1..., 54 in 8.6 M 2027 records, 3rd character '1' in 30 of them). fix_epic turned the 3rd character into a letter
# (XGI0861989) and all CRNNs agreed: a silent EPIC error (held-out U07-23). With the switch an EPIC that reads as 2 letters +
# '1' + 7 digits is kept as read ('0' stays 'O': Tesseract reads O as 0, gold U07-16 SCO0060897); it fails
# EPIC_RE (3 letters + 7 digits), so the field is flagged; with 2027 hints, epicfix keeps it when the 2027 roll has it.
TAM_EPIC_L2D8 = __import__('os').environ.get('ECI_TAM_EPIC_L2D8', '0') == '1'


def fix_epic(e):
    e = re.sub(r'\s+', '', (e or '').upper())
    if TAM_EPIC_L2D8 and re.fullmatch(r'[A-Z]{2}1[0-9]{7}', e):
        return e
    if 0 < e.count('/') < 3 and re.fullmatch(r'[A-Z0-9]{3}[0-9/]{7,8}', e) and len(e.replace('/', '7')) == 10:
        e = e.replace('/', '7')
    if '/' in e:
        return e.translate(str.maketrans('OIlSB', '01158'))
    m = re.match(r'^([A-Z0-9]{3})([A-Z0-9]{6,8})$', e)
    if m:
        return m.group(1).translate(str.maketrans('0158', 'OISB')) + m.group(2).translate(str.maketrans('OISBZ', '01582'))
    return e


def snap(v, choices, cutoff=60):
    m = process.extractOne(v, choices, scorer=fuzz.ratio)
    return m[0] if m and m[1] >= cutoff else v


def from_lines_tam(f, pp=True):
    """f: canonical dict (serial, part, epic, code, line0..line3). Same output keys as parse.from_lines."""
    out = dict(code='', serial_no='', part_no='', epic='', name='', relation_label='', relation_name='', house_no='', age='', gender='')
    s = norm(f.get('serial', ''))
    if 'code' in f:
        out['code'] = re.sub(r'\s+', '', f['code']).upper().replace('$', 'S')
    d = re.findall(r'[0-9]+', s)
    out['serial_no'] = d[-1] if d else s
    if not out['code'] and d:
        pre = s[: s.rfind(d[-1])].strip()
        if pre: out['code'] = pre.replace(' ', '')
    out['part_no'] = norm(f.get('part', ''))
    out['epic'] = norm(f.get('epic', ''))
    out['name'] = norm(_value(norm(f.get('line0', '')), 1))
    l1 = norm(f.get('line1', ''))
    m = COLON.search(l1)
    if m and m.start() <= 24:
        out['relation_label'], out['relation_name'] = norm(l1[:m.start()]), norm(l1[m.end():])
    else:
        w = l1.split()
        k = 1 if w and w[0].startswith('இதரர்') and not (len(w) > 1 and w[1].startswith('பெய')) else 2
        out['relation_label'], out['relation_name'] = ' '.join(w[:k]), ' '.join(w[k:])
    out['house_no'] = norm(_value(norm(f.get('line2', '')), 2))
    l3 = norm(f.get('line3', ''))
    m = re.search(r'^\W*\S{0,6}\W*([0-9OoIlSB]{1,3})\b', l3) or re.search(r'[:;!]\s*([0-9OoIl]{1,3})\b', l3)
    if m: out['age'] = m.group(1)
    g = re.search(r'பாலின\S*\s*[:;!|]?\s*(.*)$', l3)
    if g: out['gender'] = norm(g.group(1))
    elif ':' in l3: out['gender'] = norm(l3.split(':')[-1])
    if pp:
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(str.maketrans('OoIlSB', '001158')))
        out['epic'] = fix_epic(out['epic'])
        if out['gender']: out['gender'] = snap(out['gender'], GENDERS)
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], LABELS, 70)
        for k in ('name', 'relation_name'):
            v = re.sub(rf'^[^{TAM}A-Za-z(]+', '', out[k])           # colon / noise read before the value
            out[k] = v.strip(' :;,_|')
        out['house_no'] = re.sub(r'^[:;!|]+\s*', '', out['house_no']).strip(' ;_|')
    return out


def rel_canon(v):
    """relation label -> kind (father / husband / mother / other / wife): SIR 2026 and 2027 print other forms"""
    v = norm(v)
    for k, keys in (('father', ('தந்தை',)), ('husband', ('கணவ',)), ('mother', ('தாய',)), ('wife', ('மனைவி',)), ('other', ('இதரர்',))):
        if v.startswith(keys): return k
    return v


GENDER_SET = set(GENDERS)
LABEL_SET = set(LABELS)
NAME_RE = re.compile(rf"^[{TAM}A-Za-z][{TAM}A-Za-z .'()\-]*$")


def valid_tam(f, v, epic_re=None):
    if f == 'epic': return bool(epic_re.match(v)) if epic_re else bool(v)
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in GENDER_SET
    if f == 'relation_label': return v in LABEL_SET
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'): return bool(NAME_RE.match(v or ''))
    return True
