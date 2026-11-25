"""Bengali roll boxes (West Bengal SIR 2026; 2026-11-25, job lang-ben): engine text of the cut lines -> the 10 record fields.

The West Bengal SIR print uses the English template (segment.segment_eng: 4 labelled lines + 0-3 unlabelled wrap lines):
    নাম : <name>
    <পিতার নাম | স্বামীর নাম | মাতার নাম | অন্যান্য ...>: <relative>
    বাড়ির নং : <house>                      (Assam 2027: বাড়ী নং)
    বয়স : <age> লিঙ্গ : <পুরুষ | মহিলা | তৃতীয় লিঙ্গ>
Roles are voted over all engines (box_roles), every engine is then parsed with the same roles (canonical), as parse_eng does.
Normalisation: NFC, ZWJ / ZWNJ removed, whitespace collapsed. The Bengali visarga 'ঃ' after a label is read as the colon.
"""
import collections, re, unicodedata
from rapidfuzz import fuzz, process

LABELS = ['পিতার নাম', 'স্বামীর নাম', 'মাতার নাম', 'অন্যান্য', 'মায়ের নাম', 'স্ত্রীর নাম', 'অভিভাবকের নাম']
REL_KIND = {'পিতার নাম': 'father', 'স্বামীর নাম': 'husband', 'মাতার নাম': 'mother', 'মায়ের নাম': 'mother',
            'স্ত্রীর নাম': 'wife', 'অন্যান্য': 'other', 'অভিভাবকের নাম': 'guardian'}
GENDERS = ['পুরুষ', 'মহিলা', 'তৃতীয় লিঙ্গ']
ROLE_ORDER = ['name', 'rel', 'house', 'age']
COLON = re.compile(r'\s*[:;ː：ঃ](?=\s|$|[^ঁ-ঃ়-ৗ])')
BEN_DIG = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')


ASM_TO_BEN = str.maketrans({'ৰ': 'র', 'ৱ': 'ব'})   # the line models also learn Assam 2027 lines; WB rolls never print ৰ / ৱ


def norm(s):
    s = unicodedata.normalize('NFC', s or '').replace('‌', '').replace('‍', '').translate(ASM_TO_BEN)
    s = re.sub('(?<![ডঢয])়', '.', s)   # r2: a nukta exists only under ড ঢ য; elsewhere it is a printed full stop ('হক.' read 'হক়')
    s = s.replace('ত্স', 'ৎস')   # r2: Bengali has no printed t+s conjunct; the Assam text layer (training labels) stores khanda ta as ত্
    return re.sub(r'\s+', ' ', s).strip(' ;_')


def _prefix(t):
    m = COLON.search(t)
    if m and m.start() <= 16: return t[:m.start()].strip()
    return ' '.join(t.split()[:2])


def line_role(text):
    t = norm(text)
    if not t: return None
    p = _prefix(t)
    if p.startswith('বয়') or 'লিঙ্গ' in t[3:] or (fuzz.ratio(p, 'বয়স') >= 60 and re.search(r'[0-9০-৯]', t)):
        return 'age'
    if p.startswith('বাড়') or p.startswith('বাড') or fuzz.ratio(p, 'বাড়ির নং') >= 70 or fuzz.ratio(p, 'বাড়ী নং') >= 70:
        return 'house'
    if fuzz.ratio(p, 'নাম') >= 75: return 'name'
    m = process.extractOne(p, LABELS, scorer=fuzz.ratio)
    if m and m[1] >= 70: return 'rel'
    if p.endswith('নাম') and len(p) > 4: return 'rel'
    return None


def box_roles(per_engine_lines, n_lines):
    """= parse_eng.box_roles with the Bengali line_role."""
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
    if m and m.start() <= 18: return t[m.end():]
    return ' '.join(t.split()[nwords:])


def snap(v, choices, cutoff=60):
    m = process.extractOne(v, choices, scorer=fuzz.ratio)
    return m[0] if m and m[1] >= cutoff else v


def from_lines_ben(f, pp=True):
    """f: canonical dict (serial, part, epic, code, line0..line3) -> the parse.from_lines keys."""
    import parse_eng
    out = dict(code='', serial_no='', part_no='', epic='', name='', relation_label='', relation_name='', house_no='', age='', gender='')
    s = norm(f.get('serial', '')).translate(BEN_DIG)
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
    if m and m.start() <= 18:
        out['relation_label'], out['relation_name'] = norm(l1[:m.start()]), norm(l1[m.end():])
    else:
        w = l1.split()
        k = 1 if w and w[0].startswith('অন্য') else 2
        out['relation_label'], out['relation_name'] = ' '.join(w[:k]), ' '.join(w[k:])
    out['house_no'] = norm(_value(norm(f.get('line2', '')), 2))
    l3 = norm(f.get('line3', ''))
    # (\w does not match Bengali vowel signs / virama / nukta: the label words are matched as non-space runs)
    m = re.search(r'^\W*ব[^\s:;ঃ0-9০-৯]{0,5}\s*[:;ঃ]?\s*([0-9০-৯OoIlSB]{1,3})(?![0-9০-৯])', l3) or re.search(r'[:;ঃ]\s*([0-9০-৯]{1,3})(?![0-9০-৯])', l3)
    if m: out['age'] = m.group(1)
    g = re.search(r'লি[^\s:;ঃ]*\s*[:;ঃ]\s*(.*)$', l3)
    if g: out['gender'] = norm(g.group(1))
    elif re.search(r'[:ঃ]', l3): out['gender'] = norm(re.split(r'[:ঃ]', l3)[-1])
    if pp:
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(BEN_DIG).translate(str.maketrans('OoIlSB', '001158')))
        out['epic'] = fix_epic_ben(out['epic'])
        if out['gender']: out['gender'] = snap(out['gender'], GENDERS)
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], LABELS, 70)
        for k in ('name', 'relation_name'):
            out[k] = re.sub(r'^[\s:;ঃ.,|]+', '', out[k]).strip(' :;,_|')
        out['house_no'] = re.sub(r'^[:;ঃ|]+\s*', '', out['house_no']).strip(' ;_|')
    return out


def fix_epic_ben(e):
    """parse_eng.fix_epic, but old-style EPICs keep their letter prefix: parse_eng maps B -> 8 in the WHOLE string when it has a
    '/', so 'WB/24/161/375538' became 'W8/24/161/375538' (Tesseract baseline on the BEN gold: 105 of 106 EPIC errors)."""
    import parse_eng
    e = re.sub(r'\s+', '', (e or '').upper())
    m = re.match(r'^([A-Z0-9]{2})(/.*)$', e)
    if m and e.count('/') == 3:
        return m.group(1).translate(str.maketrans('80152', 'BOISZ')) + m.group(2).translate(str.maketrans('OIlSBZ', '011582'))
    return parse_eng.fix_epic(e)


def rel_canon(v):
    return REL_KIND.get(norm(v), norm(v))


GENDER_SET = set(GENDERS)
LABEL_SET = set(LABELS)
BEN_LETTER = re.compile(r'[অ-হড়-য়ৰৱ]')
NAME_RE = re.compile(r"^[ঀ-৿ .'()\-]+$")


def valid_ben(f, v, epic_re=None):
    if f == 'epic': return bool(epic_re.match(v)) if epic_re else bool(v)
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in GENDER_SET
    if f == 'relation_label': return v in LABEL_SET
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'): return bool(NAME_RE.match(v or '')) and bool(BEN_LETTER.search(v or ''))
    return True
