"""Gujarati roll boxes (S06 SIR 2026): engine text of the cut lines -> the 10 record fields (2026-11-25, job lang-guj).

A Gujarati box prints 4 labelled lines; a long name or relative name wraps to an unlabelled next line (segment.segment_guj
gives line0..lineN):
    મતદારનું નામ : <name>
    પિતાનું નામ: <relative>   (પતિનું નામ / માતાનું નામ / અન્ય ...)
    ઘર નંબર : <house>
    ઉંમર : <age> જાતિ : <પુરુષ | સ્ત્રી | ...>
The print sets the line in Noto Sans (Latin digits, ':' and '-' from Noto Sans) with Gujarati letters from Lohit Gujarati.
The role logic (vote over engines, order, wrap lines) is the one of parse_eng (box_roles / canonical / canonical_conf take the
role function as a parameter here). Output keys = parse.from_lines.
relation_label = the printed label without the colon (as the English gold), e.g. 'પિતાનું નામ'.
"""
import collections, re, unicodedata
from rapidfuzz import fuzz, process

NAME_LABEL = 'મતદારનું નામ'
LABELS = ['પિતાનું નામ', 'પતિનું નામ', 'માતાનું નામ', 'અન્ય', 'પત્નીનું નામ']
GENDERS = ['પુરુષ', 'સ્ત્રી', 'ત્રીજી જાતિ', 'અન્ય']
HOUSE_LABEL = 'ઘર નંબર'
AGE_LABEL = 'ઉંમર'
GEN_LABEL = 'જાતિ'
GUJ2LAT = str.maketrans('૦૧૨૩૪૫૬૭૮૯', '0123456789')
COLON = re.compile(r'\s*[:;ઃ：!|](?=\s|$|[઀-૿0-9A-Za-z(])')
ROLE_ORDER = ['name', 'rel', 'house', 'age']


def norm(s):
    s = unicodedata.normalize('NFC', s or '').replace('‌', '').replace('‍', '')
    return re.sub(r'\s+', ' ', s).strip(' ;_')


def _prefix(t):
    m = COLON.search(t)
    if m and m.start() <= 16: return t[:m.start()].strip()
    return ' '.join(t.split()[:2])


def line_role(text):
    t = norm(text)
    if not t: return None
    p = _prefix(t)
    if (fuzz.partial_ratio(AGE_LABEL, t[:8]) >= 75 and re.search(r'[0-9૦-૯]', t)) or fuzz.partial_ratio(GEN_LABEL + ' :', t) >= 85 \
            and re.search(r'[0-9૦-૯]', t):
        return 'age'
    if fuzz.ratio(p, HOUSE_LABEL) >= 70 or t.startswith('ઘર'): return 'house'
    if fuzz.ratio(p, NAME_LABEL) >= 72 or t.startswith('મતદાર'): return 'name'
    m = process.extractOne(p, LABELS, scorer=fuzz.ratio)
    if m and m[1] >= 70: return 'rel'
    if re.match(r'^(પિતા|પતિ|માતા|પત્ની)', t) and 'નામ' in t[:16]: return 'rel'
    if t.startswith('અન્ય') and COLON.search(t[:8] + ' '): return 'rel'
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
        out[f'line{j}'] = _join(texts[r])
    return out


def _join(parts):
    """wrapped lines joined with one space; no space after a trailing hyphen ('એન-' + '54' -> 'એન-54', gold convention)"""
    s = ''
    for p in parts:
        s = (s + p) if s.endswith('-') else (s + ' ' + p if s else p)
    return s


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


def fix_epic(e):
    e = re.sub(r'\s+', '', (e or '').upper())
    if 0 < e.count('/') < 3 and re.fullmatch(r'[A-Z0-9]{3}[0-9/]{7,8}', e) and len(e.replace('/', '7')) == 10:
        e = e.replace('/', '7')
    if '/' in e:
        return e.translate(str.maketrans('OIlSB', '01158'))
    m = re.match(r'^([A-Z0-9]{3})([A-Z0-9]{6,8})$', e)
    if m:
        a = m.group(1).translate(str.maketrans('0158', 'OISB')); b = m.group(2).translate(str.maketrans('OISBZ', '01582'))
        return a + b
    return e


def snap(v, choices, cutoff=60):
    m = process.extractOne(v, choices, scorer=fuzz.ratio)
    return m[0] if m and m[1] >= cutoff else v


def from_lines_guj(f, pp=True):
    out = dict(code='', serial_no='', part_no='', epic='', name='', relation_label='', relation_name='', house_no='', age='', gender='')
    s = norm(f.get('serial', '')).translate(GUJ2LAT)
    if 'code' in f:
        out['code'] = re.sub(r'\s+', '', f['code']).upper().replace('$', 'S')
    d = re.findall(r'[0-9]+', s)
    out['serial_no'] = d[-1] if d else s
    if not out['code'] and d:
        pre = s[: s.rfind(d[-1])].strip()
        if pre: out['code'] = pre.replace(' ', '')
    out['part_no'] = norm(f.get('part', ''))
    out['epic'] = norm(f.get('epic', ''))
    out['name'] = norm(_value(norm(f.get('line0', '')), 2))
    l1 = norm(f.get('line1', ''))
    m = COLON.search(l1)
    if m and m.start() <= 18:
        out['relation_label'], out['relation_name'] = norm(l1[:m.start()]), norm(l1[m.end():])
    else:
        w = l1.split()
        k = 1 if w and w[0].startswith('અન્ય') else 2
        out['relation_label'], out['relation_name'] = ' '.join(w[:k]), ' '.join(w[k:])
    out['house_no'] = norm(_value(norm(f.get('line2', '')), 2))
    l3 = norm(f.get('line3', '')).translate(GUJ2LAT)
    m = re.search(r'^\W*\S{1,6}\s*[:;ઃ!|]?\s*([0-9OoIlSB]{1,3})\b', l3) or re.search(r'[:;ઃ!]\s*([0-9OoIl]{1,3})\b', l3)
    if m: out['age'] = m.group(1)
    g = re.search(r'જા\S{0,3}\s*[:;ઃ!|]?\s*(.*)$', l3[m.end():] if m else l3)
    if g: out['gender'] = norm(g.group(1))
    elif ':' in l3: out['gender'] = norm(l3.split(':')[-1])
    if pp:
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(GUJ2LAT).translate(str.maketrans('OoIlSB', '001158')))
        out['epic'] = fix_epic(out['epic'])
        if out['gender']: out['gender'] = snap(out['gender'], GENDERS)
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], LABELS, 70)
        for k in ('name', 'relation_name'):
            out[k] = re.sub(r'^[^઀-૿A-Za-z(]+', '', out[k]).strip(' :;,_ઃ')
        out['house_no'] = re.sub(r'^[:;!|ઃ]+\s*', '', out['house_no'].translate(GUJ2LAT)).strip(' ;_|')
    return out


GENDER_SET = set(GENDERS)
LABEL_SET = set(LABELS)
NAME_RE = re.compile(r'^[઀-૿][઀-૿ .\-()]*$')


def valid_guj(f, v, epic_re=None):
    if f == 'epic': return bool(epic_re.match(v)) if epic_re else bool(v)
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in GENDER_SET
    if f == 'relation_label': return v in LABEL_SET
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'): return bool(NAME_RE.match(v or ''))
    return True
