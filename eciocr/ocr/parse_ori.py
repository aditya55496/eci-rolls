"""Odia roll boxes (S18 Odisha SIR 2026 draft + final): engine text of the cut lines -> the 10 record fields
(2026-11-25, job lang-ori; structure of parse_tam / parse_eng).

The Odia SIR box is the English SIR template (serial cell, EPIC top right, photo cell) with Odia labels; the lines are cut by
segment.segment_ori (connected components, Odia marks reach above / below the line core):
    ନାମ : <name>
    ପିତାଙ୍କ ନାମ: <relative>  (ସ୍ବାମୀଙ୍କ ନାମ / ମାତାଙ୍କ ନାମ / ଅନ୍ୟ ସମ୍ପର୍କୀୟଙ୍କ ନାମ)
    ଘର ନଂ : <house>          (Latin digits; some houses are printed with Odia digits ୦-୯ and are kept as printed)
    ବୟସ : <age> ଲିଙ୍ଗ : <ପୁରୁଷ | ସ୍ତ୍ରୀ | ତୃତୀୟ ଲିଙ୍ଗ>
A long value wraps to an unlabelled next line. EPIC: 3 letters + 7 digits, or the old form OR/17/123/272006.
Text normalisation (norm) = the gold's canonical form: NFC, ZWJ / ZWNJ dropped, subjoined WA -> subjoined BA and WA (U+0B35)
-> BA (the print font draws them the same), spaces collapsed. The print font draws Latin 8 open at the top and 9 curled:
they look like Odia ୪ / ୨; serial and age are always Latin.
"""
import collections, os, re, unicodedata
from rapidfuzz import fuzz, process

NAME_LABEL = 'ନାମ'
LABELS = ['ପିତାଙ୍କ ନାମ', 'ସ୍ବାମୀଙ୍କ ନାମ', 'ମାତାଙ୍କ ନାମ', 'ଅନ୍ୟ ସମ୍ପର୍କୀୟଙ୍କ ନାମ', 'ପତ୍ନୀଙ୍କ ନାମ']
GENDERS = ['ପୁରୁଷ', 'ସ୍ତ୍ରୀ', 'ତୃତୀୟ ଲିଙ୍ଗ']
HOUSE_LABEL = 'ଘର ନଂ'
AGE_LABEL = 'ବୟସ'
GEN_LABEL = 'ଲିଙ୍ଗ'
COLON = re.compile(r'\s*[:;：!|ः](?=\s|$|[଀-୿0-9A-Za-z(])')
ROLE_ORDER = ['name', 'rel', 'house', 'age']
ORI = '଀-୿'
ODIA_DIGITS = str.maketrans('୦୧୨୩୪୫୬୭୮୯', '0123456789')


def norm(s):
    s = unicodedata.normalize('NFC', s or '').replace('‌', '').replace('‍', '').replace(' ', ' ')
    s = unicodedata.normalize('NFC', s.replace('୍ୱ', '୍ବ').replace('ଵ', 'ବ'))
    return re.sub(r'\s+', ' ', s).strip(' ;_')


def vis(s):
    return norm(s)


def _prefix(t):
    m = COLON.search(t)
    if m and m.start() <= 26: return t[:m.start()].strip()
    return ' '.join(t.split()[:2])


def line_role(text):
    t = norm(text)
    if not t: return None
    p = _prefix(t)
    if (fuzz.partial_ratio(AGE_LABEL, t[:6]) >= 75 and re.search(r'[0-9୦-୯]', t)) or (fuzz.partial_ratio(GEN_LABEL, t) >= 85
                                                                                     and re.search(r'[0-9୦-୯]', t)):
        return 'age'
    if fuzz.ratio(p, HOUSE_LABEL) >= 70 or t.startswith('ଘର'): return 'house'
    m = process.extractOne(p, LABELS, scorer=fuzz.ratio)
    if m and m[1] >= 72: return 'rel'
    if re.match(r'^(ପିତା|ସ୍ବାମୀ|ମାତା|ଅନ୍ୟ|ପତ୍ନୀ)', t): return 'rel'
    if fuzz.ratio(p, NAME_LABEL) >= 70 or (t.startswith('ନାମ') and len(p) <= 5): return 'name'
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
    if m and m.start() <= 26: return t[m.end():]
    return ' '.join(t.split()[nwords:])


def fix_epic(e):
    """3 letters + 7 digits (GDY1412279) or the old form OR/17/123/272006 (2 letters, then digit groups)."""
    e = re.sub(r'\s+', '', (e or '').upper()).translate(ODIA_DIGITS)
    if '/' in e and not re.match(r'^[A-Z0-9]{2}/', e):   # a new-form EPIC with a stray '/': Tesseract reads 7 as '/' or '/7'
        if len(e) == 10: e = e.replace('/', '7')
        elif len(e.replace('/', '')) == 10: e = e.replace('/', '')
    if '/' in e:
        m = re.match(r'^([A-Z0-9]{2})/(.*)$', e)
        if m: return m.group(1).translate(str.maketrans('0158', 'OISB')) + '/' + m.group(2).translate(str.maketrans('OIlSBZ', '011582'))
        return e
    m = re.match(r'^([A-Z0-9]{3})([A-Z0-9]{6,8})$', e)
    if m:
        return m.group(1).translate(str.maketrans('0158', 'OISB')) + m.group(2).translate(str.maketrans('OISBZ', '01582'))
    return e


def snap(v, choices, cutoff=60):
    m = process.extractOne(v, choices, scorer=fuzz.ratio)
    return m[0] if m and m[1] >= cutoff else v


def from_lines_ori(f, pp=True):
    """f: canonical dict (serial, part, epic, code, line0..line3). Same output keys as parse.from_lines."""
    out = dict(code='', serial_no='', part_no='', epic='', name='', relation_label='', relation_name='', house_no='', age='', gender='')
    s = norm(f.get('serial', '')).translate(ODIA_DIGITS)
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
    if m and m.start() <= 30:
        out['relation_label'], out['relation_name'] = norm(l1[:m.start()]), norm(l1[m.end():])
    else:
        w = l1.split()
        k = 3 if w and w[0].startswith('ଅନ୍ୟ') else 2
        out['relation_label'], out['relation_name'] = ' '.join(w[:k]), ' '.join(w[k:])
    out['house_no'] = norm(_value(norm(f.get('line2', '')), 2))
    l3 = norm(f.get('line3', ''))
    head = re.split(r'ଲି|ଲ଼ି', l3)[0]                       # age = last number before the gender label
    head = head[COLON.search(head).end():] if COLON.search(head) and COLON.search(head).start() <= 8 else head[3:]
    ds = re.findall(r'[0-9୦-୯OoIlSB]{1,3}', head)
    if ds: out['age'] = ds[0]
    g = re.search(r'ଲି\S*\s*[:;!|]?\s*(.*)$', l3)
    if g: out['gender'] = norm(g.group(1))
    elif ':' in l3: out['gender'] = norm(l3.split(':')[-1])
    if pp:
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(ODIA_DIGITS).translate(str.maketrans('OoIlSB', '001158')))
        out['epic'] = fix_epic(out['epic'])
        if out['gender']:
            gv = re.sub(rf'^[^{ORI}]+', '', out['gender'])
            out['gender'] = snap(gv, GENDERS, 40) if gv else out['gender']
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], LABELS, 70)
        for k in ('name', 'relation_name'):
            v = re.sub(rf'^[^{ORI}A-Za-z(]+', '', out[k])
            out[k] = v.strip(' :;,_|')
        out['house_no'] = re.sub(r'^[:;!|]+\s*', '', out['house_no']).strip(' ;_|')
    return out


def rel_canon(v):
    v = norm(v)
    for k, keys in (('father', ('ପିତା',)), ('husband', ('ସ୍ବାମୀ',)), ('mother', ('ମାତା',)), ('wife', ('ପତ୍ନୀ',)), ('other', ('ଅନ୍ୟ',))):
        if v.startswith(keys): return k
    return v


GENDER_SET = set(GENDERS)
LABEL_SET = set(LABELS)
NAME_RE = re.compile(rf"^[{ORI}A-Za-z][{ORI}A-Za-z .'()\-]*$")
EPIC_OLD = re.compile(r'^[A-Z]{2}/[0-9]{1,3}/[0-9]{1,4}/[0-9]{4,7}$')


def valid_ori(f, v, epic_re=None):
    if f == 'epic': return bool((epic_re.match(v) if epic_re else v) or EPIC_OLD.match(v or ''))
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in GENDER_SET
    if f == 'relation_label': return v in LABEL_SET
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'): return bool(NAME_RE.match(v or ''))
    return True
