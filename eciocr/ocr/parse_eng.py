"""English roll boxes: engine text of the cut lines -> the 10 record fields (2026-11-25, job engpipe).

An English box has 4 labelled lines and 0-3 unlabelled wrap lines (segment.segment_eng gives line0..lineN):
    Name : <name>
    <Fathers|Father's|Husbands|Husband's|Mothers|Mother's> Name: <relative>     or   Others: <relative>
    House Number : <house>
    Age : <age> Gender : <Male|Female|Third Gender>
Roles: line_role() classifies one engine's text of one line (name / rel / house / age / None = wrap line). box_roles() takes
the roles of all engines per line, votes, and makes them consistent (order name < rel < house < age, each once); a wrap line
belongs to the labelled line above it; lines after the age line (stamp ink) are dropped. Every engine is then parsed with the
SAME roles (canonical()), so the engines' values are comparable field by field.
canonical() also maps the box to the Hindi key names (line0 = name text, line1 = relative line, line2 = house, line3 = age
line): ensemble2.decide (LINE_OF, confidences) works unchanged.
"""
import collections, re, unicodedata
from rapidfuzz import fuzz, process

# printed relative labels (gold 2,064 boxes, 14 states; 2027 rolls add the 'Father Name' / 'Other' forms of Tamil Nadu)
LABELS = ['Fathers Name', "Father's Name", 'Husbands Name', "Husband's Name", 'Mothers Name', "Mother's Name", 'Others',
          'Father Name', 'Husband Name', 'Mother Name', 'Other', 'Wifes Name', "Wife's Name"]
GENDERS = ['Male', 'Female', 'Third Gender']
APOS = str.maketrans({'’': "'", '‘': "'", '`': "'", '´': "'", 'ʼ': "'", '′': "'"})
COLON = re.compile(r'\s*[:;ː：!|](?=\s|[A-Za-z0-9(]|$)')
ROLE_ORDER = ['name', 'rel', 'house', 'age']


def norm(s):
    s = unicodedata.normalize('NFC', s or '').translate(APOS).replace('‌', '').replace('‍', '')
    return re.sub(r'\s+', ' ', s).strip(' ;_')   # '|' kept: a final I read as '|' is repaired in from_lines_eng


def _prefix(t):
    """Label part of a line: text before the first colon-like mark, else the first two words."""
    m = COLON.search(t)
    if m and m.start() <= 22: return t[:m.start()].strip()
    return ' '.join(t.split()[:2])


def line_role(text):
    t = norm(text)
    if not t: return None
    p = _prefix(t); pl = p.lower(); tl = t.lower()
    if re.match(r'^a[gq]e\b', tl) or re.search(r'\bgend[eo]r\b', tl) or (fuzz.ratio(pl, 'age') >= 67 and re.search(r'\d', t)):
        return 'age'
    if pl.startswith('house') or fuzz.partial_ratio('house number', pl) >= 80 and len(pl) >= 8: return 'house'
    if fuzz.ratio(pl, 'name') >= 75: return 'name'
    m = process.extractOne(p, LABELS, scorer=fuzz.ratio)
    if m and m[1] >= 72: return 'rel'
    if re.match(r"^(fath|husb|moth|wif|othe)", pl) and ('name' in tl[:20] or pl.startswith('othe')): return 'rel'
    return None


def box_roles(per_engine_lines, n_lines):
    """per_engine_lines: list (per engine) of {line_i: text}. -> list of role per line index (None = wrap line, 'drop')."""
    votes = []
    for i in range(n_lines):
        c = collections.Counter(line_role(L.get(f'line{i}', '')) for L in per_engine_lines)
        lab = [(r, n) for r, n in c.most_common() if r]
        none = c.get(None, 0)
        votes.append(lab[0][0] if lab and lab[0][1] * 2 >= len(per_engine_lines) - 0 and lab[0][1] >= none * 0.5 else None)
    roles, seen, last = [], set(), -1
    for r in votes:
        if r and r not in seen and ROLE_ORDER.index(r) > last:
            roles.append(r); seen.add(r); last = ROLE_ORDER.index(r)
        else:
            roles.append(None)
    # missing labels (a label read badly by most engines): fill by position. The box always has the 4 labelled lines in order;
    # wrap lines are the exceptions, so a missing role goes to the first free line after the role before it.
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
        elif cur is None or cur == 'age': out.append('drop')     # before the name line / after the age line: not voter text
        else: out.append(None)
    return out


def canonical(raw, roles):
    """raw: one engine's {serial, part, epic, code, line0..lineN}; roles from box_roles. -> dict with line0..line3 = the
    name / relative / house / age texts (wrap lines joined with a space)."""
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
    """CRNN confidences per raw crop -> per canonical line (minimum over the joined lines)."""
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
    """Value after the label: split at the first colon-like mark near the start, else drop the label words."""
    m = COLON.search(t)
    if m and m.start() <= 24: return t[m.end():]
    return ' '.join(t.split()[nwords:])


def fix_epic(e):
    e = re.sub(r'\s+', '', (e or '').upper())
    if 0 < e.count('/') < 3 and re.fullmatch(r'[A-Z0-9]{3}[0-9/]{7,8}', e) and len(e.replace('/', '7')) == 10:
        e = e.replace('/', '7')   # Tesseract (EPIC whitelist has '/') reads 7 as '/' in new-style EPICs: YKS15/0423
    if '/' in e:   # old-style 'WB/17/114/129145': letters before the first '/', digits after it (job4: was 'W8/...')
        m = re.match(r'^([A-Z0-9]{2,3})(/.*)$', e)
        if m: return m.group(1).translate(str.maketrans('0158', 'OISB')) + m.group(2).translate(str.maketrans('OIlSB', '01158'))
        return e.translate(str.maketrans('OIlSB', '01158'))
    m = re.match(r'^([A-Z0-9]{3})([A-Z0-9]{6,8})$', e)
    if m:
        a = m.group(1).translate(str.maketrans('0158', 'OISB')); b = m.group(2).translate(str.maketrans('OISBZ', '01582'))
        return a + b
    return e


def snap(v, choices, cutoff=60):
    m = process.extractOne(v, choices, scorer=fuzz.ratio)
    return m[0] if m and m[1] >= cutoff else v


def from_lines_eng(f, pp=True):
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
        k = 1 if w and w[0].lower().startswith('other') else 2
        out['relation_label'], out['relation_name'] = ' '.join(w[:k]), ' '.join(w[k:])
    out['house_no'] = norm(_value(norm(f.get('line2', '')), 2))
    l3 = norm(f.get('line3', ''))
    m = re.search(r'^\W*A\w{0,2}\W*([0-9OoIlSB]{1,3})\b', l3) or re.search(r'[:;!]\s*([0-9OoIl]{1,3})\b', l3)
    if m: out['age'] = m.group(1)
    g = re.search(r'Gen\w*\s*[:;!|ː]?\s*(.*)$', l3, re.I)
    if g: out['gender'] = norm(g.group(1))
    elif ':' in l3: out['gender'] = norm(l3.split(':')[-1])
    if pp:
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(str.maketrans('OoIlSB', '001158')))
        out['epic'] = fix_epic(out['epic'])
        if out['gender']: out['gender'] = snap(out['gender'], GENDERS)
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], LABELS, 70)
        for k in ('name', 'relation_name'):
            v = re.sub(r'^[^A-Za-z(]+', '', out[k])                 # colon / noise read before the value ('? SIMHACHALAM')
            v = re.sub(r'(?<=[A-Za-z])[!|](?=\s|$)', 'I', v)         # final I read as ! or | ('DODD!')
            v = re.sub(r'(?<=[A-Za-z])[|!](?=[A-Za-z])', 'I', v)
            v = re.sub(r'(?<=\s)[|!](?=\s|[A-Za-z])', 'I', v)
            out[k] = v.strip(' :;,_')
        out['house_no'] = re.sub(r'^[:;!|ː]+\s*', '', out['house_no']).strip(' ;_|')
    return out


def rel_canon(v):
    """'Fathers Name' / "Father's Name" / 'Father Name' -> 'father' (2027 hint and SIR print may use different forms)."""
    v = norm(v).lower()
    v = re.sub(r"'?s?\s*name$", '', v).strip()
    return {'others': 'other'}.get(v, v)


GENDER_SET = set(GENDERS)
LABEL_SET = set(LABELS)
NAME_RE = re.compile(r"^(\([A-Za-z]{1,3}\)\.? ?)?[A-Za-z][A-Za-z .'()\-]*$")   # job4: '(L) Amuthoi' (late) is a printed form


def valid_eng(f, v, epic_re=None):
    if f == 'epic': return bool(epic_re.match(v)) if epic_re else bool(v)
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in GENDER_SET
    if f == 'relation_label': return v in LABEL_SET
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'): return bool(NAME_RE.match(v or ''))
    return True
