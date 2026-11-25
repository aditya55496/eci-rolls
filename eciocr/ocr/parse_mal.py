"""Malayalam + Telugu roll boxes (S11 Kerala, U06 Lakshadweep, U07-29 Mahe: MAL; U07-30 Yanam: TEL; SIR 2026): engine text of
the cut lines -> the 10 record fields (2026-11-25, job lang-mal; copy of parse_tam with per-language tables).

The box is the English SIR template (segment.segment_eng gives line0..lineN) with the labels of the language:
  MAL  പേര് : <name> | അച്ഛന്റെ പേര്: (U06 / U07-29: പിതാവിന്റെ പേര്) / ഭർത്താവിന്റെ പേര് / അമ്മയുടെ പേര് / മറ്റുള്ളവ(ർ) |
       വീട്ടു നമ്പർ : <house> | പ്രായം : <age> ലിംഗം : <പുരുഷൻ | സ്ത്രീ | ...>
  TEL  పేరు : <name> | తండ్రి పేరు / భర్త పేరు / తల్లి పేరు / ఇతరులు | ఇంటి నంబరు : <house> | వయస్సు : <age> లింగం : <పురుషుడు | స్త్రీ>
The language is the module global LANG (set by eci_ocr per PDF, as ensemble2.LANG). Role logic = parse_tam / parse_eng.
Text normalisation (norm): NFC, Malayalam chillu written as consonant + virama + ZWJ -> the atomic chillu (both print the same;
the 2027 text layer and the gold use the atomic form), other ZWJ / ZWNJ dropped, spaces collapsed.
"""
import collections, os, re, unicodedata
from rapidfuzz import fuzz, process

LANG = 'MAL'
SPEC = {
    'MAL': dict(
        name='പേര്', house='വീട്ടു നമ്പർ', age='പ്രായം', gen='ലിംഗം', block='ഀ-ൿ',
        labels=['അച്ഛന്റെ പേര്', 'പിതാവിന്റെ പേര്', 'ഭർത്താവിന്റെ പേര്', 'അമ്മയുടെ പേര്', 'മാതാവിന്റെ പേര്', 'ഭാര്യയുടെ പേര്',
                'മറ്റുള്ളവ', 'മറ്റുള്ളവർ', 'രക്ഷിതാവിന്റെ പേര്'],
        genders=['പുരുഷൻ', 'സ്ത്രീ', 'മൂന്നാം ലിംഗം', 'ട്രാൻസ്ജെൻഡർ'],
        kinds=(('father', ('അച്ഛ', 'പിതാ')), ('husband', ('ഭർത്താ', 'ഭര്ത്താ')), ('mother', ('അമ്മ', 'മാതാ')), ('wife', ('ഭാര്യ',)),
               ('other', ('മറ്റു', 'രക്ഷി'))),
        rel_re=r'^(അച്ഛ|പിതാ|ഭർത്ത|ഭര്ത്ത|അമ്മ|മാതാ|ഭാര്യ|മറ്റു)', other_one='മറ്റുള്ള',
        male='പുരുഷൻ', female='സ്ത്രീ', husband=('ഭർത്താ', 'ഭര്ത്താ'), wife=('ഭാര്യ',)),
    'TEL': dict(
        name='పేరు', house='ఇంటి నంబరు', age='వయస్సు', gen='లింగం', block='ఀ-౿',
        labels=['తండ్రి పేరు', 'భర్త పేరు', 'తల్లి పేరు', 'భార్య పేరు', 'ఇతరులు', 'ఇతరుల పేరు', 'సంరక్షకుని పేరు'],
        genders=['పురుషుడు', 'స్త్రీ', 'ఇతరులు', 'థర్డ్ జెండర్'],
        kinds=(('father', ('తండ్రి',)), ('husband', ('భర్త',)), ('mother', ('తల్లి',)), ('wife', ('భార్య',)), ('other', ('ఇతర', 'సంరక్ష'))),
        rel_re=r'^(తండ్రి|భర్త|తల్లి|భార్య|ఇతర)', other_one='ఇతరులు',
        male='పురుషుడు', female='స్త్రీ', husband=('భర్త',), wife=('భార్య',)),
}
ROLE_ORDER = ['name', 'rel', 'house', 'age']
CHILLU = {'ണ': 'ൺ', 'ന': 'ൻ', 'ര': 'ർ', 'ല': 'ൽ', 'ള': 'ൾ', 'ക': 'ൿ'}
CHILLU_RE = re.compile('([ണനരലളക])്‍')


def S(): return SPEC[LANG]


def colon_re():
    return re.compile(r'\s*[:;：!|ःഃః](?=\s|$|[' + S()['block'] + r'0-9A-Za-z(/])')


def norm(s):
    s = unicodedata.normalize('NFC', s or '')
    s = CHILLU_RE.sub(lambda m: CHILLU[m.group(1)], s)
    s = s.replace('‌', '').replace('‍', '').replace(' ', ' ')
    return re.sub(r'\s+', ' ', s).strip(' ;_')


def vis(s):
    """visual form for comparisons (gold / 2027 hint): Malayalam old au sign ൌ and the length mark ൗ alone are kept apart
    (they print differently); only the chillu / ZWJ forms are folded (in norm)"""
    return norm(s)


def _prefix(t):
    m = colon_re().search(t)
    if m and m.start() <= 24: return t[:m.start()].strip()
    return ' '.join(t.split()[:2])


def line_role(text):
    t = norm(text)
    if not t: return None
    sp = S(); p = _prefix(t)
    if fuzz.ratio(p, sp['house']) >= 70 or t.startswith(sp['house'][:3]): return 'house'   # before age: 'വീട്ട് നമ്പർ : 1/24'
    if (max(fuzz.ratio(a, p) for a in sp['age']) >= 70 and re.search(r'[0-9]', t)) or \
            (fuzz.partial_ratio(sp['gen'], t) >= 85 and re.search(r'[0-9]', t)):
        return 'age'
    m = process.extractOne(p, sp['labels'], scorer=fuzz.ratio)
    if m and m[1] >= 72: return 'rel'
    if re.match(sp['rel_re'], t): return 'rel'
    if fuzz.ratio(p, sp['name']) >= 70 or (t.startswith(sp['name'][:2]) and len(p) <= 7): return 'name'
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
    m = colon_re().search(t)
    if m and m.start() <= 26: return t[m.end():]
    return ' '.join(t.split()[nwords:])


def fix_epic(e):
    e = re.sub(r'\s+', '', (e or '').upper())
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


def from_lines_mal(f, pp=True):
    """f: canonical dict (serial, part, epic, code, line0..line3). Same output keys as parse.from_lines."""
    sp = S(); col = colon_re(); blk = sp['block']
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
    m = col.search(l1)
    if m and m.start() <= 26:
        out['relation_label'], out['relation_name'] = norm(l1[:m.start()]), norm(l1[m.end():])
    else:
        w = l1.split()
        k = 1 if w and w[0].startswith(sp['other_one']) else 2
        out['relation_label'], out['relation_name'] = ' '.join(w[:k]), ' '.join(w[k:])
    out['house_no'] = norm(_value(norm(f.get('line2', '')), 2))
    l3 = norm(f.get('line3', ''))
    m = re.search(r'^\W*\S{0,8}\W*([0-9OoIlSB]{1,3})\b', l3) or re.search(r'[:;!]\s*([0-9OoIl]{1,3})\b', l3)
    if m: out['age'] = m.group(1)
    g = re.search(sp['gen'][:3] + r'\S*\s*[:;!|ഃః]?\s*(.*)$', l3)
    if g: out['gender'] = norm(g.group(1))
    elif ':' in l3: out['gender'] = norm(l3.split(':')[-1])
    if pp:
        for k in ('serial_no', 'part_no', 'age'):
            out[k] = re.sub(r'[^0-9]', '', out[k].translate(str.maketrans('OoIlSB', '001158')))
        out['epic'] = fix_epic(out['epic'])
        if out['gender']: out['gender'] = snap(out['gender'], sp['genders'])
        if out['relation_label']: out['relation_label'] = snap(out['relation_label'], sp['labels'], 70)
        for k in ('name', 'relation_name'):
            v = re.sub(rf'^[^{blk}A-Za-z(]+', '', out[k])           # colon / noise read before the value
            out[k] = v.strip(' :;,_|')
        out['house_no'] = re.sub(r'^[:;!|]+\s*', '', out['house_no']).strip(' ;_|')
    return out


def rel_canon(v):
    """relation label -> kind (father / husband / mother / other / wife): SIR 2026 and 2027 print other forms"""
    v = norm(v)
    for k, keys in S()['kinds']:
        if v.startswith(keys): return k
    return v


def implausible(label, gender):
    """Husband label + male gender, wife label + female gender"""
    sp = S(); label = norm(label)
    return (label.startswith(sp['husband']) and gender == sp['male']) or (label.startswith(sp['wife']) and gender == sp['female'])


def valid_mal(f, v, epic_re=None):
    sp = S()
    if f == 'epic': return bool(epic_re.match(v)) if epic_re else bool(v)
    if f == 'age': return v.isdigit() and 18 <= int(v) <= 120
    if f == 'gender': return v in sp['genders']
    if f == 'relation_label': return v in sp['labels']
    if f == 'serial_no': return v.isdigit()
    if f in ('name', 'relation_name'):
        return bool(re.match(rf"^[{sp['block']}A-Za-z][{sp['block']}A-Za-z .'()\-]*$", v or ''))
    return True
