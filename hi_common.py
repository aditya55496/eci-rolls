"""Shared parts of the Hindi-state SIR OCR tasks (ocr-any, repair-any, parquet-any; 2026-11-25).

Model sets: model_sets.json (name -> CRNN dirs, env switches, hints mode, stage1). The queue item names its set; the hub
settings of the OCR project can swap sets for all pending items at once (see choose_set). Every output record carries its
provenance (PROVENANCE_KEYS), so outputs of different sets can be told apart and compared later.
"""
import hashlib
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OCR_HI_PROJECT = "eci-rolls-hi-ocr"          # ocr/<state>/<ac>/<part>.jsonl.gz (+ .summary.json)
REPAIR_HI_PROJECT = "eci-rolls-hi-repair"    # re-runs of the repairs on stored outputs (repair_any_task.py)
PARQUET_HI_PROJECT = "eci-rolls-hi-parquet"  # parquet/<state>/<roll id>/ac_<ac>/data.parquet
SSR_TEXT_PROJECT = "eci-rolls-ssr-text"      # 2027 SSR hints of any state: hints/<roll id>/<ac>.jsonl.gz (task ssrtext)
# engocr (2026-11-25): SIR rolls in English and Tamil (eci-rolls-all-cdn), the same task ocr-any in their own projects
OCR_ALL_PROJECT = "eci-rolls-all-ocr"        # ocr/<state>/<lang>/<ac>/<part>.jsonl.gz (+ .summary.json)
PARQUET_ALL_PROJECT = "eci-rolls-all-parquet"  # parquet/<state>/<roll id>/<lang>/ac_<ac>/data.parquet
LANGS = ("HIN", "ENG", "TAM", "MAL", "TEL", "PAN", "BEN", "ORI", "GUJ")   # eci_ocr --lang; HIN = the Hindi-state items (no lang segment in key and path)
PROVENANCE_KEYS = ("ocr_model_set", "ocr_models", "ocr_code", "ocr_rules", "hints_used")


def load_model_sets(path=None):
    d = json.load(open(path or os.path.join(HERE, "model_sets.json"), encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


MODEL_SETS = load_model_sets()

def code_commit():
    """The family checkout's commit (12 hex chars, '-dirty' with uncommitted changes to tracked files), or '' when git cannot
    tell (no .git, safe.directory, git missing). GITHUB_SHA is no fallback: on the runners it names the runner repository's
    commit, not this family's."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=60)
        code = r.stdout.strip() if r.returncode == 0 else ""
        if not re.fullmatch(r"[0-9a-f]{12}", code):
            return ""
        if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE, capture_output=True, text=True,
                          timeout=60).stdout.strip():
            code += "-dirty"      # local tests with uncommitted changes; the runners check out a clean tree
        return code
    except (OSError, subprocess.SubprocessError):
        return ""


# Read once at import: the code that is loaded in this process. ocr_any_task refuses to start when it is '' and compares it with
# code_commit() after every item (a runner's refetch_family or a server worker's ensure_family can move the checkout under a
# running worker; the provenance would then name the wrong commit).
CODE = code_commit()


class NeedsNewerCode(Exception):
    """The item needs code this checkout does not have: the worker gives the item back untried ('pending', the attempt is not
    counted) and exits EXIT_NEEDS_NEWER_CODE."""


class CodeChanged(NeedsNewerCode):
    """The family checkout moved to another commit while this worker ran: the item is given back and the worker exits."""


class UnknownModelSet(NeedsNewerCode):
    """A model set this checkout does not define (newer code has it): the worker hands the work back and exits."""


class UnknownPayloadVersion(NeedsNewerCode):
    """The item's payload 'v' is newer than this checkout knows (ocrv5 review 3): a later change that adds a payload key or
    changes its meaning raises PAYLOAD_V, so workers on older code (up to 5.5 h of a runner job) give such items back instead of
    ignoring the new key."""


class MissingRoll(RuntimeError):
    """A PDF that the item names could not be read from the bucket (common.download gives None after 4 tries on any error, not
    only on a 404). The item fails and is retried; it is never done with one roll less (ocrv5 review 3: nothing re-plans a
    done item whose keys did not change)."""


# Payload format of ocr-any items (key 'v'; absent = 1). Raise it, and add the new value to PAYLOAD_VERSIONS, whenever a change
# adds a payload key or changes the meaning of one without a new model set.
# v 2 (formocr, 2026-11-25): + 'roll_group' (form-route items: GE / SSR roll families; the output path gets that segment,
# ocr_base). SIR items keep writing v 1 (no new key): workers of older code take them as before and give a v 2 item back.
# v 3 (engocr, 2026-11-25): 'lang' ENG / TAM (eci_ocr --lang, a set of that language), output ocr/<state>/<lang>/<ac>/<part> and
# key '<state>/<lang>/<ac>/<part>'; parquet-any items + 'ocr_project' and 'lang'. A worker of older code gives a v 3 item back
# (UnknownPayloadVersion) instead of failing it on the lang check (v 1 / v 2 code raises 'lang not supported').
PAYLOAD_V = 3
PAYLOAD_VERSIONS = {1, 2, 3}
ROLL_GROUP_RE = re.compile(r"[SU][0-9]{2}-20[0-9]{2}-[A-Z0-9-]+")


def ocr_base(state, ac, roll_group=None, lang=None):
    """Bucket path (in eci-rolls-hi-ocr) of a part's output without the '/<part>.jsonl.gz': ocr/<state>/<ac> for the SIR items,
    ocr/<state>/<roll group>/<ac> for the form-route items (formocr: GE 2024 and SSR 2024/25 rolls of the same state and AC
    have the same part numbers as the SIR rolls; the group segment keeps the files apart). engocr: an item in another language
    than HIN -> ocr/<state>/<lang>/<ac> (in eci-rolls-all-ocr; West Bengal has Hindi, English and Bengali ACs)."""
    if lang not in (None, "HIN"):
        if lang not in LANGS or roll_group is not None:
            raise RuntimeError(f"bad lang {lang!r} (roll_group {roll_group!r})")
        return f"ocr/{state}/{lang}/{ac}"
    if roll_group is None:
        return f"ocr/{state}/{ac}"
    if not ROLL_GROUP_RE.fullmatch(str(roll_group)) or not str(roll_group).startswith(str(state) + "-"):
        raise RuntimeError(f"bad roll_group {roll_group!r} for state {state}")
    return f"ocr/{state}/{roll_group}/{ac}"


def payload_version(payload):
    v = payload.get("v", 1)
    if v not in PAYLOAD_VERSIONS:
        raise UnknownPayloadVersion(f"payload v {v!r} (this checkout knows {sorted(PAYLOAD_VERSIONS)})")
    if v < 2 and payload.get("roll_group") is not None:
        raise RuntimeError("payload v 1 with a roll_group (v 2 key): planner error")
    if v < 3 and (payload.get("lang") or "HIN") != "HIN":
        raise RuntimeError(f"payload v {v} with lang {payload.get('lang')} (v 3 meaning): planner error")
    return v


def set_lang(ms):
    """Language of a model set (key 'lang', absent = HIN: every set before engocr)."""
    return ms.get("lang", "HIN")


class ModelSetIncomplete(UnknownModelSet):
    """A set whose model dirs are missing in this checkout or have another sha256 than model_sets.json names (a set entry pushed
    before its model files, or a hub swap to such a set): handled like UnknownModelSet (item given back as 'pending', the
    attempt is not counted), so a bad push cannot use up the attempts of the pending items (ocrv5 review 2026-11-25)."""


# Exit code of a worker that gave its item back because this checkout is too old or incomplete for it (unknown set, set with
# missing model files, checkout moved during the item). 75 = EX_TEMPFAIL. A runner that knows it fetches the family's main
# again at once and runs the next assignment with the new code (s1d-labs/runner branch family-refetch); an older runner treats
# it like any other non-zero exit (waits 30-300 s and asks again).
EXIT_NEEDS_NEWER_CODE = 75


def is_draft(roll_id):
    """S12-2026-DR, S04-2025-SIR-DR -> True; S12-2026-FIR, S28-2026-SIR-FIR-1 -> False."""
    return bool(re.search(r"-DR(-\d+)?$", roll_id))


def roll_year(roll_id):
    return int(roll_id.split("-")[1])


def roll_order(keys):
    """Roll ids of an item, draft first (its readings are the final roll's cache)."""
    return sorted(keys, key=lambda r: (0 if is_draft(r) else 1, r))


def compatible(ms, has_hints, lang="HIN"):
    """A set fits an item when its hints mode fits the item's hints and its language is the item's (engocr: an English set is
    never put on a Hindi item and the other way round, also not by a hub swap)."""
    if set_lang(ms) != (lang or "HIN"):
        return False
    mode = ms.get("hints", "optional")
    return has_hints if mode == "required" else (not has_hints) if mode == "none" else True


def _set_name(v):
    """A hub override value names a set only when it is a non-empty string (null, '' or a number = no override)."""
    return v if isinstance(v, str) and v.strip() else None


def choose_set(payload, settings, sets=None):
    """-> (set name, source). Order: hub setting 'ocr_model_set_map' {payload set: new set}, then 'ocr_model_set' (one set
    for all items), then the payload's 'model_set'. A hub choice whose hints mode does not fit the item (a set that needs
    hints on an item without hints, or a no-hint set on an item with hints) is ignored for that item. Raises UnknownModelSet
    when the chosen set is not in this checkout's model_sets.json."""
    sets = MODEL_SETS if sets is None else sets
    name = payload.get("model_set")
    has_h = bool(payload.get("hints"))
    lang = payload.get("lang") or "HIN"
    mp, one = (settings or {}).get("ocr_model_set_map"), (settings or {}).get("ocr_model_set")
    for cand, src in ((_set_name(mp.get(name)) if isinstance(mp, dict) else None, "hub ocr_model_set_map"),
                      (_set_name(one), "hub ocr_model_set")):
        if cand and cand != name:
            if cand not in sets:
                raise UnknownModelSet(cand)
            if compatible(sets[cand], has_h, lang):
                return cand, src
    if name not in sets:
        raise UnknownModelSet(str(name))
    if set_lang(sets[name]) != lang:
        raise RuntimeError(f"model set {name} reads {set_lang(sets[name])}, the item is {lang}: planner error")
    return name, "payload"


def hub_override_unknown(settings, sets=None):
    """A hub override naming a set this checkout lacks (checked before claiming, so no item is taken). Values that are not a
    non-empty string are no override (as in choose_set; ocrv5 review 3: {"j2Ps": null} stopped every worker). A typo
    ({"j2Ps": "j2Psff"}) still counts as unknown: no worker can tell it from a set of newer code, so every worker of the project
    exits 75 before it claims (metric model_set_unknown; README hot-swap steps: check that metric just after a swap)."""
    sets = MODEL_SETS if sets is None else sets
    mp, one = (settings or {}).get("ocr_model_set_map"), (settings or {}).get("ocr_model_set")
    names = [_set_name(v) for v in mp.values()] if isinstance(mp, dict) else []
    names.append(_set_name(one))
    return [n for n in names if n and n not in sets]


_VERIFIED = set()


def model_dirs(name, sets=None):
    """Model dirs of a set, checked once per process against the sha256 prefixes in model_sets.json: crnn.onnx (sha256_16) and
    charset.json (charset_sha256_16; a wrong charset gives wrong readings without an error, ocrv5 review 3)."""
    ms = (MODEL_SETS if sets is None else sets)[name]
    dirs = [os.path.join(HERE, "models", m) for m in ms["models"]]
    if name not in _VERIFIED:
        for m, d in zip(ms["models"], dirs):
            want = (ms.get("sha256_16") or {}).get(m)
            if not os.path.exists(os.path.join(d, "crnn.onnx")) or not os.path.exists(os.path.join(d, "charset.json")):
                raise ModelSetIncomplete(f"{name} (models/{m} incomplete in this checkout)")
            arch = (ms.get("arch_sha256_16") or {}).get(m)
            if arch is None and os.path.exists(os.path.join(d, "arch.json")) and ms.get("lang", "HIN") != "HIN":
                # engocr: eci_ocr reads 'xs' (horizontal stretch, job3) from arch.json: a set of a later language must name its sha
                raise ModelSetIncomplete(f"{name} (models/{m}/arch.json has no sha256 in model_sets.json)")
            for fn, w in (("crnn.onnx", want), ("charset.json", (ms.get("charset_sha256_16") or {}).get(m)), ("arch.json", arch)):
                if w:
                    if not os.path.exists(os.path.join(d, fn)):
                        raise ModelSetIncomplete(f"{name} (models/{m}/{fn} missing in this checkout)")
                    got = hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()[:16]
                    if got != w:
                        raise ModelSetIncomplete(f"{name} (models/{m}/{fn} sha256 {got} != {w})")
        for a in ms.get("aux") or []:     # files the set's env names as <models>/NAME (VALSET CRNN dirs, lexicons; formocr)
            path = os.path.join(HERE, "models", a)
            files = [("crnn.onnx", ms["sha256_16"].get(a)), ("charset.json", (ms.get("charset_sha256_16") or {}).get(a))] \
                if os.path.isdir(path) else [("", ms["sha256_16"].get(a))]
            for fn, w in files:
                f = os.path.join(path, fn) if fn else path
                if not os.path.isfile(f):
                    raise ModelSetIncomplete(f"{name} (models/{a} incomplete in this checkout)")
                if not w:
                    raise ModelSetIncomplete(f"{name} (models/{a}/{fn} has no sha256 in model_sets.json)")
                got = sha256_file(f)[:16]
                if got != w:
                    raise ModelSetIncomplete(f"{name} (models/{a}/{fn} sha256 {got} != {w})")
        _VERIFIED.add(name)
    return dirs


MODELS_TOKEN = "<models>"


def set_env(ms):
    """The set's eci_ocr switches for a run: '<models>' in a value (paths of VALSET CRNNs and lexicons, formocr j5r4) becomes
    this checkout's models/ dir. The records' ocr_rules keep the '<models>' form (the same on every runner)."""
    d = os.path.join(HERE, "models")
    return {k: str(v).replace(MODELS_TOKEN, d) for k, v in (ms.get("env") or {}).items()}


def _version(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return ((r.stdout or r.stderr).strip().splitlines() or [""])[0][:80]
    except (OSError, subprocess.SubprocessError):
        return ""


def tool_versions(extra_tess=()):
    """Versions of the parts outside the family commit that change the OCR input or the readings (ocrv5 review 2026-11-25:
    requirements.txt does not pin pymupdf; Tesseract and its hin traineddata come from apt). Read once per worker."""
    import importlib.metadata as md
    v = {}
    for pkg in ("pymupdf", "onnxruntime", "opencv-python-headless", "numpy"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    v["tesseract"] = _version(["tesseract", "--version"]).replace("tesseract ", "")
    for deb in ("libtesseract5", "tesseract-ocr-hin") + tuple(f"tesseract-ocr-{t}" for t in extra_tess):     # eci_ocr calls libtesseract through ctypes; hin = the traineddata
        v[deb] = _version(["dpkg-query", "-W", "-f", "${Version}", deb])
    return {k: x for k, x in v.items() if x}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# eci_ocr arguments a set may add (engocr): job4's set J reads with a second Tesseract (x2, Otsu) as one more voter
SET_ARGS = {"--tess2"}


def set_args(ms):
    """Extra eci_ocr arguments of a set (key 'args'); only those in SET_ARGS (a set entry cannot point eci_ocr at other files)."""
    a = list(ms.get("args") or [])
    bad = [x for x in a if x not in SET_ARGS]
    if bad:
        raise RuntimeError(f"model set arguments {bad} not allowed (allowed: {sorted(SET_ARGS)})")
    return a


def set_stage1(ms, has_hints):
    """--stage1 of a run. Hindi sets: the cascade needs hints (eci_ocr ignores it without them, so 0 then). engocr: a set with
    'stage1_nohint' true gives --stage1 also without hints (job4's English no-hint cascade ECI_ENG_NOHINT: stage 1 = the first 2
    CRNNs; lang-tam tamN2 the same)."""
    if has_hints or ms.get("stage1_nohint"):
        return int(ms.get("stage1", 0))
    return 0


def rules_of(ms, year, has_hints):
    """The decision switches a run uses: the set's env plus the arguments that change decisions (eci_ocr ignores the
    cascade without hints, so stage1 is 0 then; engocr: sets with stage1_nohint keep it). engocr sets (lang other than HIN) also
    name lang, their extra eci_ocr arguments and whether a final roll reads the draft's records (draft_records)."""
    r = dict(sorted(ms.get("env", {}).items()))
    r.update(stage1=set_stage1(ms, has_hints), roll_year=int(year))
    if set_lang(ms) != "HIN":
        r.update(lang=set_lang(ms), args=set_args(ms), draft_records=bool(ms.get("draft_records")))
    return r


CODE_RE = re.compile(r"(?:#|[SERQM])[0-9]?")    # = eci_ocr.CODE_RE (ECI_CODE_VALID)
LISTS = ("परिवर्धन की सूची", "विलोपन की सूची", "संशोधन की सूची")   # = eci_ocr.LISTS (supplement list sections)


def repair_engines(ms):
    """Reading names of the CRNNs whose agreeing code letter the final_deleted repair takes (the first two of the set; a set
    can name others in 'repair_engines', e.g. when its first model is a stage-1 screening model that does not read stage-2
    boxes: job1 speed set, ECI_S1_SCREEN)."""
    return list(ms.get("repair_engines") or ["crnn_" + m for m in ms["models"]])


def repair_part(recs, engines, rules=None):
    """Repairs of one part file (draft + final records), as repair_task.py did for UP, in place:
    final_deleted (final-roll voters deleted after the draft that read as page noise get the draft's fields) and serial_gaps
    (boxes taken for page noise that fill a serial gap are voters). The EPIC repair from the 2027 roll runs inside eci_ocr
    when hints exist (epicfix.fix + fix2), so it is not repeated here.
    engines: the reading names of the set's CRNNs, or a function record -> names (repair-any on a file of several sets).
    rules: the run's eci_ocr switches (the set's env), or None = each record's ocr_rules. With ECI_CODE_VALID=1 a repaired
    record whose code is not valid gets the 'code' flag again (final_deleted takes the draft's flags, eci_ocr flagged the
    final roll's code before). Returns {fix: records changed}."""
    import sys
    sys.path.insert(0, os.path.join(HERE, "eciocr", "prod"))
    import epicfix
    out = {"final_deleted": epicfix.final_deleted(recs, engines), "serial_gaps": epicfix.serial_gaps(recs)}
    n = 0
    for r in recs:
        rl = rules if rules is not None else (r.get("ocr_rules") or {})
        if (str(rl.get("ECI_CODE_VALID", "0")) == "1" and not r.get("not_voter_box") and r.get("code")
                and not CODE_RE.fullmatch(r["code"]) and "code" not in (r.get("flags") or [])):
            r.setdefault("flags", []).append("code")
            r["code_invalid"] = True
            n += 1
    if n:
        out["code_invalid"] = n
    return out


def serial_summary(recs):
    """eci_ocr.serial_check's numbers for the voter records of one PDF, without changing the records (ocr-any computes them
    again after repair_part: before, the hub record showed eci_ocr's check before final_deleted / serial_gaps, e.g.
    records_equal_last false for a part that the repairs made complete)."""
    s = [int(r["serial_no"]) if str(r.get("serial_no") or "").isdigit() else None for r in recs]
    n, bad = len(s), 0
    for i, v in enumerate(s):
        prev_ok = i > 0 and s[i - 1] is not None and v is not None and v == s[i - 1] + 1
        next_ok = i + 1 < n and s[i + 1] is not None and v is not None and v == s[i + 1] - 1
        if i == 0:
            prev_ok = v == 1
        if i == n - 1 and n > 1:
            next_ok = prev_ok
        bad += not (prev_ok or next_ok)
    good = sorted(v for v in s if v is not None)
    seen = {}
    for v in good:
        seen[v] = seen.get(v, 0) + 1
    last = max(good) if good else None
    missing = [k for k in range(1, (last or 0) + 1) if k not in seen]
    return dict(records=n, first=s[0] if s else None, last=last, records_equal_last=(n == last), missing=missing[:200],
                n_missing=len(missing), duplicates=[k for k, c in seen.items() if c > 1][:200], sequence_breaks=bad,
                unreadable=sum(v is None for v in s))


def additions_without_list(recs):
    """{final-roll source file: voter records with a serial above the draft roll's last serial whose section is not a
    supplement list}. In an SIR final roll the addition list continues the draft's serials; a record above the draft's last
    serial outside a list section means the list heading was not read (ocrv5 review 3: 'घटक I - परिवर्धन सूची 1' was not
    parsed in 4 states). A check for monitors, nothing is changed. The draft's last serial is capped at its records + 10 (one
    far-out misread such as 11001 must not hide the check), as epicfix.serial_gaps."""
    voters = [r for r in recs if not r.get("not_voter_box")]
    dr = [int(r["serial_no"]) for r in voters if "DraftRoll" in (r.get("source") or "") and str(r.get("serial_no") or "").isdigit()]
    dr = [x for x in dr if x <= len(dr) + 10]
    if not dr:
        return {}
    last, out = max(dr), {}
    for r in voters:
        src = r.get("source") or ""
        if "FinalRoll" in src:
            out.setdefault(src, 0)
            if str(r.get("serial_no") or "").isdigit() and int(r["serial_no"]) > last and r.get("section") not in LISTS:
                out[src] += 1
    return out
