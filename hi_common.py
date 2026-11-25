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
PAYLOAD_V = 1
PAYLOAD_VERSIONS = {1}


def payload_version(payload):
    v = payload.get("v", 1)
    if v not in PAYLOAD_VERSIONS:
        raise UnknownPayloadVersion(f"payload v {v!r} (this checkout knows {sorted(PAYLOAD_VERSIONS)})")
    return v


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


def compatible(ms, has_hints):
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
    mp, one = (settings or {}).get("ocr_model_set_map"), (settings or {}).get("ocr_model_set")
    for cand, src in ((_set_name(mp.get(name)) if isinstance(mp, dict) else None, "hub ocr_model_set_map"),
                      (_set_name(one), "hub ocr_model_set")):
        if cand and cand != name:
            if cand not in sets:
                raise UnknownModelSet(cand)
            if compatible(sets[cand], has_h):
                return cand, src
    if name not in sets:
        raise UnknownModelSet(str(name))
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
            for fn, w in (("crnn.onnx", want), ("charset.json", (ms.get("charset_sha256_16") or {}).get(m))):
                if w:
                    got = hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()[:16]
                    if got != w:
                        raise ModelSetIncomplete(f"{name} (models/{m}/{fn} sha256 {got} != {w})")
        _VERIFIED.add(name)
    return dirs


def _version(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return ((r.stdout or r.stderr).strip().splitlines() or [""])[0][:80]
    except (OSError, subprocess.SubprocessError):
        return ""


def tool_versions():
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
    for deb in ("libtesseract5", "tesseract-ocr-hin"):     # eci_ocr calls libtesseract through ctypes; hin = the traineddata
        v[deb] = _version(["dpkg-query", "-W", "-f", "${Version}", deb])
    return {k: x for k, x in v.items() if x}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rules_of(ms, year, has_hints):
    """The decision switches a run uses: the set's env plus the arguments that change decisions (eci_ocr ignores the
    cascade without hints, so stage1 is 0 then)."""
    r = dict(sorted(ms.get("env", {}).items()))
    r.update(stage1=int(ms.get("stage1", 0)) if has_hints else 0, roll_year=int(year))
    return r


CODE_RE = re.compile(r"(?:#|[SERQM])[0-9]?")    # = eci_ocr.CODE_RE (ECI_CODE_VALID)
LISTS = ("परिवर्धन की सूची", "विलोपन की सूची", "संशोधन की सूची")   # = eci_ocr.LISTS (supplement list sections)


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
