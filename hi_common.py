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

try:
    CODE = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    if CODE and subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE, capture_output=True,
                               text=True).stdout.strip():
        CODE += "-dirty"      # local tests with uncommitted changes; the runners check out a clean tree
except OSError:
    CODE = ""


class UnknownModelSet(Exception):
    """A model set this checkout does not define (newer code has it): the worker hands the work back and exits."""


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


def choose_set(payload, settings, sets=None):
    """-> (set name, source). Order: hub setting 'ocr_model_set_map' {payload set: new set}, then 'ocr_model_set' (one set
    for all items), then the payload's 'model_set'. A hub choice whose hints mode does not fit the item (a set that needs
    hints on an item without hints, or a no-hint set on an item with hints) is ignored for that item. Raises UnknownModelSet
    when the chosen set is not in this checkout's model_sets.json."""
    sets = MODEL_SETS if sets is None else sets
    name = payload.get("model_set")
    has_h = bool(payload.get("hints"))
    mp, one = (settings or {}).get("ocr_model_set_map"), (settings or {}).get("ocr_model_set")
    for cand, src in (((mp or {}).get(name) if isinstance(mp, dict) else None, "hub ocr_model_set_map"),
                      (one if isinstance(one, str) and one else None, "hub ocr_model_set")):
        if cand and cand != name:
            if cand not in sets:
                raise UnknownModelSet(cand)
            if compatible(sets[cand], has_h):
                return cand, src
    if name not in sets:
        raise UnknownModelSet(str(name))
    return name, "payload"


def hub_override_unknown(settings, sets=None):
    """A hub override naming a set this checkout lacks (checked before claiming, so no item is taken)."""
    sets = MODEL_SETS if sets is None else sets
    mp, one = (settings or {}).get("ocr_model_set_map"), (settings or {}).get("ocr_model_set")
    names = list(mp.values()) if isinstance(mp, dict) else []
    names += [one] if isinstance(one, str) and one else []
    return [n for n in names if n not in sets]


_VERIFIED = set()


def model_dirs(name, sets=None):
    """Model dirs of a set, checked once per process against the sha256 prefixes in model_sets.json."""
    ms = (MODEL_SETS if sets is None else sets)[name]
    dirs = [os.path.join(HERE, "models", m) for m in ms["models"]]
    if name not in _VERIFIED:
        for m, d in zip(ms["models"], dirs):
            want = (ms.get("sha256_16") or {}).get(m)
            if not os.path.exists(os.path.join(d, "crnn.onnx")) or not os.path.exists(os.path.join(d, "charset.json")):
                raise RuntimeError(f"model set {name}: models/{m} incomplete in this checkout")
            if want:
                got = hashlib.sha256(open(os.path.join(d, "crnn.onnx"), "rb").read()).hexdigest()[:16]
                if got != want:
                    raise RuntimeError(f"model set {name}: models/{m}/crnn.onnx sha256 {got} != {want}")
        _VERIFIED.add(name)
    return dirs


def rules_of(ms, year, has_hints):
    """The decision switches a run uses: the set's env plus the arguments that change decisions (eci_ocr ignores the
    cascade without hints, so stage1 is 0 then)."""
    r = dict(sorted(ms.get("env", {}).items()))
    r.update(stage1=int(ms.get("stage1", 0)) if has_hints else 0, roll_year=int(year))
    return r


def repair_part(recs, engines):
    """Repairs of one part file (draft + final records), as repair_task.py did for UP, in place:
    final_deleted (final-roll voters deleted after the draft that read as page noise get the draft's fields) and serial_gaps
    (boxes taken for page noise that fill a serial gap are voters). The EPIC repair from the 2027 roll runs inside eci_ocr
    when hints exist (epicfix.fix + fix2), so it is not repeated here. Returns {fix: records changed}."""
    import sys
    sys.path.insert(0, os.path.join(HERE, "eciocr", "prod"))
    import epicfix
    return {"final_deleted": epicfix.final_deleted(recs, engines), "serial_gaps": epicfix.serial_gaps(recs)}
