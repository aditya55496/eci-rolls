"""Task `repair-any`: the part repairs again on stored ocr-any output, without new OCR (2026-11-25).

ocr-any already runs the repairs on every part (hi_common.repair_part: final_deleted, serial_gaps; the EPIC repair from the 2027
roll runs inside eci_ocr when hints exist). This task is for outputs written before a repair existed or changed, as
repair_task.py was for UP, in the layout ocr/<state>/<ac>/<part>.jsonl.gz.
Queue item (project eci-rolls-hi-repair): key "<fix>/<state>/<ac>", payload {"state", "ac", "p": [parts], "fix": "parts" |
"epic", "hints": key or null}. fix "parts" = final_deleted + serial_gaps; fix "epic" = epicfix.fix + fix2 against the 2027
hints (only for items with a hints key), fix2 with the guard switches recorded in the records' ocr_rules. A changed file is written back to eci-rolls-hi-ocr (one file per call); the file it
replaced is kept once under ocr_prev/<fix>/<state>/<ac>/<part>.jsonl.gz in this project. Records changed by this task get
"repaired_by": [<fix>, ...] (the record's ocr_* provenance stays that of the OCR run). One hub record per item: counts.
"""
import gzip
import json
import os
import sys
import tempfile
import time

from scrape_hub_client import HubClient, stop_flag

import hi_common as H
from common import download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eciocr", "prod"))
import epicfix  # noqa: E402

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", H.REPAIR_HI_PROJECT)
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(600, _BUDGET / 4)
FIXES = ("parts", "epic")


def _mark(recs, before, fix):
    n = 0
    for r, b in zip(recs, before):
        if json.dumps(r, sort_keys=True, ensure_ascii=False) != b:
            r.setdefault("repaired_by", []).append(fix)
            n += 1
    return n


def do_item(hub, item, ocr_hub, fetch=None, upload_ocr=None, upload_own=None):
    p = item["payload"]
    st, ac, fix = str(p["state"]), int(p["ac"]), p.get("fix")
    if fix not in FIXES:
        raise RuntimeError(f"unknown fix {fix!r}")
    parts = [int(x) for x in p["p"]]
    fetch = fetch or (lambda ks, dest: {k: v for proj in {x.split("/", 1)[0] for x in ks}
                                        for k, v in download(hub, proj, [x for x in ks if x.startswith(proj + "/")], dest).items()})
    upload_ocr = upload_ocr or ocr_hub.upload_many
    upload_own = upload_own or hub.upload_many
    out = {"state": st, "ac": ac, "fix": fix, "parts": len(parts), "files": 0, "files_changed": 0, "records": 0, "changed": {}}
    with tempfile.TemporaryDirectory() as tmp:
        idx = None
        if fix == "epic":
            if not p.get("hints"):
                raise RuntimeError("fix epic needs a hints key")
            hp = fetch([p["hints"]], tmp)[p["hints"]]
            if not hp:
                raise RuntimeError(f"no 2027 hints: {p['hints']}")
            idx = epicfix.index(json.loads(l) for l in gzip.open(hp, "rt", encoding="utf-8") if l.strip())
        keys = [f"{H.OCR_HI_PROJECT}/ocr/{st}/{ac}/{x}.jsonl.gz" for x in parts]
        got = fetch(keys, tmp)
        for x, k in zip(parts, keys):
            if not got.get(k):
                continue
            raw = open(got[k], "rb").read()
            recs = [json.loads(l) for l in gzip.decompress(raw).decode("utf-8").splitlines() if l.strip()]
            out["files"] += 1
            out["records"] += len(recs)
            before = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in recs]
            if fix == "parts":
                sets = {r.get("ocr_model_set") for r in recs}
                ms = H.MODEL_SETS.get(next(iter(sets))) if len(sets) == 1 else None
                engines = ["crnn_" + m for m in ms["models"]] if ms else None
                c = H.repair_part(recs, engines)
            else:
                voters = [r for r in recs if not r.get("not_voter_box")]
                by_roll = {}
                for r in voters:
                    by_roll.setdefault(r.get("roll"), []).append(r)
                c = {"epic_fix1": 0, "epic_fix2": 0}
                for roll, vs in by_roll.items():
                    gap = 2027 - H.roll_year(roll) if roll else 1
                    # fix2 with the guard switches of the OCR run (ocr_rules), not of this process: without them the unguarded
                    # fix2 would put back the swaps the run refused (ocrv5 review 2: S20/14/70 serial 937)
                    rules = vs[0].get("ocr_rules") or {}
                    epicfix.FIX2_GUARD = str(rules.get("ECI_EPICFIX2_GUARD", "0")) == "1"
                    epicfix.FIX2_MIN = float(rules.get("ECI_EPICFIX2_MIN", "60"))
                    c["epic_fix1"] += epicfix.fix(vs, idx, gap)
                    c["epic_fix2"] += epicfix.fix2(vs, idx, gap)
            n = _mark(recs, before, fix)
            for a, b in c.items():
                out["changed"][a] = out["changed"].get(a, 0) + b
            if n:
                out["files_changed"] += 1
                upload_own([(f"ocr_prev/{fix}/{st}/{ac}/{x}.jsonl.gz", raw, "application/gzip")])
                upload_ocr([(f"ocr/{st}/{ac}/{x}.jsonl.gz", gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode(), mtime=0),
                             "application/gzip")])
            os.remove(got[k])
    return out


def main():
    stop = stop_flag()
    hub = HubClient(PROJECT, job_index=int(os.environ.get("SCRAPE_JOB_INDEX", 0)), total_jobs=int(os.environ.get("SCRAPE_TOTAL_JOBS", 1)))
    ocr_hub = HubClient(H.OCR_HI_PROJECT, note="repair-any: writes repaired OCR files")
    try:
        while time.time() < DEADLINE and not stop.is_set():
            if hub.settings.get("paused"):
                time.sleep(30)
                continue
            items = hub.claim(1, steal=True, retry_failed=True)
            if not items:
                break
            it = items[0]
            try:
                rec = do_item(hub, it, ocr_hub)
                hub.record(it["key"], rec)
                hub.metric("records_repaired", sum(rec["changed"].values()))
                hub.complete([it["key"]], "done", fetched=sum(rec["changed"].values()))
            except Exception as e:  # one bad AC must not stop the worker
                hub.error()
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:300])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
        ocr_hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        ocr_hub.close("failed", "repair-any worker failed")
        raise


if __name__ == "__main__":
    main()
