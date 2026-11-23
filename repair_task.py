"""Task `repair`: apply the EPIC repairs (eciocr/prod/epicfix.py: fix + fix2) to the finished SIR OCR output of one AC.

Queue item (project eci-rolls-up-repair): key "<ac>", payload {"ac": 174, "parts": [1, 2, ...]}.
Second fix (2026-11-23): key "final_deleted/<ac>", payload {"ac", "parts", "fix": "final_deleted"}: SIR final-roll voters deleted
after the draft (stamped boxes taken for page noise) get the draft's fields (epicfix.final_deleted); originals kept under ocr_v2/.
Outputs made before the repairs existed (2026-11-23) still carry EPICs whose prefix the CRNNs changed. No new OCR: the rules need
only the output and the AC's 2027 roll (hints/<ac>.jsonl.gz). For each part: ocr/<ac>/<part>.jsonl.gz is read, repaired and
written back only when something changed; the original is kept once as ocr_v1/<ac>/<part>.jsonl.gz. One hub record per AC with
the counts. Records keep the OCR value in 'epic_ocr' and get 'epic_from_2027': true.
"""
import gzip
import json
import os
import sys
import tempfile
import time

from scrape_hub_client import HubClient, stop_flag

from common import OCR_PROJECT, TEXT_PROJECT, download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eciocr", "prod"))
import epicfix  # noqa: E402

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "eci-rolls-up-repair")
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(600, _BUDGET / 4)


def do_item(hub, item):
    ac = int(item["payload"]["ac"])
    parts = [int(p) for p in item["payload"]["parts"]]
    fix = item["payload"].get("fix", "epic")
    with tempfile.TemporaryDirectory() as tmp:
        idx = None
        if fix == "epic":
            hk = f"{TEXT_PROJECT}/hints/{ac}.jsonl.gz"
            hp = download(hub, TEXT_PROJECT, [hk], tmp)[hk]
            if not hp:
                raise RuntimeError(f"no 2027 hints for AC {ac}")
            idx = epicfix.index(json.loads(l) for l in gzip.open(hp, "rt", encoding="utf-8"))
        keys = [f"{OCR_PROJECT}/ocr/{ac}/{p}.jsonl.gz" for p in parts]
        got = download(hub, OCR_PROJECT, keys, tmp)
        n_files = n_changed = n_fix1 = n_fix2 = n_recs = 0
        for p, k in zip(parts, keys):
            if not got.get(k):
                continue
            raw = open(got[k], "rb").read()
            recs = [json.loads(l) for l in gzip.decompress(raw).decode("utf-8").splitlines() if l.strip()]
            n_files += 1
            n_recs += len(recs)
            if fix == "final_deleted":                # SIR final: deleted voters taken for page noise (epicfix.final_deleted)
                a, b = epicfix.final_deleted(recs), 0
            else:
                voters = [r for r in recs if not r.get("not_voter_box")]
                a = epicfix.fix(voters, idx)
                b = epicfix.fix2(voters, idx, 1)          # SIR 2026 vs the 2027 roll: age gap 0 or 1
            if a or b:
                n_changed += 1
                n_fix1 += a
                n_fix2 += b
                out = gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode(), mtime=0)
                hub.upload_many([(f"{'ocr_v1' if fix == 'epic' else 'ocr_v2'}/{ac}/{p}.jsonl.gz", raw, "application/gzip")])
                # the repaired file belongs to the OCR project: written through a client of that project, one file per call
                # (a whole AC in one call is over the hub's limit: 422 too_long / 520)
                ocr_hub.upload_many([(f"ocr/{ac}/{p}.jsonl.gz", out, "application/gzip")])
    hub.record(str(ac) if fix == "epic" else f"{fix}/{ac}", {"ac": ac, "fix": fix, "parts": len(parts), "files": n_files, "records": n_recs, "files_changed": n_changed,
                         "epic_fix1": n_fix1, "epic_fix2": n_fix2})
    hub.metric("epics_repaired" if fix == "epic" else f"{fix}_repaired", n_fix1 + n_fix2)
    return n_fix1 + n_fix2


ocr_hub = None


def main():
    global ocr_hub
    stop = stop_flag()
    hub = HubClient(PROJECT, job_index=int(os.environ.get("SCRAPE_JOB_INDEX", 0)), total_jobs=int(os.environ.get("SCRAPE_TOTAL_JOBS", 1)))
    ocr_hub = HubClient(OCR_PROJECT, note="repair: writes repaired OCR files")
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
                n = do_item(hub, it)
                hub.complete([it["key"]], "done", fetched=n)
            except Exception as e:  # one bad AC must not stop the worker
                hub.error()
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:300])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
        ocr_hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        ocr_hub.close("failed", "repair worker failed")
        raise


if __name__ == "__main__":
    main()
