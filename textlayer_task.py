"""Task `textlayer`: the 2027 SSR draft roll of one AC from its PDF text layer (no OCR).

Queue item (project eci-rolls-up-2027): key "<ac>", payload {"ac": 174, "keys": [bucket keys of that AC's 2027 part PDFs]}.
Output in the bucket: text/<ac>.jsonl.gz (every voter record, every field), hints/<ac>.jsonl.gz (the --hint-2027 form the OCR
task uses to CONFIRM its readings) and text/<ac>.summary.json (per-PDF counts, cover totals, repairs, warnings).
One hub record per AC: counts only. The records themselves stay in the bucket.
"""
import json
import os
import sys
import tempfile
import time

from scrape_hub_client import HubClient, stop_flag

from common import FETCH_PROJECT, TEXT_PROJECT, download, gz_jsonl, hint_of

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eciocr", "textlayer"))
import parse_textlayer as T  # noqa: E402

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", TEXT_PROJECT)
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(300, _BUDGET / 4)


def do_item(hub, item):
    ac = int(item["payload"]["ac"])
    keys = sorted(item["payload"]["keys"])
    with tempfile.TemporaryDirectory() as tmp:
        got = download(hub, FETCH_PROJECT, keys, tmp)
        missing = [k for k, p in got.items() if not p]
        recs, summ = [], []
        for k in keys:
            if not got[k]:
                continue
            r, meta = T.parse_pdf(got[k])
            val = T.validate(r, meta)
            meta["validation"] = val
            meta["key"] = k
            summ.append(meta)
            recs += r
            hub.metric("pdfs_parsed")
            hub.metric("pages", meta.get("pages", 0))
    bad = [m["key"] for m in summ if not m["validation"]["ok"]]
    files = [("text/%d.jsonl.gz" % ac, gz_jsonl(recs), "application/gzip"),
             ("hints/%d.jsonl.gz" % ac, gz_jsonl(hint_of(r) for r in recs if r.get("epic")), "application/gzip"),
             ("text/%d.summary.json" % ac, json.dumps(summ, ensure_ascii=False, default=str).encode(), "application/json")]
    hub.upload_many(files)
    hub.record(str(ac), {"ac": ac, "pdfs": len(summ), "records": len(recs), "missing_pdfs": missing, "not_ok": bad,
                         "cover_total": sum((m["validation"].get("cover_total") or 0) for m in summ)})
    if missing:
        hub.failure("missing-pdf", str(ac), f"{len(missing)} of {len(keys)} PDFs not in the bucket", keys=missing[:20])
    return len(recs)


def main():
    stop = stop_flag()
    hub = HubClient(PROJECT, job_index=int(os.environ.get("SCRAPE_JOB_INDEX", 0)), total_jobs=int(os.environ.get("SCRAPE_TOTAL_JOBS", 1)))
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
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:200])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
