"""Task `parquet-any`: one AC of a Hindi-state SIR roll as Parquet files, from the ocr-any output (2026-11-25).

Queue item (project eci-rolls-hi-parquet, items from plan_hi_ocr.py --parquet): key "<state>/<ac>", payload
{"state": "S12", "ac": 230, "parts": [1, 2, ...], "rolls": ["S12-2026-DR", "S12-2026-FIR"]}.
Reads ocr/<state>/<ac>/<part>.jsonl.gz of eci-rolls-hi-ocr (both rolls in one file, after the repairs) and writes one file per
roll: parquet/<state>/<roll id>/ac_<ac>/data.parquet (zstd). Form-route items (formocr): key "<state>/<roll group>/<ac>", payload
+ "roll_group"; read from ocr/<state>/<roll group>/<ac>/; the output path is the same form (the roll id keeps it apart). Schema = the SIR schema of parquet_task.py (OCR_SCHEMA; column
'roll' = the roll id) plus five provenance columns at the end: ocr_model_set (string), ocr_models (list of strings), ocr_code
(string, git commit), ocr_rules (JSON text: env switches, stage1, roll_year), hints_used (bool). Keys outside the schema go to
the JSON column 'extra' (page_scale, extra_part, epic_by_prefix, ...), so nothing is dropped. One hub record per AC: rows per
roll, rows per model set, parts without an OCR file.
"""
import gzip
import io
import json
import os
import tempfile
import time

from scrape_hub_client import HubClient, stop_flag

import hi_common as H
import parquet_task as PT
from common import download

pa, pq = PT.pa, PT.pq
PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", H.PARQUET_HI_PROJECT)
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(600, _BUDGET / 4)
S, B = pa.string(), pa.bool_()
SCHEMA = pa.schema(list(PT.OCR_SCHEMA) + [("ocr_model_set", S), ("ocr_models", pa.list_(S)), ("ocr_code", S), ("ocr_rules", S),
                                          ("hints_used", B)])


def table(rows, roll, ac):
    for r in rows:
        if isinstance(r.get("ocr_rules"), dict):
            r["ocr_rules"] = json.dumps(r["ocr_rules"], ensure_ascii=False, sort_keys=True)
        r["hints_used"] = bool(r.get("hints_used"))
    return PT.to_table(rows, SCHEMA, roll, ac)


def do_item(hub, item, fetch=None, upload=None):
    p = item["payload"]
    st, ac, parts = str(p["state"]), int(p["ac"]), [int(x) for x in p["parts"]]
    base = H.ocr_base(st, ac, p.get("roll_group"))     # form-route outputs (formocr): ocr/<state>/<roll group>/<ac>
    fetch = fetch or (lambda ks, dest: download(hub, H.OCR_HI_PROJECT, ks, dest))
    upload = upload or hub.upload_many
    by, missing = {r: [] for r in p.get("rolls") or []}, []
    with tempfile.TemporaryDirectory() as tmp:
        keys = [f"{H.OCR_HI_PROJECT}/{base}/{x}.jsonl.gz" for x in parts]
        got = fetch(keys, tmp)
        for x, k in zip(parts, keys):
            if not got.get(k):
                missing.append(x)
                continue
            with gzip.open(got[k], "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        by.setdefault(r["roll"], []).append(r)
            os.remove(got[k])
    rec = {"state": st, "ac": ac, "parts": len(parts), "parts_without_ocr_file": missing, "rolls": {}}
    if p.get("roll_group") is not None:
        rec["roll_group"] = p["roll_group"]
    for roll in sorted(by, key=lambda r: (0 if H.is_draft(r) else 1, r)):
        rows = by[roll]
        if not rows:
            rec["rolls"][roll] = {"rows": 0}
            continue
        sets = {}
        for r in rows:
            sets[r.get("ocr_model_set")] = sets.get(r.get("ocr_model_set"), 0) + 1
        buf = io.BytesIO()
        pq.write_table(table(rows, roll, ac), buf, compression="zstd", compression_level=9, row_group_size=131072)
        data = buf.getvalue()
        upload([(f"parquet/{st}/{roll}/ac_{ac}/data.parquet", data, "application/vnd.apache.parquet")])
        nv = sum(1 for r in rows if r.get("not_voter_box"))
        rec["rolls"][roll] = {"rows": len(rows), "voters": len(rows) - nv, "not_voter_box": nv, "bytes": len(data),
                              "rows_by_model_set": sets, "hints_used_rows": sum(1 for r in rows if r.get("hints_used"))}
    return rec


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
                rec = do_item(hub, it)
                hub.record(it["key"], rec)
                n = sum(v.get("rows", 0) for v in rec["rolls"].values())
                hub.metric("parquet_rows", n)
                hub.complete([it["key"]], "done", fetched=n)
            except Exception as e:  # one bad AC must not stop the worker
                hub.error()
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:400])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
