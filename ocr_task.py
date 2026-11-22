"""Task `ocr`: OCR of one part's 4 image rolls (SIR draft 2026, SIR final 2026, supplement 2 and 3) on the runner's 4 CPUs.

Queue item (project eci-rolls-up-ocr): key "<ac>-<part>", payload {"ac": 174, "part": 1, "keys": {"2026-sir-draft": key, ...}}.
The engine is eciocr/prod/eci_ocr.py exactly as benchmarked (bench report: 97.84 % whole-record accuracy on the gold set, CI
97.13-98.43, 4.5 % of boxes flagged; 99.63 % on unflagged boxes): cascade --stage1 1 over the int8 CRNNs c11q, c9q, c4q, c3q plus
Tesseract, with the AC's 2027 text-layer records as confirmation only (hints/<ac>.jsonl.gz from the textlayer task).
Output in the bucket: ocr/<ac>/<part>.jsonl.gz (one record per voter box, all engine fields + flags) and ocr/<ac>/<part>.summary.json.
One hub record per part: counts, flag rate and serial checks. No LLM, no network other than the hub and the bucket links.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from scrape_hub_client import HubClient, stop_flag

from common import FETCH_PROJECT, OCR_PROJECT, OCR_ROLLS, TEXT_PROJECT, download

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", OCR_PROJECT)
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(600, _BUDGET / 4)     # an item takes ~2 min; keep 10 min spare
MODELS = [os.path.join(HERE, "models", m) for m in ("c11q", "c9q", "c4q", "c3q")]
CPUS = len(os.sched_getaffinity(0))


def do_item(hub, item):
    p = item["payload"]
    ac, part = int(p["ac"]), int(p["part"])
    keys, tag = p["keys"], p.get("tag", "")         # tag: "" for the SIR rolls; "-supp" when the supplements are read later
    with tempfile.TemporaryDirectory() as tmp:
        hk = f"{TEXT_PROJECT}/hints/{ac}.jsonl.gz"
        got = download(hub, FETCH_PROJECT, [keys[r] for r in OCR_ROLLS if keys.get(r)], tmp)
        hint = download(hub, TEXT_PROJECT, [hk], tmp)[hk]
        if not hint:
            raise RuntimeError(f"no 2027 hints for AC {ac} yet")
        hpath = os.path.join(tmp, "hints.jsonl")
        with gzip.open(hint) as f, open(hpath, "wb") as o:
            shutil.copyfileobj(f, o)
        pdfs = [got[keys[r]] for r in OCR_ROLLS if keys.get(r) and got.get(keys[r])]
        missing = [r for r in OCR_ROLLS if keys.get(r) and not got.get(keys[r])]
        if not pdfs:
            raise RuntimeError("none of the 4 PDFs is in the bucket")
        out = os.path.join(tmp, "out.jsonl")
        cmd = [sys.executable, os.path.join(HERE, "eciocr", "prod", "eci_ocr.py"), *pdfs, "--out", out, "--hint-2027", hpath,
               "--stage1", "1", "--workers", str(CPUS), "--ort-threads", "1", "--quiet"]
        for m in MODELS:
            cmd += ["--crnn", m]
        t = time.time()
        r = subprocess.run(cmd, timeout=3600, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode:
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and "_discovery.cc" not in l and "onnxruntime" not in l.lower()]
            tb = [l for l in lines if "Error" in l or "error" in l or "Killed" in l or "Traceback" in l]
            raise RuntimeError(f"eci_ocr exit {r.returncode}: " + " | ".join((tb[-3:] or lines[-4:]))[-350:])
        secs = time.time() - t
        summ = json.load(open(out + ".summary.json"))
        hub.upload_many([(f"ocr/{ac}/{part}{tag}.jsonl.gz", gzip.compress(open(out, "rb").read(), mtime=0), "application/gzip"),
                         (f"ocr/{ac}/{part}{tag}.summary.json", json.dumps(summ, ensure_ascii=False).encode(), "application/json")])
    hub.metric("voter_pages", summ.get("voter_pages", 0))
    hub.metric("ocr_s", round(secs, 1))
    rec = {"ac": ac, "part": part, "tag": tag, "records": summ.get("records"), "voter_pages": summ.get("voter_pages"), "flag_rate": summ.get("flag_rate"),
           "s_per_voter_page": summ.get("s_per_voter_page_wall"), "missing_rolls": missing,
           "pdfs": [{k: d.get(k) for k in ("pdf", "kind", "records", "flag_rate", "hint_matched", "serial_check")} for d in summ.get("pdfs", [])]}
    hub.record(f"{ac}-{part}{tag}", rec)
    if missing:
        hub.failure("missing-roll", f"{ac}-{part}{tag}", "a PDF named in the item was not in the bucket", rolls=missing)
    return summ.get("records", 0)


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
            except Exception as e:  # one bad part must not stop the worker
                hub.error()
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:400])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
