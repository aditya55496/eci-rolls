"""Task `ocr-any`: OCR of one part of one AC of a Hindi-state SIR roll (draft + final when both exist), 2026-11-25.

Queue item (project eci-rolls-hi-ocr, items from plan_hi_ocr.py): key "<state>/<ac>/<part>", payload
  {"state": "S12", "ac": 230, "part": 10, "lang": "HIN", "keys": {"S12-2026-DR": key, "S12-2026-FIR": key},
   "hints": "eci-rolls-ssr-text/hints/S12-2027-DR/230.jsonl.gz" or null, "model_set": "v4a" | "j2Ps" | ...,
   "extra_rolls": [roll ids where this part is above the portal's part list, optional]}
Keys and the hints key start with the project that stored them; each file is read through that project's links.
Model set: model_sets.json (CRNN dirs, eci_ocr env switches, hints mode, stage1). The hub settings of this project can swap the
set of every pending item at once (hi_common.choose_set): "ocr_model_set_map" {"j2Ps": "newset"} or "ocr_model_set" "newset".
A set this checkout does not have -> the worker takes no item and exits 3 (the runner waits and asks again; runner jobs that
start later check out the newer code).
Engine: eciocr/prod/eci_ocr.py, one run per PDF, draft first; the draft's readings are the final roll's cache (as ocr_task.py).
With hints: --hint-2027 (the whole AC's 2027 SSR records; confirmation, EPIC repair inside eci_ocr) and --stage1 of the set.
Without hints: all engines on every box, the set's no-hint rules.
Then the part repairs (hi_common.repair_part: final_deleted, serial_gaps) on the part's records.
Output in the bucket: ocr/<state>/<ac>/<part>.jsonl.gz (one record per box, all engine fields, flags, readings of flagged fields,
repair markers, and the provenance: roll, ocr_model_set, ocr_models, ocr_code, ocr_rules, hints_used) and
ocr/<state>/<ac>/<part>.summary.json. One hub record per part (id = item key): counts, flag rate, serial checks, repairs, CPU,
provenance. No LLM, no network other than the hub and the bucket links.
"""
import gzip
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

from scrape_hub_client import HubClient, stop_flag

import hi_common as H
from common import download

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", H.OCR_HI_PROJECT)
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(600, _BUDGET / 4)     # an item takes ~1-4 min; keep 10 min spare
CPUS = len(os.sched_getaffinity(0))
WORKERS = int(os.environ.get("ECI_OCR_WORKERS") or max(1, CPUS // 2))   # 2 worker processes per 4-CPU runner job, 2 OCR processes each


def fetch_hub(hub, keys, dest):
    """{key: local path or None}, each key through the project named by its first segment."""
    got = {}
    for proj in sorted({k.split("/", 1)[0] for k in keys}):
        got.update(download(hub, proj, [k for k in keys if k.startswith(proj + "/")], dest))
    return got


def _run(cmd, env, timeout=3600):
    """-> (exit code, output, rusage of this eci_ocr run and its pool workers). os.wait4 gives the CPU of this child alone
    (os.times() would also count other threads' children)."""
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    timer = threading.Timer(timeout, p.kill)
    timer.start()
    try:
        text = p.stdout.read()
        _, status, ru = os.wait4(p.pid, 0)
    finally:
        timer.cancel()
        p.stdout.close()
    p.returncode = os.waitstatus_to_exitcode(status)    # Popen must not wait for the pid again
    return p.returncode, text, ru


def run_eci(pdfs, hints_path, ms, dirs, year, tmp):
    """pdfs: [(roll id, path)] in roll order -> (records, per-run summaries, CPU seconds of the eci_ocr processes)."""
    # only the set's decision switches: an ECI_* variable from the runner or pool env must not change the run behind the
    # provenance (ECI_OCR_WORKERS and ECI_TESS_CLI change speed / call path only)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECI_") or k in ("ECI_OCR_WORKERS", "ECI_TESS_CLI")}
    env.update(ms.get("env", {}))
    recs, summs, prev, cpu = [], [], None, 0.0
    for i, (roll, pdf) in enumerate(pdfs):
        o1 = os.path.join(tmp, f"out{i}.jsonl")
        cmd = [sys.executable, os.path.join(HERE, "eciocr", "prod", "eci_ocr.py"), pdf, "--out", o1, "--workers", str(WORKERS),
               "--ort-threads", "1", "--quiet", "--roll-year", str(year)]
        if hints_path:
            cmd += ["--hint-2027", hints_path, "--stage1", str(int(ms.get("stage1", 0)))]
        for d in dirs:
            cmd += ["--crnn", d]
        if prev:
            cmd += ["--cache-in", prev]
        if i + 1 < len(pdfs):
            prev = os.path.join(tmp, f"cache{i}.pkl")
            cmd += ["--cache-out", prev]
        rc, text, ru = _run(cmd, env)
        cpu += ru.ru_utime + ru.ru_stime
        if rc:
            lines = [l.strip() for l in text.splitlines() if l.strip() and "_discovery.cc" not in l and "onnxruntime" not in l.lower()]
            tb = [l for l in lines if "Error" in l or "error" in l or "Killed" in l or "Traceback" in l]
            raise RuntimeError(f"eci_ocr exit {rc} on {roll}: " + " | ".join((tb[-3:] or lines[-4:]))[-350:])
        with open(o1, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec["roll"] = roll
                    recs.append(rec)
        s = json.load(open(o1 + ".summary.json", encoding="utf-8"))
        s["roll"] = roll
        summs.append(s)
        os.remove(o1)
    return recs, summs, cpu


def do_item(hub, item, settings=None, fetch=None, upload=None):
    """fetch(keys, dest) -> {key: path or None} (default: hub read links); upload(files) (default: hub.upload_many).
    Returns the hub record of the part."""
    p = item["payload"]
    state, ac, part = str(p["state"]), int(p["ac"]), int(p["part"])
    if (p.get("lang") or "HIN") != "HIN":
        raise RuntimeError(f"lang {p.get('lang')} not supported (eci_ocr reads Hindi SIR rolls)")
    keys = {r: k for r, k in (p.get("keys") or {}).items() if k}
    if not keys:
        raise RuntimeError("payload has no PDF keys")
    name, set_src = H.choose_set(p, settings if settings is not None else getattr(hub, "settings", {}))
    ms = H.MODEL_SETS[name]
    dirs = H.model_dirs(name)
    want_hints = bool(p.get("hints")) and ms.get("hints") != "none"
    rolls = H.roll_order(keys)
    years = {H.roll_year(r) for r in rolls}
    if len(years) != 1:
        raise RuntimeError(f"rolls of different years in one item: {rolls}")
    year = years.pop()
    fetch = fetch or (lambda ks, dest: fetch_hub(hub, ks, dest))
    upload = upload or hub.upload_many
    with tempfile.TemporaryDirectory() as tmp:
        got = fetch([keys[r] for r in rolls] + ([p["hints"]] if want_hints else []), tmp)
        hpath, nh = None, 0
        if want_hints:
            hz = got.get(p["hints"])
            if not hz:
                raise RuntimeError(f"no 2027 hints yet: {p['hints']}")
            # all 2027 records of the AC (2027 parts are numbered differently); eci_ocr loads them once before it forks
            hpath = os.path.join(tmp, "hints.jsonl")
            with gzip.open(hz, "rt", encoding="utf-8") as f, open(hpath, "w", encoding="utf-8") as o:
                for line in f:
                    if line.strip():
                        o.write(line)
                        nh += 1
            os.remove(hz)
            if not nh:
                raise RuntimeError(f"2027 hints file is empty: {p['hints']}")
        missing = [r for r in rolls if not got.get(keys[r])]
        pdfs = []
        os.makedirs(os.path.join(tmp, "pdf"), exist_ok=True)
        for r in rolls:     # the portal's file name as 'source' in the records (download() names files by the whole key)
            if got.get(keys[r]):
                d = os.path.join(tmp, "pdf", os.path.basename(keys[r]))
                os.rename(got[keys[r]], d)
                pdfs.append((r, d))
        if not pdfs:
            raise RuntimeError(f"none of the {len(rolls)} PDFs is in the bucket")
        t = time.time()
        recs, summs, cpu = run_eci(pdfs, hpath, ms, dirs, year, tmp)
        secs = time.time() - t
    engines = ["crnn_" + m for m in ms["models"]]
    repairs = H.repair_part(recs, engines)
    prov = {"ocr_model_set": name, "ocr_models": list(ms["models"]), "ocr_code": H.CODE,
            "ocr_rules": H.rules_of(ms, year, hpath is not None), "hints_used": hpath is not None}
    extra = set(p.get("extra_rolls") or [])
    for r in recs:
        r.update(prov)
        if r["roll"] in extra:
            r["extra_part"] = True
    voters = [r for r in recs if not r.get("not_voter_box")]
    nfl = sum(1 for r in voters if r.get("flags"))
    by_roll = {}
    for r in voters:
        b = by_roll.setdefault(r["roll"], {"records": 0, "flagged": 0})
        b["records"] += 1
        b["flagged"] += bool(r.get("flags"))
    vp = sum(s["voter_pages"] for s in summs)
    summ = {"item": item["key"], "state": state, "ac": ac, "part": part, **prov, "model_set_source": set_src,
            "model_sha256_16": ms.get("sha256_16"), "hints_key": p.get("hints") if want_hints else None, "hints_records": nh,
            "rolls": rolls, "missing_rolls": missing, "records": len(voters), "not_voter_boxes": len(recs) - len(voters),
            "flagged_records": nfl, "flag_rate": round(nfl / max(1, len(voters)), 4), "by_roll": by_roll, "repairs": repairs,
            "voter_pages": vp, "wall_s": round(secs, 1), "cpu_s": round(cpu, 1), "cpu_s_per_voter_page": round(cpu / max(1, vp), 2),
            "workers": WORKERS, "runs": [{k: s.get(k) for k in ("roll", "wall_s", "cache_in", "cache_out", "voter_pages", "peak_worker_rss_mb")} for s in summs],
            "pdfs": [dict(d, roll=s["roll"]) for s in summs for d in s["pdfs"]]}
    body = gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode(), mtime=0)
    upload([(f"ocr/{state}/{ac}/{part}.jsonl.gz", body, "application/gzip"),
            (f"ocr/{state}/{ac}/{part}.summary.json", json.dumps(summ, ensure_ascii=False).encode(), "application/json")])
    rec = {k: summ[k] for k in ("state", "ac", "part", "ocr_model_set", "ocr_models", "ocr_code", "ocr_rules", "hints_used",
                                "model_set_source", "hints_records", "rolls", "missing_rolls", "records", "not_voter_boxes",
                                "flag_rate", "by_roll", "repairs", "voter_pages", "wall_s", "cpu_s", "cpu_s_per_voter_page")}
    rec["pdfs"] = [{k: d.get(k) for k in ("roll", "pdf", "kind", "records", "flag_rate", "hint_matched", "epic_from_2027", "serial_check",
                                          "worker_s_total")} for d in summ["pdfs"]]
    return rec


def main():
    stop = stop_flag()
    hub = HubClient(PROJECT, job_index=int(os.environ.get("SCRAPE_JOB_INDEX", 0)), total_jobs=int(os.environ.get("SCRAPE_TOTAL_JOBS", 1)))
    code = 0
    try:
        while time.time() < DEADLINE and not stop.is_set():
            if hub.settings.get("paused"):
                time.sleep(30)
                continue
            unknown = H.hub_override_unknown(hub.settings)
            if unknown:      # the hub names a newer model set: leave the items to workers with newer code
                hub.metric("model_set_unknown")
                code = 3
                break
            items = hub.claim(1, steal=True, retry_failed=True)
            if not items:
                break
            it = items[0]
            try:
                rec = do_item(hub, it, settings=hub.settings)
                hub.record(it["key"], rec)
                if rec["missing_rolls"]:
                    hub.failure("missing-roll", it["key"], "a PDF named in the item was not in the bucket", rolls=rec["missing_rolls"])
                hub.metric("voter_pages", rec["voter_pages"])
                hub.metric("ocr_cpu_s", rec["cpu_s"])
                hub.metric(f"set_{rec['ocr_model_set']}")
                hub.complete([it["key"]], "done", fetched=rec["records"])
            except H.UnknownModelSet as e:
                hub.complete([it["key"]], "failed", error=f"model set {e} not in this checkout")
                code = 3
                break
            except Exception as e:  # one bad part must not stop the worker
                hub.error()
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:400])
        if stop.is_set():
            hub.close("failed", "stopped by signal")
        else:
            hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise
    sys.exit(code)


if __name__ == "__main__":
    main()
