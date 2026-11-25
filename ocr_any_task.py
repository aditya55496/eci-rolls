"""Task `ocr-any`: OCR of one part of one AC of a Hindi-state SIR roll (draft + final when both exist), 2026-11-25.

Queue item (project eci-rolls-hi-ocr, items from plan_hi_ocr.py): key "<state>/<ac>/<part>", payload
  {"state": "S12", "ac": 230, "part": 10, "lang": "HIN", "keys": {"S12-2026-DR": key, "S12-2026-FIR": key},
   "hints": "eci-rolls-ssr-text/hints/S12-2027-DR/230.jsonl.gz" or null, "model_set": "v4a" | "j2Ps" | ...,
   "extra_rolls": [roll ids where this part is above the portal's part list, optional], "v": 1 (payload format, absent = 1)}
Form-route items (formocr, plan_form_ocr.py, payload v 2): key "<state>/<roll group>/<ac>/<part>", payload as above + "roll_group"
  (GE: the roll id, e.g. "S12-2024-GEN-FIR"; SSR: "<state>-<year>-SSR" with the draft and/or final roll of that year in "keys");
  output ocr/<state>/<roll group>/<ac>/<part>.jsonl.gz (+ .summary.json), hub record id = item key, + "roll_group".
English / Tamil SIR items (engocr, plan_lang_ocr.py, payload v 3; project eci-rolls-all-ocr): key "<state>/<lang>/<ac>/<part>"
  (lang ENG or TAM: never the same as a Hindi key, and West Bengal's English ACs stay apart from its other ACs), payload as the SIR
  payload with "lang" ENG / TAM and a set of that language; output ocr/<state>/<lang>/<ac>/<part>.jsonl.gz (+ .summary.json).
  eci_ocr gets --lang, the set's extra arguments (job4 set: --tess2) and its --stage1 also without hints (sets with
  "stage1_nohint"); with "draft_records" the final roll's run reads the draft roll's records of the same item (--draft-records:
  flagged fields of a final record take the unflagged draft value of the same EPIC; job4, West Bengal 'UNDER ADJUDICATION' stamps).
  Tamil needs Tesseract's tam traineddata (scrape.json setup installs it; a worker without it installs it or gives the item back).
Keys and the hints key start with the project that stored them; each file is read through that project's links.
Model set: model_sets.json (CRNN dirs, eci_ocr env switches, hints mode, stage1). The hub settings of this project can swap the
set of every pending item at once (hi_common.choose_set): "ocr_model_set_map" {"j2Ps": "newset"} or "ocr_model_set" "newset".
A set this checkout does not have, or whose model files are missing / have another sha256 -> the worker gives the item back
untried (status 'pending', the attempt is not counted) and exits 75 (hi_common.EXIT_NEEDS_NEWER_CODE). A runner with the
family-refetch change fetches the family's main again at once and goes on with the new code; an older runner waits 30-300 s,
asks again, and idles like this until its job ends (jobs that start later check out the newer code).
Provenance: the worker refuses to start when the family commit is unknown, and gives an item back and exits 75 when the
checkout moved to another commit during the item (hi_common.code_commit), also when the item failed while it moved (a
sibling worker's refetch can replace files under a running eci_ocr). A payload 'v' this checkout does not know -> given back,
exit 75. A PDF the item names that cannot be read from the bucket -> the item fails and is retried (never done with a roll less).
Engine: eciocr/prod/eci_ocr.py, one run per PDF, draft first; the draft's readings are the final roll's cache (as ocr_task.py).
With hints: --hint-2027 (the whole AC's 2027 SSR records; confirmation, EPIC repair inside eci_ocr) and --stage1 of the set.
Without hints: all engines on every box, the set's no-hint rules.
Then the part repairs (hi_common.repair_part: final_deleted, serial_gaps) on the part's records.
Output in the bucket: ocr/<state>/<ac>/<part>.jsonl.gz (one record per box, all engine fields, flags, readings of flagged fields,
repair markers, and the provenance: roll, ocr_model_set, ocr_models, ocr_code, ocr_rules, hints_used) and
ocr/<state>/<ac>/<part>.summary.json (uploaded first: with two workers on one item after a stall, the jsonl that stays is
the one written last, and its records carry their own provenance). The summary and the hub record also name the tool
versions (pymupdf, onnxruntime, Tesseract, hin traineddata) and the sha256 of the hints file. One hub record per part (id = item key): counts, flag rate, serial checks, repairs, CPU,
provenance. No LLM, no network other than the hub and the bucket links.
"""
import gzip
import json
import os
import re
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
_TOOLS = {}
TESSDATA = "/usr/share/tesseract-ocr/5/tessdata"     # = eci_ocr.STD_TESSDATA
TESS_LANG = {"HIN": (), "ENG": ("eng",), "TAM": ("eng", "tam")}   # traineddata per item language besides hin (eng: numbers, EPIC)


def tools(lang="HIN"):
    if lang not in _TOOLS:
        _TOOLS[lang] = H.tool_versions(TESS_LANG.get(lang, ()))
    return _TOOLS[lang]


class MissingTool(Exception):
    """Tesseract's traineddata of the item's language is not on this runner (a job whose setup ran before scrape.json named it):
    the item goes back untried and the worker exits 2 (a new job runs the setup)."""


def ensure_tessdata(lang):
    need = [t for t in TESS_LANG.get(lang, ()) if not os.path.exists(os.path.join(TESSDATA, f"{t}.traineddata"))]
    if need:     # GitHub runners: passwordless sudo, as the setup step
        subprocess.run(["sudo", "-n", "apt-get", "install", "-y", "-qq"] + [f"tesseract-ocr-{t}" for t in need],
                       capture_output=True, timeout=900)
        need = [t for t in need if not os.path.exists(os.path.join(TESSDATA, f"{t}.traineddata"))]
    if need:
        raise MissingTool(f"Tesseract traineddata {need} missing in {TESSDATA}")
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
    """pdfs: [(roll id, path)] in roll order -> (records, per-run summaries, CPU seconds of the eci_ocr processes).
    engocr sets (lang ENG / TAM): + --lang, the set's extra arguments, --stage1 also without hints (stage1_nohint), and with
    draft_records the final roll's run gets the draft roll's output of this item as --draft-records."""
    # only the set's decision switches: an ECI_* variable from the runner or pool env must not change the run behind the
    # provenance (ECI_OCR_WORKERS and ECI_TESS_CLI change speed / call path only)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECI_") or k in ("ECI_OCR_WORKERS", "ECI_TESS_CLI")}
    env.update(H.set_env(ms))       # '<models>' paths (VALSET CRNNs, lexicons) -> this checkout's models/
    lang = H.set_lang(ms)
    recs, summs, prev, cpu, draft_out = [], [], None, 0.0, None
    for i, (roll, pdf) in enumerate(pdfs):
        o1 = os.path.join(tmp, f"out{i}.jsonl")
        cmd = [sys.executable, os.path.join(HERE, "eciocr", "prod", "eci_ocr.py"), pdf, "--out", o1, "--workers", str(WORKERS),
               "--ort-threads", "1", "--quiet", "--roll-year", str(year)]
        if hints_path:
            cmd += ["--hint-2027", hints_path, "--stage1", str(H.set_stage1(ms, True))]
        elif H.set_stage1(ms, False):
            cmd += ["--stage1", str(H.set_stage1(ms, False))]
        if lang != "HIN":
            cmd += ["--lang", lang] + H.set_args(ms)
            if ms.get("draft_records") and draft_out and not H.is_draft(roll):
                cmd += ["--draft-records", draft_out]
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
        if lang != "HIN" and ms.get("draft_records") and H.is_draft(roll):
            draft_out = o1      # read by the final roll's run (--draft-records); removed with tmp
        else:
            os.remove(o1)
    return recs, summs, cpu


def do_item(hub, item, settings=None, fetch=None, upload=None):
    """fetch(keys, dest) -> {key: path or None} (default: hub read links); upload(files) (default: hub.upload_many).
    Returns the hub record of the part."""
    if not H.CODE:
        raise RuntimeError("the family commit is unknown (git rev-parse failed); no output without its provenance")
    p = item["payload"]
    H.payload_version(p)        # raises UnknownPayloadVersion for a newer payload format
    state, ac, part = str(p["state"]), int(p["ac"]), int(p["part"])
    group = p.get("roll_group")        # form-route items (payload v 2): GE / SSR roll family, own output path
    lang = p.get("lang") or "HIN"      # engocr (payload v 3): ENG / TAM SIR rolls, output ocr/<state>/<lang>/<ac>
    if lang not in H.LANGS:
        raise RuntimeError(f"lang {lang} not supported (eci_ocr reads {', '.join(H.LANGS)})")
    base = H.ocr_base(state, ac, group, lang)
    keys = {r: k for r, k in (p.get("keys") or {}).items() if k}
    if not keys:
        raise RuntimeError("payload has no PDF keys")
    name, set_src = H.choose_set(p, settings if settings is not None else getattr(hub, "settings", {}))
    ms = H.MODEL_SETS[name]
    dirs = H.model_dirs(name)
    if lang != "HIN":
        ensure_tessdata(lang)
    want_hints = bool(p.get("hints")) and ms.get("hints") != "none"
    if ms.get("hints") == "required" and not want_hints:
        raise RuntimeError(f"model set {name} needs hints; the item has none")
    rolls = H.roll_order(keys)
    if group is not None and any(not r.startswith(state + "-") for r in rolls):
        raise RuntimeError(f"rolls {rolls} not of state {state}")
    years = {H.roll_year(r) for r in rolls}
    if len(years) != 1:
        raise RuntimeError(f"rolls of different years in one item: {rolls}")
    year = years.pop()
    fetch = fetch or (lambda ks, dest: fetch_hub(hub, ks, dest))
    upload = upload or hub.upload_many
    with tempfile.TemporaryDirectory() as tmp:
        got = fetch([keys[r] for r in rolls] + ([p["hints"]] if want_hints else []), tmp)
        hpath, nh, hsha = None, 0, None
        if want_hints:
            hz = got.get(p["hints"])
            if not hz:
                raise RuntimeError(f"no 2027 hints yet: {p['hints']}")
            hsha = H.sha256_file(hz)      # which version of the AC's hints this OCR used (ssrtext may run the AC again)
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
        if missing:     # every planned key is a stored PDF: a missing one is a read error, so the item is retried
            raise H.MissingRoll(f"PDF of {', '.join(missing)} not read from the bucket: " + ", ".join(keys[r] for r in missing))
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
    now = H.code_commit()
    if now != H.CODE:     # eci_ocr runs from the checkout for each PDF: the output may mix code; the item is done again
        raise H.CodeChanged(f"family checkout moved {H.CODE} -> {now or '?'} during the item")
    engines = ["crnn_" + m for m in ms["models"]]
    repairs = H.repair_part(recs, engines, rules=ms.get("env", {}))
    # serial checks again after the repairs (the hub record's pdfs[].serial_check; eci_ocr's own is serial_check_ocr)
    by_src = {}
    for r in recs:
        if not r.get("not_voter_box"):
            by_src.setdefault(r.get("source"), []).append(r)
    no_list = H.additions_without_list(recs)
    for s in summs:
        for d in s["pdfs"]:
            if "serial_check" in d:
                d["serial_check_ocr"] = d.pop("serial_check")
                d["serial_check"] = H.serial_summary(by_src.get(d["pdf"], []))
            if d["pdf"] in no_list:
                d["additions_without_list_section"] = no_list[d["pdf"]]
    prov = {"ocr_model_set": name, "ocr_models": list(ms["models"]), "ocr_code": H.CODE,
            "ocr_rules": H.rules_of(ms, year, hpath is not None), "hints_used": hpath is not None}
    extra = set(p.get("extra_rolls") or [])
    for r in recs:
        r.update(prov)
        if hsha:
            r["hints_sha256_16"] = hsha[:16]      # Parquet: in 'extra'
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
    summ = {"item": item["key"], "state": state, "ac": ac, "part": part, **({"roll_group": group} if group is not None else {}),
            **({"lang": lang} if lang != "HIN" else {}), **prov, "model_set_source": set_src,
            "model_sha256_16": ms.get("sha256_16"), "hints_key": p.get("hints") if want_hints else None, "hints_records": nh,
            "hints_sha256": hsha, "ocr_tools": tools(lang),
            "epic_fix2_refused": sum(1 for r in voters if "fix2_candidate" in (((r.get("readings") or {}).get("epic")) or {})),
            "rolls": rolls, "missing_rolls": missing, "records": len(voters), "not_voter_boxes": len(recs) - len(voters),
            "flagged_records": nfl, "flag_rate": round(nfl / max(1, len(voters)), 4), "by_roll": by_roll, "repairs": repairs,
            "additions_without_list_section": sum(no_list.values()),
            "voter_pages": vp, "wall_s": round(secs, 1), "cpu_s": round(cpu, 1), "cpu_s_per_voter_page": round(cpu / max(1, vp), 2),
            "workers": WORKERS, "runs": [{k: s.get(k) for k in ("roll", "wall_s", "cache_in", "cache_out", "voter_pages", "peak_worker_rss_mb")} for s in summs],
            "pdfs": [dict(d, roll=s["roll"]) for s in summs for d in s["pdfs"]]}
    body = gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode(), mtime=0)
    # summary first, records last (ocrv5 review: a second worker on the same item after a stall must not leave a summary that
    # names another set than the records)
    upload([(f"{base}/{part}.summary.json", json.dumps(summ, ensure_ascii=False).encode(), "application/json")])
    upload([(f"{base}/{part}.jsonl.gz", body, "application/gzip")])
    rec = {k: summ[k] for k in ("state", "ac", "part", "ocr_model_set", "ocr_models", "ocr_code", "ocr_rules", "hints_used",
                                "model_set_source", "hints_records", "hints_sha256", "ocr_tools", "epic_fix2_refused", "rolls",
                                "missing_rolls", "records", "not_voter_boxes",
                                "flag_rate", "by_roll", "repairs", "voter_pages", "wall_s", "cpu_s", "cpu_s_per_voter_page")}
    rec["additions_without_list_section"] = sum(no_list.values())
    if group is not None:
        rec["roll_group"] = group
    if lang != "HIN":
        rec["lang"] = lang
    rec["pdfs"] = [{k: d.get(k) for k in ("roll", "pdf", "kind", "records", "flag_rate", "hint_matched", "epic_from_2027", "serial_check",
                                          "serial_check_ocr", "post_rules", "additions_without_list_section", "worker_s_total")
                    + (("draft_fill", "epic_by_prefix") if lang != "HIN" else ())}
                   for d in summ["pdfs"]]
    return rec


def give_back(hub, it, e):
    """Item back as 'pending' (attempt not counted) with the metric of the reason."""
    if isinstance(e, H.ModelSetIncomplete):
        err, metric = f"model set {e} incomplete in this checkout ({H.CODE})", "model_set_incomplete"
    elif isinstance(e, H.UnknownModelSet):
        err, metric = f"model set {e} not in this checkout ({H.CODE})", "model_set_unknown"
    elif isinstance(e, H.UnknownPayloadVersion):
        err, metric = f"{e} ({H.CODE})", "payload_version_unknown"
    else:
        err, metric = str(e), "code_changed"
    hub.complete([it["key"]], "pending", error=err[:400])
    hub.metric(metric)


def main():
    stop = stop_flag()
    hub = HubClient(PROJECT, job_index=int(os.environ.get("SCRAPE_JOB_INDEX", 0)), total_jobs=int(os.environ.get("SCRAPE_TOTAL_JOBS", 1)))
    if not H.CODE:      # every record names its code commit: without it the worker takes no item
        hub.metric("no_code_commit")
        hub.close("failed", "family commit unknown: git rev-parse failed in the checkout")
        sys.exit(2)
    code = 0
    try:
        while time.time() < DEADLINE and not stop.is_set():
            if hub.settings.get("paused"):
                time.sleep(30)
                continue
            unknown = H.hub_override_unknown(hub.settings)
            if unknown:      # the hub names a newer model set: leave the items to workers with newer code
                hub.metric("model_set_unknown")
                code = H.EXIT_NEEDS_NEWER_CODE
                break
            items = hub.claim(1, steal=True, retry_failed=True)
            if not items:
                break
            it = items[0]
            try:
                rec = do_item(hub, it, settings=hub.settings)
                hub.record(it["key"], rec)
                if rec["additions_without_list_section"]:
                    hub.metric("additions_without_list_section", rec["additions_without_list_section"])
                hub.metric("voter_pages", rec["voter_pages"])
                hub.metric("ocr_cpu_s", rec["cpu_s"])
                hub.metric("set_" + re.sub(r"[^a-z0-9_-]", "_", rec["ocr_model_set"].lower()))   # metric names: a-z 0-9 - _
                hub.complete([it["key"]], "done", fetched=rec["records"])
            except MissingTool as e:     # this runner lacks the language's traineddata: a later job (new setup) takes it
                hub.complete([it["key"]], "pending", error=f"{e} ({H.CODE})"[:400])
                hub.metric("tool_missing")
                code = 2
                break
            except H.NeedsNewerCode as e:
                # 'pending' = given back untried: the hub does not count the attempt (scrape-hub app.py complete).
                # UnknownModelSet, ModelSetIncomplete (model files missing / wrong sha256), UnknownPayloadVersion, CodeChanged
                give_back(hub, it, e)
                code = H.EXIT_NEEDS_NEWER_CODE
                break
            except Exception as e:  # one bad part must not stop the worker
                now = H.code_commit()
                if now != H.CODE:   # it failed while a sibling's refetch replaced the files (e.g. an ImportError): not the item's fault
                    give_back(hub, it, H.CodeChanged(f"family checkout moved {H.CODE} -> {now or '?'} during the item "
                                                     f"({type(e).__name__}: {e})"[:300]))
                    code = H.EXIT_NEEDS_NEWER_CODE
                    break
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
