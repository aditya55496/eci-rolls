#!/usr/bin/env python3
"""Uttarakhand (S28) 2026 SIR ingestion - additive, hub-free (CC1 / Window 1, 2026 strike team).

Owns 2026 SIR Draft + Final end-to-end on GitHub Actions (unattended, Rs.0). It NEVER touches 2027:
it does not import, read or edit uk_driver.py's textlayer path or uk_rolls_runner.ipynb.

Reuses the repo's own eci_api.py (ECI's encrypted part-list request + public PDF paths) and runs
eciocr/prod/eci_ocr.py (CRNN-int8 ensemble + Tesseract 5, no LLM) as a subprocess for the scanned
2026 rolls - exactly the invocation ocr_task.py benchmarked (44.2 s draft, 10.0 s final with cache).

Assumptions (karpathy: stated, not hidden):
  - S28 2026 SIR Draft/Final live at the same public EROLLGEN paths eci_api.roll_pdf_url builds
    (S24 verified live 2026-11-22; S28 2027 verified 2026-11-26). Captcha-free.
  - 2026 rolls are SCANNED -> OCR required (unlike 2027's text layer). eci_ocr.py needs the CRNN
    model dirs under models/ (c11q/c9q/c4q/c3q, ~16 MB each, committed as plain blobs) and
    Tesseract 5 with hin+eng data at /usr/share/tesseract-ocr/5/tessdata (STD_TESSDATA in eci_ocr).
  - 2027 text-layer hints are OPTIONAL confirmation for OCR. Absent them, eci_ocr runs the full
    ensemble on every box -> wall_s is an UPPER BOUND vs. production (which skips with hints).

PII (non-negotiable - public-repo Actions logs/artifacts are PUBLIC):
  raw PDFs live in memory / an unlinked temp file and are discarded; the per-voter OCR JSONL and the
  reading cache stay in a temp dir deleted here. Every mode prints ONLY aggregate counts - zero
  names / EPICs / addresses / serials.

Modes:
  probe   - P1 proof (write for AG1 review; run only after sign-off): S28 AC 1 ->
            (1) part list for 2026-sir-draft (gateway host, always works),
            (2) timed fetch of Draft Part 1 with retry+backoff (file host; tarpits some subnets) ->
                logs IP + elapsed + RESULT=OK/TARPIT,
            (3) a 1-part CPU-OCR benchmark Draft(--cache-out) -> Final(--cache-in), printing ONLY
                records / voter_pages / wall_s / s_per_page / flag_rate. OCR is skipped if the PDF
                tarpits (fetch result still reported); a wall-clock deadline bounds the job.
  ingest  - P2 skeleton (DO NOT run in P1): per-AC skip-if-exists resume + fetch with the SLOW_BPS
            fast-bail from fetch_task.py + OCR draft->final; writes PII-safe aggregate JSON only.

Run (after AG1 sign-off):  python -u uk/uk_2026.py probe
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

# --- make the repo's own modules importable (this file lives in uk/, repo root is one up) ---
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import eci_api as E  # noqa: E402  (roll_part_list_request, roll_pdf_url, is_pdf, HEADERS)
import requests      # noqa: E402

STATE = "S28"
ROLLS = {  # the two scanned EROLLGEN rolls we own; shapes match eci_api.EROLL / uk_driver.UK_ROLLS
    "2026-sir-draft": {"year": 2026, "ref": "SIR-DraftRoll", "gen": "EROLLGEN", "rev": 1},
    "2026-sir-final": {"year": 2026, "ref": "SIR-FinalRoll", "gen": "EROLLGEN", "rev": 1},
}
OCR = os.path.join(REPO, "eciocr", "prod", "eci_ocr.py")
MODELS = [os.path.join(REPO, "models", m) for m in ("c11q", "c9q", "c4q", "c3q")]

# fetch tuning (mirrors uk_driver.get_pdf + fetch_task.py)
ATTEMPTS, TIMEOUT = 4, 180
# fast-bail (fetch_task.SLOW_BPS/SLOW_AFTER): a runner whose median speed over the last SLOW_AFTER
# PDFs is below SLOW_BPS is on a tar-pitting subnet; it hands the AC back so a matrix reshuffle /
# later run draws a faster subnet instead of burning the whole budget in place.
SLOW_BPS = float(os.environ.get("ECI_SLOW_BPS", "300000"))
SLOW_AFTER = 8
# keep the probe comfortably under the job's timeout-minutes even if BOTH legs tarpit:
# draft fetch gets the full 4x180 budget; the final leg is only started while under this deadline
# and with a tighter per-request budget (it exists only to show the cache speedup).
PROBE_DEADLINE_S = int(os.environ.get("ECI_PROBE_DEADLINE", "1080"))  # 18 min; < 25-min job cap
OCR_TIMEOUT_S = int(os.environ.get("ECI_OCR_TIMEOUT", "900"))         # 1-part OCR ceiling

_S = None


def session():
    global _S
    if _S is None:
        _S = requests.Session()
        _S.headers.update(E.HEADERS)
    return _S


def my_ip():
    try:
        return session().get("https://api.ipify.org", timeout=10).text
    except requests.RequestException:
        return "?"


def parse_acs(spec, lo=1, hi=70):
    out = set()
    for tok in (spec or f"{lo}-{hi}").split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); out.update(range(int(a), int(b) + 1))
        elif tok:
            out.add(int(tok))
    return sorted(x for x in out if lo <= x <= hi)


def part_list(roll_key, ac):
    """Parts array for a roll+AC via the gateway API (encrypted body; plaintext payload back)."""
    url, body = E.roll_part_list_request(ROLLS[roll_key], STATE, ac)
    for attempt in range(5):
        try:
            r = session().post(url, json=body, timeout=60)
            if r.status_code == 200:
                d = r.json()
                if d.get("payload") is not None:
                    return d["payload"]
        except requests.RequestException:
            pass
        time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"part list failed: {roll_key} AC {ac}")


def get_pdf(roll_key, ac, part, speeds=None, attempts=ATTEMPTS, timeout=TIMEOUT):
    """One part PDF from its public path, validated (%PDF.. / %%EOF / Content-Length).
    Appends bytes/sec to `speeds` on success (for the fast-bail). Returns bytes or None."""
    url = E.roll_pdf_url(ROLLS[roll_key], STATE, ac, part)
    for attempt in range(attempts):
        t = time.time()
        try:
            r = session().get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and E.is_pdf(r.content):
                cl = r.headers.get("Content-Length")
                if not cl or int(cl) == len(r.content):
                    if speeds is not None:
                        speeds.append(len(r.content) / max(0.01, time.time() - t))
                    return r.content
        except requests.RequestException:
            pass
        time.sleep(6 * (attempt + 1))
    return None


class SlowRunner(Exception):
    """Raised when this runner's subnet is tar-pitting (median speed < SLOW_BPS)."""


def run_ocr(pdf_bytes, work, tag, roll_year=2026, cache_in=None, cache_out=None, hints=None):
    """Run eci_ocr.py on one PDF inside caller-owned `work`; return ONLY its aggregate summary
    (records / voter_pages / wall_s / flag_rate). The PDF and the per-voter JSONL are unlinked here
    so no PII survives. Invocation matches ocr_task.py (cascade stage1=1 over the 4 int8 CRNNs)."""
    pdf = os.path.join(work, f"{tag}.pdf")
    with open(pdf, "wb") as f:
        f.write(pdf_bytes)
    out = os.path.join(work, f"{tag}.jsonl")
    cmd = [sys.executable, OCR, pdf, "--out", out, "--stage1", "1",
           "--ort-threads", "1", "--quiet", "--roll-year", str(roll_year)]
    for m in MODELS:
        cmd += ["--crnn", m]
    if hints:
        cmd += ["--hint-2027", hints]
    if cache_in:
        cmd += ["--cache-in", cache_in]
    if cache_out:
        cmd += ["--cache-out", cache_out]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=OCR_TIMEOUT_S)
    if r.returncode:
        tail = "\n".join(line for line in r.stdout.splitlines()[-8:])
        raise RuntimeError(f"eci_ocr {tag} exit {r.returncode}:\n{tail}")
    summ = json.load(open(out + ".summary.json"))
    for p in (out, out + ".summary.json", pdf):  # discard per-voter output immediately (PII)
        try:
            os.unlink(p)
        except OSError:
            pass
    return {k: summ.get(k) for k in ("records", "voter_pages", "wall_s", "flag_rate")}


def _per(d):
    vp = d.get("voter_pages") or 0
    return round((d.get("wall_s") or 0) / vp, 2) if vp else None


# ---------------- modes ----------------
def cmd_probe(args):
    ac, start = 1, time.time()
    ip = my_ip()
    hints = args.hints if args.hints and os.path.exists(args.hints) else None
    print(f"[probe] S28 AC {ac} ip={ip} hints={'yes' if hints else 'none (wall_s is an upper bound)'}",
          flush=True)

    parts = part_list("2026-sir-draft", ac)
    p1 = sorted(parts, key=lambda p: int(p["partNumber"]))[0]
    part1 = int(p1["partNumber"])
    print(f"[probe] 2026-sir-draft AC {ac}: {len(parts)} parts listed; benchmarking part {part1}",
          flush=True)

    # (2) timed fetch of Draft Part 1 - full retry budget (this is the P1 fetch proof)
    t = time.time()
    draft = get_pdf("2026-sir-draft", ac, part1)
    dt = time.time() - t
    print(f"[probe] FETCH draft part {part1}: {len(draft) if draft else 0}B {dt:.1f}s "
          f"RESULT={'OK' if draft else 'TARPIT'} ip={ip}", flush=True)
    if not draft:
        print("[probe] draft tarpitted on this subnet; OCR benchmark skipped "
              "(the matrix gives other shards a cleaner subnet). done.", flush=True)
        return

    # (3) 1-part CPU-OCR benchmark: Draft (cache-out) -> Final (cache-in)
    work = tempfile.mkdtemp(prefix="uk2026-")
    try:
        cache = os.path.join(work, "draft.cache.pkl")
        d = run_ocr(draft, work, "draft", 2026, cache_out=cache, hints=hints)
        print(f"[probe] OCR draft part {part1}: records={d['records']} voter_pages={d['voter_pages']} "
              f"wall_s={d['wall_s']} s_per_page={_per(d)} flag_rate={d['flag_rate']}", flush=True)

        if time.time() - start > PROBE_DEADLINE_S:
            print("[probe] deadline reached after draft; skipping final leg (draft-only). done.",
                  flush=True)
            return
        # final leg: tighter budget (attempts=2, timeout=120) so a tarpit here can't blow the job cap
        tf = time.time()
        final = get_pdf("2026-sir-final", ac, part1, attempts=2, timeout=120)
        ft = time.time() - tf
        if not final:
            print(f"[probe] final part {part1} fetch RESULT=TARPIT {ft:.1f}s; draft-only reported. done.",
                  flush=True)
            return
        f = run_ocr(final, work, "final", 2026, cache_in=cache, hints=hints)
        print(f"[probe] OCR final part {part1} (cache-in): records={f['records']} "
              f"voter_pages={f['voter_pages']} wall_s={f['wall_s']} s_per_page={_per(f)} "
              f"flag_rate={f['flag_rate']}", flush=True)
        combined = round((d['wall_s'] or 0) + (f['wall_s'] or 0), 1)
        print(f"[probe] COMBINED draft+final wall_s={combined} | est per-AC at {len(parts)} parts "
              f"~{round(combined * len(parts) / 3600, 2)} h (Option A fits the 6-h job cap if < ~6).",
              flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("[probe] done. All raw PDFs/records/cache discarded; only aggregates printed.", flush=True)


def _ingest_ac(ac, args):
    """P2 skeleton: fetch every part (draft+final) with fast-bail, OCR each, return counts only."""
    draft_parts = sorted(part_list("2026-sir-draft", ac), key=lambda p: int(p["partNumber"]))
    speeds, totals = [], {"ac": ac, "parts": 0, "draft_records": 0, "draft_voter_pages": 0,
                          "final_records": 0, "final_voter_pages": 0, "fetched": E.now()}
    for p in draft_parts:
        part = int(p["partNumber"])
        d = get_pdf("2026-sir-draft", ac, part, speeds)
        if len(speeds) >= SLOW_AFTER and statistics.median(speeds[-SLOW_AFTER:]) < SLOW_BPS:
            raise SlowRunner(f"median {statistics.median(speeds[-SLOW_AFTER:]) / 1e3:.0f} kB/s over "
                             f"last {SLOW_AFTER} parts < {SLOW_BPS / 1e3:.0f} kB/s")
        if not d:
            continue
        f_pdf = get_pdf("2026-sir-final", ac, part, speeds)
        work = tempfile.mkdtemp(prefix="uk2026-")
        try:
            cache = os.path.join(work, "d.cache.pkl")
            dd = run_ocr(d, work, "d", 2026, cache_out=cache, hints=args.hints)
            totals["draft_records"] += dd["records"] or 0
            totals["draft_voter_pages"] += dd["voter_pages"] or 0
            if f_pdf:
                ff = run_ocr(f_pdf, work, "f", 2026, cache_in=cache, hints=args.hints)
                totals["final_records"] += ff["records"] or 0
                totals["final_voter_pages"] += ff["voter_pages"] or 0
        finally:
            shutil.rmtree(work, ignore_errors=True)
        totals["parts"] += 1
    return totals


def cmd_ingest(args):
    """P2 (DO NOT run in P1). Per-AC skip-if-exists resume (mirrors uk_driver.cmd_textlayer);
    emits PII-safe aggregate counts only. The Draft->Final DIFF is P3, not here."""
    acs = parse_acs(args.acs)
    agg_dir = os.path.join(args.out, "aggregates", "2026")
    os.makedirs(agg_dir, exist_ok=True)
    for ac in acs:
        agg_path = os.path.join(agg_dir, f"ac={ac}.json")
        if os.path.exists(agg_path) and os.path.getsize(agg_path) > 0:
            print(f"  AC {ac}: aggregate present, skip (resume-safe)", flush=True)
            continue
        try:
            totals = _ingest_ac(ac, args)
        except SlowRunner as e:
            print(f"  AC {ac}: slow runner ({e}) -> handed back (fast-bail; matrix reshuffle)",
                  flush=True)
            continue
        with open(agg_path, "w") as f:
            json.dump(totals, f)  # counts only - safe to keep / upload
        print(f"  AC {ac}: {totals['parts']} parts -> aggregate written "
              f"(draft {totals['draft_records']} rec / final {totals['final_records']} rec)",
              flush=True)
    print("[ingest] PII-safe aggregates written. Raw PDFs/records discarded.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["probe", "ingest"])
    ap.add_argument("--out", default="./out")
    ap.add_argument("--acs", default="1-70", help="e.g. 1-70 or 1,21,19 (ingest)")
    ap.add_argument("--hints", default=os.environ.get("ECI_HINTS"),
                    help="optional 2027 text-layer hints JSONL (confirmation only; prod uses it to skip)")
    args = ap.parse_args()
    {"probe": cmd_probe, "ingest": cmd_ingest}[args.mode](args)


if __name__ == "__main__":
    main()
