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
    model dirs under models/ (c11q/c9q/c4q/c3q) and Tesseract 5 with hin+eng data at
    /usr/share/tesseract-ocr/5/tessdata (STD_TESSDATA in eci_ocr).
  - The portal's part list is the AC's CURRENT (2027) list and UNDER-COUNTS 2026 (~4k deleted/merged
    parts live above the listed max). ingest walks max+1.. until EXTRA_MISS consecutive 404s
    (fetch_task.py pattern) to recover them.
  - OCR records expose score.OUT_FIELDS incl. 'epic'/'age'/'gender'/'house_no' (same keys uk/detect.py
    reads), so the within-draft radars and the Draft->Final EPIC diff key off them.
  - 2027 text-layer hints are OPTIONAL confirmation for OCR; absent them wall_s is an UPPER BOUND.

PII (non-negotiable - public-repo Actions logs/artifacts are PUBLIC):
  raw PDFs live in memory / an unlinked temp file and are discarded. In ingest the per-voter OCR
  JSONLs of one AC are kept ONLY in a runner /tmp dir until end-of-AC (so detect + the Draft->Final
  diff can run on the runner), then the whole dir is unlinked. Every mode prints / writes ONLY
  aggregate counts - zero names / EPICs / addresses / serials.

Modes:
  probe   - P1 proof (APPROVED, merged, dispatched): S28 AC 1 -> timed Draft-Part-1 fetch +
            1-part CPU-OCR benchmark Draft(--cache-out)->Final(--cache-in); aggregates only.
  ingest  - P2: per-AC skip-if-exists resume + fetch with SLOW_BPS fast-bail + the EXTRA_MISS 404
            walk + OCR draft->final (JSONLs kept to end-of-AC) -> detect radars + Draft->Final EPIC
            diff -> PII-safe aggregate JSON. The headline mass-deletion signal is diff_deleted_epics.

Run:  python -u uk/uk_2026.py probe        # P1
      python -u uk/uk_2026.py ingest --acs 1-70 --out ./out   # P2 (after AG1 sign-off + scale wf)
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
from collections import Counter

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
SLOW_CPU_SPP = float(os.environ.get("ECI_SLOW_CPU_SPP", "7.5"))  # bail >7.5 s/page on Part 1
# extra-parts 404-walk (fetch_task.EXTRA_MISS): 2026 rolls have ~4k parts above the current list.
EXTRA_MISS = int(os.environ.get("ECI_EXTRA_MISS", "5"))
EXTRA_CAP = int(os.environ.get("ECI_EXTRA_CAP", "3000"))   # hard bound, like fetch_task (top0 + 3000)
# keep the probe comfortably under the job's timeout-minutes even if BOTH legs tarpit.
PROBE_DEADLINE_S = int(os.environ.get("ECI_PROBE_DEADLINE", "1080"))  # 18 min; < 25-min job cap
OCR_TIMEOUT_S = int(os.environ.get("ECI_OCR_TIMEOUT", "900"))        # 1-part OCR ceiling

PLACEHOLDER_HOUSE = {"", "0", "00", "000", "-", ".", "nan", "none", "null", "na"}  # uk/detect.py

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


def fetch_pdf(roll_key, ac, part, speeds=None, attempts=ATTEMPTS, timeout=TIMEOUT):
    """One part PDF from its public path, validated (%PDF.. / %%EOF / Content-Length).
    Returns (bytes_or_None, status) with status in {'ok','404','fail'} so the extra-parts walk can
    tell a true 404 (part absent) from a tarpit give-up. Appends bytes/sec to `speeds` on success."""
    url = E.roll_pdf_url(ROLLS[roll_key], STATE, ac, part)
    for attempt in range(attempts):
        t = time.time()
        try:
            r = session().get(url, timeout=timeout)
            if r.status_code == 404:
                return None, "404"
            if r.status_code == 200 and E.is_pdf(r.content):
                cl = r.headers.get("Content-Length")
                if not cl or int(cl) == len(r.content):
                    if speeds is not None:
                        speeds.append(len(r.content) / max(0.01, time.time() - t))
                    return r.content, "ok"
        except requests.RequestException:
            pass
        time.sleep(6 * (attempt + 1))
    return None, "fail"


def get_pdf(roll_key, ac, part, speeds=None, attempts=ATTEMPTS, timeout=TIMEOUT):
    """Bytes-or-None convenience wrapper (used by the probe)."""
    return fetch_pdf(roll_key, ac, part, speeds, attempts, timeout)[0]


class SlowRunner(Exception):
    """Raised when this runner's subnet is tar-pitting (median speed < SLOW_BPS)."""


def _check_slow(speeds):
    if len(speeds) >= SLOW_AFTER and statistics.median(speeds[-SLOW_AFTER:]) < SLOW_BPS:
        raise SlowRunner(f"median {statistics.median(speeds[-SLOW_AFTER:]) / 1e3:.0f} kB/s over "
                         f"last {SLOW_AFTER} PDFs < {SLOW_BPS / 1e3:.0f} kB/s")


def run_ocr(pdf_bytes, work, tag, roll_year=2026, cache_in=None, cache_out=None, hints=None,
            keep_jsonl=False):
    """Run eci_ocr.py on one PDF inside caller-owned `work`; return its aggregate summary
    (records / voter_pages / wall_s / flag_rate). The raw PDF is always unlinked here. With
    keep_jsonl the per-voter JSONL is KEPT (path returned as 'jsonl') for an end-of-AC detect/diff
    and the caller MUST unlink it; without it the JSONL is unlinked immediately (P1 posture)."""
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
    res = {k: summ.get(k) for k in ("records", "voter_pages", "wall_s", "flag_rate")}
    try:
        os.unlink(pdf)                    # raw PDF always discarded
    except OSError:
        pass
    try:
        os.unlink(out + ".summary.json")  # counts already captured
    except OSError:
        pass
    if keep_jsonl:
        res["jsonl"] = out                # PII: kept in /tmp until end-of-AC, unlinked by the caller
    else:
        try:
            os.unlink(out)
        except OSError:
            pass
    return res


def _per(d):
    vp = d.get("voter_pages") or 0
    return round((d.get("wall_s") or 0) / vp, 2) if vp else None


# ---------------- probe (P1 - APPROVED, unchanged) ----------------
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


# ---------------- ingest (P2) ----------------
def _to_int(x):
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        return None


def _epics(jsonls):
    """Set of normalised, non-empty EPICs across the given per-part JSONLs. Values are NEVER
    returned to the caller as data - only this set, used for |set| counts."""
    s = set()
    for path in jsonls:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                e = str(r.get("epic") or "").strip().upper()
                if e and e.lower() not in ("none", "null", "nan"):
                    s.add(e)
    return s


def _detect_counts(jsonls, thresh=25):
    """Within-draft discrepancy radars - a stdlib reimplementation of uk/detect.py's definitions
    (so the OCR runner needs no pandas/pyarrow). Returns COUNTS only, no row values."""
    epic_counts, house = Counter(), Counter()
    n = age_imp = age_cent = m = f = 0
    for path in jsonls:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                n += 1
                e = str(r.get("epic") or "").strip().upper()
                if e and e.lower() not in ("none", "null", "nan"):
                    epic_counts[e] += 1
                a = _to_int(r.get("age"))
                if a is not None:
                    if a < 18 or a > 120:
                        age_imp += 1
                    if a >= 100:
                        age_cent += 1
                g = str(r.get("gender") or "").strip()
                gu = g.upper()
                if "पुरुष" in g or gu in ("M", "MALE"):
                    m += 1
                elif "महिला" in g or gu in ("F", "FEMALE"):
                    f += 1
                h = str(r.get("house_no") or "").strip()
                if h.lower() not in PLACEHOLDER_HOUSE:
                    house[h] += 1
    return {
        "records": n,
        "dup_epic_within_ac": sum(1 for c in epic_counts.values() if c > 1),
        "age_impossible": age_imp,
        "age_centenarian": age_cent,
        "gender_m": m,
        "gender_f": f,
        "females_per_1000_males": round(1000 * f / max(1, m)),
        "congestion_houses_gt_thresh": sum(1 for c in house.values() if c > thresh),  # [Plausible]
        "congestion_max": max(house.values()) if house else 0,                        # [Plausible]
        "congestion_confidence": "Plausible (house_no noisy; validate before quoting)",
    }


def _process_part(ac, part, work, speeds, totals, draft_jsonls, final_jsonls,
                  paired_draft_jsonls, paired_final_jsonls, args, extra=False):
    """Fetch + OCR one part's Draft (and Final, same part number).
    Returns 'ok' (draft found & OCR'd), '404' (draft absent) or 'fail' (tarpit give-up)."""
    d, st = fetch_pdf("2026-sir-draft", ac, part, speeds)
    _check_slow(speeds)
    if st == "fail":
        totals["draft_failed_parts"] += 1
        return "fail"
    if st != "ok":
        return st
    cache = os.path.join(work, f"cache.{part}.pkl")
    dd = run_ocr(d, work, f"draft.{part}", 2026, cache_out=cache, hints=args.hints, keep_jsonl=True)
    # CPU fast-bail on first part of AC so slow-CPU runners (e.g. >7.5 s/page) hand back in ~2 min
    if totals["parts"] == 0 and _per(dd) and _per(dd) > SLOW_CPU_SPP:
        raise SlowRunner(f"slow CPU {_per(dd)} s/page > {SLOW_CPU_SPP} s/page on part {part}")
    draft_jsonls.append(dd["jsonl"])
    totals["draft_records"] += dd["records"] or 0
    totals["draft_voter_pages"] += dd["voter_pages"] or 0

    fb, fst = fetch_pdf("2026-sir-final", ac, part, speeds)
    _check_slow(speeds)
    if fst == "ok":
        ff = run_ocr(fb, work, f"final.{part}", 2026, cache_in=cache, hints=args.hints, keep_jsonl=True)
        final_jsonls.append(ff["jsonl"])
        totals["final_records"] += ff["records"] or 0
        totals["final_voter_pages"] += ff["voter_pages"] or 0
        paired_draft_jsonls.append(dd["jsonl"])
        paired_final_jsonls.append(ff["jsonl"])
    elif fst == "404":
        # Legitimate booth deletion/merge in Final -> keep Draft in paired diff
        paired_draft_jsonls.append(dd["jsonl"])
    else:
        # Network tarpit on Final -> do NOT count Draft voters as deleted
        totals["final_failed_parts"] += 1
    try:
        os.unlink(cache)
    except OSError:
        pass
    totals["parts"] += 1
    totals["extra_parts"] += 1 if extra else 0
    return "ok"


def _ingest_ac(ac, args):
    """One AC end-to-end: listed Draft+Final parts + EXTRA_MISS 404-walk, OCR draft->final
    (JSONLs kept to end-of-AC), then detect radars (draft+final) + paired Draft->Final EPIC diff,
    then unlink all PII. Returns the PII-safe aggregate. May raise SlowRunner (fast-bail)."""
    work = tempfile.mkdtemp(prefix=f"uk2026-ac{ac}-")
    speeds, draft_jsonls, final_jsonls = [], [], []
    paired_draft_jsonls, paired_final_jsonls = [], []
    totals = {"ac": ac, "parts": 0, "extra_parts": 0, "final_only_parts": 0,
              "draft_failed_parts": 0, "final_failed_parts": 0, "fetched": E.now(),
              "draft_records": 0, "draft_voter_pages": 0,
              "final_records": 0, "final_voter_pages": 0}
    try:
        listed = sorted(part_list("2026-sir-draft", ac), key=lambda p: int(p["partNumber"]))
        part_nums = [int(p["partNumber"]) for p in listed]
        try:
            final_listed = {int(p["partNumber"]) for p in part_list("2026-sir-final", ac)}
        except RuntimeError:
            final_listed = set()
        seen_parts = set(part_nums)

        for part in part_nums:
            if _process_part(ac, part, work, speeds, totals, draft_jsonls, final_jsonls,
                             paired_draft_jsonls, paired_final_jsonls, args) == "fail":
                print(f"    AC {ac} part {part}: draft fetch gave up (tarpit); skipped", flush=True)

        # EXTRA_MISS 404-walk above the listed max (2026 deleted/merged parts; fetch_task.py)
        top0, miss, n = max(part_nums), 0, max(part_nums)
        while miss < EXTRA_MISS and n < top0 + EXTRA_CAP:
            n += 1
            seen_parts.add(n)
            st = _process_part(ac, n, work, speeds, totals, draft_jsonls, final_jsonls,
                               paired_draft_jsonls, paired_final_jsonls, args, extra=True)
            if st == "404":
                miss += 1
            elif st == "ok":
                miss = 0
            else:
                print(f"    AC {ac}: extra-walk halted at {n} (tarpit, not 404)", flush=True)
                break

        # Final-only parts (new booths listed in 2026-sir-final absent from Draft)
        for part in sorted(final_listed - seen_parts):
            fb, fst = fetch_pdf("2026-sir-final", ac, part, speeds)
            _check_slow(speeds)
            if fst == "ok":
                ff = run_ocr(fb, work, f"final.{part}", 2026, hints=args.hints, keep_jsonl=True)
                final_jsonls.append(ff["jsonl"])
                paired_final_jsonls.append(ff["jsonl"])
                totals["final_records"] += ff["records"] or 0
                totals["final_voter_pages"] += ff["voter_pages"] or 0
                totals["final_only_parts"] += 1
            elif fst == "fail":
                totals["final_failed_parts"] += 1

        # End-of-AC: paired Draft->Final diff + detect_draft & detect_final (counts only)
        draft_epics, final_epics = _epics(paired_draft_jsonls), _epics(paired_final_jsonls)
        deleted = len(draft_epics - final_epics)
        incomplete = (totals["draft_failed_parts"] + totals["final_failed_parts"]) > 0
        agg = dict(totals,
                   incomplete=incomplete,
                   draft_distinct_epics=len(draft_epics),
                   final_distinct_epics=len(final_epics),
                   diff_deleted_epics=deleted,
                   diff_added_epics=len(final_epics - draft_epics),
                   diff_retained_epics=len(draft_epics & final_epics),
                   diff_net_deletion_pct=round(-100 * deleted / max(1, len(draft_epics)), 2),
                   detect_draft=_detect_counts(draft_jsonls),
                   detect_final=_detect_counts(final_jsonls),
                   confidence="Plausible - OCR EPICs (no hints); validate Draft->Final diff before quoting")
        return agg
    finally:
        shutil.rmtree(work, ignore_errors=True)

def cmd_ingest(args):
    """P2 per-AC pipeline (dispatch only after AG1 sign-off). skip-if-exists resume mirrors
    uk_driver.cmd_textlayer; emits PII-safe aggregate counts only."""
    acs = parse_acs(args.acs)
    agg_dir = os.path.join(args.out, "aggregates", "2026")
    os.makedirs(agg_dir, exist_ok=True)
    for ac in acs:
        agg_path = os.path.join(agg_dir, f"ac={ac}.json")
        if os.path.exists(agg_path) and os.path.getsize(agg_path) > 0:
            print(f"  AC {ac}: aggregate present, skip (resume-safe)", flush=True)
            continue
        try:
            agg = _ingest_ac(ac, args)
        except SlowRunner as e:
            print(f"  AC {ac}: slow runner ({e}) -> handed back (fast-bail; matrix reshuffle)",
                  flush=True)
            continue
        with open(agg_path, "w") as f:
            json.dump(agg, f, ensure_ascii=False)  # counts only - safe to keep / upload
        print(f"  AC {ac}: {agg['parts']} parts ({agg['extra_parts']} extra) | "
              f"draft {agg['draft_distinct_epics']} epics, final {agg['final_distinct_epics']} | "
              f"DELETED {agg['diff_deleted_epics']} ({agg['diff_net_deletion_pct']}%)", flush=True)
    print("[ingest] PII-safe aggregates written. All raw PDFs/records/JSONLs discarded.", flush=True)


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
