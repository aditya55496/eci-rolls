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
sys.path.insert(0, os.path.join(REPO, "eciocr", "textlayer"))  # parse_textlayer (2027 hint pull)
import eci_api as E  # noqa: E402  (roll_part_list_request, roll_pdf_url, is_pdf, HEADERS)
import requests      # noqa: E402
import hashlib       # noqa: E402  (EPIC/house HMAC for the PII-safe public release)
import hmac          # noqa: E402

STATE = "S28"
ROLLS = {  # scanned EROLLGEN rolls we own + the 2027 text-layer roll used ONLY to pull OCR hints
    "2026-sir-draft": {"year": 2026, "ref": "SIR-DraftRoll", "gen": "EROLLGEN", "rev": 1},
    "2026-sir-final": {"year": 2026, "ref": "SIR-FinalRoll", "gen": "EROLLGEN", "rev": 1},
    "2027-ssr-draft": {"year": 2027, "ref": "SSR-DraftRoll", "gen": "EROLLGEN", "rev": 1},  # hints only
}
OCR = os.path.join(REPO, "eciocr", "prod", "eci_ocr.py")
MODEL_SETS = os.path.join(REPO, "model_sets.json")
MODELS = [os.path.join(REPO, "models", m) for m in ("c11q", "c9q", "c4q", "c3q")]  # legacy default

# PII-safe public release (--public-out): STRICT allowlist. Only these columns are ever written;
# everything else (name, relation_name, house_no, epic, serial_no, page, section, section_name,
# readings, bbox, extra, code_ink, ...) is dropped by construction. EPIC/house are keyed one-way HMACs.
PUBLIC_FIELDS = ("epic_hmac", "house_hmac", "house_is_placeholder", "state", "roll", "revision_year",
                 "ac", "part", "age", "gender", "relation_type", "status_code", "flags",
                 "deleted_in_final", "from_draft", "ocr_model_set")

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


def load_model_set(name):
    """Resolve a model_sets.json entry -> (crnn_dirs, env, stage1). Loading BOTH `models` (as
    models/<name> dirs for --crnn, in voting order) AND `env` (the exact ECI_* switch block) so the
    provenance is reproducible. Returns (None, {}, 1) for name in (None, '') -> legacy MODELS path."""
    if not name:
        return None, {}, 1
    sets = json.load(open(MODEL_SETS, encoding="utf-8"))
    if name not in sets:
        raise SystemExit(f"model set '{name}' not in {MODEL_SETS} (have: {[k for k in sets if not k.startswith('_')]})")
    s = sets[name]
    dirs = [os.path.join(REPO, "models", m) for m in s["models"]]
    env = {k: str(v) for k, v in (s.get("env") or {}).items()}
    return dirs, env, int(s.get("stage1", 1))


def _hmac(salt_bytes, value):
    """Keyed one-way digest: HMAC-SHA256(salt, normalised value) -> hex. Raw value never stored."""
    v = str(value or "").strip().upper()
    if not v:
        return None
    return hmac.new(salt_bytes, v.encode("utf-8"), hashlib.sha256).hexdigest()


def _norm_epic(e):
    e = str(e or "").strip().upper()
    return e if e and e.lower() not in ("none", "null", "nan") else ""


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
            keep_jsonl=False, models=None, env=None, stage1=1):
    """Run eci_ocr.py on one PDF inside caller-owned `work`; return its aggregate summary
    (records / voter_pages / wall_s / flag_rate). The raw PDF is always unlinked here. With
    keep_jsonl the per-voter JSONL is KEPT (path returned as 'jsonl') for an end-of-AC detect/diff
    and the caller MUST unlink it; without it the JSONL is unlinked immediately (P1 posture)."""
    pdf = os.path.join(work, f"{tag}.pdf")
    with open(pdf, "wb") as f:
        f.write(pdf_bytes)
    out = os.path.join(work, f"{tag}.jsonl")
    cmd = [sys.executable, OCR, pdf, "--out", out, "--stage1", str(stage1),
           "--ort-threads", "1", "--quiet", "--roll-year", str(roll_year)]
    for m in (models or MODELS):
        cmd += ["--crnn", m]
    if hints:
        cmd += ["--hint-2027", hints]
    if cache_in:
        cmd += ["--cache-in", cache_in]
    if cache_out:
        cmd += ["--cache-out", cache_out]
    run_env = {**os.environ, **(env or {})}   # model-set ECI_* switches, set explicitly for provenance
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=OCR_TIMEOUT_S, env=run_env)
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


def _parse_range(spec):
    """'LO-HI' -> (lo, hi) inclusive part bounds; None/'' -> (0, 0) meaning the whole AC."""
    if not spec:
        return 0, 0
    a, b = spec.split("-")
    return int(a), int(b)


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
                if r.get("not_voter_box"):      # eci_ocr writes these garbage boxes to JSONL but
                    continue                    # excludes them from summary.records (fixes 73,356>73,247)
                e = _norm_epic(r.get("epic"))
                if e:
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
                if r.get("not_voter_box"):      # excluded from counts, as eci_ocr excludes them
                    continue
                n += 1
                e = _norm_epic(r.get("epic"))
                if e:
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


def pull_2027_hints(ac, work):
    """Pull ONE AC's 2027 SSR text-layer records -> work/hints.jsonl (keyed by EPIC) for --hint-2027.
    The 2027 PDFs carry a text layer (no OCR): each is fetched, parsed, and unlinked immediately. The
    hints file is the CALLER's to unlink. Returns the path, or None if nothing parsed."""
    import parse_textlayer as T
    parts = sorted(part_list("2027-ssr-draft", ac), key=lambda p: int(p["partNumber"]))
    path = os.path.join(work, "hints.jsonl")
    n = 0
    with open(path, "w", encoding="utf-8") as out:
        for p in parts:
            part = int(p["partNumber"])
            b = get_pdf("2027-ssr-draft", ac, part)
            if not b:
                continue
            fd, pp = tempfile.mkstemp(suffix=".pdf", dir=work)
            try:
                os.write(fd, b)
                os.close(fd)
                recs, _ = T.parse_pdf(pp)
            except Exception as ex:                     # a bad part must not sink the whole AC
                print(f"    [hints] parse fail AC{ac} part{part}: {ex}", flush=True)
                recs = []
            finally:
                try:
                    os.unlink(pp)                       # discard the 2027 PDF immediately (0 footprint)
                except OSError:
                    pass
            for r in recs:
                if r.get("epic"):
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
    print(f"    [hints] AC {ac}: {n} 2027 SSR hint records pulled (ephemeral)", flush=True)
    return path if n else None


def _public_rows(jsonls, roll, ac, model_set, salt, draft_epics, final_epics):
    """Allowlisted PII-safe rows from kept JSONLs (not_voter_box excluded). EPIC/house -> keyed HMAC;
    deleted_in_final / from_draft from the Draft<->Final EPIC sets. No raw PII ever emitted."""
    from_draft_roll = roll.endswith("draft")
    rows = []
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
                if r.get("not_voter_box"):
                    continue
                ep = _norm_epic(r.get("epic"))
                house = str(r.get("house_no") or "").strip()
                placeholder = house.lower() in PLACEHOLDER_HOUSE
                g = str(r.get("gender") or "").strip()
                gu = g.upper()
                gender = ("M" if ("पुरुष" in g or gu in ("M", "MALE"))
                          else "F" if ("महिला" in g or gu in ("F", "FEMALE")) else None)
                if from_draft_roll:
                    deleted = bool(ep and final_epics is not None and ep not in final_epics)
                    from_draft = True
                else:
                    deleted = False
                    from_draft = bool(ep and draft_epics is not None and ep in draft_epics)
                rows.append({
                    "epic_hmac": _hmac(salt, ep) if ep else None,
                    "house_hmac": _hmac(salt, house) if (house and not placeholder) else None,
                    "house_is_placeholder": placeholder,
                    "state": STATE, "roll": roll, "revision_year": 2026, "ac": int(ac),
                    "part": _to_int(r.get("part_no")),
                    "age": _to_int(r.get("age")),
                    "gender": gender,
                    "relation_type": (str(r.get("relation_label") or "").strip() or None),
                    "status_code": (str(r.get("code") or "").strip() or None),
                    "flags": list(r.get("flags") or []),
                    "deleted_in_final": deleted,
                    "from_draft": from_draft,
                    "ocr_model_set": model_set,
                })
    return rows


def write_public_release(ac, out_root, model_set, draft_jsonls, final_jsonls, part_suffix=""):
    """Write public_out/release/<roll>/ac=<ac>/data[.pLO-HI].parquet (zstd-9) for Draft & Final,
    allowlisted + HMAC'd, plus a sidecar epic_hmac_tok[.pLO-HI].txt (unique sorted) for cheap cross-AC
    collision counting. part_suffix keeps a shard's files from colliding (reduce concatenates later).
    An allowlist guard refuses to write if any column outside PUBLIC_FIELDS appears."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    salt_env = os.environ.get("EPIC_HMAC_SALT")
    if not salt_env:
        raise SystemExit("--public-out needs EPIC_HMAC_SALT in the environment (repo secret)")
    salt = salt_env.encode("utf-8")
    draft_epics = _epics(draft_jsonls)     # full-roll sets -> correct deleted_in_final / from_draft
    final_epics = _epics(final_jsonls)
    for roll, jsonls in (("2026-sir-draft", draft_jsonls), ("2026-sir-final", final_jsonls)):
        rows = _public_rows(jsonls, roll, ac, model_set, salt, draft_epics, final_epics)
        if not rows:
            continue
        extra = {k for row in rows for k in row} - set(PUBLIC_FIELDS)
        if extra:                           # defensive: never ship a column that isn't on the allowlist
            raise SystemExit(f"public release leak guard tripped: unexpected columns {sorted(extra)}")
        d = os.path.join(out_root, "release", roll, f"ac={ac}")
        os.makedirs(d, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(d, f"data{part_suffix}.parquet"),
                       compression="zstd", compression_level=9)
        toks = sorted({row["epic_hmac"] for row in rows if row["epic_hmac"]})
        with open(os.path.join(d, f"epic_hmac_tok{part_suffix}.txt"), "w") as f:
            f.write("\n".join(toks) + ("\n" if toks else ""))
        print(f"    [public] {roll} ac={ac}{part_suffix}: {len(rows)} rows, {len(toks)} epic toks "
              f"-> data{part_suffix}.parquet (zstd-9) + tok", flush=True)


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
    dd = run_ocr(d, work, f"draft.{part}", 2026, cache_out=cache, hints=args.hints, keep_jsonl=True,
                 models=args._models, env=args._env, stage1=args._stage1)
    # CPU fast-bail on first part of AC so slow-CPU runners (e.g. >7.5 s/page) hand back in ~2 min
    if totals["parts"] == 0 and _per(dd) and _per(dd) > SLOW_CPU_SPP:
        raise SlowRunner(f"slow CPU {_per(dd)} s/page > {SLOW_CPU_SPP} s/page on part {part}")
    draft_jsonls.append(dd["jsonl"])
    totals["draft_records"] += dd["records"] or 0
    totals["draft_voter_pages"] += dd["voter_pages"] or 0

    fb, fst = fetch_pdf("2026-sir-final", ac, part, speeds)
    _check_slow(speeds)
    if fst == "ok":
        ff = run_ocr(fb, work, f"final.{part}", 2026, cache_in=cache, hints=args.hints, keep_jsonl=True,
                     models=args._models, env=args._env, stage1=args._stage1)
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
        if getattr(args, "pull_2027_hints", False):
            args.hints = pull_2027_hints(ac, work)   # ephemeral; work/ is rmtree'd in finally
        lo, hi = _parse_range(getattr(args, "part_range", None))
        listed = sorted(part_list("2026-sir-draft", ac), key=lambda p: int(p["partNumber"]))
        all_part_nums = [int(p["partNumber"]) for p in listed]
        listed_max = max(all_part_nums) if all_part_nums else 0
        if lo or hi:
            part_nums = [p for p in all_part_nums if lo <= p <= hi]
            is_top = hi >= listed_max      # only the TOP shard walks extras + final-only
            totals["shard"] = f"{lo}-{hi}"
        else:
            part_nums = all_part_nums
            is_top = True
            totals["shard"] = "full"
        try:
            final_listed = {int(p["partNumber"]) for p in part_list("2026-sir-final", ac)}
        except RuntimeError:
            final_listed = set()
        seen_parts = set(all_part_nums)    # FULL draft list, so final-only is measured once (top shard)

        for part in part_nums:
            if _process_part(ac, part, work, speeds, totals, draft_jsonls, final_jsonls,
                             paired_draft_jsonls, paired_final_jsonls, args) == "fail":
                print(f"    AC {ac} part {part}: draft fetch gave up (tarpit); skipped", flush=True)

        if is_top and all_part_nums:
            # EXTRA_MISS 404-walk above the listed max (2026 deleted/merged parts; fetch_task.py)
            top0, miss, n = listed_max, 0, listed_max
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
                    ff = run_ocr(fb, work, f"final.{part}", 2026, hints=args.hints, keep_jsonl=True,
                                 models=args._models, env=args._env, stage1=args._stage1)
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
                   diff_gross_deleted_pct=round(100 * deleted / max(1, len(draft_epics)), 2),
                   diff_net_change_pct=round(100 * (len(final_epics) - len(draft_epics))
                                             / max(1, len(draft_epics)), 2),
                   detect_draft=_detect_counts(draft_jsonls),
                   detect_final=_detect_counts(final_jsonls),
                   ocr_model_set=getattr(args, "model_set", None) or "legacy",
                   confidence="Plausible - OCR EPICs; validate Draft->Final diff before quoting")
        if getattr(args, "public_out", None):
            psuffix = f".p{lo}-{hi}" if (lo or hi) else ""
            write_public_release(ac, args.public_out, getattr(args, "model_set", None) or "legacy",
                                 draft_jsonls, final_jsonls, part_suffix=psuffix)
        return agg
    finally:
        shutil.rmtree(work, ignore_errors=True)

def cmd_ingest(args):
    """P2 per-AC pipeline (dispatch only after AG1 sign-off). skip-if-exists resume mirrors
    uk_driver.cmd_textlayer; emits PII-safe aggregate counts only."""
    acs = parse_acs(args.acs)
    agg_dir = os.path.join(args.out, "aggregates", "2026")
    os.makedirs(agg_dir, exist_ok=True)
    args._models, args._env, args._stage1 = load_model_set(getattr(args, "model_set", None))
    if getattr(args, "model_set", None):
        print(f"[ingest] model-set {args.model_set}: {len(args._models)} CRNNs, "
              f"{len(args._env)} env switches, stage1={args._stage1}", flush=True)
    lo, hi = _parse_range(getattr(args, "part_range", None))
    suffix = f".p{lo}-{hi}" if (lo or hi) else ""   # shard files don't collide; reduce merges them
    for ac in acs:
        agg_path = os.path.join(agg_dir, f"ac={ac}{suffix}.json")
        if os.path.exists(agg_path) and os.path.getsize(agg_path) > 0:
            print(f"  AC {ac}{suffix}: aggregate present, skip (resume-safe)", flush=True)
            continue
        try:
            agg = _ingest_ac(ac, args)
        except SlowRunner as e:
            print(f"  AC {ac}: slow runner ({e}) -> handed back (fast-bail; matrix reshuffle)",
                  flush=True)
            continue
        if agg.get("incomplete"):
            # Resume contract: a tarpit-truncated shard writes NO file -> no artifact -> re-run next pass
            print(f"  AC {ac}{suffix}: INCOMPLETE (tarpit) -> NOT written; will re-run on resume",
                  flush=True)
            continue
        with open(agg_path, "w") as f:
            json.dump(agg, f, ensure_ascii=False)  # counts only - safe to keep / upload
        print(f"  AC {ac}{suffix}: {agg['parts']} parts ({agg['extra_parts']} extra) | "
              f"draft {agg['draft_distinct_epics']} epics, final {agg['final_distinct_epics']} | "
              f"DELETED {agg['diff_deleted_epics']} (gross {agg['diff_gross_deleted_pct']}%, "
              f"net {agg['diff_net_change_pct']}%)", flush=True)
    print("[ingest] PII-safe aggregates written. All raw PDFs/records/JSONLs discarded.", flush=True)


# ---------------- plan + reduce (shard sizing & cross-shard merge) ----------------
def shard_windows(part_nums, shard_parts):
    """Split sorted listed part numbers into <=shard_parts windows -> [(lo, hi), ...] inclusive.
    Each listed part lands in exactly one window. Empty list -> whole AC (no sharding needed)."""
    pn = sorted({int(p) for p in part_nums})
    if len(pn) <= shard_parts:
        return []
    out = []
    for i in range(0, len(pn), shard_parts):
        chunk = pn[i:i + shard_parts]
        out.append((chunk[0], chunk[-1]))
    return out


def cmd_plan(args):
    """Sweep draft part counts per AC (gateway API only, no PDFs), shard ACs above --shard-parts,
    subtract already-done ACs (--done-acs), emit the (ac, part-range) matrix to GITHUB_OUTPUT/stdout.
    Cell = {ac, lo, hi, tag}; lo==hi==0 -> whole AC."""
    done = set(parse_acs(args.done_acs, 1, 70)) if args.done_acs else set()
    cells = []
    for ac in parse_acs(args.acs):
        if ac in done:
            print(f"[plan] AC {ac}: clean artifact exists -> skip (resume)", flush=True)
            continue
        try:
            nums = [int(p["partNumber"]) for p in part_list("2026-sir-draft", ac)]
        except RuntimeError as e:
            print(f"[plan] AC {ac}: part-list failed ({e}) -> whole-AC cell", flush=True)
            cells.append({"ac": ac, "lo": 0, "hi": 0, "tag": str(ac)})
            continue
        wins = shard_windows(nums, args.shard_parts)
        if not wins:
            cells.append({"ac": ac, "lo": 0, "hi": 0, "tag": str(ac)})
            print(f"[plan] AC {ac}: {len(nums)} parts -> 1 cell (single-shard)", flush=True)
        else:
            for wlo, whi in wins:
                cells.append({"ac": ac, "lo": wlo, "hi": whi, "tag": f"{ac}.p{wlo}-{whi}"})
            print(f"[plan] AC {ac}: {len(nums)} parts -> {len(wins)} shards "
                  f"{[f'{wlo}-{whi}' for wlo, whi in wins]}", flush=True)
    payload = json.dumps(cells)
    acs_out = json.dumps(sorted({c["ac"] for c in cells}))
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write("cells=" + payload + "\n")
            f.write("acs=" + acs_out + "\n")       # distinct ACs -> reduce matrix
            f.write("count=" + str(len(cells)) + "\n")
    print("[plan] cells=" + payload, flush=True)
    print("[plan] acs=" + acs_out, flush=True)


_SUM_KEYS = ("parts", "extra_parts", "final_only_parts", "draft_failed_parts", "final_failed_parts",
             "draft_records", "draft_voter_pages", "final_records", "final_voter_pages",
             "draft_distinct_epics", "final_distinct_epics",
             "diff_deleted_epics", "diff_added_epics", "diff_retained_epics")
_DETECT_SUM = ("records", "dup_epic_within_ac", "age_impossible", "age_centenarian",
               "gender_m", "gender_f", "congestion_houses_gt_thresh")


def _merge_detect(ds):
    if not ds:
        return None
    m = {k: sum(int(d.get(k) or 0) for d in ds) for k in _DETECT_SUM}
    m["congestion_max"] = max((int(d.get("congestion_max") or 0) for d in ds), default=0)
    mm = m["gender_m"]
    m["females_per_1000_males"] = round(1000 * m["gender_f"] / mm) if mm else None  # recomputed, not summed
    m["congestion_confidence"] = ds[0].get("congestion_confidence")
    return m


def merge_shards(shards):
    """Sum N per-shard aggregates of ONE AC into a single PII-safe AC aggregate. Part populations
    are disjoint across shards, so counts sum; percentages and F/1000M are recomputed from the sums."""
    ac = shards[0]["ac"]
    m = {"ac": ac}
    for k in _SUM_KEYS:
        m[k] = sum(int(s.get(k) or 0) for s in shards)
    m["incomplete"] = any(bool(s.get("incomplete")) for s in shards)
    dd = max(1, m["draft_distinct_epics"])
    m["diff_gross_deleted_pct"] = round(100 * m["diff_deleted_epics"] / dd, 2)
    m["diff_net_change_pct"] = round(100 * (m["final_distinct_epics"] - m["draft_distinct_epics"]) / dd, 2)
    m["detect_draft"] = _merge_detect([s["detect_draft"] for s in shards if s.get("detect_draft")])
    m["detect_final"] = _merge_detect([s["detect_final"] for s in shards if s.get("detect_final")])
    m["shards"] = sorted(str(s.get("shard", "full")) for s in shards)
    m["ocr_model_set"] = shards[0].get("ocr_model_set")
    m["confidence"] = shards[0].get("confidence")
    return m


def cmd_reduce(args):
    """Merge this run's downloaded shard jsons (--shard-dir, any depth) into per-AC ac={n}.json.
    An AC is emitted ONLY if ALL its expected shards (from --expect cells) are present and clean;
    otherwise it is left unwritten so the next dispatch re-runs it (resume contract)."""
    sel = set(parse_acs(args.acs)) if getattr(args, "acs", None) else None  # reduce one AC per matrix cell
    expect = {}
    for c in json.loads(args.expect or "[]"):
        ac = int(c["ac"])
        if sel is not None and ac not in sel:
            continue
        expect.setdefault(ac, set()).add(c["tag"])
    found = {}   # ac -> {tag: agg}
    for root, _, files in os.walk(args.shard_dir or "."):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            try:
                agg = json.load(open(os.path.join(root, fn)))
            except (ValueError, OSError):
                continue
            ac = agg.get("ac")
            if ac is None:
                continue
            sh = str(agg.get("shard", "full"))
            tag = str(ac) if sh == "full" else f"{ac}.p{sh}"
            found.setdefault(int(ac), {})[tag] = agg
    agg_dir = os.path.join(args.out, "aggregates", "2026")
    os.makedirs(agg_dir, exist_ok=True)
    for ac, want in sorted(expect.items()):
        have = found.get(ac, {})
        if not want.issubset(have.keys()):
            print(f"[reduce] AC {ac}: missing shard(s) {sorted(want - set(have))} -> NOT merged (resume)",
                  flush=True)
            continue
        merged = merge_shards([have[t] for t in sorted(want)])
        if merged["incomplete"]:
            print(f"[reduce] AC {ac}: a shard is incomplete -> NOT merged", flush=True)
            continue
        with open(os.path.join(agg_dir, f"ac={ac}.json"), "w") as f:
            json.dump(merged, f, ensure_ascii=False)
        print(f"[reduce] AC {ac}: merged {len(want)} shard(s) -> {merged['parts']} parts | "
              f"DELETED {merged['diff_deleted_epics']} (gross {merged['diff_gross_deleted_pct']}%, "
              f"net {merged['diff_net_change_pct']}%)", flush=True)
    print("[reduce] per-AC merges complete.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["probe", "ingest", "plan", "reduce"])
    ap.add_argument("--out", default="./out")
    ap.add_argument("--acs", default="1-70", help="e.g. 1-70 or 1,21,19 (ingest/plan)")
    ap.add_argument("--part-range", dest="part_range", default=None,
                    help="ingest: LO-HI inclusive draft-part window for one shard (extras+final-only run in top shard)")
    ap.add_argument("--shard-parts", dest="shard_parts", type=int, default=200,
                    help="plan: max listed parts per shard (ACs above this split into windows)")
    ap.add_argument("--done-acs", dest="done_acs", default=None,
                    help="plan: ACs whose clean merged artifact already exists (skipped for resume)")
    ap.add_argument("--shard-dir", dest="shard_dir", default=None,
                    help="reduce: directory of downloaded per-shard jsons")
    ap.add_argument("--expect", default=None,
                    help="reduce: plan cells JSON (which shards each AC expects)")
    ap.add_argument("--model-set", dest="model_set", default=None,
                    help="ingest: model_sets.json entry (e.g. v4a) -> its CRNNs + env + stage1 (else legacy c11q..c3q)")
    ap.add_argument("--pull-2027-hints", dest="pull_2027_hints", action="store_true",
                    help="ingest: pull each AC's 2027 SSR text-layer -> ephemeral --hint-2027 (hinted sets like v4a need it)")
    ap.add_argument("--public-out", dest="public_out", default=None,
                    help="ingest: dir for the PII-safe public release (release/<roll>/ac=<ac>/data.parquet, HMAC'd via EPIC_HMAC_SALT)")
    ap.add_argument("--hints", default=os.environ.get("ECI_HINTS"),
                    help="optional 2027 text-layer hints JSONL (confirmation only; prod uses it to skip)")
    args = ap.parse_args()
    {"probe": cmd_probe, "ingest": cmd_ingest, "plan": cmd_plan, "reduce": cmd_reduce}[args.mode](args)


if __name__ == "__main__":
    main()
