#!/usr/bin/env python3
"""Uttarakhand (S28) ingestion driver - hub-free.

Reuses this repo's own eci_api.py (part-list API + public PDF paths, replicating ECI's own
request crypto with ECI's own public key) and eciocr/textlayer/parse_textlayer.py (2027 SSR
records straight from the PDF text layer, no OCR). It does NOT use scrape_hub_client: it loops
locally and writes to an output dir (a Google Drive folder in Colab, or ./out in Actions).

Assumptions (karpathy: stated, not hidden):
  - S28 rolls live at the same public paths as S24 (verified live 2026-11-26, AC1 HTTP 206 %PDF).
  - Only the 3 EROLLGEN rolls below have public paths (no captcha). 2024 GE is out of scope here.
  - 2027 SSR has a text layer -> parse_pdf. 2026 SIR rolls are scanned -> OCR stage (CC#2), not here.

Modes:
  check     - AC 1, 2027: part list -> 1 PDF -> parse -> print RECORD COUNT only (self-test; no names printed)
  parts     - part-list sweep across ACs x 3 rolls -> out/parts/<roll>/<ac>.json + out/manifest.json (sizing)
  textlayer - 2027 SSR: download each part, parse_pdf -> rows -> Parquet(zstd) per AC; PDFs streamed + unlinked

PII: 2027 records contain real names. They are written ONLY to the private out dir (never the public site).
PDFs are held in memory / a unlinked temp file and discarded immediately (0 lasting footprint).

Run (Colab/Actions): pip install requests cryptography pymupdf pyarrow
  python uk_driver.py check
  python uk_driver.py parts     --out /content/drive/MyDrive/voter-engine/data
  python uk_driver.py textlayer --out /content/drive/MyDrive/voter-engine/data --acs 1-70
"""
import argparse, json, os, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

# --- make the repo's own modules importable (driver lives in uk/, repo root is one up) ---
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eciocr", "textlayer"))
import eci_api as E  # noqa: E402  (part-list request, pdf url, is_pdf, HEADERS, now)

STATE = "S28"
# Generic roll records (eci_api.roll_* takes these dicts). Confirmed live for S28 2026-11-26.
UK_ROLLS = {
    "2026-sir-draft": {"year": 2026, "ref": "SIR-DraftRoll", "gen": "EROLLGEN", "rev": 1},
    "2026-sir-final": {"year": 2026, "ref": "SIR-FinalRoll", "gen": "EROLLGEN", "rev": 1},
    "2027-ssr-draft": {"year": 2027, "ref": "SSR-DraftRoll", "gen": "EROLLGEN", "rev": 1},
}
TEXTLAYER_ROLL = "2027-ssr-draft"  # the only one with a text layer (no OCR)

import requests  # noqa: E402

_S = None
def session():
    global _S
    if _S is None:
        _S = requests.Session()
        _S.headers.update(E.HEADERS)
    return _S


def part_list(roll_key, ac):
    """Return the parts array (each: partNumber, partName, districtCd, partId). Plaintext payload."""
    roll = UK_ROLLS[roll_key]
    url, body = E.roll_part_list_request(roll, STATE, ac)
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


def get_pdf(roll_key, ac, part):
    roll = UK_ROLLS[roll_key]
    url = E.roll_pdf_url(roll, STATE, ac, part)
    for attempt in range(4):
        try:
            r = session().get(url, timeout=180)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and E.is_pdf(r.content):
                cl = r.headers.get("Content-Length")
                if not cl or int(cl) == len(r.content):
                    return r.content
        except requests.RequestException:
            pass
        time.sleep(6 * (attempt + 1))
    return None


def parse_acs(spec, lo=1, hi=70):
    out = set()
    for tok in (spec or f"{lo}-{hi}").split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); out.update(range(int(a), int(b) + 1))
        elif tok:
            out.add(int(tok))
    return sorted(x for x in out if lo <= x <= hi)


# ---------------- modes ----------------
def cmd_check(args):
    ac = 1
    parts = part_list(TEXTLAYER_ROLL, ac)
    print(f"[check] {TEXTLAYER_ROLL} AC {ac}: {len(parts)} parts listed")
    p1 = sorted(parts, key=lambda p: int(p["partNumber"]))[0]
    b = get_pdf(TEXTLAYER_ROLL, ac, int(p1["partNumber"]))
    assert b and E.is_pdf(b), "part 1 PDF not fetched/valid"
    print(f"[check] part {p1['partNumber']} ({p1.get('partName')}): {len(b)} bytes, valid PDF")
    import parse_textlayer as T
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(fd, b); os.close(fd)
        records, meta = T.parse_pdf(path)
    finally:
        os.unlink(path)
    print(f"[check] parsed {len(records)} voter records from 1 part "
          f"(cover total={meta.get('cover_total') if isinstance(meta, dict) else '?'}). "
          f"Chain OK: part-list -> download -> parse. (names NOT printed)")


def cmd_parts(args):
    acs = parse_acs(args.acs)
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)
    manifest = {"state": STATE, "generated": E.now(), "rolls": {}}
    for roll_key in UK_ROLLS:
        os.makedirs(os.path.join(args.out, "parts", roll_key), exist_ok=True)
        total = 0
        for ac in acs:
            try:
                parts = part_list(roll_key, ac)
            except RuntimeError as e:
                print(f"  ! {e}"); continue
            total += len(parts)
            with open(os.path.join(args.out, "parts", roll_key, f"{ac}.json"), "w") as f:
                json.dump({"roll": roll_key, "ac": ac, "fetched": E.now(), "parts": parts},
                          f, ensure_ascii=False)
            print(f"  {roll_key} AC {ac}: {len(parts)} parts (running {total})")
            time.sleep(0.4)
        manifest["rolls"][roll_key] = {"acs": len(acs), "parts": total}
    manifest["total_parts"] = sum(r["parts"] for r in manifest["rolls"].values())
    manifest["est_gb_at_4mb"] = round(manifest["total_parts"] * 4 / 1024, 1)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[parts] TOTAL {manifest['total_parts']} PDFs across {len(UK_ROLLS)} rolls "
          f"(~{manifest['est_gb_at_4mb']} GB raw). manifest.json written.")


def cmd_textlayer(args):
    import pyarrow as pa, pyarrow.parquet as pq
    import parse_textlayer as T
    acs = parse_acs(args.acs)
    base = os.path.join(args.out, "release", "2027-ssr")
    for ac in acs:
        parts = sorted(part_list(TEXTLAYER_ROLL, ac), key=lambda p: int(p["partNumber"]))
        rows = []
        def one(p):
            part = int(p["partNumber"])
            b = get_pdf(TEXTLAYER_ROLL, ac, part)
            if not b:
                return []
            fd, path = tempfile.mkstemp(suffix=".pdf")
            try:
                os.write(fd, b); os.close(fd)
                recs, _ = T.parse_pdf(path)
            except Exception as ex:
                print(f"    parse fail AC{ac} part{part}: {ex}"); return []
            finally:
                os.unlink(path)  # discard the PDF immediately (0 footprint)
            for r in recs:
                r["ac_no"] = ac; r["part_no"] = part
            return recs
        with ThreadPoolExecutor(4) as ex:
            for recs in ex.map(one, parts):
                rows.extend(recs)
        if not rows:
            print(f"  AC {ac}: no records"); continue
        os.makedirs(os.path.join(base, f"ac={ac}"), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows),
                       os.path.join(base, f"ac={ac}", "data.parquet"), compression="zstd")
        print(f"  AC {ac}: {len(rows)} records -> ac={ac}/data.parquet (zstd)")
    print("[textlayer] 2027 SSR records written to private Parquet. PDFs discarded.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["check", "parts", "textlayer"])
    ap.add_argument("--out", default="./out")
    ap.add_argument("--acs", default="1-70", help="e.g. 1-70 or 1,21,19")
    args = ap.parse_args()
    {"check": cmd_check, "parts": cmd_parts, "textlayer": cmd_textlayer}[args.mode](args)


if __name__ == "__main__":
    main()
