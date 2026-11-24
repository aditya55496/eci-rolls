"""Task `parquet`: one AC's electoral-roll records as Parquet files (owner 2026-11-24).

Queue item (project eci-rolls-up-parquet): key "<ac>", payload {"ac": 174, "parts": [1, 2, ...], "rolls": [...]}
(GE 2024: key "ge/<ac>", payload {"ac", "ge_parts": [...], "rolls": ["2024-ge-final"]}).
Rolls: "2026-sir-draft" and "2026-sir-final" from the OCR output (ocr/<ac>/<part>.jsonl.gz: both rolls in one file, after the
EPIC and final-roll repairs), "2027-ssr-draft" from the text layer (text/<ac>.jsonl.gz); later "2024-ge-final"
(ocr/<ac>/<part>-ge2024.jsonl.gz).
Every record is kept: flagged records, not_voter_box rows and repaired records, all fields, the flags list, the stored readings
(JSON text column) and the repair markers (epic_from_2027, epic_ocr, deleted_in_final, from_draft, final_ocr, voter_by_serial,
not_voter_box). Keys outside the fixed schema go to the JSON column 'extra', so nothing is dropped.
Output: parquet/<roll>/ac_<ac>/data.parquet (zstd; the hub allows no '=' in paths, so ac_<n>, not ac=<n>). One hub record per
AC: rows per roll (all rows, voter rows, not_voter_box rows) and the parts without an OCR file. No OCR, no network but the hub
and the bucket links. pyarrow is installed at the first run of this task only (the OCR tasks' setup is not changed).
"""
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import time

from scrape_hub_client import HubClient, stop_flag

from common import OCR_PROJECT, TEXT_PROJECT, download

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow==26.0.1"], check=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "eci-rolls-up-parquet")
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(600, _BUDGET / 4)

S, I, B, F = pa.string(), pa.int32(), pa.bool_(), pa.float32()
OCR_SCHEMA = pa.schema([
    ("roll", S), ("ac", I), ("part", I), ("source", S), ("state", S), ("page", I), ("box", I), ("kind", S), ("section", S),
    ("code", S), ("serial_no", S), ("part_no", S), ("epic", S), ("name", S), ("relation_label", S), ("relation_name", S),
    ("house_no", S), ("age", S), ("gender", S), ("flags", pa.list_(S)), ("readings", S), ("hint_2027", B), ("code_ink", B),
    ("bbox", pa.list_(I)), ("t_box_s", F), ("not_voter_box", B), ("epic_from_2027", B), ("epic_ocr", S), ("deleted_in_final", B),
    ("from_draft", B), ("final_ocr", S), ("voter_by_serial", B), ("extra", S)])
TEXT_SCHEMA = pa.schema([
    ("roll", S), ("ac", I), ("part", I), ("page", I), ("section_no", I), ("section_name", S), ("serial_no", I), ("epic", S),
    ("name", S), ("relation_type", S), ("relation_name", S), ("house_no", S), ("age", I), ("gender", S), ("status_code", S),
    ("repairs", pa.list_(S)), ("text_warnings", pa.list_(S)), ("extra", S)])
JSON_COLS = {"readings", "final_ocr"}
BOOL_COLS = {"not_voter_box", "epic_from_2027", "deleted_in_final", "from_draft", "voter_by_serial", "hint_2027", "code_ink"}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def to_table(rows, schema, roll, ac=None):
    names = [f.name for f in schema]
    cols = {n: [] for n in names}
    for r in rows:
        r = dict(r)
        r.setdefault("part", r.get("roll_part"))
        if ac is not None and _int(r.get("ac")) != ac:
            # the AC is the item's; the text layer reads it from the PDF header, which is wrong for some ACs (2026-11-24:
            # AC 320's 2027 PDFs give 64; the voters are AC 320's). The header value is kept in 'extra' as ac_in_pdf.
            r["ac_in_pdf"], r["ac"] = r.get("ac"), ac
        extra = {k: v for k, v in r.items() if k not in cols and k != "roll_part"}
        for n in names:
            if n == "roll":
                v = roll
            elif n == "extra":
                v = json.dumps(extra, ensure_ascii=False) if extra else None
            else:
                v = r.get(n)
                t = schema.field(n).type
                if n in JSON_COLS:
                    v = json.dumps(v, ensure_ascii=False) if v is not None else None
                elif n in BOOL_COLS:
                    v = bool(v)
                elif t == I:
                    v = _int(v)
                elif t == F:
                    v = float(v) if v is not None else None
                elif t == S:
                    v = None if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                elif pa.types.is_list(t) and v is not None:
                    v = [None if x is None else (str(x) if t.value_type == S else _int(x)) for x in v]
            cols[n].append(v)
    return pa.table(cols, schema=schema)


def write(hub, roll, ac, table):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd", compression_level=9, row_group_size=131072)
    data = buf.getvalue()
    hub.upload_many([(f"parquet/{roll}/ac_{ac}/data.parquet", data, "application/vnd.apache.parquet")])
    return len(data)


def counts(rows):
    nv = sum(1 for r in rows if r.get("not_voter_box"))
    return {"rows": len(rows), "voters": len(rows) - nv, "not_voter_box": nv}


def do_item(hub, item):
    p = item["payload"]
    # GE items carry no "rolls" (the key says it): a worker with older code of this task fails on them (KeyError) instead
    # of overwriting the AC's SIR record with an empty one (2026-11-24)
    rolls = p.get("rolls") or (["2024-ge-final"] if item["key"].startswith("ge/") else [])
    ac, parts = int(p["ac"]), [int(x) for x in p.get("parts", [])]
    rec = {"ac": ac, "rolls": {}}
    with tempfile.TemporaryDirectory() as tmp:
        ocr_rolls = [r for r in rolls if r in ("2026-sir-draft", "2026-sir-final")]
        if ocr_rolls:
            keys = [f"{OCR_PROJECT}/ocr/{ac}/{x}.jsonl.gz" for x in parts]
            got = download(hub, OCR_PROJECT, keys, tmp)
            by = {r: [] for r in ocr_rolls}
            missing = []
            for x, k in zip(parts, keys):
                if not got.get(k):
                    missing.append(x)
                    continue
                with gzip.open(got[k], "rt", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        src = r.get("source") or ""
                        roll = "2026-sir-final" if "FinalRoll" in src else "2026-sir-draft" if "DraftRoll" in src else None
                        if roll in by:
                            by[roll].append(r)
                os.remove(got[k])
            for roll in ocr_rolls:
                rows = by[roll]
                c = counts(rows)
                c["bytes"] = write(hub, roll, ac, to_table(rows, OCR_SCHEMA, roll, ac)) if rows else 0
                c["parts_without_ocr_file"] = missing
                rec["rolls"][roll] = c
            by = None
        if "2024-ge-final" in rolls:   # GE 2024: its own OCR files, ocr/<ac>/<part>-ge2024.jsonl.gz (parts: payload ge_parts)
            gparts = [int(x) for x in p.get("ge_parts", [])]
            keys = [f"{OCR_PROJECT}/ocr/{ac}/{x}-ge2024.jsonl.gz" for x in gparts]
            got = download(hub, OCR_PROJECT, keys, tmp)
            rows, missing = [], []
            for x, k in zip(gparts, keys):
                if not got.get(k):
                    missing.append(x)
                    continue
                with gzip.open(got[k], "rt", encoding="utf-8") as fh:
                    rows += [json.loads(line) for line in fh if line.strip()]
                os.remove(got[k])
            c = counts(rows)
            c["bytes"] = write(hub, "2024-ge-final", ac, to_table(rows, OCR_SCHEMA, "2024-ge-final", ac)) if rows else 0
            c["parts_without_ocr_file"] = missing
            rec["rolls"]["2024-ge-final"] = c
            rows = None
        if "2027-ssr-draft" in rolls:
            k = f"{TEXT_PROJECT}/text/{ac}.jsonl.gz"
            path = download(hub, TEXT_PROJECT, [k], tmp)[k]
            if not path:
                raise RuntimeError(f"no 2027 text layer for AC {ac}")
            rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8") if l.strip()]
            c = {"rows": len(rows), "ac_in_pdf_differs": sum(1 for r in rows if _int(r.get("ac")) != ac),
                 "bytes": write(hub, "2027-ssr-draft", ac, to_table(rows, TEXT_SCHEMA, "2027-ssr-draft", ac))}
            rec["rolls"]["2027-ssr-draft"] = c
    hub.record(item["key"], rec)   # "<ac>" for SIR + 2027, "ge/<ac>" for GE 2024
    hub.metric("parquet_rows", sum(v["rows"] for v in rec["rolls"].values()))
    return sum(v["rows"] for v in rec["rolls"].values())


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
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:400])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
