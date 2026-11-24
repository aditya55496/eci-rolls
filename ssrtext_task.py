"""Task `ssrtext`: the 2027 SSR draft roll of one AC, any state and language, from the PDF text layer (no OCR). 2026-11-25.

Queue item: key "<roll id>/<ac>", payload {"state": "S19", "ac": 21, "lang": "PAN", "roll": {"id": "S19-2027-DR", ...},
"src_project": "eci-rolls-all-cdn", "keys": [bucket keys of the AC's part PDFs], "extra_parts": [parts above the portal's
part list, optional]} (items from plan_ssrtext.py). The PDFs are downloaded through the hub's read links of src_project.
Parser by language: HIN eciocr/textlayer/parse_textlayer.py (all Hindi states), ENG parse_textlayer_eng.py, any other
(ASM, BEN, MAL, KAN, PAN, ...) parse_textlayer_any.py (box layout and line position; no label text needed).
Output in the bucket of this project:
  text/<roll id>/<ac>.jsonl.gz           every voter record (all fields, repairs, warnings, raw_lines = the box rows as printed,
                                         without the photo placeholder); relation_type in one vocabulary for all languages
                                         (Father / Husband / Mother / Wife / Others / Guardian), the word read in relation_type_printed
  hints/<roll id>/<ac>.jsonl.gz          the --hint-2027 form eci_ocr uses (one per record with an EPIC)
  text/<roll id>/<ac>.summary.json       per PDF: records, printed totals (cover, summary page, page count), match, validation
  parquet/<roll id>/ac_<ac>/data.parquet the records in the TEXT_SCHEMA of parquet_task.py (other keys in the JSON column 'extra')
One hub record per AC (id = item key): counts and the per-PDF check against the printed totals. A PDF "matches" when its
record count equals the cover total and the summary-page total (when the page has one); gender counts are checked separately.
"""
import collections
import concurrent.futures as cf
import json
import os
import sys
import tempfile
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eciocr", "textlayer"))
import parse_textlayer as TH  # noqa: E402  Hindi
import parse_textlayer_eng as TE  # noqa: E402  English
import parse_textlayer_any as TA  # noqa: E402  any other language

from common import gz_jsonl, hint_of  # noqa: E402

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "eci-rolls-ssr-text")
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(600, _BUDGET / 4)
WORKERS = int(os.environ.get("ECI_SSR_WORKERS", str(min(4, os.cpu_count() or 1))))

GENDER_STD = {"पुरुष": "M", "महिला": "F", "तृतीय लिंग": "T", "Male": "M", "Female": "F", "Third Gender": "T"}
REL_STD = {"पिता": "Father", "पति": "Husband", "माता": "Mother", "पत्नी": "Wife", "अन्य": "Others",
           "Father": "Father", "Husband": "Husband", "Mother": "Mother", "Wife": "Wife", "Others": "Others", "Other": "Others",
           "Guardian": "Guardian"}


def parser_of(lang):
    return "hin" if lang == "HIN" else "eng" if lang == "ENG" else "any"


ENG_PHOTO = ("Photo", "Available", "Photo Available", "Photo Not Available")


def photo_span(c, box, parser):
    """The photo placeholder span of a box ('फोटो उपलब्ध', 'Photo' / 'Available'), as the parsers drop it."""
    t = c[4].strip()
    if parser == "eng" and t in ENG_PHOTO:
        return True
    word = TH.PHOTO_TEXT if parser == "hin" else TE.PHOTO_TEXT
    return t.startswith(word) and c[1] > box.x0 + 0.55 * box.width


def raw_lines_of(path, parser="hin"):
    """{(page, serial_no or EPIC): [row texts of the box, top row first]} for the Hindi and English parsers, which do not
    keep them (same box finder, character assignment and rows as the parsers; the photo placeholder spans left out)."""
    import pymupdf
    out = {}
    doc = pymupdf.open(path)
    for pno, page in enumerate(doc):
        boxes = TH.find_boxes(page, page.get_drawings())
        if not boxes:
            continue
        chars, _ = TH.page_chars(page)
        own = TH.assign([(c[1], c[2], c[3]) for c in chars], boxes)
        per = collections.defaultdict(list)
        for c, k in zip(chars, own):
            if k is not None and not photo_span(c, boxes[k], parser):
                per[k].append(c)
        for i in range(len(boxes)):
            rws = TH.rows_of(per[i], key_y=lambda c: c[2])
            rows = [TH.row_text(r["c"]) for r in rws]
            if rows:
                for tok in TH.row_text(rws[0]["c"], gap=0.5).split():
                    out.setdefault((pno + 1, tok), rows)
    return out


def parse_one(path, lang):
    """One PDF -> (records, meta with 'validation'). Runs in a worker process."""
    p = parser_of(lang)
    if p == "hin":
        recs, meta = TH.parse_pdf(path)
        val = TH.validate(recs, meta)
    elif p == "eng":
        recs, meta = TE.parse_pdf(path)
        val = TE.validate(recs, meta)
    else:
        recs, meta = TA.parse_pdf(path, lang)
        val = TA.validate(recs, meta)
    if p == "eng" and not meta.get("printed_pages"):
        import pymupdf
        meta["printed_pages"] = TA.printed_pages_of(pymupdf.open(path))
        if meta["printed_pages"]:
            val["pages_match"] = meta["printed_pages"] == meta.get("pages")
            val["ok"] = bool(val["ok"] and val["pages_match"])
    if p != "any":
        raw = raw_lines_of(path, p)
        for r in recs:
            rl = raw.get((r["page"], str(r.get("serial_no")))) or raw.get((r["page"], r.get("epic")))
            if rl:
                r["raw_lines"] = rl
    meta["validation"] = val
    meta["parser"] = p
    return recs, meta


def normalise(r, item, part, meta, extra_part):
    """Common record form for all parsers. The AC and part come from the item and the file name; the values the PDF
    prints are kept as ac_in_pdf / part_in_pdf when they differ."""
    p = item["payload"]
    ac = int(p["ac"])
    out = {"state": p["state"], "roll": p["roll"]["id"], "lang": p["lang"]}
    out.update(r)
    out["state"] = p["state"]
    if r.get("ac") is not None and r.get("ac") != ac:
        out["ac_in_pdf"] = r.get("ac")
    if r.get("part") is not None and r.get("part") != part:
        out["part_in_pdf"] = r.get("part")
    out["ac"], out["part"] = ac, part
    if meta["parser"] == "hin":
        rt = r.get("relation_type") or ""
        out["relation_label"] = (rt + " का नाम") if rt and rt != "अन्य" else rt
    if "gender_std" not in out:
        out["gender_std"] = GENDER_STD.get(r.get("gender"))
    # relation_type: one vocabulary for all languages (Father / Husband / Mother / Wife / Others / Guardian, None when the
    # printed word is unknown); the word as the parser read it ('पिता', 'Other') is relation_type_printed, the printed label
    # relation_label
    if r.get("relation_type") is not None:
        out["relation_type_printed"] = r["relation_type"]
    out["relation_type"] = REL_STD.get(r.get("relation_type"))
    if extra_part:
        out["extra_part"] = True
    return out


def hint(r, parser):
    if parser == "hin":
        h = hint_of(r)   # its relation_label comes from relation_type, which is now the common vocabulary
        h["relation_label"] = r.get("relation_label")
        h["state"] = r.get("state")
        for k in ("age", "serial_no"):
            if r.get(k) is None:
                h[k] = None
        return h
    return TE.hint_of_eng(r) if parser == "eng" else TA.hint_of_any(r)


def part_of(key):
    m = TA.FILE_RE.search(os.path.basename(key))
    return int(m.group(5)) if m else None


def pdf_check(meta, recs):
    """Per-PDF summary: records against the printed totals."""
    v = meta["validation"]
    cov = meta.get("cover") or {}
    summ = meta.get("summary") or {}
    n = len(recs)
    g = collections.Counter(r.get("gender_std") for r in recs)
    printed = {"cover_total": cov.get("total"), "cover_male": cov.get("male"), "cover_female": cov.get("female"),
               "cover_third_gender": cov.get("third_gender"), "summary_total": summ.get("total"),
               "printed_pages": meta.get("printed_pages"), "pages": meta.get("pages")}
    match = cov.get("total") is not None and n == cov["total"] and (summ.get("total") is None or n == summ["total"])
    gender_match = (cov.get("total") is not None and g["M"] == cov.get("male") and g["F"] == cov.get("female")
                    and g["T"] == cov.get("third_gender"))
    return {"records": n, "printed": printed, "match": bool(match), "gender_match": bool(gender_match),
            "by_gender": {"M": g["M"], "F": g["F"], "T": g["T"], "unknown": g[None]}}


def download_all(hub, project, keys, dest, threads=8):
    """Bucket objects of `project` through the hub's read links, 8 at a time. {key: local path or None}."""
    urls = {}
    for i in range(0, len(keys), 100):
        urls.update(hub.read_urls(project, keys[i:i + 100]))
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=threads))

    def one(k):
        path = os.path.join(dest, k.replace("/", "__"))
        for attempt in range(4):
            try:
                r = s.get(urls[k], timeout=120)
                if r.status_code == 404:
                    return k, None
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                return k, path
            except (requests.RequestException, KeyError):
                time.sleep(5 * (attempt + 1))
        return k, None

    with cf.ThreadPoolExecutor(threads) as ex:
        return dict(ex.map(one, keys))


def do_item(hub, item, fetch=None, upload=None, workers=WORKERS):
    """fetch(keys, dest) -> {key: path or None}; upload(files) (default: hub.upload_many). Returns the hub record."""
    t0 = time.time()
    p = item["payload"]
    roll, ac, lang = p["roll"]["id"], int(p["ac"]), p["lang"]
    parser = parser_of(lang)
    keys = sorted(p["keys"], key=lambda k: (part_of(k) or 0, k))
    extra = {int(x) for x in p.get("extra_parts") or []}
    fetch = fetch or (lambda ks, dest: download_all(hub, p["src_project"], ks, dest))
    upload = upload or hub.upload_many
    recs, summ = [], []
    with tempfile.TemporaryDirectory() as tmp:
        got = fetch(keys, tmp)
        missing = [k for k in keys if not got.get(k)]
        todo = [k for k in keys if got.get(k)]
        with cf.ProcessPoolExecutor(workers) if workers > 1 else _Inline() as ex:
            futs = {k: ex.submit(parse_one, got[k], lang) for k in todo}
            for k in todo:
                part = part_of(k)
                try:
                    r, meta = futs[k].result()
                except Exception as e:  # a broken PDF: listed, the AC goes on
                    summ.append({"key": k, "file": os.path.basename(k), "part": part, "error": f"{type(e).__name__}: {e}"[:300],
                                 "match": False})
                    continue
                rows = [normalise(x, item, part, meta, part in extra) for x in r]
                s = {"key": k, "file": meta.get("file"), "part": part, "parser": parser, **pdf_check(meta, rows),
                     "ok": bool(meta["validation"].get("ok")), "seconds": meta.get("seconds"),
                     "repairs": meta.get("repairs"), "meta": {x: meta[x] for x in meta if x not in ("validation", "repairs", "seconds")},
                     "validation": meta["validation"]}
                if part in extra:
                    s["extra_part"] = True
                summ.append(s)
                recs += rows
                if hub is not None:
                    hub.metric("pdfs_parsed")
                    hub.metric("pages", meta.get("pages", 0))
                try:
                    os.remove(got[k])
                except OSError:
                    pass
    from parquet_task import TEXT_SCHEMA, to_table, pq
    import io
    buf = io.BytesIO()
    pq.write_table(to_table(recs, TEXT_SCHEMA, roll, ac), buf, compression="zstd", compression_level=9, row_group_size=131072)
    pqb = buf.getvalue()
    hints = [hint(r, parser) for r in recs if r.get("epic")]
    files = [(f"text/{roll}/{ac}.jsonl.gz", gz_jsonl(recs), "application/gzip"),
             (f"hints/{roll}/{ac}.jsonl.gz", gz_jsonl(hints), "application/gzip"),
             (f"text/{roll}/{ac}.summary.json", json.dumps(summ, ensure_ascii=False, default=str).encode(), "application/json"),
             (f"parquet/{roll}/ac_{ac}/data.parquet", pqb, "application/vnd.apache.parquet")]
    upload(files)
    ok = [s for s in summ if s.get("match")]
    rec = {"state": p["state"], "roll": roll, "ac": ac, "lang": lang, "parser": parser, "src_project": p["src_project"],
           "pdfs": len(keys), "pdfs_parsed": len(summ) - sum(1 for s in summ if "error" in s), "records": len(recs),
           "hints": len(hints), "printed_total": sum((s.get("printed") or {}).get("cover_total") or 0 for s in summ),
           "match_pdfs": len(ok), "gender_match_pdfs": sum(1 for s in summ if s.get("gender_match")),
           "ok_pdfs": sum(1 for s in summ if s.get("ok")),
           "mismatch": [{"part": s["part"], "records": s.get("records"), "cover_total": (s.get("printed") or {}).get("cover_total"),
                         "summary_total": (s.get("printed") or {}).get("summary_total"), "error": s.get("error")}
                        for s in summ if not s.get("match")][:100],
           "missing_pdfs": missing[:100], "n_missing_pdfs": len(missing), "extra_parts": sorted(extra),
           "repairs": dict(sum((collections.Counter(s.get("repairs") or {}) for s in summ), collections.Counter())),
           "warnings": dict(collections.Counter(w.split(":")[0] for r in recs for w in r.get("text_warnings") or [])),
           "bytes_parquet": len(pqb), "seconds": round(time.time() - t0, 1)}
    if hub is not None:
        hub.record(item["key"], rec)
        if missing:
            hub.failure("missing-pdf", item["key"], f"{len(missing)} of {len(keys)} PDFs not in the bucket", keys=missing[:20])
        bad = [m for m in rec["mismatch"]]
        if bad:
            hub.failure("count-mismatch", item["key"], f"{len(bad)} of {len(summ)} PDFs: records differ from the printed total",
                        parts=bad[:20])
        hub.metric("ssr_records", len(recs))
    return rec


class _Inline:
    """ProcessPoolExecutor stand-in for workers=1 (tests, small runs)."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *a):
        f = cf.Future()
        try:
            f.set_result(fn(*a))
        except Exception as e:
            f.set_exception(e)
        return f


def main():
    from scrape_hub_client import HubClient, stop_flag
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
                hub.complete([it["key"]], "done", fetched=rec["records"])
            except Exception as e:  # one bad AC must not stop the worker
                hub.error()
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:400])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


def local(argv):
    """ssrtext_task.py --local OUTDIR STATE AC LANG ROLL_ID PDF [PDF ...]: one AC from local PDFs, files into OUTDIR (no hub)."""
    out, state, ac, lang, roll = argv[:5]
    pdfs = argv[5:]
    keys = [f"local/pdf/{roll}/{ac}/{os.path.basename(x)}" for x in pdfs]
    src = dict(zip(keys, pdfs))
    item = {"key": f"{roll}/{ac}", "payload": {"state": state, "ac": int(ac), "lang": lang, "roll": {"id": roll},
                                               "src_project": "local", "keys": keys}}

    def fetch(ks, dest):
        import shutil
        res = {}
        for k in ks:
            res[k] = os.path.join(dest, os.path.basename(k))
            shutil.copy(src[k], res[k])
        return res

    def upload(files):
        for path, data, _ in files:
            fp = os.path.join(out, path)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "wb") as f:
                f.write(data)

    rec = do_item(None, item, fetch=fetch, upload=upload)
    print(json.dumps(rec, ensure_ascii=False))
    return rec


if __name__ == "__main__":
    if sys.argv[1:2] == ["--local"]:
        local(sys.argv[2:])
    else:
        main()
