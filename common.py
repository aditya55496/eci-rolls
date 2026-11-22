"""Shared names for the eci-rolls family: rolls, bucket keys, the 2027 hint conversion, downloads through hub read links."""
import gzip
import json
import os
import time

import requests

FETCH_PROJECT = "eci-rolls-up"          # PDFs as the portal serves them: <FETCH_PROJECT>/pdf/<roll>/<ac>/<file name>
TEXT_PROJECT = "eci-rolls-up-2027"      # 2027 SSR draft roll from its text layer: text/<ac>.jsonl.gz, hints/<ac>.jsonl.gz
OCR_PROJECT = "eci-rolls-up-ocr"        # OCR of the 4 image rolls: ocr/<ac>/<part>.jsonl.gz (+ .summary.json)

# The 5 UP rolls in scope (owner 2026-11-22: "the 5 tested rolls", Hindi only). id -> roll id on the portal.
ROLLS = {
    "2026-sir-draft": "S24-2026-DR",
    "2026-sir-final": "S24-2026-FIR-1",
    "2026-supp-2": "S24-2026-SUPP-2",
    "2026-supp-3": "S24-2026-SUPP-3",
    "2027-ssr-draft": "S24-2027-DR",
}
OCR_ROLLS = ["2026-sir-draft", "2026-sir-final", "2024-ge-final"]   # order matters: the draft's readings are a cache for the final


def pdf_key(roll, ac, name):
    return f"{FETCH_PROJECT}/pdf/{roll}/{int(ac)}/{name}"


def hint_of(r):
    """A parse_textlayer record -> the --hint-2027 record eci_ocr expects (same as bench hint_2027_v2.jsonl; checked equal on
    869 of 869 records of AC 174 part 1)."""
    rt = r.get("relation_type") or ""
    return {"epic": r["epic"], "name": r.get("name"), "relation_label": (rt + " का नाम") if rt and rt != "अन्य" else rt,
            "relation_name": r.get("relation_name"), "house_no": r.get("house_no"), "age": str(r.get("age")), "gender": r.get("gender"),
            "serial_no": str(r.get("serial_no")), "ac": str(r["ac"]), "part": str(r["part"])}


def gz_jsonl(rows):
    return gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode(), mtime=0)


def download(hub, target, keys, dest, session=None):
    """Download bucket objects of project `target` through the hub's short-lived links. Returns {key: local path or None}."""
    s = session or requests.Session()
    urls = {}
    for i in range(0, len(keys), 100):
        urls.update(hub.read_urls(target, keys[i:i + 100]))
    out = {}
    for k in keys:
        path = os.path.join(dest, k.replace("/", "__"))
        for attempt in range(4):
            try:
                r = s.get(urls[k], timeout=120)
                if r.status_code == 404:
                    out[k] = None
                    break
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                out[k] = path
                break
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        else:
            out[k] = None
    return out
