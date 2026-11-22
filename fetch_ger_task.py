"""Task `fetch-ger`: the General Election 2024 roll of one UP AC -> every part PDF once into the bucket.

Queue item (project eci-rolls-up-ger): key "2024-ge-final/<ac>", payload {"roll": "2024-ge-final", "ac": 174}.
This roll has no public file path: the portal gives it only through its captcha form. Per AC:
1. Part list (no captcha) -> parts/<roll>/<ac>.json.
2. Chunks of CHUNK parts: get a captcha, solve it with the offline model in captcha/ (top-5 guesses; a wrong answer leaves the
   captcha valid), call generate-published-pdfs at once -> one single-use file id per part (ids expire within minutes), and
   download all of them in parallel at once. A part whose id fails goes into the next chunk (new captcha). At most MAX_ROUNDS.
3. Each PDF: checked (%PDF- ... %%EOF), stored as pdf/<roll>/<ac>/<file name the portal gives>, one hub record per PDF.
Captchas are paced (>= CAPTCHA_GAP_S per worker): ECI answered 429 after ~15 captchas in 10 minutes from one IP in a burst test.
A 400 "Invalid Catpcha" after all guesses keeps the image (captcha-miss/...) for retraining.
"""
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import time

import requests
from scrape_hub_client import HubClient, stop_flag, timed_get, timed_post

import eci_api as E

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "captcha"))
from captcha_solver import solve  # noqa: E402

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "eci-rolls-up-ger")
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(900, _BUDGET / 4)
CHUNK = int(os.environ.get("ECI_GER_CHUNK", "60"))
CAPTCHA_GAP_S = float(os.environ.get("ECI_CAPTCHA_GAP_S", "40"))
MAX_ROUNDS = 12
MAX_PARTS = int(os.environ.get("ECI_MAX_PARTS", "0"))     # tests only
_local = threading.local()
_last_captcha = [0.0]


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(E.HEADERS)
    return _local.s


def part_list(hub, roll, ac):
    url, body = E.ger_part_list_request(roll, ac)
    for attempt in range(5):
        r = timed_post(hub, session(), url, json=body, timeout=60)
        if r is not None and r.status_code == 200 and r.json().get("payload") is not None:
            return r.json()["payload"]
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"part list failed for {roll} AC {ac}")


def file_ids(hub, roll, ac, district, parts):
    """One captcha (solved offline, up to 5 guesses) -> file ids (one per part the portal serves; sometimes fewer than asked)."""
    for attempt in range(4):
        wait = _last_captcha[0] + CAPTCHA_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_captcha[0] = time.time()
        r = timed_get(hub, session(), E.captcha_url(), headers={"appName": "VSP"}, timeout=30)
        if r is None or r.status_code != 200:
            time.sleep(30 * (attempt + 1))
            continue
        cid, jpg = E.captcha_decode(r.json()["data"])
        hub.metric("captchas")
        for guess in solve(jpg, k=5):
            url, body = E.generate_request(roll, ac, district, parts, guess, cid)
            g = timed_post(hub, session(), url, json=body, timeout=120)
            if g is None:
                continue
            if g.status_code == 200 and g.json().get("payload"):
                ids = g.json()["payload"]
                hub.metric("captcha_ok")
                if len(ids) != len(parts):
                    hub.metric("generate_short", len(parts) - len(ids))
                return ids                  # the part of each file comes from its file name (refId), not from the order
            if g.status_code == 400 and "aptcha" in g.text:
                hub.metric("captcha_wrong_guess")
                continue
            raise RuntimeError(f"generate: HTTP {g.status_code} {g.text[:120]}")
        hub.metric("captcha_miss")
        hub.upload_many([(f"captcha-miss/{int(time.time())}-{hashlib.md5(jpg).hexdigest()[:8]}.jpg", jpg, "image/jpeg")])
    raise RuntimeError("no captcha accepted in 4 tries")


def get_file(hub, fid):
    r = timed_get(hub, session(), E.file_url(fid), headers={"CurrentRole": "citizen", "Accept": "application/json"}, timeout=180)
    if r is None or r.status_code != 200:
        return None, None, (r.status_code if r is not None else "no answer")
    j = r.json()
    if not j.get("payload"):
        return None, None, (j.get("message") or "empty")
    b = base64.b64decode(j["payload"])
    return (b, j.get("refId"), "") if E.is_pdf(b) else (None, None, "not a PDF")


def do_item(hub, item):
    roll, ac = item["payload"]["roll"], int(item["payload"]["ac"])
    plist = sorted(part_list(hub, roll, ac), key=lambda p: int(p["partNumber"]))
    if MAX_PARTS:
        plist = plist[:MAX_PARTS]
    hub.upload_many([(f"parts/{roll}/{ac}.json", json.dumps({"roll": roll, "ac": ac, "fetched": E.now(), "parts": plist},
                                                           ensure_ascii=False).encode(), "application/json")])
    byno = {int(p["partNumber"]): p for p in plist}
    district = plist[0]["districtCd"]
    todo = sorted(byno)
    if (item.get("attempts") or 0) > 1:                     # retry: skip parts already in the bucket
        todo = [n for n in todo if not stored(hub, roll, ac, n)]
    stored_n, rounds, why, absent = 0, 0, {}, {}
    part_re = re.compile(r"-HIN-(\d+)-WI\.pdf$")
    while todo and rounds < MAX_ROUNDS and time.time() < DEADLINE:
        rounds += 1
        chunk, todo = todo[:CHUNK], todo[CHUNK:]
        ids = file_ids(hub, roll, ac, district, chunk)
        got = set()

        def one(fid):
            b, name, err = get_file(hub, fid)
            if b is None:
                return None, str(err)
            m = part_re.search(name or "")
            n = int(m.group(1)) if m else None
            if n not in byno:
                return None, f"unexpected file {name}"
            up = hub.upload_many([(f"pdf/{roll}/{ac}/{name}", b, "application/pdf")])[0]
            p = byno[n]
            hub.record(f"{roll}/{ac}/{n}", {"roll": roll, "ac": ac, "part": n, "part_name": p.get("partName"), "district": p.get("districtCd"),
                                            "part_id": p.get("partId"), "file": name, "key": up["key"], "bytes": len(b),
                                            "sha256": hashlib.sha256(b).hexdigest(), "fetched": E.now()})
            hub.metric("pdf_bytes", len(b))
            return n, ""

        threads = max(1, min(16, int(hub.settings.get("threads", 8))))
        errs = []
        with cf.ThreadPoolExecutor(threads) as ex:
            for n, err in ex.map(one, ids):
                if n is not None:
                    got.add(n)
                    stored_n += 1
                elif err:
                    errs.append(err)
        for n in chunk:
            if n in got:
                why.pop(n, None)
                continue
            if len(ids) < len(chunk) and not errs:      # the portal did not give a file for this part
                absent[n] = absent.get(n, 0) + 1
            why[n] = errs[0] if errs else "no file id"
            if absent.get(n, 0) >= 3:
                hub.failure("pdf", f"{roll}/{ac}/{n}", "the portal gives no file for this part (3 tries)")
                continue
            todo.append(n)                               # next chunk, new captcha and new ids
    if todo:
        raise RuntimeError(f"{len(todo)} parts not fetched after {rounds} rounds ({stored_n} stored); e.g. "
                           + ", ".join(f"{n}: {why.get(n, '?')}" for n in todo[:3]))
    hub.record(f"{roll}/{ac}", {"roll": roll, "ac": ac, "parts": len(plist), "stored_now": stored_n, "rounds": rounds,
                                "part_numbers": sorted(byno)})
    return stored_n


def stored(hub, roll, ac, n):
    """Is part n's PDF in the bucket already? The portal's file name is only known after a download, so look for the record's
    object through the parts JSON naming rule (2024-FC-EROLLGEN-S24-<ac>-FinalRoll-Revision2-HIN-<n>-WI.pdf)."""
    e = E.GER[roll]
    k = f"{PROJECT}/pdf/{roll}/{ac}/{e['year']}-{e['gen']}-S24-{ac}-{e['ref']}-Revision{e['rev']}-HIN-{n}-WI.pdf"
    try:
        url = hub.read_urls(PROJECT, [k])[k]
        return requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30).status_code in (200, 206)
    except Exception:
        return False


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
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:300])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
