"""Task `fetch`: one roll of one UP assembly constituency (AC) -> every part PDF once into the bucket.

Queue item (project eci-rolls-up): key "<roll>/<ac>", payload {"roll": "2026-sir-draft", "ac": 174}.
1. Part list from the portal API (one call; see eci_api.py). Saved as parts/<roll>/<ac>.json.
2. Each part PDF from its public path, checked (starts with %PDF-, ends with %%EOF, Content-Length matches), stored as
   pdf/<roll>/<ac>/<file name as the portal names it>. One hub record per PDF: <roll>/<ac>/<part> with bytes, sha256, the
   server's Last-Modified and the part (polling station) name. PDFs already stored are skipped (hub.known).
A missing PDF (HTTP 404) is a failure entry, not a stop. 403/429 are counted as blocks; the hub then slows the project down.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time

import requests
from scrape_hub_client import HubClient, stop_flag, timed_get, timed_post

import eci_api as E

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "eci-rolls-up")
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(900, _BUDGET / 4)   # an AC takes ~5-15 min
MAX_PARTS = int(os.environ.get("ECI_MAX_PARTS", "0"))
# Some GitHub runner locations get ECI's CDN at 10-100x lower speed than others from their first request (measured 2026-11-22:
# 4 s per 8 MB PDF on most runners, 10-370 s on 128.85.x). A worker that is slow hands its AC back and stops fetching in
# this runner job (the marker file), so the dispatcher gives the AC to another runner; this runner can still do OCR.
SLOW_BPS = float(os.environ.get("ECI_SLOW_BPS", "300000"))
SLOW_AFTER = 8
SLOW_MARK = "/tmp/eci-rolls-slow-runner"
_local = threading.local()


class SlowRunner(Exception):
    pass


def leave_pool():
    """A slow runner ends its pool worker (SIGTERM to the parent: the runner's own "finish" path), so the hub's fetch slot is
    free for a fast runner. Exiting alone is not enough: the hub would hand this runner a fetch slot again every minute
    (seen 2026-11-22: 20 of 32 slots held by slow runners). The worker runs only this family's work."""
    import signal
    print("slow runner for voters.eci.gov.in: ending this pool worker so a faster runner gets the slot", flush=True)
    try:
        os.kill(os.getppid(), signal.SIGTERM)
    except OSError:
        pass
    time.sleep(5)


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(E.HEADERS)
    return _local.s


def part_list(hub, roll, ac):
    url, body = E.part_list_request(roll, ac)
    for attempt in range(5):
        r = timed_post(hub, session(), url, json=body, timeout=60)
        if r is not None and r.status_code == 200:
            d = r.json()
            if d.get("payload") is not None:
                return d["payload"]
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"part list failed for {roll} AC {ac}")


def get_pdf(hub, roll, ac, part, speeds):
    url = E.pdf_url(roll, ac, part)
    for attempt in range(4):
        t = time.time()
        r = timed_get(hub, session(), url, timeout=180)
        if r is None:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None, "404"
        if r.status_code != 200:
            time.sleep(15 * (attempt + 1))
            continue
        b = r.content
        cl = r.headers.get("Content-Length")
        if (cl and int(cl) != len(b)) or not E.is_pdf(b):
            hub.metric("bad_pdf")
            time.sleep(5)
            continue
        speeds.append(len(b) / max(0.01, time.time() - t))
        return b, r.headers.get("Last-Modified", "")
    return None, "gave up"


def stored_already(hub, roll, ac, ids):
    """On a retry: the parts whose PDF is already in the bucket (1-byte read through the hub's short-lived links)."""
    keys = {f"{PROJECT}/pdf/{roll}/{ac}/{E.pdf_name(roll, ac, ids[rid]['partNumber'])}": rid for rid in ids}
    urls = {}
    ks = list(keys)
    for i in range(0, len(ks), 100):
        urls.update(hub.read_urls(PROJECT, ks[i:i + 100]))
    s = requests.Session()
    have = set()
    for k, rid in keys.items():
        try:
            if s.get(urls[k], headers={"Range": "bytes=0-0"}, timeout=30).status_code in (200, 206):
                have.add(rid)
        except requests.RequestException:
            pass
    return have


def do_item(hub, item):
    roll, ac = item["payload"]["roll"], int(item["payload"]["ac"])
    parts = sorted(part_list(hub, roll, ac), key=lambda p: int(p["partNumber"]))
    if MAX_PARTS:                       # tests only (scrape-check): the first N parts
        parts = parts[:MAX_PARTS]
    hub.upload_many([(f"parts/{roll}/{ac}.json", json.dumps({"roll": roll, "ac": ac, "fetched": E.now(), "parts": parts},
                                                           ensure_ascii=False).encode(), "application/json")])
    ids = {f"{roll}/{ac}/{int(p['partNumber'])}": p for p in parts}
    have = stored_already(hub, roll, ac, ids) if (item.get("attempts") or 0) > 1 else set()
    todo = [(rid, p) for rid, p in ids.items() if rid not in have]
    hub.metric("pdfs_skipped_known", len(ids) - len(todo))
    stored, speeds, gave_up, slow = 0, [], [], threading.Event()

    def one(rid_p):
        if slow.is_set():
            return 0
        rid, p = rid_p
        part = int(p["partNumber"])
        b, info = get_pdf(hub, roll, ac, part, speeds)
        if len(speeds) >= SLOW_AFTER and sorted(speeds[-SLOW_AFTER:])[SLOW_AFTER // 2] < SLOW_BPS:
            slow.set()
        if b is None:
            hub.failure("pdf", rid, f"no PDF ({info})", url=E.pdf_url(roll, ac, part))
            if info != "404":
                gave_up.append(rid)
            return 0
        name = E.pdf_name(roll, ac, part)
        up = hub.upload_many([(f"pdf/{roll}/{ac}/{name}", b, "application/pdf")])[0]
        hub.record(rid, {"roll": roll, "ac": ac, "part": part, "part_name": p.get("partName"), "district": p.get("districtCd"),
                         "part_id": p.get("partId"), "file": name, "key": up["key"], "bytes": len(b),
                         "sha256": hashlib.sha256(b).hexdigest(), "last_modified": info, "fetched": E.now()})
        hub.metric("pdf_bytes", len(b))
        return 1

    threads = max(1, min(16, int(hub.settings.get("threads", 4))))
    with cf.ThreadPoolExecutor(threads) as ex:
        for n in ex.map(one, todo):
            stored += n
            delay = hub.settings.get("delay_ms", 0)
            if delay:
                time.sleep(delay / 1000)
    if slow.is_set():
        med = sorted(speeds[-SLOW_AFTER:])[SLOW_AFTER // 2]
        hub.metric("slow_runner")
        raise SlowRunner(f"slow runner: median {med / 1e3:.0f} kB/s over the last {SLOW_AFTER} PDFs; {stored} stored, AC handed back")
    if gave_up:
        raise RuntimeError(f"{len(gave_up)} PDFs not fetched (retry later), {stored} stored")
    hub.record(f"{roll}/{ac}", {"roll": roll, "ac": ac, "parts": len(parts), "stored_now": stored, "known_before": len(ids) - len(todo),
                                "part_numbers": [int(p["partNumber"]) for p in parts]})
    return stored


def main():
    stop = stop_flag()
    if os.path.exists(SLOW_MARK):          # this runner was too slow for the CDN earlier in this job
        leave_pool()
        return
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
            except SlowRunner as e:
                hub.complete([it["key"]], "failed", error=str(e)[:200])
                open(SLOW_MARK, "w").write(str(e))
                hub.close("finished", "slow runner: leaving the pool")
                leave_pool()
                return
            except Exception as e:  # one bad AC must not stop the worker
                hub.error()
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:200])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
