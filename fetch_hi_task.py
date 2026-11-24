"""Task `fetch-hi`: one roll of one assembly constituency (AC), any state -> every part PDF once into the bucket (owner 2026-11-24:
every roll of the states that publish Hindi rolls, in Hindi).

Queue item: key "<roll id>/<ac>", payload {"state": "S12", "ac": 1, "lang": "HIN", "route": "cdn" | "form",
"roll": {"id": "S12-2026-DR", "year": 2026, "ref": "SIR-DraftRoll", "gen": "EROLLGEN", "rev": 1}}.
- route "cdn" (SIR draft/final 2026, SSR draft 2027; any IP, the GitHub runners): the portal's public file path, as task `fetch`
  for UP (checks: %PDF- ... %%EOF, Content-Length; a runner that gets the CDN slowly hands the AC back and leaves the pool).
- route "form" (every other roll; India IPs only: the file server gateway-vpd answers only from India): the captcha form, as
  task `fetch-ger` for UP (offline captcha solver, one captcha per chunk of parts, paced captchas, single-use file ids
  downloaded at once, parts without a file id go into the next chunk).
Output: parts/<roll id>/<ac>.json (the part list), pdf/<roll id>/<ac>/<file name the portal gives>, one hub record per PDF
("<roll id>/<ac>/<part>": bytes, sha256, part name) and one per AC. A retry skips parts already in the bucket.
"""
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time

import requests
from scrape_hub_client import HubClient, stop_flag, timed_get, timed_post

import eci_api as E

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "captcha"))
from captcha_solver import solve  # noqa: E402

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "eci-rolls-hi-cdn")
_BUDGET = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 3600
DEADLINE = time.time() + _BUDGET - min(900, _BUDGET / 4)
CHUNK = int(os.environ.get("ECI_FORM_CHUNK", "200"))
CAPTCHA_GAP_S = float(os.environ.get("ECI_CAPTCHA_GAP_S", "45"))
MAX_ROUNDS = 12
MAX_PARTS = int(os.environ.get("ECI_MAX_PARTS", "0"))     # tests only (scrape-check): the first N parts
SLOW_BPS = float(os.environ.get("ECI_SLOW_BPS", "300000"))
SLOW_AFTER = 8
SLOW_MARK = "/tmp/eci-rolls-slow-runner"
_local = threading.local()
_last_captcha = [0.0]


class SlowRunner(Exception):
    pass


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(E.HEADERS)
    return _local.s


def leave_pool():
    print("slow runner for voters.eci.gov.in: ending this pool worker so a faster runner gets the slot", flush=True)
    try:
        os.kill(os.getppid(), signal.SIGTERM)
    except OSError:
        pass
    time.sleep(5)


def part_list(hub, roll, state, ac):
    url, body = E.roll_part_list_request(roll, state, ac)
    for attempt in range(5):
        r = timed_post(hub, session(), url, json=body, timeout=60)
        if r is not None and r.status_code == 200:
            d = r.json()
            if d.get("payload") is not None:
                return d["payload"]
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"part list failed for {roll['id']} AC {ac}")


def in_bucket(hub, keys):
    """The keys already in the bucket (1-byte reads through the hub's short-lived links)."""
    urls, have, s = {}, set(), requests.Session()
    for i in range(0, len(keys), 100):
        urls.update(hub.read_urls(PROJECT, keys[i:i + 100]))
    for k in keys:
        try:
            if s.get(urls[k], headers={"Range": "bytes=0-0"}, timeout=30).status_code in (200, 206):
                have.add(k)
        except requests.RequestException:
            pass
    return have


def store(hub, roll, state, ac, p, name, b, extra=None):
    part = int(p["partNumber"])
    up = hub.upload_many([(f"pdf/{roll['id']}/{ac}/{name}", b, "application/pdf")])[0]
    hub.record(f"{roll['id']}/{ac}/{part}", {"state": state, "roll": roll["id"], "ac": ac, "part": part, "part_name": p.get("partName"),
                                             "district": p.get("districtCd"), "part_id": p.get("partId"), "file": name, "key": up["key"],
                                             "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "fetched": E.now(), **(extra or {})})
    hub.metric("pdf_bytes", len(b))


# ---------------------------------------------------------------- route cdn
def get_cdn(hub, url, speeds):
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
        b, cl = r.content, r.headers.get("Content-Length")
        if (cl and int(cl) != len(b)) or not E.is_pdf(b):
            hub.metric("bad_pdf")
            time.sleep(5)
            continue
        speeds.append(len(b) / max(0.01, time.time() - t))
        return b, r.headers.get("Last-Modified", "")
    return None, "gave up"


def do_cdn(hub, roll, state, ac, lang, parts):
    names = {int(p["partNumber"]): E.roll_pdf_name(roll, state, ac, int(p["partNumber"]), lang) for p in parts}
    byno = {int(p["partNumber"]): p for p in parts}
    todo = sorted(byno)
    if hub.attempts > 1:
        have = in_bucket(hub, [f"{PROJECT}/pdf/{roll['id']}/{ac}/{names[n]}" for n in todo])
        todo = [n for n in todo if f"{PROJECT}/pdf/{roll['id']}/{ac}/{names[n]}" not in have]
    hub.metric("pdfs_skipped_known", len(byno) - len(todo))
    stored, speeds, gave_up, slow = 0, [], [], threading.Event()

    def one(n):
        if slow.is_set():
            return 0
        url = E.roll_pdf_url(roll, state, ac, n, lang)
        b, info = get_cdn(hub, url, speeds)
        if len(speeds) >= SLOW_AFTER and sorted(speeds[-SLOW_AFTER:])[SLOW_AFTER // 2] < SLOW_BPS:
            slow.set()
        if b is None:
            hub.failure("pdf", f"{roll['id']}/{ac}/{n}", f"no PDF ({info})", url=url)
            if info != "404":
                gave_up.append(n)
            return 0
        store(hub, roll, state, ac, byno[n], names[n], b, {"last_modified": info})
        return 1

    threads = max(1, min(16, int(hub.settings.get("threads", 8))))
    with cf.ThreadPoolExecutor(threads) as ex:
        for k in ex.map(one, todo):
            stored += k
    if slow.is_set():
        hub.metric("slow_runner")
        raise SlowRunner(f"slow runner: {stored} stored, AC handed back")
    if gave_up:
        raise RuntimeError(f"{len(gave_up)} PDFs not fetched (retry later), {stored} stored")
    return stored, len(byno) - len(todo)


# ---------------------------------------------------------------- route form
def file_ids(hub, roll, state, ac, district, parts, lang):
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
            url, body = E.roll_generate_request(roll, state, ac, district, parts, guess, cid, lang)
            g = timed_post(hub, session(), url, json=body, timeout=120)
            if g is None:
                continue
            if g.status_code == 200 and g.json().get("payload"):
                ids = g.json()["payload"]
                hub.metric("captcha_ok")
                if len(ids) != len(parts):
                    hub.metric("generate_short", len(parts) - len(ids))
                return ids
            if g.status_code == 400 and "aptcha" in g.text:
                hub.metric("captcha_wrong_guess")
                continue
            if g.status_code == 200 and "not found" in g.text.lower():
                return []                    # the portal has no file for these parts ("File not found")
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


def do_form(hub, roll, state, ac, lang, parts):
    byno = {int(p["partNumber"]): p for p in parts}
    district = parts[0]["districtCd"]
    todo = sorted(byno)
    if hub.attempts > 1:      # the portal's file names follow this rule for the rolls seen so far; a miss only means a re-download
        guess = {n: f"{PROJECT}/pdf/{roll['id']}/{ac}/{E.roll_pdf_name(roll, state, ac, n, lang)}" for n in todo}
        have = in_bucket(hub, list(guess.values()))
        todo = [n for n in todo if guess[n] not in have]
    hub.metric("pdfs_skipped_known", len(byno) - len(todo))
    stored_n, rounds, why, absent = 0, 0, {}, {}
    part_re = re.compile(rf"-{lang}-(\d+)-WI\.pdf$")
    threads = int(os.environ.get("ECI_FORM_THREADS") or 0) or max(1, min(16, int(hub.settings.get("threads", 8))))
    while todo and rounds < MAX_ROUNDS and time.time() < DEADLINE:
        rounds += 1
        chunk, todo = todo[:CHUNK], todo[CHUNK:]
        ids = file_ids(hub, roll, state, ac, district, chunk, lang)
        got, errs = set(), []

        def one(fid):
            b, name, err = get_file(hub, fid)
            if b is None:
                return None, str(err)
            m = part_re.search(name or "")
            n = int(m.group(1)) if m else None
            if n not in byno:
                return None, f"unexpected file {name}"
            store(hub, roll, state, ac, byno[n], name, b)
            return n, ""

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
            if len(ids) < len(chunk) and not errs:
                absent[n] = absent.get(n, 0) + 1
            why[n] = errs[0] if errs else "no file id"
            if absent.get(n, 0) >= 3:
                hub.failure("pdf", f"{roll['id']}/{ac}/{n}", "the portal gives no file for this part (3 tries)")
                continue
            todo.append(n)
    if todo:
        raise RuntimeError(f"{len(todo)} parts not fetched after {rounds} rounds ({stored_n} stored); e.g. "
                           + ", ".join(f"{n}: {why.get(n, '?')}" for n in todo[:3]))
    return stored_n, len(byno) - len(todo)


def do_item(hub, item):
    p = item["payload"]
    roll, state, ac, lang, route = p["roll"], p["state"], int(p["ac"]), p.get("lang", "HIN"), p["route"]
    hub.attempts = item.get("attempts") or 0
    parts = sorted(part_list(hub, roll, state, ac), key=lambda x: int(x["partNumber"]))
    if MAX_PARTS:
        parts = parts[:MAX_PARTS]
    if not parts:
        hub.record(f"{roll['id']}/{ac}", {"state": state, "roll": roll["id"], "ac": ac, "parts": 0, "note": "empty part list"})
        return 0
    hub.upload_many([(f"parts/{roll['id']}/{ac}.json", json.dumps({"state": state, "roll": roll, "ac": ac, "fetched": E.now(),
                                                                   "parts": parts}, ensure_ascii=False).encode(), "application/json")])
    stored, known = (do_cdn if route == "cdn" else do_form)(hub, roll, state, ac, lang, parts)
    hub.record(f"{roll['id']}/{ac}", {"state": state, "roll": roll["id"], "ac": ac, "lang": lang, "route": route, "parts": len(parts),
                                      "stored_now": stored, "known_before": known, "part_numbers": [int(x["partNumber"]) for x in parts]})
    return stored


def main():
    stop = stop_flag()
    if os.path.exists(SLOW_MARK) and "cdn" in PROJECT:
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
                hub.complete([it["key"]], "failed", error=f"{type(e).__name__}: {e}"[:300])
        hub.close("failed", "stopped by signal") if stop.is_set() else hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
