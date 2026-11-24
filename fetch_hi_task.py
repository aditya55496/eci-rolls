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
("<roll id>/<ac>/<part>": bytes, sha256, part name) and one per AC. A retry (or payload "recheck": true) skips parts already
in the bucket. Payload "check_only": true: the printed-totals check of the PDFs already in the bucket, plus the language check
and the probe above the list (fetches nothing that is stored). Payload "probe_only": true (ACs done before the part-list fix): no listed part is fetched; only the language check
and the probe above the list run.

Part-list gap (checked 2026-11-24): get-publish-part-list ignores the roll and gives the AC's CURRENT part list. A roll can have
parts above that list (UP SIR 2026: ~4,015 parts in 294 ACs; Gujarat, Kerala and Uttarakhand SIR 2026 seen too) or fewer parts
(Bihar 2025 rolls: 12-15 % of the listed parts have no file). So after the listed parts the task probes higher part numbers until
EXTRA_MISS numbers in a row have no file, stores what it finds and writes "extra_parts" and "missing" into the AC record. Part
names / ids in the records come from the current list: for an older roll they can belong to a different polling station.
Language (cdn): the planned language is checked on the public path first; if it has no file there, the first language that
has one is used (2026-11-24: Tamil Nadu and Puducherry SIR 2026 rolls exist only in Tamil, Lakshadweep and Kerala ACs 1-3 only in
Malayalam on the public path, although the discovery table lists English / Kannada). The AC record says "lang_used".

Completeness (AC record "completeness"), not against the part list: every PDF is checked against what it prints itself
(printcheck.py: page total in the footer = pages in the file, part number in the header = the file's part, cover electors
total = the summary's net total); the AC's stored part numbers must run 1..highest without a hole, and the probe above the
highest must end in EXTRA_MISS numbers without a file. "electors_printed" is the sum of the printed totals. The PDFs do not
print how many parts the AC has. Per-part results: printcheck/<roll id>/<ac>.json in the bucket.
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
import printcheck

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
EXTRA_MISS = int(os.environ.get("ECI_EXTRA_MISS", "5"))      # numbers in a row without a file that end the probe above the list
FORM_PROBE = 30                                               # part numbers per probe request on the form route
# completeness check against the totals printed in each PDF (printcheck.py; Tesseract, ~0.3 s CPU per PDF). On by default where
# Tesseract is installed (the runners); the local form units on dev-vps-3 set ECI_PRINT_CHECK=0 (no bulk OCR on that server):
# their ACs are checked later with payload "check_only" on the runners.
PRINT_CHECK = os.environ.get("ECI_PRINT_CHECK", "1") == "1" and printcheck.available()
LANGS = ["ENG", "HIN", "TAM", "MAL", "KAN", "TEL", "BEN", "MAR", "GUJ", "ORI", "PAN", "ASM", "URD"]
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


def upload(hub, files):
    """hub.upload_many, riding out a busy bucket: B2 answers 503 under heavy load (2026-11-24, ~470 MB/s from the pool); the
    client's own retries are short, so wait 20 / 40 / 80 s and try again before giving up on the file."""
    for wait in (20, 40, 80, None):
        try:
            return hub.upload_many(files)
        except RuntimeError as e:
            if wait is None or "HTTP 503" not in str(e):
                raise
            hub.metric("upload_503_wait")
            time.sleep(wait)


def print_check(hub, part, b):
    """printcheck.check() reduced to what the records keep; None when the check is off."""
    if not PRINT_CHECK:
        return None
    try:
        c = printcheck.check(b, part, hub.lang_used)
    except Exception as e:  # a PDF the reader cannot open is a fault of the file, not of the task
        c = {"ok": False, "error": f"{type(e).__name__}: {e}"[:120]}
    c = {k: c.get(k) for k in ("ok", "pages", "part", "totals", "electors", "pdf_pages", "pages_printed", "pages_printed_last", "error")
         if c.get(k) is not None}
    hub.checks[part] = c
    hub.metric("print_check_bad", 0 if c["ok"] else 1)
    return c


def store(hub, roll, state, ac, p, name, b, extra=None):
    part = int(p["partNumber"])
    up = upload(hub, [(f"pdf/{roll['id']}/{ac}/{name}", b, "application/pdf")])[0]
    c = print_check(hub, part, b)
    hub.record(f"{roll['id']}/{ac}/{part}", {"state": state, "roll": roll["id"], "ac": ac, "part": part, "part_name": p.get("partName"),
                                             "district": p.get("districtCd"), "part_id": p.get("partId"), "file": name, "key": up["key"],
                                             "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "fetched": E.now(),
                                             **({"printed": c} if c else {}), **(extra or {})})
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


def head_ok(url):
    """True / False for a file / no file (404) on the public path; None when the site does not answer clearly."""
    for attempt in range(3):
        try:
            r = session().head(url, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (200, 404):
            return r.status_code == 200
        time.sleep(10 * (attempt + 1))
    return None


def cdn_lang(hub, roll, state, ac, lang, nums):
    """The language the public path serves for this AC roll: the planned one, else the first language with a file."""
    sample = sorted({nums[0], nums[len(nums) // 2], nums[-1]})
    unclear = False
    for lg in [lang] + [x for x in LANGS if x != lang]:
        ans = [head_ok(E.roll_pdf_url(roll, state, ac, n, lg)) for n in sample]
        if any(ans):
            if lg != lang:
                hub.metric("lang_fallback")
            return lg
        unclear |= None in ans
    if unclear:
        raise RuntimeError("public path did not answer clearly while checking the language (retry later)")
    return lang


def check_stored(hub, roll, state, ac, lang, byno, key, cdn_probe):
    """check_only: the printed-totals check of the listed parts already in the bucket; parts not in the bucket are "missing";
    then the probe above the list as in a normal run (cdn: do_cdn in probe_only mode; form: do_form in probe_only mode)."""
    keys = {n: key(n) for n in sorted(byno)}
    urls = {}
    ks = list(keys.values())
    for i in range(0, len(ks), 100):
        urls.update(hub.read_urls(PROJECT, ks[i:i + 100]))

    def one(n):
        for attempt in range(3):
            try:
                r = requests.get(urls[keys[n]], timeout=180)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 404:
                return n, None
            if r.status_code == 200:
                return n, r.content
            time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"bucket read failed for part {n} (retry later)")

    missing, seen = [], 0
    with cf.ThreadPoolExecutor(4) as ex:
        for n, b in ex.map(one, sorted(byno)):
            if b is None:
                missing.append(n)
            else:
                print_check(hub, n, b)
                seen += 1
    hub.probe_only = True
    hub.check_only = False
    parts = [byno[n] for n in sorted(byno)]
    stored, known, info = (do_cdn if cdn_probe else do_form)(hub, roll, state, ac, lang, parts)
    info["missing"] = sorted(missing)
    return stored, known, info


def do_cdn(hub, roll, state, ac, lang, parts):
    byno = {int(p["partNumber"]): p for p in parts}
    lang = hub.lang_used = cdn_lang(hub, roll, state, ac, lang, sorted(byno))
    key = lambda n: f"{PROJECT}/pdf/{roll['id']}/{ac}/{E.roll_pdf_name(roll, state, ac, n, lang)}"  # noqa: E731
    todo = sorted(byno)
    if hub.check_only:
        return check_stored(hub, roll, state, ac, lang, byno, key, cdn_probe=True)
    if hub.probe_only:
        todo = []
    elif hub.skip_known:
        have = in_bucket(hub, [key(n) for n in todo])
        todo = [n for n in todo if key(n) not in have]
    known = len(byno) - len(todo)
    hub.metric("pdfs_skipped_known", known)
    stored, speeds, gave_up, missing, slow = [0], [], [], [], threading.Event()

    def one(n, p=None):
        """'ok' (stored), '404' (no file) or 'err'."""
        if slow.is_set():
            return "err"
        url = E.roll_pdf_url(roll, state, ac, n, lang)
        b, info = get_cdn(hub, url, speeds)
        if len(speeds) >= SLOW_AFTER and sorted(speeds[-SLOW_AFTER:])[SLOW_AFTER // 2] < SLOW_BPS:
            slow.set()
        if b is None:
            if p is not None or info != "404":       # a probe above the list that finds nothing is not a failure
                hub.failure("pdf", f"{roll['id']}/{ac}/{n}", f"no PDF ({info})", url=url)
            if info != "404":
                gave_up.append(n)
            return "404" if info == "404" else "err"
        store(hub, roll, state, ac, p or {"partNumber": n}, E.roll_pdf_name(roll, state, ac, n, lang), b,
              {"last_modified": info, **({} if p else {"extra_part": True})})
        stored[0] += 1
        return "ok"

    threads = max(1, min(16, int(hub.settings.get("threads", 8))))
    with cf.ThreadPoolExecutor(threads) as ex:
        for n, res in zip(todo, ex.map(lambda n: one(n, byno[n]), todo)):
            if res == "404":
                missing.append(n)
        # the part list is the AC's current list, not this roll's: probe above it until EXTRA_MISS numbers in a row have no file
        extra, top, miss = [], max(byno), 0
        while miss < EXTRA_MISS and not slow.is_set() and not gave_up and top < max(byno) + 3000:
            batch = list(range(top + 1, top + 1 + max(EXTRA_MISS, threads)))
            have = in_bucket(hub, [key(n) for n in batch]) if hub.skip_known else set()
            res = list(ex.map(lambda n: "known" if key(n) in have else one(n), batch))
            for n, r in zip(batch, res):
                miss = miss + 1 if r == "404" else 0
                if r in ("ok", "known"):
                    extra.append(n)
                known += r == "known"
            top = batch[-1]
    if slow.is_set():
        hub.metric("slow_runner")
        raise SlowRunner(f"slow runner: {stored[0]} stored, AC handed back")
    if gave_up:
        raise RuntimeError(f"{len(gave_up)} PDFs not fetched (retry later), {stored[0]} stored")
    hub.metric("extra_parts", len(extra))
    return stored[0], known, {"lang_used": lang, "extra_parts": extra, "missing": sorted(missing)}


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
    hub.lang_used = lang
    guess = lambda n: f"{PROJECT}/pdf/{roll['id']}/{ac}/{E.roll_pdf_name(roll, state, ac, n, lang)}"  # noqa: E731
    if hub.check_only:
        return check_stored(hub, roll, state, ac, lang, byno, guess, cdn_probe=False)
    if hub.probe_only:
        todo = []
    elif hub.skip_known:      # the portal's file names follow this rule for the rolls seen so far; a miss only means a re-download
        have = in_bucket(hub, [guess(n) for n in todo])
        todo = [n for n in todo if guess(n) not in have]
    known = len(byno) - len(todo)
    hub.metric("pdfs_skipped_known", known)
    stored_n, rounds, why, absent, missing = 0, 0, {}, {}, []
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
                missing.append(n)
                continue
            todo.append(n)
    if todo:
        raise RuntimeError(f"{len(todo)} parts not fetched after {rounds} rounds ({stored_n} stored); e.g. "
                           + ", ".join(f"{n}: {why.get(n, '?')}" for n in todo[:3]))
    # the part list is the AC's current list, not this roll's: ask for FORM_PROBE numbers above it (one captcha); go on while
    # files come back near the top of the probed range. The portal answers only for numbers that have a file.
    extra, top = [], max(byno)
    while time.time() < DEADLINE:
        probe = [n for n in range(top + 1, top + 1 + FORM_PROBE)]
        if hub.skip_known:
            have = in_bucket(hub, [guess(n) for n in probe])
            extra += [n for n in probe if guess(n) in have]
            known += sum(guess(n) in have for n in probe)
            probe = [n for n in probe if guess(n) not in have]
        got = set()
        for fid in file_ids(hub, roll, state, ac, district, probe, lang) if probe else []:
            b, name, err = get_file(hub, fid)
            m = part_re.search(name or "")
            if b is None or not m or int(m.group(1)) not in probe:
                raise RuntimeError(f"probe above the list: {err or 'unexpected file ' + str(name)} (retry later)")
            n = int(m.group(1))
            store(hub, roll, state, ac, {"partNumber": n, "districtCd": district}, name, b, {"extra_part": True})
            got.add(n)
            stored_n += 1
        extra += sorted(got)
        found = [n for n in extra if n > top]
        if not found or max(found) <= top + FORM_PROBE - EXTRA_MISS:
            break
        top += FORM_PROBE
    hub.metric("extra_parts", len(extra))
    return stored_n, known, {"lang_used": lang, "extra_parts": sorted(extra), "missing": sorted(missing)}


def completeness(hub, parts, info, mode):
    """The AC's completeness against what the PDFs print (see the module doc). "complete" is None when the printed check did
    not run on every PDF of the AC (check off, or a probe_only run)."""
    listed = {int(x["partNumber"]) for x in parts}
    have = sorted((listed - set(info["missing"])) | set(info["extra_parts"]))
    holes = [n for n in range(1, (have[-1] if have else 0) + 1) if n not in set(have)]
    C = hub.checks
    bad = sorted(n for n, c in C.items() if not c.get("ok"))
    unread = {k: sum(1 for c in C.values() if c.get(k) == "unread") for k in ("pages", "part", "totals")}
    checked_all = bool(have) and set(have) <= set(C)      # a retry or probe_only run does not re-read PDFs stored earlier
    return {"parts_with_file": len(have), "highest_part": have[-1] if have else None, "holes": holes[:200], "n_holes": len(holes),
            "printed_checked": len(C), "printed_bad": bad[:200], "n_printed_bad": len(bad), "printed_unread": unread,
            "electors_printed": sum(c.get("electors") or 0 for c in C.values()),
            "electors_unread": sum(1 for c in C.values() if not c.get("electors")),
            "complete": (not holes and not bad) if checked_all else None}


def do_item(hub, item):
    p = item["payload"]
    roll, state, ac, lang, route = p["roll"], p["state"], int(p["ac"]), p.get("lang", "HIN"), p["route"]
    hub.attempts = item.get("attempts") or 0
    hub.skip_known = hub.attempts > 1 or bool(p.get("recheck"))
    hub.probe_only = bool(p.get("probe_only"))    # an AC done before this change: only the language check and the probe above the list
    hub.check_only = bool(p.get("check_only"))    # printed-totals check of the stored PDFs (+ language check and probe)
    hub.checks, hub.lang_used = {}, lang
    parts = sorted(part_list(hub, roll, state, ac), key=lambda x: int(x["partNumber"]))
    if MAX_PARTS:
        parts = parts[:MAX_PARTS]
    if not parts:
        hub.record(f"{roll['id']}/{ac}", {"state": state, "roll": roll["id"], "ac": ac, "parts": 0, "note": "empty part list"})
        return 0
    upload(hub, [(f"parts/{roll['id']}/{ac}.json", json.dumps({"state": state, "roll": roll, "ac": ac, "fetched": E.now(),
                                                                   "parts": parts}, ensure_ascii=False).encode(), "application/json")])
    mode = "check_only" if hub.check_only else "probe_only" if hub.probe_only else "fetch"
    stored, known, info = (do_cdn if route == "cdn" else do_form)(hub, roll, state, ac, lang, parts)
    info["completeness"] = completeness(hub, parts, info, mode)
    if hub.checks:
        upload(hub, [(f"printcheck/{roll['id']}/{ac}.json", json.dumps({"roll": roll, "state": state, "ac": ac, "lang_used": info["lang_used"],
                     "mode": mode, "checked": E.now(), "parts": {str(k): v for k, v in sorted(hub.checks.items())}}).encode(), "application/json")])
    hub.record(f"{roll['id']}/{ac}", {"state": state, "roll": roll["id"], "ac": ac, "lang": lang, "route": route, "parts": len(parts),
                                      "stored_now": stored, "known_before": known, "part_numbers": [int(x["partNumber"]) for x in parts],
                                      **info, "mode": mode, **({"probe_only": True} if mode == "probe_only" else
                                                 {"files": len(parts) - len(info["missing"]) + len(info["extra_parts"])}),
                                      "part_list": "current list of the AC (the portal ignores the roll)", "checked": E.now()})
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
