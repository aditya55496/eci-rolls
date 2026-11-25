"""Planner for task ocr-any (and parquet-any) of the Hindi-state SIR rolls (2026-11-25). Runs on the hub server (reads the hub's
Postgres, read-only, through psql); writes the items as JSON for seeding, like plan_hi.py.

  plan_hi_ocr.py OUT.json [--dry-run] [--states S12,S07] [--acs 1,2] [--parts 1,10]
                 [--hint-set v4a] [--nohint-set j2Ps] [--state-set S07=j2Psf,S25=j2Ps] [--hints-project eci-rolls-ssr-text]
                 [--assume-hints] [--include-fetching] [--include-partial] [--strict-hints] [--resets OUT.sql]
  plan_hi_ocr.py OUT.json --parquet [--states ...] [--resets OUT.sql]   parquet-any items for ACs whose ocr-any items are all done;
                 --resets also writes SQL for parquet items older than an OCR item of their AC (seed_queue keeps done keys)

OCR items (project eci-rolls-hi-ocr): one per part of one AC of one state, both SIR rolls when both exist. Parts = the PDFs
stored in the bucket (one hub record "<roll id>/<ac>/<part>" per PDF in eci-rolls-hi-cdn / eci-rolls-hi-form, with its key);
this includes the extra parts the probe above the part list found and leaves out the listed parts that have no file. The AC
record ("<roll id>/<ac>": part_numbers, extra_parts, missing) is the cross-check: listed parts without a PDF and PDFs not in
the record are counted in the summary (2026-11-25: the probe_only pass rewrote "missing" as [], so listed parts without a
file show up there, e.g. S12 DR 122 parts).
Model set by state: the states with a 2027 SSR roll (S04 S12 S20 S26 S28) get --hint-set with the hints key of task ssrtext
(<hints project>/hints/<state>-2027-DR/<ac>.jsonl.gz); the others get --nohint-set and hints null. An AC of a hint state is
planned only when its ssrtext record exists and its hints are complete (no 2027 PDF missing from the bucket, every PDF
parsed; --strict-hints also holds ACs whose record counts differ from the printed totals; see ssr_done). --assume-hints skips
that check, for tests. Running the planner again later adds the rest (seed_queue keeps existing keys).
Listed parts without a PDF get no item; the output summary lists them per roll and AC (listed_parts_without_pdf) for the
part-gap follow-up.
An AC whose fetch item of one of the state's SIR rolls is not done (Bihar's form route, 2026-11-25: 108 + 75 failed) is held
too (an item seeded now would keep only the PDFs stored so far); --include-fetching plans it anyway.
Held states (HOLD): a state there is planned only when --state-set names its set (2026-11-25: S07 Haryana, j2Ps flags 22-50 %
of the records, j2Psf 3-7 % in ACs 1 3 12 40 66 but 22-33 % in AC 88, 40-48 % in AC 28, 41-56 % in AC 78, no gold run of j2Psf; S25 West Bengal, Nepali print, 78 % flagged).
Partial parts: a part whose PDF of one of the state's SIR rolls is missing although that roll's AC record lists the part (or
the roll's fetch of the AC is not done) is held (seed_queue keeps keys: an item seeded with one roll never gets the other);
--include-partial seeds it with the rolls stored now. --resets OUT.sql: for items already in the queue whose stored rolls are
now more than the item's, SQL that gives the item the new payload and sets it pending (the main session runs it; done items
are OCR'd again for that part).
Lang: from the AC record ('lang', default HIN); ACs in another language are counted and not planned (ocr-any reads HIN only).
Partition order: the no-hint states first, then the hint states; within a state by AC and part (an AC finishes together);
the partition id is the same on every run: RANK[state] x 1,000,000 + AC x 1,000 + part. RANK is a fixed table (a new state
gets the next free rank; never renumber).
--dry-run prints the counts per state and writes nothing.
"""
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hi_common as H  # noqa: E402

DB = os.environ.get("SCRAPE_HUB_DB", "scrape_hub")
SRC_PROJECTS = ("eci-rolls-hi-cdn", "eci-rolls-hi-form")
# SIR rolls of the Hindi states (plan_hi roll list 2026-11-24: rollTypeRefId SIR-DraftRoll / SIR-FinalRoll, pdfGenType
# EROLLGEN). UP (S24) is done (eci-rolls-up-ocr). Not here: West Bengal S25-2026-FIR-2 (FC-EROLLGEN print, form route) and
# bye-election rolls; Himachal (S08) and J&K (U08) have no Hindi SIR roll. Bihar's SIR is 2025 (form route).
SIR = {
    "S07": ["S07-2026-DR"], "S27": ["S27-2026-DR"], "U05": ["U05-2026-DR"], "U02": ["U02-2026-DR", "U02-2026-FIR"],
    "S25": ["S25-2026-DR", "S25-2026-FIR"],
    "S12": ["S12-2026-DR", "S12-2026-FIR"], "S20": ["S20-2026-DR", "S20-2026-FIR"], "S26": ["S26-2026-DR", "S26-2026-FIR"],
    "S28": ["S28-2026-DR", "S28-2026-SIR-FIR-1"], "S04": ["S04-2025-SIR-DR", "S04-2025-SIR-FIR"],
}
# Partition rank per state: fixed (a dict order change or a new state must not change the ids of seeded items).
RANK = {"S07": 0, "S27": 1, "U05": 2, "U02": 3, "S25": 4, "S12": 5, "S20": 6, "S26": 7, "S28": 8, "S04": 9}
# States planned only when --state-set names their set (the main session / owner decides; see the docstring).
HOLD = {"S07": "j2Ps flags 22-58 % (old-format EPICs); j2Psf 3-7 % in ACs 1 3 12 40 66 but 22-33 % in AC 88, 40-48 % in AC 28 "
               "and 41-56 % in AC 78 (names printed with a missing-glyph box, read '' or '0', flagged), not gold-run: "
               "choose --state-set S07=j2Ps|j2Psf",
        "S25": "Nepali print: 78 % of records flagged with j2Ps; hold until a suited set exists (--state-set S25=SET to plan)"}
HINT_ROLL = {"S04": "S04-2027-DR", "S12": "S12-2027-DR", "S20": "S20-2027-DR", "S26": "S26-2027-DR", "S28": "S28-2027-DR"}
NAMES = {"S04": "Bihar", "S07": "Haryana", "S12": "Madhya Pradesh", "S20": "Rajasthan", "S25": "West Bengal (Hindi ACs)",
         "S26": "Chhattisgarh", "S27": "Jharkhand", "S28": "Uttarakhand", "U02": "Chandigarh", "U05": "Delhi"}


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def ints(s):
    return {int(x) for x in s.split(",") if x.strip()} if s else None


def psql_json(sql):
    """Rows of a read-only query as dicts (one JSON object per line)."""
    r = subprocess.run(["psql", "-X", "-q", "-At", "-d", DB, "-v", "ON_ERROR_STOP=1", "-c", "SET default_transaction_read_only = on",
                        "-c", "SET statement_timeout = '600s'", "-c", f"SELECT row_to_json(t) FROM ({sql}) t"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"psql failed: {r.stderr.strip()[-400:]}")
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]


def lit(s):
    return "'" + str(s).replace("'", "''") + "'"


def prefix_cond(roll):
    """record_id starts with '<roll>/', as a range the primary key serves. The database collation (en_US) ignores '/' and '-'
    at the first level, so the upper bound is '<roll>/zzzz' (letters sort after the digits that follow) and starts_with()
    keeps the result exact."""
    p = roll + "/"
    return f" AND record_id >= {lit(p)} AND record_id < {lit(p + 'zzzz')} AND starts_with(record_id, {lit(p)})"


def roll_rows(roll):
    """Per-PDF and per-AC hub records of one roll (range on the records primary key: project IN (...), record_id prefix)."""
    return psql_json(
        "SELECT project, record_id, data->>'key' AS key, (data->>'ac')::int AS ac, (data->>'part')::int AS part, "
        "data->'part_numbers' AS part_numbers, data->'extra_parts' AS extra_parts, data->'missing' AS missing, data->>'lang' AS lang "
        f"FROM records WHERE project IN ({', '.join(lit(p) for p in SRC_PROJECTS)}) "
        + prefix_cond(roll))


def fetch_status(roll):
    """{ac: queue status} of the fetch items "<roll id>/<ac>" (queue primary key range, as prefix_cond)."""
    p = roll + "/"
    rows = psql_json(f"SELECT key, status FROM queue WHERE project IN ({', '.join(lit(x) for x in SRC_PROJECTS)}) "
                     f"AND key >= {lit(p)} AND key < {lit(p + 'zzzz')} AND starts_with(key, {lit(p)})")
    return {int(r["key"].split("/")[1]): r["status"] for r in rows if r["key"].count("/") == 1 and r["key"].split("/")[1].isdigit()}


def ssr_done(project, roll, strict=False):
    """-> (ACs whose 2027 hints are complete, {ac: reason} for ACs whose ssrtext record exists but whose hints are not complete).
    ssrtext writes its record (record id '<roll id>/<ac>') also when PDFs of the 2027 roll are missing from the bucket
    (n_missing_pdfs, failure 'missing-pdf') or did not parse (pdfs_parsed < pdfs). OCR made with such partial hints would keep
    more flags and fewer EPIC confirmations / repairs, and seed_queue keeps the keys (no second OCR): those ACs are held.
    A count mismatch against the printed totals (match_pdfs < pdfs) holds the AC only with strict (--strict-hints); the
    planner lists those ACs either way (ocrv5 review 2026-11-25)."""
    rows = psql_json(f"SELECT record_id, (data->>'pdfs')::int AS pdfs, (data->>'pdfs_parsed')::int AS parsed, "
                     f"(data->>'n_missing_pdfs')::int AS missing, (data->>'match_pdfs')::int AS match FROM records "
                     f"WHERE project = {lit(project)} " + prefix_cond(roll))
    ready, bad = set(), {}
    for r in rows:
        if r["record_id"].count("/") != 1 or not r["record_id"].split("/")[1].isdigit():
            continue
        ac, why = int(r["record_id"].split("/")[1]), []
        if r["missing"] is None or r["missing"] > 0:
            why.append(f"missing_pdfs {r['missing']}")
        if r["pdfs"] is None or r["parsed"] is None or r["parsed"] < r["pdfs"]:
            why.append(f"parsed {r['parsed']} of {r['pdfs']}")
        if r["pdfs"] is not None and r["match"] is not None and r["match"] < r["pdfs"]:
            why.append(f"count_mismatch {r['pdfs'] - r['match']} of {r['pdfs']} PDFs" + ("" if strict else " (not held)"))
        held = [w for w in why if not w.endswith("(not held)")]
        if held:
            bad[ac] = "; ".join(why)
        else:
            ready.add(ac)
            if why:
                bad[ac] = "; ".join(why)
    return ready, bad


def plan_ocr(states, acs_only, parts_only, hint_set, nohint_set, hints_project, assume_hints, include_fetching=False,
             state_set=None, include_partial=False, strict_hints=False):
    state_set = state_set or {}
    for s in [hint_set, nohint_set] + list(state_set.values()):
        if s not in H.MODEL_SETS:
            raise SystemExit(f"model set {s} not in model_sets.json")
    items, summ = [], {}
    order = sorted(SIR, key=lambda st: RANK[st])
    for st in order:
        if states and st not in states:
            continue
        hinted = st in HINT_ROLL
        mset = state_set.get(st) or (hint_set if hinted else nohint_set)
        if not H.compatible(H.MODEL_SETS[mset], hinted):
            raise SystemExit(f"model set {mset} ({H.MODEL_SETS[mset].get('hints')} hints) does not fit {st} (hints: {hinted})")
        ready, hint_issues = None, {}
        if hinted and not assume_hints:
            ready, hint_issues = ssr_done(hints_project, HINT_ROLL[st], strict_hints)
        by_ac = collections.defaultdict(lambda: collections.defaultdict(dict))     # ac -> part -> {roll: key}
        extra = collections.defaultdict(lambda: collections.defaultdict(set))      # ac -> part -> {roll}
        expect = collections.defaultdict(lambda: collections.defaultdict(set))     # ac -> part -> {roll whose AC record lists it}
        lang = {}                                                                   # ac -> lang of its AC records
        s = {"name": NAMES.get(st), "model_set": mset, "rolls": {}}
        fetching = set()      # ACs with a fetch item of one of the state's SIR rolls that is not done (still downloading)
        roll_fetching = {}    # roll -> ACs whose fetch item of that roll is not done
        gaps = {}             # roll -> {ac: listed parts without a stored PDF} (no item; follow-up: part-gap work)
        for roll in SIR[st]:
            roll_fetching[roll] = {ac for ac, stt in fetch_status(roll).items() if stt != "done"}
            fetching |= roll_fetching[roll]
            rows = roll_rows(roll)
            pdf = [r for r in rows if r["record_id"].count("/") == 2 and r["key"]]
            acr = {r["ac"]: r for r in rows if r["record_id"].count("/") == 1}
            have = collections.defaultdict(set)
            for r in pdf:
                if acs_only and r["ac"] not in acs_only:
                    continue
                have[r["ac"]].add(r["part"])
                if not parts_only or r["part"] in parts_only:
                    by_ac[r["ac"]][r["part"]][roll] = r["key"]
            listed_not_stored = stored_not_listed = 0
            for ac, a in acr.items():
                if acs_only and ac not in acs_only:
                    continue
                want = (set(a.get("part_numbers") or []) | set(a.get("extra_parts") or [])) - set(a.get("missing") or [])
                gap = sorted(int(x) for x in want - have.get(ac, set()))
                listed_not_stored += len(gap)
                if gap:
                    gaps.setdefault(roll, {})[ac] = gap
                stored_not_listed += len(have.get(ac, set()) - want)
                for x in a.get("extra_parts") or []:
                    extra[ac][int(x)].add(roll)
                for x in want:
                    expect[ac][int(x)].add(roll)
                if a.get("lang"):
                    lang.setdefault(ac, set()).add(a["lang"])
            s["rolls"][roll] = {"acs": len(have), "pdfs": sum(len(v) for v in have.values()), "ac_records": len(acr),
                                "listed_not_stored": listed_not_stored, "stored_not_listed": stored_not_listed,
                                "langs": sorted({r["lang"] for r in acr.values() if r.get("lang")})}
        held = held_fetch = held_partial = held_lang = held_hint_incomplete = 0
        n_items = n_pdfs = 0
        state_items = []
        for ac in sorted(by_ac):
            if ac in fetching and not include_fetching:
                # an item seeded now would keep only the PDFs stored so far (seed_queue keeps existing keys): wait for the fetch
                held_fetch += len(by_ac[ac])
                continue
            if hinted and ready is not None and ac not in ready:
                if ac in hint_issues:
                    held_hint_incomplete += len(by_ac[ac])
                else:
                    held += len(by_ac[ac])
                continue
            langs = lang.get(ac) or {"HIN"}
            if langs != {"HIN"}:
                held_lang += len(by_ac[ac])
                continue
            for part in sorted(by_ac[ac]):
                keys = by_ac[ac][part]
                want = expect[ac][part] | {r for r in SIR[st] if ac in roll_fetching[r]}
                if set(keys) < want and not include_partial:
                    held_partial += 1       # a listed roll has no PDF yet: the item would never get it (seed_queue keeps keys)
                    continue
                pl = {"state": st, "ac": ac, "part": part, "lang": "HIN", "keys": dict(sorted(keys.items())),
                      "hints": f"{hints_project}/hints/{HINT_ROLL[st]}/{ac}.jsonl.gz" if hinted else None,
                      "model_set": mset, "v": H.PAYLOAD_V}
                ex = sorted(extra[ac][part] & set(keys))
                if ex:
                    pl["extra_rolls"] = ex
                # stable partition id (the same on every planner run, with or without --states): fixed state rank (no-hint
                # states first), then AC, then part; parts < 1000, ACs < 1000
                state_items.append({"key": f"{st}/{ac}/{part}", "partition_id": RANK[st] * 1_000_000 + ac * 1000 + part, "payload": pl})
                n_items += 1
                n_pdfs += len(keys)
        on_hold = st in HOLD and st not in state_set
        if not on_hold:
            items += state_items
        s.update(acs=len(by_ac), items=n_items, pdfs=n_pdfs, items_waiting_for_hints=held, items_waiting_for_fetch=held_fetch,
                 items_partial_held=held_partial, items_other_lang=held_lang, items_hints_incomplete=held_hint_incomplete,
                 hint_issues={str(a): w for a, w in sorted(hint_issues.items()) if a in by_ac},
                 listed_parts_without_pdf={r: {str(a): v for a, v in sorted(g.items())} for r, g in gaps.items()}, acs_fetch_not_done=len(fetching & set(by_ac)),
                 acs_with_hints=(len(ready & set(by_ac)) if ready is not None else ("assumed" if hinted else None)),
                 held_state=HOLD[st] if on_hold else None)
        summ[st] = s
    return items, summ


def resets_sql(project, items, also=()):
    """SQL for items already in the queue of `project` whose planned rolls are more than the queued payload's (or whose key is
    in `also`: out-of-date parquet items): new payload, status pending, attempts 0. Returns (sql text, {status: count})."""
    rows = psql_json(f"SELECT key, status, payload FROM queue WHERE project = {lit(project)}")
    q = {r["key"]: r for r in rows}
    out, n = [], collections.Counter()
    for it in items:
        r = q.get(it["key"])
        if not r:
            continue
        old = set((r["payload"] or {}).get("keys") or (r["payload"] or {}).get("rolls") or [])
        new = set(it["payload"].get("keys") or it["payload"].get("rolls") or [])
        if new > old or it["key"] in also:
            n[r["status"]] += 1
            out.append(f"UPDATE queue SET payload = {lit(json.dumps(it['payload'], ensure_ascii=False))}::jsonb, status = 'pending', "
                       f"attempts = 0, claimed_by = NULL, last_error = '' WHERE project = {lit(project)} AND key = {lit(it['key'])} "
                       f"AND status <> 'claimed';")
    return ("BEGIN;\n" + "\n".join(out) + "\nCOMMIT;\n") if out else "", dict(n)


def plan_parquet(states):
    """parquet-any items: one per AC whose ocr-any items are all done (queue of eci-rolls-hi-ocr). Also the parquet items that
    are out of date: an OCR item of the AC finished after the AC's parquet item (OCR again with a new set, a key reset, or a
    repair-any rewrite is NOT seen here: reset those by hand). Returns (items, summary, stale keys)."""
    rows = psql_json(f"SELECT key, status, payload->'keys' AS keys, extract(epoch FROM done_at) AS done_at FROM queue "
                     f"WHERE project = {lit(H.OCR_HI_PROJECT)}")
    pq_done = {r["key"]: r["done_at"] for r in psql_json(
        f"SELECT key, extract(epoch FROM done_at) AS done_at FROM queue WHERE project = {lit(H.PARQUET_HI_PROJECT)} AND status = 'done'")}
    by = collections.defaultdict(list)
    for r in rows:
        if r["key"].count("/") != 2:      # form-route items '<state>/<roll group>/<ac>/<part>' (plan_form_ocr.py --parquet)
            continue
        st, ac, part = r["key"].split("/")
        if states and st not in states:
            continue
        by[(st, int(ac))].append((int(part), r["status"], sorted((r["keys"] or {}).keys()), r["done_at"]))
    items, summ, stale = [], collections.Counter(), []
    for (st, ac), parts in sorted(by.items()):
        if all(s == "done" for _, s, _, _ in parts):
            rolls = sorted({x for _, _, ks, _ in parts for x in ks}, key=lambda r: (0 if H.is_draft(r) else 1, r))
            key = f"{st}/{ac}"
            items.append({"key": key, "partition_id": RANK[st] * 1000 + ac,
                          "payload": {"state": st, "ac": ac, "parts": sorted(p for p, _, _, _ in parts), "rolls": rolls}})
            summ[st + " ready"] += 1
            if key in pq_done and max(d or 0 for _, _, _, d in parts) > (pq_done[key] or 0):
                stale.append(key)
                summ[st + " parquet out of date"] += 1
        else:
            summ[st + " not ready"] += 1
    return items, dict(summ), stale


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    dry = "--dry-run" in sys.argv
    states = set(arg("--states").split(",")) if arg("--states") else None
    if "--parquet" in sys.argv:
        items, summ, stale = plan_parquet(states)
        print(json.dumps(summ, indent=1))
        project = H.PARQUET_HI_PROJECT
    else:
        state_set = dict(kv.split("=", 1) for kv in arg("--state-set").split(",")) if arg("--state-set") else {}
        items, summ = plan_ocr(states, ints(arg("--acs")), ints(arg("--parts")), arg("--hint-set", "v4a"), arg("--nohint-set", "j2Ps"),
                               arg("--hints-project", H.SSR_TEXT_PROJECT), "--assume-hints" in sys.argv,
                               "--include-fetching" in sys.argv, state_set, "--include-partial" in sys.argv,
                               "--strict-hints" in sys.argv)
        print(f"{'state':5} {'name':24} {'set':6} {'ACs':>5} {'items':>7} {'PDFs':>7} {'w-hint':>6} {'w-fetch':>7} {'partial':>7}  rolls (ACs / PDFs / listed parts without a PDF / PDFs not in the AC record)")
        for st, s in summ.items():
            rl = "; ".join(f"{r} {v['acs']}/{v['pdfs']}/{v['listed_not_stored']}/{v['stored_not_listed']}" for r, v in s["rolls"].items())
            print(f"{st:5} {str(s['name'])[:24]:24} {s['model_set']:6} {s['acs']:5} {s['items']:7} {s['pdfs']:7} {s['items_waiting_for_hints']:6} "
                  f"{s['items_waiting_for_fetch']:7} {s['items_partial_held']:7}  {rl}" + (f"  other lang: {s['items_other_lang']}" if s['items_other_lang'] else ""))
            if s["held_state"]:
                print(f"      HELD (not in the output): {s['held_state']}")
            if s["hint_issues"]:
                print(f"      2027 hints not complete ({s['items_hints_incomplete']} items held): "
                      + "; ".join(f"AC {a}: {w}" for a, w in list(s["hint_issues"].items())[:12])
                      + (" ..." if len(s["hint_issues"]) > 12 else ""))
        print(f"total items {len(items)}, PDFs {sum(len(i['payload']['keys']) for i in items)}")
        project = H.OCR_HI_PROJECT
    if arg("--resets"):
        sql, n = resets_sql(project, items, stale if "--parquet" in sys.argv else ())
        print(f"resets for items already queued in {project}: {n or 'none'}")
        if sql and not dry:
            open(arg("--resets"), "w").write(sql)
            print("wrote", arg("--resets"))
    if out and not dry:
        json.dump({"items": items, "summary": summ}, open(out, "w"), ensure_ascii=False, indent=0)
        print("wrote", out)


if __name__ == "__main__":
    main()
