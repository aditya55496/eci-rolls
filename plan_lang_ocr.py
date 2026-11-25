"""Planner for task ocr-any (and parquet-any) of the ENGLISH and TAMIL SIR rolls (engocr, 2026-11-25). Runs on the hub server
(reads the hub's Postgres, read-only, through psql); writes the items as JSON for seeding, like plan_hi_ocr.py.

  plan_lang_ocr.py OUT.json [--dry-run] [--states S05,S22] [--acs 1,2] [--parts 1,10] [--project eci-rolls-all-ocr] [--include-partial]
                   [--state-set S05=engJ,S22=tamN2] [--release S25] [--wait-hints] [--tam-hints S22=ROLL,U07=ROLL]
                   [--new-only] [--include-fetching] [--partition-base 0] [--resets OUT.sql]
  plan_lang_ocr.py OUT.json --parquet [--states ...] [--resets OUT.sql]    parquet-any items (project eci-rolls-all-parquet) for the
                   (state, lang, AC) whose ocr-any items are all done

Source: eci-rolls-all-cdn (fetch-hi, CDN rolls of all other states): per-PDF hub records '<roll id>/<ac>/<part>' with the key,
AC records '<roll id>/<ac>' (lang_used = the language the portal served; part_numbers / extra_parts / missing).
Rolls: the SIR 2026 draft and final rolls of the states in ROLLS (one table; DR-only states have no final roll on the portal).
Language per AC: lang_used of its AC records, else the language in its PDF file names ('-ENG-', '-TAM-'); ENG and TAM are
planned, other languages (S25 Bengali ACs, U06 / U07 Malayalam, U07 Telugu) are counted and left out. An AC whose rolls name
different languages is held (counted). Items: one per part of one AC, draft + final when both PDFs are stored (the draft first:
its readings are the final roll's cache and, for engK11 / engJ, the source of draft_fill), else the one stored roll ('parts_one_roll').
The fetch of every roll is 'done' in eci-rolls-all-cdn (2026-11-25), so a missing PDF of one roll will not come later; an AC
whose fetch item of a roll is not done is held (--include-fetching plans it).
A part whose PDF of one roll is missing although that roll's AC record lists the part (a fetch gap: S22 has 5 draft and 3 final AC
records without any PDF) is held ('parts_partial_held'; listed in the summary for the part-gap work): seed_queue keeps keys, so an
item seeded with one roll would never get the other one. --include-partial plans it with the stored roll.
Model set: English with complete 2027 hints (S05 S14 U01: eci-rolls-ssr-text record '<state>-2027-DR/<ac>' with no missing and no
unparsed PDF, plan_hi_ocr.ssr_done) -> engD with the hints key; English without -> engK11 (an AC of a hint state whose hints are
not complete also gets the no-hint set; --wait-hints holds it instead). No-hint English set: engK11 (job4 round 3, adopted 10:20;
engJ = round 2 stays defined, --state-set STATE=engJ or a hub swap ocr_model_set_map {"engK11": "engJ"} uses it). Tamil -> tamN2; tamH2 only with --tam-hints STATE=ROLL (a Tamil
2027 roll in eci-rolls-ssr-text: none exists on 2026-11-25, its hints are the ENGLISH 2027 text, which cannot confirm Tamil
names) and complete hints for the AC. --state-set overrides the set of a state (it must fit the state's language and hints).
Held states (HOLD): planned only with --release STATE. 2026-11-25: S25 West Bengal English ACs (job4: final rolls without the
draft roll stay far below target).
Key '<state>/<lang>/<ac>/<part>' (lang ENG / TAM): never the same as a Hindi SIR key ('<state>/<ac>/<part>') or a form key
('<state>/<roll group>/<ac>/<part>', the group starts with the state code). Payload v 3 (hi_common.PAYLOAD_V): workers of older
code give the items back untried.
Partition id: --partition-base + RANK[state] x 1,000,000 + AC x 1,000 + part (fixed RANK table: hint states first, then the
English no-hint states by size, then Tamil; never renumber).
--new-only leaves out the (state, lang, AC) that have an item in the target project already (batches as more is planned).
--resets OUT.sql: SQL for queued items whose planned rolls are more than the queued payload's (plan_hi_ocr.resets_sql).
--dry-run prints the counts per state and writes nothing.
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hi_common as H  # noqa: E402
import plan_hi_ocr as P  # noqa: E402

SRC = "eci-rolls-all-cdn"
# SIR 2026 rolls on the portal's CDN path (queue of eci-rolls-all-cdn, 2026-11-25). States with only a draft roll there: S01 S13
# S17 S29. Not here: S03 (2027 only), S06 Gujarati, S11 Kerala, S18 Odisha, S19 Punjab (other scripts), U06 Lakshadweep (MAL).
ROLLS = {
    "U01": ["U01-2026-DR", "U01-2026-FIR"], "S05": ["S05-2026-DR", "S05-2026-FIR"], "S14": ["S14-2026-DR", "S14-2026-SIR-FIR-1"],
    "S02": ["S02-2026-DR", "S02-2026-SIR-FIR-1"], "S10": ["S10-2026-DR", "S10-2026-SIR-FIR-1"], "S13": ["S13-2026-DR"],
    "S15": ["S15-2026-DR", "S15-2026-SIR-FIR-1"], "S16": ["S16-2026-DR", "S16-2026-SIR-FIR-1"], "S17": ["S17-2026-DR"],
    "S21": ["S21-2026-DR", "S21-2026-SIR-FIR-1"], "S29": ["S29-2026-DR"], "U03": ["U03-2026-DR", "U03-2026-SIR-FIR-1"],
    "S01": ["S01-2026-DR"], "S22": ["S22-2026-DR", "S22-2026-FIR"], "U07": ["U07-2026-DR", "U07-2026-FIR"],
    "S25": ["S25-2026-DR", "S25-2026-FIR"],
}
RANK = {"U01": 0, "S05": 1, "S14": 2, "S02": 3, "S10": 4, "S13": 5, "S15": 6, "S16": 7, "S17": 8, "S21": 9, "S29": 10, "U03": 11,
        "S01": 12, "S22": 13, "U07": 14, "S25": 15}
NAMES = {"U01": "Andaman & Nicobar", "S05": "Goa", "S14": "Manipur", "S02": "Arunachal Pradesh", "S10": "Karnataka",
         "S13": "Maharashtra", "S15": "Meghalaya", "S16": "Mizoram", "S17": "Nagaland", "S21": "Sikkim", "S29": "Telangana",
         "U03": "Dadra & Nagar Haveli, Daman & Diu", "S01": "Andhra Pradesh", "S22": "Tamil Nadu", "U07": "Puducherry",
         "S25": "West Bengal (English ACs)"}
# English 2027 SSR roll in eci-rolls-ssr-text (task ssrtext, English parser): the hints of these states' English SIR rolls
ENG_HINT_ROLL = {"S05": "S05-2027-DR", "S14": "S14-2027-DR", "U01": "U01-2027-DR"}
SETS = {("ENG", True): "engD", ("ENG", False): "engK11", ("TAM", True): "tamH2", ("TAM", False): "tamN2"}
PLAN_LANGS = ("ENG", "TAM")
HOLD = {"S25": "job4 (2026-11-25 10:20, set K11): West Bengal FINAL rolls without their draft stay below target (held-out stamp set: "
               "97.52 % record / 10.14 % flagged; J 95.14 / 10.27); with the draft roll (draft_records) the English gold is 99.95 % "
               "record / 0.29 % flagged. --release S25 plans it (pairs read with draft_fill; final-only parts are held as partial "
               "parts unless --include-partial)"}
FILE_LANG = re.compile(r"-(HIN|ENG|TAM|BEN|MAL|TEL|KAN|GUJ|ORI|PAN|ASM|MAR|URD)-\d+-")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def kv(s):
    return dict(x.split("=", 1) for x in s.split(",") if x) if s else {}


def roll_rows(roll):
    """Per-PDF and per-AC hub records of one roll of eci-rolls-all-cdn (primary-key range: project, record_id prefix)."""
    return P.psql_json(
        "SELECT record_id, data->>'key' AS key, data->>'file' AS file, (data->>'ac')::int AS ac, (data->>'part')::int AS part, "
        "data->'part_numbers' AS part_numbers, data->'extra_parts' AS extra_parts, data->'missing' AS missing, "
        "data->>'lang_used' AS lang_used "
        f"FROM records WHERE project = {P.lit(SRC)}" + P.prefix_cond(roll))


def fetch_status(roll):
    """{ac: queue status} of the fetch items '<roll id>/<ac>' of eci-rolls-all-cdn."""
    p = roll + "/"
    rows = P.psql_json(f"SELECT key, status FROM queue WHERE project = {P.lit(SRC)} "
                       f"AND key >= {P.lit(p)} AND key < {P.lit(p + 'zzzz')} AND starts_with(key, {P.lit(p)})")
    return {int(r["key"].split("/")[1]): r["status"] for r in rows if r["key"].count("/") == 1 and r["key"].split("/")[1].isdigit()}


def queued_acs(project):
    """(state, lang, ac) that have an item in the project's queue (for --new-only)."""
    rows = P.psql_json(f"SELECT key FROM queue WHERE project = {P.lit(project)}")
    out = set()
    for r in rows:
        k = r["key"].split("/")
        if len(k) == 4 and k[1] in PLAN_LANGS and k[2].isdigit():
            out.add((k[0], k[1], int(k[2])))
    return out


def listed(a):
    """Parts an AC record lists (part_numbers + extra_parts - missing)."""
    if not a:
        return set()
    return {int(x) for x in (set(a.get("part_numbers") or []) | set(a.get("extra_parts") or [])) - set(a.get("missing") or [])}


def ac_lang(rows_by_roll, ac):
    """-> (lang or None, note): lang_used of the AC records, else the PDF file names' language; None when the rolls disagree."""
    langs = set()
    for rows in rows_by_roll.values():
        a = rows["acr"].get(ac)
        if a and a.get("lang_used"):
            langs.add(a["lang_used"])
        else:
            for f in rows["files"].get(ac, ()):
                m = FILE_LANG.search(f or "")
                if m:
                    langs.add(m.group(1))
    if len(langs) == 1:
        return langs.pop(), None
    return None, ("no language known" if not langs else "rolls in different languages: " + ",".join(sorted(langs)))


def plan(states=None, acs_only=None, parts_only=None, state_set=None, release=(), wait_hints=False, tam_hints=None,
         include_fetching=False, new_only=False, project=H.OCR_ALL_PROJECT, partition_base=0, hints_project=H.SSR_TEXT_PROJECT,
         include_partial=False):
    state_set, tam_hints = state_set or {}, tam_hints or {}
    for s in list(state_set.values()) + list(SETS.values()):
        if s not in H.MODEL_SETS:
            raise SystemExit(f"model set {s} not in model_sets.json")
    have_q = queued_acs(project) if new_only else set()
    items, summ = [], {}
    for st in sorted(ROLLS, key=lambda s: RANK[s]):
        if states and st not in states:
            continue
        hint_roll = ENG_HINT_ROLL.get(st)
        ready, hint_issues = set(), {}
        if hint_roll:
            ready, hint_issues = P.ssr_done(hints_project, hint_roll)
        tam_ready, tam_issues = set(), {}
        if st in tam_hints:
            tam_ready, tam_issues = P.ssr_done(hints_project, tam_hints[st])
        R = {}
        fetching = set()
        for roll in ROLLS[st]:
            fs = fetch_status(roll)
            fetching |= {ac for ac, s in fs.items() if s != "done"}
            rows = roll_rows(roll)
            pdf = [r for r in rows if r["record_id"].count("/") == 2 and r["key"]]
            acr = {r["ac"]: r for r in rows if r["record_id"].count("/") == 1}
            parts, files = collections.defaultdict(dict), collections.defaultdict(list)
            for r in pdf:
                if acs_only and r["ac"] not in acs_only:
                    continue
                parts[r["ac"]][r["part"]] = r["key"]
                files[r["ac"]].append(r["file"] or os.path.basename(r["key"]))
            R[roll] = {"pdf": parts, "acr": acr, "files": files}
        s = collections.OrderedDict(name=NAMES.get(st), rolls={}, langs=collections.Counter(), sets=collections.Counter(),
                                    items=0, pdfs=0, pairs=0, parts_one_roll=collections.Counter(), acs=0,
                                    parts_partial_held=collections.Counter(), partial_parts={},
                                    acs_other_lang={}, acs_lang_unknown={}, acs_fetch_not_done=0, items_fetch_held=0,
                                    acs_hints_incomplete={}, items_waiting_for_hints=0, acs_new_only_skipped=0,
                                    listed_parts_without_pdf={}, acs_records_without_pdf=[])
        all_acs = sorted({ac for rr in R.values() for ac in set(rr["pdf"]) | set(rr["acr"]) if not acs_only or ac in acs_only})
        state_items = []
        for roll, rr in R.items():
            s["rolls"][roll] = {"acs_with_pdf": len(rr["pdf"]), "pdfs": sum(len(v) for v in rr["pdf"].values()),
                                "ac_records": len(rr["acr"])}
            for ac, a in rr["acr"].items():
                if acs_only and ac not in acs_only:
                    continue
                want = (set(a.get("part_numbers") or []) | set(a.get("extra_parts") or [])) - set(a.get("missing") or [])
                gap = sorted(int(x) for x in want - set(rr["pdf"].get(ac, {})))
                if gap:
                    s["listed_parts_without_pdf"].setdefault(roll, {})[str(ac)] = gap if len(gap) <= 50 else gap[:50] + [f"... {len(gap)} parts"]
                if not rr["pdf"].get(ac):
                    s["acs_records_without_pdf"].append(f"{roll}/{ac}")
        for ac in all_acs:
            if not any(rr["pdf"].get(ac) for rr in R.values()):
                continue
            lang, note = ac_lang(R, ac)
            if lang is None:
                s["acs_lang_unknown"][str(ac)] = note
                continue
            s["langs"][lang] += 1
            if lang not in PLAN_LANGS:
                s["acs_other_lang"][str(ac)] = lang
                continue
            s["acs"] += 1
            nparts = len({p for rr in R.values() for p in rr["pdf"].get(ac, {})})
            if ac in fetching and not include_fetching:
                s["acs_fetch_not_done"] += 1
                s["items_fetch_held"] += nparts
                continue
            if new_only and (st, lang, ac) in have_q:
                s["acs_new_only_skipped"] += 1
                continue
            hints = None
            if lang == "ENG" and hint_roll:
                if ac in ready:
                    hints = f"{hints_project}/hints/{hint_roll}/{ac}.jsonl.gz"
                else:
                    s["acs_hints_incomplete"][str(ac)] = hint_issues.get(ac, "no ssrtext record")
                    if wait_hints:
                        s["items_waiting_for_hints"] += nparts
                        continue
            if lang == "TAM" and st in tam_hints:
                if ac in tam_ready:
                    hints = f"{hints_project}/hints/{tam_hints[st]}/{ac}.jsonl.gz"
                else:
                    s["acs_hints_incomplete"][str(ac)] = tam_issues.get(ac, "no ssrtext record")
                    if wait_hints:
                        s["items_waiting_for_hints"] += nparts
                        continue
            mset = state_set.get(st) or SETS[(lang, hints is not None)]
            if H.MODEL_SETS[mset].get("hints") == "none":
                hints = None        # --state-set with a no-hint set: the item carries no hints key (a hub swap cannot add hints)
            if not H.compatible(H.MODEL_SETS[mset], hints is not None, lang):
                raise SystemExit(f"model set {mset} does not fit {st} AC {ac} (lang {lang}, hints {hints is not None})")
            for part in sorted({p for rr in R.values() for p in rr["pdf"].get(ac, {})}):
                if parts_only and part not in parts_only:
                    continue
                keys = {roll: rr["pdf"][ac][part] for roll, rr in R.items() if part in rr["pdf"].get(ac, {})}
                lack = [roll for roll, rr in R.items() if roll not in keys and part in listed(rr["acr"].get(ac))]
                if lack and not include_partial:
                    for roll in lack:
                        s["parts_partial_held"][roll] += 1
                        s["partial_parts"].setdefault(roll, {}).setdefault(str(ac), []).append(part)
                    continue
                extra = sorted(roll for roll in keys if int(part) in set((R[roll]["acr"].get(ac) or {}).get("extra_parts") or []))
                pl = {"state": st, "ac": ac, "part": part, "lang": lang, "keys": dict(sorted(keys.items())), "hints": hints,
                      "model_set": mset, "v": H.PAYLOAD_V}
                if extra:
                    pl["extra_rolls"] = extra
                state_items.append({"key": f"{st}/{lang}/{ac}/{part}",
                                    "partition_id": partition_base + RANK[st] * 1_000_000 + ac * 1000 + part, "payload": pl})
                s["items"] += 1
                s["pdfs"] += len(keys)
                s["sets"][mset] += 1
                if len(keys) == 2:
                    s["pairs"] += 1
                elif len(ROLLS[st]) == 2:
                    s["parts_one_roll"][next(iter(keys))] += 1
        on_hold = st in HOLD and st not in release
        if not on_hold:
            items += state_items
        s["held_state"] = HOLD[st] if on_hold else None
        s["acs_hints_complete"] = len(ready) if hint_roll else None
        summ[st] = s
    return items, summ


def plan_parquet(states=None, ocr_project=H.OCR_ALL_PROJECT):
    """parquet-any items: one per (state, lang, AC) whose ocr-any items in the OCR project are all done. Key '<state>/<lang>/<ac>',
    payload {state, ac, lang, parts, rolls, ocr_project, v 3}. Also the keys of parquet items older than an OCR item of the AC
    (for --resets)."""
    rows = P.psql_json(f"SELECT key, status, payload->'keys' AS keys, extract(epoch FROM done_at) AS done_at FROM queue "
                       f"WHERE project = {P.lit(ocr_project)}")
    pq_done = {r["key"]: r["done_at"] for r in P.psql_json(
        f"SELECT key, extract(epoch FROM done_at) AS done_at FROM queue WHERE project = {P.lit(H.PARQUET_ALL_PROJECT)} AND status = 'done'")}
    by = collections.defaultdict(list)
    for r in rows:
        k = r["key"].split("/")
        if len(k) != 4 or k[1] not in PLAN_LANGS:
            continue
        if states and k[0] not in states:
            continue
        by[(k[0], k[1], int(k[2]))].append((int(k[3]), r["status"], sorted((r["keys"] or {}).keys()), r["done_at"]))
    items, summ, stale = [], collections.Counter(), []
    for (st, lang, ac), parts in sorted(by.items()):
        if all(x == "done" for _, x, _, _ in parts):
            rolls = sorted({x for _, _, ks, _ in parts for x in ks}, key=lambda r: (0 if H.is_draft(r) else 1, r))
            key = f"{st}/{lang}/{ac}"
            items.append({"key": key, "partition_id": RANK.get(st, 99) * 1000 + ac,
                          "payload": {"state": st, "ac": ac, "lang": lang, "parts": sorted(p for p, _, _, _ in parts), "rolls": rolls,
                                      "ocr_project": ocr_project, "v": H.PAYLOAD_V}})
            summ[f"{st} {lang} ready"] += 1
            if key in pq_done and max(d or 0 for _, _, _, d in parts) > (pq_done[key] or 0):
                stale.append(key)
                summ[f"{st} {lang} parquet out of date"] += 1
        else:
            summ[f"{st} {lang} not ready"] += 1
    return items, dict(summ), stale


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    dry = "--dry-run" in sys.argv
    states = set(arg("--states").split(",")) if arg("--states") else None
    stale = []
    if "--parquet" in sys.argv:
        items, summ, stale = plan_parquet(states, arg("--project", H.OCR_ALL_PROJECT))
        print(json.dumps(summ, indent=1))
        project = H.PARQUET_ALL_PROJECT
    else:
        project = arg("--project", H.OCR_ALL_PROJECT)
        items, summ = plan(states, P.ints(arg("--acs")), P.ints(arg("--parts")), kv(arg("--state-set")),
                           set((arg("--release") or "").split(",")) - {""}, "--wait-hints" in sys.argv, kv(arg("--tam-hints")),
                           "--include-fetching" in sys.argv, "--new-only" in sys.argv, project, int(arg("--partition-base", "0")),
                           include_partial="--include-partial" in sys.argv)
        print(f"{'state':5} {'name':22} {'ACs':>4} {'items':>7} {'PDFs':>7} {'pairs':>6} {'1-roll':>6}  sets / notes")
        for st, s in summ.items():
            one = sum(s["parts_one_roll"].values())
            print(f"{st:5} {str(s['name'])[:22]:22} {s['acs']:4} {s['items']:7} {s['pdfs']:7} {s['pairs']:6} {one:6}  "
                  f"{dict(s['sets'])} langs {dict(s['langs'])}"
                  + (f" | other lang ACs {len(s['acs_other_lang'])}" if s["acs_other_lang"] else "")
                  + (f" | lang unknown {s['acs_lang_unknown']}" if s["acs_lang_unknown"] else "")
                  + (f" | fetch not done: {s['acs_fetch_not_done']} ACs" if s["acs_fetch_not_done"] else "")
                  + (f" | AC records without a PDF: {s['acs_records_without_pdf']}" if s["acs_records_without_pdf"] else ""))
            if s["acs_hints_complete"] is not None:
                print(f"      2027 hints complete for {s['acs_hints_complete']} ACs; not complete: "
                      + (("; ".join(f"AC {a}: {w}" for a, w in list(s["acs_hints_incomplete"].items())[:10]) or "none")
                         + (" ..." if len(s["acs_hints_incomplete"]) > 10 else "")))
            if one:
                print(f"      parts with one roll only (the other roll's AC record does not list them): {dict(s['parts_one_roll'])}")
            if s["parts_partial_held"]:
                print(f"      HELD parts (a listed roll has no PDF; --include-partial plans them): missing roll {dict(s['parts_partial_held'])}")
            n_gap = sum(len(v) for g in s["listed_parts_without_pdf"].values() for v in g.values())
            if n_gap:
                print(f"      listed parts without a PDF: {n_gap} entries in {sum(len(g) for g in s['listed_parts_without_pdf'].values())} AC rolls")
            if s["held_state"]:
                print(f"      HELD (not in the output): {s['held_state']}")
        print(f"total items {len(items)}, PDFs {sum(len(i['payload']['keys']) for i in items)}, by set "
              f"{dict(collections.Counter(i['payload']['model_set'] for i in items))}")
        summ = {st: {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in s.items()} for st, s in summ.items()}
    if arg("--resets"):
        sql, n = P.resets_sql(project, items, stale)
        print(f"resets for items already queued in {project}: {n or 'none'}")
        if sql and not dry:
            open(arg("--resets"), "w").write(sql)
            print("wrote", arg("--resets"))
    if out and not dry:
        json.dump({"items": items, "summary": summ}, open(out, "w"), ensure_ascii=False, indent=0)
        print("wrote", out)


if __name__ == "__main__":
    main()
