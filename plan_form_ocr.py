"""Planner for task ocr-any: the FORM-ROUTE rolls of the Hindi states (formocr, 2026-11-25). Runs on the hub server (reads the
hub's Postgres read-only through psql, as plan_hi_ocr.py); writes the items as JSON for seeding into eci-rolls-hi-ocr.

  plan_form_ocr.py OUT.json [--dry-run] [--new-only] [--states S12,S04] [--types GE,SSR] [--acs 1,2] [--parts 1,10]
                   [--set j5r4] [--type-set GE=SET,SSR=SET] [--release S07] [--include-partial] [--partition-base N]
                   [--resets OUT.sql]
  plan_form_ocr.py OUT.json --parquet [--states ...] [--resets OUT.sql]     parquet-any items of form-route ACs whose OCR is done

Source: hub project eci-rolls-hi-form (task fetch-hi, form route). Queue item '<roll id>/<ac>' per AC roll ('done' = every part
of that AC roll the portal serves is stored), AC record '<roll id>/<ac>' (part_numbers, extra_parts, missing, stored_now,
known_before, lang), one record per stored PDF '<roll id>/<ac>/<part>' (key).
Roll types (roll id '<state>-<year>-<suffix>', the portal's roll list in the queue payload):
  GE   '<st>-<year>-GEN-FIR' (GE 2024 final roll; Delhi U05 2025): one item per part, roll group = the roll id.
  SSR  '<st>-<year>-DR' / '<st>-<year>-FIR' (SSR draft / final, rev 1): draft and final of the SAME year are one item (draft
       first: its readings are the final's cache, as for SIR); a year with only one of them gives one-roll items. Roll group
       '<st>-<year>-SSR'. MP has S12-2024-DR and S12-2025-FIR: two groups of one roll each.
  Not planned (counted): SIR rolls (S04-2025-SIR-DR/FIR: plan_hi_ocr.py plans them), supplements (SUPP: job5 SUP gold
  95.65 % < 97 %), adjudication / tribunal lists (SUPP-JO-*), bye-election rolls (BY-*), other prints (S04-2025-ST-FIR = FC
  SIR final rev 2, S25-2026-FIR-2: no gold).
States: every state of eci-rolls-hi-form. HELD (planned only with --release ST): S07 Haryana (owner 2026-11-25 09:23: waits for
job2's fix of the name-final glyph printed as an empty box), S25 West Bengal (Nepali / stamped prints, 76-83 % flagged); group
S04-2025-SSR (Bihar: S04-2025-DR mixes SIR 2025 draft prints into the SSR draft roll; --release S04-2025-SSR).
Item: key '<state>/<roll group>/<ac>/<part>' (never collides with the SIR keys '<state>/<ac>/<part>'), payload v 2 =
the SIR payload + 'roll_group'; output ocr/<state>/<roll group>/<ac>/<part>.jsonl.gz (hi_common.ocr_base).
Only AC rolls whose fetch item is 'done' are planned; a group waits until every roll of the group that the state has is done
for the AC (--include-partial plans it with the done rolls; --resets then gives the item the other roll later, as
plan_hi_ocr). Parts = PDFs stored; a part with the PDF of only one roll of a two-roll group is planned with that roll (counted:
parts_one_roll; the AC records list the AC's CURRENT parts, not the roll's, so a listed part without a PDF is no proof of a
gap: Bihar S04-2025-DR has up to 63 parts per AC above the FIR's highest part). Listed parts without a PDF get no item
(listed_parts_without_pdf in the summary, for the part-gap work).
Model set: --set (default j5r4) for every type, --type-set per type. Hints per type and state: a set whose hints mode is not
'none' gets the AC's 2027 hints where the state has a 2027 roll (S04 S12 S20 S26 S28: eci-rolls-ssr-text
hints/<st>-2027-DR/<ac>.jsonl.gz, complete per ssr_done; S24 UP: eci-rolls-up-2027 hints/<ac>.jsonl.gz, its queue item done);
a 'required' set waits for them. j5r4 is a no-hint set (job5 round 1: 2027 hints confirm 2027 spellings on the 2024/25
prints), so its items carry hints null. Roll year = the roll id's year (eci_ocr age window with hints: 2027 - year - 1 .. 2027 -
year: GE 2024 / SSR 2024 2-3, SSR 2025 1-2; without hints the year sets only the GE stamp template, year < 2026).
--new-only: leaves out (state, group, AC) that already have items in eci-rolls-hi-ocr (incremental batches as downloads finish).
Partition id (fixed, the same on every run): --partition-base (default 100,000,000: after every SIR item, whose ids are below
10,000,000) + (FORM_RANK[state] x 4 + slot) x 1,000,000 + AC x 1,000 + part; slot GE 2024 0, SSR 2024 1, GE 2025 2, SSR 2025 3.
--dry-run prints counts per state and roll type and writes nothing.
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hi_common as H  # noqa: E402
import plan_hi_ocr as P  # noqa: E402

FORM_PROJECT = "eci-rolls-hi-form"
UP_TEXT_PROJECT = "eci-rolls-up-2027"     # UP 2027 hints: hints/<ac>.jsonl.gz (task textlayer, one queue item per AC)
# Fixed partition rank per state (never renumber; a new state gets the next free rank)
FORM_RANK = {"U05": 0, "U02": 1, "S27": 2, "U08": 3, "S08": 4, "S12": 5, "S20": 6, "S26": 7, "S28": 8, "S04": 9, "S24": 10,
             "S07": 11, "S25": 12}
NAMES = dict(P.NAMES, S08="Himachal Pradesh", S24="Uttar Pradesh", U08="Jammu and Kashmir")
HOLD = {"S07": "owner 2026-11-25 09:23: Haryana waits for job2's fix of the name-final glyph (empty box; 22-58 % flags in ACs "
               "28 78 88); --release S07 to plan",
        "S25": "West Bengal (Nepali / stamped prints, 76-83 % flagged with j2Ps); --release S25 to plan",
        # roll groups (formocr 2026-11-25, checked by eye on page 1): S04-2025-DR holds SIR 2025 draft prints ('vishesh gahan
        # punarikshan 2025 ... 01-08-2025') in some ACs (AC 14 parts 15 and 300; 11 ACs have DR PDFs of the same byte size as
        # the S04-2025-SIR-DR PDF of the same part), the SSR draft ('29-10-2024') in others (AC 164 part 5); S04-2025-FIR has
        # PDFs of the S04-2025-SIR-FIR size in ACs 26 and 29. Pairing a SIR draft with the SSR final is wrong, and the SIR
        # draft is OCR'd by plan_hi_ocr already.
        "S04-2025-SSR": "Bihar S04-2025-DR mixes SIR 2025 draft prints into the SSR draft roll (fetch-side; a per-PDF roll check "
                        "of page 1 is needed first); --release S04-2025-SSR to plan"}
HINT_STATES = dict(P.HINT_ROLL)            # state -> 2027 SSR roll in eci-rolls-ssr-text
UP_HINTS = "S24"
ROLL_RE = re.compile(r"^([SU][0-9]{2})-(20[0-9]{2})-(.+)$")


def classify(roll):
    """roll id -> (type, year) with type GE / SSR-DR / SSR-FIR, or (excluded reason, year)."""
    m = ROLL_RE.match(roll)
    if not m:
        return "other", None
    year, suf = int(m.group(2)), m.group(3)
    if "SUPP-JO" in suf:
        return "excluded: adjudication / tribunal list", year
    if "SUPP" in suf:
        return "excluded: supplement", year
    if suf.startswith("BY-"):
        return "excluded: bye-election", year
    if suf.startswith("SIR-"):
        return "excluded: SIR (plan_hi_ocr.py)", year
    if suf == "GEN-FIR":
        return "GE", year
    if suf == "DR":
        return "SSR-DR", year
    if suf == "FIR":
        return "SSR-FIR", year
    return "excluded: other print (no gold)", year


def group_of(roll):
    """roll id -> (roll group, type 'GE' / 'SSR', year) or None for a roll not planned here."""
    t, y = classify(roll)
    st = roll.split("-")[0]
    if t == "GE":
        return roll, "GE", y
    if t in ("SSR-DR", "SSR-FIR"):
        return f"{st}-{y}-SSR", "SSR", y
    return None


def slot(gtype, year):
    s = (year - 2024) * 2 + (0 if gtype == "GE" else 1)
    if not 0 <= s < 4:
        raise SystemExit(f"no partition slot for {gtype} {year} (planner covers 2024-2025)")
    return s


def form_rolls():
    """{roll id: {ac: queue status}} of every AC roll item in eci-rolls-hi-form (one read of that project's queue)."""
    rows = P.psql_json(f"SELECT key, status FROM queue WHERE project = {P.lit(FORM_PROJECT)}")
    out = collections.defaultdict(dict)
    for r in rows:
        roll, _, ac = r["key"].partition("/")
        if ac.isdigit():
            out[roll][int(ac)] = r["status"]
    return out


def planned_groups():
    """{(state, group, ac)} that already have form-route items in eci-rolls-hi-ocr (keys with 4 segments)."""
    rows = P.psql_json(f"SELECT key FROM queue WHERE project = {P.lit(H.OCR_HI_PROJECT)} "
                       "AND length(key) - length(replace(key, '/', '')) = 3")
    out = set()
    for r in rows:
        st, g, ac, _ = r["key"].split("/")
        out.add((st, g, int(ac)))
    return out


def up_hints_done():
    rows = P.psql_json(f"SELECT key FROM queue WHERE project = {P.lit(UP_TEXT_PROJECT)} AND status = 'done'")
    return {int(r["key"]) for r in rows if r["key"].isdigit()}


def hints_for(st, ac, ready):
    """-> (hints key or None, ready bool) of a state's AC; ready = None when the state has no 2027 roll."""
    if st in HINT_STATES:
        return f"{H.SSR_TEXT_PROJECT}/hints/{HINT_STATES[st]}/{ac}.jsonl.gz", ac in ready.get(st, set())
    if st == UP_HINTS:
        return f"{UP_TEXT_PROJECT}/hints/{ac}.jsonl.gz", ac in ready.get(st, set())
    return None, None


def plan(states=None, types=None, acs_only=None, parts_only=None, set_name="j5r4", type_set=None, release=(), include_partial=False,
         new_only=False, base=100_000_000, rolls=None, planned=None, hints_ready=None):
    """-> (items, summary). rolls / planned / hints_ready: injected in tests (default: read from the hub database)."""
    type_set = type_set or {}
    for s in [set_name] + list(type_set.values()):
        if s not in H.MODEL_SETS:
            raise SystemExit(f"model set {s} not in model_sets.json")
    rolls = form_rolls() if rolls is None else rolls
    planned = (planned_groups() if new_only else set()) if planned is None else planned
    groups = collections.defaultdict(dict)        # (state, group) -> {roll: {ac: status}}
    gmeta = {}
    excluded = collections.defaultdict(lambda: collections.Counter())    # reason -> {state: AC rolls}
    for roll, acs in sorted(rolls.items()):
        st = roll.split("-")[0]
        g = group_of(roll)
        if g is None:
            excluded[classify(roll)[0]][st] += len(acs)
            continue
        groups[(st, g[0])][roll] = acs
        gmeta[(st, g[0])] = g[1:]
    need_hints = {t: H.MODEL_SETS[type_set.get(t, set_name)].get("hints", "optional") != "none" for t in ("GE", "SSR")}
    ready = {}
    if hints_ready is not None:
        ready = hints_ready
    elif any(need_hints.values()):
        for st in sorted({k[0] for k in groups}):
            if st in HINT_STATES:
                ready[st] = P.ssr_done(H.SSR_TEXT_PROJECT, HINT_STATES[st])[0]
            elif st == UP_HINTS:
                ready[st] = up_hints_done()
    items, summ = [], {}
    for (st, grp) in sorted(groups, key=lambda k: (FORM_RANK.get(k[0], 99), slot(*gmeta[k]), k[1])):
        gtype, year = gmeta[(st, grp)]
        if states and st not in states or types and gtype not in types:
            continue
        if st not in FORM_RANK:
            raise SystemExit(f"state {st} has no FORM_RANK entry (add the next free rank)")
        mset = type_set.get(gtype, set_name)
        ms = H.MODEL_SETS[mset]
        grolls = H.roll_order(groups[(st, grp)])
        all_acs = sorted(set().union(*[set(groups[(st, grp)][r]) for r in grolls]))
        if acs_only:
            all_acs = [a for a in all_acs if a in acs_only]
        s = {"state": st, "name": NAMES.get(st), "type": gtype, "year": year, "rolls": grolls, "model_set": mset,
             "acs": len(all_acs), "acs_done": 0, "acs_waiting_fetch": 0, "acs_partial_planned": 0, "acs_planned_before": 0,
             "acs_waiting_hints": 0, "acs_other_lang": 0, "items": 0, "pdfs": 0, "parts_one_roll": 0,
             "listed_parts_without_pdf": {}, "stored_not_listed": 0, "held_state": None}
        summ[f"{st}/{grp}"] = s
        if st in HOLD and st not in release:
            s["held_state"] = HOLD[st]
        elif grp in HOLD and grp not in release:
            s["held_state"] = HOLD[grp]
        # ACs whose every roll of the group is done (or, with --include-partial, at least one)
        ok = {}
        for ac in all_acs:
            st_by_roll = {r: groups[(st, grp)][r].get(ac) for r in grolls}
            done = [r for r, x in st_by_roll.items() if x == "done"]
            if (st, grp, ac) in planned:
                s["acs_planned_before"] += 1
                continue
            if len(done) == len([r for r, x in st_by_roll.items() if x is not None]) and done:
                ok[ac] = done
                s["acs_done"] += 1
            elif done and include_partial:
                ok[ac] = done
                s["acs_partial_planned"] += 1
            else:
                s["acs_waiting_fetch"] += 1
        if not ok:
            continue
        by_ac = collections.defaultdict(lambda: collections.defaultdict(dict))
        extra = collections.defaultdict(lambda: collections.defaultdict(set))
        lang = {}
        for roll in grolls:
            if not any(roll in v for v in ok.values()):
                continue
            rows = P.roll_rows(roll)
            pdf = [r for r in rows if r["record_id"].count("/") == 2 and r["key"] and r["ac"] in ok and roll in ok[r["ac"]]]
            acr = {r["ac"]: r for r in rows if r["record_id"].count("/") == 1 and r["ac"] in ok and roll in ok[r["ac"]]}
            have = collections.defaultdict(set)
            for r in pdf:
                have[r["ac"]].add(r["part"])
                if not parts_only or r["part"] in parts_only:
                    by_ac[r["ac"]][r["part"]][roll] = r["key"]
            for ac, a in acr.items():
                want = (set(a.get("part_numbers") or []) | set(a.get("extra_parts") or [])) - set(a.get("missing") or [])
                gap = sorted(int(x) for x in want - have.get(ac, set()))
                if gap:
                    s["listed_parts_without_pdf"].setdefault(roll, {})[str(ac)] = gap
                s["stored_not_listed"] += len(have.get(ac, set()) - want)
                for x in a.get("extra_parts") or []:
                    extra[ac][int(x)].add(roll)
                if a.get("lang"):
                    lang.setdefault(ac, set()).add(a["lang"])
        hint_mode = ms.get("hints", "optional")
        for ac in sorted(by_ac):
            if (lang.get(ac) or {"HIN"}) != {"HIN"}:
                s["acs_other_lang"] += 1
                continue
            hk, hready = hints_for(st, ac, ready) if hint_mode != "none" else (None, None)
            if hint_mode == "required" and not hready:
                s["acs_waiting_hints"] += 1
                continue
            hk = hk if hready else None
            for part in sorted(by_ac[ac]):
                keys = by_ac[ac][part]
                if len(keys) < len(ok[ac]):
                    # every planned roll of the AC is done: what is stored is all the portal served. No hold (the AC records'
                    # part lists are the AC's CURRENT list, not the roll's: Bihar S04-2025-DR has parts above the FIR's
                    # highest part); counted. A later PDF of the other roll -> --resets gives the item that roll.
                    s["parts_one_roll"] += 1
                pl = {"state": st, "ac": ac, "part": part, "lang": "HIN", "keys": dict(sorted(keys.items())), "hints": hk,
                      "model_set": mset, "roll_group": grp, "v": 2}
                ex = sorted(extra[ac][part] & set(keys))
                if ex:
                    pl["extra_rolls"] = ex
                pid = base + (FORM_RANK[st] * 4 + slot(gtype, year)) * 1_000_000 + ac * 1000 + part
                s["items"] += 1
                s["pdfs"] += len(keys)
                if not s["held_state"]:
                    items.append({"key": f"{st}/{grp}/{ac}/{part}", "partition_id": pid, "payload": pl})
    for reason, by in sorted(excluded.items()):
        summ[reason] = dict(sorted(by.items()))
    return items, summ


def plan_parquet(states=None):
    """parquet-any items '<state>/<group>/<ac>' for form-route ACs whose ocr-any items are all done."""
    rows = P.psql_json(f"SELECT key, status, payload->'keys' AS keys FROM queue WHERE project = {P.lit(H.OCR_HI_PROJECT)} "
                       "AND length(key) - length(replace(key, '/', '')) = 3")
    by = collections.defaultdict(list)
    for r in rows:
        st, grp, ac, part = r["key"].split("/")
        if states and st not in states:
            continue
        by[(st, grp, int(ac))].append((int(part), r["status"], sorted((r["keys"] or {}).keys())))
    items, summ = [], collections.Counter()
    for (st, grp, ac), parts in sorted(by.items()):
        gtype, year = group_of(H.roll_order({x for _, _, ks in parts for x in ks})[0])[1:]
        if all(s == "done" for _, s, _ in parts):
            items.append({"key": f"{st}/{grp}/{ac}", "partition_id": 1_000_000 + (FORM_RANK[st] * 4 + slot(gtype, year)) * 1000 + ac,
                          "payload": {"state": st, "ac": ac, "form_parts": sorted(p for p, _, _ in parts), "roll_group": grp,
                                      "rolls": H.roll_order({x for _, _, ks in parts for x in ks}), "v": 2}})
            summ[f"{st}/{grp} ready"] += 1
        else:
            summ[f"{st}/{grp} not ready"] += 1
    return items, dict(summ)


def main():
    a = P.arg
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    dry = "--dry-run" in sys.argv
    states = set(a("--states").split(",")) if a("--states") else None
    if "--parquet" in sys.argv:
        items, summ = plan_parquet(states)
        print(json.dumps(summ, indent=1))
        project = H.PARQUET_HI_PROJECT
    else:
        type_set = dict(kv.split("=", 1) for kv in a("--type-set").split(",")) if a("--type-set") else {}
        items, summ = plan(states, set(a("--types").split(",")) if a("--types") else None, P.ints(a("--acs")), P.ints(a("--parts")),
                           a("--set", "j5r4"), type_set, set(a("--release").split(",")) if a("--release") else set(),
                           "--include-partial" in sys.argv, "--new-only" in sys.argv, int(a("--partition-base", "100000000")))
        print(f"{'group':22} {'name':18} {'type':4} {'set':5} {'ACs':>4} {'done':>4} {'w-fetch':>7} {'before':>6} {'items':>7} {'PDFs':>7} "
              f"{'1-roll':>6} {'gaps':>5}")
        tot = collections.Counter()
        for k, s in summ.items():
            if "type" not in s:
                continue
            gaps = sum(len(v) for g in s["listed_parts_without_pdf"].values() for v in g.values())
            print(f"{k:22} {str(s['name'])[:18]:18} {s['type']:4} {s['model_set']:5} {s['acs']:4} {s['acs_done']:4} {s['acs_waiting_fetch']:7} "
                  f"{s['acs_planned_before']:6} {s['items']:7} {s['pdfs']:7} {s['parts_one_roll']:6} {gaps:5}"
                  + (f"  other lang ACs {s['acs_other_lang']}" if s["acs_other_lang"] else "")
                  + (f"  w-hints ACs {s['acs_waiting_hints']}" if s["acs_waiting_hints"] else "")
                  + ("  HELD (not in the output)" if s["held_state"] else ""))
            if not s["held_state"]:
                tot[s["type"] + " items"] += s["items"]
                tot[s["type"] + " PDFs"] += s["pdfs"]
        for k, v in summ.items():
            if "type" not in v:
                print(f"{k}: {sum(v.values())} AC rolls ({', '.join(f'{x} {n}' for x, n in v.items())})")
        print(f"total items {len(items)}, PDFs {sum(len(i['payload']['keys']) for i in items)}; {dict(tot)}")
        project = H.OCR_HI_PROJECT
    if a("--resets"):
        sql, n = P.resets_sql(project, items)
        print(f"resets for items already queued in {project}: {n or 'none'}")
        if sql and not dry:
            open(a("--resets"), "w").write(sql)
            print("wrote", a("--resets"))
    if out and not dry:
        json.dump({"items": items, "summary": summ}, open(out, "w"), ensure_ascii=False, indent=0)
        print("wrote", out)


if __name__ == "__main__":
    main()
