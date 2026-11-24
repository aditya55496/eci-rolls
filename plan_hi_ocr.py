"""Planner for task ocr-any (and parquet-any) of the Hindi-state SIR rolls (2026-11-25). Runs on the hub server (reads the hub's
Postgres, read-only, through psql); writes the items as JSON for seeding, like plan_hi.py.

  plan_hi_ocr.py OUT.json [--dry-run] [--states S12,S07] [--acs 1,2] [--parts 1,10]
                 [--hint-set v4a] [--nohint-set j2Ps] [--hints-project eci-rolls-ssr-text] [--assume-hints]
  plan_hi_ocr.py OUT.json --parquet [--states ...]       parquet-any items for ACs whose ocr-any items are all done

OCR items (project eci-rolls-hi-ocr): one per part of one AC of one state, both SIR rolls when both exist. Parts = the PDFs
stored in the bucket (one hub record "<roll id>/<ac>/<part>" per PDF in eci-rolls-hi-cdn / eci-rolls-hi-form, with its key);
this includes the extra parts the probe above the part list found and leaves out the listed parts that have no file. The AC
record ("<roll id>/<ac>": part_numbers, extra_parts, missing) is the cross-check: differences are counted in the summary.
Model set by state: the states with a 2027 SSR roll (S04 S12 S20 S26 S28) get --hint-set with the hints key of task ssrtext
(<hints project>/hints/<state>-2027-DR/<ac>.jsonl.gz); the others get --nohint-set and hints null. An AC of a hint state is
planned only when its ssrtext record exists (the 2027 text is done; --assume-hints skips that check, for tests), so running
the planner again later adds the rest (seed_queue keeps existing keys).
Partition order: the no-hint states first, then the hint states; within a state by AC and part (an AC finishes together).
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


def ssr_done(project, roll):
    """ACs whose ssrtext record exists (record id '<roll id>/<ac>')."""
    rows = psql_json(f"SELECT record_id FROM records WHERE project = {lit(project)} "
                     + prefix_cond(roll))
    return {int(r["record_id"].split("/")[1]) for r in rows if r["record_id"].count("/") == 1 and r["record_id"].split("/")[1].isdigit()}


def plan_ocr(states, acs_only, parts_only, hint_set, nohint_set, hints_project, assume_hints):
    for s in (hint_set, nohint_set):
        if s not in H.MODEL_SETS:
            raise SystemExit(f"model set {s} not in model_sets.json")
    items, summ = [], {}
    order = [s for s in SIR if s not in HINT_ROLL] + [s for s in SIR if s in HINT_ROLL]
    for st in order:
        if states and st not in states:
            continue
        hinted = st in HINT_ROLL
        ready = None
        if hinted and not assume_hints:
            ready = ssr_done(hints_project, HINT_ROLL[st])
        by_ac = collections.defaultdict(lambda: collections.defaultdict(dict))     # ac -> part -> {roll: key}
        extra = collections.defaultdict(lambda: collections.defaultdict(set))      # ac -> part -> {roll}
        s = {"name": NAMES.get(st), "model_set": hint_set if hinted else nohint_set, "rolls": {}}
        for roll in SIR[st]:
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
            diff = 0
            for ac, a in acr.items():
                if acs_only and ac not in acs_only:
                    continue
                want = (set(a.get("part_numbers") or []) | set(a.get("extra_parts") or [])) - set(a.get("missing") or [])
                diff += len(want ^ have.get(ac, set()))
                for x in a.get("extra_parts") or []:
                    extra[ac][int(x)].add(roll)
            s["rolls"][roll] = {"acs": len(have), "pdfs": sum(len(v) for v in have.values()), "ac_records": len(acr),
                                "parts_differing_from_ac_record": diff,
                                "langs": sorted({r["lang"] for r in acr.values() if r.get("lang")})}
        held = 0
        n_items = n_pdfs = 0
        for ac in sorted(by_ac):
            if hinted and ready is not None and ac not in ready:
                held += len(by_ac[ac])
                continue
            for part in sorted(by_ac[ac]):
                keys = by_ac[ac][part]
                pl = {"state": st, "ac": ac, "part": part, "lang": "HIN", "keys": dict(sorted(keys.items())),
                      "hints": f"{hints_project}/hints/{HINT_ROLL[st]}/{ac}.jsonl.gz" if hinted else None,
                      "model_set": hint_set if hinted else nohint_set}
                ex = sorted(extra[ac][part] & set(keys))
                if ex:
                    pl["extra_rolls"] = ex
                items.append({"key": f"{st}/{ac}/{part}", "payload": pl})
                n_items += 1
                n_pdfs += len(keys)
        s.update(acs=len(by_ac), items=n_items, pdfs=n_pdfs, items_waiting_for_hints=held,
                 acs_with_hints=(len(ready & set(by_ac)) if ready is not None else ("assumed" if hinted else None)))
        summ[st] = s
    for i, it in enumerate(items):
        it["partition_id"] = i
    return items, summ


def plan_parquet(states):
    """parquet-any items: one per AC whose ocr-any items are all done (queue of eci-rolls-hi-ocr)."""
    rows = psql_json(f"SELECT key, status, payload->'keys' AS keys FROM queue WHERE project = {lit(H.OCR_HI_PROJECT)}")
    by = collections.defaultdict(list)
    for r in rows:
        st, ac, part = r["key"].split("/")
        if states and st not in states:
            continue
        by[(st, int(ac))].append((int(part), r["status"], sorted((r["keys"] or {}).keys())))
    items, summ = [], collections.Counter()
    for (st, ac), parts in sorted(by.items()):
        if all(s == "done" for _, s, _ in parts):
            rolls = sorted({x for _, _, ks in parts for x in ks}, key=lambda r: (0 if H.is_draft(r) else 1, r))
            items.append({"key": f"{st}/{ac}", "payload": {"state": st, "ac": ac, "parts": sorted(p for p, _, _ in parts), "rolls": rolls}})
            summ[st + " ready"] += 1
        else:
            summ[st + " not ready"] += 1
    for i, it in enumerate(items):
        it["partition_id"] = i
    return items, dict(summ)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    dry = "--dry-run" in sys.argv
    states = set(arg("--states").split(",")) if arg("--states") else None
    if "--parquet" in sys.argv:
        items, summ = plan_parquet(states)
        print(json.dumps(summ, indent=1))
    else:
        items, summ = plan_ocr(states, ints(arg("--acs")), ints(arg("--parts")), arg("--hint-set", "v4a"), arg("--nohint-set", "j2Ps"),
                               arg("--hints-project", H.SSR_TEXT_PROJECT), "--assume-hints" in sys.argv)
        print(f"{'state':5} {'name':24} {'set':6} {'ACs':>5} {'items':>7} {'PDFs':>7} {'wait':>6}  rolls (ACs / PDFs / parts differing from the AC record)")
        for st, s in summ.items():
            rl = "; ".join(f"{r} {v['acs']}/{v['pdfs']}/{v['parts_differing_from_ac_record']}" for r, v in s["rolls"].items())
            print(f"{st:5} {str(s['name'])[:24]:24} {s['model_set']:6} {s['acs']:5} {s['items']:7} {s['pdfs']:7} {s['items_waiting_for_hints']:6}  {rl}")
        print(f"total items {len(items)}, PDFs {sum(len(i['payload']['keys']) for i in items)}")
    if out and not dry:
        json.dump({"items": items, "summary": summ}, open(out, "w"), ensure_ascii=False, indent=0)
        print("wrote", out)


if __name__ == "__main__":
    main()
