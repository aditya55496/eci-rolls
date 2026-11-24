"""Plan for task fetch-hi, CDN rolls of ALL states (owner 2026-11-24: "complete the CDN downloads for all states"; language:
English where ECI offers it, else the state's language).

  plan_cdn_all.py OUT.json --ac-languages eci_ac_languages.csv --skip PLAN_HI.json
CDN rolls = SIR draft/final rolls and SSR draft rolls of 2026 and later (EROLLGEN), as plan_hi.py. Per AC one language: ENG if
the AC offers it for that roll family (discovery table of 2026-11-22; a family the table does not know uses the AC's languages
from any family), else the first other language. Items already in the Hindi plan (same roll and AC) are left out.
"""
import collections, csv, json, sys

import plan_hi as P

ALL = ["U01", "S01", "S02", "S03", "S04", "U02", "S26", "U03", "S05", "S06", "S07", "S08", "U08", "S27", "S10", "S11", "U09", "U06",
       "S12", "S13", "S14", "S15", "S16", "S17", "U05", "S18", "U07", "S19", "S20", "S21", "S22", "S29", "S23", "S24", "S28", "S25"]
CODE = {"HINDI": "HIN", "ENGLISH": "ENG", "BENGALI": "BEN", "TAMIL": "TAM", "TELUGU": "TEL", "MARATHI": "MAR", "KANNADA": "KAN",
        "MALAYALAM": "MAL", "GUJRATI": "GUJ", "URDU": "URD", "ORIYA": "ORI", "PUNJABI": "PAN", "ASSAMESE": "ASM"}


def main():
    out = sys.argv[1]
    langs_csv = sys.argv[sys.argv.index("--ac-languages") + 1]
    skip_plan = json.load(open(sys.argv[sys.argv.index("--skip") + 1]))
    skip = {it["key"] for it in skip_plan["items"]}
    fam = collections.defaultdict(dict)       # (state, rt, pg) -> {ac: [languages]}
    any_fam = collections.defaultdict(dict)   # state -> {ac: languages (union)}
    for row in csv.DictReader(open(langs_csv)):
        L = row["languages"].split()
        if not L:
            continue
        fam[(row["state_cd"], row["roll_type_ref"], row["pdf_gen_type"])][int(row["ac"])] = L
        any_fam[row["state_cd"]].setdefault(int(row["ac"]), set()).update(L)
    items, summ, rolls = [], collections.Counter(), []
    for st in ALL:
        for y in (2026, 2027):
            for r in P.eroll_types(st, y):
                if P.route(r) != "cdn":
                    continue
                roll = {"id": r["id"], "year": int(r["year"]), "ref": r["rollTypeRefId"], "gen": r["pdfGenType"], "rev": int(r["revisionNo"])}
                acs = fam.get((st, r["rollTypeRefId"], r["pdfGenType"])) or {a: sorted(v) for a, v in any_fam[st].items()}
                n = 0
                for ac, L in sorted(acs.items()):
                    key = f"{r['id']}/{ac}"
                    if key in skip or r["id"] in P.HAVE:
                        continue
                    lang = "ENG" if "ENGLISH" in L else CODE.get(next(x for x in L if x != "ENGLISH"), "ENG")
                    items.append({"key": key, "payload": {"state": st, "ac": ac, "lang": lang, "route": "cdn", "roll": roll}})
                    summ[lang] += 1
                    n += 1
                rolls.append({**roll, "state": st, "name": r.get("displayName"), "acs": n})
    for i, it in enumerate(items):
        it["partition_id"] = 100000 + i
    json.dump({"items": items, "rolls": rolls, "summary": dict(summ)}, open(out, "w"), indent=1, ensure_ascii=False)
    print(len(rolls), "CDN rolls,", len(items), "items, by language", dict(summ))


if __name__ == "__main__":
    main()
