"""Plan for task fetch-hi (owner 2026-11-24: every roll of the states that publish Hindi rolls, in Hindi).

  plan_hi.py OUT.json [--ac-languages eci_ac_languages.csv]
1. The roll list of each state for 2024-2027 from the portal (get-publish-eroll-type; about 52 small calls, no captcha).
2. The ACs that offer Hindi per roll family, from the discovery's per-AC language table (2026-11-22; a roll family the table
   does not know gets the state's Hindi ACs).
3. Route per roll: "cdn" for SIR draft/final and SSR draft rolls of 2026 and later (EROLLGEN; measured: the public path serves
   only these), "form" for everything else (captcha form; India IPs only).
4. Rolls already in the bucket are left out (UP: SIR draft/final 2026, SSR draft 2027, GE 2024; projects eci-rolls-up*).
Writes {"items": [{"key", "partition_id", "payload"}], "rolls": [...], "summary": {...}}.
"""
import base64, collections, csv, json, os, sys
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import eci_api as E

STATES = {"S04": "Bihar", "U02": "Chandigarh", "S26": "Chhattisgarh", "S07": "Haryana", "S08": "Himachal Pradesh",
          "U08": "Jammu and Kashmir", "S27": "Jharkhand", "S12": "Madhya Pradesh", "U05": "NCT OF Delhi", "S20": "Rajasthan",
          "S24": "Uttar Pradesh", "S28": "Uttarakhand", "S25": "West Bengal"}
HAVE = {"S24-2026-DR", "S24-2026-FIR", "S24-2026-FIR-1", "S24-2027-DR", "S24-2024-GEN-FIR"}     # already stored (eci-rolls-up, -up-ger)
CDN_REFS = {"SIR-DraftRoll", "SIR-FinalRoll", "SSR-DraftRoll"}


def enc_params(*vals):
    k, iv = os.urandom(32), os.urandom(12)
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    u = lambda x: base64.urlsafe_b64encode(x).decode().rstrip("=")  # noqa: E731
    wrapped = E._PUB.encrypt(k, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    a = AESGCM(k)
    return u(wrapped), u(iv), [u(a.encrypt(iv, json.dumps(v).encode(), None)) for v in vals]


def eroll_types(state, year):
    yek, rot, (s, y, m) = enc_params(state, str(year), E.MISKEY)
    r = requests.get(E.GW + f"api/v1/printing-publish/get-publish-eroll-type?stateCd={s}&year={y}&misKey={m}",
                     headers={**E.HEADERS, "accept_yek": yek, "accept_rotcev": rot}, timeout=60)
    r.raise_for_status()
    return r.json().get("payload") or []


def route(r):
    return "cdn" if r["pdfGenType"] == "EROLLGEN" and r["rollTypeRefId"] in CDN_REFS and int(r["year"]) >= 2026 else "form"


def main():
    out = sys.argv[1]
    langs = sys.argv[sys.argv.index("--ac-languages") + 1] if "--ac-languages" in sys.argv else None
    hindi = collections.defaultdict(set)          # (state, rt, pg) -> ACs with HINDI
    state_hindi = collections.defaultdict(set)
    for row in csv.DictReader(open(langs)) if langs else []:
        if row["state_cd"] in STATES and "HINDI" in row["languages"].split():
            hindi[(row["state_cd"], row["roll_type_ref"], row["pdf_gen_type"])].add(int(row["ac"]))
            state_hindi[row["state_cd"]].add(int(row["ac"]))
    items, rolls, summ = [], [], collections.Counter()
    for st in STATES:
        for y in (2024, 2025, 2026, 2027):
            for r in eroll_types(st, y):
                roll = {"id": r["id"], "year": int(r["year"]), "ref": r["rollTypeRefId"], "gen": r["pdfGenType"], "rev": int(r["revisionNo"])}
                acs = sorted(hindi.get((st, r["rollTypeRefId"], r["pdfGenType"])) or state_hindi[st])
                by = [int(a["acNo"]) for a in (r.get("byElecAcList") or [])]
                if by:                                   # a bye-election roll covers only its ACs
                    acs = [a for a in acs if a in by] or by
                skip = r["id"] in HAVE
                rolls.append({**roll, "state": st, "name": r.get("displayName"), "route": route(r), "acs": len(acs), "skipped_have": skip})
                if skip:
                    continue
                for ac in acs:
                    items.append({"key": f"{r['id']}/{ac}", "payload": {"state": st, "ac": ac, "lang": "HIN", "route": route(r), "roll": roll}})
                    summ[route(r)] += 1
    def rank(it):    # form items: main rolls first, then supplements, bye-elections, tribunal lists (the hub claims by partition)
        rid = it["payload"]["roll"]["id"]
        return (3 if "-JO-" in rid else 2 if "-BY-" in rid else 1 if "-SUPP-" in rid else 0, rid, it["payload"]["ac"])
    items.sort(key=rank)
    for i, it in enumerate(items):
        it["partition_id"] = i
    json.dump({"items": items, "rolls": rolls, "summary": dict(summ)}, open(out, "w"), indent=1, ensure_ascii=False)
    print(len(rolls), "rolls,", dict(summ), "items")


if __name__ == "__main__":
    main()
