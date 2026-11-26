#!/usr/bin/env python3
"""Gather step for the 2027 matrix pull.

Merges every per-AC shard output (`*_agg.json` + `*_tok.txt`, one pair per AC) into the final
PII-safe flags.json. Same schema as uk/detect.py so the UI consumes it identically.

Cross-AC duplicate EPIC = count of HMAC tokens appearing in more than one AC. Tokens are
HMAC-SHA256(epic, secret salt) emitted by uk/shard_detect.py — opaque pseudonyms, not reversible
without the salt. The final flags.json contains only aggregate counts (no tokens, names or EPICs).

Run: python uk/gather_detect.py --in <dir with shard_ac<n>_agg.json + shard_ac<n>_tok.txt> --out flags.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

AC_RE = re.compile(r"shard_ac(\d+)_")


def ac_of(path):
    m = AC_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", default="flags.json")
    a = ap.parse_args()

    per_ac = {}
    for p in sorted(glob.glob(os.path.join(a.indir, "**", "shard_ac*_agg.json"), recursive=True)):
        d = json.load(open(p))
        per_ac[str(d["ac"])] = {k: v for k, v in d.items() if k != "ac"}

    # token -> set of ACs it appears in. Each shard's tokens are DISTINCT epics within that AC,
    # so a token in >1 shard file means the same EPIC appears across >1 AC (cross-AC dup).
    tok_files = sorted(glob.glob(os.path.join(a.indir, "**", "shard_ac*_tok.txt"), recursive=True))
    tok_acs = defaultdict(set)
    for p in tok_files:
        ac = ac_of(p)
        with open(p) as fh:
            for line in fh:
                t = line.strip()
                if t:
                    tok_acs[t].add(ac)
    cross = sum(1 for acs in tok_acs.values() if len(acs) > 1) if tok_files else None

    res = {
        "state": "S28",
        "roll_scope": "2027-ssr",
        "acs_analyzed": len(per_ac),
        "dup_epic_cross_ac": cross,
        "congestion_confidence": "Plausible (house_no noisy; validate before quoting)",
        "per_ac": dict(sorted(per_ac.items(), key=lambda kv: int(kv[0]))),
    }
    if cross is None:
        res["dup_epic_cross_ac_note"] = "not computed this run (no HMAC tokens; set EPIC_HMAC_SALT to enable)"
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    tot = sum(v["records"] for v in per_ac.values())
    print(f"gather: {len(per_ac)} ACs, {tot} records, cross-AC dup EPIC = {cross} -> {a.out}")


if __name__ == "__main__":
    main()
