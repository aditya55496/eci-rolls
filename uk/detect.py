#!/usr/bin/env python3
"""Within-roll discrepancy detection for the Uttarakhand (S28) voter rolls.

Reads the Parquet produced by uk_driver.py (release/<roll>/ac=<n>/data.parquet) and computes the
discrepancy radars as PII-safe AGGREGATES -> flags.json (counts only; never names/EPICs/addresses).
Validated 2026-11-26 on real AC 1 2027 data (73,940 records): gender 917 F/1000M, 7 centenarians,
0 impossible ages, 0 within-AC dup EPICs.

Radars:
  dup_epic_within_ac   EPIC appearing >1 time inside one AC (a clean roll ~ 0)
  dup_epic_cross_ac    same EPIC across >1 AC (the stronger ghost signal; needs >=2 ACs loaded)
  age_impossible       age <18 or >120 (electors must be 18+)
  age_centenarian      age >=100
  gender_skew          females per 1000 males (gender is Devanagari: पुरुष / महिला / तृतीय लिंग)
  address_congestion   distinct house_no shared by > threshold electors
                       CAVEAT (Loop-9, 2026-11-26): house_no in these rolls is noisy and
                       low-granularity (placeholders + shared values). Congestion is tagged
                       [Plausible], NOT [Confirmed] — validate against the source roll before quoting.

PII: only aggregate counts are produced. No row values are written or printed.
Run (Colab/Actions, never the Mac): python uk/detect.py --data <dir> [--congestion-threshold 25]
"""
import argparse
import glob
import json
import os

import pandas as pd
import pyarrow.parquet as pq

PLACEHOLDER_HOUSE = {"", "0", "00", "000", "-", ".", "nan", "none", "null", "na"}


def read_ac(path):
    # single-file read avoids pyarrow inferring the ac= dir as a dict-typed partition column
    return pq.ParquetFile(path).read().to_pandas()


def gender_mf(series):
    gv = series.astype(str).str.strip()
    gu = gv.str.upper()
    m = int((gv.str.contains("पुरुष", na=False) | gu.isin(["M", "MALE"])).sum())
    f = int((gv.str.contains("महिला", na=False) | gu.isin(["F", "FEMALE"])).sum())
    return m, f


def detect(data_dir, thresh=25):
    files = sorted(glob.glob(os.path.join(data_dir, "release", "*", "ac=*", "data.parquet")))
    per_ac = {}
    epic_to_acs = {}
    for path in files:
        ac = os.path.basename(os.path.dirname(path)).split("=")[-1]
        df = read_ac(path)
        n = len(df)
        epic = df["epic"].astype(str).str.strip()
        epic = epic[epic.str.len() > 0]
        vc = epic.value_counts()
        age = pd.to_numeric(df["age"], errors="coerce")
        m, f = gender_mf(df["gender"])
        hh = df["house_no"].astype(str).str.strip()
        hh = hh[~hh.str.lower().isin(PLACEHOLDER_HOUSE)]
        occ = hh.value_counts()
        per_ac[ac] = {
            "records": n,
            "dup_epic_within_ac": int((vc > 1).sum()),
            "age_impossible": int(((age > 120) | (age < 18)).sum()),
            "age_centenarian": int((age >= 100).sum()),
            "gender_m": m,
            "gender_f": f,
            "females_per_1000_males": round(1000 * f / max(1, m)),
            "congestion_houses_gt_thresh": int((occ > thresh).sum()),  # [Plausible]: house_no noisy
            "congestion_max": int(occ.max()) if len(occ) else 0,       # [Plausible]
        }
        for e in vc.index:
            epic_to_acs.setdefault(e, set()).add(ac)
    cross = sum(1 for acs in epic_to_acs.values() if len(acs) > 1)
    return {
        "state": "S28",
        "roll_scope": "2027-ssr",
        "acs_analyzed": len(per_ac),
        "dup_epic_cross_ac": cross,
        "congestion_confidence": "Plausible (house_no noisy; validate before quoting)",
        "per_ac": per_ac,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/content/out")
    ap.add_argument("--congestion-threshold", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = detect(a.data, a.congestion_threshold)
    out = a.out or os.path.join(a.data, "flags.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("ACs:", res["acs_analyzed"], "| cross-AC dup EPIC:", res["dup_epic_cross_ac"], "-> wrote", out)
    for ac, d in list(res["per_ac"].items())[:5]:
        print(f"  AC {ac}: {d['records']} recs | F/1000M {d['females_per_1000_males']} | "
              f"centenarians {d['age_centenarian']} | dupEPIC_within {d['dup_epic_within_ac']} | "
              f"congestion {d['congestion_houses_gt_thresh']} [plausible]")


if __name__ == "__main__":
    main()
