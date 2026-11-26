#!/usr/bin/env python3
"""Per-AC detection for the 2027 matrix pull (one AC per runner).

Reads ONE AC's parquet (produced by uk/uk_driver.py textlayer) and emits, PII-safe:
  <prefix>_agg.json  per-AC aggregate counts (identical fields to uk/detect.py's per_ac entry)
  <prefix>_tok.txt   HMAC-SHA256(epic, secret salt) of each DISTINCT epic in the AC, one per line
                     -> opaque pseudonyms; the gather step counts tokens shared across ACs =
                        cross-AC dup EPIC. Not reversible without the salt. No names/EPICs/addresses.

Run: python uk/shard_detect.py --data <dir> --ac <n> --out-prefix shard_ac<n> [--salt-env EPIC_HMAC_SALT]
"""
import argparse
import hashlib
import hmac
import json
import os

import pandas as pd
import pyarrow.parquet as pq

PLACEHOLDER_HOUSE = {"", "0", "00", "000", "-", ".", "nan", "none", "null", "na"}


def read_ac(path):
    return pq.ParquetFile(path).read().to_pandas()


def gender_mf(series):
    gv = series.astype(str).str.strip()
    gu = gv.str.upper()
    m = int((gv.str.contains("पुरुष", na=False) | gu.isin(["M", "MALE"])).sum())
    f = int((gv.str.contains("महिला", na=False) | gu.isin(["F", "FEMALE"])).sum())
    return m, f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ac", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--salt-env", default="EPIC_HMAC_SALT")
    ap.add_argument("--congestion-threshold", type=int, default=25)
    a = ap.parse_args()

    salt = os.environ.get(a.salt_env, "").encode()
    if not salt:
        raise SystemExit(f"missing HMAC salt in env {a.salt_env}")

    path = os.path.join(a.data, "release", "2027-ssr", f"ac={a.ac}", "data.parquet")
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

    agg = {
        "ac": str(a.ac),
        "records": n,
        "dup_epic_within_ac": int((vc > 1).sum()),
        "age_impossible": int(((age > 120) | (age < 18)).sum()),
        "age_centenarian": int((age >= 100).sum()),
        "gender_m": m,
        "gender_f": f,
        "females_per_1000_males": round(1000 * f / max(1, m)),
        "congestion_houses_gt_thresh": int((occ > a.congestion_threshold).sum()),
        "congestion_max": int(occ.max()) if len(occ) else 0,
    }
    with open(a.out_prefix + "_agg.json", "w") as fh:
        json.dump(agg, fh, ensure_ascii=False)
    with open(a.out_prefix + "_tok.txt", "w") as fh:
        for e in vc.index:  # distinct epics only
            fh.write(hmac.new(salt, str(e).encode(), hashlib.sha256).hexdigest() + "\n")
    print(f"shard AC {a.ac}: {n} records, {len(vc)} distinct epics -> {a.out_prefix}_agg.json + _tok.txt")


if __name__ == "__main__":
    main()
