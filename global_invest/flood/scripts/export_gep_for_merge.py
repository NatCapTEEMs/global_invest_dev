#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
export_gep_for_merge.py -- the three columns the other services join on
NatCap TEEMs / GEP Flood
===============================================================================

WHY THIS IS SEPARATE

The valuation writes a full country table: exposure, the protection split, both
counterfactuals, prevention shares, attributed damage, the design standard and
its evidence class. Twenty-one columns, because anyone auditing the account needs
all of them.

A cross-service table needs three. This writes those three, from the table the
valuation already produced, so combining flood with erosion and the rest is a
join rather than a reconciliation.

It is a script rather than a step inside `task_compute_flood_gep` for the same
reason the figures are: it presents the account rather than computing it, and a
change here should not oblige anyone to re-verify the valuation. Nothing it reads
is modified, and running it twice gives the same file.

    iso3_r250_label     what every service keys on
    iso3_r250_name      carried so the file reads without a lookup
    gep_const2019_usd   the reported value, bare-soil counterfactual

BLANK RATHER THAN ZERO

Countries where the account ran and found no service are blank. Zero would say
ecosystems prevent nothing there, which is a claim; blank says there is nothing
to report, which is the truth for a country whose rivers the hazard product does
not model. It also matches how the other services' columns treat absence, so a
count of non-blank cells gives the number of countries with a service rather than
the number of rows.

USAGE
    python export_gep_for_merge.py
    python export_gep_for_merge.py --scenario insitu --out /path/to/file.csv
===============================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from global_invest.flood import flood_paths as FP
except ImportError:                       # run directly from the module directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import flood_paths as FP

SCENARIOS = {
    "bare": ("gep_flood_bare_usd2019",
             "bare soil: land cover removed entirely, the reported figure"),
    "insitu": ("gep_flood_insitu_usd2019",
               "in-situ degradation: soils compacted, canopy thinned, land present"),
}

# Below this the value is numerical noise from differencing two large integrals
# rather than a service, and publishing it invites a reader to divide by it.
MIN_GEP_USD = 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="bare", choices=sorted(SCENARIOS))
    ap.add_argument("--results", default=None,
                    help="country table; defaults to the valuation's own output")
    ap.add_argument("--lookup", default=None,
                    help="country master carrying iso3_r250_id; defaults to inputs/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results = Path(a.results) if a.results else (
        FP.OUTPUTS / "_global" / "flood_gep_country_results_v2_2024hazard.csv")
    if not results.exists():
        results = FP.OUTPUTS / "_global" / "step4e_flood_gep_USD2019.csv"
    if not results.exists():
        print(f"[FAIL] no country table at {results}")
        print("       run the valuation first, or pass --results")
        return 1

    lookup = Path(a.lookup) if a.lookup else (
        FP.INPUTS / "country_vector" / "country_codes_master_justin.csv")
    if not lookup.exists():
        print(f"[FAIL] no country master at {lookup}")
        print("       it carries iso3_r250_id, which the account's table does not")
        return 1

    col, note = SCENARIOS[a.scenario]
    d = pd.read_csv(results)
    if col not in d.columns:
        print(f"[FAIL] {results.name} has no column {col}")
        return 1

    # Territories share their parent's ISO3 in the boundary layer -- Reunion and
    # French Guiana are both FRA -- so a naive merge inflates 250 rows to 264.
    # The master table has been generated more than one way -- from the boundary
    # gpkg and from a wider correspondence file -- and the r250 id column is named
    # differently or absent depending on which. In every version `iso3` already
    # holds the r250 label, so fall back to it rather than requiring the id.
    raw = pd.read_csv(lookup)
    id_col = next((c for c in ("iso3_r250_id", "iso3_r250_label", "adm0_a3")
                   if c in raw.columns), "iso3")
    name_col = next((c for c in ("iso3_r250_name", "country_name", "name_long")
                     if c in raw.columns), None)
    if name_col is None:
        print(f"[FAIL] {lookup.name} carries no country-name column")
        return 1
    m = (raw[["iso3", id_col, name_col]]
         .rename(columns={id_col: "iso3_r250_id", name_col: "iso3_r250_name"})
         .drop_duplicates(subset="iso3"))

    if "iso3_r250_id" in d.columns:
        out = d.copy()
    else:
        out = d.merge(m, on="iso3", how="left")

    unmatched = out[out.iso3_r250_id.isna()].iso3.tolist()
    if unmatched:
        # Palestine shares ISR in that layer and South Sudan postdates it. Both
        # report damage, so leaving them unlabelled would silently drop them from
        # any merge on the r250 id.
        print(f"[warn] no r250 id for {len(unmatched)}: {' '.join(unmatched)}")
        print("       falling back to the ISO3 code for these")
        for iso in unmatched:
            out.loc[out.iso3 == iso, "iso3_r250_id"] = iso
            out.loc[out.iso3 == iso, "iso3_r250_name"] = out.loc[out.iso3 == iso, "country_name"]

    pub = out[["iso3_r250_id", "iso3_r250_name", col]].copy()
    pub.columns = ["iso3_r250_label", "iso3_r250_name", "gep_const2019_usd"]
    pub.loc[pub.gep_const2019_usd < MIN_GEP_USD, "gep_const2019_usd"] = pd.NA
    pub = pub.sort_values("iso3_r250_label").reset_index(drop=True)

    dest = Path(a.out) if a.out else (
        results.parent / f"flood_gep_for_merge_{a.scenario}.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pub.to_csv(dest, index=False)

    served = int(pub.gep_const2019_usd.notna().sum())
    print(f"[OK] {dest}")
    print(f"     scenario   {a.scenario} -- {note}")
    print(f"     rows       {len(pub)}")
    print(f"     reporting  {served} with a service, {len(pub) - served} blank")
    print(f"     total      {pub.gep_const2019_usd.sum():,.0f} USD 2019/yr")
    return 0


if __name__ == "__main__":
    sys.exit(main())
