"""Aggregate the CLUE per-building census into the site-year attributes the supervisor asked for.

Height and land use, per building, every year -- what the satellite pipeline is being asked to
recover from pixels. This is the ceiling arm: give the model the true numbers and see how much
traffic signal they carry. Whatever it reaches is the target the imagery-only route is aiming
at, and the gap between them is the finding.

Two things this is not. It is not a replacement for the imagery route: a per-building planning
census exists for the City of Melbourne and almost nowhere else, which is the whole reason the
project reads satellite photographs instead. And it is not a model input for the imagery arm --
these columns go into their own run, never into the one that claims to need only pixels.

The footprint is the satellite patch, not a radius. stitch_patch cuts a square 500 m across
-- 250 m either side of the site -- so a 500 m radius would have covered 785,000 square metres
against the imagery's 250,000, reaching twice as far in every direction. The two arms are meant
to describe the same ground, so this reads the same square.

Distance inside that square is kept rather than averaged away: a forty-storey tower on the far
corner and the same tower across the road are not the same fact about a junction. Two cheap
ways to say so without changing the model -- an inner square at half the width, and sums
weighted by 1/distance -- plus the plain distance to the nearest tall building.

    python outputs/phase2_offnadir_masking_v1/attribute_site_features.py
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SITES = HERE / "clue_area/area_sites_clue.csv"
OUT = HERE / "clue_area/attribute_features"
BUILDINGS = OUT / "clue_buildings.csv"

# the space uses worth their own column; everything rarer is pooled into "other" rather than
# spread across a long tail of columns that are zero almost everywhere
USES = ["House/Townhouse", "Residential Apartment", "Office", "Retail - Shop",
        "Entertainment/Recreation - Indoor", "Educational/Research", "Storage",
        "Commercial Accommodation", "Hospital/Clinic", "Student Accommodation",
        "Workshop/Studio", "Unoccupied - Unused"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-m", type=float, default=500.0,
                        help="side of the square, matching the satellite patch")
    parser.add_argument("--min-buildings", type=int, default=20,
                        help="a site counts as covered if this many buildings sit "
                             "inside the patch in every year")
    args = parser.parse_args()

    frame = pd.read_csv(BUILDINGS, encoding="utf-8-sig", low_memory=False)
    frame = frame[frame.longitude.notna()].copy()
    frame["floors"] = pd.to_numeric(frame.number_of_floors_above_ground,
                                    errors="coerce").fillna(1.0)
    frame["bicycle_spaces"] = pd.to_numeric(frame.bicycle_spaces, errors="coerce").fillna(0.0)
    frame["built"] = pd.to_numeric(frame.construction_year, errors="coerce")
    frame["use"] = np.where(frame.predominant_space_use.isin(USES),
                            frame.predominant_space_use, "other")

    sites = pd.read_csv(SITES)
    reference = float(sites.LATITUDE.mean())
    metres_per_lon = 111_320.0 * math.cos(math.radians(reference))
    site_x = sites.LONGITUDE.values * metres_per_lon
    site_y = sites.LATITUDE.values * 110_540.0

    years = sorted(frame.census_year.unique())
    print(f"{len(frame)} building-years, census years {years}")
    print(f"{len(sites)} sites, square {args.patch_m:.0f} m across (the satellite patch), "
          f"inner square {args.patch_m / 2:.0f} m\n")

    rows = []
    for year in years:
        near_year = frame[frame.census_year == year]
        bx = near_year.longitude.values * metres_per_lon
        by = near_year.latitude.values * 110_540.0
        dx = np.abs(site_x[:, None] - bx[None, :])
        dy = np.abs(site_y[:, None] - by[None, :])
        half = args.patch_m / 2.0
        within = (dx <= half) & (dy <= half)
        inner = (dx <= half / 2.0) & (dy <= half / 2.0)
        distance = np.hypot(site_x[:, None] - bx[None, :], site_y[:, None] - by[None, :])
        for index, site_no in enumerate(sites.SITE_NO.astype(int).values):
            near = near_year[within[index]]
            record = {"SITE_NO": site_no, "year": int(year), "buildings": len(near)}
            if len(near):
                floors = near.floors.values
                record.update({
                    "floors_mean": float(floors.mean()),
                    "floors_max": float(floors.max()),
                    "floors_sum": float(floors.sum()),
                    "floors_p90": float(np.percentile(floors, 90)),
                    "tall_over_10": float((floors >= 10).sum()),
                    "tall_over_30": float((floors >= 30).sum()),
                    "bicycle_spaces": float(near.bicycle_spaces.sum()),
                    "built_last_5y": float((near.built >= year - 5).sum()),
                    "median_age": float(year - near.built.median())
                    if near.built.notna().any() else np.nan,
                })
                # the same measures again for the inner quarter of the patch, and weighted by
                # nearness, so "a tower at the junction" reads differently from "a tower at
                # the corner of the frame"
                close = near_year[inner[index]]
                record["buildings_inner"] = len(close)
                record["floors_sum_inner"] = float(close.floors.sum()) if len(close) else 0.0
                record["floors_max_inner"] = float(close.floors.max()) if len(close) else 0.0
                metres = np.maximum(distance[index][within[index]], 20.0)
                record["floors_sum_weighted"] = float((floors / metres).sum())
                record["buildings_weighted"] = float((1.0 / metres).sum())
                for threshold in (10, 30):
                    tall = metres[floors >= threshold]
                    record[f"nearest_over_{threshold}_m"] = (float(tall.min()) if len(tall)
                                                            else float(args.patch_m))
                # land use as shares of the buildings standing there, and as floor-weighted
                # shares, because one office tower is not one house
                counts = near.use.value_counts()
                weighted = near.groupby("use").floors.sum()
                for use in USES + ["other"]:
                    key = use.replace(" ", "_").replace("/", "_").replace("-", "").lower()
                    record[f"share_{key}"] = float(counts.get(use, 0)) / len(near)
                    record[f"floorshare_{key}"] = float(weighted.get(use, 0.0)) / float(
                        near.floors.sum() or 1.0)
            rows.append(record)

    table = pd.DataFrame(rows).fillna(0.0)
    table["nearest_over_10_m"] = table.nearest_over_10_m.fillna(args.patch_m)
    table["nearest_over_30_m"] = table.nearest_over_30_m.fillna(args.patch_m)
    covered = (table.groupby("SITE_NO").buildings.min() >= args.min_buildings)
    keep = sorted(covered[covered].index.tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"site_year_attributes_sq{int(args.patch_m)}m.csv", index=False)
    pd.DataFrame({"SITE_NO": keep}).to_csv(OUT / "covered_sites.csv", index=False)

    print(f"{'year':>6}{'sites with >=1 building':>26}{'median buildings':>19}"
          f"{'median floors_max':>20}")
    for year, group in table.groupby("year"):
        hit = group[group.buildings > 0]
        print(f"{year:>6}{len(hit):>26}{hit.buildings.median():>19.0f}"
              f"{hit.floors_max.median():>20.0f}")
    print(f"\ncovered sites (>= {args.min_buildings} buildings in every year): "
          f"{len(keep)} of {len(sites)}")
    print(f"feature columns: {len([c for c in table.columns if c not in ('SITE_NO','year')])}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
