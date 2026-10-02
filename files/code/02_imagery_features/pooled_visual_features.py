"""Rebuild the visual features so they do not depend on matching regions across years.

The 2846-column table lays each patch's regions into 104 fixed slots, ordered by SAM's own
ranking. That only works if slot seven means the same piece of ground in 2025 as in 2026, and
it does not: a different capture angle, season or light gives SAM a different segmentation, so
a car park split in two one year merges the next and everything after it shifts. Sorting by
something else does not fix it -- when the segmenter itself is not stable across years, no
ordering can make the slots correspond.

So drop correspondence as a requirement. Everything here is a statistic over the region set,
which does not care what order the regions arrive in or how many there are:

    38   the whole-patch summaries the old table already carried
    16   the mean DINO vector over all regions
    96   the mean DINO vector within each of six semantic classes
    24   area-weighted DINO mean, and how the regions sit relative to the detector

2846 -> 174, and the DINO information survives -- which the 38 summaries alone would not have,
since they are entirely SAM-side counts and areas.

Region position is included because the token table has it and the site-year table threw it
away: centroid_x and centroid_y say whether a roof sits at the junction or at the frame edge,
and 128 px is the detector at the patch centre.

    python outputs/phase2_offnadir_masking_v1/pooled_visual_features.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REGIONS = PROJECT / "outputs/phase2_cf_generated_v1/embeddings/regions_original.csv"
FULL = PROJECT / "outputs/phase2_cf_generated_v1/embeddings/features_original.csv"
OUT = HERE / "pooled_visual"

CLASSES = ["vegetation", "building_roof", "road", "parking_paved", "water", "other"]
# The detector sits at the centre of the patch, so the centre in pixels is half the width,
# and a pixel is the patch's ground width divided by that same number. Both are arguments
# now: the archive is 256 px covering 121 m, the rebuild is 1024 px covering 500 m, and
# hard-coding either turns every distance into a wrong number rather than a missing one.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rings", type=float, nargs="+", default=[60.0, 120.0, 250.0],
                        help="distance bands in metres for the roof-area histogram")
    parser.add_argument("--region-csv", type=Path, default=REGIONS)
    parser.add_argument("--summary-csv", type=Path, default=FULL,
                        help="table to take the 38 whole-patch summaries from; omit to skip")
    parser.add_argument("--image-px", type=int, default=256)
    parser.add_argument("--patch-m", type=float, default=121.0)
    parser.add_argument("--out-csv", type=Path, default=OUT / "site_year_pooled_visual.csv")
    # Keeping only the regions near the detector, so the same imagery can be pooled over a
    # smaller footprint. Every column here is a mean or a sum over whatever regions are in
    # the patch, so widening the patch spreads the same statistic over more ground -- a
    # 500 m mean of roof area says much less about one junction than a 121 m mean does. This
    # separates that effect from the imagery itself.
    parser.add_argument("--max-region-distance-m", type=float, default=None)
    args = parser.parse_args()
    centre = args.image_px / 2.0
    metres_per_px = args.patch_m / args.image_px

    regions = pd.read_csv(args.region_csv)
    dino = [c for c in regions.columns if c.startswith("dino_")]
    print(f"{len(regions)} regions, {len(dino)} DINO components, "
          f"{regions.SITE_NO.nunique()} sites, years {sorted(regions.year.unique())}")

    regions["distance_m"] = np.hypot(regions.centroid_x - centre,
                                     regions.centroid_y - centre) * metres_per_px
    regions["label"] = np.where(regions.semantic_label.isin(CLASSES),
                                regions.semantic_label, "other")
    if args.max_region_distance_m is not None:
        before = len(regions)
        regions = regions[regions.distance_m <= args.max_region_distance_m].copy()
        print(f"kept {len(regions)} of {before} regions within "
              f"{args.max_region_distance_m:.0f} m of the detector")

    rows = []
    for (site, year), group in regions.groupby(["SITE_NO", "year"]):
        vectors = group[dino].to_numpy(dtype=np.float32)
        weights = group.area_ratio.to_numpy(dtype=np.float32)
        record = {"SITE_NO": int(site), "year": int(year)}

        for index, value in enumerate(vectors.mean(axis=0)):
            record[f"dino_mean_{index}"] = float(value)
        total = float(weights.sum()) or 1.0
        for index, value in enumerate((vectors * weights[:, None]).sum(axis=0) / total):
            record[f"dino_areaw_{index}"] = float(value)
        for name in CLASSES:
            mask = (group.label == name).to_numpy()
            block = vectors[mask].mean(axis=0) if mask.any() else np.zeros(len(dino), np.float32)
            for index, value in enumerate(block):
                record[f"dino_{name}_{index}"] = float(value)

        # where things sit, which the slot table dropped
        distance = group.distance_m.to_numpy(dtype=np.float32)
        record["region_distance_mean"] = float(distance.mean())
        record["region_distance_areaw"] = float((distance * weights).sum() / total)
        roofs = group.label == "building_roof"
        far = args.patch_m / 2.0
        record["roof_distance_min"] = float(distance[roofs].min()) if roofs.any() else far
        record["roof_area_total"] = float(weights[roofs.to_numpy()].sum())
        edges = [0.0, *args.rings]
        for lower, upper in zip(edges[:-1], edges[1:]):
            band = roofs.to_numpy() & (distance >= lower) & (distance < upper)
            record[f"roof_area_{int(lower)}_{int(upper)}m"] = float(weights[band].sum())
        rows.append(record)

    pooled = pd.DataFrame(rows)
    if args.summary_csv and Path(args.summary_csv).is_file():
        summaries = pd.read_csv(args.summary_csv).drop_duplicates(["SITE_NO", "year"])
        keep = ["SITE_NO", "year"] + [c for c in summaries.columns
                                      if not c.startswith("region_")
                                      and c not in ("SITE_NO", "year")]
        table = summaries[keep].merge(pooled, on=["SITE_NO", "year"], how="inner")
    else:
        print("no summary table given; the 38 whole-patch columns are omitted")
        keep = ["SITE_NO", "year"]
        table = pooled
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)

    width = table.shape[1] - 2
    print(f"\n{len(table)} site-years, {width} features (was 2846)")
    print(f"  {len(keep) - 2:>4}  SAM whole-patch summaries")
    print(f"  {len(dino):>4}  mean DINO")
    print(f"  {len(dino):>4}  area-weighted DINO")
    print(f"  {len(CLASSES) * len(dino):>4}  per-class DINO ({', '.join(CLASSES)})")
    print(f"  {width - (len(keep) - 2) - len(dino) * (2 + len(CLASSES)):>4}  region position "
          f"and roof-area rings")
    print(f"\nregions per site-year: median {regions.groupby(['SITE_NO','year']).size().median():.0f}"
          f", min {regions.groupby(['SITE_NO','year']).size().min()}"
          f", max {regions.groupby(['SITE_NO','year']).size().max()}"
          f"  -- none of which changes the width now")
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
