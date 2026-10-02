"""Collapse every sensitivity run to a profile per road group, day type and slot.

The supervisor's two criticisms both land here. Averaging a main road with a back street is
what made the response read as four vehicles per fifteen minutes, and the peak windows the
figures used were taken from habit rather than from the data. Both are fixed by aggregating
once, finely enough that the windows can be chosen afterwards and the groups compared.

The office arms were scored on Spartan and the residential and retail arms locally, so this
runs in both places and writes the same tidy table; the figure script merges them and checks
the two base arms agree before it does. Nothing here picks a peak window -- that is left to
whoever reads the profile, which is the point.

    python outputs/experiment_ledger_v1/lu_roadclass_aggregate.py --root <dir> --out <csv>
"""

import argparse
import pathlib
import re

import numpy as np
import pandas as pd

PATTERN = re.compile(r"^(?:lus|lux)_(r[12]attr)_(.+)_s(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="directory of run folders")
    parser.add_argument("--groups", required=True, help="csv with SITE_NO and road_group")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    groups = pd.read_csv(args.groups)[["SITE_NO", "road_group", "osm_class"]]
    binary = dict(zip(groups.SITE_NO, groups.road_group))
    by_class = dict(zip(groups.SITE_NO, groups.osm_class))
    print(f"  {len(binary)} sites: " +
          ", ".join(f"{k} {v}" for k, v in groups.road_group.value_counts().items()))
    print("  classes: " +
          ", ".join(f"{k} {v}" for k, v in groups.osm_class.value_counts().items()))

    rows = []
    runs = sorted(p for p in pathlib.Path(args.root).iterdir() if p.is_dir())
    for i, run in enumerate(runs, 1):
        m = PATTERN.match(run.name)
        path = run / "test_predictions.parquet"
        if not m or not path.is_file():
            continue
        arm, dose, seed = m.group(1), m.group(2), int(m.group(3))
        d = pd.read_parquet(path, columns=["SITE_NO", "date", "slot_idx", "y_pred", "y_true"])
        d["road_group"] = d.SITE_NO.map(binary)
        d = d[d.road_group.notna()]
        d["daytype"] = np.where(pd.to_datetime(d.date).dt.dayofweek >= 5, "weekend", "weekday")
        # three levels at once: "all" keeps the new numbers comparable with the old, the binary
        # split answers the supervisor's question, and the class shows the gradient behind it --
        # OSM tertiary in the CBD spans Collins Street and Little Bourke Street alike
        stacked = pd.concat([d.assign(road_group="all"), d,
                             d.assign(road_group=d.SITE_NO.map(by_class))], ignore_index=True)
        g = stacked.groupby(["road_group", "daytype", "slot_idx"]).agg(
            mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean"), n=("y_pred", "size"))
        g = g.reset_index()
        g["arm"], g["dose"], g["seed"] = arm, dose, seed
        rows.append(g)
        if i % 20 == 0 or i == len(runs):
            print(f"    {i}/{len(runs)}", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\n  {len(out)} rows, {out.dose.nunique()} doses, {out.seed.nunique()} seeds "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
