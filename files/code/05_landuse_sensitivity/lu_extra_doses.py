"""Fill in the doses the sensitivity design is missing, under the rule the existing tables used.

Office was dosed at 0.5, 1.5, 2 and 3, so its response can be fitted as a line and read for
monotonicity. Residential and retail were dosed once each, at 2. That asymmetry is a real
weakness: a model that only responds to residential at high doses would look flat at 2, and the
conclusion "it distinguishes use from volume" would rest on a single point per comparison use.

So residential and retail get the same four multipliers, and each gets a swap arm -- the
constant-floorspace version that moves the share alone. If office swap moves the prediction and
residential swap does not, that is the finding; if both move, the model is reading floorspace
share in general rather than office in particular, which would be worth knowing before the
result is presented as evidence about land use.

The construction rule is not invented here. It is recovered from the tables already on disk and
checked by rebuilding them: the script reproduces attr_office_x2.csv and attr_office_x2_swap.csv
from attr_base.csv and refuses to write anything if either fails to match. That is the only way
the new doses are comparable to the old ones.

    python outputs/experiment_ledger_v1/lu_extra_doses.py
"""

import pathlib

import numpy as np
import pandas as pd

SCRATCH = pathlib.Path("C:/Users/LIANGZ~1/AppData/Local/Temp/claude/D--Melbourne/"
                       "0c849c9d-9acc-4e93-af30-ad378575207c/scratchpad")
WORK = SCRATCH / "lu_extra"
KEY = ["SITE_NO", "year"]
SHARE = "floorshare_"
VOLUME = ["floors_sum", "floors_sum_inner", "floors_mean", "floors_sum_weighted"]
USES = {"office": "floorshare_office",
        "resi": "floorshare_residential_apartment",
        "retail": "floorshare_retail__shop"}
MULTIPLIERS = [0.5, 1.5, 2.0, 3.0]


TARGET_YEAR = 2024
TRAIN_SITES = SCRATCH / "area_sites_attr.csv"


def share_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(SHARE)]


def dosed_rows(base: pd.DataFrame, column: str) -> np.ndarray:
    """Which rows the archived tables actually moved, recovered from them and then verified.

    Only the test year, only the junctions the model trains on, and only those with some of
    the use already present -- a site with no office cannot have its office doubled, and
    including it would dilute the response with rows that cannot move. Checked against
    attr_office_x2_swap.csv: exactly the 70 rows it changed.
    """
    train = set(pd.read_csv(TRAIN_SITES).SITE_NO.astype(int))
    return ((base.year == TARGET_YEAR).to_numpy()
            & base.SITE_NO.astype(int).isin(train).to_numpy()
            & (base[column].to_numpy(dtype=float) > 0))


def dose_add(base: pd.DataFrame, column: str, multiplier: float) -> pd.DataFrame:
    """Scale one use's floorspace, letting the total rise with it, then renormalise the shares.

    This is the arm where share and volume move together, which is what makes a swap arm
    necessary to tell them apart.
    """
    out = base.copy()
    live = dosed_rows(base, column)
    shares = share_columns(base)
    target = base[column].to_numpy(dtype=float)
    rest = base[shares].to_numpy(dtype=float).sum(axis=1) - target
    total = rest + target * multiplier
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(total > 0, 1.0 / total, 1.0)
    for c in shares:
        v = base[c].to_numpy(dtype=float)
        out[c] = np.where(live, np.where(c == column, v * multiplier, v) * scale, v)
    # the volume columns follow the floorspace that was added
    grew = np.where(rest + target > 0, total / np.maximum(rest + target, 1e-12), 1.0)
    for c in VOLUME:
        if c in base.columns:
            v = base[c].to_numpy(dtype=float)
            out[c] = np.where(live, v * grew, v)
    return out


def dose_swap(base: pd.DataFrame, column: str, multiplier: float) -> pd.DataFrame:
    """Raise one use's share by converting the others, with every volume column untouched."""
    out = base.copy()
    live = dosed_rows(base, column)
    shares = share_columns(base)
    target = base[column].to_numpy(dtype=float)
    rest = base[shares].to_numpy(dtype=float).sum(axis=1) - target
    wanted = np.minimum(target * multiplier, target + rest)
    with np.errstate(invalid="ignore", divide="ignore"):
        shrink = np.where(rest > 0, (rest - (wanted - target)) / rest, 1.0)
    for c in shares:
        v = base[c].to_numpy(dtype=float)
        out[c] = np.where(live, wanted if c == column else v * shrink, v)
    return out


def matches(a: pd.DataFrame, b: pd.DataFrame, tolerance: float = 5e-4) -> tuple[bool, str]:
    cols = [c for c in a.columns if c not in KEY]
    worst, name = 0.0, ""
    for c in cols:
        gap = float(np.nanmax(np.abs(a[c].to_numpy(dtype=float) - b[c].to_numpy(dtype=float))))
        if gap > worst:
            worst, name = gap, c
    return worst <= tolerance, f"largest difference {worst:.6f} on {name}"


def main() -> None:
    base = pd.read_csv(WORK / "attr_base.csv").sort_values(KEY).reset_index(drop=True)
    print(f"  base table: {len(base)} rows, {len(share_columns(base))} share columns")

    print("\n  recovering the construction rule by rebuilding tables that already exist")
    checks = [("attr_office_x2.csv", dose_add(base, USES["office"], 2.0)),
              ("attr_office_x2_swap.csv", dose_swap(base, USES["office"], 2.0)),
              ("attr_resi_x2.csv", dose_add(base, USES["resi"], 2.0)),
              ("attr_retail_x2.csv", dose_add(base, USES["retail"], 2.0))]
    ok = True
    for name, rebuilt in checks:
        archived = pd.read_csv(WORK / name).sort_values(KEY).reset_index(drop=True)
        rebuilt = rebuilt[archived.columns]
        good, detail = matches(archived, rebuilt)
        print(f"    {name:<28}{'reproduced' if good else 'DOES NOT MATCH':<16}{detail}")
        ok &= good
    if not ok:
        raise SystemExit("\n  the rule is not the one the archived tables used; not writing "
                         "anything, because doses built a different way are not comparable "
                         "to the existing ones")

    print("\n  writing the missing doses")
    written = []
    for use, column in (("resi", USES["resi"]), ("retail", USES["retail"])):
        for multiplier in MULTIPLIERS:
            tag = f"attr_{use}_x{multiplier:g}.csv"
            if (WORK / tag).is_file():
                print(f"    {tag:<28}already on disk")
                continue
            dose_add(base, column, multiplier).to_csv(WORK / tag, index=False)
            written.append(tag)
        tag = f"attr_{use}_x2_swap.csv"
        if not (WORK / tag).is_file():
            dose_swap(base, column, 2.0).to_csv(WORK / tag, index=False)
            written.append(tag)
    for tag in written:
        print(f"    {tag}")
    print(f"\n  {len(written)} new tables; {len(list(WORK.glob('attr_*.csv')))} in total")


if __name__ == "__main__":
    main()
