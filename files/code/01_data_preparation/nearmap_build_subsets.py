"""Cut the site lists each arm can actually train on, and say why each site was dropped.

Four inputs constrain the experiment, and after the rebuilds only one of them still binds:

    imagery      Nearmap, 2014-2026, all 143 sites
    flow         the target, now 2014-2026 with no gap. 2019 was never missing -- its zip
                 uses a compression method Python's zipfile rejects, so an earlier pass
                 skipped it silently and the merged table inherited the hole. The early
                 years carry 139 sites rather than 143.
    net/traffic  Route2's JSON inputs, rebuilt for every year, so Route2 no longer has to
                 sit in a shorter window than Route1
    attributes   CLUE, 2014-2024, and only 81 sites have any buildings in range at all --
                 the City of Melbourne census stops at the municipal boundary and 62 of our
                 junctions sit outside it, where every statistic is a zero

So the binding constraint is the attribute footprint, not the imagery it was supposed to be.

Training on a site whose attribute vector is all zeros does not test the attributes; it
feeds the network a constant and dilutes the sites that do carry a signal. So the attribute
arms take the 81, not the 143.

Note this recovers 17 sites the earlier attr_64 list left out. Those 17 have real CLUE
attributes that vary across years -- across-year standard deviations from 4.3 to 389 -- so
they were not excluded for want of data, and leaving them out cost a quarter of the sample
in exactly the experiment that was already short of it.

    python outputs/phase2_offnadir_masking_v1/nearmap_build_subsets.py
"""

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE = PROJECT / "outputs/timestamp_conditioned_siteyear_slot_graph_wayback_gapfill_v1"
ATTR = HERE / "attribute_features"
FLOW = ATTR / "slot_wide_2014_2026_with2019.parquet"
OUT = HERE / "subsets"


def adjacency_for(full: pd.DataFrame, keep: list[int]) -> pd.DataFrame:
    """Keep the rows and columns of the retained sites, in the retained order.

    The matrix is square and labelled by site number on both axes, so a subset is a
    symmetric selection. Doing it any other way leaves the graph pointing at sites the
    trainer will never see.
    """
    index_column = full.columns[0]
    frame = full.set_index(index_column)
    frame.index = frame.index.astype(int)
    frame.columns = [int(c) for c in frame.columns]
    block = frame.loc[keep, keep]
    block.index.name = index_column
    return block.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(BASE / "area_sites.csv")
    adjacency = pd.read_csv(BASE / "area_adjacency.csv")

    flow = pd.read_parquet(FLOW, columns=["SITE_NO", "date"])
    flow["year"] = pd.to_datetime(flow.date).dt.year
    # A site is only usable if it has flow in every year the model will see. 2020 and 2021
    # are not among them: COVID moved the median daily total from 28,016 to 18,931 and back
    # to 25,117, and year_norm -- one linear scalar, constant across the day -- cannot
    # express a dip. Landformer drops the same two years for the same reason.
    used_years = [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]
    per_year = {year: set(flow[flow.year == year].SITE_NO.astype(int))
                for year in used_years}
    complete = set.intersection(*per_year.values())
    print("flow coverage in the years the model will see:")
    for year in used_years:
        print(f"  {year}  {len(per_year[year]):>4} sites")
    print(f"  present in all of them: {len(complete)}")

    attributes = pd.read_csv(ATTR / "site_year_attributes_sq500m.csv")
    value_columns = [c for c in attributes.columns if c not in ("SITE_NO", "year")]
    magnitude = attributes.groupby("SITE_NO")[value_columns].apply(
        lambda g: float(np.abs(g.to_numpy(float)).sum()))
    with_clue = set(int(s) for s in magnitude[magnitude > 0].index)

    all_sites = set(sites.SITE_NO.astype(int))
    print(f"\n{len(all_sites)} sites in the study area")
    print(f"  usable flow in every year  {len(complete):>4}   "
          f"(dropped {sorted(all_sites - complete)})")
    print(f"  CLUE attributes            {len(with_clue):>4}   "
          f"({len(all_sites - with_clue)} sites are all-zero, outside the CoM boundary)")

    # Route1 and Route2 now share a site list and a split: rebuilding the JSON inputs for
    # every year removed the reason Route2 had to live in 2020-2026 on its own.
    definitions = {
        "visual": sorted(complete),
        # anything using attributes is limited to the sites CLUE describes at all
        "attr": sorted(complete & with_clue),
    }

    print()
    for name, keep in definitions.items():
        site_frame = sites[sites.SITE_NO.astype(int).isin(keep)].copy()
        site_frame["site_idx"] = range(len(site_frame))
        site_frame.to_csv(OUT / f"area_sites_{name}.csv", index=False)
        adjacency_for(adjacency, keep).to_csv(OUT / f"area_adjacency_{name}.csv", index=False)
        print(f"  {name:<11} {len(keep):>4} sites -> area_sites_{name}.csv, "
              f"area_adjacency_{name}.csv")

    years = {"visual": used_years, "attr": used_years}
    print()
    print(f"{'subset':<11}{'sites':>7}{'years':>7}{'site-years':>12}   years")
    print("-" * 78)
    for name, keep in definitions.items():
        span = years[name]
        print(f"{name:<11}{len(keep):>7}{len(span):>7}{len(keep) * len(span):>12}   "
              f"{span[0]}-{span[-1]}" + (" (no 2019)" if 2019 not in span
                                         and span[0] < 2019 else ""))
    pd.DataFrame([{"subset": n, "n_sites": len(k), "years": " ".join(map(str, years[n])),
                   "site_years": len(k) * len(years[n])}
                  for n, k in definitions.items()]).to_csv(
        OUT / "subset_summary.csv", index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

