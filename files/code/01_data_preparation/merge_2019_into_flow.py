"""Check 2019 belongs beside its neighbours, then merge it in.

A rebuilt year can pass every structural test and still be wrong. If the hand-extracted CSVs
were partial -- a detector missing, a column shifted, half a day truncated -- the row count
and the date range both still look right and only the magnitude gives it away. So the merge
is gated on 2019 sitting between 2018 and 2020 rather than on the file existing.

Traffic does move year to year, so the test is deliberately loose: 2019's median daily total
has to fall within a quarter of the average of the years either side. A genuine trend will
pass that easily; a year that lost a third of its detectors will not.

    python outputs/phase2_offnadir_masking_v1/merge_2019_into_flow.py
"""

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EXISTING = HERE / "attribute_features/slot_wide_2014_2026.parquet"
NEW_2019 = PROJECT / "outputs/slot_history_2019_fill_v1/Traffic_Signal_Volume_Data_2019_site_slot_wide.parquet"
OUT = HERE / "attribute_features/slot_wide_2014_2026_with2019.parquet"
TOLERANCE = 0.25


def slot_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("SITE_NO", "date", "year", "site_idx")]


def profile(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    columns = slot_columns(frame)
    # to_numpy can hand back a read-only view of the parquet buffer, so copy before masking
    daily = np.array(frame[columns].to_numpy(dtype=np.float32), copy=True)
    daily[daily < 0] = np.nan
    table = pd.DataFrame({
        "SITE_NO": frame.SITE_NO.values,
        "year": pd.to_datetime(frame.date).dt.year.values,
        "date": frame.date.values,
        "daily_total": np.nansum(daily, axis=1),
    })
    summary = table.groupby("year").agg(
        days=("date", "nunique"), sites=("SITE_NO", "nunique"),
        median_daily=("daily_total", "median")).reset_index()
    summary["source"] = label
    return summary


def main() -> None:
    existing = pd.read_parquet(EXISTING)
    fresh = pd.read_parquet(NEW_2019)
    # the rebuilt year carries V00..V95; the archive carries whatever it carries. Align on
    # position rather than name, since a rename between builds would otherwise pass silently.
    old_columns, new_columns = slot_columns(existing), slot_columns(fresh)
    if len(old_columns) != len(new_columns):
        raise SystemExit(f"slot count differs: archive {len(old_columns)}, "
                         f"rebuilt 2019 {len(new_columns)}")
    fresh = fresh.rename(columns=dict(zip(new_columns, old_columns)))
    fresh["date"] = pd.to_datetime(fresh.date).dt.strftime("%Y-%m-%d")
    existing["date"] = pd.to_datetime(existing.date).dt.strftime("%Y-%m-%d")

    combined_profile = pd.concat([profile(existing, "archive"), profile(fresh, "rebuilt")])
    print(f"{'year':<7}{'days':>6}{'sites':>7}{'median daily total':>22}  source")
    print("-" * 62)
    for row in combined_profile.sort_values("year").itertuples():
        print(f"{row.year:<7}{row.days:>6}{row.sites:>7}{row.median_daily:>22,.0f}  {row.source}")

    levels = combined_profile.set_index("year").median_daily
    neighbours = (levels[2018] + levels[2020]) / 2.0
    drift = abs(levels[2019] - neighbours) / neighbours
    print(f"\n2019 median daily total {levels[2019]:,.0f} against neighbours' mean "
          f"{neighbours:,.0f}: {drift:+.1%}")
    if drift > TOLERANCE:
        raise SystemExit(f"2019 is {drift:.1%} away from its neighbours, past the "
                         f"{TOLERANCE:.0%} tolerance -- not merging. Check the extraction "
                         f"before trusting this year.")
    print(f"within the {TOLERANCE:.0%} tolerance, so 2019 is consistent with the years "
          f"around it")

    merged = pd.concat([existing, fresh[existing.columns]], ignore_index=True)
    merged = merged.sort_values(["SITE_NO", "date"]).reset_index(drop=True)
    merged.to_parquet(OUT, index=False)

    check = merged.copy()
    check["year"] = pd.to_datetime(check.date).dt.year
    coverage = check.groupby("year").agg(days=("date", "nunique"),
                                         sites=("SITE_NO", "nunique"))
    print(f"\nmerged: {len(merged):,} site-days")
    print(coverage.to_string())
    gaps = [y for y in range(2014, 2027) if y not in coverage.index]
    print(f"\nyears missing between 2014 and 2026: {gaps or 'none'}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
