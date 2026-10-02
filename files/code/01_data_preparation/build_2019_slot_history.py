"""Build the 2019 slot history that every earlier pass skipped.

The flow archive has 2014-2018 and 2020-2026 and a hole at 2019, which looked like missing
data and is not. Traffic_Signal_Volume_Data_2019.zip is on disk, 1.36 GB, all 364 days --
but it is written with a compression method Python's zipfile refuses:

    NotImplementedError: That compression method is not supported

That is almost certainly why 2019 was dropped in the first place. The directory built next to
it is called slot_history_2014_2019_fill_v1 and its overview.json stops at 2017-12-31; someone
extracted the year by hand into manual_extract_2019 and the pipeline never picked it back up.

So this does not re-extract anything. The loose CSVs are already on disk, and the aggregation
that turns a day of detector readings into one row of 96 slots is imported from the original
builder rather than rewritten -- reimplementing it would risk 2019 being subtly unlike the
years it has to sit beside, which is a worse failure than the gap.

    python outputs/phase2_offnadir_masking_v1/build_2019_slot_history.py
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = PROJECT / "data/raw/vs2019"
FALLBACK = PROJECT / "outputs/longyear_2014_2026_prep_v1/manual_extract_2019"
SITES = PROJECT / "outputs/timestamp_conditioned_siteyear_slot_graph_wayback_gapfill_v1/area_sites.csv"
OUT = PROJECT / "outputs/slot_history_2019_fill_v1"

spec = importlib.util.spec_from_file_location(
    "builder", PROJECT / "build_square_area_slot_history.py")
builder = importlib.util.module_from_spec(spec)
sys.modules["builder"] = builder
spec.loader.exec_module(builder)


def main() -> None:
    source = RAW if RAW.is_dir() and any(RAW.glob("VSDATA_*.csv")) else FALLBACK
    days = sorted(source.glob("VSDATA_*.csv"))
    if not days:
        raise SystemExit(f"no VSDATA_*.csv under {RAW} or {FALLBACK}")
    sites = pd.read_csv(SITES)
    keep = set(pd.to_numeric(sites.SITE_NO, errors="coerce").dropna().astype(int))
    print(f"{len(days)} days from {source}")
    print(f"{len(keep)} study-area sites\n")

    parts, progress = [], []
    for position, day in enumerate(days, start=1):
        with day.open("r", newline="") as handle:
            frame = builder.aggregate_slot_wide_from_stream(
                handle, keep, chunksize=50000)
        progress.append({
            "zip_name": "Traffic_Signal_Volume_Data_2019 (extracted)",
            "source": day.name,
            "n_area_rows": len(frame),
            "n_sites_hit": int(frame.SITE_NO.nunique()) if len(frame) else 0,
            "date_min": str(frame.date.min()) if len(frame) else "",
            "date_max": str(frame.date.max()) if len(frame) else "",
        })
        if len(frame):
            parts.append(frame)
        if position % 30 == 0 or position == len(days):
            print(f"  {position}/{len(days)} days, {sum(len(p) for p in parts)} rows",
                  flush=True)

    wide = pd.concat(parts, ignore_index=True)
    wide = wide.groupby(["SITE_NO", "date"], as_index=False)[
        builder.VOLUME_COLS].sum(min_count=1)
    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT / "Traffic_Signal_Volume_Data_2019_site_slot_wide.parquet",
                    index=False)
    pd.DataFrame(progress).to_csv(OUT / "source_progress.csv", index=False)

    slots = wide[builder.VOLUME_COLS]
    print(f"\n{len(wide)} site-days, {wide.SITE_NO.nunique()} sites, "
          f"{wide.date.nunique()} dates")
    print(f"  {wide.date.min()} -> {wide.date.max()}")
    print(f"  missing slot readings: {int(slots.isna().sum().sum()):,} of "
          f"{slots.size:,} ({100 * slots.isna().sum().sum() / slots.size:.2f}%)")
    print(f"  daily total, median over site-days: {slots.sum(axis=1).median():,.0f} vehicles")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
