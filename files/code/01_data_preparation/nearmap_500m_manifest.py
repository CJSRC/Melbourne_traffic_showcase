"""Write the manifest and year catalogue the segmentation pipeline reads, for the 500 m set.

The 121 m archive was the geometry the pooled models were trained on, and every Nearmap run
so far has held it fixed. The ESRI comparison says that was the wrong constant: at 121 m the
visual branch is noise -- adding it costs 0.55 MAE pooled and 3.83 slot -- and at 500 m it
becomes signal, best arm pooled at 51.54 against 60.73 for time alone. The field of view was
carrying the whole effect.

So this rebuilds the same 143 sites and the same seven years at 500 m and 1024 px, which is
0.488 m/px -- within a hundredth of the 121 m set's 0.472, so the resolution is held and the
window is what changes.

    python outputs/phase2_offnadir_masking_v1/nearmap_500m_manifest.py
"""

import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PLAN = HERE / "nearmap_site_year_plan.csv"
LOCAL = pathlib.Path("D:/Melbourne/nearmap_500m_v1")
REMOTE = "/data/gpfs/projects/punim2970/xhe13561/nearmap_v1/imagery500"
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
PATCH_M = 500.0
PIXELS = 1024
ZOOM = 18


def split_of(year: int) -> str:
    return "test" if year == 2026 else ("val" if year == 2025 else "train")


def main() -> None:
    plan = pd.read_csv(PLAN)
    plan = plan[plan.year.isin(YEARS)].copy()

    rows = pd.DataFrame({
        "SITE_NO": plan.SITE_NO.astype(int),
        "SITE_NAME": plan.SITE_NO.astype(int),
        "LATITUDE": plan.LATITUDE,
        "LONGITUDE": plan.LONGITUDE,
        "year": plan.year.astype(int),
        "dataset_split": plan.year.map(split_of),
        "patch_size_m": PATCH_M,
        "image_size_px": PIXELS,
        "zoom": ZOOM,
    })
    rows["satellite_patch_id"] = (
        "site_" + rows.SITE_NO.astype(str) + "__year_" + rows.year.astype(str))
    rows["satellite_relative_image_path"] = (
        "satellite/year_" + rows.year.astype(str) + "/site_" + rows.SITE_NO.astype(str)
        + "__year_" + rows.year.astype(str) + f"__ps{int(PATCH_M)}m__z{ZOOM}.png")
    rows = rows.sort_values(["year", "SITE_NO"]).reset_index(drop=True)

    out = LOCAL / "manifest.csv"
    rows.to_csv(out, index=False)

    catalog = pd.DataFrame({
        "year": YEARS,
        "patch_root": [f"{REMOTE}/site_year_satellite_nearmap_{y}_ps500m_v1" for y in YEARS],
    })
    catalog.to_csv(LOCAL / "year_root_catalog.csv", index=False)

    present = sum(1 for row in rows.itertuples()
                  if (LOCAL / f"site_year_satellite_nearmap_{row.year}_ps500m_v1"
                      / row.satellite_relative_image_path).is_file())
    print(f"{len(rows)} rows, {rows.SITE_NO.nunique()} sites, years {sorted(rows.year.unique())}")
    print(f"  split: " + ", ".join(f"{k} {v}" for k, v in rows.dataset_split.value_counts().items()))
    print(f"  images on disk so far: {present}/{len(rows)}")
    print(f"-> {out}")
    print(f"-> {LOCAL / 'year_root_catalog.csv'}")


if __name__ == "__main__":
    main()
