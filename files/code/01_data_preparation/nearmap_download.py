"""Fetch the site patches from Nearmap, at the geometry the archive already uses.

Why this source replaces the ESRI archive:

    distinct years   Wayback releases are global basemap versions, so a release that did not
                     refresh Melbourne re-serves the previous pixels. 2014, 2015 and 2016 are
                     byte-identical; so are 2018 and 2019. Six years, three photographs.
                     Nearmap surveys are flights with dates, and all thirteen differ.
    one season       four to eleven captures a year means the season is a choice. The plan
                     picks each year's capture closest to the same calendar date, so the
                     imagery does not swing between summer glare and winter shadow.
    near nadir       roofs sit over their footprints instead of leaning off them, which is
                     the displacement the off-nadir masking was built to undo.

Why 121 m and 256 px rather than something finer: the pooled models were trained on patches
of exactly this size and ground resolution, and a Nearmap z18 tile covers 120.8 m at 0.472
m/px, so this is a swap of imagery with the geometry held fixed. Anything finer changes two
things at once and the comparison stops meaning anything. Finer runs are cheap to add later;
--patch-size-m and --image-size-px are arguments for that reason.

The key comes from NEARMAP_API_KEY and is never written to a file, a name or a log line.

    $env:NEARMAP_API_KEY = '...'
    python outputs/phase2_offnadir_masking_v1/nearmap_download.py --out E:/nearmap_v1
"""

import argparse
import io
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
PLAN = HERE / "nearmap_site_year_plan.csv"
TILE = "https://api.nearmap.com/tiles/v3/surveys/{survey}/Vert/{z}/{x}/{y}.jpg"
HEADERS = {"User-Agent": "Mozilla/5.0"}
KEY = os.environ.get("NEARMAP_API_KEY", "").strip()

_requests_made = 0
_counter_lock = threading.Lock()


def resolution(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** zoom


def world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 256 * 2 ** zoom
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def tile(session: requests.Session, survey: str, zoom: int, x: int, y: int,
         attempts: int = 3) -> Image.Image | None:
    global _requests_made
    for attempt in range(attempts):
        try:
            response = session.get(TILE.format(survey=survey, z=zoom, x=x, y=y),
                                   params={"apikey": KEY}, timeout=90, headers=HEADERS)
            with _counter_lock:
                _requests_made += 1
            if response.status_code == 200 and len(response.content) > 400:
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            if response.status_code in (429, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return None


def stitch(session: requests.Session, lat: float, lon: float, survey: str, zoom: int,
           patch_m: float, out_px: int) -> tuple[Image.Image | None, int]:
    """Cut the window the archive cuts: patch_m metres of ground, then resample to out_px.

    The bug this whole line of work started from was cutting image_size_px tile-pixels and
    calling it patch_size_m metres. Here the window is in metres from the start and out_px
    only ever describes the file that lands on disk.
    """
    span = patch_m / resolution(lat, zoom)
    cx, cy = world_px(lat, lon, zoom)
    left, top = cx - span / 2.0, cy - span / 2.0
    tx0, ty0 = int(left // 256), int(top // 256)
    tx1, ty1 = int((left + span) // 256), int((top + span) // 256)

    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    missing = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            piece = tile(session, survey, zoom, tx, ty)
            if piece is None:
                missing += 1
                continue
            canvas.paste(piece, ((tx - tx0) * 256, (ty - ty0) * 256))
    if missing:
        return None, missing
    ox, oy = left - tx0 * 256, top - ty0 * 256
    patch = canvas.crop((round(ox), round(oy), round(ox + span), round(oy + span)))
    if patch.size != (out_px, out_px):
        patch = patch.resize((out_px, out_px), Image.LANCZOS)
    return patch, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, default=PLAN)
    parser.add_argument("--years", type=int, nargs="*", default=None)
    parser.add_argument("--sites", type=int, nargs="*", default=None)
    parser.add_argument("--patch-size-m", type=float, default=121.0)
    parser.add_argument("--image-size-px", type=int, default=256)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default=None, help="defaults to ps<patch>m")
    args = parser.parse_args()
    if not KEY:
        sys.exit("NEARMAP_API_KEY is not set in this process")

    tag = args.tag or f"ps{int(args.patch_size_m)}m"
    plan = pd.read_csv(args.plan_csv)
    if args.years:
        plan = plan[plan.year.isin(args.years)]
    if args.sites:
        plan = plan[plan.SITE_NO.isin(args.sites)]

    sample_lat = float(plan.LATITUDE.iloc[0])
    native = resolution(sample_lat, args.zoom)
    span_px = args.patch_size_m / native
    per_patch = (int(span_px // 256) + 2) ** 2
    print(f"{len(plan)} patches, {plan.year.nunique()} years, {plan.SITE_NO.nunique()} sites")
    print(f"  z{args.zoom} native {native:.3f} m/px; window {span_px:.0f} native px "
          f"-> {args.image_size_px} px = {args.patch_size_m / args.image_size_px:.3f} m/px")
    print(f"  at most {per_patch} tiles per patch, so up to "
          f"{len(plan) * per_patch:,} requests\n")

    roots = {}
    for year in sorted(plan.year.unique()):
        directory = args.out / f"site_year_satellite_nearmap_{year}_{tag}_v1"
        (directory / "satellite" / f"year_{year}").mkdir(parents=True, exist_ok=True)
        block = plan[plan.year == year]
        (directory / "download_metadata.json").write_text(json.dumps({
            "provider": "nearmap",
            "tile_api": "https://api.nearmap.com/tiles/v3/surveys/{survey}/Vert/{z}/{x}/{y}.jpg",
            "survey_ids": sorted(set(block.survey_id)),
            "capture_dates": sorted(set(block.capture_date)),
            "patch_size_m": args.patch_size_m,
            "image_size_px": args.image_size_px,
            "zoom": args.zoom,
            "ground_resolution_m_per_px": args.patch_size_m / args.image_size_px,
            "note": "survey chosen per site as the capture closest to a fixed calendar "
                    "target, so season is held roughly constant across years",
        }, indent=2), encoding="utf-8")
        roots[int(year)] = directory

    jobs = []
    for row in plan.itertuples():
        year = int(row.year)
        name = (f"site_{int(row.SITE_NO)}__year_{year}__{tag}__z{args.zoom}.png")
        jobs.append((row, roots[year] / "satellite" / f"year_{year}" / name,
                     f"satellite/year_{year}/{name}"))

    started, done, failures = time.time(), 0, []
    records = []
    lock = threading.Lock()
    local = threading.local()

    def fetch(job):
        row, path, relative = job
        if not hasattr(local, "session"):
            local.session = requests.Session()
        if path.is_file():
            return row, relative, "exists", 0
        patch, missing = stitch(local.session, float(row.LATITUDE), float(row.LONGITUDE),
                                row.survey_id, args.zoom, args.patch_size_m,
                                args.image_size_px)
        if patch is None:
            return row, relative, "failed", missing
        patch.save(path)
        return row, relative, "downloaded", 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, relative, status, missing in pool.map(fetch, jobs):
            with lock:
                done += 1
                records.append({
                    "SITE_NO": int(row.SITE_NO), "year": int(row.year),
                    "capture_date": row.capture_date, "survey_id": row.survey_id,
                    "status": status, "missing_tiles": missing,
                    "satellite_patch_id": f"site_{int(row.SITE_NO)}__year_{int(row.year)}",
                    "satellite_relative_image_path": relative,
                    "LATITUDE": row.LATITUDE, "LONGITUDE": row.LONGITUDE,
                })
                if status == "failed":
                    failures.append((int(row.SITE_NO), int(row.year), missing))
                if done % 50 == 0 or done == len(jobs):
                    rate = done / max(time.time() - started, 1e-9)
                    print(f"  {done}/{len(jobs)}  {rate:.1f} patch/s  "
                          f"{_requests_made:,} requests  failed {len(failures)}", flush=True)

    index = pd.DataFrame(records)
    args.out.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out / "nearmap_patch_index.csv", index=False)
    pd.DataFrame([{"year": year, "patch_root": str(path)}
                  for year, path in sorted(roots.items())]).to_csv(
        args.out / "year_root_catalog.csv", index=False)

    print(f"\n{index.status.value_counts().to_dict()}")
    print(f"tile requests spent: {_requests_made:,}")
    for site, year, missing in failures[:10]:
        print(f"  failed site {site} year {year}: {missing} tiles missing")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
