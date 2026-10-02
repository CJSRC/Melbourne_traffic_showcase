"""Give every SCATS site a road class, because the project has none.

The supervisor's point was that a response of four vehicles per fifteen minutes looks small
only because it averages a main road with a back street, and that restricting to main roads
should lift it. Nothing in the project can make that split: `area_sites_attr.csv` carries a
TYPE column that reads INT for every row, and the site name is only a street pair.

So the class has to come from OpenStreetMap. One Overpass query covers the whole study area
rather than 371 separate ones, and the ways come back with geometry, `lanes` and `maxspeed`
attached -- the last two are not needed for the split but are exactly what the road-network
project will want next, and asking for them now costs nothing.

A SCATS site sits at an intersection, so its class is the highest class among the roads that
pass close to it. Distance is measured in metres on a local projection, not in degrees.

    python outputs/experiment_ledger_v1/osm_site_road_class.py
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

SCRATCH = pathlib.Path("C:/Users/LIANGZ~1/AppData/Local/Temp/claude/D--Melbourne/"
                       "6730435c-09fe-491f-9366-7890a4f73ed3/scratchpad")
SITES = SCRATCH / "area_sites_trainable.csv"
RAW = SCRATCH / "osm_ways.json"
OUT = SCRATCH / "site_road_class.csv"
ENDPOINT = "https://overpass-api.de/api/interpreter"
# Overpass refuses a request without a real User-Agent, and rate-limits by IP
UA = "melbourne-scats-research/1.0 (University of Melbourne)"

# highest first: a site takes the best class among the roads that reach it
CLASSES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
           "secondary", "secondary_link", "tertiary", "tertiary_link",
           "unclassified", "residential", "living_street", "service"]
RANK = {name: i for i, name in enumerate(CLASSES)}
MAIN = {"motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link"}
RADIUS_M = 40.0


def fetch(bbox: tuple[float, float, float, float]) -> dict:
    """One query for the whole area. Cached, because Overpass is a shared public service."""
    if RAW.is_file():
        print(f"  reusing {RAW.name}")
        return json.loads(RAW.read_text("utf-8"))
    south, west, north, east = bbox
    query = (
        "[out:json][timeout:300];"
        f'way["highway"~"^({"|".join(CLASSES)})$"]'
        f"({south:.5f},{west:.5f},{north:.5f},{east:.5f});"
        "out geom;"
    )
    print(f"  querying Overpass for {south:.3f},{west:.3f} .. {north:.3f},{east:.3f}")
    for attempt in range(6):
        response = requests.post(ENDPOINT, data={"data": query}, timeout=600,
                                 headers={"User-Agent": UA})
        if response.status_code == 200:
            RAW.write_text(response.text, "utf-8")
            return response.json()
        wait = 90 if response.status_code == 429 else 30
        print(f"    attempt {attempt + 1}: HTTP {response.status_code}, waiting {wait}s",
              flush=True)
        time.sleep(wait)
    raise SystemExit("Overpass did not answer")


def main() -> None:
    # the settled model's 81 junctions are not a subset of the 371 trainable ones, so the list
    # is an argument: a figure that drops a junction for want of a class is a different figure
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=pathlib.Path, default=SITES)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    sites = pd.read_csv(args.sites)
    pad = 0.01
    data = fetch((sites.LATITUDE.min() - pad, sites.LONGITUDE.min() - pad,
                  sites.LATITUDE.max() + pad, sites.LONGITUDE.max() + pad))
    ways = [e for e in data["elements"] if e.get("type") == "way" and e.get("geometry")]
    print(f"  {len(ways)} ways")

    # metres, not degrees: MGA94 zone 55 covers Melbourne
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:28355", always_xy=True)
    site_x, site_y = to_m.transform(sites.LONGITUDE.to_numpy(), sites.LATITUDE.to_numpy())
    site_xy = np.column_stack([site_x, site_y])

    best = np.full(len(sites), len(CLASSES), dtype=int)
    lanes = np.full(len(sites), np.nan)
    speed = np.full(len(sites), np.nan)
    names: list[set[str]] = [set() for _ in range(len(sites))]

    for way in ways:
        tags = way.get("tags", {})
        rank = RANK.get(tags.get("highway", ""), len(CLASSES))
        if rank >= len(CLASSES):
            continue
        lon = np.array([p["lon"] for p in way["geometry"]])
        lat = np.array([p["lat"] for p in way["geometry"]])
        wx, wy = to_m.transform(lon, lat)
        pts = np.column_stack([wx, wy])
        # bounding-box reject first; the full distance matrix over every way would be huge
        near = ((site_xy[:, 0] >= pts[:, 0].min() - RADIUS_M) &
                (site_xy[:, 0] <= pts[:, 0].max() + RADIUS_M) &
                (site_xy[:, 1] >= pts[:, 1].min() - RADIUS_M) &
                (site_xy[:, 1] <= pts[:, 1].max() + RADIUS_M))
        if not near.any():
            continue
        idx = np.flatnonzero(near)
        d = np.sqrt(((site_xy[idx, None, :] - pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        hit = idx[d <= RADIUS_M]
        for i in hit:
            if rank < best[i]:
                best[i] = rank
                lanes[i] = float(tags["lanes"]) if str(tags.get("lanes", "")).isdigit() else np.nan
                raw = str(tags.get("maxspeed", "")).split()
                speed[i] = float(raw[0]) if raw and raw[0].isdigit() else np.nan
            if tags.get("name"):
                names[i].add(tags["name"])

    sites["osm_class"] = [CLASSES[b] if b < len(CLASSES) else "none" for b in best]
    sites["road_group"] = ["main" if c in MAIN else ("local" if c != "none" else "none")
                           for c in sites.osm_class]
    sites["osm_lanes"] = lanes
    sites["osm_maxspeed"] = speed
    sites["osm_names"] = ["; ".join(sorted(n)) for n in names]
    sites.to_csv(args.out, index=False)

    print(f"\n  {'class':<18}{'sites':>7}")
    for name, count in sites.osm_class.value_counts().items():
        print(f"  {name:<18}{count:>7}")
    print(f"\n  main {int((sites.road_group == 'main').sum())}   "
          f"local {int((sites.road_group == 'local').sum())}   "
          f"unmatched {int((sites.road_group == 'none').sum())}")
    print(f"  lanes known for {int((~np.isnan(lanes)).sum())}, "
          f"speed limit for {int((~np.isnan(speed)).sum())} of {len(sites)}")

    print("\n  spot check -- SCATS name against the OSM street names found at that point")
    for _, row in sites.sample(10, random_state=42).iterrows():
        print(f"    {row.SITE_NO:<8}{str(row.SITE_NAME)[:26]:<28}{row.osm_class:<14}"
              f"{str(row.osm_names)[:44]}")
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
