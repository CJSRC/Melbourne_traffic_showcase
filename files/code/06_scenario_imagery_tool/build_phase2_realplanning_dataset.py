from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

from baseline_pipeline import write_json


YEAR_TO_SATELLITE_ROOT = {
    2020: Path("outputs/site_year_satellite_2020_exact_full_v1"),
    2021: Path("outputs/site_year_satellite_2021_exact_full_v1"),
    2022: Path("outputs/site_year_satellite_2022_exact_full_v1"),
    2023: Path("outputs/site_year_satellite_2023_greenline_full_v1"),
    2024: Path("outputs/site_year_satellite_wayback_2024_full_v1"),
    2025: Path("outputs/site_year_satellite_wayback_2025_full_v1"),
    2026: Path("outputs/site_year_satellite_wayback_2026_full_v1"),
}

WFS_URL = "https://opendata.maps.vic.gov.au/geoserver/wfs"
LAYER_PLAN_ZONE = "open-data-platform:plan_zone"
LAYER_PLAN_OVERLAY = "open-data-platform:plan_overlay"
LAYER_ROAD = "open-data-platform:tr_road"
LAYER_RAIL = "open-data-platform:tr_rail"
LAYER_WATER = "open-data-platform:hy_water_area_polygon"
LAYER_BUILDING = "open-data-platform:building_polygon"

COM_BUILDING_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/2023-building-footprints/exports/geojson"
COM_DEV_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/development-activity-model-footprints/exports/geojson"

RESIDENTIAL_CODES = {"GRZ", "NRZ", "RGZ", "LDRZ", "RZ", "MUZ"}
COMMERCIAL_CODES = {"ACZ", "CCZ", "CDZ", "B1Z", "B2Z", "B3Z", "B4Z", "MUZ", "CZ", "FCZ"}
TRANSPORT_CODES = {"TRZ1", "TRZ2", "PUZ4", "PCRZ", "PPRZ"}
DEVELOPMENT_OVERLAY_PREFIX = ("DDO", "CDO", "IPO", "UGZ", "SBO", "ESO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official-planning-input site-year rasters aligned to satellite patch geometry")
    parser.add_argument("--study-area-csv", type=Path, required=True)
    parser.add_argument("--patch-manifest-dir", type=Path, required=True)
    parser.add_argument("--siteyear-feature-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def meters_per_degree_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def latlon_to_pixel(lat: float, lon: float, *, north: float, south: float, west: float, east: float, size_px: int) -> tuple[float, float]:
    x = (lon - west) / max(east - west, 1e-9) * (size_px - 1)
    y = (north - lat) / max(north - south, 1e-9) * (size_px - 1)
    return x, y


def bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    south_a, west_a, north_a, east_a = a
    south_b, west_b, north_b, east_b = b
    return not (north_a < south_b or north_b < south_a or east_a < west_b or east_b < west_a)


def geometry_bounds(geometry: dict) -> tuple[float, float, float, float]:
    coords = []
    geom_type = geometry.get("type")
    if geom_type == "Polygon":
        for ring in geometry["coordinates"]:
            coords.extend(ring)
    elif geom_type == "MultiPolygon":
        for poly in geometry["coordinates"]:
            for ring in poly:
                coords.extend(ring)
    elif geom_type == "LineString":
        coords.extend(geometry["coordinates"])
    elif geom_type == "MultiLineString":
        for line in geometry["coordinates"]:
            coords.extend(line)
    else:
        return (-90.0, -180.0, 90.0, 180.0)
    lons = [float(x) for x, _ in coords]
    lats = [float(y) for x, y in coords]
    return min(lats), min(lons), max(lats), max(lons)


def fetch_geojson(url: str, cache_path: Path, timeout: int, force: bool) -> dict:
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_wfs_layer(layer_name: str, bbox: tuple[float, float, float, float], cache_path: Path, timeout: int, force: bool) -> dict:
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    south, west, north, east = bbox
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": layer_name,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{west},{south},{east},{north},EPSG:4326",
    }
    response = requests.get(WFS_URL, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def iter_lines(geometry: dict) -> Iterable[list[tuple[float, float]]]:
    geom_type = geometry.get("type")
    if geom_type == "LineString":
        yield [(float(lat), float(lon)) for lon, lat in geometry["coordinates"]]
    elif geom_type == "MultiLineString":
        for line in geometry["coordinates"]:
            yield [(float(lat), float(lon)) for lon, lat in line]


def iter_polygons(geometry: dict) -> Iterable[list[list[tuple[float, float]]]]:
    geom_type = geometry.get("type")
    if geom_type == "Polygon":
        yield [[(float(lat), float(lon)) for lon, lat in ring] for ring in geometry["coordinates"]]
    elif geom_type == "MultiPolygon":
        for poly in geometry["coordinates"]:
            yield [[(float(lat), float(lon)) for lon, lat in ring] for ring in poly]


def to_pixel_ring(ring: list[tuple[float, float]], *, north: float, south: float, west: float, east: float, size_px: int) -> list[tuple[float, float]]:
    return [latlon_to_pixel(lat, lon, north=north, south=south, west=west, east=east, size_px=size_px) for lat, lon in ring]


def zone_channel(properties: dict) -> int | None:
    code = str(properties.get("zone_code", "")).upper()
    desc = str(properties.get("zone_description", "")).upper()
    prefix = code[:3]
    if prefix in RESIDENTIAL_CODES or "RESIDENTIAL" in desc:
        return 4
    if prefix in COMMERCIAL_CODES or "COMMERCIAL" in desc or "ACTIVITY" in desc or "MIXED USE" in desc:
        return 5
    if code in TRANSPORT_CODES or "TRANSPORT" in desc or "ROAD NETWORK" in desc:
        return 6
    return None


def is_development_overlay(properties: dict) -> bool:
    code = str(properties.get("zone_code", "")).upper()
    return code.startswith(DEVELOPMENT_OVERLAY_PREFIX)


def development_status_weight(properties: dict) -> int:
    status = str(properties.get("status", "")).upper()
    mapping = {
        "APPLIED": 96,
        "APPROVED": 160,
        "UNDER CONSTRUCTION": 224,
        "COMPLETE": 128,
    }
    return mapping.get(status, 96)


def road_width_and_value(properties: dict) -> tuple[int, int]:
    road_type = str(properties.get("road_type", "")).upper()
    feature_type_code = str(properties.get("feature_type_code", "")).lower()
    if "FREEWAY" in road_type or "HIGHWAY" in road_type:
        return 6, 255
    if road_type in {"ROAD", "AVENUE", "BOULEVARD"}:
        return 4, 200
    if feature_type_code == "road":
        return 2, 120
    return 1, 80


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def make_preview(channels: np.ndarray) -> np.ndarray:
    preview = np.zeros((channels.shape[1], channels.shape[2], 3), dtype=np.uint8)
    preview[..., 0] = np.maximum.reduce([channels[0], channels[7], channels[9]])
    preview[..., 1] = np.maximum.reduce([channels[4], channels[5], channels[8]])
    preview[..., 2] = np.maximum.reduce([channels[1], channels[2], channels[3]])
    return preview


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    area = pd.read_csv(args.study_area_csv)
    area["SITE_NO"] = pd.to_numeric(area["SITE_NO"], errors="coerce").astype(int)
    area = area.sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

    manifests = []
    for year in args.years:
        manifest = pd.read_csv(args.patch_manifest_dir / f"patch_manifest_year_{year}.csv")
        manifest["SITE_NO"] = pd.to_numeric(manifest["SITE_NO"], errors="coerce").astype(int)
        manifest["year"] = int(year)
        manifests.append(manifest)
    manifest_df = pd.concat(manifests, ignore_index=True)

    feature_df = pd.read_csv(args.siteyear_feature_csv)
    feature_df["SITE_NO"] = pd.to_numeric(feature_df["SITE_NO"], errors="coerce").astype(int)
    feature_df["year"] = pd.to_numeric(feature_df["year"], errors="coerce").astype(int)
    feature_df = feature_df[feature_df["year"].isin(args.years)].copy()
    feature_cols = [column for column in feature_df.columns if column not in {"SITE_NO", "year"}]

    merged = manifest_df.merge(feature_df, on=["SITE_NO", "year"], how="inner")
    merged["dataset_split"] = np.select(
        [merged["year"] <= 2024, merged["year"] == 2025, merged["year"] == 2026],
        ["train", "val", "test"],
        default="other",
    )

    half_patch_m = float(merged["patch_size_m"].iloc[0]) / 2.0
    lat_pad = (half_patch_m + 100.0) / 111_320.0
    mean_lat = float(area["LATITUDE"].mean())
    lon_pad = (half_patch_m + 100.0) / meters_per_degree_lon(mean_lat)
    south = float(area["LATITUDE"].min()) - lat_pad
    north = float(area["LATITUDE"].max()) + lat_pad
    west = float(area["LONGITUDE"].min()) - lon_pad
    east = float(area["LONGITUDE"].max()) + lon_pad
    bbox = (south, west, north, east)

    cache_dir = out_dir / "cache"
    raw_layers = {
        "plan_zone": fetch_wfs_layer(LAYER_PLAN_ZONE, bbox, cache_dir / "plan_zone.geojson", args.timeout_seconds, args.force_refresh),
        "plan_overlay": fetch_wfs_layer(LAYER_PLAN_OVERLAY, bbox, cache_dir / "plan_overlay.geojson", args.timeout_seconds, args.force_refresh),
        "tr_road": fetch_wfs_layer(LAYER_ROAD, bbox, cache_dir / "tr_road.geojson", args.timeout_seconds, args.force_refresh),
        "tr_rail": fetch_wfs_layer(LAYER_RAIL, bbox, cache_dir / "tr_rail.geojson", args.timeout_seconds, args.force_refresh),
        "water": fetch_wfs_layer(LAYER_WATER, bbox, cache_dir / "water.geojson", args.timeout_seconds, args.force_refresh),
        "building_polygon": fetch_wfs_layer(LAYER_BUILDING, bbox, cache_dir / "building_polygon.geojson", args.timeout_seconds, args.force_refresh),
        "com_buildings": fetch_geojson(COM_BUILDING_URL, cache_dir / "com_buildings.geojson", args.timeout_seconds, args.force_refresh),
        "com_development": fetch_geojson(COM_DEV_URL, cache_dir / "com_development.geojson", args.timeout_seconds, args.force_refresh),
    }

    rows = []
    min_year = min(args.years)
    max_year = max(args.years)

    for row in merged.itertuples(index=False):
        size_px = int(row.image_size_px)
        half_lat = half_patch_m / 111_320.0
        half_lon = half_patch_m / meters_per_degree_lon(float(row.LATITUDE))
        north_patch = float(row.LATITUDE) + half_lat
        south_patch = float(row.LATITUDE) - half_lat
        west_patch = float(row.LONGITUDE) - half_lon
        east_patch = float(row.LONGITUDE) + half_lon
        patch_bbox = (south_patch, west_patch, north_patch, east_patch)

        channel_images = [Image.new("L", (size_px, size_px), 0) for _ in range(12)]

        road_draw_any = ImageDraw.Draw(channel_images[0])
        road_draw_hier = ImageDraw.Draw(channel_images[1])
        rail_draw = ImageDraw.Draw(channel_images[2])

        for feature in raw_layers["tr_road"].get("features", []):
            geom = feature.get("geometry")
            if not geom or not bbox_overlap(geometry_bounds(geom), patch_bbox):
                continue
            width, value = road_width_and_value(feature.get("properties", {}))
            for line in iter_lines(geom):
                pts = [latlon_to_pixel(lat, lon, north=north_patch, south=south_patch, west=west_patch, east=east_patch, size_px=size_px) for lat, lon in line]
                road_draw_any.line(pts, fill=255, width=1)
                road_draw_hier.line(pts, fill=value, width=width)

        for feature in raw_layers["tr_rail"].get("features", []):
            geom = feature.get("geometry")
            if not geom or not bbox_overlap(geometry_bounds(geom), patch_bbox):
                continue
            for line in iter_lines(geom):
                pts = [latlon_to_pixel(lat, lon, north=north_patch, south=south_patch, west=west_patch, east=east_patch, size_px=size_px) for lat, lon in line]
                rail_draw.line(pts, fill=255, width=2)

        for feature in raw_layers["water"].get("features", []):
            geom = feature.get("geometry")
            if not geom or not bbox_overlap(geometry_bounds(geom), patch_bbox):
                continue
            draw = ImageDraw.Draw(channel_images[3])
            for poly in iter_polygons(geom):
                exterior = to_pixel_ring(poly[0], north=north_patch, south=south_patch, west=west_patch, east=east_patch, size_px=size_px)
                draw.polygon(exterior, fill=255)

        for feature in raw_layers["plan_zone"].get("features", []):
            geom = feature.get("geometry")
            if not geom or not bbox_overlap(geometry_bounds(geom), patch_bbox):
                continue
            ch = zone_channel(feature.get("properties", {}))
            if ch is None:
                continue
            draw = ImageDraw.Draw(channel_images[ch])
            for poly in iter_polygons(geom):
                exterior = to_pixel_ring(poly[0], north=north_patch, south=south_patch, west=west_patch, east=east_patch, size_px=size_px)
                draw.polygon(exterior, fill=255)

        for feature in raw_layers["plan_overlay"].get("features", []):
            geom = feature.get("geometry")
            if not geom or not bbox_overlap(geometry_bounds(geom), patch_bbox):
                continue
            if not is_development_overlay(feature.get("properties", {})):
                continue
            draw = ImageDraw.Draw(channel_images[7])
            for poly in iter_polygons(geom):
                exterior = to_pixel_ring(poly[0], north=north_patch, south=south_patch, west=west_patch, east=east_patch, size_px=size_px)
                draw.polygon(exterior, fill=255)

        building_draw = ImageDraw.Draw(channel_images[8])
        height_draw = ImageDraw.Draw(channel_images[9])
        for source_name in ("building_polygon", "com_buildings"):
            for feature in raw_layers[source_name].get("features", []):
                geom = feature.get("geometry")
                if not geom or not bbox_overlap(geometry_bounds(geom), patch_bbox):
                    continue
                for poly in iter_polygons(geom):
                    exterior = to_pixel_ring(poly[0], north=north_patch, south=south_patch, west=west_patch, east=east_patch, size_px=size_px)
                    building_draw.polygon(exterior, fill=255)
                    if source_name == "com_buildings":
                        props = feature.get("properties", {})
                        height_val = float(props.get("structure_extrusion") or props.get("footprint_extrusion") or 0.0)
                        if height_val > 0:
                            draw_val = int(np.clip(height_val * 3.0, 0.0, 255.0))
                            height_draw.polygon(exterior, fill=draw_val)

        dev_any_draw = ImageDraw.Draw(channel_images[10])
        dev_status_draw = ImageDraw.Draw(channel_images[11])
        for feature in raw_layers["com_development"].get("features", []):
            geom = feature.get("geometry")
            if not geom or not bbox_overlap(geometry_bounds(geom), patch_bbox):
                continue
            weight = development_status_weight(feature.get("properties", {}))
            for poly in iter_polygons(geom):
                exterior = to_pixel_ring(poly[0], north=north_patch, south=south_patch, west=west_patch, east=east_patch, size_px=size_px)
                dev_any_draw.polygon(exterior, fill=255)
                dev_status_draw.polygon(exterior, fill=weight)

        year_plane = np.full((size_px, size_px), int(round(255.0 * ((int(row.year) - min_year) / max(max_year - min_year, 1)))), dtype=np.uint8)
        channels = np.stack([np.asarray(img, dtype=np.uint8) for img in channel_images] + [year_plane], axis=0)
        planning_float = (channels.astype(np.float32) / 255.0).astype(np.float32)

        raster_path = out_dir / "real_planning_raster" / f"year_{int(row.year)}" / f"{row.patch_id}__planning.npy"
        raster_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raster_path, planning_float)

        preview_path = out_dir / "real_planning_preview" / f"year_{int(row.year)}" / f"{row.patch_id}__planning_preview.png"
        save_png(make_preview(channels), preview_path)

        sat_root = root / YEAR_TO_SATELLITE_ROOT[int(row.year)]
        sat_path = sat_root / row.relative_image_path

        row_dict = {
            "SITE_NO": int(row.SITE_NO),
            "year": int(row.year),
            "SITE_NAME": row.SITE_NAME,
            "LATITUDE": float(row.LATITUDE),
            "LONGITUDE": float(row.LONGITUDE),
            "dataset_split": row.dataset_split,
            "patch_id": row.patch_id,
            "pseudo_cad_npy": str(raster_path),
            "pseudo_cad_preview_png": str(preview_path),
            "satellite_patch_png": str(sat_path),
        }
        for feature_col in feature_cols:
            row_dict[feature_col] = float(getattr(row, feature_col))
        rows.append(row_dict)

    index_df = pd.DataFrame(rows).sort_values(["year", "SITE_NO"]).reset_index(drop=True)
    index_df.to_csv(out_dir / "phase2_realplanning_dataset_index.csv", index=False)
    write_json(
        out_dir / "dataset_metadata.json",
        {
            "study_area_csv": str(args.study_area_csv),
            "patch_manifest_dir": str(args.patch_manifest_dir),
            "siteyear_feature_csv": str(args.siteyear_feature_csv),
            "years": [int(v) for v in args.years],
            "sample_count": int(len(index_df)),
            "feature_count": int(len(feature_cols)),
            "channels": [
                "road_centerlines",
                "road_hierarchy",
                "rail_tram",
                "water",
                "zone_residential",
                "zone_commercial_mixed",
                "zone_transport_special",
                "development_overlay",
                "building_footprint",
                "building_height_proxy",
                "development_any",
                "development_status",
                "year_plane",
            ],
        },
    )
    print(out_dir / "phase2_realplanning_dataset_index.csv")


if __name__ == "__main__":
    main()
