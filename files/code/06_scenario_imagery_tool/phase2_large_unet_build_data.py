from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import transform

from build_phase2_melbourne_spatial_cad_rasters import (
    MANUAL_BUILDING_ROIS,
    align_to_footprint,
    auto_plan_crop,
    choose_spatial_page,
    normalized_roi,
    square_canvas,
    structural_line_map,
    trim_crop_to_ink,
)
from validate_phase2_real_building_events import rasterize_footprint


TILE_SIZE = 256
WEB_MERCATOR_INITIAL_RESOLUTION = 156543.03392804097
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe manifests and pseudo-CAD assets for large-scale U-Net training."
    )
    parser.add_argument(
        "--footprints-geojson",
        type=Path,
        default=Path(
            "outputs/phase2_real_building_events_2014_2026_expansion_v1/raw/"
            "development_activity_model_footprints.geojson"
        ),
    )
    parser.add_argument(
        "--monitor-json",
        type=Path,
        default=Path(
            "outputs/phase2_real_building_events_2014_2026_expansion_v1/raw/"
            "development_activity_monitor.json"
        ),
    )
    parser.add_argument(
        "--release-plan-csv",
        type=Path,
        default=Path("outputs/phase2_rgb_expanded_release_plan_2014_2026_v1.csv"),
    )
    parser.add_argument(
        "--external-test-csv",
        type=Path,
        default=Path(
            "outputs/phase2_melbourne_real_cad_pairs_v2_expanded/final_paired_dataset_v2/"
            "melbourne_cad_satellite_train_eval_manifest.csv"
        ),
    )
    parser.add_argument(
        "--real-cad-multiview-csv",
        type=Path,
        default=Path(
            "outputs/phase2_melbourne_cad_targetonly_multiview_v3/"
            "melbourne_cad_targetonly_multiview_manifest.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/phase2_large_unet_v1/data"),
    )
    parser.add_argument("--materialize-pretrain", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--minimum-patch-m", type=float, default=96.0)
    parser.add_argument("--maximum-patch-m", type=float, default=360.0)
    parser.add_argument("--footprint-margin-m", type=float, default=48.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-snapshots-per-footprint", type=int, default=4)
    return parser.parse_args()


def normalize_identifier(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def parse_year(value: object) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")


def stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_release_plan(path: Path) -> dict[int, dict[str, str]]:
    frame = pd.read_csv(path)
    if "target_year" not in frame or "official_template" not in frame:
        raise RuntimeError("Release plan must contain target_year and official_template")
    return {
        int(row.target_year): {
            "wayback_date": str(row.wayback_date),
            "wayback_identifier": str(row.wayback_identifier),
            "official_template": str(row.official_template),
        }
        for row in frame.itertuples(index=False)
    }


def choose_target_year(completion_year: int | None, available_years: list[int]) -> tuple[int | None, str]:
    if not available_years:
        return None, "no_release"
    minimum, maximum = min(available_years), max(available_years)
    if completion_year is None:
        return maximum, "latest_completed_snapshot_unknown_year"
    if completion_year > maximum:
        return None, "completion_after_last_release"
    if completion_year < minimum:
        return minimum, "first_available_post_completion_snapshot"
    if completion_year + 1 <= maximum:
        return completion_year + 1, "completion_plus_one"
    return completion_year, "completion_year_last_available"


def choose_target_years(
    completion_year: int | None,
    available_years: list[int],
    maximum_snapshots: int,
) -> list[tuple[int, str]]:
    first_year, first_rule = choose_target_year(completion_year, available_years)
    if first_year is None:
        return []
    eligible = [year for year in available_years if year >= first_year]
    if len(eligible) <= maximum_snapshots:
        selected = eligible
    else:
        positions = np.linspace(0, len(eligible) - 1, num=maximum_snapshots)
        selected = sorted({eligible[int(round(position))] for position in positions})
    return [
        (year, first_rule if year == first_year else "additional_post_completion_snapshot")
        for year in selected
    ]


def application_group(application: object, development_key: str) -> str:
    normalized = normalize_identifier(application)
    if not normalized or normalized in {"0", "NONE", "NAN"}:
        return f"DEV_{normalize_identifier(development_key)}"
    return f"APP_{normalized}"


def geometry_size_m(geometry: dict[str, object], transformer: Transformer) -> tuple[float, float, float]:
    geom = shape(geometry)
    projected = transform(transformer.transform, geom)
    if projected.is_empty:
        return 0.0, 0.0, 0.0
    min_x, min_y, max_x, max_y = projected.bounds
    return float(max_x - min_x), float(max_y - min_y), float(projected.area)


def build_pretrain_candidates(
    *,
    footprint_payload: dict[str, object],
    monitor_records: list[dict[str, object]],
    releases: dict[int, dict[str, str]],
    external_development_keys: set[str],
    external_application_keys: set[str],
    minimum_patch_m: float,
    maximum_patch_m: float,
    footprint_margin_m: float,
    max_snapshots_per_footprint: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    monitor = pd.DataFrame(monitor_records)
    monitor["development_key"] = monitor.get("development_key", "").astype(str).str.strip()
    monitor["completion_year"] = monitor.get("year_completed", "").map(parse_year)
    monitor["status_normalized"] = monitor.get("status", "").fillna("").astype(str).str.upper()
    monitor = monitor[monitor["status_normalized"].str.contains("COMPLET", na=False)].copy()
    monitor["metadata_nonzero"] = (
        monitor.reindex(
            columns=["floors_above", "resi_dwellings", "office_flr", "retail_flr", "industrial_flr"],
            fill_value=0,
        )
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .ne(0)
        .sum(axis=1)
    )
    monitor = monitor.sort_values(
        ["development_key", "metadata_nonzero", "completion_year"],
        ascending=[True, False, False],
    ).drop_duplicates("development_key")
    monitor_by_key = monitor.set_index("development_key").to_dict(orient="index")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:7855", always_xy=True)
    available_years = sorted(releases)
    records: list[dict[str, object]] = []
    counts = {
        "footprint_features": 0,
        "missing_monitor": 0,
        "not_completed": 0,
        "external_excluded": 0,
        "invalid_geometry": 0,
        "completion_after_last_release": 0,
    }
    for feature in footprint_payload.get("features", []):
        counts["footprint_features"] += 1
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry")
        development_key = str(properties.get("dev_key") or "").strip()
        monitor_row = monitor_by_key.get(development_key)
        if monitor_row is None:
            counts["missing_monitor"] += 1
            continue
        if not geometry:
            counts["invalid_geometry"] += 1
            continue
        application = monitor_row.get("town_planning_application", "")
        development_norm = normalize_identifier(development_key)
        application_norm = normalize_identifier(application)
        if development_norm in external_development_keys or (
            application_norm and application_norm in external_application_keys
        ):
            counts["external_excluded"] += 1
            continue
        completion_year = parse_year(monitor_row.get("year_completed"))
        target_years = choose_target_years(
            completion_year, available_years, max(max_snapshots_per_footprint, 1)
        )
        if not target_years:
            counts["completion_after_last_release"] += 1
            continue
        try:
            geom = shape(geometry)
            if geom.is_empty:
                raise ValueError("empty geometry")
            centroid = geom.centroid
            width_m, height_m, footprint_area_m2 = geometry_size_m(geometry, transformer)
        except Exception:  # noqa: BLE001
            counts["invalid_geometry"] += 1
            continue
        patch_size_m = float(
            np.clip(max(width_m, height_m) + 2.0 * footprint_margin_m, minimum_patch_m, maximum_patch_m)
        )
        geometry_token = hashlib.sha1(
            json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        height = max(0.0, finite(properties.get("bldhgt_ahd")) - finite(properties.get("base_ahd")))
        group_id = application_group(application, development_key)
        split = "val" if stable_fraction(group_id) < 0.10 else "train"
        for target_year, target_rule in target_years:
            release = releases[target_year]
            records.append(
                {
                    "sample_id": f"pseudo_{safe_name(development_key)}_{geometry_token}_{target_year}",
                "data_stage": "pseudo_cad_pretrain",
                "dataset_split": split,
                "group_id": group_id,
                "development_key": development_key,
                "footprint_instance_id": f"{development_key}__{geometry_token}",
                "town_planning_application": str(application or ""),
                "completion_year": completion_year,
                "target_year": target_year,
                "target_year_rule": target_rule,
                "wayback_date": release["wayback_date"],
                "wayback_identifier": release["wayback_identifier"],
                "imagery_url_template": release["official_template"],
                "centroid_longitude": float(centroid.x),
                "centroid_latitude": float(centroid.y),
                "patch_size_m": patch_size_m,
                "footprint_width_m": width_m,
                "footprint_length_m": height_m,
                "footprint_area_m2": footprint_area_m2,
                "geometry_geojson": json.dumps(geometry, separators=(",", ":")),
                "metadata_height_m": max(height, finite(monitor_row.get("floors_above")) * 3.2),
                "metadata_residential": finite(monitor_row.get("resi_dwellings")),
                "metadata_commercial_m2": finite(monitor_row.get("office_flr"))
                + finite(monitor_row.get("retail_flr")),
                "metadata_industrial_m2": finite(monitor_row.get("industrial_flr")),
                "target_rgb_png": "",
                "footprint_mask_png": "",
                "roi_mask_png": "",
                "cad_line_png": "",
                "cad_density_png": "",
                "cad_proximity_png": "",
                "cad_page_png": "",
                "cad_source_type": "pseudo_footprint_geometry",
                "asset_ready": False,
                    "materialization_status": "pending",
                }
            )
    return pd.DataFrame(records), counts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_bundle_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def add_application_balanced_weights(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    counts = output.groupby("group_id")["sample_id"].transform("count").astype(int)
    group_count = max(int(output["group_id"].nunique()), 1)
    output["application_sample_count"] = counts
    output["application_balanced_weight"] = len(output) / (group_count * counts)
    return output


def standardize_real_cad(
    frame: pd.DataFrame,
    *,
    external_development_keys: set[str],
    external_application_keys: set[str],
    project_root: Path,
    asset_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame.to_dict(orient="records")
    application_development_counts = (
        frame.groupby("town_planning_application")["development_key"].nunique().to_dict()
    )
    page_developments: dict[str, set[str]] = {}
    for row in rows:
        page_type, page_png, ink_png = choose_spatial_page(pd.Series(row))
        ink_path = resolve_path(Path(ink_png.replace("\\", "/")), project_root)
        if not ink_path.is_file():
            raise FileNotFoundError(ink_path)
        page_hash = file_sha256(ink_path)
        row["_large_unet_page_type"] = page_type
        row["_large_unet_page_png"] = page_png
        row["_large_unet_ink_png"] = ink_png
        row["_large_unet_page_hash"] = page_hash
        page_developments.setdefault(page_hash, set()).add(str(row.get("development_key", "")))

    records: list[dict[str, object]] = []
    for row in rows:
        development_key = str(row.get("development_key", ""))
        application = str(row.get("town_planning_application", ""))
        development_norm = normalize_identifier(development_key)
        application_norm = normalize_identifier(application)
        application_development_count = int(application_development_counts.get(application, 1))
        multi_development_application = application_development_count > 1
        if multi_development_application and development_key not in MANUAL_BUILDING_ROIS:
            raise RuntimeError(
                "A multi-development application lacks a building-specific ROI: "
                f"{application} / {development_key}"
            )
        is_external = development_norm in external_development_keys or (
            application_norm and application_norm in external_application_keys
        )
        group_id = application_group(application, development_key)
        if is_external:
            split = "external_test"
            stage = "external_test"
        else:
            split = "val" if stable_fraction(group_id) < 0.20 else "train"
            stage = "real_cad_finetune"

        page_type = str(row["_large_unet_page_type"])
        ink_png = str(row["_large_unet_ink_png"])
        page_hash = str(row["_large_unet_page_hash"])
        full_ink = np.asarray(
            Image.open(resolve_path(Path(ink_png.replace("\\", "/")), project_root)).convert("L"),
            dtype=np.uint8,
        )
        if development_key in MANUAL_BUILDING_ROIS:
            manual_roi = MANUAL_BUILDING_ROIS[development_key]
            x0, y0, x1, y1 = normalized_roi(full_ink.shape, manual_roi[:4])
            roi_strategy = f"manual_{manual_roi[4]}"
        else:
            x0, y0, x1, y1, _ = auto_plan_crop(full_ink)
            roi_strategy = "automatic_single_development_plan"
        crop_area = max((x1 - x0) * (y1 - y0), 1)
        page_area = max(full_ink.shape[0] * full_ink.shape[1], 1)
        full_ink_pixels = int((full_ink > 18).sum())
        crop_ink_pixels = int((full_ink[y0:y1, x0:x1] > 18).sum())
        panel = square_canvas(trim_crop_to_ink(full_ink[y0:y1, x0:x1]), 512)
        panel_line, panel_support = structural_line_map(panel)
        footprint = (
            np.asarray(
                Image.open(
                    resolve_path(
                        Path(str(row["footprint_mask_png"]).replace("\\", "/")), project_root
                    )
                ).convert("L")
            )
            > 0
        )
        aligned, density, proximity = align_to_footprint(panel_line, panel_support, footprint)
        spatial_hash = array_bundle_sha256(aligned, density, proximity)
        sample_asset_dir = asset_root / safe_name(row.get("sample_id", development_key))
        sample_asset_dir.mkdir(parents=True, exist_ok=True)
        panel_path = sample_asset_dir / "building_specific_cad_panel.png"
        line_path = sample_asset_dir / "cad_aligned_raster.png"
        density_path = sample_asset_dir / "cad_aligned_density.png"
        proximity_path = sample_asset_dir / "cad_aligned_proximity.png"
        Image.fromarray(panel).save(panel_path)
        Image.fromarray(aligned).save(line_path)
        Image.fromarray(density).save(density_path)
        Image.fromarray(proximity).save(proximity_path)
        relative = lambda path: str(path.resolve().relative_to(project_root.resolve()))

        record = {
            "sample_id": str(row.get("sample_id", "")),
            "data_stage": stage,
            "dataset_split": split,
            "group_id": group_id,
            "development_key": development_key,
            "town_planning_application": application,
            "application_development_count": application_development_count,
            "multi_development_application": multi_development_application,
            "completion_year": parse_year(row.get("completion_year")),
            "target_year": parse_year(row.get("target_year")),
            "target_year_rule": "existing_real_cad_target",
            "wayback_date": "",
            "wayback_identifier": "",
            "imagery_url_template": "",
            "centroid_longitude": finite(row.get("centroid_longitude")),
            "centroid_latitude": finite(row.get("centroid_latitude")),
            "patch_size_m": finite(row.get("crop_size_m")),
            "footprint_width_m": "",
            "footprint_length_m": "",
            "footprint_area_m2": finite(row.get("footprint_area_m2")),
            "geometry_geojson": str(row.get("geometry_geojson", "")),
            "metadata_height_m": max(
                finite(row.get("footprint_height_m")), finite(row.get("floors_above")) * 3.2
            ),
            "metadata_residential": finite(row.get("resi_dwellings")),
            "metadata_commercial_m2": finite(row.get("office_floor_m2"))
            + finite(row.get("retail_floor_m2")),
            "metadata_industrial_m2": finite(row.get("industrial_floor_m2")),
            "target_rgb_png": str(row.get("aligned_target_crop_png", "")),
            "footprint_mask_png": str(row.get("footprint_mask_png", "")),
            "roi_mask_png": str(row.get("roi_mask_png", "")),
            "cad_line_png": relative(line_path),
            "cad_density_png": relative(density_path),
            "cad_proximity_png": relative(proximity_path),
            "cad_page_png": ink_png,
            "building_specific_cad_panel_png": relative(panel_path),
            "cad_source_type": f"real_{row.get('cad_view_type', 'primary')}_plan",
            "cad_page_type": page_type,
            "full_page_cad_sha256": page_hash,
            "full_page_development_count": len(page_developments[page_hash]),
            "duplicate_full_page_across_developments": len(page_developments[page_hash]) > 1,
            "building_specific_spatial_sha256": spatial_hash,
            "building_specific_roi_strategy": roi_strategy,
            "building_specific_roi_applied": True,
            "building_specific_roi_x0": x0,
            "building_specific_roi_y0": y0,
            "building_specific_roi_x1": x1,
            "building_specific_roi_y1": y1,
            "building_specific_roi_page_fraction": crop_area / page_area,
            "building_specific_roi_ink_coverage": crop_ink_pixels / max(full_ink_pixels, 1),
            "asset_ready": True,
            "materialization_status": "building_specific_real_cad_ready",
            "is_primary_view": str(row.get("is_primary_view", "")).lower() in {"true", "1"},
            "cad_view_type": str(row.get("cad_view_type", "primary")),
        }
        records.append(record)
    standardized = add_application_balanced_weights(pd.DataFrame(records))
    return (
        standardized[standardized["data_stage"].eq("real_cad_finetune")].copy(),
        standardized[standardized["data_stage"].eq("external_test")].copy(),
    )


def lonlat_to_tile_fraction(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    latitude = np.clip(lat, -85.05112878, 85.05112878)
    lat_rad = math.radians(latitude)
    scale = 2.0**zoom
    x = (lon + 180.0) / 360.0 * scale
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return x, y


def ground_resolution_m_per_pixel(latitude: float, zoom: int) -> float:
    return WEB_MERCATOR_INITIAL_RESOLUTION * math.cos(math.radians(latitude)) / (2.0**zoom)


def session_for_thread() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": "melbourne-scats-phase2-large-unet/1.0"})
        THREAD_LOCAL.session = session
    return session


def fetch_tile(url: str, timeout: int) -> Image.Image:
    last_error: Exception | None = None
    for _ in range(4):
        try:
            response = session_for_thread().get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Unable to download tile: {last_error}")


def stitch_patch(
    *,
    longitude: float,
    latitude: float,
    patch_size_m: float,
    output_size_px: int,
    zoom: int,
    url_template: str,
    timeout: int,
) -> Image.Image:
    tile_x, tile_y = lonlat_to_tile_fraction(longitude, latitude, zoom)
    center_x, center_y = tile_x * TILE_SIZE, tile_y * TILE_SIZE
    native_span = patch_size_m / ground_resolution_m_per_pixel(latitude, zoom)
    half = native_span / 2.0
    xmin, ymin = center_x - half, center_y - half
    xmax, ymax = center_x + half, center_y + half
    tile_x_min = int(math.floor(xmin / TILE_SIZE))
    tile_x_max = int(math.floor((xmax - 1) / TILE_SIZE))
    tile_y_min = int(math.floor(ymin / TILE_SIZE))
    tile_y_max = int(math.floor((ymax - 1) / TILE_SIZE))
    canvas = Image.new(
        "RGB",
        ((tile_x_max - tile_x_min + 1) * TILE_SIZE, (tile_y_max - tile_y_min + 1) * TILE_SIZE),
    )
    for current_y in range(tile_y_min, tile_y_max + 1):
        for current_x in range(tile_x_min, tile_x_max + 1):
            url = url_template.format(
                z=zoom,
                zoom=zoom,
                x=current_x,
                y=current_y,
                TileMatrixSet="default028mm",
                TileMatrix=zoom,
                TileRow=current_y,
                TileCol=current_x,
            )
            canvas.paste(
                fetch_tile(url, timeout),
                ((current_x - tile_x_min) * TILE_SIZE, (current_y - tile_y_min) * TILE_SIZE),
            )
    left = int(round(xmin - tile_x_min * TILE_SIZE))
    top = int(round(ymin - tile_y_min * TILE_SIZE))
    native_size = max(int(round(native_span)), 1)
    patch = canvas.crop((left, top, left + native_size, top + native_size))
    return patch.resize((output_size_px, output_size_px), Image.Resampling.LANCZOS)


def pseudo_cad_maps(footprint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    footprint_u8 = footprint.astype(np.uint8)
    boundary = cv2.morphologyEx(footprint_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    skeleton = np.zeros_like(footprint_u8)
    working = footprint_u8.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while working.any():
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, kernel)
        skeleton |= working & (1 - opened)
        working = cv2.erode(working, kernel)
    line = np.clip(boundary + skeleton, 0, 1).astype(np.uint8)
    density = cv2.GaussianBlur(line.astype(np.float32), (0, 0), sigmaX=4.0)
    density /= max(float(density.max()), 1e-6)
    distance = cv2.distanceTransform((1 - line).astype(np.uint8), cv2.DIST_L2, 5)
    proximity = np.exp(-distance / 16.0).astype(np.float32)
    return line, density, proximity


def materialize_row(
    row: dict[str, object],
    *,
    project_root: Path,
    asset_root: Path,
    image_size: int,
    zoom: int,
    timeout: int,
    overwrite: bool,
) -> dict[str, object]:
    sample_dir = asset_root / str(row["dataset_split"]) / safe_name(row["sample_id"])
    target_path = sample_dir / "target_rgb.png"
    footprint_path = sample_dir / "footprint_mask.png"
    roi_path = sample_dir / "roi_mask.png"
    line_path = sample_dir / "pseudo_cad_line.png"
    density_path = sample_dir / "pseudo_cad_density.png"
    proximity_path = sample_dir / "pseudo_cad_proximity.png"
    required = [target_path, footprint_path, roi_path, line_path, density_path, proximity_path]
    if overwrite or not all(path.is_file() for path in required):
        sample_dir.mkdir(parents=True, exist_ok=True)
        target = stitch_patch(
            longitude=float(row["centroid_longitude"]),
            latitude=float(row["centroid_latitude"]),
            patch_size_m=float(row["patch_size_m"]),
            output_size_px=image_size,
            zoom=zoom,
            url_template=str(row["imagery_url_template"]),
            timeout=timeout,
        )
        target_array = np.asarray(target, dtype=np.uint8)
        if float(target_array.std()) < 4.0 or float(target_array.mean()) < 5.0:
            raise RuntimeError("Downloaded imagery is blank or near-constant")
        footprint = rasterize_footprint(
            json.loads(str(row["geometry_geojson"])),
            latitude=float(row["centroid_latitude"]),
            longitude=float(row["centroid_longitude"]),
            patch_size_m=float(row["patch_size_m"]),
            shape_hw=(image_size, image_size),
        )
        if int(footprint.sum()) < 32:
            raise RuntimeError("Rasterized footprint is too small")
        roi_iterations = max(int(round(image_size * 12.0 / float(row["patch_size_m"]))), 2)
        roi = ndimage.binary_dilation(footprint, iterations=roi_iterations)
        line, density, proximity = pseudo_cad_maps(footprint)
        target.save(target_path)
        Image.fromarray(footprint.astype(np.uint8) * 255).save(footprint_path)
        Image.fromarray(roi.astype(np.uint8) * 255).save(roi_path)
        Image.fromarray(line * 255).save(line_path)
        Image.fromarray(np.clip(density * 255.0, 0, 255).astype(np.uint8)).save(density_path)
        Image.fromarray(np.clip(proximity * 255.0, 0, 255).astype(np.uint8)).save(proximity_path)
    relative = lambda path: str(path.resolve().relative_to(project_root.resolve()))
    result = dict(row)
    result.update(
        {
            "target_rgb_png": relative(target_path),
            "footprint_mask_png": relative(footprint_path),
            "roi_mask_png": relative(roi_path),
            "cad_line_png": relative(line_path),
            "cad_density_png": relative(density_path),
            "cad_proximity_png": relative(proximity_path),
            "asset_ready": True,
            "materialization_status": "ready",
        }
    )
    return result


def asset_exists(value: object, project_root: Path) -> bool:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return False
    return resolve_path(Path(text.replace("\\", "/")), project_root).is_file()


def build_cad_duplicate_audit(
    finetune: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    combined = pd.concat([finetune, external], ignore_index=True)
    page_rows: list[dict[str, object]] = []
    for page_hash, group in combined.groupby("full_page_cad_sha256"):
        page_rows.append(
            {
                "full_page_cad_sha256": page_hash,
                "sample_count": int(len(group)),
                "development_count": int(group["development_key"].nunique()),
                "application_count": int(group["town_planning_application"].nunique()),
                "applications": ";".join(sorted(group["town_planning_application"].unique())),
                "developments": ";".join(sorted(group["development_key"].unique())),
                "page_types": ";".join(sorted(group["cad_page_type"].unique())),
                "building_specific_spatial_hash_count": int(
                    group["building_specific_spatial_sha256"].nunique()
                ),
                "roi_applied_fraction": float(group["building_specific_roi_applied"].mean()),
                "roi_page_fraction_mean": float(
                    group["building_specific_roi_page_fraction"].mean()
                ),
                "duplicate_across_developments": bool(group["development_key"].nunique() > 1),
            }
        )
    page_audit = pd.DataFrame(page_rows).sort_values(
        ["development_count", "sample_count"], ascending=False
    )
    spatial_duplicates: list[dict[str, object]] = []
    for spatial_hash, group in combined.groupby("building_specific_spatial_sha256"):
        if group["development_key"].nunique() > 1:
            spatial_duplicates.append(
                {
                    "building_specific_spatial_sha256": spatial_hash,
                    "developments": sorted(group["development_key"].unique()),
                    "applications": sorted(group["town_planning_application"].unique()),
                }
            )
    multi = combined[combined["multi_development_application"].astype(bool)].copy()
    summary = {
        "full_page_unique_hash_count": int(combined["full_page_cad_sha256"].nunique()),
        "full_page_duplicate_hash_group_count": int(
            page_audit["duplicate_across_developments"].sum()
        ),
        "multi_development_application_count": int(
            multi["town_planning_application"].nunique()
        ),
        "multi_development_count": int(multi["development_key"].nunique()),
        "multi_development_view_count": int(len(multi)),
        "building_specific_roi_coverage": float(
            multi["building_specific_roi_applied"].mean() if len(multi) else 1.0
        ),
        "multi_development_manual_roi_coverage": float(
            multi["building_specific_roi_strategy"].str.startswith("manual_").mean()
            if len(multi)
            else 1.0
        ),
        "cross_development_duplicate_spatial_hash_count": len(spatial_duplicates),
        "cross_development_duplicate_spatial_hashes": spatial_duplicates,
        "full_page_input_used_by_model": False,
        "model_uses_building_specific_spatial_raster": True,
        "application_balanced_sampling": True,
    }
    if spatial_duplicates:
        raise RuntimeError(
            "Building-specific CAD generation produced identical spatial rasters for different "
            f"developments: {spatial_duplicates}"
        )
    return page_audit, summary


def audit_manifests(
    pretrain: pd.DataFrame,
    finetune: pd.DataFrame,
    external: pd.DataFrame,
    project_root: Path,
) -> dict[str, object]:
    def values(frame: pd.DataFrame, column: str) -> set[str]:
        return {normalize_identifier(value) for value in frame.get(column, []) if normalize_identifier(value)}

    pretrain_dev = values(pretrain, "development_key")
    finetune_dev = values(finetune, "development_key")
    external_dev = values(external, "development_key")
    pretrain_app = values(pretrain, "town_planning_application")
    finetune_app = values(finetune, "town_planning_application")
    external_app = values(external, "town_planning_application")
    leakage = {
        "pretrain_external_development_overlap": sorted(pretrain_dev & external_dev),
        "finetune_external_development_overlap": sorted(finetune_dev & external_dev),
        "pretrain_external_application_overlap": sorted(pretrain_app & external_app),
        "finetune_external_application_overlap": sorted(finetune_app & external_app),
        "finetune_split_group_overlap": sorted(
            set(finetune.loc[finetune["dataset_split"].eq("train"), "group_id"])
            & set(finetune.loc[finetune["dataset_split"].eq("val"), "group_id"])
        ),
        "pretrain_split_group_overlap": sorted(
            set(pretrain.loc[pretrain["dataset_split"].eq("train"), "group_id"])
            & set(pretrain.loc[pretrain["dataset_split"].eq("val"), "group_id"])
        ),
    }
    leakage_free = not any(leakage.values())
    required_real_assets = [
        "target_rgb_png",
        "footprint_mask_png",
        "roi_mask_png",
        "cad_line_png",
        "cad_density_png",
        "cad_proximity_png",
    ]
    real_missing: list[dict[str, str]] = []
    for stage_name, frame in [("finetune", finetune), ("external_test", external)]:
        for row in frame.to_dict(orient="records"):
            for column in required_real_assets:
                if not asset_exists(row.get(column), project_root):
                    real_missing.append(
                        {"stage": stage_name, "sample_id": str(row.get("sample_id")), "asset": column}
                    )
    return {
        "leakage_free": leakage_free,
        "leakage": leakage,
        "real_asset_missing_count": len(real_missing),
        "real_asset_missing": real_missing,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)
    footprints_path = resolve_path(args.footprints_geojson, project_root)
    monitor_path = resolve_path(args.monitor_json, project_root)
    release_path = resolve_path(args.release_plan_csv, project_root)
    external_path = resolve_path(args.external_test_csv, project_root)
    real_cad_path = resolve_path(args.real_cad_multiview_csv, project_root)
    for path in [footprints_path, monitor_path, release_path, external_path, real_cad_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    footprint_payload = json.loads(footprints_path.read_text(encoding="utf-8"))
    monitor_records = json.loads(monitor_path.read_text(encoding="utf-8"))
    releases = load_release_plan(release_path)
    external_raw = pd.read_csv(external_path, dtype={"development_key": str})
    real_cad_raw = pd.read_csv(real_cad_path, dtype={"development_key": str})
    external_development_keys = {
        normalize_identifier(value) for value in external_raw["development_key"].dropna().unique()
    }
    external_application_keys = {
        normalize_identifier(value)
        for value in external_raw["town_planning_application"].dropna().unique()
    }
    pretrain, source_counts = build_pretrain_candidates(
        footprint_payload=footprint_payload,
        monitor_records=monitor_records,
        releases=releases,
        external_development_keys=external_development_keys,
        external_application_keys=external_application_keys,
        minimum_patch_m=args.minimum_patch_m,
        maximum_patch_m=args.maximum_patch_m,
        footprint_margin_m=args.footprint_margin_m,
        max_snapshots_per_footprint=args.max_snapshots_per_footprint,
    )
    pretrain = add_application_balanced_weights(pretrain)
    finetune, external = standardize_real_cad(
        real_cad_raw,
        external_development_keys=external_development_keys,
        external_application_keys=external_application_keys,
        project_root=project_root,
        asset_root=args.out_dir / "real_cad_spatial_assets",
    )

    if args.materialize_pretrain:
        working = pretrain if args.limit is None else pretrain.iloc[: args.limit].copy()
        asset_root = args.out_dir / "pretrain_assets"
        results: dict[str, dict[str, object]] = {}
        failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            future_to_id = {
                pool.submit(
                    materialize_row,
                    row,
                    project_root=project_root,
                    asset_root=asset_root,
                    image_size=args.image_size,
                    zoom=args.zoom,
                    timeout=args.timeout,
                    overwrite=args.overwrite,
                ): str(row["sample_id"])
                for row in working.to_dict(orient="records")
            }
            for index, future in enumerate(as_completed(future_to_id), start=1):
                sample_id = future_to_id[future]
                try:
                    results[sample_id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append({"sample_id": sample_id, "error": str(exc)})
                if index % 25 == 0 or index == len(future_to_id):
                    print(f"materialized {index}/{len(future_to_id)}; failures={len(failures)}", flush=True)
        if results:
            indexed = pretrain.set_index("sample_id")
            for sample_id, record in results.items():
                for key, value in record.items():
                    if key in indexed.columns:
                        indexed.at[sample_id, key] = value
            pretrain = indexed.reset_index()
        (args.out_dir / "materialization_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )

    pretrain_path = args.out_dir / "phase2_large_unet_pretrain_manifest.csv"
    finetune_path = args.out_dir / "phase2_large_unet_finetune_manifest.csv"
    external_manifest_path = args.out_dir / "phase2_large_unet_external_test_manifest.csv"
    pretrain.to_csv(pretrain_path, index=False)
    finetune.to_csv(finetune_path, index=False)
    external.to_csv(external_manifest_path, index=False)

    cad_page_audit, cad_audit_summary = build_cad_duplicate_audit(finetune, external)
    cad_page_audit_path = args.out_dir / "phase2_large_unet_cad_duplicate_audit.csv"
    cad_page_audit.to_csv(cad_page_audit_path, index=False)
    (args.out_dir / "phase2_large_unet_cad_duplicate_audit.json").write_text(
        json.dumps(cad_audit_summary, indent=2), encoding="utf-8"
    )

    audit = audit_manifests(pretrain, finetune, external, project_root)
    if not audit["leakage_free"]:
        raise RuntimeError(f"Data leakage detected: {audit['leakage']}")
    summary = {
        "pretrain_candidate_count": int(len(pretrain)),
        "pretrain_unique_sample_count": int(pretrain["sample_id"].nunique()),
        "pretrain_development_count": int(pretrain["development_key"].nunique()),
        "pretrain_ready_count": int(pretrain["asset_ready"].astype(str).str.lower().isin(["true", "1"]).sum()),
        "pretrain_split_counts": pretrain["dataset_split"].value_counts().to_dict(),
        "pretrain_application_groups": int(pretrain["group_id"].nunique()),
        "pretrain_application_balanced_weight_range": [
            float(pretrain["application_balanced_weight"].min()),
            float(pretrain["application_balanced_weight"].max()),
        ],
        "finetune_view_count": int(len(finetune)),
        "finetune_development_count": int(finetune["development_key"].nunique()),
        "finetune_application_count": int(finetune["town_planning_application"].nunique()),
        "finetune_split_counts": finetune["dataset_split"].value_counts().to_dict(),
        "finetune_application_balanced_weight_range": [
            float(finetune["application_balanced_weight"].min()),
            float(finetune["application_balanced_weight"].max()),
        ],
        "external_test_view_count": int(len(external)),
        "external_test_development_count": int(external["development_key"].nunique()),
        "external_test_application_count": int(external["town_planning_application"].nunique()),
        "external_test_primary_view_count": int(external.get("is_primary_view", False).astype(bool).sum()),
        "external_test_locked": True,
        "external_test_training_use": False,
        "source_audit_counts": source_counts,
        "release_years": sorted(releases),
        "spatial_condition_channels": ["footprint", "cad_line", "cad_density", "cad_proximity"],
        "metadata_vector": ["height", "residential", "commercial", "industrial"],
        "leakage_free": audit["leakage_free"],
        "real_asset_missing_count": audit["real_asset_missing_count"],
        "cad_duplicate_audit": cad_audit_summary,
        "manifests": {
            "pretrain": str(pretrain_path),
            "finetune": str(finetune_path),
            "external_test": str(external_manifest_path),
            "cad_duplicate_audit": str(cad_page_audit_path),
        },
    }
    (args.out_dir / "phase2_large_unet_leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (args.out_dir / "phase2_large_unet_data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
