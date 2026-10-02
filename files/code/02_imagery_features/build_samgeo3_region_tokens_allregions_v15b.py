from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, closing, disk, remove_small_objects

from build_samgeo3_region_tokens import (
    CLASS_PRIORITY,
    CLASS_PROMPTS,
    add_local_sam3_repo,
    build_year_root_map,
    get_class_min_merge_score,
    get_class_score_bias,
    render_component_overlay,
    render_flat_mask,
    render_overlay,
    render_topk_montage,
    resolve_image_path,
    save_component_mask,
    selection_score,
)
from build_samgeo3_region_tokens_allregions_v2 import run_semantic_pass, sum_kept_masks
from run_samgeo3_semantic_map import maybe_preprocess_image_for_sam


TILE_SIZE = 256

WIDTHS = {
    "conservative": {1: 24, 2: 24, 3: 20, 4: 16, 5: 12, 6: 8},
    "balanced": {1: 30, 2: 30, 3: 24, 4: 19, 5: 15, 6: 10},
    "loose": {1: 36, 2: 36, 3: 28, 4: 22, 5: 18, 6: 13},
}


@dataclass(frozen=True)
class RoadLine:
    points: list[tuple[float, float]]
    class_code: int
    bbox: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SamGeo3 all-region tokens with v15b road-recall semantic cleaning."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--year-root-catalog-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--bpe-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--min-mask-size", type=int, default=48)
    parser.add_argument("--road-min-component", type=int, default=160)
    parser.add_argument("--parking-min-component", type=int, default=24)
    parser.add_argument("--building-min-component", type=int, default=64)
    parser.add_argument("--vegetation-min-component", type=int, default=96)
    parser.add_argument("--parking-road-margin", type=float, default=0.04)
    parser.add_argument("--parking-max-mask-area-ratio", type=float, default=0.10)
    parser.add_argument("--parking-min-support-prompts", type=int, default=2)
    parser.add_argument("--parking-lot-min-score", type=float, default=0.44)
    parser.add_argument("--parking-lot-strong-score", type=float, default=0.56)
    parser.add_argument("--parking-aux-min-score", type=float, default=0.40)
    parser.add_argument("--parking-road-suppress-margin", type=float, default=0.14)
    parser.add_argument("--parking-thin-max-width", type=int, default=5)
    parser.add_argument("--parking-thin-min-aspect", type=float, default=5.0)
    parser.add_argument("--parking-thin-max-area", type=int, default=180)
    parser.add_argument("--road-score-bias", type=float, default=0.0)
    parser.add_argument("--parking-score-bias", type=float, default=0.0)
    parser.add_argument("--building-score-bias", type=float, default=0.05)
    parser.add_argument("--vegetation-score-bias", type=float, default=0.0)
    parser.add_argument("--road-min-merge-score", type=float, default=0.12)
    parser.add_argument("--parking-min-merge-score", type=float, default=0.40)
    parser.add_argument("--building-min-merge-score", type=float, default=0.29)
    parser.add_argument("--vegetation-min-merge-score", type=float, default=0.42)
    parser.add_argument("--building-road-margin", type=float, default=0.12)
    parser.add_argument("--parking-context-radius", type=int, default=5)
    parser.add_argument("--parking-context-max-road-ratio", type=float, default=0.55)
    parser.add_argument("--parking-context-min-building-ratio", type=float, default=0.02)
    parser.add_argument("--road-hole-area", type=int, default=96)
    parser.add_argument("--parking-hole-area", type=int, default=48)
    parser.add_argument("--building-hole-area", type=int, default=48)
    parser.add_argument("--vegetation-hole-area", type=int, default=64)
    parser.add_argument("--road-closing-radius", type=int, default=1)
    parser.add_argument("--parking-closing-radius", type=int, default=1)
    parser.add_argument("--building-closing-radius", type=int, default=1)
    parser.add_argument("--preprocess-autocontrast-cutoff", type=float, default=0.0)
    parser.add_argument("--preprocess-contrast", type=float, default=1.0)
    parser.add_argument("--preprocess-sharpness", type=float, default=1.0)
    parser.add_argument("--adaptive-preprocess", action="store_true")
    parser.add_argument("--adaptive-mean-min", type=float, default=90.0)
    parser.add_argument("--adaptive-mean-max", type=float, default=145.0)
    parser.add_argument("--adaptive-std-min", type=float, default=40.0)
    parser.add_argument("--adaptive-std-max", type=float, default=62.0)
    parser.add_argument("--adaptive-p10-min", type=float, default=35.0)
    parser.add_argument("--adaptive-autocontrast-cutoff", type=float, default=0.8)
    parser.add_argument("--adaptive-contrast", type=float, default=1.18)
    parser.add_argument("--adaptive-sharpness", type=float, default=1.08)
    parser.add_argument("--retry-strong-preprocess-when-no-road", action="store_true")
    parser.add_argument("--retry-preprocess-autocontrast-cutoff", type=float, default=1.0)
    parser.add_argument("--retry-preprocess-contrast", type=float, default=1.35)
    parser.add_argument("--retry-preprocess-sharpness", type=float, default=1.15)
    parser.add_argument(
        "--road-geojson",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "phase2_realplanning_input_v1" / "cache" / "tr_road.geojson",
    )
    parser.add_argument("--disable-road-prior-rescue", action="store_true")
    parser.add_argument(
        "--road-prior-coordinate-mode",
        choices=["wmts", "webmercator_meters", "true_ground_meters"],
        default="wmts",
    )
    parser.add_argument("--road-rescue-green-guard-radius", type=int, default=2)
    parser.add_argument("--road-rescue-safe-min-component", type=int, default=72)
    parser.add_argument("--road-rescue-accepted-min-component", type=int, default=36)
    parser.add_argument("--road-rescue-score", type=float, default=0.34)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--render-qa-count", type=int, default=8)
    parser.add_argument("--qa-top-k", type=int, default=12)
    return parser.parse_args()


def lonlat_to_world_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


def lonlat_to_webmercator(lon: float, lat: float) -> tuple[float, float]:
    origin_shift = 20037508.342789244
    x = lon * origin_shift / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * origin_shift / 180.0
    return x, y


def iter_line_parts(geometry: dict[str, object]) -> Iterable[list[list[float]]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "LineString" and isinstance(coords, list):
        yield coords
    elif geom_type == "MultiLineString" and isinstance(coords, list):
        for part in coords:
            if isinstance(part, list):
                yield part


def feature_class_code(feature: dict[str, object]) -> int:
    props = feature.get("properties", {})
    if not isinstance(props, dict):
        return 5
    try:
        return int(props.get("class_code", 5))
    except Exception:
        return 5


def load_road_lines(road_geojson: Path, *, zoom: int, coordinate_mode: str) -> list[RoadLine]:
    if not road_geojson.exists():
        return []
    data = json.loads(road_geojson.read_text(encoding="utf-8"))
    lines: list[RoadLine] = []
    for feature in data.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        class_code = feature_class_code(feature)
        for part in iter_line_parts(geometry):
            points: list[tuple[float, float]] = []
            for item in part:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                lon = float(item[0])
                lat = float(item[1])
                if coordinate_mode in {"webmercator_meters", "true_ground_meters"}:
                    points.append(lonlat_to_webmercator(lon, lat))
                else:
                    points.append(lonlat_to_world_px(lon, lat, zoom))
            if len(points) < 2:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            lines.append(RoadLine(points=points, class_code=class_code, bbox=(min(xs), min(ys), max(xs), max(ys))))
    return lines


def line_width(class_code: int, variant: str) -> int:
    widths = WIDTHS[variant]
    if class_code <= 2:
        key = 2
    elif class_code >= 6:
        key = 6
    else:
        key = class_code
    return widths.get(key, widths[5])


def rasterize_patch_prior(row: pd.Series, lines: list[RoadLine], variant: str, mode: str) -> np.ndarray:
    image_size = int(row.get("image_size_px", TILE_SIZE))
    canvas = Image.new("L", (image_size, image_size), 0)
    painter = ImageDraw.Draw(canvas)
    max_width = max(WIDTHS[variant].values())
    if mode in {"webmercator_meters", "true_ground_meters"}:
        center_x, center_y = lonlat_to_webmercator(float(row["LONGITUDE"]), float(row["LATITUDE"]))
        patch_size_m = float(row.get("patch_size_m", 500.0))
        projected_patch_size_m = patch_size_m
        if mode == "true_ground_meters":
            projected_patch_size_m = patch_size_m / math.cos(math.radians(float(row["LATITUDE"])))
        xmin = center_x - projected_patch_size_m / 2.0
        xmax = center_x + projected_patch_size_m / 2.0
        ymin = center_y - projected_patch_size_m / 2.0
        ymax = center_y + projected_patch_size_m / 2.0
        scale = image_size / projected_patch_size_m
        width_scale = 1.0 if mode == "true_ground_meters" else max(1.0, image_size / TILE_SIZE)
        for line in lines:
            lx0, ly0, lx1, ly1 = line.bbox
            margin_m = max_width / max(scale, 1e-6)
            if lx1 < xmin - margin_m or lx0 > xmax + margin_m or ly1 < ymin - margin_m or ly0 > ymax + margin_m:
                continue
            pts = [((px - xmin) * scale, (ymax - py) * scale) for px, py in line.points]
            painter.line(pts, fill=255, width=max(1, int(round(line_width(line.class_code, variant) * width_scale))), joint="curve")
        return np.array(canvas) > 0

    center_x, center_y = lonlat_to_world_px(float(row["LONGITUDE"]), float(row["LATITUDE"]), int(row.get("zoom", 18)))
    x0 = center_x - image_size / 2.0
    y0 = center_y - image_size / 2.0
    x1 = center_x + image_size / 2.0
    y1 = center_y + image_size / 2.0
    for line in lines:
        lx0, ly0, lx1, ly1 = line.bbox
        if lx1 < x0 - max_width or lx0 > x1 + max_width or ly1 < y0 - max_width or ly0 > y1 + max_width:
            continue
        pts = [(float(px - x0), float(py - y0)) for px, py in line.points]
        painter.line(pts, fill=255, width=line_width(line.class_code, variant), joint="curve")
    return np.array(canvas) > 0


def color_features(original: np.ndarray) -> dict[str, np.ndarray]:
    arr = original.astype(np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    mean = (r + g + b) / 3.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    span = maxc - minc
    sat = span / np.maximum(mean, 1.0)
    return {"r": r, "g": g, "b": b, "mean": mean, "span": span, "sat": sat}


def road_like_color(original: np.ndarray, mode: str) -> np.ndarray:
    f = color_features(original)
    r, g, b, mean, span, sat = f["r"], f["g"], f["b"], f["mean"], f["span"], f["sat"]
    not_too_dark = mean > 30
    not_too_bright = mean < 235
    grayish = (sat < 0.48) | (span < 58)
    asphaltish = (mean > 45) & (mean < 190) & ((sat < 0.62) | (span < 78))
    strong_green = (g > r + 18) & (g > b + 14) & (mean > 45)
    strong_blue = b > r + 22
    strong_blue &= b > g + 15
    red_roof = (r > g + 34) & (r > b + 24) & (mean > 55)
    yellow_roof = (r > b + 35) & (g > b + 25) & (mean > 95)
    if mode == "balanced":
        return not_too_dark & not_too_bright & (grayish | asphaltish) & ~strong_green & ~strong_blue & ~red_roof
    return not_too_dark & not_too_bright & ~strong_green & ~strong_blue & ~red_roof & ~yellow_roof


def safe_candidate(
    original: np.ndarray,
    semantic_map: np.ndarray,
    prior: np.ndarray,
    *,
    green_guard_radius: int,
    min_component: int,
) -> np.ndarray:
    f = color_features(original)
    r, g, b, mean, span, sat = f["r"], f["g"], f["b"], f["mean"], f["span"], f["sat"]
    other = semantic_map == "other"
    roadish_gray = ((sat < 0.40) | (span < 48)) & (mean > 38) & (mean < 205)
    asphalt_shadow = (mean > 35) & (mean < 130) & (sat < 0.58) & (span < 72)
    strong_green = (g > r + 10) & (g > b + 8) & (mean > 38)
    tree_dark_green = (g >= r + 4) & (g >= b + 4) & (mean < 105)
    strong_blue = (b > r + 20) & (b > g + 14)
    red_or_yellow_roof = ((r > g + 30) & (r > b + 22) & (mean > 55)) | (
        (r > b + 36) & (g > b + 24) & (mean > 95)
    )
    very_bright_roof = (mean > 188) & (sat < 0.24)
    green_guard = strong_green | tree_dark_green
    if green_guard_radius > 0:
        green_guard = binary_dilation(green_guard, footprint=disk(int(green_guard_radius)))
    candidate = other & prior & (roadish_gray | asphalt_shadow) & ~green_guard & ~strong_blue & ~red_or_yellow_roof & ~very_bright_roof
    candidate = remove_small_objects(candidate.astype(bool), min_size=int(min_component))
    return candidate.astype(bool)


def strict_accepted_color(original: np.ndarray) -> np.ndarray:
    f = color_features(original)
    r, g, b, mean, span, sat = f["r"], f["g"], f["b"], f["mean"], f["span"], f["sat"]
    grayish = ((sat < 0.44) | (span < 56)) & (mean > 38) & (mean < 190)
    asphaltish = (mean > 42) & (mean < 170) & ((sat < 0.50) | (span < 66))
    very_green = ((g > r + 18) & (g > b + 14) & (mean > 42)) | ((g >= r + 8) & (g >= b + 7) & (mean < 112))
    strong_blue = (b > r + 24) & (b > g + 16)
    red_or_yellow_roof = ((r > g + 36) & (r > b + 24) & (mean > 52)) | (
        (r > b + 40) & (g > b + 26) & (mean > 96)
    )
    very_bright_roof = (mean > 192) & (sat < 0.25)
    return (grayish | asphaltish) & ~very_green & ~strong_blue & ~red_or_yellow_roof & ~very_bright_roof


def apply_road_prior_rescue(
    *,
    image_arr: np.ndarray,
    semantic_map: np.ndarray,
    class_score_maps: dict[str, np.ndarray],
    manifest_row: pd.Series,
    road_lines: list[RoadLine],
    args: argparse.Namespace,
) -> dict[str, int | float | bool]:
    if bool(args.disable_road_prior_rescue) or not road_lines:
        return {
            "enabled": False,
            "road_before": int(np.sum(semantic_map == "road")),
            "safe_add_pixels": 0,
            "accepted_extra_pixels": 0,
            "road_after": int(np.sum(semantic_map == "road")),
        }
    conservative_prior = rasterize_patch_prior(manifest_row, road_lines, "conservative", args.road_prior_coordinate_mode)
    balanced_prior = rasterize_patch_prior(manifest_row, road_lines, "balanced", args.road_prior_coordinate_mode)
    loose_prior = rasterize_patch_prior(manifest_row, road_lines, "loose", args.road_prior_coordinate_mode)
    other = semantic_map == "other"
    road_before = int(np.sum(semantic_map == "road"))

    safe_add = safe_candidate(
        image_arr,
        semantic_map,
        conservative_prior,
        green_guard_radius=int(args.road_rescue_green_guard_radius),
        min_component=int(args.road_rescue_safe_min_component),
    )
    balanced_add = other & balanced_prior & road_like_color(image_arr, "balanced")
    loose_add = other & loose_prior & road_like_color(image_arr, "loose")
    accepted_seed = balanced_add & ~safe_add & other
    accepted_extra = accepted_seed & strict_accepted_color(image_arr)
    accepted_extra = closing(accepted_extra, footprint=disk(1))
    accepted_extra = remove_small_objects(accepted_extra.astype(bool), min_size=int(args.road_rescue_accepted_min_component))
    rescue = (safe_add | accepted_extra) & (semantic_map == "other")

    if np.any(rescue):
        semantic_map[rescue] = "road"
        road_score = max(float(args.road_rescue_score), float(args.road_min_merge_score) + 0.02)
        class_score_maps["road"][rescue] = np.maximum(class_score_maps["road"][rescue], road_score)

    return {
        "enabled": True,
        "road_before": road_before,
        "safe_add_pixels": int(np.sum(safe_add)),
        "accepted_extra_pixels": int(np.sum(accepted_extra)),
        "rescued_pixels": int(np.sum(rescue)),
        "road_after": int(np.sum(semantic_map == "road")),
        "conservative_prior_pixels": int(np.sum(conservative_prior)),
        "balanced_prior_pixels": int(np.sum(balanced_prior)),
        "loose_prior_pixels": int(np.sum(loose_prior)),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    add_local_sam3_repo(repo_root)
    from samgeo import SamGeo3

    manifest = pd.read_csv(args.manifest_csv)
    if args.limit_rows is not None:
        manifest = manifest.head(args.limit_rows).copy()
    manifest["SITE_NO"] = pd.to_numeric(manifest["SITE_NO"], errors="coerce").astype(int)
    manifest["year"] = pd.to_numeric(manifest["year"], errors="coerce").astype(int)
    year_root_map = build_year_root_map(args.year_root_catalog_csv)
    max_zoom = int(manifest["zoom"].max()) if "zoom" in manifest.columns and len(manifest) else 18
    road_lines = load_road_lines(args.road_geojson, zoom=max_zoom, coordinate_mode=args.road_prior_coordinate_mode)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_root = out_dir / "qa"
    mask_root = out_dir / "masks"

    checkpoint_path = args.checkpoint_path or (repo_root / "external" / "hf_models" / "facebook--sam3.1" / "sam3.1_multiplex.pt")
    bpe_path = args.bpe_path or (repo_root / "external" / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    sam = SamGeo3(
        backend="meta",
        checkpoint_path=str(checkpoint_path),
        bpe_path=str(bpe_path),
        load_from_HF=False,
        device=args.device,
        enable_inst_interactivity=True,
        resolution=args.resolution,
        confidence_threshold=args.confidence_threshold,
    )

    rows: list[dict[str, object]] = []
    qa_records: list[dict[str, object]] = []
    summary_samples: list[dict[str, object]] = []
    region_counts: list[int] = []
    rescue_records: list[dict[str, object]] = []

    for row_idx, manifest_row in enumerate(manifest.itertuples(index=False), start=1):
        manifest_series = pd.Series(manifest_row._asdict())
        site_no = int(getattr(manifest_row, "SITE_NO"))
        year = int(getattr(manifest_row, "year"))
        patch_id = str(getattr(manifest_row, "satellite_patch_id"))
        image_rel = str(getattr(manifest_row, "satellite_relative_image_path"))
        image_path = resolve_image_path(manifest_series, year_root_map)
        raw_image_arr = np.asarray(Image.open(image_path).convert("RGB"))
        sam_input_path, image_arr, preprocess_info = maybe_preprocess_image_for_sam(
            image_path,
            out_dir / "sam_inputs" / f"year_{year}" / f"{patch_id}.png",
            args,
        )
        sam.set_image(str(sam_input_path))
        pass_result = run_semantic_pass(sam=sam, image_arr=image_arr, args=args)
        chosen_sam_input_path = sam_input_path
        chosen_image_arr = image_arr
        chosen_preprocess_info = preprocess_info
        retry_record: dict[str, object] | None = None

        initial_road_kept = sum_kept_masks(pass_result["prompt_records"], "road")
        if bool(args.retry_strong_preprocess_when_no_road) and initial_road_kept == 0 and str(preprocess_info.get("mode", "none")) == "adaptive":
            retry_args = argparse.Namespace(**vars(args))
            retry_args.adaptive_preprocess = False
            retry_args.preprocess_autocontrast_cutoff = float(args.retry_preprocess_autocontrast_cutoff)
            retry_args.preprocess_contrast = float(args.retry_preprocess_contrast)
            retry_args.preprocess_sharpness = float(args.retry_preprocess_sharpness)
            retry_input_path, retry_image_arr, retry_preprocess_info = maybe_preprocess_image_for_sam(
                image_path,
                out_dir / "sam_inputs_retry" / f"year_{year}" / f"{patch_id}.png",
                retry_args,
            )
            sam.set_image(str(retry_input_path))
            retry_pass_result = run_semantic_pass(sam=sam, image_arr=retry_image_arr, args=retry_args)
            retry_road_kept = sum_kept_masks(retry_pass_result["prompt_records"], "road")
            accepted = retry_road_kept > initial_road_kept
            retry_record = {
                "triggered": True,
                "accepted": bool(accepted),
                "initial_road_kept": int(initial_road_kept),
                "retry_road_kept": int(retry_road_kept),
                "retry_preprocess": retry_preprocess_info,
            }
            if accepted:
                pass_result = retry_pass_result
                chosen_sam_input_path = retry_input_path
                chosen_image_arr = retry_image_arr
                chosen_preprocess_info = retry_preprocess_info

        class_score_maps = pass_result["class_score_maps"]
        prompt_records = pass_result["prompt_records"]
        extra_maps_by_class = pass_result["extra_maps_by_class"]
        parking_refine_stats = pass_result["parking_refine_stats"]
        semantic_map = pass_result["semantic_map"]

        rescue_stats = apply_road_prior_rescue(
            image_arr=chosen_image_arr,
            semantic_map=semantic_map,
            class_score_maps=class_score_maps,
            manifest_row=manifest_series,
            road_lines=road_lines,
            args=args,
        )
        rescue_stats = {
            "SITE_NO": site_no,
            "year": year,
            "patch_id": patch_id,
            **rescue_stats,
        }
        rescue_records.append(rescue_stats)

        height, width = chosen_image_arr.shape[:2]
        components: list[dict[str, object]] = []
        image_area = float(height * width)
        for class_name in CLASS_PRIORITY:
            class_mask = semantic_map == class_name
            if not np.any(class_mask):
                continue
            labeled = label(class_mask.astype(np.uint8), connectivity=1)
            score_map = class_score_maps[class_name]
            support_map = extra_maps_by_class.get(class_name, {}).get("support_count_map", np.zeros((height, width), dtype=np.uint8))
            prompt_count = max(len(CLASS_PROMPTS.get(class_name, [])), 1)
            for prop in regionprops(labeled):
                component_mask = labeled == prop.label
                min_row, min_col, max_row, max_col = prop.bbox
                x0 = int(min_col)
                y0 = int(min_row)
                x1 = int(max_col - 1)
                y1 = int(max_row - 1)
                bbox = (x0, y0, x1, y1)
                area_pixels = int(prop.area)
                area_ratio = float(area_pixels / image_area)
                bbox_w = int(x1 - x0 + 1)
                bbox_h = int(y1 - y0 + 1)
                fill_ratio = float(area_pixels / max(bbox_w * bbox_h, 1))
                semantic_confidence = float(score_map[component_mask].mean()) if np.any(component_mask) else 0.0
                sam_score = float(score_map[component_mask].max()) if np.any(component_mask) else 0.0
                support_strength = float(support_map[component_mask].mean() / prompt_count) if np.any(component_mask) else 0.0
                components.append(
                    {
                        "SITE_NO": site_no,
                        "year": year,
                        "dataset_split": str(getattr(manifest_row, "dataset_split")),
                        "patch_id": patch_id,
                        "site_name": str(getattr(manifest_row, "SITE_NAME")),
                        "relative_image_path": image_rel,
                        "resolved_image_path": str(image_path),
                        "sam_input_image_path": str(chosen_sam_input_path),
                        "sam_preprocess_mode": str(chosen_preprocess_info["mode"]),
                        "semantic_label": class_name,
                        "prompt_semantic_class": class_name,
                        "proposal_source": "samgeo3_v5_component_allregions_v15b",
                        "semantic_source": "samgeo3_v5_allregions_v15b_road_rescue",
                        "prompt_confidence": semantic_confidence,
                        "bbox": bbox,
                        "bbox_x0": x0,
                        "bbox_y0": y0,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "centroid_x": float(prop.centroid[1]),
                        "centroid_y": float(prop.centroid[0]),
                        "area_pixels": area_pixels,
                        "area_ratio": area_ratio,
                        "sam_score": sam_score,
                        "semantic_confidence": semantic_confidence,
                        "prior_support": support_strength,
                        "consistency_support": fill_ratio,
                        "selection_score": selection_score(area_ratio, semantic_confidence, support_strength, fill_ratio),
                        "component_mask": component_mask,
                    }
                )

        components = sorted(components, key=lambda item: (float(item["selection_score"]), float(item["area_ratio"])), reverse=True)
        for region_rank, comp in enumerate(components, start=1):
            comp["region_rank"] = region_rank

        region_counts.append(len(components))
        for comp in components:
            region_rank = int(comp["region_rank"])
            mask_rel = Path("masks") / f"year_{year}" / f"{patch_id}__region_{region_rank:03d}.png"
            mask_abs = mask_root / f"year_{year}" / f"{patch_id}__region_{region_rank:03d}.png"
            save_component_mask(comp["component_mask"], mask_abs)
            rows.append(
                {
                    "SITE_NO": comp["SITE_NO"],
                    "year": comp["year"],
                    "dataset_split": comp["dataset_split"],
                    "patch_id": comp["patch_id"],
                    "site_name": comp["site_name"],
                    "relative_image_path": comp["relative_image_path"],
                    "resolved_image_path": comp["resolved_image_path"],
                    "sam_input_image_path": str(chosen_sam_input_path),
                    "sam_preprocess_mode": str(chosen_preprocess_info["mode"]),
                    "region_rank": region_rank,
                    "mask_relative_path": str(mask_rel),
                    "mask_path": str(mask_abs),
                    "bbox_x0": comp["bbox_x0"],
                    "bbox_y0": comp["bbox_y0"],
                    "bbox_x1": comp["bbox_x1"],
                    "bbox_y1": comp["bbox_y1"],
                    "bbox_w": comp["bbox_w"],
                    "bbox_h": comp["bbox_h"],
                    "centroid_x": comp["centroid_x"],
                    "centroid_y": comp["centroid_y"],
                    "area_pixels": comp["area_pixels"],
                    "area_ratio": comp["area_ratio"],
                    "sam_score": comp["sam_score"],
                    "prompt_semantic_class": comp["prompt_semantic_class"],
                    "proposal_source": comp["proposal_source"],
                    "semantic_source": comp["semantic_source"],
                    "prompt_confidence": comp["prompt_confidence"],
                    "semantic_label": comp["semantic_label"],
                    "semantic_confidence": comp["semantic_confidence"],
                    "prior_support": comp["prior_support"],
                    "consistency_support": comp["consistency_support"],
                    "selection_score": comp["selection_score"],
                }
            )

        if row_idx <= args.render_qa_count:
            qa_dir = qa_root / f"year_{year}" / patch_id
            qa_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(raw_image_arr, mode="RGB").save(qa_dir / "original_raw.png")
            Image.fromarray(chosen_image_arr, mode="RGB").save(qa_dir / "sam_input.png")
            render_overlay(raw_image_arr, semantic_map, qa_dir / "semantic_overlay.png")
            render_flat_mask(semantic_map, qa_dir / "semantic_mask.png")
            render_component_overlay(raw_image_arr, components, qa_dir / "sam_overlay_allregions.png")
            render_topk_montage(raw_image_arr, components[: args.qa_top_k], qa_dir / "topk_montage.png")
            qa_records.append({"SITE_NO": site_no, "year": year, "patch_id": patch_id, "qa_dir": str(qa_dir)})

        summary_samples.append(
            {
                "SITE_NO": site_no,
                "year": year,
                "patch_id": patch_id,
                "all_region_count": int(len(components)),
                "sam_input_image_path": str(chosen_sam_input_path),
                "sam_preprocess": chosen_preprocess_info,
                "sam_retry": retry_record,
                "road_rescue": rescue_stats,
                "parking_pixels_after_refine": int(parking_refine_stats["pixels_after"]),
                "prompt_records": prompt_records,
            }
        )
        print(f"[{row_idx}/{len(manifest)}] built SamGeo3 v15b all-region tokens for site {site_no} year {year}", flush=True)

    token_df = pd.DataFrame(rows)
    token_csv = out_dir / "sam_region_tokens.csv"
    token_df.to_csv(token_csv, index=False)
    pd.DataFrame(rescue_records).to_csv(out_dir / "road_rescue_stats.csv", index=False)
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "year_root_catalog_csv": str(args.year_root_catalog_csv),
        "row_count": int(len(token_df)),
        "site_year_count": int(token_df[["SITE_NO", "year"]].drop_duplicates().shape[0]) if not token_df.empty else 0,
        "mean_regions_per_site_year": float(0.0 if not region_counts else np.mean(region_counts)),
        "max_regions_per_site_year": int(0 if not region_counts else max(region_counts)),
        "frontend": "samgeo3_v5_semantic_components_allregions_v15b",
        "all_regions_preserved": True,
        "road_rescue_enabled": not bool(args.disable_road_prior_rescue),
        "road_geojson": str(args.road_geojson),
        "road_line_count": int(len(road_lines)),
        "road_rescue_pixels_sum": int(sum(int(record.get("rescued_pixels", 0)) for record in rescue_records)),
        "preprocess_settings": {
            "autocontrast_cutoff": float(args.preprocess_autocontrast_cutoff),
            "contrast": float(args.preprocess_contrast),
            "sharpness": float(args.preprocess_sharpness),
            "adaptive_preprocess": bool(args.adaptive_preprocess),
            "adaptive_mean_min": float(args.adaptive_mean_min),
            "adaptive_mean_max": float(args.adaptive_mean_max),
            "adaptive_std_min": float(args.adaptive_std_min),
            "adaptive_std_max": float(args.adaptive_std_max),
            "adaptive_p10_min": float(args.adaptive_p10_min),
            "adaptive_autocontrast_cutoff": float(args.adaptive_autocontrast_cutoff),
            "adaptive_contrast": float(args.adaptive_contrast),
            "adaptive_sharpness": float(args.adaptive_sharpness),
            "retry_strong_preprocess_when_no_road": bool(args.retry_strong_preprocess_when_no_road),
            "retry_preprocess_autocontrast_cutoff": float(args.retry_preprocess_autocontrast_cutoff),
            "retry_preprocess_contrast": float(args.retry_preprocess_contrast),
            "retry_preprocess_sharpness": float(args.retry_preprocess_sharpness),
        },
        "class_prompts": CLASS_PROMPTS,
        "merge_settings": {
            "class_score_bias": get_class_score_bias(args),
            "class_min_merge_score": get_class_min_merge_score(args),
            "parking_road_margin": float(args.parking_road_margin),
            "parking_road_suppress_margin": float(args.parking_road_suppress_margin),
            "parking_thin_max_width": int(args.parking_thin_max_width),
            "parking_thin_min_aspect": float(args.parking_thin_min_aspect),
            "parking_thin_max_area": int(args.parking_thin_max_area),
            "building_road_margin": float(args.building_road_margin),
            "road_rescue_green_guard_radius": int(args.road_rescue_green_guard_radius),
            "road_rescue_safe_min_component": int(args.road_rescue_safe_min_component),
            "road_rescue_accepted_min_component": int(args.road_rescue_accepted_min_component),
            "road_rescue_score": float(args.road_rescue_score),
        },
        "qa_count": int(len(qa_records)),
        "qa_root": str(qa_root),
        "samples": summary_samples[: min(len(summary_samples), args.render_qa_count)],
    }
    (out_dir / "sam_region_tokens_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "qa_index.json").write_text(json.dumps(qa_records, indent=2), encoding="utf-8")
    print(token_csv)


if __name__ == "__main__":
    main()
