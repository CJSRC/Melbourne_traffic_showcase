from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

from build_phase2_cad_to_semantic_mask_dataset import CLASS_COLORS, CLASS_TO_ID, colorize_label_map
from build_phase2_realplanning_dataset import iter_polygons, meters_per_degree_lon, to_pixel_ring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real development footprints against annual SAM building changes.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--study-area-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-size-m", type=float, default=500.0)
    parser.add_argument("--tolerance-pixels", type=int, default=6)
    parser.add_argument("--max-gallery", type=int, default=48)
    return parser.parse_args()


def load_binary(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def load_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def rasterize_footprint(
    geometry: dict,
    *,
    latitude: float,
    longitude: float,
    patch_size_m: float,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    height, width = shape_hw
    half_lat = (patch_size_m / 2.0) / 111_320.0
    half_lon = (patch_size_m / 2.0) / meters_per_degree_lon(latitude)
    north = latitude + half_lat
    south = latitude - half_lat
    west = longitude - half_lon
    east = longitude + half_lon
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in iter_polygons(geometry):
        if not polygon:
            continue
        exterior = to_pixel_ring(
            polygon[0], north=north, south=south, west=west, east=east, size_px=width
        )
        holes = [
            to_pixel_ring(ring, north=north, south=south, west=west, east=east, size_px=width)
            for ring in polygon[1:]
        ]
        draw.polygon(exterior, fill=255)
        for hole in holes:
            draw.polygon(hole, fill=0)
    return np.asarray(image, dtype=np.uint8) > 0


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    total = int(denominator.sum())
    return float(np.logical_and(numerator, denominator).sum() / total) if total else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def satellite_contrast(
    current_path: Path,
    target_path: Path,
    footprint: np.ndarray,
    tolerance: int,
) -> tuple[float, float]:
    if not current_path.exists() or not target_path.exists() or not footprint.any():
        return 0.0, 0.0
    size = footprint.shape[::-1]
    current = np.asarray(Image.open(current_path).convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    target = np.asarray(Image.open(target_path).convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    difference = np.mean(np.abs(current - target), axis=2)
    inside = float(difference[footprint].mean())
    outer = ndimage.binary_dilation(footprint, iterations=max(tolerance * 3, 8))
    inner = ndimage.binary_dilation(footprint, iterations=max(tolerance, 2))
    ring = outer & ~inner
    outside = float(difference[ring].mean()) if ring.any() else float(difference.mean())
    return inside, inside - outside


def verification_tier(row: dict[str, object]) -> str:
    footprint_pixels = int(row["footprint_pixels"])
    add_recall = float(row["add_recall_tolerant"])
    target_coverage = float(row["target_building_coverage"])
    gain = float(row["building_coverage_gain"])
    satellite = float(row["footprint_satellite_change_contrast"])
    if footprint_pixels >= 12 and target_coverage >= 0.45 and (add_recall >= 0.30 or gain >= 0.20) and satellite >= -0.01:
        return "high"
    if footprint_pixels >= 8 and target_coverage >= 0.25 and (add_recall >= 0.12 or gain >= 0.08) and satellite >= -0.04:
        return "medium"
    return "low"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidate_csv)
    area = pd.read_csv(args.study_area_csv)
    area["SITE_NO"] = pd.to_numeric(area["SITE_NO"], errors="coerce").astype(int)
    site_lookup = area.set_index("SITE_NO")[["LATITUDE", "LONGITUDE"]].to_dict(orient="index")
    records: list[dict[str, object]] = []

    for row in candidates.itertuples(index=False):
        site_no = int(row.SITE_NO)
        site = site_lookup.get(site_no)
        if site is None:
            continue
        required = [
            Path(str(row.add_building_mask_png)),
            Path(str(row.current_semantic_mask_png)),
            Path(str(row.target_semantic_mask_png)),
        ]
        if not all(path.exists() for path in required):
            continue
        add = load_binary(required[0])
        current = load_label(required[1])
        target = load_label(required[2])
        footprint = rasterize_footprint(
            json.loads(str(row.geometry_geojson)),
            latitude=float(site["LATITUDE"]),
            longitude=float(site["LONGITUDE"]),
            patch_size_m=args.patch_size_m,
            shape_hw=add.shape,
        )
        if not footprint.any():
            continue
        tolerant_add = ndimage.binary_dilation(add, iterations=args.tolerance_pixels)
        tolerant_footprint = ndimage.binary_dilation(footprint, iterations=args.tolerance_pixels)
        current_building = current == CLASS_TO_ID["building_roof"]
        target_building = target == CLASS_TO_ID["building_roof"]
        current_coverage = safe_ratio(current_building, footprint)
        target_coverage = safe_ratio(target_building, footprint)
        current_satellite = Path(str(getattr(row, "current_satellite_path", "")))
        target_satellite = Path(str(getattr(row, "target_satellite_path", "")))
        satellite_inside, satellite_change_contrast = satellite_contrast(
            current_satellite, target_satellite, footprint, args.tolerance_pixels
        )
        record = row._asdict()
        record.update(
            {
                "footprint_pixels": int(footprint.sum()),
                "add_pixels": int(add.sum()),
                "add_footprint_iou": mask_iou(add, footprint),
                "add_recall_exact": safe_ratio(add, footprint),
                "add_recall_tolerant": safe_ratio(tolerant_add, footprint),
                "add_precision_tolerant": safe_ratio(tolerant_footprint, add),
                "current_building_coverage": current_coverage,
                "target_building_coverage": target_coverage,
                "building_coverage_gain": target_coverage - current_coverage,
                "footprint_satellite_change_mean": satellite_inside,
                "footprint_satellite_change_contrast": satellite_change_contrast,
            }
        )
        record["spatial_verification_score"] = (
            0.40 * float(record["add_recall_tolerant"])
            + 0.25 * float(record["target_building_coverage"])
            + 0.20 * max(float(record["building_coverage_gain"]), 0.0)
            + 0.10 * float(record["add_precision_tolerant"])
            + 0.05 * max(min(5.0 * satellite_change_contrast + 0.5, 1.0), 0.0)
        )
        record["verification_tier"] = verification_tier(record)
        records.append(record)

    verified = pd.DataFrame(records)
    if verified.empty:
        raise RuntimeError("No candidate masks were available for real-event validation.")
    timing_rank = {"completion_year": 0, "completion_plus_1": 1}
    verified["timing_rank"] = verified["timing_hypothesis"].map(timing_rank).fillna(2)
    verified = verified.sort_values(
        ["development_key", "SITE_NO", "spatial_verification_score", "timing_rank"],
        ascending=[True, True, False, True],
    )
    verified["best_timing_for_event_site"] = ~verified.duplicated(["development_key", "SITE_NO"], keep="first")
    selected = verified[
        verified["best_timing_for_event_site"] & verified["verification_tier"].isin(["high", "medium"])
    ].copy()
    verified.to_csv(args.out_dir / "real_event_spatial_validation_all.csv", index=False)
    selected.to_csv(args.out_dir / "real_event_spatial_validation_selected.csv", index=False)

    gallery_rows = selected.sort_values("spatial_verification_score", ascending=False).head(args.max_gallery)
    if not gallery_rows.empty:
        fig, axes = plt.subplots(len(gallery_rows), 5, figsize=(16, max(3, len(gallery_rows) * 3)))
        if len(gallery_rows) == 1:
            axes = np.asarray([axes])
        for row_idx, row in enumerate(gallery_rows.itertuples(index=False)):
            site = site_lookup[int(row.SITE_NO)]
            add = load_binary(Path(str(row.add_building_mask_png)))
            current = load_label(Path(str(row.current_semantic_mask_png)))
            target = load_label(Path(str(row.target_semantic_mask_png)))
            footprint = rasterize_footprint(
                json.loads(str(row.geometry_geojson)),
                latitude=float(site["LATITUDE"]),
                longitude=float(site["LONGITUDE"]),
                patch_size_m=args.patch_size_m,
                shape_hw=add.shape,
            )
            overlay = colorize_label_map(target).copy()
            boundary = footprint & ~ndimage.binary_erosion(footprint, iterations=2)
            overlay[boundary] = [255, 0, 255]
            delta = np.zeros((*add.shape, 3), dtype=np.uint8)
            delta[add] = [60, 220, 80]
            delta[boundary] = [255, 0, 255]
            current_sat = Path(str(row.current_satellite_path))
            target_sat = Path(str(row.target_satellite_path))
            panels = [
                np.asarray(Image.open(current_sat).convert("RGB")) if current_sat.exists() else colorize_label_map(current),
                np.asarray(Image.open(target_sat).convert("RGB")) if target_sat.exists() else colorize_label_map(target),
                colorize_label_map(current),
                overlay,
                delta,
            ]
            titles = ["Current satellite", "Target satellite", "Current semantic", "Target + real footprint", "SAM add + footprint"]
            for col, (panel, title) in enumerate(zip(panels, titles)):
                axes[row_idx, col].imshow(panel)
                axes[row_idx, col].set_title(
                    f"{int(row.SITE_NO)} {int(row.source_year)}->{int(row.target_year)} | {title}", fontsize=8
                )
                axes[row_idx, col].axis("off")
        fig.tight_layout()
        fig.savefig(args.out_dir / "real_event_spatial_validation_gallery.jpg", dpi=180)
        plt.close(fig)

    summary = {
        "candidate_count": int(len(candidates)),
        "validated_count": int(len(verified)),
        "best_timing_event_site_count": int(verified["best_timing_for_event_site"].sum()),
        "selected_medium_high_count": int(len(selected)),
        "selected_unique_events": int(selected["development_key"].nunique()),
        "selected_unique_sites": int(selected["SITE_NO"].nunique()),
        "verification_tier_counts_all": verified["verification_tier"].value_counts().to_dict(),
        "selected_target_year_counts": {
            str(int(year)): int(count) for year, count in selected.groupby("target_year").size().items()
        },
        "selected_timing_counts": selected["timing_hypothesis"].value_counts().to_dict(),
        "mean_selected_add_recall_tolerant": float(selected["add_recall_tolerant"].mean()) if not selected.empty else 0.0,
        "mean_selected_building_coverage_gain": float(selected["building_coverage_gain"].mean()) if not selected.empty else 0.0,
    }
    (args.out_dir / "real_event_spatial_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
