from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from build_phase2_roof_retrieval_composites import affine_between_masks


# Building-level subregions for applications whose drawing package covers
# several development footprints. Coordinates are normalized x0, y0, x1, y1.
MANUAL_BUILDING_ROIS: dict[str, tuple[float, float, float, float, str]] = {
    # TPD-2012-41/A roof plan: two separately recorded development footprints.
    "X001112": (0.44, 0.29, 0.69, 0.52, "stage_a"),
    "X001113": (0.30, 0.47, 0.56, 0.70, "stage_b"),
    # TPMR-2019-29/A concept plan: isolate the hatched future-development block.
    "X0012807": (0.27, 0.35, 0.57, 0.71, "future_development"),
    # TPM-2021-19 site plan: Building 01-05 map west-to-east/north-to-south
    # to the five geocoded development footprints.
    "X0011609": (0.15, 0.53, 0.37, 0.72, "building_01"),
    "X0011610": (0.31, 0.52, 0.55, 0.72, "building_02"),
    "X0011611": (0.29, 0.25, 0.50, 0.58, "building_03"),
    "X0011612": (0.48, 0.23, 0.70, 0.51, "building_04"),
    "X0011613": (0.50, 0.44, 0.70, 0.72, "building_05"),
    # TPM-2014-5: western Stage 1 and eastern Stage 2.
    "X000752": (0.24, 0.29, 0.46, 0.58, "stage_1_west"),
    "X000753": (0.41, 0.29, 0.63, 0.58, "stage_2_east"),
    # TPM-2012-4/A: large western L-shaped building and eastern oval building.
    "X000618": (0.24, 0.32, 0.55, 0.72, "western_large_building"),
    "X000619": (0.46, 0.31, 0.76, 0.70, "eastern_oval_building"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build building-specific, footprint-aligned CAD rasters for Melbourne pairs."
    )
    parser.add_argument(
        "--pair-manifest-csv",
        type=Path,
        default=Path(
            "outputs/phase2_melbourne_real_cad_pairs_v1/final_paired_dataset_v1/"
            "melbourne_cad_satellite_train_eval_manifest.csv"
        ),
    )
    parser.add_argument(
        "--page-selection-csv",
        type=Path,
        default=Path(
            "outputs/phase2_melbourne_real_cad_pairs_v1/final_paired_dataset_v1/"
            "selected_cad_pages_by_application.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/phase2_melbourne_real_cad_pairs_v1/spatial_cad_rasters_v1"
        ),
    )
    parser.add_argument("--gallery-rows", type=int, default=9)
    return parser.parse_args()


def resolve_path(value: object, project_root: Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else project_root / path


def choose_spatial_page(row: pd.Series) -> tuple[str, str, str]:
    forced = str(row.get("forced_spatial_page_type", "") or "")
    if forced and forced.lower() != "nan":
        ink = str(row.get(f"selected_{forced}_ink_png", "") or "")
        page = str(row.get(f"selected_{forced}_page_png", "") or "")
        if ink and ink.lower() != "nan":
            return forced, page, ink
    for page_type in ["roof", "site", "floor"]:
        ink = str(row.get(f"selected_{page_type}_ink_png", "") or "")
        page = str(row.get(f"selected_{page_type}_page_png", "") or "")
        if ink and ink.lower() != "nan":
            return page_type, page, ink
    raise RuntimeError(f"No spatial CAD page for {row['town_planning_application']}")


def normalized_roi(shape: tuple[int, int], roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = roi
    return (
        max(int(round(x0 * width)), 0),
        max(int(round(y0 * height)), 0),
        min(int(round(x1 * width)), width),
        min(int(round(y1 * height)), height),
    )


def auto_plan_crop(ink: np.ndarray) -> tuple[int, int, int, int, dict[str, float]]:
    height, width = ink.shape
    binary = ink > 18
    # Suppress page-number/header and common right/bottom title-block zones.
    binary[: int(0.08 * height)] = False
    binary[int(0.93 * height) :] = False
    binary[:, : int(0.03 * width)] = False
    binary[:, int(0.96 * width) :] = False

    grid_size = 48
    small = cv2.resize(binary.astype(np.uint8), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    active = small > 0
    active = ndimage.binary_dilation(active, iterations=2)
    active = ndimage.binary_closing(active, iterations=2)
    labels, count = ndimage.label(active)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if len(xs) < 4:
            continue
        gx0, gx1 = int(xs.min()), int(xs.max()) + 1
        gy0, gy1 = int(ys.min()), int(ys.max()) + 1
        box_area = (gx1 - gx0) * (gy1 - gy0)
        aspect = max((gx1 - gx0) / max(gy1 - gy0, 1), (gy1 - gy0) / max(gx1 - gx0, 1))
        centre_x = (gx0 + gx1) / 2 / grid_size
        centre_y = (gy0 + gy1) / 2 / grid_size
        centrality = math.exp(-2.0 * ((centre_x - 0.48) ** 2 + (centre_y - 0.48) ** 2))
        edge_penalty = 0.55 if gx0 > int(0.72 * grid_size) or gy0 > int(0.78 * grid_size) else 1.0
        aspect_penalty = 0.55 if aspect > 5.0 else 1.0
        score = box_area * centrality * edge_penalty * aspect_penalty
        if best is None or score > best[0]:
            best = (score, (gx0, gy0, gx1, gy1))
    if best is None:
        return 0, 0, width, height, {"auto_crop_score": 0.0}
    gx0, gy0, gx1, gy1 = best[1]
    margin = 2
    gx0, gy0 = max(gx0 - margin, 0), max(gy0 - margin, 0)
    gx1, gy1 = min(gx1 + margin, grid_size), min(gy1 + margin, grid_size)
    x0 = int(round(gx0 / grid_size * width))
    y0 = int(round(gy0 / grid_size * height))
    x1 = int(round(gx1 / grid_size * width))
    y1 = int(round(gy1 / grid_size * height))
    return x0, y0, x1, y1, {
        "auto_crop_score": float(best[0]),
        "auto_crop_area_ratio": float((x1 - x0) * (y1 - y0) / (width * height)),
    }


def trim_crop_to_ink(crop: np.ndarray, padding_ratio: float = 0.05) -> np.ndarray:
    mask = crop > 14
    if not mask.any():
        return crop
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = max(int(round(max(x1 - x0, y1 - y0) * padding_ratio)), 4)
    return crop[
        max(y0 - padding, 0) : min(y1 + padding, crop.shape[0]),
        max(x0 - padding, 0) : min(x1 + padding, crop.shape[1]),
    ]


def square_canvas(image: np.ndarray, size: int = 512) -> np.ndarray:
    height, width = image.shape
    scale = min((size - 24) / max(width, 1), (size - 24) / max(height, 1))
    resized = cv2.resize(
        image,
        (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def structural_line_map(panel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enhanced = cv2.normalize(panel, None, 0, 255, cv2.NORM_MINMAX)
    line = enhanced > max(int(np.percentile(enhanced[enhanced > 0], 35)) if (enhanced > 0).any() else 20, 16)
    line = ndimage.binary_opening(line, structure=np.ones((2, 2)))
    line = ndimage.binary_dilation(line, iterations=1)
    support = ndimage.binary_dilation(line, iterations=4)
    support = ndimage.binary_closing(support, iterations=5)
    labels, count = ndimage.label(support)
    if count:
        sizes = ndimage.sum(support, labels, range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        support = labels == keep
        support = ndimage.binary_fill_holes(support)
    if support.sum() < 25:
        support = line.copy()
    return line, support


def align_to_footprint(
    panel_line: np.ndarray,
    panel_support: np.ndarray,
    footprint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = affine_between_masks(panel_support, footprint)
    height, width = footprint.shape
    density = ndimage.gaussian_filter(panel_line.astype(np.float32), sigma=3.0)
    if density.max() > 0:
        density /= density.max()
    distance = ndimage.distance_transform_edt(~panel_line)
    proximity = np.exp(-distance / 10.0).astype(np.float32) * panel_support.astype(np.float32)

    warped_line = cv2.warpAffine(
        panel_line.astype(np.uint8) * 255,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_density = cv2.warpAffine(
        (density * 255.0).round().astype(np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_proximity = cv2.warpAffine(
        (proximity * 255.0).round().astype(np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    clip = ndimage.binary_dilation(footprint, iterations=5)
    warped_line[~clip] = 0
    warped_density[~clip] = 0
    warped_proximity[~clip] = 0
    return warped_line, warped_density, warped_proximity


def save_galleries(records: list[dict[str, object]], out_dir: Path, rows_per_part: int) -> list[str]:
    paths: list[str] = []
    for part_index in range(math.ceil(len(records) / rows_per_part)):
        part = records[part_index * rows_per_part : (part_index + 1) * rows_per_part]
        fig, axes = plt.subplots(len(part), 6, figsize=(19, len(part) * 3.05), squeeze=False)
        for row_index, record in enumerate(part):
            overlay = record["source"].copy()
            aligned = record["aligned"] > 12
            overlay[aligned] = [0, 255, 255]
            cad_multichannel = np.stack(
                [record["aligned"], record["density"], record["proximity"]], axis=2
            )
            panels = [
                record["full_ink"],
                record["panel"],
                record["line"],
                cad_multichannel,
                overlay,
                record["target"],
            ]
            titles = [
                "full CAD page",
                "building panel",
                "structural lines",
                "aligned line/density/proximity",
                "before + CAD",
                "true after",
            ]
            prefix = f"{record['sample_id']}\n{record['strategy']} | {record['page_type']}"
            for column, (panel, title) in enumerate(zip(panels, titles)):
                axes[row_index, column].imshow(panel, cmap="gray" if column < 3 else None)
                axes[row_index, column].set_title(f"{prefix}\n{title}", fontsize=7)
                axes[row_index, column].axis("off")
        fig.tight_layout()
        path = out_dir / f"spatial_cad_review_gallery_part_{part_index + 1}.jpg"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path.cwd()
    pairs = pd.read_csv(args.pair_manifest_csv, dtype={"development_key": str})
    pages = pd.read_csv(args.page_selection_csv)
    page_by_application = pages.set_index("town_planning_application")
    records: list[dict[str, object]] = []
    galleries: list[dict[str, object]] = []

    for _, row in pairs.iterrows():
        application = str(row["town_planning_application"])
        development_key = str(row["development_key"])
        page_record = (
            row
            if str(row.get("forced_spatial_page_type", "") or "").lower() not in {"", "nan"}
            else page_by_application.loc[application]
        )
        page_type, page_png, ink_png = choose_spatial_page(page_record)
        full_ink = np.asarray(Image.open(resolve_path(ink_png, project_root)).convert("L"), dtype=np.uint8)
        if development_key in MANUAL_BUILDING_ROIS:
            roi = MANUAL_BUILDING_ROIS[development_key]
            x0, y0, x1, y1 = normalized_roi(full_ink.shape, roi[:4])
            strategy = f"manual_{roi[4]}"
            crop_meta = {"auto_crop_score": np.nan, "auto_crop_area_ratio": np.nan}
        else:
            x0, y0, x1, y1, crop_meta = auto_plan_crop(full_ink)
            strategy = "automatic_main_plan"
        panel_raw = full_ink[y0:y1, x0:x1]
        panel = square_canvas(trim_crop_to_ink(panel_raw), 512)
        line, support = structural_line_map(panel)
        footprint = np.asarray(Image.open(resolve_path(row["footprint_mask_png"], project_root)).convert("L")) > 0
        aligned, density, proximity = align_to_footprint(line, support, footprint)

        sample_dir = args.out_dir / str(row["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        panel_path = sample_dir / "cad_building_panel.png"
        line_path = sample_dir / "cad_building_line_map.png"
        aligned_path = sample_dir / "cad_aligned_raster.png"
        density_path = sample_dir / "cad_aligned_density.png"
        proximity_path = sample_dir / "cad_aligned_proximity.png"
        Image.fromarray(panel).save(panel_path, optimize=True)
        Image.fromarray(line.astype(np.uint8) * 255).save(line_path, optimize=True)
        Image.fromarray(aligned).save(aligned_path, optimize=True)
        Image.fromarray(density).save(density_path, optimize=True)
        Image.fromarray(proximity).save(proximity_path, optimize=True)

        record = row.to_dict()
        record.update(
            {
                "spatial_cad_page_type": page_type,
                "spatial_cad_page_png": page_png,
                "spatial_cad_ink_png": ink_png,
                "spatial_cad_crop_strategy": strategy,
                "spatial_cad_crop_x0": x0,
                "spatial_cad_crop_y0": y0,
                "spatial_cad_crop_x1": x1,
                "spatial_cad_crop_y1": y1,
                "spatial_cad_panel_png": str(panel_path),
                "spatial_cad_line_png": str(line_path),
                "spatial_cad_aligned_png": str(aligned_path),
                "spatial_cad_density_png": str(density_path),
                "spatial_cad_proximity_png": str(proximity_path),
                "spatial_cad_line_fraction_panel": float(line.mean()),
                "spatial_cad_support_fraction_panel": float(support.mean()),
                "spatial_cad_line_fraction_footprint": float((aligned[footprint] > 12).mean()),
                "spatial_cad_density_mean_footprint": float(density[footprint].mean() / 255.0),
                "spatial_cad_proximity_mean_footprint": float(
                    proximity[footprint].mean() / 255.0
                ),
                "spatial_cad_usable": bool((aligned[footprint] > 12).mean() > 0.01),
                **crop_meta,
            }
        )
        records.append(record)
        source = np.asarray(Image.open(resolve_path(row["aligned_source_crop_png"], project_root)).convert("RGB"))
        target = np.asarray(Image.open(resolve_path(row["aligned_target_crop_png"], project_root)).convert("RGB"))
        galleries.append(
            {
                "sample_id": row["sample_id"],
                "strategy": strategy,
                "page_type": page_type,
                "full_ink": full_ink,
                "panel": panel,
                "line": line,
                "aligned": aligned,
                "density": density,
                "proximity": proximity,
                "source": source,
                "target": target,
            }
        )
        print(f"{row['sample_id']} -> {strategy} ({page_type})", flush=True)

    output = pd.DataFrame(records)
    manifest_path = args.out_dir / "melbourne_spatial_cad_pair_manifest.csv"
    output.to_csv(manifest_path, index=False)
    galleries_out = save_galleries(galleries, args.out_dir, args.gallery_rows)
    summary = {
        "pair_count": int(len(output)),
        "application_count": int(output["town_planning_application"].nunique()),
        "manual_building_roi_count": int(output["spatial_cad_crop_strategy"].str.startswith("manual_").sum()),
        "automatic_crop_count": int(output["spatial_cad_crop_strategy"].eq("automatic_main_plan").sum()),
        "page_type_counts": output["spatial_cad_page_type"].value_counts().to_dict(),
        "usable_count": int(output["spatial_cad_usable"].sum()),
        "median_line_fraction_footprint": float(output["spatial_cad_line_fraction_footprint"].median()),
        "manifest": str(manifest_path),
        "galleries": galleries_out,
        "manual_roi_mapping": {
            key: {"normalized_roi": list(value[:4]), "label": value[4]}
            for key, value in MANUAL_BUILDING_ROIS.items()
        },
    }
    (args.out_dir / "spatial_cad_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
