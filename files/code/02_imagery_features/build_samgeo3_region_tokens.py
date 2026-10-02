from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw
from skimage.measure import label, regionprops

from hf_cache_utils import rebase_output_root
from run_samgeo3_semantic_map import (
    CLASS_PALETTE,
    CLASS_PRIORITY,
    CLASS_PROMPTS,
    add_local_sam3_repo,
    build_class_mask,
    get_class_min_merge_score,
    get_class_score_bias,
    merge_semantic_maps,
    refine_parking_mask,
    render_flat_mask,
    render_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Route 1/2 region tokens from the SamGeo3 semantic frontend.")
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
    parser.add_argument("--parking-lot-min-score", type=float, default=0.50)
    parser.add_argument("--parking-lot-strong-score", type=float, default=0.62)
    parser.add_argument("--parking-aux-min-score", type=float, default=0.45)
    parser.add_argument("--parking-road-suppress-margin", type=float, default=0.16)
    parser.add_argument("--parking-thin-max-width", type=int, default=5)
    parser.add_argument("--parking-thin-min-aspect", type=float, default=5.0)
    parser.add_argument("--parking-thin-max-area", type=int, default=180)
    parser.add_argument("--road-score-bias", type=float, default=0.0)
    parser.add_argument("--parking-score-bias", type=float, default=0.02)
    parser.add_argument("--building-score-bias", type=float, default=0.0)
    parser.add_argument("--vegetation-score-bias", type=float, default=0.0)
    parser.add_argument("--road-min-merge-score", type=float, default=0.38)
    parser.add_argument("--parking-min-merge-score", type=float, default=0.46)
    parser.add_argument("--building-min-merge-score", type=float, default=0.40)
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
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--render-qa-count", type=int, default=8)
    return parser.parse_args()


def build_year_root_map(catalog_csv: Path) -> dict[int, Path]:
    catalog = pd.read_csv(catalog_csv)
    year_col = "target_year" if "target_year" in catalog.columns else "year"
    outputs_root = Path(__file__).resolve().parent / "outputs"
    root_map: dict[int, Path] = {}
    for row in catalog.itertuples(index=False):
        year = int(getattr(row, year_col))
        if hasattr(row, "patch_root") and getattr(row, "patch_root"):
            patch_root = rebase_output_root(str(getattr(row, "patch_root")), outputs_root)
        elif hasattr(row, "embedding_csv") and getattr(row, "embedding_csv"):
            patch_root = rebase_output_root(str(Path(getattr(row, "embedding_csv")).resolve().parents[1]), outputs_root)
        else:
            raise AttributeError(f"Catalog row for year {year} has neither patch_root nor embedding_csv")
        root_map[year] = patch_root
    return root_map


def resolve_image_path(row: pd.Series, year_root_map: dict[int, Path]) -> Path:
    rel = Path(str(row["satellite_relative_image_path"]))
    root = year_root_map[int(row["year"])]
    return root / rel


def selection_score(area_ratio: float, semantic_confidence: float, support_strength: float, fill_ratio: float) -> float:
    return float(
        semantic_confidence
        + min(area_ratio * 12.0, 0.90)
        + 0.08 * support_strength
        + 0.05 * fill_ratio
    )


def select_diverse_components(components: list[dict[str, object]], max_regions: int) -> list[dict[str, object]]:
    if not components:
        return []
    by_class: dict[str, list[dict[str, object]]] = {class_name: [] for class_name in CLASS_PRIORITY}
    for comp in components:
        by_class.setdefault(str(comp["semantic_label"]), []).append(comp)
    for class_name in by_class:
        by_class[class_name] = sorted(
            by_class[class_name],
            key=lambda item: (float(item["selection_score"]), float(item["area_ratio"])),
            reverse=True,
        )

    selected: list[dict[str, object]] = []
    used_ids: set[int] = set()

    # First pass: guarantee one representative component for each present class.
    for class_name in CLASS_PRIORITY:
        class_components = by_class.get(class_name, [])
        if not class_components:
            continue
        comp = class_components[0]
        selected.append(comp)
        used_ids.add(id(comp))
        if len(selected) >= max_regions:
            return selected[:max_regions]

    # Second pass: fill the remaining slots with the strongest leftovers.
    leftovers = [comp for comp in components if id(comp) not in used_ids]
    leftovers = sorted(leftovers, key=lambda item: (float(item["selection_score"]), float(item["area_ratio"])), reverse=True)
    for comp in leftovers:
        selected.append(comp)
        if len(selected) >= max_regions:
            break
    return selected[:max_regions]


def crop_masked_region(image_arr: np.ndarray, component_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = image_arr[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = component_mask[y0 : y1 + 1, x0 : x1 + 1]
    crop[~crop_mask] = 0
    return Image.fromarray(crop, mode="RGB")


def save_component_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def render_component_overlay(image_arr: np.ndarray, components: list[dict[str, object]], out_path: Path) -> None:
    rgba = Image.fromarray(image_arr).convert("RGBA")
    for comp in components:
        class_name = str(comp["semantic_label"])
        color = ImageColor.getrgb(CLASS_PALETTE.get(class_name, CLASS_PALETTE["other"]))
        mask = comp["component_mask"]
        layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        layer[mask] = (*color, 110)
        rgba = Image.alpha_composite(rgba, Image.fromarray(layer, mode="RGBA"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)


def render_topk_montage(image_arr: np.ndarray, components: list[dict[str, object]], out_path: Path) -> None:
    if not components:
        Image.new("RGB", (256, 64), "white").save(out_path)
        return
    tile = 96
    label_h = 18
    cols = 4
    rows = math.ceil(len(components) / cols)
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, comp in enumerate(components):
        bbox = comp["bbox"]
        crop = crop_masked_region(image_arr, comp["component_mask"], bbox).resize((tile, tile))
        x = (idx % cols) * tile
        y = (idx // cols) * (tile + label_h)
        canvas.paste(crop, (x, y))
        draw.text((x + 2, y + tile + 2), f"{comp['region_rank']} {comp['semantic_label']}", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


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

    for row_idx, manifest_row in enumerate(manifest.itertuples(index=False), start=1):
        site_no = int(getattr(manifest_row, "SITE_NO"))
        year = int(getattr(manifest_row, "year"))
        patch_id = str(getattr(manifest_row, "satellite_patch_id"))
        image_rel = str(getattr(manifest_row, "satellite_relative_image_path"))
        image_path = resolve_image_path(pd.Series(manifest_row._asdict()), year_root_map)
        sam.set_image(str(image_path))
        image_arr = np.asarray(Image.open(image_path).convert("RGB"))

        class_masks: dict[str, np.ndarray] = {}
        class_score_maps: dict[str, np.ndarray] = {}
        prompt_records: dict[str, list[dict[str, object]]] = {}
        extra_maps_by_class: dict[str, dict[str, np.ndarray]] = {}

        for class_name, prompts in CLASS_PROMPTS.items():
            class_mask, class_score_map, prompt_record_list, extra_maps = build_class_mask(
                sam=sam,
                prompts=prompts,
                min_mask_size=args.min_mask_size,
                class_name=class_name,
                args=args,
            )
            class_masks[class_name] = class_mask
            class_score_maps[class_name] = class_score_map
            prompt_records[class_name] = prompt_record_list
            extra_maps_by_class[class_name] = extra_maps

        parking_mask, parking_score_map, parking_refine_stats = refine_parking_mask(
            parking_mask=class_masks["parking_paved"],
            parking_score_map=class_score_maps["parking_paved"],
            parking_extra_maps=extra_maps_by_class.get("parking_paved", {}),
            road_score_map=class_score_maps["road"],
            building_score_map=class_score_maps["building_roof"],
            args=args,
        )
        class_masks["parking_paved"] = parking_mask
        class_score_maps["parking_paved"] = parking_score_map

        height, width = image_arr.shape[:2]
        semantic_map, best_score = merge_semantic_maps(class_score_maps=class_score_maps, args=args)

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
                area_pixels = int(prop.area)
                area_ratio = float(area_pixels / image_area)
                bbox_w = int(x1 - x0 + 1)
                bbox_h = int(y1 - y0 + 1)
                fill_ratio = float(area_pixels / max(bbox_w * bbox_h, 1))
                semantic_confidence = float(score_map[component_mask].mean()) if np.any(component_mask) else 0.0
                sam_score = float(score_map[component_mask].max()) if np.any(component_mask) else 0.0
                support_strength = float(support_map[component_mask].mean() / prompt_count) if np.any(component_mask) else 0.0
                consistency_support = fill_ratio
                components.append(
                    {
                        "SITE_NO": site_no,
                        "year": year,
                        "dataset_split": str(getattr(manifest_row, "dataset_split")),
                        "patch_id": patch_id,
                        "site_name": str(getattr(manifest_row, "SITE_NAME")),
                        "relative_image_path": image_rel,
                        "resolved_image_path": str(image_path),
                        "semantic_label": class_name,
                        "prompt_semantic_class": class_name,
                        "proposal_source": "samgeo3_v5_component",
                        "semantic_source": "samgeo3_v5",
                        "prompt_confidence": semantic_confidence,
                        "bbox": (x0, y0, x1, y1),
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
                        "consistency_support": consistency_support,
                        "selection_score": selection_score(area_ratio, semantic_confidence, support_strength, fill_ratio),
                        "component_mask": component_mask,
                    }
                )

        components = sorted(components, key=lambda item: (float(item["selection_score"]), float(item["area_ratio"])), reverse=True)
        selected_components = select_diverse_components(components, args.max_regions)
        for region_rank, comp in enumerate(selected_components, start=1):
            mask_rel = Path("masks") / f"year_{year}" / f"{patch_id}__region_{region_rank:02d}.png"
            mask_abs = out_dir / mask_rel
            save_component_mask(comp["component_mask"], mask_abs)
            comp["region_rank"] = region_rank
            comp["mask_relative_path"] = str(mask_rel)
            comp["mask_path"] = str(mask_abs)
            rows.append(
                {
                    "SITE_NO": comp["SITE_NO"],
                    "year": comp["year"],
                    "dataset_split": comp["dataset_split"],
                    "patch_id": comp["patch_id"],
                    "site_name": comp["site_name"],
                    "relative_image_path": comp["relative_image_path"],
                    "resolved_image_path": comp["resolved_image_path"],
                    "region_rank": comp["region_rank"],
                    "mask_relative_path": comp["mask_relative_path"],
                    "mask_path": comp["mask_path"],
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
            Image.fromarray(image_arr, mode="RGB").save(qa_dir / "original.png")
            render_overlay(image_arr, semantic_map, qa_dir / "semantic_overlay.png")
            render_flat_mask(semantic_map, qa_dir / "semantic_mask.png")
            render_component_overlay(image_arr, selected_components, qa_dir / "sam_overlay.png")
            render_topk_montage(image_arr, selected_components, qa_dir / "topk_montage.png")
            qa_records.append(
                {
                    "SITE_NO": site_no,
                    "year": year,
                    "patch_id": patch_id,
                    "qa_dir": str(qa_dir),
                }
            )

        summary_samples.append(
            {
                "SITE_NO": site_no,
                "year": year,
                "patch_id": patch_id,
                "selected_region_count": int(len(selected_components)),
                "parking_pixels_after_refine": int(parking_refine_stats["pixels_after"]),
                "prompt_records": prompt_records,
            }
        )
        print(f"[{row_idx}/{len(manifest)}] built SamGeo3 tokens for site {site_no} year {year}", flush=True)

    token_df = pd.DataFrame(rows)
    token_csv = out_dir / "sam_region_tokens.csv"
    token_df.to_csv(token_csv, index=False)
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "year_root_catalog_csv": str(args.year_root_catalog_csv),
        "row_count": int(len(token_df)),
        "site_year_count": int(token_df[["SITE_NO", "year"]].drop_duplicates().shape[0]) if not token_df.empty else 0,
        "mean_regions_per_site_year": float(0.0 if token_df.empty else len(token_df) / max(len(manifest), 1)),
        "max_regions": int(args.max_regions),
        "frontend": "samgeo3_v5_semantic_components",
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
