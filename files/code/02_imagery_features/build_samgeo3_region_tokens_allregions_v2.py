from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops

from build_samgeo3_region_tokens import (
    CLASS_PRIORITY,
    CLASS_PROMPTS,
    add_local_sam3_repo,
    build_class_mask,
    get_class_min_merge_score,
    get_class_score_bias,
    merge_semantic_maps,
    refine_parking_mask,
    render_component_overlay,
    render_flat_mask,
    render_overlay,
    render_topk_montage,
    resolve_image_path,
    save_component_mask,
    selection_score,
    build_year_root_map,
)
from run_samgeo3_semantic_map import maybe_preprocess_image_for_sam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SamGeo3 region tokens while preserving all semantic components (v2 safe QA).")
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
    parser.add_argument("--parking-score-bias", type=float, default=0.0)
    parser.add_argument("--building-score-bias", type=float, default=0.05)
    parser.add_argument("--vegetation-score-bias", type=float, default=0.0)
    parser.add_argument("--road-min-merge-score", type=float, default=0.34)
    parser.add_argument("--parking-min-merge-score", type=float, default=0.46)
    parser.add_argument("--building-min-merge-score", type=float, default=0.38)
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
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--render-qa-count", type=int, default=8)
    parser.add_argument("--qa-top-k", type=int, default=12)
    return parser.parse_args()


def sum_kept_masks(prompt_records: dict[str, list[dict[str, object]]], class_name: str) -> int:
    return int(sum(int(record.get("mask_count_kept", 0)) for record in prompt_records.get(class_name, [])))


def run_semantic_pass(
    sam: object,
    image_arr: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
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

    return {
        "class_masks": class_masks,
        "class_score_maps": class_score_maps,
        "prompt_records": prompt_records,
        "extra_maps_by_class": extra_maps_by_class,
        "parking_refine_stats": parking_refine_stats,
        "semantic_map": semantic_map,
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

    for row_idx, manifest_row in enumerate(manifest.itertuples(index=False), start=1):
        site_no = int(getattr(manifest_row, "SITE_NO"))
        year = int(getattr(manifest_row, "year"))
        patch_id = str(getattr(manifest_row, "satellite_patch_id"))
        image_rel = str(getattr(manifest_row, "satellite_relative_image_path"))
        image_path = resolve_image_path(pd.Series(manifest_row._asdict()), year_root_map)
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
                        "sam_input_image_path": str(chosen_sam_input_path),
                        "sam_preprocess_mode": str(chosen_preprocess_info["mode"]),
                        "semantic_label": class_name,
                        "prompt_semantic_class": class_name,
                        "proposal_source": "samgeo3_v5_component_allregions_v2",
                        "semantic_source": "samgeo3_v5_allregions_v2",
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
                        "consistency_support": consistency_support,
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
                "parking_pixels_after_refine": int(parking_refine_stats["pixels_after"]),
                "prompt_records": prompt_records,
            }
        )
        print(f"[{row_idx}/{len(manifest)}] built SamGeo3 all-region tokens for site {site_no} year {year}", flush=True)

    token_df = pd.DataFrame(rows)
    token_csv = out_dir / "sam_region_tokens.csv"
    token_df.to_csv(token_csv, index=False)
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "year_root_catalog_csv": str(args.year_root_catalog_csv),
        "row_count": int(len(token_df)),
        "site_year_count": int(token_df[["SITE_NO", "year"]].drop_duplicates().shape[0]) if not token_df.empty else 0,
        "mean_regions_per_site_year": float(0.0 if not region_counts else np.mean(region_counts)),
        "max_regions_per_site_year": int(0 if not region_counts else max(region_counts)),
        "frontend": "samgeo3_v5_semantic_components_allregions_v2",
        "all_regions_preserved": True,
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
