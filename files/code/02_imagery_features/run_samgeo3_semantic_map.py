import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageEnhance, ImageOps
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_holes, remove_small_objects


CLASS_PROMPTS = {
    "road": [
        "asphalt road carriageways and intersections",
        "street lanes and roadway surfaces seen from above",
        "urban traffic lanes between kerbs in satellite view",
    ],
    "building_roof": [
        "building roofs",
        "residential house roofs in overhead satellite view",
        "commercial and industrial building rooftops",
        "dark roofs and light roofs of urban buildings",
        "light colored roof surfaces and roof blocks",
    ],
    "vegetation": [
        "trees and vegetation",
        "tree canopy and green vegetation",
    ],
    "parking_paved": [
        "surface parking lot with parked cars",
        "parking bays with white markings",
        "parking area beside buildings with parked cars",
        "open parking lot next to buildings",
    ],
}

CLASS_PRIORITY = ["road", "parking_paved", "building_roof", "vegetation"]
CLASS_PALETTE = {
    "road": "#FB8C00",
    "parking_paved": "#FDD835",
    "building_roof": "#8E24AA",
    "vegetation": "#4CAF50",
    "other": "#78909C",
}

CLASS_SCORE_BIAS = {
    "road": 0.00,
    "parking_paved": 0.00,
    "building_roof": 0.05,
    "vegetation": 0.00,
}

CLASS_MIN_MERGE_SCORE = {
    "road": 0.34,
    "parking_paved": 0.46,
    "building_roof": 0.38,
    "vegetation": 0.42,
}

PARKING_PROMPT_GROUPS = {
    "surface parking lot with parked cars": "cars",
    "parking bays with white markings": "markings",
    "parking area beside buildings with parked cars": "cars",
    "open parking lot next to buildings": "lot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a merged semantic map using SamGeo3 class-wise prompts.")
    parser.add_argument("--image-path", type=Path, required=True)
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
    parser.add_argument("--road-score-bias", type=float, default=CLASS_SCORE_BIAS["road"])
    parser.add_argument("--parking-score-bias", type=float, default=CLASS_SCORE_BIAS["parking_paved"])
    parser.add_argument("--building-score-bias", type=float, default=CLASS_SCORE_BIAS["building_roof"])
    parser.add_argument("--vegetation-score-bias", type=float, default=CLASS_SCORE_BIAS["vegetation"])
    parser.add_argument("--road-min-merge-score", type=float, default=CLASS_MIN_MERGE_SCORE["road"])
    parser.add_argument("--parking-min-merge-score", type=float, default=CLASS_MIN_MERGE_SCORE["parking_paved"])
    parser.add_argument("--building-min-merge-score", type=float, default=CLASS_MIN_MERGE_SCORE["building_roof"])
    parser.add_argument("--vegetation-min-merge-score", type=float, default=CLASS_MIN_MERGE_SCORE["vegetation"])
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
    return parser.parse_args()


def add_local_sam3_repo(project_root: Path) -> None:
    repo = project_root / "external" / "sam3"
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return clean or "prompt"


def mask_to_bool(mask: object) -> np.ndarray:
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr > 0


def compute_grayscale_stats(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "p10": float(np.percentile(gray, 10)),
        "p50": float(np.percentile(gray, 50)),
        "p90": float(np.percentile(gray, 90)),
    }


def should_apply_adaptive_preprocess(stats: dict[str, float], args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "adaptive_preprocess", False)):
        return False
    return (
        float(args.adaptive_mean_min) <= float(stats["mean"]) <= float(args.adaptive_mean_max)
        and float(args.adaptive_std_min) <= float(stats["std"]) <= float(args.adaptive_std_max)
        and float(stats["p10"]) >= float(args.adaptive_p10_min)
    )


def maybe_preprocess_image_for_sam(
    image_path: Path,
    out_path: Path | None,
    args: argparse.Namespace,
) -> tuple[Path, np.ndarray, dict[str, object]]:
    cutoff = float(getattr(args, "preprocess_autocontrast_cutoff", 0.0))
    contrast = float(getattr(args, "preprocess_contrast", 1.0))
    sharpness = float(getattr(args, "preprocess_sharpness", 1.0))

    image = Image.open(image_path).convert("RGB")
    original_stats = compute_grayscale_stats(image)
    applied_mode = "none"

    if should_apply_adaptive_preprocess(original_stats, args):
        cutoff = float(getattr(args, "adaptive_autocontrast_cutoff", cutoff))
        contrast = float(getattr(args, "adaptive_contrast", contrast))
        sharpness = float(getattr(args, "adaptive_sharpness", sharpness))
        applied_mode = "adaptive"
    elif cutoff > 0.0 or abs(contrast - 1.0) > 1e-6 or abs(sharpness - 1.0) > 1e-6:
        applied_mode = "fixed"

    use_preprocess = cutoff > 0.0 or abs(contrast - 1.0) > 1e-6 or abs(sharpness - 1.0) > 1e-6
    if cutoff > 0.0:
        image = ImageOps.autocontrast(image, cutoff=cutoff)
    if abs(contrast - 1.0) > 1e-6:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if abs(sharpness - 1.0) > 1e-6:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)

    processed_stats = compute_grayscale_stats(image)
    info = {
        "applied": bool(use_preprocess),
        "mode": applied_mode,
        "original_stats": original_stats,
        "processed_stats": processed_stats,
        "settings": {
            "autocontrast_cutoff": float(cutoff),
            "contrast": float(contrast),
            "sharpness": float(sharpness),
        },
    }

    if use_preprocess and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return out_path, np.asarray(image), info
    return image_path, np.asarray(image), info


def cleanup_class_mask(mask: np.ndarray, class_name: str, args: argparse.Namespace) -> np.ndarray:
    thresholds = {
        "road": int(args.road_min_component),
        "parking_paved": int(args.parking_min_component),
        "building_roof": int(args.building_min_component),
        "vegetation": int(args.vegetation_min_component),
    }
    cleaned = remove_small_objects(mask.astype(bool), min_size=thresholds.get(class_name, int(args.min_mask_size)))
    if class_name == "road":
        if int(args.road_closing_radius) > 0:
            cleaned = binary_closing(cleaned, footprint=disk(int(args.road_closing_radius)))
        if int(args.road_hole_area) > 0:
            cleaned = remove_small_holes(cleaned, area_threshold=int(args.road_hole_area))
    elif class_name in {"building_roof", "parking_paved"}:
        closing_radius = int(args.building_closing_radius) if class_name == "building_roof" else int(args.parking_closing_radius)
        hole_area = int(args.building_hole_area) if class_name == "building_roof" else int(args.parking_hole_area)
        if closing_radius > 0:
            cleaned = binary_closing(cleaned, footprint=disk(closing_radius))
        if hole_area > 0:
            cleaned = remove_small_holes(cleaned, area_threshold=hole_area)
    elif class_name == "vegetation":
        if int(args.vegetation_hole_area) > 0:
            cleaned = remove_small_holes(cleaned, area_threshold=int(args.vegetation_hole_area))
    return cleaned.astype(bool)


def get_class_score_bias(args: argparse.Namespace) -> dict[str, float]:
    return {
        "road": float(args.road_score_bias),
        "parking_paved": float(args.parking_score_bias),
        "building_roof": float(args.building_score_bias),
        "vegetation": float(args.vegetation_score_bias),
    }


def get_class_min_merge_score(args: argparse.Namespace) -> dict[str, float]:
    return {
        "road": float(args.road_min_merge_score),
        "parking_paved": float(args.parking_min_merge_score),
        "building_roof": float(args.building_min_merge_score),
        "vegetation": float(args.vegetation_min_merge_score),
    }


def build_class_mask(
    sam: object,
    prompts: list[str],
    min_mask_size: int,
    class_name: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, np.ndarray]]:
    combined = None
    score_map = None
    prompt_records: list[dict[str, object]] = []
    image_area = None
    support_count_map = None
    group_score_maps: dict[str, np.ndarray] = {}
    for prompt in prompts:
        sam.generate_masks(prompt, quiet=True)
        masks = sam.masks if sam.masks is not None else []
        scores = sam.scores if sam.scores is not None else []
        if combined is None:
            combined = np.zeros((sam.image_height, sam.image_width), dtype=bool)
            score_map = np.zeros((sam.image_height, sam.image_width), dtype=np.float32)
            image_area = float(sam.image_height * sam.image_width)
            support_count_map = np.zeros((sam.image_height, sam.image_width), dtype=np.uint8)
        kept = 0
        skipped_large = 0
        prompt_union = np.zeros((sam.image_height, sam.image_width), dtype=bool)
        prompt_score_map = np.zeros((sam.image_height, sam.image_width), dtype=np.float32)
        for idx, mask in enumerate(masks):
            mask_bool = mask_to_bool(mask)
            if int(mask_bool.sum()) < min_mask_size:
                continue
            if class_name == "parking_paved" and image_area is not None:
                if float(mask_bool.sum()) / image_area > float(args.parking_max_mask_area_ratio):
                    skipped_large += 1
                    continue
            score = float(scores[idx]) if idx < len(scores) else 0.0
            combined |= mask_bool
            score_map[mask_bool] = np.maximum(score_map[mask_bool], score)
            prompt_union |= mask_bool
            prompt_score_map[mask_bool] = np.maximum(prompt_score_map[mask_bool], score)
            kept += 1
        if support_count_map is not None and np.any(prompt_union):
            support_count_map[prompt_union] = np.clip(support_count_map[prompt_union] + 1, 0, 255)
        if class_name == "parking_paved":
            group_name = PARKING_PROMPT_GROUPS.get(prompt, "other")
            if group_name not in group_score_maps:
                group_score_maps[group_name] = np.zeros((sam.image_height, sam.image_width), dtype=np.float32)
            group_score_maps[group_name] = np.maximum(group_score_maps[group_name], prompt_score_map)
        prompt_records.append(
            {
                "prompt": prompt,
                "mask_count_raw": int(len(masks)),
                "mask_count_kept": int(kept),
                "mask_count_skipped_large": int(skipped_large),
                "scores_preview": [float(scores[i]) for i in range(min(len(scores), 5))],
            }
        )
    if combined is None:
        combined = np.zeros((sam.image_height, sam.image_width), dtype=bool)
        score_map = np.zeros((sam.image_height, sam.image_width), dtype=np.float32)
        support_count_map = np.zeros((sam.image_height, sam.image_width), dtype=np.uint8)
    combined = cleanup_class_mask(combined, class_name, args)
    score_map = np.where(combined, score_map, 0.0).astype(np.float32)
    extra_maps: dict[str, np.ndarray] = {
        "support_count_map": np.where(combined, support_count_map, 0).astype(np.uint8),
    }
    for group_name, group_score_map in group_score_maps.items():
        extra_maps[f"group_score_map__{group_name}"] = np.where(combined, group_score_map, 0.0).astype(np.float32)
    return combined, score_map, prompt_records, extra_maps


def refine_parking_mask(
    parking_mask: np.ndarray,
    parking_score_map: np.ndarray,
    parking_extra_maps: dict[str, np.ndarray],
    road_score_map: np.ndarray,
    building_score_map: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    support_count = parking_extra_maps.get("support_count_map", np.zeros_like(parking_score_map, dtype=np.uint8))
    lot_score = parking_extra_maps.get("group_score_map__lot", np.zeros_like(parking_score_map, dtype=np.float32))
    cars_score = parking_extra_maps.get("group_score_map__cars", np.zeros_like(parking_score_map, dtype=np.float32))
    markings_score = parking_extra_maps.get("group_score_map__markings", np.zeros_like(parking_score_map, dtype=np.float32))

    aux_support = (markings_score >= float(args.parking_aux_min_score)) | (cars_score >= float(args.parking_aux_min_score))
    lot_support = lot_score >= float(args.parking_lot_min_score)
    strong_lot_support = lot_score >= float(args.parking_lot_strong_score)

    consensus_keep = strong_lot_support | (lot_support & aux_support) | (cars_score >= float(args.parking_lot_min_score))
    road_dominates = road_score_map >= (parking_score_map + float(args.parking_road_suppress_margin))
    road_override = road_dominates & ~strong_lot_support

    refined_mask = parking_mask & consensus_keep & ~road_override
    thin_marking_mask = np.zeros_like(refined_mask, dtype=bool)
    context_suppress_mask = np.zeros_like(refined_mask, dtype=bool)
    labeled = label(refined_mask.astype(np.uint8), connectivity=2)
    for prop in regionprops(labeled):
        min_row, min_col, max_row, max_col = prop.bbox
        height = int(max_row - min_row)
        width = int(max_col - min_col)
        short_side = max(min(height, width), 1)
        long_side = max(height, width)
        aspect = float(long_side / short_side)
        if (
            short_side <= int(args.parking_thin_max_width)
            and aspect >= float(args.parking_thin_min_aspect)
            and int(prop.area) <= int(args.parking_thin_max_area)
        ):
            thin_marking_mask[labeled == prop.label] = True
            continue
        comp_mask = labeled == prop.label
        ring = binary_dilation(comp_mask, footprint=disk(int(args.parking_context_radius))) & ~comp_mask
        if np.any(ring):
            road_like = road_score_map[ring] >= max(float(args.road_min_merge_score) - 0.02, 0.30)
            building_like = building_score_map[ring] >= max(float(args.building_min_merge_score) - 0.04, 0.30)
            road_ratio = float(np.mean(road_like))
            building_ratio = float(np.mean(building_like))
            if road_ratio >= float(args.parking_context_max_road_ratio) and building_ratio <= float(args.parking_context_min_building_ratio):
                context_suppress_mask[comp_mask] = True
    refined_mask &= ~thin_marking_mask
    refined_mask &= ~context_suppress_mask
    refined_score_map = np.where(refined_mask, parking_score_map, 0.0).astype(np.float32)
    stats = {
        "pixels_before": int(np.sum(parking_mask)),
        "pixels_after": int(np.sum(refined_mask)),
        "pixels_removed_by_consensus": int(np.sum(parking_mask & ~consensus_keep)),
        "pixels_removed_by_road_override": int(np.sum(parking_mask & consensus_keep & road_override)),
        "pixels_removed_as_thin_markings": int(np.sum(thin_marking_mask)),
        "pixels_removed_by_context": int(np.sum(context_suppress_mask)),
    }
    return refined_mask.astype(bool), refined_score_map, stats


def merge_semantic_maps(
    class_score_maps: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    any_score_map = next(iter(class_score_maps.values()))
    semantic_map = np.full(any_score_map.shape, "other", dtype=object)
    best_score = np.zeros(any_score_map.shape, dtype=np.float32)
    class_score_bias = get_class_score_bias(args)
    class_min_merge_score = get_class_min_merge_score(args)
    for class_name in CLASS_PRIORITY:
        score_map = class_score_maps[class_name]
        effective_score = score_map + float(class_score_bias.get(class_name, 0.0))
        confident_mask = score_map >= float(class_min_merge_score.get(class_name, 0.0))
        update_mask = confident_mask & (effective_score > best_score)
        if class_name == "parking_paved":
            road_overlap = semantic_map == "road"
            near_tie = confident_mask.copy()
            near_tie &= score_map >= (best_score - float(args.parking_road_margin))
            update_mask |= road_overlap & near_tie
        elif class_name == "building_roof":
            road_overlap = semantic_map == "road"
            near_tie = confident_mask.copy()
            near_tie &= score_map >= (best_score - float(args.building_road_margin))
            update_mask |= road_overlap & near_tie
        semantic_map[update_mask] = class_name
        best_score[update_mask] = effective_score[update_mask]
        tie_mask = (effective_score == best_score) & confident_mask & (semantic_map == "other")
        semantic_map[tie_mask] = class_name
    return semantic_map, best_score


def render_overlay(image: np.ndarray, semantic_map: np.ndarray, out_path: Path) -> None:
    rgba = Image.fromarray(image).convert("RGBA")
    for class_name in CLASS_PRIORITY:
        color = ImageColor.getrgb(CLASS_PALETTE[class_name])
        mask = semantic_map == class_name
        if not np.any(mask):
            continue
        layer = np.zeros((semantic_map.shape[0], semantic_map.shape[1], 4), dtype=np.uint8)
        layer[mask] = (*color, 105)
        rgba = Image.alpha_composite(rgba, Image.fromarray(layer, mode="RGBA"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)


def render_flat_mask(semantic_map: np.ndarray, out_path: Path) -> None:
    canvas = np.zeros((semantic_map.shape[0], semantic_map.shape[1], 3), dtype=np.uint8)
    for class_name, color_hex in CLASS_PALETTE.items():
        mask = semantic_map == class_name
        if not np.any(mask):
            continue
        canvas[mask] = ImageColor.getrgb(color_hex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(out_path)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    add_local_sam3_repo(project_root)
    from samgeo import SamGeo3

    checkpoint_path = args.checkpoint_path or (project_root / "external" / "hf_models" / "facebook--sam3.1" / "sam3.1_multiplex.pt")
    bpe_path = args.bpe_path or (project_root / "external" / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
    sam_input_path, image_arr, preprocess_info = maybe_preprocess_image_for_sam(
        args.image_path,
        out_dir / "sam_input_preprocessed.png",
        args,
    )
    sam.set_image(str(sam_input_path))

    class_masks: dict[str, np.ndarray] = {}
    class_score_maps: dict[str, np.ndarray] = {}
    records: dict[str, list[dict[str, object]]] = {}
    extra_maps_by_class: dict[str, dict[str, np.ndarray]] = {}

    for class_name, prompts in CLASS_PROMPTS.items():
        class_mask, class_score_map, prompt_records, extra_maps = build_class_mask(
            sam,
            prompts,
            min_mask_size=args.min_mask_size,
            class_name=class_name,
            args=args,
        )
        class_masks[class_name] = class_mask
        class_score_maps[class_name] = class_score_map
        records[class_name] = prompt_records
        extra_maps_by_class[class_name] = extra_maps

        # Re-run first prompt for a clean class-specific overlay export.
        first_prompt = prompts[0]
        class_dir = out_dir / slugify(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        sam.generate_masks(first_prompt, quiet=True)
        if sam.masks is None or len(sam.masks) == 0:
            (class_dir / "no_masks.txt").write_text(f"No masks for class {class_name} with prompt: {first_prompt}\n", encoding="utf-8")
        else:
            sam.show_anns(
                output=str(class_dir / "overlay_clean.png"),
                show_bbox=False,
                show_score=False,
                blend=True,
                alpha=0.55,
            )
            sam.save_masks(output=str(class_dir / "instance_mask_unique.png"), unique=True, dtype="uint16")
            sam.save_masks(output=str(class_dir / "instance_mask_binary.png"), unique=False, dtype="uint8")

    parking_refine_stats = None
    if "parking_paved" in class_masks and "road" in class_score_maps:
        refined_mask, refined_score_map, parking_refine_stats = refine_parking_mask(
            parking_mask=class_masks["parking_paved"],
            parking_score_map=class_score_maps["parking_paved"],
            parking_extra_maps=extra_maps_by_class.get("parking_paved", {}),
            road_score_map=class_score_maps["road"],
            building_score_map=class_score_maps["building_roof"],
            args=args,
        )
        class_masks["parking_paved"] = refined_mask
        class_score_maps["parking_paved"] = refined_score_map

    semantic_map, best_score = merge_semantic_maps(class_score_maps=class_score_maps, args=args)
    class_score_bias = get_class_score_bias(args)
    class_min_merge_score = get_class_min_merge_score(args)

    render_overlay(image_arr, semantic_map, out_dir / "merged_semantic_overlay.png")
    render_flat_mask(semantic_map, out_dir / "merged_semantic_mask.png")

    summary = {
        "image_path": str(args.image_path),
        "sam_input_image_path": str(sam_input_path),
        "class_prompts": CLASS_PROMPTS,
        "class_priority": CLASS_PRIORITY,
        "min_mask_size": int(args.min_mask_size),
        "confidence_threshold": float(args.confidence_threshold),
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
        },
        "preprocess_runtime": preprocess_info,
        "component_cleanup": {
            "road_min_component": int(args.road_min_component),
            "parking_min_component": int(args.parking_min_component),
            "building_min_component": int(args.building_min_component),
            "vegetation_min_component": int(args.vegetation_min_component),
        },
        "merge_settings": {
            "class_score_bias": class_score_bias,
            "class_min_merge_score": class_min_merge_score,
            "parking_road_margin": float(args.parking_road_margin),
            "parking_max_mask_area_ratio": float(args.parking_max_mask_area_ratio),
            "parking_min_support_prompts": int(args.parking_min_support_prompts),
            "parking_lot_min_score": float(args.parking_lot_min_score),
            "parking_lot_strong_score": float(args.parking_lot_strong_score),
            "parking_aux_min_score": float(args.parking_aux_min_score),
            "parking_road_suppress_margin": float(args.parking_road_suppress_margin),
            "parking_thin_max_width": int(args.parking_thin_max_width),
            "parking_thin_min_aspect": float(args.parking_thin_min_aspect),
            "parking_thin_max_area": int(args.parking_thin_max_area),
            "building_road_margin": float(args.building_road_margin),
            "road_hole_area": int(args.road_hole_area),
            "parking_hole_area": int(args.parking_hole_area),
            "building_hole_area": int(args.building_hole_area),
            "vegetation_hole_area": int(args.vegetation_hole_area),
            "road_closing_radius": int(args.road_closing_radius),
            "parking_closing_radius": int(args.parking_closing_radius),
            "building_closing_radius": int(args.building_closing_radius),
        },
        "class_pixel_counts": {
            class_name: int(np.sum(semantic_map == class_name))
            for class_name in list(CLASS_PRIORITY) + ["other"]
        },
        "class_score_max": {
            class_name: float(class_score_maps[class_name].max()) if class_name in class_score_maps else 0.0
            for class_name in CLASS_PRIORITY
        },
        "prompt_records": records,
    }
    if parking_refine_stats is not None:
        summary["parking_refine_stats"] = parking_refine_stats
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
