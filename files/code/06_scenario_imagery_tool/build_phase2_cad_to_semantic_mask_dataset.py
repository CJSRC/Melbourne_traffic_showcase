from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REMOTE_REPO_PREFIX = (
    "/data/gpfs/projects/punim2970/xhe13561/"
    "melbourne_scats_spartan_wayback_screened_v3_v12c/melbourne_scats_baseline"
)

CLASS_NAMES = ["background", "vegetation", "road", "parking_paved", "building_roof"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_COLORS = {
    0: (28, 28, 28),
    1: (70, 165, 75),
    2: (205, 78, 62),
    3: (236, 184, 72),
    4: (88, 138, 224),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase2 pseudo-CAD/planning raster -> SAM semantic mask dataset."
    )
    parser.add_argument("--quality-subset-csv", type=Path, required=True)
    parser.add_argument("--sam-region-tokens-csv", type=Path, required=True)
    parser.add_argument("--tokens-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mask-size", type=int, default=256)
    parser.add_argument("--local-repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--remote-repo-prefix", default=REMOTE_REPO_PREFIX)
    parser.add_argument("--max-gallery", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def localize_remote_path(raw: object, remote_prefix: str, local_repo_root: Path) -> str:
    if raw is None or pd.isna(raw):
        return ""
    text = str(raw).replace("\\", "/")
    prefix = remote_prefix.replace("\\", "/")
    if text.startswith(prefix):
        rel = text[len(prefix) :].lstrip("/")
        return str(local_repo_root / rel)
    return str(raw)


def read_mask(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.NEAREST)
    return np.asarray(image) > 0


def write_label_map(label_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_map.astype(np.uint8), mode="L").save(path)


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        out[label_map == class_id] = color
    return out


def load_rgb_if_exists(path: str, size: int) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    image = Image.open(p).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image)


def load_cad_preview_from_raster(path: str, size: int) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim != 3:
        return None
    chw = arr.astype(np.float32) if arr.shape[0] < arr.shape[-1] else np.moveaxis(arr, -1, 0).astype(np.float32)
    if chw.max() > 1.0:
        chw = chw / 255.0
    road = chw[0] if chw.shape[0] > 0 else np.zeros_like(chw[0])
    existing = chw[1] if chw.shape[0] > 1 else np.zeros_like(road)
    height = chw[2] if chw.shape[0] > 2 else np.zeros_like(road)
    proposed = chw[3] if chw.shape[0] > 3 else np.zeros_like(road)
    rgb = np.stack([np.maximum(proposed, height), existing, road], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    if rgb.shape[:2] != (size, size):
        image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
        image = image.resize((size, size), Image.BILINEAR)
        rgb = np.asarray(image).astype(np.float32) / 255.0
    return (rgb * 255).astype(np.uint8)


def overlay_semantic(rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    semantic = colorize_label_map(label_map).astype(np.float32) / 255.0
    base = rgb.astype(np.float32) / 255.0
    mask = label_map > 0
    out = base.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * semantic[mask]
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def save_gallery(df: pd.DataFrame, out_path: Path, mask_size: int, max_gallery: int, seed: int) -> None:
    if df.empty:
        return
    rng = np.random.default_rng(seed)
    candidates = df.sort_values(["foreground_area_ratio", "quality_score"], ascending=False)
    top = candidates.head(max_gallery // 2)
    rest = candidates.iloc[max_gallery // 2 :]
    if len(rest) > 0:
        rand_idx = rng.choice(rest.index.to_numpy(), size=min(max_gallery - len(top), len(rest)), replace=False)
        selected = pd.concat([top, rest.loc[rand_idx]], ignore_index=True)
    else:
        selected = top.reset_index(drop=True)

    cols = 4
    fig, axes = plt.subplots(len(selected), cols, figsize=(cols * 3.6, max(1, len(selected)) * 3.1))
    if len(selected) == 1:
        axes = np.asarray([axes])
    blank = np.full((mask_size, mask_size, 3), 245, dtype=np.uint8)
    for i, row in enumerate(selected.itertuples(index=False)):
        sat = load_rgb_if_exists(getattr(row, "local_satellite_path"), mask_size)
        cad = load_cad_preview_from_raster(getattr(row, "local_building_raster_path"), mask_size)
        label_map = np.asarray(Image.open(getattr(row, "target_semantic_mask_png")).convert("L"), dtype=np.uint8)
        sat_view = sat if sat is not None else blank
        cad_view = cad if cad is not None else blank
        panels = [
            sat_view,
            cad_view,
            colorize_label_map(label_map),
            overlay_semantic(sat_view, label_map) if sat is not None else colorize_label_map(label_map),
        ]
        titles = [
            f"{int(getattr(row, 'SITE_NO'))} | {int(getattr(row, 'year'))}",
            "pseudo-CAD",
            "SAM semantic target",
            "target overlay",
        ]
        for j, (panel, title) in enumerate(zip(panels, titles)):
            axes[i, j].imshow(panel)
            axes[i, j].set_title(title, fontsize=8)
            axes[i, j].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_dir = args.out_dir / "target_semantic_masks"

    quality = pd.read_csv(args.quality_subset_csv)
    tokens = pd.read_csv(args.sam_region_tokens_csv)
    for frame in [quality, tokens]:
        frame["SITE_NO"] = frame["SITE_NO"].astype(int)
        frame["year"] = frame["year"].astype(int)

    tokens = tokens[tokens["semantic_label"].astype(str).isin(CLASS_TO_ID)].copy()
    if "selection_score" not in tokens.columns:
        tokens["selection_score"] = tokens["semantic_confidence"].astype(float)
    grouped = {key: group.copy() for key, group in tokens.groupby(["SITE_NO", "year"], sort=False)}

    records: list[dict[str, object]] = []
    missing_region_masks = 0
    for row in quality.itertuples(index=False):
        site = int(getattr(row, "SITE_NO"))
        year = int(getattr(row, "year"))
        label_map = np.zeros((args.mask_size, args.mask_size), dtype=np.uint8)
        score_map = np.full((args.mask_size, args.mask_size), -np.inf, dtype=np.float32)
        region_counts = {name: 0 for name in CLASS_NAMES}

        group = grouped.get((site, year))
        if group is not None:
            group = group.sort_values(["selection_score", "semantic_confidence"], ascending=True)
            for token in group.itertuples(index=False):
                label = str(getattr(token, "semantic_label"))
                class_id = CLASS_TO_ID[label]
                rel = Path(str(getattr(token, "mask_relative_path")).replace("\\", "/"))
                local_mask = args.tokens_root / rel
                if not local_mask.exists():
                    missing_region_masks += 1
                    continue
                mask = read_mask(local_mask, args.mask_size)
                score = float(getattr(token, "selection_score"))
                update = mask & (score >= score_map)
                label_map[update] = class_id
                score_map[update] = score
                region_counts[label] += 1

        out_mask = target_dir / f"year_{year}" / f"site_{site}__year_{year}__ps500m__z18__semantic.png"
        write_label_map(label_map, out_mask)

        remote_sat = str(getattr(row, "resolved_satellite_path"))
        remote_raster = str(getattr(row, "resolved_building_raster_path"))
        local_sat = localize_remote_path(remote_sat, args.remote_repo_prefix, args.local_repo_root)
        local_raster = localize_remote_path(remote_raster, args.remote_repo_prefix, args.local_repo_root)

        rec: dict[str, object] = {
            "SITE_NO": site,
            "year": year,
            "dataset_split": str(getattr(row, "dataset_split")),
            "quality_tier": getattr(row, "quality_tier"),
            "quality_score": float(getattr(row, "quality_score")),
            "remote_satellite_path": remote_sat,
            "local_satellite_path": local_sat,
            "remote_building_raster_path": remote_raster,
            "local_building_raster_path": local_raster,
            "target_semantic_mask_png": str(out_mask),
            "foreground_area_ratio": float((label_map > 0).mean()),
        }
        for name, class_id in CLASS_TO_ID.items():
            rec[f"target_{name}_area_ratio"] = float((label_map == class_id).mean())
            rec[f"target_{name}_pixel_count"] = int((label_map == class_id).sum())
            rec[f"target_{name}_region_count"] = int(region_counts[name])
        records.append(rec)

    index = pd.DataFrame(records)
    index_path = args.out_dir / "phase2_cad_to_semantic_mask_dataset_index.csv"
    index.to_csv(index_path, index=False)

    class_summary = []
    total_pixels = float(len(index) * args.mask_size * args.mask_size)
    for name in CLASS_NAMES:
        pixels = int(index[f"target_{name}_pixel_count"].sum())
        class_summary.append(
            {
                "class_name": name,
                "class_id": CLASS_TO_ID[name],
                "pixel_count": pixels,
                "pixel_ratio": float(pixels / total_pixels) if total_pixels else 0.0,
                "rows_with_class": int((index[f"target_{name}_pixel_count"] > 0).sum()),
                "region_count": int(index[f"target_{name}_region_count"].sum()),
            }
        )
    pd.DataFrame(class_summary).to_csv(args.out_dir / "semantic_class_summary.csv", index=False)

    split_summary = (
        index.groupby("dataset_split", dropna=False)
        .agg(
            sample_count=("SITE_NO", "size"),
            site_count=("SITE_NO", "nunique"),
            foreground_area_mean=("foreground_area_ratio", "mean"),
            building_area_mean=("target_building_roof_area_ratio", "mean"),
            road_area_mean=("target_road_area_ratio", "mean"),
            vegetation_area_mean=("target_vegetation_area_ratio", "mean"),
            parking_area_mean=("target_parking_paved_area_ratio", "mean"),
        )
        .reset_index()
    )
    split_summary.to_csv(args.out_dir / "semantic_split_summary.csv", index=False)

    summary = {
        "quality_subset_csv": str(args.quality_subset_csv),
        "sam_region_tokens_csv": str(args.sam_region_tokens_csv),
        "tokens_root": str(args.tokens_root),
        "sample_count": int(len(index)),
        "site_count": int(index["SITE_NO"].nunique()),
        "year_count": int(index["year"].nunique()),
        "years": [int(x) for x in sorted(index["year"].unique())],
        "mask_size": args.mask_size,
        "class_names": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "missing_region_masks": int(missing_region_masks),
        "index_csv": str(index_path),
    }
    (args.out_dir / "semantic_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "semantic_class_map.json").write_text(
        json.dumps({"class_names": CLASS_NAMES, "class_to_id": CLASS_TO_ID, "class_colors": CLASS_COLORS}, indent=2),
        encoding="utf-8",
    )
    save_gallery(index, args.out_dir / "semantic_dataset_gallery.jpg", args.mask_size, args.max_gallery, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
