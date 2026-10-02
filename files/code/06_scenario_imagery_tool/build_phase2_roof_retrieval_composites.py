from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


METHODS = ("alpha", "alpha_shadow", "poisson_shadow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval-based roof composites for Phase 2 CAD scenarios.")
    parser.add_argument("--dataset-index-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--gallery-count", type=int, default=12)
    return parser.parse_args()


@dataclass
class Sample:
    sample_id: str
    development_key: str
    source: np.ndarray
    target: np.ndarray
    footprint: np.ndarray
    roi: np.ndarray
    condition: np.ndarray
    descriptor: np.ndarray
    event_score: float


def load_sample(row: pd.Series) -> Sample:
    with np.load(str(row.sample_npz)) as payload:
        source = payload["source_rgb"].astype(np.uint8)
        target = payload["target_rgb"].astype(np.uint8)
        footprint = payload["footprint_mask"].astype(bool)
        roi = payload["roi_mask"].astype(bool)
        condition = payload["condition"].astype(np.float32)
    return Sample(
        sample_id=str(row.sample_id),
        development_key=str(row.development_key),
        source=source,
        target=target,
        footprint=footprint,
        roi=roi,
        condition=condition,
        descriptor=shape_condition_descriptor(footprint, condition),
        event_score=float(row.get("event_score", 0.0)),
    )


def principal_frame(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack(np.where(mask))[:, ::-1].astype(np.float32)
    if len(points) < 3:
        raise ValueError("Footprint has fewer than three pixels.")
    centre = points.mean(axis=0)
    centred = points - centre
    covariance = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major = vectors[:, order[0]].astype(np.float32)
    minor = vectors[:, order[1]].astype(np.float32)
    if major[0] < 0:
        major = -major
    if np.linalg.det(np.stack([major, minor], axis=1)) < 0:
        minor = -minor
    projected = centred @ np.stack([major, minor], axis=1)
    extents = np.maximum(np.percentile(np.abs(projected), 95, axis=0), 1.0).astype(np.float32)
    return centre, major, minor, extents


def shape_condition_descriptor(mask: np.ndarray, condition: np.ndarray) -> np.ndarray:
    centre, major, minor, extents = principal_frame(mask)
    del centre, major, minor
    area = float(mask.sum())
    bbox_area = float(max(4.0 * extents[0] * extents[1], 1.0))
    aspect = float(max(extents) / max(min(extents), 1.0))
    intensity = [float(np.max(channel)) for channel in condition[1:5]]
    return np.asarray(
        [np.log1p(area), np.log1p(aspect), np.clip(area / bbox_area, 0.0, 1.5), *intensity],
        dtype=np.float32,
    )


def affine_between_masks(source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    source_centre, source_major, source_minor, source_extents = principal_frame(source_mask)
    target_centre, target_major, target_minor, target_extents = principal_frame(target_mask)
    source_points = np.float32(
        [
            source_centre,
            source_centre + source_major * source_extents[0],
            source_centre + source_minor * source_extents[1],
        ]
    )
    target_points = np.float32(
        [
            target_centre,
            target_centre + target_major * target_extents[0],
            target_centre + target_minor * target_extents[1],
        ]
    )
    return cv2.getAffineTransform(source_points, target_points)


def colour_match(texture: np.ndarray, source: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    ring = ndimage.binary_dilation(footprint, iterations=12) & ~ndimage.binary_dilation(footprint, iterations=3)
    if ring.sum() < 32:
        return texture
    output = texture.astype(np.float32)
    source_f = source.astype(np.float32)
    for channel in range(3):
        roof_values = output[..., channel][footprint]
        context_values = source_f[..., channel][ring]
        if roof_values.size == 0 or context_values.size == 0:
            continue
        roof_mean = float(roof_values.mean())
        roof_std = max(float(roof_values.std()), 8.0)
        context_mean = float(context_values.mean())
        context_std = max(float(context_values.std()), 8.0)
        matched = (output[..., channel] - roof_mean) * np.clip(context_std / roof_std, 0.65, 1.35)
        matched += 0.65 * roof_mean + 0.35 * context_mean
        output[..., channel] = matched
    return np.clip(output, 0, 255).astype(np.uint8)


def feather_alpha(mask: np.ndarray, width: float = 2.5) -> np.ndarray:
    inside = ndimage.distance_transform_edt(mask)
    return np.clip(inside / width, 0.0, 1.0).astype(np.float32)


def add_height_shadow(image: np.ndarray, footprint: np.ndarray, height_norm: float) -> np.ndarray:
    shift = int(round(2 + 11 * np.clip(height_norm, 0.0, 1.0)))
    shifted = ndimage.shift(footprint.astype(np.float32), shift=(0.65 * shift, 0.85 * shift), order=0) > 0.5
    shadow = ndimage.binary_dilation(shifted & ~footprint, iterations=2)
    alpha = ndimage.gaussian_filter(shadow.astype(np.float32), sigma=1.5)[..., None] * 0.32
    result = image.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def render_composite(sample: Sample, donor: Sample, method: str) -> np.ndarray:
    height, width = sample.footprint.shape
    transform = affine_between_masks(donor.footprint, sample.footprint)
    warped = cv2.warpAffine(
        donor.target,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped = colour_match(warped, sample.source, sample.footprint)
    base = sample.source.copy()
    if "shadow" in method:
        base = add_height_shadow(base, sample.footprint, float(np.max(sample.condition[1])))

    if method.startswith("poisson"):
        mask = sample.footprint.astype(np.uint8) * 255
        ys, xs = np.where(sample.footprint)
        centre = (int(round((xs.min() + xs.max()) / 2)), int(round((ys.min() + ys.max()) / 2)))
        try:
            return cv2.seamlessClone(warped, base, mask, centre, cv2.NORMAL_CLONE)
        except cv2.error:
            pass

    alpha = feather_alpha(sample.footprint)[..., None]
    composite = base.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
    return np.clip(composite, 0, 255).astype(np.uint8)


def standardized_distances(query: np.ndarray, donors: list[Sample]) -> np.ndarray:
    matrix = np.stack([donor.descriptor for donor in donors])
    scale = np.maximum(matrix.std(axis=0), 0.05)
    distances = np.sqrt(np.square((matrix - query[None]) / scale[None]).mean(axis=1))
    quality = np.asarray([donor.event_score for donor in donors], dtype=np.float32)
    quality_range = float(np.ptp(quality))
    if quality_range > 1e-8:
        quality = (quality - quality.min()) / quality_range
        distances -= 0.08 * quality
    return distances


def save_gallery(records: list[dict[str, object]], path: Path, count: int) -> None:
    if not records:
        return
    selected = records[: min(count, len(records))]
    fig, axes = plt.subplots(len(selected), 5, figsize=(15, 3 * len(selected)), squeeze=False)
    for row_index, record in enumerate(selected):
        panels = [
            record["source"],
            record["donor"],
            record["alpha"],
            record["poisson_shadow"],
            record["target"],
        ]
        titles = ["Source", f"Donor: {record['donor_key']}", "Alpha", "Poisson + shadow", "Target"]
        for column, (panel, title) in enumerate(zip(panels, titles)):
            axes[row_index, column].imshow(panel)
            axes[row_index, column].set_title(f"{record['sample_id']} | {title}", fontsize=8)
            axes[row_index, column].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset_index_csv)
    train_frame = frame[frame["dataset_split"].eq("train")].copy()
    test_frame = frame[frame["dataset_split"].eq("test")].copy()
    if train_frame.empty or test_frame.empty:
        raise RuntimeError("Retrieval compositing requires non-empty train and test projects.")

    donors = [load_sample(row) for _, row in train_frame.iterrows()]
    tests = [load_sample(row) for _, row in test_frame.iterrows()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        (args.out_dir / "methods" / method / "test_predictions").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    gallery_records: list[dict[str, object]] = []
    for sample in tests:
        distances = standardized_distances(sample.descriptor, donors)
        candidate_indices = np.argsort(distances)[: max(1, min(args.top_k, len(donors)))]
        donor_index = min(
            candidate_indices,
            key=lambda index: (float(distances[index]), -float(donors[index].event_score)),
        )
        donor = donors[int(donor_index)]
        generated: dict[str, np.ndarray] = {}
        for method in METHODS:
            image = render_composite(sample, donor, method)
            output_path = args.out_dir / "methods" / method / "test_predictions" / f"{sample.sample_id}__generated.png"
            Image.fromarray(image).save(output_path)
            generated[method] = image
        rows.append(
            {
                "sample_id": sample.sample_id,
                "development_key": sample.development_key,
                "donor_sample_id": donor.sample_id,
                "donor_development_key": donor.development_key,
                "retrieval_distance": float(distances[donor_index]),
                "query_height_norm": float(sample.descriptor[3]),
                "donor_height_norm": float(donor.descriptor[3]),
                "query_landuse_signature": json.dumps(sample.descriptor[4:].round(5).tolist()),
                "donor_landuse_signature": json.dumps(donor.descriptor[4:].round(5).tolist()),
            }
        )
        gallery_records.append(
            {
                "sample_id": sample.sample_id,
                "donor_key": donor.development_key,
                "source": sample.source,
                "donor": donor.target,
                "alpha": generated["alpha"],
                "poisson_shadow": generated["poisson_shadow"],
                "target": sample.target,
            }
        )

    pd.DataFrame(rows).to_csv(args.out_dir / "retrieval_matches.csv", index=False)
    save_gallery(gallery_records, args.out_dir / "retrieval_composite_gallery.jpg", args.gallery_count)
    summary = {
        "donor_split": "train only",
        "donor_count": len(donors),
        "test_project_count": len(tests),
        "methods": list(METHODS),
        "top_k": args.top_k,
        "information_leakage_prevented": True,
        "output_contract": "Each method is exposed as a run-like test_predictions directory for the unchanged SamGeo3 evaluator.",
    }
    (args.out_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
