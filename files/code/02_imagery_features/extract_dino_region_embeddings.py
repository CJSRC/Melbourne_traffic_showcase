from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel

from hf_cache_utils import local_files_only_default, rebase_output_root, resolve_hf_snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINO region embeddings from SAM token crops.")
    parser.add_argument("--region-token-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, default=None)
    parser.add_argument("--year-root-catalog-csv", type=Path, default=None)
    parser.add_argument("--root-search-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--dino-model-id", type=str, default="facebook/dinov2-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--train-years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--mask-background", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def should_trust_remote_code(model_id: str, resolved_model_id: str, explicit_flag: bool) -> bool:
    if explicit_flag:
        return True
    model_tokens = (model_id + " " + resolved_model_id).lower()
    return "radio" in model_tokens


def load_local_radio_model(resolved_model_id: str, local_files_only: bool) -> tuple[AutoImageProcessor, torch.nn.Module]:
    model_dir = Path(resolved_model_id).resolve()
    package_name = f"_radio_local_{abs(hash(str(model_dir)))}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    module_name = f"{package_name}.hf_model"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, model_dir / "hf_model.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load RADIO hf_model.py from {model_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=local_files_only)
    config = module.RADIOConfig.from_pretrained(str(model_dir), local_files_only=local_files_only)
    model = module.RADIOModel.from_pretrained(str(model_dir), config=config, local_files_only=local_files_only)
    return processor, model


def build_year_root_map(args: argparse.Namespace) -> dict[int, Path]:
    root_map: dict[int, Path] = {}
    if args.patch_root is not None or args.year_root_catalog_csv is None:
        return root_map
    catalog = pd.read_csv(args.year_root_catalog_csv)
    year_col = "target_year" if "target_year" in catalog.columns else "year"
    for row in catalog.itertuples(index=False):
        year = int(getattr(row, year_col))
        if hasattr(row, "patch_root") and getattr(row, "patch_root"):
            root = Path(getattr(row, "patch_root"))
        elif hasattr(row, "embedding_csv") and getattr(row, "embedding_csv"):
            root = Path(getattr(row, "embedding_csv")).resolve().parents[1]
        else:
            continue
        root_map[year] = rebase_output_root(str(root), args.root_search_dir)
    return root_map


def resolve_image_path(row: pd.Series, patch_root: Path | None, year_root_map: dict[int, Path]) -> Path:
    if "resolved_image_path" in row and isinstance(row["resolved_image_path"], str):
        path = Path(row["resolved_image_path"])
        if path.exists():
            return path
    rel = Path(str(row["relative_image_path"]))
    if rel.is_absolute() and rel.exists():
        return rel
    if patch_root is not None:
        return patch_root / rel
    year = int(row["year"])
    if year in year_root_map:
        return year_root_map[year] / rel
    raise FileNotFoundError(f"Could not resolve image path for row {row.to_dict()}")


def crop_region(row: pd.Series, image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).copy()
    x0 = int(row["bbox_x0"])
    y0 = int(row["bbox_y0"])
    x1 = int(row["bbox_x1"])
    y1 = int(row["bbox_y1"])
    if "mask_path" in row and Path(str(row["mask_path"])).exists():
        mask_path = Path(str(row["mask_path"]))
    else:
        mask_path = Path(row["mask_relative_path"])
        if not mask_path.is_absolute():
            mask_path = Path(row["_token_root"]) / mask_path
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    crop = arr[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    if bool(row.get("_mask_background", False)):
        crop[~crop_mask] = 0
    return Image.fromarray(crop)


def encode_image_batch(
    processor: AutoImageProcessor,
    images: list[Image.Image],
    *,
    is_radio_model: bool,
    device: torch.device,
):
    processor_kwargs = {"images": images, "return_tensors": "pt"}
    if is_radio_model:
        processor_kwargs["do_resize"] = True
        processor_kwargs["size"] = {"height": 512, "width": 512}
    return processor(**processor_kwargs).to(device)


def forward_image_batch(model: torch.nn.Module, encoded, *, is_radio_model: bool):
    if is_radio_model:
        return model(encoded["pixel_values"])
    return model(**encoded)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    token_df = pd.read_csv(args.region_token_csv)
    if args.limit_rows is not None:
        token_df = token_df.head(args.limit_rows).copy()
    token_df["_token_root"] = str(args.region_token_csv.parent)
    token_df["_mask_background"] = bool(args.mask_background)
    token_df["SITE_NO"] = pd.to_numeric(token_df["SITE_NO"], errors="coerce").astype(int)
    token_df["year"] = pd.to_numeric(token_df["year"], errors="coerce").astype(int)
    year_root_map = build_year_root_map(args)

    resolved_model_id = resolve_hf_snapshot_path(args.dino_model_id)
    local_files_only = args.local_files_only or local_files_only_default(args.dino_model_id)
    trust_remote_code = should_trust_remote_code(args.dino_model_id, resolved_model_id, args.trust_remote_code)
    is_radio_model = "radio" in (args.dino_model_id + " " + resolved_model_id).lower()
    if trust_remote_code:
        hf_modules_cache = out_dir / ".hf_modules_cache"
        hf_modules_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = str(hf_modules_cache.resolve())
    if is_radio_model and Path(resolved_model_id).exists():
        processor, model = load_local_radio_model(resolved_model_id, local_files_only=local_files_only)
    else:
        processor = AutoImageProcessor.from_pretrained(
            resolved_model_id,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            resolved_model_id,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    images: list[Image.Image] = []
    raw_vectors: list[np.ndarray] = []
    for record in token_df.to_dict(orient="records"):
        row = pd.Series(record)
        image_path = resolve_image_path(row, args.patch_root, year_root_map)
        images.append(crop_region(row, image_path))
        if len(images) >= args.batch_size:
            encoded = encode_image_batch(processor, images, is_radio_model=is_radio_model, device=device)
            with torch.inference_mode():
                outputs = forward_image_batch(model, encoded, is_radio_model=is_radio_model)
            if isinstance(outputs, (tuple, list)):
                pooled = outputs[0]
            else:
                pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]
            raw_vectors.append(pooled.float().detach().cpu().numpy())
            images = []
    if images:
        encoded = encode_image_batch(processor, images, is_radio_model=is_radio_model, device=device)
        with torch.inference_mode():
            outputs = forward_image_batch(model, encoded, is_radio_model=is_radio_model)
        if isinstance(outputs, (tuple, list)):
            pooled = outputs[0]
        else:
            pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]
        raw_vectors.append(pooled.float().detach().cpu().numpy())

    raw = np.concatenate(raw_vectors, axis=0).astype(np.float32)
    np.save(out_dir / "raw_region_embeddings.npy", raw)
    train_mask = token_df["year"].isin(args.train_years).to_numpy()
    n_components = min(args.pca_components, raw.shape[1], max(int(train_mask.sum()) - 1, 1))
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    raw_train_scaled = scaler.fit_transform(raw[train_mask]) if bool(train_mask.any()) else scaler.fit_transform(raw)
    pca.fit(raw_train_scaled)
    reduced = pca.transform(scaler.transform(raw)).astype(np.float32)

    out_df = token_df.drop(columns=["_token_root", "_mask_background"]).copy()
    for idx in range(reduced.shape[1]):
        out_df[f"dino_emb_{idx:03d}"] = reduced[:, idx]
    out_csv = out_dir / "region_dino_embeddings.csv"
    out_df.to_csv(out_csv, index=False)
    summary = {
        "region_token_csv": str(args.region_token_csv),
        "row_count": int(len(out_df)),
        "site_year_count": int(out_df[["SITE_NO", "year"]].drop_duplicates().shape[0]),
        "raw_dim": int(raw.shape[1]),
        "pca_components": int(reduced.shape[1]),
        "train_years": args.train_years,
        "dino_model_id": args.dino_model_id,
        "resolved_model_id": resolved_model_id,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
        "is_radio_model": bool(is_radio_model),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    (out_dir / "region_dino_embeddings_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_csv)


if __name__ == "__main__":
    main()
