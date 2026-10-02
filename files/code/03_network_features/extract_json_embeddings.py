from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from hf_cache_utils import local_files_only_default, resolve_hf_snapshot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hidden-state embeddings from structured JSON files.")
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--feature-prefix", type=str, required=True)
    parser.add_argument("--hidden-layer", type=int, default=-2)
    parser.add_argument("--pooling", choices=["mean", "last_token", "cls"], default="mean")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="float16")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--train-years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_idx, lengths]


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return mean_pool(hidden, attention_mask)
    if pooling == "last_token":
        return last_token_pool(hidden, attention_mask)
    return hidden[:, 0]


def suggested_dtype(model_id: str, requested_dtype: str) -> str:
    if requested_dtype != "float16":
        return requested_dtype
    model_name = model_id.lower()
    if "embedding" in model_name:
        return "float32"
    return requested_dtype


def resolve_json_payload_path(json_root: Path, json_path_value: object) -> Path:
    raw = str(json_path_value).strip()
    normalized = raw.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    return json_root / candidate


def should_trust_remote_code(model_id: str, explicit_flag: bool) -> bool:
    if explicit_flag:
        return True
    model_name = model_id.lower()
    return "alibaba-nlp/gte-" in model_name


def main() -> None:
    args = parse_args()
    index_df = pd.read_csv(args.index_csv)
    index_df["SITE_NO"] = pd.to_numeric(index_df["SITE_NO"], errors="coerce").astype(int)
    index_df["year"] = pd.to_numeric(index_df["year"], errors="coerce").astype(int)
    index_df = index_df.sort_values(["year", "SITE_NO"]).reset_index(drop=True)
    if args.limit_rows is not None:
        index_df = index_df.head(args.limit_rows).copy()

    texts: list[str] = []
    for record in index_df.to_dict(orient="records"):
        payload_path = resolve_json_payload_path(args.json_root, record["json_path"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        texts.append(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))

    resolved_model_id = resolve_hf_snapshot_path(args.model_id)
    local_files_only = args.local_files_only or local_files_only_default(args.model_id)
    trust_remote_code = should_trust_remote_code(args.model_id, args.trust_remote_code)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    effective_dtype = suggested_dtype(args.model_id, args.dtype)
    dtype = torch.float16 if effective_dtype == "float16" and device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        resolved_model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype if effective_dtype != "auto" else "auto",
    )
    model.to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    token_lengths: list[int] = []
    truncated_count = 0
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            batch_lengths = encoded["attention_mask"].sum(dim=1).detach().cpu().numpy().astype(int).tolist()
            token_lengths.extend(batch_lengths)
            truncated_count += int(sum(length >= args.max_length for length in batch_lengths))
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[args.hidden_layer]
            pooled = pool_hidden(hidden, encoded["attention_mask"], args.pooling)
            vectors.append(pooled.float().detach().cpu().numpy())

    raw = np.concatenate(vectors, axis=0).astype(np.float32)
    nonfinite_mask = ~np.isfinite(raw)
    nonfinite_count = int(nonfinite_mask.sum())
    if nonfinite_count:
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e4, neginf=-1e4)
    train_mask = index_df["year"].isin(args.train_years).to_numpy()
    n_components = min(args.pca_components, raw.shape[1], max(int(train_mask.sum()) - 1, 1))
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    raw_train_scaled = scaler.fit_transform(raw[train_mask]) if bool(train_mask.any()) else scaler.fit_transform(raw)
    pca.fit(raw_train_scaled)
    reduced = pca.transform(scaler.transform(raw)).astype(np.float32)

    out_df = index_df[["SITE_NO", "year"]].copy()
    for idx in range(reduced.shape[1]):
        out_df[f"{args.feature_prefix}_{idx:03d}"] = reduced[:, idx]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    summary = {
        "json_root": str(args.json_root),
        "index_csv": str(args.index_csv),
        "model_id": args.model_id,
        "resolved_model_id": resolved_model_id,
        "row_count": int(len(out_df)),
        "feature_prefix": args.feature_prefix,
        "raw_embedding_dim": int(raw.shape[1]),
        "pca_components": int(reduced.shape[1]),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "hidden_layer": int(args.hidden_layer),
        "pooling": args.pooling,
        "requested_dtype": args.dtype,
        "effective_dtype": effective_dtype,
        "mean_token_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
        "max_token_length": int(np.max(token_lengths)) if token_lengths else 0,
        "truncated_count": int(truncated_count),
        "truncated_pct": float(truncated_count / len(token_lengths)) if token_lengths else 0.0,
        "nonfinite_count": nonfinite_count,
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
    }
    args.out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.out_csv)


if __name__ == "__main__":
    main()
