from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pipeline import (
    SIGNALS_CSV_URL,
    VOLUME_JAN_2026_URL,
    build_generic_matrix,
    download_file,
    normalize_adjacency,
    read_signals_csv,
    read_volume_zip,
    reshape_volume_long,
    aggregate_site_timeseries,
    summarize_site_activity,
    run_training,
    sanitize_slug,
    write_json,
    haversine_m,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Area-level Melbourne SCATS site baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--study-area-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--signals-url", type=str, default=SIGNALS_CSV_URL)
    parser.add_argument("--volume-url", type=str, default=VOLUME_JAN_2026_URL)
    parser.add_argument("--variant", type=str, default="dcrnn_like")
    parser.add_argument("--node-feature-csv", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-nearest", type=int, default=4)
    parser.add_argument("--max-edge-m", type=float, default=1200.0)
    return parser.parse_args()


def load_or_build_interim(data_dir: Path, signals_url: str, volume_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    signals_path = download_file(signals_url, raw_dir / "victorian_traffic_signals.csv")
    _ = read_signals_csv(signals_path)
    volume_path = download_file(volume_url, raw_dir / "traffic_signal_volume_data.zip")

    long_path = interim_dir / "volume_long.parquet"
    site_ts_path = interim_dir / "site_timeseries.parquet"
    site_summary_path = interim_dir / "site_summary.csv"

    if site_ts_path.exists() and site_summary_path.exists():
        site_ts = pd.read_parquet(site_ts_path)
        site_summary = pd.read_csv(site_summary_path)
        if not pd.api.types.is_datetime64_any_dtype(site_ts["timestamp_end"]):
            site_ts["timestamp_end"] = pd.to_datetime(site_ts["timestamp_end"])
        return site_ts, site_summary

    volume_df = read_volume_zip(volume_path)
    long_df = reshape_volume_long(volume_df)
    site_ts = aggregate_site_timeseries(long_df)
    site_summary = summarize_site_activity(site_ts)
    long_df.to_parquet(long_path, index=False)
    site_ts.to_parquet(site_ts_path, index=False)
    site_summary.to_csv(site_summary_path, index=False)
    return site_ts, site_summary


def build_area_graph(nodes: pd.DataFrame, k_nearest: int, max_edge_m: float) -> np.ndarray:
    nodes = nodes.reset_index(drop=True)
    n = len(nodes)
    adj = np.zeros((n, n), dtype=float)
    distances = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m(
                float(nodes.loc[i, "LATITUDE"]),
                float(nodes.loc[i, "LONGITUDE"]),
                float(nodes.loc[j, "LATITUDE"]),
                float(nodes.loc[j, "LONGITUDE"]),
            )
            distances[i, j] = d
            distances[j, i] = d

    for i in range(n):
        candidate_idx = np.argsort(distances[i])
        added = 0
        for j in candidate_idx:
            if i == j or not np.isfinite(distances[i, j]):
                continue
            if distances[i, j] > max_edge_m:
                continue
            weight = 1.0 / max(distances[i, j], 1.0)
            adj[i, j] = max(adj[i, j], weight)
            adj[j, i] = max(adj[j, i], weight)
            added += 1
            if added >= k_nearest:
                break

    np.fill_diagonal(adj, 1.0)
    return normalize_adjacency(adj, mode="row")


def load_node_features(node_feature_csv: Path | None, area_sites: pd.DataFrame) -> np.ndarray | None:
    if node_feature_csv is None:
        return None
    feature_df = pd.read_csv(node_feature_csv)
    if "SITE_NO" not in feature_df.columns:
        raise ValueError("node feature CSV must contain SITE_NO.")
    feature_df["SITE_NO"] = pd.to_numeric(feature_df["SITE_NO"], errors="coerce").astype(int)
    merged = area_sites[["SITE_NO"]].merge(feature_df, on="SITE_NO", how="left")
    feature_cols = [column for column in merged.columns if column != "SITE_NO"]
    numeric_feature_cols = []
    for column in feature_cols:
        if pd.api.types.is_numeric_dtype(merged[column]):
            numeric_feature_cols.append(column)
    feature_cols = numeric_feature_cols
    if not feature_cols:
        raise ValueError("node feature CSV must include at least one feature column besides SITE_NO.")
    merged[feature_cols] = merged[feature_cols].fillna(0.0)
    return merged[feature_cols].to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    site_ts, site_summary = load_or_build_interim(args.data_dir, args.signals_url, args.volume_url)

    area_sites = pd.read_csv(args.study_area_csv)
    area_sites["SITE_NO"] = pd.to_numeric(area_sites["SITE_NO"], errors="coerce").astype(int)
    area_sites = area_sites.merge(site_summary, on="SITE_NO", how="left")
    area_sites = area_sites.sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    area_sites["area_order"] = np.arange(len(area_sites))

    adjacency = build_area_graph(area_sites, k_nearest=args.k_nearest, max_edge_m=args.max_edge_m)
    node_static_features = load_node_features(args.node_feature_csv, area_sites)
    matrix = build_generic_matrix(
        site_ts[site_ts["SITE_NO"].isin(area_sites["SITE_NO"])].rename(columns={"site_volume": "value"}),
        area_sites,
        node_key_col="SITE_NO",
        value_col="value",
    )

    area_sites.to_csv(out_dir / "area_sites.csv", index=False)
    matrix.to_csv(out_dir / "area_site_matrix.csv")
    pd.DataFrame(adjacency, index=area_sites["SITE_NO"], columns=area_sites["SITE_NO"]).to_csv(
        out_dir / "area_adjacency.csv"
    )

    run_config = {
        "study_area_csv": str(args.study_area_csv),
        "study_area_name": sanitize_slug(args.study_area_csv.stem),
        "node_count": int(len(area_sites)),
        "variant": args.variant,
        "node_feature_csv": str(args.node_feature_csv) if args.node_feature_csv is not None else None,
        "node_static_dim": int(0 if node_static_features is None else node_static_features.shape[1]),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "seed": args.seed,
        "k_nearest": args.k_nearest,
        "max_edge_m": args.max_edge_m,
    }
    write_json(out_dir / "run_config.json", run_config)

    training_summary = run_training(
        matrix=matrix,
        adj_norm=adjacency,
        out_dir=out_dir,
        model_variant=args.variant,
        node_static_features=node_static_features,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    run_config["training_summary"] = training_summary
    write_json(out_dir / "run_config.json", run_config)
    print(json.dumps(training_summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
