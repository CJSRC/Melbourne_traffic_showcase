from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from baseline_pipeline import set_global_seed, write_json
from experiment_metrics import count_trainable_parameters, evaluate_regression_arrays
from timestamp_conditioned_siteyear_slot_residual_mlp import (
    build_long_run_array,
    build_profile_array,
    fill_profile_nans,
    make_date_arrays,
)
from train_route_multimodal_geh import (
    HOLIDAY_FEATURE_COLS,
    SLOT_COLS,
    SLOT_COUNT,
    TemporalConvBlock,
    apply_graph_smoothing,
    best_blend_with_base,
    best_graph_smoothing,
    build_lag_row_indices,
    geh_surrogate,
    gradients_are_finite,
    hourly_huber_loss,
    load_adjacency_from_csv,
    reconstruct_flow,
    resolve_lag_vector,
)
from victoria_holidays import add_victoria_holiday_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervisor v4 multimodal route trainer with balanced branch fusion.")
    parser.add_argument("--slot-wide-parquet", type=Path, required=True)
    parser.add_argument("--study-area-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--train-years", nargs="+", type=int, required=True)
    # Which years the log_ratio base is averaged over. Defaults to the training years, which
    # reproduces every run made before this argument existed.
    parser.add_argument("--anchor-years", nargs="*", type=int, default=None)
    parser.add_argument("--val-years", nargs="+", type=int, required=True)
    parser.add_argument("--test-years", nargs="+", type=int, required=True)
    parser.add_argument("--visual-feature-csv", type=Path, default=None)
    parser.add_argument("--net-embedding-csv", type=Path, default=None)
    parser.add_argument("--traffic-embedding-csv", type=Path, default=None)
    # Planning attributes get their own branch rather than being appended to the
    # visual table: 43 columns concatenated onto 2846 share one linear layer with
    # them, and a channel outnumbered sixty-six to one cannot be judged on its own.
    parser.add_argument("--attr-feature-csv", type=Path, default=None)
    parser.add_argument("--adjacency-csv", type=Path, default=None)
    parser.add_argument("--target-mode", choices=["residual", "log_flow", "log_ratio"], default="log_ratio")
    parser.add_argument("--loss-mode", choices=["huber", "huber_geh", "huber_geh_hourly"], default="huber_geh_hourly")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--geh-weight", type=float, default=0.10)
    parser.add_argument("--hourly-weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-days", type=int, default=120000)
    parser.add_argument("--max-val-days", type=int, default=None)
    parser.add_argument("--max-test-days", type=int, default=None)
    parser.add_argument("--base-input", action="store_true")
    parser.add_argument("--use-lag-features", action="store_true")
    parser.add_argument("--use-holiday-features", action="store_true")
    parser.add_argument("--disable-time-features", action="store_true")
    parser.add_argument("--time-feature-preset", choices=["full", "planning_light"], default="full")
    parser.add_argument("--graph-smoothing", action="store_true")
    parser.add_argument("--blend-base", action="store_true")
    parser.add_argument("--fixed-blend-alpha", type=float, default=None)
    # the searched grid; the defaults are the range that was hard-coded before
    parser.add_argument("--blend-alpha-min", type=float, default=0.60)
    parser.add_argument("--blend-alpha-max", type=float, default=1.10)
    parser.add_argument("--fixed-graph-alpha", type=float, default=None)
    parser.add_argument("--region-selection-mode", choices=["all", "topk"], default="all")
    parser.add_argument("--visual-backbone-id", type=str, default=None)
    parser.add_argument("--json-embed-model-id", type=str, default=None)
    parser.add_argument("--visual-compressor", choices=["none", "pca64", "ae64", "vae64"], default="none")
    parser.add_argument("--visual-latent-dim", type=int, default=64)
    parser.add_argument("--visual-compressor-hidden-dim", type=int, default=192)
    parser.add_argument("--visual-recon-weight", type=float, default=0.05)
    parser.add_argument("--visual-kl-weight", type=float, default=0.001)
    parser.add_argument("--proj-visual-dim", type=int, default=128)
    parser.add_argument("--proj-net-dim", type=int, default=32)
    parser.add_argument("--proj-traffic-dim", type=int, default=32)
    parser.add_argument("--proj-attr-dim", type=int, default=32)
    parser.add_argument("--proj-time-dim", type=int, default=32)
    parser.add_argument("--checkpoint-in", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def resolve_time_feature_flags(args: argparse.Namespace) -> dict[str, bool]:
    if args.disable_time_features:
        return {
            "include_calendar_core": False,
            "include_profile_features": False,
            "include_base_input": False,
            "include_holiday": False,
            "include_lag_features": False,
        }
    if args.time_feature_preset == "planning_light":
        return {
            "include_calendar_core": True,
            "include_profile_features": False,
            "include_base_input": False,
            "include_holiday": bool(args.use_holiday_features),
            "include_lag_features": False,
        }
    return {
        "include_calendar_core": True,
        "include_profile_features": True,
        "include_base_input": bool(args.base_input),
        "include_holiday": bool(args.use_holiday_features),
        "include_lag_features": bool(args.use_lag_features),
    }


def coerce_numeric_feature_df(df: pd.DataFrame, keep_years: list[int]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["SITE_NO"] = pd.to_numeric(out["SITE_NO"], errors="coerce").astype(int)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out = out[out["year"].isin(keep_years)].copy()
    feature_cols: list[str] = []
    numeric_cols: dict[str, pd.Series] = {"SITE_NO": out["SITE_NO"], "year": out["year"]}
    for column in out.columns:
        if column in {"SITE_NO", "year"}:
            continue
        coerced = pd.to_numeric(out[column], errors="coerce")
        if coerced.notna().any():
            numeric_cols[column] = coerced.astype(np.float32)
            feature_cols.append(column)
    return pd.DataFrame(numeric_cols), feature_cols


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    return payload


def load_siteyear_table(feature_csv: Path | None, site_ids: list[int], keep_years: list[int]) -> tuple[np.ndarray, list[str]]:
    if feature_csv is None:
        return np.zeros((len(site_ids), len(keep_years), 0), dtype=np.float32), []
    df = pd.read_csv(feature_csv)
    df, feature_cols = coerce_numeric_feature_df(df, keep_years)
    year_to_idx = {year: idx for idx, year in enumerate(keep_years)}
    site_to_idx = {site: idx for idx, site in enumerate(site_ids)}
    arr = np.zeros((len(site_ids), len(keep_years), len(feature_cols)), dtype=np.float32)
    for row in df.itertuples(index=False):
        site_no = int(row[0])
        year = int(row[1])
        if site_no not in site_to_idx or year not in year_to_idx:
            continue
        arr[site_to_idx[site_no], year_to_idx[year], :] = np.nan_to_num(np.asarray(row[2:], dtype=np.float32), nan=0.0)
    return arr, feature_cols


def scale_branch_by_train_years(arr: np.ndarray, train_year_idx: list[int]) -> tuple[np.ndarray, dict[str, float]]:
    if arr.shape[-1] == 0:
        return arr.astype(np.float32), {"input_dim": 0, "scaled_dim": 0}
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    train_flat = arr[:, train_year_idx, :].reshape(-1, arr.shape[-1]).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(train_flat)
    scaled = scaler.transform(flat).reshape(arr.shape).astype(np.float32)
    return scaled, {
        "input_dim": int(arr.shape[-1]),
        "scaled_dim": int(arr.shape[-1]),
        "train_rows": int(train_flat.shape[0]),
    }


def apply_fixed_visual_pca(
    arr: np.ndarray,
    train_year_idx: list[int],
    latent_dim: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if arr.shape[-1] == 0:
        return arr.astype(np.float32), {"input_dim": 0, "pca_components": 0}
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    train_flat = arr[:, train_year_idx, :].reshape(-1, arr.shape[-1]).astype(np.float32)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
    n_components = min(latent_dim, arr.shape[-1], max(int(train_flat.shape[0]) - 1, 1))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_scaled)
    reduced = pca.transform(scaler.transform(flat)).reshape(arr.shape[0], arr.shape[1], n_components).astype(np.float32)
    return reduced, {
        "input_dim": int(arr.shape[-1]),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


@dataclass
class PreparedDataV4:
    site_ids: list[int]
    keep_years: list[int]
    year_to_idx: dict[int, int]
    wide: pd.DataFrame
    values: np.ndarray
    valid_mask: np.ndarray
    date_arrays: dict[str, np.ndarray]
    dow_profile: np.ndarray
    month_profile: np.ndarray
    long_run_profile: np.ndarray
    lag_row_indices: dict[int, np.ndarray]
    flow_feature_scale: float
    visual_features: np.ndarray
    net_features: np.ndarray
    traffic_features: np.ndarray
    attr_features: np.ndarray
    visual_feature_dim: int
    net_feature_dim: int
    traffic_feature_dim: int
    attr_feature_dim: int
    branch_summary: dict[str, object]


def prepare_data(args: argparse.Namespace) -> PreparedDataV4:
    area_sites = pd.read_csv(args.study_area_csv)
    area_sites["SITE_NO"] = pd.to_numeric(area_sites["SITE_NO"], errors="coerce").astype(int)
    area_sites = area_sites.sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    site_ids = area_sites["SITE_NO"].tolist()
    site_to_idx = {site: idx for idx, site in enumerate(site_ids)}

    wide = pd.read_parquet(args.slot_wide_parquet)
    wide["SITE_NO"] = pd.to_numeric(wide["SITE_NO"], errors="coerce").astype(int)
    wide["date"] = pd.to_datetime(wide["date"]).dt.normalize()
    wide = wide[wide["SITE_NO"].isin(site_ids)].copy()
    keep_years = sorted(set(args.train_years) | set(args.val_years) | set(args.test_years))
    wide = wide[wide["date"].dt.year.isin(keep_years)].copy()
    wide["site_idx"] = wide["SITE_NO"].map(site_to_idx).astype(int)
    wide = wide.sort_values(["date", "SITE_NO"]).reset_index(drop=True)

    values = wide[SLOT_COLS].to_numpy(dtype=np.float32).copy()
    values[values < 0] = np.nan
    valid_mask = np.isfinite(values)
    date_arrays = make_date_arrays(wide["date"])
    if args.use_holiday_features:
        holiday_df = add_victoria_holiday_features(wide[["date"]].copy())
        for column in HOLIDAY_FEATURE_COLS:
            date_arrays[column] = holiday_df[column].to_numpy(dtype=np.float32)
    row_years = date_arrays["year"]
    train_row_mask = np.isin(row_years, np.asarray(args.train_years))
    train_site_idx_arr = wide.loc[train_row_mask, "site_idx"].to_numpy(dtype=np.int32)
    train_values = values[train_row_mask]

    # The profiles the target is expressed against need not come from the same years the
    # network trains on. Tying them together forces a choice between sample count and a base
    # calibrated to the year being predicted: over 2014-2019 and 2022 the base runs 11% above
    # 2024, and narrowing the window to 2022 alone brings that to -1.9% while costing seven
    # eighths of the training rows. --anchor-years lets both be had. Default is the training
    # years, which is what every earlier run did.
    anchor_years = args.anchor_years or args.train_years
    anchor_row_mask = np.isin(row_years, np.asarray(anchor_years))
    if not anchor_row_mask.any():
        raise SystemExit(f"no rows for --anchor-years {anchor_years}")
    anchor_site_idx_arr = wide.loc[anchor_row_mask, "site_idx"].to_numpy(dtype=np.int32)
    anchor_values = values[anchor_row_mask]

    dow_profile_raw = build_profile_array(
        site_idx_arr=anchor_site_idx_arr,
        group_idx_arr=date_arrays["dow"][anchor_row_mask],
        values=anchor_values,
        n_sites=len(site_ids),
        n_groups=7,
    )
    month_profile_raw = build_profile_array(
        site_idx_arr=anchor_site_idx_arr,
        group_idx_arr=date_arrays["month"][anchor_row_mask],
        values=anchor_values,
        n_sites=len(site_ids),
        n_groups=12,
    )
    long_run_raw = build_long_run_array(anchor_site_idx_arr, anchor_values, len(site_ids))
    fallback = np.nanmean(anchor_values, axis=0, keepdims=True).astype(np.float32)
    long_run_profile = np.where(np.isfinite(long_run_raw), long_run_raw, fallback).astype(np.float32)
    dow_profile = fill_profile_nans(dow_profile_raw, long_run_profile)
    month_profile = fill_profile_nans(month_profile_raw, long_run_profile)
    flow_feature_scale = float(np.nanstd(train_values))
    if not np.isfinite(flow_feature_scale) or flow_feature_scale < 1e-3:
        flow_feature_scale = 1.0
    lag_row_indices = build_lag_row_indices(wide, [1, 7, 14]) if args.use_lag_features else {}

    year_to_idx = {year: idx for idx, year in enumerate(keep_years)}
    train_year_idx = [year_to_idx[year] for year in args.train_years if year in year_to_idx]
    visual_raw, visual_cols = load_siteyear_table(args.visual_feature_csv, site_ids, keep_years)
    net_raw, net_cols = load_siteyear_table(args.net_embedding_csv, site_ids, keep_years)
    traffic_raw, traffic_cols = load_siteyear_table(args.traffic_embedding_csv, site_ids, keep_years)
    attr_raw, attr_cols = load_siteyear_table(args.attr_feature_csv, site_ids, keep_years)

    visual_scaled, visual_scale_meta = scale_branch_by_train_years(visual_raw, train_year_idx)
    net_scaled, net_scale_meta = scale_branch_by_train_years(net_raw, train_year_idx)
    traffic_scaled, traffic_scale_meta = scale_branch_by_train_years(traffic_raw, train_year_idx)
    attr_scaled, attr_scale_meta = scale_branch_by_train_years(attr_raw, train_year_idx)

    branch_summary: dict[str, object] = {
        "visual_scale": visual_scale_meta,
        "net_scale": net_scale_meta,
        "traffic_scale": traffic_scale_meta,
        "attr_scale": attr_scale_meta,
    }
    if args.visual_compressor == "pca64":
        visual_features, visual_pca_meta = apply_fixed_visual_pca(visual_scaled, train_year_idx, args.visual_latent_dim)
        branch_summary["visual_pca"] = visual_pca_meta
    else:
        visual_features = visual_scaled

    return PreparedDataV4(
        site_ids=site_ids,
        keep_years=keep_years,
        year_to_idx=year_to_idx,
        wide=wide,
        values=values,
        valid_mask=valid_mask,
        date_arrays=date_arrays,
        dow_profile=dow_profile,
        month_profile=month_profile,
        long_run_profile=long_run_profile,
        lag_row_indices=lag_row_indices,
        flow_feature_scale=flow_feature_scale,
        visual_features=visual_features.astype(np.float32),
        net_features=net_scaled.astype(np.float32),
        traffic_features=traffic_scaled.astype(np.float32),
        attr_features=attr_scaled.astype(np.float32),
        visual_feature_dim=int(visual_features.shape[-1]),
        net_feature_dim=int(net_scaled.shape[-1]),
        traffic_feature_dim=int(traffic_scaled.shape[-1]),
        attr_feature_dim=int(attr_scaled.shape[-1]),
        branch_summary=branch_summary,
    )


class DaySequenceDatasetV4(Dataset):
    def __init__(
        self,
        *,
        data: PreparedDataV4,
        row_indices: np.ndarray,
        base_input: bool,
        use_time_features: bool,
        use_lag_features: bool,
        use_holiday_features: bool,
        time_feature_preset: str,
    ):
        self.data = data
        self.row_indices = row_indices.astype(np.int32)
        self.base_input = base_input
        self.use_time_features = use_time_features
        self.use_lag_features = use_lag_features
        self.use_holiday_features = use_holiday_features
        self.time_feature_preset = time_feature_preset
        self.time_feature_flags = resolve_time_feature_flags(
            argparse.Namespace(
                disable_time_features=not use_time_features,
                time_feature_preset=time_feature_preset,
                base_input=base_input,
                use_lag_features=use_lag_features,
                use_holiday_features=use_holiday_features,
            )
        )
        slot_frac = np.arange(SLOT_COUNT, dtype=np.float32) / float(SLOT_COUNT)
        self.slot_sin = np.sin(2 * np.pi * slot_frac).astype(np.float32)
        self.slot_cos = np.cos(2 * np.pi * slot_frac).astype(np.float32)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, idx: int):
        row_idx = int(self.row_indices[idx])
        site_idx = int(self.data.wide.iloc[row_idx]["site_idx"])
        year = int(self.data.date_arrays["year"][row_idx])
        year_idx = int(self.data.year_to_idx[year])
        dow = int(self.data.date_arrays["dow"][row_idx])
        month = int(self.data.date_arrays["month"][row_idx])
        doy = int(self.data.date_arrays["doy"][row_idx])
        is_weekend = float(self.data.date_arrays["is_weekend"][row_idx])
        year_norm = float(self.data.date_arrays["year_norm"][row_idx])

        y_true = self.data.values[row_idx].astype(np.float32)
        valid = self.data.valid_mask[row_idx].astype(np.float32)
        base = self.data.dow_profile[site_idx, dow].astype(np.float32)
        month_base = self.data.month_profile[site_idx, month].astype(np.float32) / self.data.flow_feature_scale
        long_base = self.data.long_run_profile[site_idx].astype(np.float32) / self.data.flow_feature_scale
        base_input_arr = base / self.data.flow_feature_scale

        if self.use_time_features:
            doy_frac = (float(doy) + np.arange(SLOT_COUNT, dtype=np.float32) / SLOT_COUNT) / 366.0
            dyn_parts = [
                np.full(SLOT_COUNT, float(self.data.date_arrays["dow_sin"][row_idx]), dtype=np.float32),
                np.full(SLOT_COUNT, float(self.data.date_arrays["dow_cos"][row_idx]), dtype=np.float32),
                np.full(SLOT_COUNT, float(self.data.date_arrays["month_sin"][row_idx]), dtype=np.float32),
                np.full(SLOT_COUNT, float(self.data.date_arrays["month_cos"][row_idx]), dtype=np.float32),
                np.sin(2 * np.pi * doy_frac).astype(np.float32),
                np.cos(2 * np.pi * doy_frac).astype(np.float32),
                np.full(SLOT_COUNT, is_weekend, dtype=np.float32),
                np.full(SLOT_COUNT, year_norm, dtype=np.float32),
                self.slot_sin,
                self.slot_cos,
            ]
            if self.time_feature_flags["include_profile_features"]:
                dyn_parts.extend([month_base, long_base])
            if self.time_feature_flags["include_base_input"]:
                dyn_parts.append(base_input_arr)
            if self.time_feature_flags["include_holiday"] and self.data.date_arrays.get(HOLIDAY_FEATURE_COLS[0]) is not None:
                for column in HOLIDAY_FEATURE_COLS:
                    dyn_parts.append(np.full(SLOT_COUNT, float(self.data.date_arrays[column][row_idx]), dtype=np.float32))
            if self.time_feature_flags["include_lag_features"] and self.data.lag_row_indices:
                lag1, lag7, lag14, lag_mean, same_dow_mean = resolve_lag_vector(
                    self.data.values,
                    self.data.lag_row_indices,
                    row_idx,
                    base,
                )
                dyn_parts.extend(
                    [
                        lag1 / self.data.flow_feature_scale,
                        lag7 / self.data.flow_feature_scale,
                        lag14 / self.data.flow_feature_scale,
                        lag_mean / self.data.flow_feature_scale,
                        same_dow_mean / self.data.flow_feature_scale,
                        (lag1 - base) / self.data.flow_feature_scale,
                        (lag7 - base) / self.data.flow_feature_scale,
                        (lag14 - base) / self.data.flow_feature_scale,
                    ]
                )
        else:
            dyn_parts = [np.zeros(SLOT_COUNT, dtype=np.float32)]

        return {
            "x_dyn": torch.tensor(np.stack(dyn_parts, axis=1).astype(np.float32), dtype=torch.float32),
            "x_visual": torch.tensor(self.data.visual_features[site_idx, year_idx].astype(np.float32), dtype=torch.float32),
            "x_net": torch.tensor(self.data.net_features[site_idx, year_idx].astype(np.float32), dtype=torch.float32),
            "x_traffic": torch.tensor(self.data.traffic_features[site_idx, year_idx].astype(np.float32), dtype=torch.float32),
            "x_attr": torch.tensor(self.data.attr_features[site_idx, year_idx].astype(np.float32), dtype=torch.float32),
            "site_idx": torch.tensor(site_idx, dtype=torch.long),
            "y_true": torch.tensor(np.nan_to_num(y_true, nan=0.0), dtype=torch.float32),
            "base": torch.tensor(base, dtype=torch.float32),
            "valid": torch.tensor(valid, dtype=torch.float32),
        }


class OptionalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if self.in_dim > 0 and self.out_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_dim <= 0:
            return x.new_zeros((*x.shape[:-1], 0))
        if self.net is None:
            return x.new_zeros((*x.shape[:-1], self.out_dim))
        return self.net(x)


class IdentityVisualCompressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = int(input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = x.new_zeros(())
        return x, {"visual_recon_loss": zero, "visual_kl_loss": zero}


class AutoEncoderVisualCompressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, variational: bool):
        super().__init__()
        self.variational = variational
        self.output_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        if variational:
            self.mu_head = nn.Linear(hidden_dim, latent_dim)
            self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        else:
            self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.encoder(x)
        if self.variational:
            mu = self.mu_head(h)
            logvar = self.logvar_head(h).clamp(min=-8.0, max=8.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            else:
                z = mu
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        else:
            z = self.latent_head(h)
            kl = x.new_zeros(())
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, x)
        return z, {"visual_recon_loss": recon_loss, "visual_kl_loss": kl}


class DayTCNBalancedV4(nn.Module):
    def __init__(
        self,
        *,
        dyn_dim: int,
        visual_dim: int,
        net_dim: int,
        traffic_dim: int,
        attr_dim: int,
        site_count: int,
        hidden_dim: int,
        emb_dim: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        proj_visual_dim: int,
        proj_net_dim: int,
        proj_traffic_dim: int,
        proj_attr_dim: int,
        proj_time_dim: int,
        visual_compressor: str,
        visual_latent_dim: int,
        visual_compressor_hidden_dim: int,
    ):
        super().__init__()
        self.site_emb = nn.Embedding(site_count, emb_dim)
        if visual_dim <= 0:
            visual_compressor = "none"
        if visual_compressor == "ae64":
            self.visual_compressor = AutoEncoderVisualCompressor(visual_dim, visual_compressor_hidden_dim, visual_latent_dim, False)
        elif visual_compressor == "vae64":
            self.visual_compressor = AutoEncoderVisualCompressor(visual_dim, visual_compressor_hidden_dim, visual_latent_dim, True)
        else:
            self.visual_compressor = IdentityVisualCompressor(visual_dim)

        self.effective_visual_compressor = visual_compressor
        self.effective_proj_visual_dim = int(proj_visual_dim if self.visual_compressor.output_dim > 0 else 0)
        self.effective_proj_net_dim = int(proj_net_dim if net_dim > 0 else 0)
        self.effective_proj_traffic_dim = int(proj_traffic_dim if traffic_dim > 0 else 0)
        self.effective_proj_attr_dim = int(proj_attr_dim if attr_dim > 0 else 0)
        self.effective_proj_time_dim = int(proj_time_dim)

        self.visual_proj = OptionalProjector(self.visual_compressor.output_dim, self.effective_proj_visual_dim, dropout)
        self.net_proj = OptionalProjector(net_dim, self.effective_proj_net_dim, dropout)
        self.traffic_proj = OptionalProjector(traffic_dim, self.effective_proj_traffic_dim, dropout)
        # with no attribute csv this projector holds no parameters and adds nothing to
        # static_in_dim, so an older checkpoint still loads into the same shapes
        self.attr_proj = OptionalProjector(attr_dim, self.effective_proj_attr_dim, dropout)
        self.time_proj = OptionalProjector(dyn_dim, proj_time_dim, dropout)

        static_in_dim = (emb_dim + self.effective_proj_visual_dim + self.effective_proj_net_dim
                         + self.effective_proj_traffic_dim + self.effective_proj_attr_dim)
        fusion_in_dim = hidden_dim + max(self.effective_proj_time_dim, 0)
        self.static_fuse = nn.Sequential(
            nn.Linear(static_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        dilations = [2**idx for idx in range(layers)]
        self.blocks = nn.ModuleList([TemporalConvBlock(hidden_dim, kernel_size, dilation, dropout) for dilation in dilations])
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_visual: torch.Tensor,
        x_net: torch.Tensor,
        x_traffic: torch.Tensor,
        x_attr: torch.Tensor,
        site_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        site_vec = self.site_emb(site_idx)
        visual_latent, aux_losses = self.visual_compressor(x_visual)
        visual_h = self.visual_proj(visual_latent)
        net_h = self.net_proj(x_net)
        traffic_h = self.traffic_proj(x_traffic)
        attr_h = self.attr_proj(x_attr)
        static_h = self.static_fuse(torch.cat([site_vec, visual_h, net_h, traffic_h, attr_h], dim=-1))
        static_h = static_h.unsqueeze(1).expand(-1, x_dyn.shape[1], -1)
        time_h = self.time_proj(x_dyn)
        h = self.fusion(torch.cat([static_h, time_h], dim=-1))
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        out = self.head(h).squeeze(1)
        return out, aux_losses


def build_dyn_dim(args: argparse.Namespace) -> int:
    if args.disable_time_features:
        return 1
    flags = resolve_time_feature_flags(args)
    dim = 10
    if flags["include_profile_features"]:
        dim += 2
    if flags["include_base_input"]:
        dim += 1
    if flags["include_holiday"]:
        dim += len(HOLIDAY_FEATURE_COLS)
    if flags["include_lag_features"]:
        dim += 8
    return dim


def train_day_tcn_v4(args: argparse.Namespace, data: PreparedDataV4) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    row_years = data.date_arrays["year"]
    train_rows = np.flatnonzero(np.isin(row_years, np.asarray(args.train_years))).astype(np.int32)
    val_rows = np.flatnonzero(np.isin(row_years, np.asarray(args.val_years))).astype(np.int32)
    test_rows = np.flatnonzero(np.isin(row_years, np.asarray(args.test_years))).astype(np.int32)
    if args.max_train_days is not None and len(train_rows) > args.max_train_days:
        train_rows = np.sort(rng.choice(train_rows, size=args.max_train_days, replace=False).astype(np.int32))
    if args.max_val_days is not None and len(val_rows) > args.max_val_days:
        val_rows = np.sort(rng.choice(val_rows, size=args.max_val_days, replace=False).astype(np.int32))
    if args.max_test_days is not None and len(test_rows) > args.max_test_days:
        test_rows = np.sort(rng.choice(test_rows, size=args.max_test_days, replace=False).astype(np.int32))

    use_time_features = not args.disable_time_features
    train_ds = DaySequenceDatasetV4(
        data=data,
        row_indices=train_rows,
        base_input=args.base_input,
        use_time_features=use_time_features,
        use_lag_features=args.use_lag_features,
        use_holiday_features=args.use_holiday_features,
        time_feature_preset=args.time_feature_preset,
    )
    val_ds = DaySequenceDatasetV4(
        data=data,
        row_indices=val_rows,
        base_input=args.base_input,
        use_time_features=use_time_features,
        use_lag_features=args.use_lag_features,
        use_holiday_features=args.use_holiday_features,
        time_feature_preset=args.time_feature_preset,
    )
    test_ds = DaySequenceDatasetV4(
        data=data,
        row_indices=test_rows,
        base_input=args.base_input,
        use_time_features=use_time_features,
        use_lag_features=args.use_lag_features,
        use_holiday_features=args.use_holiday_features,
        time_feature_preset=args.time_feature_preset,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(str(args.device).startswith("cuda")),
        "persistent_workers": bool(int(args.num_workers) > 0),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    dyn_dim = build_dyn_dim(args)
    model = DayTCNBalancedV4(
        dyn_dim=dyn_dim,
        visual_dim=data.visual_feature_dim,
        net_dim=data.net_feature_dim,
        traffic_dim=data.traffic_feature_dim,
        attr_dim=data.attr_feature_dim,
        site_count=len(data.site_ids),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        proj_visual_dim=args.proj_visual_dim,
        proj_net_dim=args.proj_net_dim,
        proj_traffic_dim=args.proj_traffic_dim,
        proj_attr_dim=args.proj_attr_dim,
        proj_time_dim=args.proj_time_dim,
        visual_compressor=args.visual_compressor,
        visual_latent_dim=args.visual_latent_dim,
        visual_compressor_hidden_dim=args.visual_compressor_hidden_dim,
    ).to(args.device)
    initial_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.HuberLoss(delta=60.0)

    best_state = None
    best_score = None
    stale = 0
    history: list[dict[str, float]] = []
    if args.eval_only:
        if args.checkpoint_in is None:
            raise ValueError("--eval-only requires --checkpoint-in")
        best_state = load_checkpoint_state(args.checkpoint_in)

    for epoch in range(1, args.epochs + 1) if not args.eval_only else []:
        model.train()
        train_losses: list[float] = []
        train_recon_losses: list[float] = []
        train_kl_losses: list[float] = []
        for batch in train_loader:
            x_dyn = batch["x_dyn"].to(args.device)
            x_visual = batch["x_visual"].to(args.device)
            x_net = batch["x_net"].to(args.device)
            x_traffic = batch["x_traffic"].to(args.device)
            x_attr = batch["x_attr"].to(args.device)
            site_idx = batch["site_idx"].to(args.device)
            y_true = batch["y_true"].to(args.device)
            base = batch["base"].to(args.device)
            valid = batch["valid"].to(args.device)

            optimizer.zero_grad()
            raw, aux = model(x_dyn, x_visual, x_net, x_traffic, x_attr, site_idx)
            pred = reconstruct_flow(raw, base, args.target_mode)
            loss = huber(pred * valid, y_true * valid)
            if args.loss_mode in {"huber_geh", "huber_geh_hourly"}:
                loss = loss + args.geh_weight * geh_surrogate(y_true, pred, valid)
            if args.loss_mode == "huber_geh_hourly":
                loss = loss + args.hourly_weight * hourly_huber_loss(y_true, pred, valid)
            recon_loss = aux["visual_recon_loss"] if "visual_recon_loss" in aux else loss.new_zeros(())
            kl_loss = aux["visual_kl_loss"] if "visual_kl_loss" in aux else loss.new_zeros(())
            if args.visual_compressor in {"ae64", "vae64"}:
                loss = loss + args.visual_recon_weight * recon_loss + args.visual_kl_weight * kl_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_recon_losses.append(float(recon_loss.item()))
            train_kl_losses.append(float(kl_loss.item()))

        model.eval()
        val_pred_parts: list[np.ndarray] = []
        val_true_parts: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                raw, _ = model(
                    batch["x_dyn"].to(args.device),
                    batch["x_visual"].to(args.device),
                    batch["x_net"].to(args.device),
                    batch["x_traffic"].to(args.device),
                    batch["x_attr"].to(args.device),
                    batch["site_idx"].to(args.device),
                )
                pred = reconstruct_flow(raw, batch["base"].to(args.device), args.target_mode)
                val_pred_parts.append(pred.detach().cpu().numpy().astype(np.float32))
                val_true_parts.append(batch["y_true"].numpy().astype(np.float32))
        val_pred = np.concatenate(val_pred_parts).reshape(-1)
        val_true = np.concatenate(val_true_parts).reshape(-1)
        if not np.isfinite(val_pred).any():
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses) if train_losses else np.nan), "val_GEH_median": float("inf")})
            stale += 1
            if stale >= args.patience:
                break
            continue
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1e6, neginf=0.0)
        val_metrics = evaluate_regression_arrays(val_true, val_pred, geh_scale_factor=4.0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses) if train_losses else np.nan),
                "train_visual_recon_loss": float(np.mean(train_recon_losses) if train_recon_losses else 0.0),
                "train_visual_kl_loss": float(np.mean(train_kl_losses) if train_kl_losses else 0.0),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        score = (val_metrics["GEH_median"], val_metrics["MAE"])
        if best_score is None or score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        best_state = initial_state
    model.load_state_dict(best_state)
    # Eval-only skips the training loop, so explicitly disable dropout before inference.
    model.eval()

    def predict_days(loader: DataLoader, row_indices: np.ndarray) -> pd.DataFrame:
        pred_rows: list[pd.DataFrame] = []
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                raw, _ = model(
                    batch["x_dyn"].to(args.device),
                    batch["x_visual"].to(args.device),
                    batch["x_net"].to(args.device),
                    batch["x_traffic"].to(args.device),
                    batch["x_attr"].to(args.device),
                    batch["site_idx"].to(args.device),
                )
                pred = reconstruct_flow(raw, batch["base"].to(args.device), args.target_mode).detach().cpu().numpy().astype(np.float32)
                y_true = batch["y_true"].numpy().astype(np.float32)
                base = batch["base"].numpy().astype(np.float32)
                bs = pred.shape[0]
                rows = row_indices[cursor : cursor + bs]
                cursor += bs
                for local_idx, row_idx in enumerate(rows):
                    meta_row = data.wide.iloc[int(row_idx)]
                    date_val = pd.Timestamp(meta_row["date"])
                    pred_rows.append(
                        pd.DataFrame(
                            {
                                "SITE_NO": int(meta_row["SITE_NO"]),
                                "date": np.repeat(date_val, SLOT_COUNT),
                                "year": np.repeat(int(date_val.year), SLOT_COUNT),
                                "slot_idx": np.arange(SLOT_COUNT, dtype=np.int16),
                                "y_true": y_true[local_idx],
                                "y_pred": pred[local_idx],
                                "base_dow_slot_profile": base[local_idx],
                            }
                        )
                    )
        return pd.concat(pred_rows, ignore_index=True)

    return {
        "history": history,
        "val_df": predict_days(val_loader, val_rows),
        "test_df": predict_days(test_loader, test_rows),
        "model": model,
        "model_summary": {
            "dyn_dim": int(dyn_dim),
            "visual_feature_dim": int(data.visual_feature_dim),
            "net_feature_dim": int(data.net_feature_dim),
            "traffic_feature_dim": int(data.traffic_feature_dim),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "kernel_size": int(args.kernel_size),
            "emb_dim": int(args.emb_dim),
            "proj_visual_dim": int(model.effective_proj_visual_dim),
            "proj_net_dim": int(model.effective_proj_net_dim),
            "proj_traffic_dim": int(model.effective_proj_traffic_dim),
            "proj_time_dim": int(model.effective_proj_time_dim),
            "requested_proj_visual_dim": int(args.proj_visual_dim),
            "requested_proj_net_dim": int(args.proj_net_dim),
            "requested_proj_traffic_dim": int(args.proj_traffic_dim),
            "requested_proj_time_dim": int(args.proj_time_dim),
            "visual_compressor": model.effective_visual_compressor,
            "requested_visual_compressor": args.visual_compressor,
            "visual_latent_dim": int(args.visual_latent_dim),
            "time_feature_preset": args.time_feature_preset,
            "trainable_params": count_trainable_parameters(model),
        },
        "branch_summary": data.branch_summary,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    torch.manual_seed(args.seed)

    data = prepare_data(args)
    result = train_day_tcn_v4(args, data)
    val_df = result["val_df"].copy()
    test_df = result["test_df"].copy()
    posthoc: dict[str, float] = {}

    if args.blend_base:
        if args.fixed_blend_alpha is not None:
            alpha = float(args.fixed_blend_alpha)
            posthoc["blend_alpha"] = alpha
            blended_val = np.clip(
                alpha * val_df["y_pred"].to_numpy(dtype=np.float32)
                + (1.0 - alpha) * val_df["base_dow_slot_profile"].to_numpy(dtype=np.float32),
                0.0,
                None,
            )
        else:
            blended_val, params = best_blend_with_base(
                val_df["y_true"].to_numpy(dtype=np.float32),
                val_df["y_pred"].to_numpy(dtype=np.float32),
                val_df["base_dow_slot_profile"].to_numpy(dtype=np.float32),
                alpha_min=args.blend_alpha_min,
                alpha_max=args.blend_alpha_max,
            )
            posthoc.update(params)
            alpha = params["blend_alpha"]
        val_df["y_pred"] = blended_val
        test_df["y_pred"] = np.clip(
            alpha * test_df["y_pred"].to_numpy(dtype=np.float32)
            + (1.0 - alpha) * test_df["base_dow_slot_profile"].to_numpy(dtype=np.float32),
            0.0,
            None,
        )

    if args.graph_smoothing and args.adjacency_csv is not None:
        adjacency = load_adjacency_from_csv(args.adjacency_csv, data.site_ids)
        if args.fixed_graph_alpha is not None:
            graph_alpha = float(args.fixed_graph_alpha)
            posthoc["graph_alpha"] = graph_alpha
            if graph_alpha > 0.0:
                val_df["y_pred"] = apply_graph_smoothing(
                    val_df,
                    adjacency=adjacency,
                    site_ids=data.site_ids,
                    alpha=graph_alpha,
                )
                test_df["y_pred"] = apply_graph_smoothing(
                    test_df,
                    adjacency=adjacency,
                    site_ids=data.site_ids,
                    alpha=graph_alpha,
                )
        else:
            smoothed_val, params = best_graph_smoothing(val_df, adjacency=adjacency, site_ids=data.site_ids)
            posthoc.update(params)
            val_df["y_pred"] = smoothed_val
            if params["graph_alpha"] > 0.0:
                test_df["y_pred"] = apply_graph_smoothing(test_df, adjacency=adjacency, site_ids=data.site_ids, alpha=float(params["graph_alpha"]))

    val_metrics = evaluate_regression_arrays(val_df["y_true"], val_df["y_pred"], geh_scale_factor=4.0)
    test_metrics = evaluate_regression_arrays(test_df["y_true"], test_df["y_pred"], geh_scale_factor=4.0)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["history"]).to_csv(run_dir / "training_history.csv", index=False)
    val_df.to_parquet(run_dir / "val_predictions.parquet", index=False)
    test_df.to_parquet(run_dir / "test_predictions.parquet", index=False)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in result["model"].state_dict().items()}}, run_dir / "model_state.pt")
    write_json(run_dir / "val_metrics.json", val_metrics)
    write_json(run_dir / "metrics.json", test_metrics)
    write_json(
        run_dir / "run_config.json",
        {
            "slot_wide_parquet": str(args.slot_wide_parquet),
            "study_area_csv": str(args.study_area_csv),
            "visual_feature_csv": str(args.visual_feature_csv) if args.visual_feature_csv else None,
            "net_embedding_csv": str(args.net_embedding_csv) if args.net_embedding_csv else None,
            "traffic_embedding_csv": str(args.traffic_embedding_csv) if args.traffic_embedding_csv else None,
            "adjacency_csv": str(args.adjacency_csv) if args.adjacency_csv else None,
            "train_years": args.train_years,
            # recorded resolved rather than as given, so a run that took the default is
            # still readable later without knowing what the default was
            "anchor_years": args.anchor_years or args.train_years,
            "val_years": args.val_years,
            "test_years": args.test_years,
            "target_mode": args.target_mode,
            "loss_mode": args.loss_mode,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "layers": args.layers,
            "kernel_size": args.kernel_size,
            "emb_dim": args.emb_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "geh_weight": args.geh_weight,
            "hourly_weight": args.hourly_weight,
            "patience": args.patience,
            "device": args.device,
            "seed": args.seed,
            "base_input": bool(args.base_input),
            "use_lag_features": bool(args.use_lag_features),
            "use_holiday_features": bool(args.use_holiday_features),
            "disable_time_features": bool(args.disable_time_features),
            "time_feature_preset": args.time_feature_preset,
            "time_feature_flags": resolve_time_feature_flags(args),
            "graph_smoothing": bool(args.graph_smoothing),
            "blend_base": bool(args.blend_base),
            "fixed_blend_alpha": args.fixed_blend_alpha,
            "blend_alpha_min": args.blend_alpha_min,
            "blend_alpha_max": args.blend_alpha_max,
            "fixed_graph_alpha": args.fixed_graph_alpha,
            "region_selection_mode": args.region_selection_mode,
            "visual_backbone_id": args.visual_backbone_id,
            "json_embed_model_id": args.json_embed_model_id,
            "visual_compressor": args.visual_compressor,
            "visual_latent_dim": args.visual_latent_dim,
            "visual_compressor_hidden_dim": args.visual_compressor_hidden_dim,
            "visual_recon_weight": args.visual_recon_weight,
            "visual_kl_weight": args.visual_kl_weight,
            "proj_visual_dim": args.proj_visual_dim,
            "proj_net_dim": args.proj_net_dim,
            "proj_traffic_dim": args.proj_traffic_dim,
            "proj_attr_dim": args.proj_attr_dim,
            "proj_time_dim": args.proj_time_dim,
            "checkpoint_in": str(args.checkpoint_in) if args.checkpoint_in else None,
            "eval_only": bool(args.eval_only),
            "model_summary": result["model_summary"],
            "branch_summary": result["branch_summary"],
            "posthoc": posthoc,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    print(run_dir)


if __name__ == "__main__":
    main()
