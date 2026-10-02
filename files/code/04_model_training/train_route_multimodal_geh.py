from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from area_baseline import build_area_graph
from baseline_pipeline import set_global_seed, write_json
from experiment_metrics import count_trainable_parameters, evaluate_regression_arrays
from timestamp_conditioned_siteyear_slot_residual_mlp import (
    build_long_run_array,
    build_profile_array,
    fill_profile_nans,
    make_date_arrays,
)
from victoria_holidays import add_victoria_holiday_features


SLOT_COUNT = 96
SLOT_COLS = [f"V{slot:02d}" for slot in range(SLOT_COUNT)]
HOLIDAY_FEATURE_COLS = [
    "is_public_holiday",
    "is_restricted_trading_day",
    "is_holiday_eve",
    "is_day_before_holiday",
    "is_melbourne_cup",
    "is_afl_grand_final_friday",
    "is_easter_period",
    "is_christmas_period",
    "is_new_year_period",
    "is_australia_day",
    "is_labour_day",
    "is_anzac_day",
    "is_monarch_birthday",
    "is_oneoff_national_event",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced GEH-oriented route trainer with slot/day models.")
    parser.add_argument("--slot-wide-parquet", type=Path, required=True)
    parser.add_argument("--study-area-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-years", nargs="+", type=int, required=True)
    parser.add_argument("--val-years", nargs="+", type=int, required=True)
    parser.add_argument("--test-years", nargs="+", type=int, required=True)
    parser.add_argument("--visual-feature-csv", type=Path, default=None)
    parser.add_argument("--net-embedding-csv", type=Path, default=None)
    parser.add_argument("--traffic-embedding-csv", type=Path, default=None)
    parser.add_argument("--adjacency-csv", type=Path, default=None)
    parser.add_argument("--model-family", choices=["slot_resmlp", "day_tcn"], default="day_tcn")
    parser.add_argument("--target-mode", choices=["residual", "log_flow", "log_ratio"], default="log_ratio")
    parser.add_argument("--loss-mode", choices=["huber", "huber_geh", "huber_geh_hourly"], default="huber_geh")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--geh-weight", type=float, default=0.20)
    parser.add_argument("--hourly-weight", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=1500000, help="Slot model only.")
    parser.add_argument("--max-val-samples", type=int, default=300000, help="Slot model only.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Slot model only.")
    parser.add_argument("--max-train-days", type=int, default=120000, help="Day-TCN only.")
    parser.add_argument("--max-val-days", type=int, default=None, help="Day-TCN only.")
    parser.add_argument("--max-test-days", type=int, default=None, help="Day-TCN only.")
    parser.add_argument("--base-input", action="store_true")
    parser.add_argument("--use-lag-features", action="store_true")
    parser.add_argument("--use-holiday-features", action="store_true")
    parser.add_argument("--disable-time-features", action="store_true")
    parser.add_argument("--graph-smoothing", action="store_true")
    parser.add_argument("--blend-base", action="store_true")
    parser.add_argument("--fixed-blend-alpha", type=float, default=None)
    parser.add_argument("--fixed-graph-alpha", type=float, default=None)
    parser.add_argument("--checkpoint-in", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--run-name", type=str, default="route_advanced")
    return parser.parse_args()


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


@dataclass
class PreparedData:
    site_ids: list[int]
    site_to_idx: dict[int, int]
    keep_years: list[int]
    year_to_idx: dict[int, int]
    wide: pd.DataFrame
    values: np.ndarray
    valid_mask: np.ndarray
    date_arrays: dict[str, np.ndarray]
    dow_profile: np.ndarray
    month_profile: np.ndarray
    long_run_profile: np.ndarray
    static_features: np.ndarray
    static_feature_dim: int
    flow_feature_scale: float
    lag_row_indices: dict[int, np.ndarray]


def build_lag_row_indices(wide: pd.DataFrame, lags: list[int]) -> dict[int, np.ndarray]:
    anchor = wide[["SITE_NO", "date"]].copy()
    anchor["row_idx"] = np.arange(len(anchor), dtype=np.int32)
    out: dict[int, np.ndarray] = {}
    for lag in lags:
        shifted = anchor.rename(columns={"date": "lag_date", "row_idx": f"lag_{lag}_row_idx"})
        query = anchor[["SITE_NO", "date"]].copy()
        query["lag_date"] = query["date"] - pd.to_timedelta(lag, unit="D")
        merged = query.merge(shifted[["SITE_NO", "lag_date", f"lag_{lag}_row_idx"]], on=["SITE_NO", "lag_date"], how="left")
        out[lag] = merged[f"lag_{lag}_row_idx"].fillna(-1).to_numpy(dtype=np.int32)
    return out


def resolve_lag_vector(
    values: np.ndarray,
    lag_row_indices: dict[int, np.ndarray],
    row_idx: int,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lag_vectors: list[np.ndarray] = []
    for lag in (1, 7, 14):
        prev_idx = int(lag_row_indices[lag][row_idx]) if lag in lag_row_indices else -1
        if prev_idx >= 0:
            prev = values[prev_idx].astype(np.float32)
            prev = np.where(np.isfinite(prev), prev, fallback).astype(np.float32)
            lag_vectors.append(prev)
        else:
            lag_vectors.append(fallback.astype(np.float32, copy=True))
    lag1, lag7, lag14 = lag_vectors
    lag_mean = (0.5 * (lag1 + lag7)).astype(np.float32)
    same_dow_mean = ((lag7 + lag14) / 2.0).astype(np.float32)
    return lag1, lag7, lag14, lag_mean, same_dow_mean


def resolve_lag_scalar(
    values: np.ndarray,
    lag_row_indices: dict[int, np.ndarray],
    row_indices: np.ndarray,
    slot_indices: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lag1 = fallback.astype(np.float32, copy=True)
    lag7 = fallback.astype(np.float32, copy=True)
    lag14 = fallback.astype(np.float32, copy=True)
    if 1 in lag_row_indices:
        prev1 = lag_row_indices[1][row_indices]
        valid1 = prev1 >= 0
        if np.any(valid1):
            lag1_vals = values[prev1[valid1], slot_indices[valid1]].astype(np.float32)
            lag1[valid1] = np.where(np.isfinite(lag1_vals), lag1_vals, fallback[valid1]).astype(np.float32)
    if 7 in lag_row_indices:
        prev7 = lag_row_indices[7][row_indices]
        valid7 = prev7 >= 0
        if np.any(valid7):
            lag7_vals = values[prev7[valid7], slot_indices[valid7]].astype(np.float32)
            lag7[valid7] = np.where(np.isfinite(lag7_vals), lag7_vals, fallback[valid7]).astype(np.float32)
    if 14 in lag_row_indices:
        prev14 = lag_row_indices[14][row_indices]
        valid14 = prev14 >= 0
        if np.any(valid14):
            lag14_vals = values[prev14[valid14], slot_indices[valid14]].astype(np.float32)
            lag14[valid14] = np.where(np.isfinite(lag14_vals), lag14_vals, fallback[valid14]).astype(np.float32)
    lag_mean = (0.5 * (lag1 + lag7)).astype(np.float32)
    same_dow_mean = ((lag7 + lag14) / 2.0).astype(np.float32)
    return lag1.astype(np.float32), lag7.astype(np.float32), lag14.astype(np.float32), lag_mean, same_dow_mean


def prepare_data(args: argparse.Namespace) -> PreparedData:
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

    dow_profile_raw = build_profile_array(
        site_idx_arr=train_site_idx_arr,
        group_idx_arr=date_arrays["dow"][train_row_mask],
        values=train_values,
        n_sites=len(site_ids),
        n_groups=7,
    )
    month_profile_raw = build_profile_array(
        site_idx_arr=train_site_idx_arr,
        group_idx_arr=date_arrays["month"][train_row_mask],
        values=train_values,
        n_sites=len(site_ids),
        n_groups=12,
    )
    long_run_raw = build_long_run_array(train_site_idx_arr, train_values, len(site_ids))
    fallback = np.nanmean(train_values, axis=0, keepdims=True).astype(np.float32)
    long_run_profile = np.where(np.isfinite(long_run_raw), long_run_raw, fallback).astype(np.float32)
    dow_profile = fill_profile_nans(dow_profile_raw, long_run_profile)
    month_profile = fill_profile_nans(month_profile_raw, long_run_profile)
    flow_feature_scale = float(np.nanstd(train_values))
    if not np.isfinite(flow_feature_scale) or flow_feature_scale < 1e-3:
        flow_feature_scale = 1.0
    lag_row_indices = build_lag_row_indices(wide, [1, 7, 14]) if args.use_lag_features else {}

    year_to_idx = {year: idx for idx, year in enumerate(keep_years)}
    visual_arr, visual_cols = load_siteyear_table(args.visual_feature_csv, site_ids, keep_years)
    net_arr, net_cols = load_siteyear_table(args.net_embedding_csv, site_ids, keep_years)
    traffic_arr, traffic_cols = load_siteyear_table(args.traffic_embedding_csv, site_ids, keep_years)
    static_features = np.concatenate([visual_arr, net_arr, traffic_arr], axis=-1).astype(np.float32)
    if static_features.shape[-1] > 0:
        train_year_idx = [year_to_idx[year] for year in args.train_years if year in year_to_idx]
        train_flat = static_features[:, train_year_idx, :].reshape(-1, static_features.shape[-1])
        scaler = StandardScaler()
        scaler.fit(train_flat)
        static_features = scaler.transform(static_features.reshape(-1, static_features.shape[-1])).reshape(static_features.shape).astype(
            np.float32
        )

    return PreparedData(
        site_ids=site_ids,
        site_to_idx=site_to_idx,
        keep_years=keep_years,
        year_to_idx=year_to_idx,
        wide=wide,
        values=values,
        valid_mask=valid_mask,
        date_arrays=date_arrays,
        dow_profile=dow_profile,
        month_profile=month_profile,
        long_run_profile=long_run_profile,
        static_features=static_features,
        static_feature_dim=int(static_features.shape[-1]),
        flow_feature_scale=flow_feature_scale,
        lag_row_indices=lag_row_indices,
    )


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    return payload


class SlotDataset(Dataset):
    def __init__(
        self,
        *,
        x: np.ndarray,
        site_idx: np.ndarray,
        y_true: np.ndarray,
        base: np.ndarray,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.site_idx = torch.tensor(site_idx, dtype=torch.long)
        self.y_true = torch.tensor(y_true, dtype=torch.float32)
        self.base = torch.tensor(base, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_true)

    def __getitem__(self, idx: int):
        return self.x[idx], self.site_idx[idx], self.y_true[idx], self.base[idx]


class DaySequenceDataset(Dataset):
    def __init__(
        self,
        *,
        data: PreparedData,
        row_indices: np.ndarray,
        base_input: bool,
        use_time_features: bool,
    ):
        self.data = data
        self.row_indices = row_indices.astype(np.int32)
        self.base_input = base_input
        self.use_time_features = use_time_features
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
        month_base = (self.data.month_profile[site_idx, month].astype(np.float32) / self.data.flow_feature_scale)
        long_base = (self.data.long_run_profile[site_idx].astype(np.float32) / self.data.flow_feature_scale)
        static = self.data.static_features[site_idx, year_idx].astype(np.float32)
        base_input_arr = base / self.data.flow_feature_scale
        lag1, lag7, lag14, lag_mean, same_dow_mean = resolve_lag_vector(self.data.values, self.data.lag_row_indices, row_idx, base)
        lag1_s = lag1 / self.data.flow_feature_scale
        lag7_s = lag7 / self.data.flow_feature_scale
        lag14_s = lag14 / self.data.flow_feature_scale
        lag_mean_s = lag_mean / self.data.flow_feature_scale
        same_dow_mean_s = same_dow_mean / self.data.flow_feature_scale
        lag1_delta = (lag1 - base) / self.data.flow_feature_scale
        lag7_delta = (lag7 - base) / self.data.flow_feature_scale
        lag14_delta = (lag14 - base) / self.data.flow_feature_scale

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
                month_base,
                long_base,
            ]
            if self.base_input:
                dyn_parts.append(base_input_arr)
            if self.data.date_arrays.get(HOLIDAY_FEATURE_COLS[0]) is not None:
                for column in HOLIDAY_FEATURE_COLS:
                    dyn_parts.append(np.full(SLOT_COUNT, float(self.data.date_arrays[column][row_idx]), dtype=np.float32))
            if self.data.lag_row_indices:
                dyn_parts.extend([lag1_s, lag7_s, lag14_s, lag_mean_s, same_dow_mean_s, lag1_delta, lag7_delta, lag14_delta])
        else:
            dyn_parts = [np.zeros(SLOT_COUNT, dtype=np.float32)]
        x_dyn = np.stack(dyn_parts, axis=1).astype(np.float32)
        return {
            "x_dyn": torch.tensor(x_dyn, dtype=torch.float32),
            "x_static": torch.tensor(static, dtype=torch.float32),
            "site_idx": torch.tensor(site_idx, dtype=torch.long),
            "y_true": torch.tensor(np.nan_to_num(y_true, nan=0.0), dtype=torch.float32),
            "base": torch.tensor(base, dtype=torch.float32),
            "valid": torch.tensor(valid, dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SlotResMLP(nn.Module):
    def __init__(self, *, input_dim: int, site_count: int, hidden_dim: int, emb_dim: int, layers: int, dropout: float):
        super().__init__()
        self.site_emb = nn.Embedding(site_count, emb_dim)
        self.in_proj = nn.Linear(input_dim + emb_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(layers)])
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, site_idx: torch.Tensor) -> torch.Tensor:
        site_vec = self.site_emb(site_idx)
        h = F.silu(self.in_proj(torch.cat([x, site_vec], dim=-1)))
        for block in self.blocks:
            h = block(h)
        return self.out(h).squeeze(-1)


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.dropout(self.act(self.norm1(self.conv1(x))))
        h = self.dropout(self.act(self.norm2(self.conv2(h))))
        return self.act(h + residual)


class DayTCN(nn.Module):
    def __init__(
        self,
        *,
        dyn_dim: int,
        static_dim: int,
        site_count: int,
        hidden_dim: int,
        emb_dim: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.site_emb = nn.Embedding(site_count, emb_dim)
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim + emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.dyn_proj = nn.Linear(dyn_dim, hidden_dim)
        dilations = [2**idx for idx in range(layers)]
        self.blocks = nn.ModuleList([TemporalConvBlock(hidden_dim, kernel_size, dilation, dropout) for dilation in dilations])
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor, site_idx: torch.Tensor) -> torch.Tensor:
        # x_dyn: [B, T, F]
        site_vec = self.site_emb(site_idx)
        static_h = self.static_proj(torch.cat([x_static, site_vec], dim=-1)).unsqueeze(1)
        h = self.dyn_proj(x_dyn) + static_h
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        out = self.head(h).squeeze(1)
        return out


def geh_surrogate(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    yt = y_true * 4.0
    yp = torch.clamp(y_pred, min=0.0) * 4.0
    denom = torch.clamp(yt + yp, min=1e-4)
    geh = torch.sqrt(torch.clamp((2.0 * (yp - yt) ** 2) / denom, min=0.0))
    return (geh * mask).sum() / mask.sum().clamp_min(1.0)


def hourly_huber_loss(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size = y_true.shape[0]
    yt = y_true.view(batch_size, 24, 4).sum(dim=-1)
    yp = torch.clamp(y_pred, min=0.0).view(batch_size, 24, 4).sum(dim=-1)
    valid = (mask.view(batch_size, 24, 4).sum(dim=-1) > 0).float()
    diff = torch.abs(yp - yt)
    delta = 80.0
    huber = torch.where(diff <= delta, 0.5 * diff * diff / delta, diff - 0.5 * delta)
    return (huber * valid).sum() / valid.sum().clamp_min(1.0)


def reconstruct_flow(
    raw_output: torch.Tensor,
    base: torch.Tensor,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "residual":
        return torch.clamp(base + raw_output, min=0.0)
    if target_mode == "log_flow":
        return torch.clamp(torch.expm1(torch.clamp(raw_output, min=-6.0, max=6.0)), min=0.0)
    if target_mode == "log_ratio":
        return torch.clamp(
            torch.expm1(torch.log1p(torch.clamp(base, min=0.0)) + torch.clamp(raw_output, min=-6.0, max=6.0)),
            min=0.0,
        )
    raise ValueError(f"Unknown target_mode: {target_mode}")


def build_slot_arrays(
    *,
    data: PreparedData,
    row_indices: np.ndarray,
    slot_indices: np.ndarray,
    base_input: bool,
    use_time_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    site_idx = data.wide.iloc[row_indices]["site_idx"].to_numpy(dtype=np.int64)
    year = data.date_arrays["year"][row_indices].astype(int)
    year_idx = np.asarray([data.year_to_idx[int(v)] for v in year], dtype=np.int16)
    dow = data.date_arrays["dow"][row_indices].astype(int)
    month = data.date_arrays["month"][row_indices].astype(int)
    doy_frac = (data.date_arrays["doy"][row_indices].astype(np.float32) + slot_indices.astype(np.float32) / SLOT_COUNT) / 366.0
    slot_frac = slot_indices.astype(np.float32) / SLOT_COUNT
    base = data.dow_profile[site_idx, dow, slot_indices].astype(np.float32)
    month_base = (data.month_profile[site_idx, month, slot_indices].astype(np.float32) / data.flow_feature_scale)
    long_base = (data.long_run_profile[site_idx, slot_indices].astype(np.float32) / data.flow_feature_scale)
    static = data.static_features[site_idx, year_idx].astype(np.float32)
    lag1, lag7, lag14, lag_mean, same_dow_mean = resolve_lag_scalar(data.values, data.lag_row_indices, row_indices, slot_indices, base)
    if use_time_features:
        parts = [
            data.date_arrays["dow_sin"][row_indices][:, None],
            data.date_arrays["dow_cos"][row_indices][:, None],
            data.date_arrays["month_sin"][row_indices][:, None],
            data.date_arrays["month_cos"][row_indices][:, None],
            np.sin(2 * np.pi * doy_frac)[:, None].astype(np.float32),
            np.cos(2 * np.pi * doy_frac)[:, None].astype(np.float32),
            data.date_arrays["is_weekend"][row_indices][:, None],
            data.date_arrays["year_norm"][row_indices][:, None],
            np.sin(2 * np.pi * slot_frac)[:, None].astype(np.float32),
            np.cos(2 * np.pi * slot_frac)[:, None].astype(np.float32),
            month_base[:, None],
            long_base[:, None],
        ]
        if base_input:
            parts.append((base / data.flow_feature_scale)[:, None])
        if data.date_arrays.get(HOLIDAY_FEATURE_COLS[0]) is not None:
            for column in HOLIDAY_FEATURE_COLS:
                parts.append(data.date_arrays[column][row_indices][:, None].astype(np.float32))
        if data.lag_row_indices:
            parts.extend(
                [
                    (lag1 / data.flow_feature_scale)[:, None],
                    (lag7 / data.flow_feature_scale)[:, None],
                    (lag14 / data.flow_feature_scale)[:, None],
                    (lag_mean / data.flow_feature_scale)[:, None],
                    (same_dow_mean / data.flow_feature_scale)[:, None],
                    ((lag1 - base) / data.flow_feature_scale)[:, None],
                    ((lag7 - base) / data.flow_feature_scale)[:, None],
                    ((lag14 - base) / data.flow_feature_scale)[:, None],
                ]
            )
    else:
        parts = [np.zeros((len(row_indices), 1), dtype=np.float32)]
    if static.shape[1] > 0:
        parts.append(static)
    x = np.concatenate(parts, axis=1).astype(np.float32)
    y_true = data.values[row_indices, slot_indices].astype(np.float32)
    return x, site_idx, y_true, base


def best_blend_with_base(y_true: np.ndarray, y_pred: np.ndarray, base: np.ndarray,
                        alpha_min: float = 0.60, alpha_max: float = 1.10
                        ) -> tuple[np.ndarray, dict[str, float]]:
    """Pick the blend against the day-of-week base that scores best on validation.

    The range defaults to the grid this used before it was settable, so existing runs are
    unaffected. The step is held at 0.05.
    """
    best = None
    best_score = None
    best_params = {"blend_alpha": 1.0}
    steps = max(2, int(round((alpha_max - alpha_min) / 0.05)) + 1)
    for alpha in np.linspace(alpha_min, alpha_max, num=steps):
        cand = np.clip(alpha * y_pred + (1.0 - alpha) * base, 0.0, None)
        metrics = evaluate_regression_arrays(y_true, cand, geh_scale_factor=4.0)
        score = (metrics["GEH_median"], metrics["MAE"])
        if best_score is None or score < best_score:
            best_score = score
            best = cand
            best_params = {"blend_alpha": float(alpha)}
    assert best is not None
    return best.astype(np.float32), best_params


def apply_graph_smoothing(
    pred_df: pd.DataFrame,
    *,
    adjacency: np.ndarray,
    site_ids: list[int],
    alpha: float,
) -> np.ndarray:
    site_to_pos = {site: idx for idx, site in enumerate(site_ids)}
    df = pred_df.copy()
    df["site_pos"] = df["SITE_NO"].map(site_to_pos).astype(int)
    out = np.zeros(len(df), dtype=np.float32)
    for _, group in df.groupby(["date", "slot_idx"], sort=False):
        order = np.argsort(group["site_pos"].to_numpy(dtype=np.int32))
        grp = group.iloc[order]
        pos = grp["site_pos"].to_numpy(dtype=np.int32)
        pred_vec = grp["y_pred"].to_numpy(dtype=np.float32)
        smoothed = pred_vec.copy()
        smoothed = (1.0 - alpha) * pred_vec + alpha * adjacency[np.ix_(pos, pos)] @ pred_vec
        out[grp.index.to_numpy(dtype=np.int64)] = np.clip(smoothed, 0.0, None)
    return out


def _smoothing_layout(pred_df: pd.DataFrame, site_ids: list[int]):
    """Row -> (group, site) placement, which every alpha in the search shares.

    Building it once turns the search from ten walks over 34,944 groups into ten matrix
    products. The values are identical either way; only the bookkeeping is hoisted.
    """
    site_to_pos = {site: idx for idx, site in enumerate(site_ids)}
    pos = pred_df["SITE_NO"].map(site_to_pos).to_numpy()
    group = pd.factorize(pd.MultiIndex.from_arrays(
        [pred_df["date"].to_numpy(), pred_df["slot_idx"].to_numpy()]), sort=False)[0]
    dense = np.zeros((int(group.max()) + 1, len(site_ids)), dtype=np.float32)
    dense[group, pos.astype(np.int64)] = pred_df["y_pred"].to_numpy(dtype=np.float32)
    return group, pos.astype(np.int64), dense


def best_graph_smoothing(
    pred_df: pd.DataFrame,
    *,
    adjacency: np.ndarray,
    site_ids: list[int],
) -> tuple[np.ndarray, dict[str, float]]:
    best = pred_df["y_pred"].to_numpy(dtype=np.float32)
    best_score = evaluate_regression_arrays(pred_df["y_true"], best, geh_scale_factor=4.0)
    best_params = {"graph_alpha": 0.0}
    group, pos, dense = _smoothing_layout(pred_df, site_ids)
    neighbour = dense @ adjacency.T.astype(np.float32)
    for alpha in np.linspace(0.05, 0.45, num=9):
        cand = np.clip(((1.0 - float(alpha)) * dense
                        + float(alpha) * neighbour)[group, pos], 0.0, None)
        metrics = evaluate_regression_arrays(pred_df["y_true"], cand, geh_scale_factor=4.0)
        score = (metrics["GEH_median"], metrics["MAE"])
        best_current = (best_score["GEH_median"], best_score["MAE"])
        if score < best_current:
            best = cand
            best_score = metrics
            best_params = {"graph_alpha": float(alpha)}
    return best.astype(np.float32), best_params


def gradients_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def load_adjacency_from_csv(adjacency_csv: Path, site_ids: list[int]) -> np.ndarray:
    df = pd.read_csv(adjacency_csv, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    df = df.reindex(index=site_ids, columns=site_ids).fillna(0.0)
    arr = df.to_numpy(dtype=np.float32)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 1e-6] = 1.0
    return arr / row_sums


def train_slot_model(args: argparse.Namespace, data: PreparedData) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    row_years = data.date_arrays["year"]
    train_row_mask = np.isin(row_years, np.asarray(args.train_years))
    val_row_mask = np.isin(row_years, np.asarray(args.val_years))
    test_row_mask = np.isin(row_years, np.asarray(args.test_years))

    def sample_pairs(mask: np.ndarray, max_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
        row_positions = np.flatnonzero(mask)
        valid_pairs = np.argwhere(data.valid_mask[row_positions])
        if max_samples is not None and len(valid_pairs) > max_samples:
            chosen = rng.choice(len(valid_pairs), size=max_samples, replace=False)
            valid_pairs = valid_pairs[chosen]
        row_idx = row_positions[valid_pairs[:, 0]].astype(np.int32)
        slot_idx = valid_pairs[:, 1].astype(np.int16)
        order = np.lexsort((slot_idx, row_idx))
        return row_idx[order], slot_idx[order]

    train_rows, train_slots = sample_pairs(train_row_mask, args.max_train_samples)
    val_rows, val_slots = sample_pairs(val_row_mask, args.max_val_samples)
    test_rows, test_slots = sample_pairs(test_row_mask, args.max_test_samples)

    use_time_features = not args.disable_time_features
    x_train, site_train, y_train, base_train = build_slot_arrays(
        data=data,
        row_indices=train_rows,
        slot_indices=train_slots,
        base_input=args.base_input,
        use_time_features=use_time_features,
    )
    x_val, site_val, y_val, base_val = build_slot_arrays(
        data=data,
        row_indices=val_rows,
        slot_indices=val_slots,
        base_input=args.base_input,
        use_time_features=use_time_features,
    )
    x_test, site_test, y_test, base_test = build_slot_arrays(
        data=data,
        row_indices=test_rows,
        slot_indices=test_slots,
        base_input=args.base_input,
        use_time_features=use_time_features,
    )

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = x_scaler.transform(x_val).astype(np.float32)
    x_test_s = x_scaler.transform(x_test).astype(np.float32)

    train_ds = SlotDataset(x=x_train_s, site_idx=site_train, y_true=y_train, base=base_train)
    val_ds = SlotDataset(x=x_val_s, site_idx=site_val, y_true=y_val, base=base_val)
    test_ds = SlotDataset(x=x_test_s, site_idx=site_test, y_true=y_test, base=base_test)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(str(args.device).startswith("cuda")),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = SlotResMLP(
        input_dim=x_train_s.shape[1],
        site_count=len(data.site_ids),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(args.device)
    initial_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.HuberLoss(delta=60.0)

    best_state = None
    best_score = None
    history: list[dict[str, float]] = []
    stale = 0
    if args.eval_only:
        if args.checkpoint_in is None:
            raise ValueError("--eval-only requires --checkpoint-in")
        best_state = load_checkpoint_state(args.checkpoint_in)

    for epoch in range(1, args.epochs + 1) if not args.eval_only else []:
        model.train()
        train_losses = []
        for xb, sb, yb, bb in train_loader:
            xb = xb.to(args.device)
            sb = sb.to(args.device)
            yb = yb.to(args.device)
            bb = bb.to(args.device)
            optimizer.zero_grad()
            raw = model(xb, sb)
            pred = reconstruct_flow(raw, bb, args.target_mode)
            loss = huber(pred, yb)
            if args.loss_mode in {"huber_geh", "huber_geh_hourly"}:
                loss = loss + args.geh_weight * geh_surrogate(yb, pred, torch.ones_like(yb))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_pred_parts: list[np.ndarray] = []
        with torch.no_grad():
            for xb, sb, _, bb in val_loader:
                raw = model(xb.to(args.device), sb.to(args.device))
                pred = reconstruct_flow(raw, bb.to(args.device), args.target_mode)
                val_pred_parts.append(pred.detach().cpu().numpy().astype(np.float32))
        val_pred = np.concatenate(val_pred_parts)
        if not np.isfinite(val_pred).any():
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_GEҲ_median": float("inf"), "nan_failure": 1.0})
            stale += 1
            if stale >= args.patience:
                break
            continue
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1e6, neginf=0.0)
        val_metrics = evaluate_regression_arrays(y_val, val_pred, geh_scale_factor=4.0)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        score = (val_metrics["GEH_median"], val_metrics["MAE"])
        if best_score is None or score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1  # type: ignore[has-type]
        if stale >= args.patience:
            break

    if best_state is None:
        best_state = initial_state
    model.load_state_dict(best_state)

    def predict(loader: DataLoader) -> np.ndarray:
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for xb, sb, _, bb in loader:
                raw = model(xb.to(args.device), sb.to(args.device))
                pred = reconstruct_flow(raw, bb.to(args.device), args.target_mode)
                preds.append(pred.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(preds)

    val_pred = predict(val_loader)
    test_pred = predict(test_loader)
    val_df = pd.DataFrame(
        {
            "SITE_NO": data.wide.iloc[val_rows]["SITE_NO"].to_numpy(dtype=np.int32),
            "date": data.wide.iloc[val_rows]["date"].to_numpy(),
            "year": data.date_arrays["year"][val_rows].astype(np.int16),
            "slot_idx": val_slots.astype(np.int16),
            "y_true": y_val.astype(np.float32),
            "y_pred": val_pred.astype(np.float32),
            "base_dow_slot_profile": base_val.astype(np.float32),
        }
    )
    test_df = pd.DataFrame(
        {
            "SITE_NO": data.wide.iloc[test_rows]["SITE_NO"].to_numpy(dtype=np.int32),
            "date": data.wide.iloc[test_rows]["date"].to_numpy(),
            "year": data.date_arrays["year"][test_rows].astype(np.int16),
            "slot_idx": test_slots.astype(np.int16),
            "y_true": y_test.astype(np.float32),
            "y_pred": test_pred.astype(np.float32),
            "base_dow_slot_profile": base_test.astype(np.float32),
        }
    )
    return {
        "history": history,
        "val_df": val_df,
        "test_df": test_df,
        "model": model,
        "model_summary": {
            "input_dim": int(x_train_s.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "emb_dim": int(args.emb_dim),
            "trainable_params": count_trainable_parameters(model),
        },
    }


def train_day_tcn(args: argparse.Namespace, data: PreparedData) -> dict[str, object]:
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
    train_ds = DaySequenceDataset(data=data, row_indices=train_rows, base_input=args.base_input, use_time_features=use_time_features)
    val_ds = DaySequenceDataset(data=data, row_indices=val_rows, base_input=args.base_input, use_time_features=use_time_features)
    test_ds = DaySequenceDataset(data=data, row_indices=test_rows, base_input=args.base_input, use_time_features=use_time_features)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(str(args.device).startswith("cuda")),
        "persistent_workers": bool(int(args.num_workers) > 0),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    dyn_dim = 1 if args.disable_time_features else (13 if args.base_input else 12) + (len(HOLIDAY_FEATURE_COLS) if args.use_holiday_features else 0) + (8 if args.use_lag_features else 0)
    model = DayTCN(
        dyn_dim=dyn_dim,
        static_dim=data.static_feature_dim,
        site_count=len(data.site_ids),
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
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
        train_losses = []
        for batch in train_loader:
            x_dyn = batch["x_dyn"].to(args.device)
            x_static = batch["x_static"].to(args.device)
            site_idx = batch["site_idx"].to(args.device)
            y_true = batch["y_true"].to(args.device)
            base = batch["base"].to(args.device)
            valid = batch["valid"].to(args.device)
            optimizer.zero_grad()
            raw = model(x_dyn, x_static, site_idx)
            pred = reconstruct_flow(raw, base, args.target_mode)
            mask = valid
            loss = huber(pred * mask, y_true * mask)
            if args.loss_mode in {"huber_geh", "huber_geh_hourly"}:
                loss = loss + args.geh_weight * geh_surrogate(y_true, pred, mask)
            if args.loss_mode == "huber_geh_hourly":
                loss = loss + args.hourly_weight * hourly_huber_loss(y_true, pred, mask)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_pred_parts: list[np.ndarray] = []
        val_true_parts: list[np.ndarray] = []
        val_base_parts: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                raw = model(batch["x_dyn"].to(args.device), batch["x_static"].to(args.device), batch["site_idx"].to(args.device))
                pred = reconstruct_flow(raw, batch["base"].to(args.device), args.target_mode)
                val_pred_parts.append(pred.detach().cpu().numpy().astype(np.float32))
                val_true_parts.append(batch["y_true"].numpy().astype(np.float32))
                val_base_parts.append(batch["base"].numpy().astype(np.float32))
        val_pred = np.concatenate(val_pred_parts).reshape(-1)
        val_true = np.concatenate(val_true_parts).reshape(-1)
        if not np.isfinite(val_pred).any():
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_GEH_median": float("inf"), "nan_failure": 1.0})
            stale += 1
            if stale >= args.patience:
                break
            continue
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1e6, neginf=0.0)
        val_metrics = evaluate_regression_arrays(val_true, val_pred, geh_scale_factor=4.0)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
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

    def predict_days(loader: DataLoader, row_indices: np.ndarray) -> pd.DataFrame:
        pred_rows: list[pd.DataFrame] = []
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                raw = model(batch["x_dyn"].to(args.device), batch["x_static"].to(args.device), batch["site_idx"].to(args.device))
                pred = reconstruct_flow(raw, batch["base"].to(args.device), args.target_mode).detach().cpu().numpy().astype(np.float32)
                y_true = batch["y_true"].numpy().astype(np.float32)
                base = batch["base"].numpy().astype(np.float32)
                bs = pred.shape[0]
                rows = row_indices[cursor : cursor + bs]
                cursor += bs
                for local_idx, row_idx in enumerate(rows):
                    meta_row = data.wide.iloc[int(row_idx)]
                    date_val = pd.Timestamp(meta_row["date"])
                    df = pd.DataFrame(
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
                    pred_rows.append(df)
        return pd.concat(pred_rows, ignore_index=True)

    val_df = predict_days(val_loader, val_rows)
    test_df = predict_days(test_loader, test_rows)
    return {
        "history": history,
        "val_df": val_df,
        "test_df": test_df,
        "model": model,
        "model_summary": {
            "dyn_dim": int(dyn_dim),
            "static_dim": int(data.static_feature_dim),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "kernel_size": int(args.kernel_size),
            "emb_dim": int(args.emb_dim),
            "trainable_params": count_trainable_parameters(model),
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    torch.manual_seed(args.seed)

    data = prepare_data(args)
    if args.model_family == "slot_resmlp":
        result = train_slot_model(args, data)
    else:
        result = train_day_tcn(args, data)

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
            )
            posthoc.update(params)
            alpha = params["blend_alpha"]
        val_df["y_pred"] = blended_val
        test_df["y_pred"] = np.clip(
            alpha * test_df["y_pred"].to_numpy(dtype=np.float32) + (1.0 - alpha) * test_df["base_dow_slot_profile"].to_numpy(dtype=np.float32),
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
                test_df["y_pred"] = apply_graph_smoothing(
                    test_df,
                    adjacency=adjacency,
                    site_ids=data.site_ids,
                    alpha=float(params["graph_alpha"]),
                )
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
            "val_years": args.val_years,
            "test_years": args.test_years,
            "model_family": args.model_family,
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
            "base_input": bool(args.base_input),
            "use_lag_features": bool(args.use_lag_features),
            "use_holiday_features": bool(args.use_holiday_features),
            "time_features_enabled": bool(not args.disable_time_features),
            "blend_base": bool(args.blend_base),
            "graph_smoothing": bool(args.graph_smoothing),
            "fixed_blend_alpha": args.fixed_blend_alpha,
            "fixed_graph_alpha": args.fixed_graph_alpha,
            "eval_only": bool(args.eval_only),
            "checkpoint_in": str(args.checkpoint_in) if args.checkpoint_in else None,
            "checkpoint_out": str(run_dir / "model_state.pt"),
            "posthoc": posthoc,
            "static_feature_dim": int(data.static_feature_dim),
            "model_summary": result["model_summary"],
        },
    )
    print(json.dumps({"val_metrics": val_metrics, "test_metrics": test_metrics, "posthoc": posthoc}, indent=2))


if __name__ == "__main__":
    main()
