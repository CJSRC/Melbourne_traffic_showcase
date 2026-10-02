from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from baseline_pipeline import set_global_seed, write_json
from experiment_metrics import count_trainable_parameters, evaluate_regression_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timestamp-conditioned slot-level residual MLP with site-year semantic features"
    )
    parser.add_argument("--slot-wide-parquet", type=Path, required=True)
    parser.add_argument("--study-area-csv", type=Path, required=True)
    parser.add_argument("--siteyear-feature-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-years", nargs="+", type=int, required=True)
    parser.add_argument("--val-years", nargs="+", type=int, required=True)
    parser.add_argument("--test-years", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=1500000)
    parser.add_argument("--max-val-samples", type=int, default=300000)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()

class RowDataset(Dataset):
    def __init__(self, x: np.ndarray, site_idx: np.ndarray, residual_y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.site_idx = torch.tensor(site_idx, dtype=torch.long)
        self.y = torch.tensor(residual_y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.site_idx[idx], self.y[idx]


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, site_count: int, emb_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.site_emb = nn.Embedding(site_count, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, site_idx: torch.Tensor) -> torch.Tensor:
        site_vec = self.site_emb(site_idx)
        return self.net(torch.cat([x, site_vec], dim=-1)).squeeze(-1)


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float32)
    valid = den > 0
    out[valid] = (num[valid] / den[valid]).astype(np.float32)
    return out


def build_profile_array(
    site_idx_arr: np.ndarray,
    group_idx_arr: np.ndarray,
    values: np.ndarray,
    n_sites: int,
    n_groups: int,
    slot_count: int = 96,
) -> np.ndarray:
    out = np.full((n_sites, n_groups, slot_count), np.nan, dtype=np.float32)
    for group in range(n_groups):
        group_mask = group_idx_arr == group
        if not group_mask.any():
            continue
        site_idx_group = site_idx_arr[group_mask]
        value_group = values[group_mask]
        for slot in range(slot_count):
            slot_vals = value_group[:, slot]
            valid = np.isfinite(slot_vals)
            if not valid.any():
                continue
            sums = np.bincount(site_idx_group[valid], weights=slot_vals[valid], minlength=n_sites).astype(np.float32)
            counts = np.bincount(site_idx_group[valid], minlength=n_sites).astype(np.float32)
            out[:, group, slot] = safe_divide(sums, counts)
    return out


def build_long_run_array(site_idx_arr: np.ndarray, values: np.ndarray, n_sites: int, slot_count: int = 96) -> np.ndarray:
    out = np.full((n_sites, slot_count), np.nan, dtype=np.float32)
    for slot in range(slot_count):
        slot_vals = values[:, slot]
        valid = np.isfinite(slot_vals)
        if not valid.any():
            continue
        sums = np.bincount(site_idx_arr[valid], weights=slot_vals[valid], minlength=n_sites).astype(np.float32)
        counts = np.bincount(site_idx_arr[valid], minlength=n_sites).astype(np.float32)
        out[:, slot] = safe_divide(sums, counts)
    return out


def fill_profile_nans(profile: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if profile.ndim == 3:
        fallback_exp = fallback[:, None, :]
    else:
        fallback_exp = fallback
    return np.where(np.isfinite(profile), profile, fallback_exp).astype(np.float32)


def make_date_arrays(dates: pd.Series) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    dow = idx.dayofweek.to_numpy(dtype=np.int16)
    month = (idx.month - 1).to_numpy(dtype=np.int16)
    doy = (idx.dayofyear - 1).to_numpy(dtype=np.int16)
    years = idx.year.to_numpy(dtype=np.int16)
    min_year = int(years.min())
    max_year = int(years.max())
    denom = max(max_year - min_year, 1)
    year_norm = ((years - min_year) / denom).astype(np.float32)
    return {
        "dow": dow,
        "month": month,
        "doy": doy,
        "year": years,
        "dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
        "month_sin": np.sin(2 * np.pi * month / 12.0).astype(np.float32),
        "month_cos": np.cos(2 * np.pi * month / 12.0).astype(np.float32),
        "doy_sin": np.sin(2 * np.pi * doy / 366.0).astype(np.float32),
        "doy_cos": np.cos(2 * np.pi * doy / 366.0).astype(np.float32),
        "is_weekend": (dow >= 5).astype(np.float32),
        "year_norm": year_norm,
    }


def build_siteyear_lookup(
    siteyear_features: pd.DataFrame,
    site_ids: list[int],
    keep_years: list[int],
) -> tuple[np.ndarray, list[str], dict[int, int]]:
    feature_cols = [column for column in siteyear_features.columns if column not in {"SITE_NO", "year"}]
    year_to_idx = {year: idx for idx, year in enumerate(sorted(keep_years))}
    site_to_idx = {site: idx for idx, site in enumerate(site_ids)}
    feature_arr = np.zeros((len(site_ids), len(year_to_idx), len(feature_cols)), dtype=np.float32)
    for row in siteyear_features.itertuples(index=False):
        site_no = int(row[0])
        year = int(row[1])
        if site_no not in site_to_idx or year not in year_to_idx:
            continue
        feature_arr[site_to_idx[site_no], year_to_idx[year], :] = np.asarray(row[2:], dtype=np.float32)
    return feature_arr, feature_cols, year_to_idx


def build_sample_indices(
    row_years: np.ndarray,
    valid_mask: np.ndarray,
    target_years: set[int],
    rng: np.random.Generator,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    row_mask = np.isin(row_years, sorted(target_years))
    if not row_mask.any():
        return np.array([], dtype=np.int32), np.array([], dtype=np.int16)
    sub_valid = valid_mask[row_mask]
    sub_row_local, slot_idx = np.nonzero(sub_valid)
    row_idx = np.where(row_mask)[0][sub_row_local].astype(np.int32)
    slot_idx = slot_idx.astype(np.int16)
    if max_samples is not None and len(row_idx) > max_samples:
        choice = rng.choice(len(row_idx), size=max_samples, replace=False)
        row_idx = row_idx[choice]
        slot_idx = slot_idx[choice]
    return row_idx, slot_idx


def build_feature_matrix(
    *,
    row_idx: np.ndarray,
    slot_idx: np.ndarray,
    site_idx_arr: np.ndarray,
    date_arrays: dict[str, np.ndarray],
    dow_profile: np.ndarray,
    month_profile: np.ndarray,
    long_run_profile: np.ndarray,
    siteyear_feature_arr: np.ndarray,
    year_to_idx: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    site_idx = site_idx_arr[row_idx]
    year = date_arrays["year"][row_idx].astype(int)
    dow = date_arrays["dow"][row_idx].astype(int)
    month = date_arrays["month"][row_idx].astype(int)
    doy_frac = (date_arrays["doy"][row_idx].astype(np.float32) + slot_idx.astype(np.float32) / 96.0) / 366.0
    slot_frac = slot_idx.astype(np.float32) / 96.0
    year_idx = np.asarray([year_to_idx[int(v)] for v in year], dtype=np.int16)

    base = dow_profile[site_idx, dow, slot_idx].astype(np.float32)
    month_base = month_profile[site_idx, month, slot_idx].astype(np.float32)
    long_base = long_run_profile[site_idx, slot_idx].astype(np.float32)
    siteyear_feats = siteyear_feature_arr[site_idx, year_idx, :].astype(np.float32)

    feature_parts = [
        date_arrays["dow_sin"][row_idx][:, None],
        date_arrays["dow_cos"][row_idx][:, None],
        date_arrays["month_sin"][row_idx][:, None],
        date_arrays["month_cos"][row_idx][:, None],
        np.sin(2 * np.pi * doy_frac)[:, None].astype(np.float32),
        np.cos(2 * np.pi * doy_frac)[:, None].astype(np.float32),
        date_arrays["is_weekend"][row_idx][:, None],
        date_arrays["year_norm"][row_idx][:, None],
        np.sin(2 * np.pi * slot_frac)[:, None].astype(np.float32),
        np.cos(2 * np.pi * slot_frac)[:, None].astype(np.float32),
        month_base[:, None],
        long_base[:, None],
        siteyear_feats,
    ]
    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return x, site_idx.astype(np.int64), base, month_base


def run_variant(
    *,
    name: str,
    x_train: np.ndarray,
    train_site_idx: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    val_site_idx: np.ndarray,
    y_val: np.ndarray,
    base_val: np.ndarray,
    x_test: np.ndarray,
    test_site_idx: np.ndarray,
    y_test: np.ndarray,
    base_test: np.ndarray,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    site_count: int,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
) -> dict[str, float]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32).reshape(-1)

    x_val_scaled = x_scaler.transform(x_val).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32).reshape(-1)
    x_test_scaled = x_scaler.transform(x_test).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32).reshape(-1)

    train_ds = RowDataset(x_train_scaled, train_site_idx, y_train_scaled)
    val_ds = RowDataset(x_val_scaled, val_site_idx, y_val_scaled)
    test_ds = RowDataset(x_test_scaled, test_site_idx, y_test_scaled)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ResidualMLP(in_dim=x_train.shape[1], site_count=site_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, sb, yb in train_loader:
            xb = xb.to(device)
            sb = sb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, sb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, sb, yb in val_loader:
                pred = model(xb.to(device), sb.to(device))
                val_losses.append(criterion(pred, yb.to(device)).item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pred_scaled = []
    with torch.no_grad():
        for xb, sb, _ in test_loader:
            pred_scaled.append(model(xb.to(device), sb.to(device)).cpu().numpy())

    pred_scaled = np.concatenate(pred_scaled)
    pred_residual = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_pred = np.clip(base_test + pred_residual, 0.0, None)
    y_true = base_test + y_test.astype(np.float32)

    variant_dir = out_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(variant_dir / "training_history.csv", index=False)
    np.save(variant_dir / "y_true.npy", y_true)
    np.save(variant_dir / "y_pred.npy", y_pred)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_scaler_mean": x_scaler.mean_.astype(np.float32),
            "x_scaler_scale": x_scaler.scale_.astype(np.float32),
            "y_scaler_mean": y_scaler.mean_.astype(np.float32),
            "y_scaler_scale": y_scaler.scale_.astype(np.float32),
        },
        variant_dir / "model_artifacts.pt",
    )
    pred_df = test_meta.copy()
    pred_df["y_true"] = y_true.astype(np.float32)
    pred_df["y_pred"] = y_pred.astype(np.float32)
    pred_df["base_dow_slot_profile"] = base_test.astype(np.float32)
    pred_df.to_parquet(variant_dir / "test_predictions.parquet", index=False)

    val_pred_scaled = []
    with torch.no_grad():
        for xb, sb, _ in val_loader:
            val_pred_scaled.append(model(xb.to(device), sb.to(device)).cpu().numpy())
    val_pred_scaled = np.concatenate(val_pred_scaled)
    val_pred_residual = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).reshape(-1).astype(np.float32)
    val_base = base_val.astype(np.float32)
    y_val_true = val_base + y_val.astype(np.float32)
    y_val_pred = np.clip(val_base + val_pred_residual, 0.0, None)
    val_pred_df = val_meta.copy()
    val_pred_df["y_true"] = y_val_true.astype(np.float32)
    val_pred_df["y_pred"] = y_val_pred.astype(np.float32)
    val_pred_df["base_dow_slot_profile"] = val_base.astype(np.float32)
    val_pred_df.to_parquet(variant_dir / "val_predictions.parquet", index=False)
    metrics = evaluate_regression_arrays(y_true, y_pred, geh_scale_factor=4.0)
    write_json(variant_dir / "metrics.json", metrics)
    write_json(
        variant_dir / "run_config.json",
        {
            "variant": name,
            "feature_count": int(x_train.shape[1]),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "seed": seed,
            "train_sample_count": int(len(x_train)),
            "val_sample_count": int(len(x_val)),
            "test_sample_count": int(len(x_test)),
            "geh_scale_factor": 4.0,
            "model_summary": {
                "input_dim": int(x_train.shape[1]),
                "site_emb_dim": 16,
                "hidden_dim": 128,
                "output_dim": 1,
                "trainable_params": count_trainable_parameters(model),
            },
        },
    )
    return metrics


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

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

    slot_cols = [f"V{slot:02d}" for slot in range(96)]
    values = wide[slot_cols].to_numpy(dtype=np.float32).copy()
    values[values < 0] = np.nan
    valid_mask = np.isfinite(values)

    date_arrays = make_date_arrays(wide["date"])
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
    long_run_profile = np.where(np.isfinite(long_run_raw), long_run_raw, np.nanmean(train_values, axis=0, keepdims=True)).astype(
        np.float32
    )
    dow_profile = fill_profile_nans(dow_profile_raw, long_run_profile)
    month_profile = fill_profile_nans(month_profile_raw, long_run_profile)

    siteyear_features = pd.read_csv(args.siteyear_feature_csv)
    siteyear_features["SITE_NO"] = pd.to_numeric(siteyear_features["SITE_NO"], errors="coerce").astype(int)
    siteyear_features["year"] = pd.to_numeric(siteyear_features["year"], errors="coerce").astype(int)
    siteyear_features = siteyear_features[siteyear_features["year"].isin(keep_years)].copy()
    siteyear_feature_arr, siteyear_feature_cols, year_to_idx = build_siteyear_lookup(
        siteyear_features, site_ids, keep_years
    )

    train_row_idx, train_slot_idx = build_sample_indices(
        row_years=row_years,
        valid_mask=valid_mask,
        target_years=set(args.train_years),
        rng=rng,
        max_samples=args.max_train_samples,
    )
    val_row_idx, val_slot_idx = build_sample_indices(
        row_years=row_years,
        valid_mask=valid_mask,
        target_years=set(args.val_years),
        rng=rng,
        max_samples=args.max_val_samples,
    )
    test_row_idx, test_slot_idx = build_sample_indices(
        row_years=row_years,
        valid_mask=valid_mask,
        target_years=set(args.test_years),
        rng=rng,
        max_samples=args.max_test_samples,
    )

    x_train_full, train_site_idx, base_train, month_base_train = build_feature_matrix(
        row_idx=train_row_idx,
        slot_idx=train_slot_idx,
        site_idx_arr=wide["site_idx"].to_numpy(dtype=np.int32),
        date_arrays=date_arrays,
        dow_profile=dow_profile,
        month_profile=month_profile,
        long_run_profile=long_run_profile,
        siteyear_feature_arr=siteyear_feature_arr,
        year_to_idx=year_to_idx,
    )
    x_val_full, val_site_idx, base_val, month_base_val = build_feature_matrix(
        row_idx=val_row_idx,
        slot_idx=val_slot_idx,
        site_idx_arr=wide["site_idx"].to_numpy(dtype=np.int32),
        date_arrays=date_arrays,
        dow_profile=dow_profile,
        month_profile=month_profile,
        long_run_profile=long_run_profile,
        siteyear_feature_arr=siteyear_feature_arr,
        year_to_idx=year_to_idx,
    )
    x_test_full, test_site_idx, base_test, month_base_test = build_feature_matrix(
        row_idx=test_row_idx,
        slot_idx=test_slot_idx,
        site_idx_arr=wide["site_idx"].to_numpy(dtype=np.int32),
        date_arrays=date_arrays,
        dow_profile=dow_profile,
        month_profile=month_profile,
        long_run_profile=long_run_profile,
        siteyear_feature_arr=siteyear_feature_arr,
        year_to_idx=year_to_idx,
    )

    y_train_flow = values[train_row_idx, train_slot_idx].astype(np.float32)
    y_val_flow = values[val_row_idx, val_slot_idx].astype(np.float32)
    y_test_flow = values[test_row_idx, test_slot_idx].astype(np.float32)
    y_train_residual = y_train_flow - base_train
    y_val_residual = y_val_flow - base_val
    y_test_residual = y_test_flow - base_test

    common_feature_count = 12
    x_train_common = x_train_full[:, :common_feature_count]
    x_val_common = x_val_full[:, :common_feature_count]
    x_test_common = x_test_full[:, :common_feature_count]
    val_meta = pd.DataFrame(
        {
            "SITE_NO": wide.iloc[val_row_idx]["SITE_NO"].to_numpy(dtype=np.int32),
            "date": wide.iloc[val_row_idx]["date"].to_numpy(),
            "year": row_years[val_row_idx].astype(np.int16),
            "slot_idx": val_slot_idx.astype(np.int16),
            "slot_label": [f"V{int(v):02d}" for v in val_slot_idx],
        }
    )
    test_meta = pd.DataFrame(
        {
            "SITE_NO": wide.iloc[test_row_idx]["SITE_NO"].to_numpy(dtype=np.int32),
            "date": wide.iloc[test_row_idx]["date"].to_numpy(),
            "year": row_years[test_row_idx].astype(np.int16),
            "slot_idx": test_slot_idx.astype(np.int16),
            "slot_label": [f"V{int(v):02d}" for v in test_slot_idx],
        }
    )

    results = {
        "historical_dow_slot_profile": evaluate_regression_arrays(y_test_flow, base_test, geh_scale_factor=4.0),
        "historical_month_slot_profile": evaluate_regression_arrays(y_test_flow, month_base_test, geh_scale_factor=4.0),
        "long_run_slot_mean": evaluate_regression_arrays(
            y_test_flow,
            long_run_profile[test_site_idx, test_slot_idx].astype(np.float32),
            geh_scale_factor=4.0,
        ),
    }

    results["residual_mlp_slot_no_siteyear"] = run_variant(
        name="residual_mlp_slot_no_siteyear",
        x_train=x_train_common,
        train_site_idx=train_site_idx,
        y_train=y_train_residual,
        x_val=x_val_common,
        val_site_idx=val_site_idx,
        y_val=y_val_residual,
        base_val=base_val,
        x_test=x_test_common,
        test_site_idx=test_site_idx,
        y_test=y_test_residual,
        base_test=base_test,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        site_count=len(site_ids),
        val_meta=val_meta,
        test_meta=test_meta,
    )
    results["residual_mlp_slot_qwen_siteyear"] = run_variant(
        name="residual_mlp_slot_qwen_siteyear",
        x_train=x_train_full,
        train_site_idx=train_site_idx,
        y_train=y_train_residual,
        x_val=x_val_full,
        val_site_idx=val_site_idx,
        y_val=y_val_residual,
        base_val=base_val,
        x_test=x_test_full,
        test_site_idx=test_site_idx,
        y_test=y_test_residual,
        base_test=base_test,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        site_count=len(site_ids),
        val_meta=val_meta,
        test_meta=test_meta,
    )

    write_json(
        out_dir / "run_config.json",
        {
            "slot_wide_parquet": str(args.slot_wide_parquet),
            "study_area_csv": str(args.study_area_csv),
            "siteyear_feature_csv": str(args.siteyear_feature_csv),
            "train_years": args.train_years,
            "val_years": args.val_years,
            "test_years": args.test_years,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "seed": args.seed,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
            "row_count": int(len(wide)),
            "site_count": int(len(site_ids)),
            "train_sample_count": int(len(train_row_idx)),
            "val_sample_count": int(len(val_row_idx)),
            "test_sample_count": int(len(test_row_idx)),
            "siteyear_feature_dim": int(len(siteyear_feature_cols)),
            "common_feature_count": int(common_feature_count),
            "geh_scale_factor": 4.0,
        },
    )
    write_json(out_dir / "metrics.json", results)
    pd.DataFrame([{"variant": name, **vals} for name, vals in results.items()]).sort_values("MAE").to_csv(
        out_dir / "benchmark_summary.csv", index=False
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
