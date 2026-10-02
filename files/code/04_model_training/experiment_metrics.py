from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


def flatten_valid_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float32).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    valid = np.isfinite(yt) & np.isfinite(yp)
    return yt[valid], yp[valid]


def compute_geh(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    scale_factor: float = 1.0,
) -> dict[str, float]:
    yt, yp = flatten_valid_arrays(y_true, y_pred)
    yt = yt * float(scale_factor)
    yp = yp * float(scale_factor)
    denom = np.clip(yt + yp, 1e-6, None)
    geh = np.sqrt((2.0 * (yp - yt) ** 2) / denom)
    return {
        "GEH_mean": float(np.mean(geh)),
        "GEH_median": float(np.median(geh)),
        "GEH_p95": float(np.percentile(geh, 95)),
        "GEH_pct_lt_5": float(np.mean(geh < 5.0) * 100.0),
        "GEH_pct_lt_10": float(np.mean(geh < 10.0) * 100.0),
    }


def evaluate_regression_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    geh_scale_factor: float | None = None,
) -> dict[str, float]:
    yt, yp = flatten_valid_arrays(y_true, y_pred)
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    denom = np.clip(np.abs(yt), 1.0, None)
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_pct": float(np.mean(np.abs(yt - yp) / denom) * 100.0),
    }
    if geh_scale_factor is not None:
        metrics.update(compute_geh(yt, yp, scale_factor=geh_scale_factor))
    return metrics


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def collect_sklearn_params(model: Any) -> dict[str, Any]:
    if not hasattr(model, "get_params"):
        return {}
    params = model.get_params(deep=False)
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned
