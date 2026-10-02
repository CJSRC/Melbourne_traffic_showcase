"""Retail at five and ten times, to see whether its small response is only a small change.

Retail floors are a small share wherever they exist (median 1.6% at the 57 junctions that have
any), so doubling retail moves the attribute table far less than doubling office does. A flat
retail response at x2 cannot tell "the model ignores retail" from "x2 of almost nothing is
almost nothing". Five and ten times move retail's share by amounts comparable to office at x2.

Same frozen checkpoints, same rule for building the tables, only larger multipliers; the base
is scored again here and checked against the original sensitivity runs before anything is
compared.

    python lu_retail_extra.py tables            build and check the attribute tables
    python lu_retail_extra.py run <1..50>       one evaluation (slurm array task)
    python lu_retail_extra.py collect           per-junction means, as ch5_extract.py writes

Runs on Spartan, in the route12_allregions env.
"""

import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd

B = pathlib.Path("/data/gpfs/projects/punim2970/xhe13561")
N = B / "nearmap_v1"
TRAINER = B / "melbourne_scats_spartan_wayback_screened_v3_v12c/melbourne_scats_baseline"
ORIGINAL = B / "work/cfx/attr"                 # the tables the settled sensitivity runs used
W = B / "work/lu_retail_extra"
ATTR = W / "attr"
RUNS = W / "runs"
OUT = B / "work/report_data/ch5"
KEY = ["SITE_NO", "year"]
COLUMN = "floorshare_retail__shop"
VOLUME = ["floors_sum", "floors_sum_inner", "floors_mean", "floors_sum_weighted"]
TEST_YEAR = 2024
ARMS = ["r1attr", "r2attr"]
SEEDS = [42, 73, 128, 2024, 7]
DOSES = ["base", "retail_x5", "retail_x10", "retail_x5_swap", "retail_x10_swap"]
WINDOWS = {"am": ("weekday", 32, 36), "pm": ("weekday", 67, 71), "wk": ("weekend", 48, 52)}
SKIP = {"run_name", "out_dir", "posthoc", "branch_summary", "device_used", "started_at",
        "finished_at", "elapsed_seconds", "git_commit", "hostname", "epochs_ran",
        "best_epoch", "checkpoint_in", "eval_only", "device"}
FLAG_KEYS = {"graph_smoothing", "blend_base", "use_holiday_features", "disable_time_features",
             "mask_background", "local_files_only", "trust_remote_code"}


# ---------------------------------------------------------------- the rule (as lu_extra_doses.py)

def shares(frame):
    return [c for c in frame.columns if c.startswith("floorshare_")]


def dosed(base, sites):
    return ((base.year == TEST_YEAR) & base.SITE_NO.isin(sites) & (base[COLUMN] > 0)).to_numpy()


def dose_add(base, multiplier, sites):
    out = base.copy()
    live = dosed(base, sites)
    target = base[COLUMN].to_numpy(float)
    rest = base[shares(base)].to_numpy(float).sum(axis=1) - target
    total = rest + target * multiplier
    scale = np.where(total > 0, 1.0 / np.where(total > 0, total, 1.0), 1.0)
    for c in shares(base):
        v = base[c].to_numpy(float)
        out[c] = np.where(live, np.where(c == COLUMN, v * multiplier, v) * scale, v)
    grew = np.where(rest + target > 0, total / np.maximum(rest + target, 1e-12), 1.0)
    for c in VOLUME:
        out[c] = np.where(live, base[c].to_numpy(float) * grew, base[c].to_numpy(float))
    return out


def dose_swap(base, multiplier, sites):
    out = base.copy()
    live = dosed(base, sites)
    target = base[COLUMN].to_numpy(float)
    rest = base[shares(base)].to_numpy(float).sum(axis=1) - target
    wanted = np.minimum(target * multiplier, target + rest)
    shrink = np.where(rest > 0, (rest - (wanted - target)) / np.where(rest > 0, rest, 1.0), 1.0)
    for c in shares(base):
        v = base[c].to_numpy(float)
        out[c] = np.where(live, wanted if c == COLUMN else v * shrink, v)
    return out


def tables() -> None:
    ATTR.mkdir(parents=True, exist_ok=True)
    sites = set(pd.read_csv(N / "data/subsets/area_sites_attr.csv").SITE_NO.astype(int))
    base = pd.read_csv(ORIGINAL / "attr_base.csv").sort_values(KEY).reset_index(drop=True)
    cols = [c for c in base.columns if c not in KEY]
    # the rule must rebuild the retail tables the settled runs used, or nothing is comparable
    for name, rebuilt in (("attr_retail_x2.csv", dose_add(base, 2.0, sites)),
                          ("attr_retail_x2_swap.csv", dose_swap(base, 2.0, sites))):
        archived = pd.read_csv(ORIGINAL / name).sort_values(KEY).reset_index(drop=True)
        gap = float(np.nanmax(np.abs(archived[cols].to_numpy(float)
                                     - rebuilt[archived.columns][cols].to_numpy(float))))
        print(f"  {name}: largest difference {gap:.2e}")
        if gap > 5e-4:
            raise SystemExit("the rule does not reproduce the archived table; stopping")
    base.to_csv(ATTR / "attr_base.csv", index=False)
    live = dosed(base, sites)
    before = base.loc[live, COLUMN] * 100
    print(f"  {live.sum()} junctions; retail share median {before.median():.2f}%, "
          f"max {before.max():.2f}%")
    for m in (5, 10):
        for kind, t in (("", dose_add(base, m, sites)), ("_swap", dose_swap(base, m, sites))):
            name = f"attr_retail_x{m}{kind}.csv"
            t.to_csv(ATTR / name, index=False)
            after = t.loc[live, COLUMN] * 100
            print(f"  {name:<26} share after: median {after.median():5.1f}%, max {after.max():5.1f}%;"
                  f" change: mean {(after - before).mean():5.1f} pp")


# ---------------------------------------------------------------- one evaluation

def task(index: int):
    i = index - 1
    return (ARMS[i // (len(DOSES) * len(SEEDS))], DOSES[(i // len(SEEDS)) % len(DOSES)],
            SEEDS[i % len(SEEDS)])


def command(arm: str, dose: str, seed: int, device: str) -> list[str]:
    run = N / f"training/g_p500_tr_ln_{arm}_s{seed}"
    config = json.loads((run / "run_config.json").read_text())
    argv = [sys.executable, str(TRAINER / "train_route_multimodal_geh_v4.py"),
            "--out-dir", str(RUNS), "--run-name", f"lux3_{arm}_{dose}_s{seed}",
            "--eval-only", "--checkpoint-in", str(run / "model_state.pt"),
            "--attr-feature-csv", str(ATTR / f"attr_{dose}.csv"), "--device", device]
    for key, value in config.items():
        if key in SKIP or value is None or isinstance(value, dict):
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value and key in FLAG_KEYS:
                argv.append(flag)
            continue
        if isinstance(value, list):
            if value:
                argv += [flag] + [str(v) for v in value]
            continue
        argv += [flag, str(value)]
    return argv


def run(index: int, device: str) -> None:
    arm, dose, seed = task(index)
    name = f"lux3_{arm}_{dose}_s{seed}"
    if (RUNS / name / "test_predictions.parquet").is_file():
        print(f"{name} cached")
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"task {index} -> {name}", flush=True)
    subprocess.run(command(arm, dose, seed, device), cwd=TRAINER, check=True)


# ---------------------------------------------------------------- collect

def collect() -> None:
    parts, checks = [], []
    for index in range(1, len(ARMS) * len(DOSES) * len(SEEDS) + 1):
        arm, dose, seed = task(index)
        path = RUNS / f"lux3_{arm}_{dose}_s{seed}" / "test_predictions.parquet"
        d = pd.read_parquet(path, columns=["SITE_NO", "date", "slot_idx", "y_pred"])
        weekend = pd.to_datetime(d.date).dt.dayofweek.to_numpy() >= 5
        slot = d.slot_idx.to_numpy()
        frames = [d.assign(window="all")]
        for key, (daytype, lo, hi) in WINDOWS.items():
            keep = (weekend if daytype == "weekend" else ~weekend) & (slot >= lo) & (slot <= hi)
            frames.append(d[keep].assign(window=key))
        g = pd.concat(frames).groupby(["SITE_NO", "window"]).y_pred.agg(["mean", "size"])
        g = g.rename(columns={"mean": "mean_pred", "size": "n"}).reset_index()
        parts.append(g.assign(arm=arm, dose=dose, seed=seed))
        if dose == "base":
            # the base scored here against the base of the settled sensitivity runs
            old = pd.read_parquet(B / f"work/lu_settled/lus_{arm}_base_s{seed}/test_predictions.parquet",
                                  columns=["SITE_NO", "date", "slot_idx", "y_pred"])
            # scored on CPU here and on GPU there: expect floating-point noise, not a shift
            m = d.merge(old, on=["SITE_NO", "date", "slot_idx"], suffixes=("", "_old"))
            gap = m.y_pred - m.y_pred_old
            rel = float((gap.abs() / m.y_pred_old.abs().clip(lower=1)).max())
            checks.append((arm, seed, float(gap.abs().mean()), rel))
    for arm, seed, mean_gap, rel in checks:
        print(f"  base {arm} s{seed}: mean |difference| from the settled base {mean_gap:.4f}, "
              f"largest relative {rel:.1e}")
    if max(c[2] for c in checks) > 0.01 or max(c[3] for c in checks) > 0.01:
        raise SystemExit("the base does not reproduce; the new doses are not comparable")
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT / "site_response_retail_extra.csv", index=False)
    print(f"  {len(out)} rows -> {OUT / 'site_response_retail_extra.csv'}")


if __name__ == "__main__":
    if sys.argv[1] == "tables":
        tables()
    elif sys.argv[1] == "run":
        run(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "cuda")
    elif sys.argv[1] == "collect":
        collect()
