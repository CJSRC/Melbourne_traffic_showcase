from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import osmnx as ox
except Exception:  # pragma: no cover
    ox = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SIGNALS_CSV_URL = (
    "https://opendata.transport.vic.gov.au/dataset/923af458-363d-469f-bc5e-84746a80b9a2/"
    "resource/d094415e-7b73-414a-88f5-6a3a6b5a903d/download/victorian_traffic_signals.csv"
)
VOLUME_JAN_2026_URL = (
    "https://opendata.transport.vic.gov.au/dataset/331b846b-1e18-415f-a3f3-ce4198d86c82/"
    "resource/790754eb-5e3b-49a0-ae64-fd44dbca87af/download/traffic_signal_volume_data_january_2026.zip"
)

MODEL_VARIANTS = {
    "graph_only",
    "graph_time",
    "diffusion_residual",
    "gru_only",
    "stgcn",
    "dcrnn_like",
    "dcrnn_like_nores",
    "dcrnn_like_static",
    "dcrnn_like_static_gated",
}
GRAPH_LEVELS = {"site", "approach"}
CORRIDOR_CHAINS = {"", "fwd", "rev"}
REQUIRED_MAPPING_COLUMNS = [
    "site_no",
    "detector_no",
    "approach_id",
    "approach_name",
    "corridor_chain",
    "movement_notes",
]


# ----------------------------
# Download and loading
# ----------------------------


def download_file(url: str, dest: Path, chunk_size: int = 2**20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return dest


def read_signals_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["SITE_NO", "SITE_NAME", "TYPE", "MUNICIPALITY", "LATITUDE", "LONGITUDE"]
    missing = [column for column in keep if column not in df.columns]
    if missing:
        raise ValueError(f"Signals CSV missing expected columns: {missing}")
    df = df[keep].copy()
    df["SITE_NO"] = pd.to_numeric(df["SITE_NO"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["SITE_NO", "LATITUDE", "LONGITUDE"])
    return df


def _pick_csv_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zip_file:
        csvs = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No CSV found inside {zip_path}")
        csvs.sort()
        return csvs[0]


def read_volume_zip(zip_path: Path) -> pd.DataFrame:
    member = _pick_csv_from_zip(zip_path)
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(member) as handle:
            df = pd.read_csv(handle)
    return df


# ----------------------------
# Volume reshaping
# ----------------------------


def reshape_volume_long(df: pd.DataFrame) -> pd.DataFrame:
    volume_cols = [f"V{i:02d}" for i in range(96) if f"V{i:02d}" in df.columns]
    if len(volume_cols) != 96:
        raise ValueError("Expected V00-V95 columns in volume data.")

    keep = [
        "NB_SCATS_SITE",
        "QT_INTERVAL_COUNT",
        "NB_DETECTOR",
        "NM_REGION",
        "CT_RECORDS",
        "QT_VOLUME_24HOUR",
        "CT_ALARM_24HOUR",
    ]
    id_vars = [column for column in keep if column in df.columns]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=volume_cols,
        var_name="slot",
        value_name="volume",
    )
    long_df["slot_idx"] = long_df["slot"].str[1:].astype(int)
    long_df["volume"] = pd.to_numeric(long_df["volume"], errors="coerce").fillna(0.0)
    long_df["NB_SCATS_SITE"] = pd.to_numeric(long_df["NB_SCATS_SITE"], errors="coerce").astype("Int64")
    long_df["NB_DETECTOR"] = pd.to_numeric(long_df["NB_DETECTOR"], errors="coerce").astype("Int64")

    dt = pd.to_datetime(long_df["QT_INTERVAL_COUNT"], errors="coerce")
    if dt.dt.hour.fillna(0).eq(0).all() and dt.dt.minute.fillna(0).eq(0).all():
        long_df["timestamp_end"] = dt + pd.to_timedelta((long_df["slot_idx"] + 1) * 15, unit="m")
    else:
        long_df["timestamp_end"] = dt
        same_dt = long_df.groupby(["NB_SCATS_SITE", "NB_DETECTOR", "QT_INTERVAL_COUNT"]).size().gt(1).any()
        if same_dt:
            long_df["timestamp_end"] = dt.dt.normalize() + pd.to_timedelta((long_df["slot_idx"] + 1) * 15, unit="m")

    long_df = long_df.dropna(subset=["NB_SCATS_SITE", "NB_DETECTOR", "timestamp_end"]).copy()
    long_df["NB_SCATS_SITE"] = long_df["NB_SCATS_SITE"].astype(int)
    long_df["NB_DETECTOR"] = long_df["NB_DETECTOR"].astype(int)
    return long_df


def aggregate_site_timeseries(long_df: pd.DataFrame) -> pd.DataFrame:
    site_ts = (
        long_df.groupby(["NB_SCATS_SITE", "timestamp_end"], as_index=False)["volume"]
        .sum()
        .rename(columns={"NB_SCATS_SITE": "SITE_NO", "volume": "site_volume"})
    )
    site_ts["SITE_NO"] = site_ts["SITE_NO"].astype(int)
    return site_ts


def summarize_site_activity(site_ts: pd.DataFrame) -> pd.DataFrame:
    tmp = site_ts.copy()
    tmp["date"] = tmp["timestamp_end"].dt.date
    daily = tmp.groupby(["SITE_NO", "date"], as_index=False)["site_volume"].sum()
    summary = daily.groupby("SITE_NO")["site_volume"].agg(["mean", "median", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "avg_daily_volume", "median": "med_daily_volume", "count": "n_days"})
    return summary


# ----------------------------
# Corridor selection
# ----------------------------


ROAD_SPLIT_RE = re.compile(r"\s*(?:/|\\|&|\bAT\b|\bAND\b|@|\bCORNER OF\b)\s*", re.IGNORECASE)
JUNK_TOKENS = {"MELBOURNE", "CBD", "CITY", "UNKNOWN"}


def normalize_road_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name).upper()).strip()
    name = re.sub(r"\bNTH\b", "NORTH", name)
    name = re.sub(r"\bSTH\b", "SOUTH", name)
    name = re.sub(r"\bRD\b", "ROAD", name)
    name = re.sub(r"\bST\b", "STREET", name)
    name = re.sub(r"\bAVE\b", "AVENUE", name)
    name = re.sub(r"\bHWY\b", "HIGHWAY", name)
    return name


def extract_road_names(site_name: str) -> list[str]:
    parts = [normalize_road_name(part) for part in ROAD_SPLIT_RE.split(str(site_name))]
    parts = [part for part in parts if part and part not in JUNK_TOKENS]
    dedup: list[str] = []
    for part in parts:
        if part not in dedup:
            dedup.append(part)
    return dedup


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def project_along_principal_axis(df: pd.DataFrame) -> np.ndarray:
    xy = df[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    xy = xy - xy.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(xy, full_matrices=False)
    axis = vh[0]
    return xy @ axis


@dataclass
class CorridorChoice:
    road_name: str
    sites: pd.DataFrame
    score: float
    reason: str


def choose_corridor(
    signals: pd.DataFrame,
    site_summary: pd.DataFrame,
    min_sites: int = 5,
    max_neighbor_gap_m: float = 1200.0,
) -> CorridorChoice:
    sig = signals.merge(site_summary, on="SITE_NO", how="inner")
    sig = sig[sig["TYPE"].eq("INT")].copy()
    if sig.empty:
        raise ValueError("No INT signal sites remain after joining summaries.")

    sig["road_names"] = sig["SITE_NAME"].apply(extract_road_names)
    exploded = sig.explode("road_names").dropna(subset=["road_names"]).rename(columns={"road_names": "road_name"})

    best: CorridorChoice | None = None
    for road_name, group in exploded.groupby("road_name"):
        group = group.drop_duplicates(subset=["SITE_NO"]).copy()
        if len(group) < min_sites:
            continue

        group["axis_pos"] = project_along_principal_axis(group)
        group = group.sort_values("axis_pos").reset_index(drop=True)

        chains: list[pd.DataFrame] = []
        start = 0
        for index in range(1, len(group)):
            distance = haversine_m(
                group.loc[index - 1, "LATITUDE"],
                group.loc[index - 1, "LONGITUDE"],
                group.loc[index, "LATITUDE"],
                group.loc[index, "LONGITUDE"],
            )
            if distance > max_neighbor_gap_m:
                chains.append(group.iloc[start:index].copy())
                start = index
        chains.append(group.iloc[start:].copy())

        for chain in chains:
            if len(chain) < min_sites:
                continue
            mean_gap = (
                np.mean(
                    [
                        haversine_m(
                            chain.iloc[index - 1]["LATITUDE"],
                            chain.iloc[index - 1]["LONGITUDE"],
                            chain.iloc[index]["LATITUDE"],
                            chain.iloc[index]["LONGITUDE"],
                        )
                        for index in range(1, len(chain))
                    ]
                )
                if len(chain) > 1
                else 0.0
            )
            score = float(len(chain) * chain["avg_daily_volume"].mean() / (1.0 + mean_gap / 500.0))
            reason = (
                f"road={road_name}; n_sites={len(chain)}; avg_daily_volume={chain['avg_daily_volume'].mean():.1f}; "
                f"mean_gap_m={mean_gap:.1f}"
            )
            if best is None or score > best.score:
                best = CorridorChoice(road_name=road_name, sites=chain.copy(), score=score, reason=reason)

    if best is None:
        raise ValueError("Could not identify a corridor with enough active INT signal sites.")

    best.sites["corridor_order"] = np.arange(len(best.sites))
    return best


# ----------------------------
# Graph utilities
# ----------------------------


def normalize_adjacency(adj: np.ndarray, mode: str = "symmetric") -> np.ndarray:
    if mode == "symmetric":
        deg = adj.sum(axis=1)
        deg_inv_sqrt = np.zeros_like(deg, dtype=float)
        np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
        deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
        d_inv = np.diag(deg_inv_sqrt)
        return d_inv @ adj @ d_inv
    if mode == "row":
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return adj / row_sum
    raise ValueError(f"Unsupported normalization mode: {mode}")


def is_structured_variant(variant: str) -> bool:
    return variant == "diffusion_residual"


def build_corridor_graph(
    corridor_sites: pd.DataFrame,
    use_osmnx: bool = False,
    *,
    structured: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    nodes = corridor_sites[["SITE_NO", "SITE_NAME", "LATITUDE", "LONGITUDE", "corridor_order"]].copy()
    nodes = nodes.sort_values("corridor_order").reset_index(drop=True)
    node_count = len(nodes)
    adj = np.zeros((node_count, node_count), dtype=float)

    route_distances: dict[tuple[int, int], float] = {}
    if use_osmnx and ox is not None:
        north = nodes["LATITUDE"].max() + 0.01
        south = nodes["LATITUDE"].min() - 0.01
        east = nodes["LONGITUDE"].max() + 0.01
        west = nodes["LONGITUDE"].min() - 0.01
        graph = ox.graph_from_bbox((north, south, east, west), network_type="drive", simplify=True)
        nearest_nodes = [ox.distance.nearest_nodes(graph, row.LONGITUDE, row.LATITUDE) for row in nodes.itertuples()]
        nodes["osmnx_node"] = nearest_nodes

        for index in range(node_count - 1):
            src = nodes.loc[index, "osmnx_node"]
            dst = nodes.loc[index + 1, "osmnx_node"]
            try:
                distance = nx.shortest_path_length(graph, src, dst, weight="length")
            except Exception:
                distance = haversine_m(
                    nodes.loc[index, "LATITUDE"],
                    nodes.loc[index, "LONGITUDE"],
                    nodes.loc[index + 1, "LATITUDE"],
                    nodes.loc[index + 1, "LONGITUDE"],
                )
            route_distances[(index, index + 1)] = distance
            route_distances[(index + 1, index)] = distance
    else:
        for index in range(node_count - 1):
            distance = haversine_m(
                nodes.loc[index, "LATITUDE"],
                nodes.loc[index, "LONGITUDE"],
                nodes.loc[index + 1, "LATITUDE"],
                nodes.loc[index + 1, "LONGITUDE"],
            )
            route_distances[(index, index + 1)] = distance
            route_distances[(index + 1, index)] = distance

    for index in range(node_count - 1):
        distance = max(route_distances[(index, index + 1)], 1.0)
        weight = 1.0 / distance
        adj[index, index + 1] = weight
        adj[index + 1, index] = weight

    if structured:
        for index in range(node_count - 2):
            hop_distance = max(route_distances[(index, index + 1)] + route_distances[(index + 1, index + 2)], 1.0)
            skip_weight = 0.35 / hop_distance
            adj[index, index + 2] = max(adj[index, index + 2], skip_weight)
            adj[index + 2, index] = max(adj[index + 2, index], skip_weight)

    np.fill_diagonal(adj, 1.0)
    adj_norm = normalize_adjacency(adj, mode="row" if structured else "symmetric")
    return adj_norm, nodes


def build_site_matrix(site_ts: pd.DataFrame, corridor_nodes: pd.DataFrame) -> pd.DataFrame:
    node_order = corridor_nodes["SITE_NO"].tolist()
    pivot = site_ts[site_ts["SITE_NO"].isin(node_order)].pivot(index="timestamp_end", columns="SITE_NO", values="site_volume")
    pivot = pivot.reindex(columns=node_order)
    pivot = pivot.sort_index()
    pivot = pivot.asfreq("15min")
    pivot = pivot.ffill().fillna(0.0)
    return pivot


def build_generic_matrix(
    ts_df: pd.DataFrame,
    node_table: pd.DataFrame,
    node_key_col: str,
    value_col: str,
) -> pd.DataFrame:
    node_order = node_table[node_key_col].tolist()
    pivot = ts_df.pivot(index="timestamp_end", columns=node_key_col, values=value_col)
    pivot = pivot.reindex(columns=node_order)
    pivot = pivot.sort_index()
    pivot = pivot.asfreq("15min")
    pivot = pivot.ffill().fillna(0.0)
    return pivot


def make_time_features(index: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    slot = idx.hour * 4 + (idx.minute // 15)
    dow = idx.dayofweek
    return np.column_stack(
        [
            np.sin(2 * np.pi * slot / 96.0),
            np.cos(2 * np.pi * slot / 96.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            (dow >= 5).astype(float),
        ]
    ).astype(np.float32)


# ----------------------------
# Approach mapping workflow
# ----------------------------


def build_detector_template(long_df: pd.DataFrame, corridor_nodes: pd.DataFrame) -> pd.DataFrame:
    template = (
        long_df[long_df["NB_SCATS_SITE"].isin(corridor_nodes["SITE_NO"])][["NB_SCATS_SITE", "NB_DETECTOR"]]
        .drop_duplicates()
        .sort_values(["NB_SCATS_SITE", "NB_DETECTOR"])
        .rename(columns={"NB_SCATS_SITE": "site_no", "NB_DETECTOR": "detector_no"})
        .reset_index(drop=True)
    )
    return template


def generate_approach_mapping_template(long_df: pd.DataFrame, corridor_nodes: pd.DataFrame, out_csv: Path) -> Path:
    template = build_detector_template(long_df, corridor_nodes)
    for column in REQUIRED_MAPPING_COLUMNS[2:]:
        template[column] = ""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_csv, index=False)
    return out_csv


def _clean_mapping_df(mapping_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = mapping_df.copy()
    missing = [column for column in REQUIRED_MAPPING_COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"mapping CSV missing columns: {missing}")

    cleaned = cleaned[REQUIRED_MAPPING_COLUMNS].copy()
    cleaned["site_no"] = pd.to_numeric(cleaned["site_no"], errors="coerce")
    cleaned["detector_no"] = pd.to_numeric(cleaned["detector_no"], errors="coerce")
    if cleaned["site_no"].isna().any() or cleaned["detector_no"].isna().any():
        raise ValueError("site_no and detector_no must be numeric for every mapping row.")
    cleaned["site_no"] = cleaned["site_no"].astype(int)
    cleaned["detector_no"] = cleaned["detector_no"].astype(int)

    for column in ["approach_id", "approach_name", "corridor_chain", "movement_notes"]:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    invalid_chains = sorted(set(cleaned["corridor_chain"]) - CORRIDOR_CHAINS)
    if invalid_chains:
        raise ValueError(f"corridor_chain must be one of {sorted(CORRIDOR_CHAINS)}; got {invalid_chains}")

    duplicated_pairs = cleaned.duplicated(subset=["site_no", "detector_no"], keep=False)
    if duplicated_pairs.any():
        dup_rows = cleaned.loc[duplicated_pairs, ["site_no", "detector_no"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Detector rows must be unique in mapping CSV; duplicates found for {dup_rows}")

    return cleaned


def validate_approach_mapping(
    mapping_df: pd.DataFrame,
    long_df: pd.DataFrame,
    corridor_nodes: pd.DataFrame,
    *,
    allow_partial_sites: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = _clean_mapping_df(mapping_df)
    expected = build_detector_template(long_df, corridor_nodes)
    merged = expected.merge(cleaned, on=["site_no", "detector_no"], how="left", indicator=True)

    unexpected = cleaned.merge(expected, on=["site_no", "detector_no"], how="left", indicator=True)
    unexpected = unexpected[unexpected["_merge"] == "left_only"][["site_no", "detector_no"]]
    if not unexpected.empty:
        records = unexpected.drop_duplicates().to_dict("records")
        raise ValueError(f"Mapping CSV contains detectors outside the selected corridor: {records}")

    merged["approach_id"] = merged["approach_id"].fillna("").astype(str).str.strip()
    merged["approach_name"] = merged["approach_name"].fillna("").astype(str).str.strip()
    merged["corridor_chain"] = merged["corridor_chain"].fillna("").astype(str).str.strip()
    merged["movement_notes"] = merged["movement_notes"].fillna("").astype(str).str.strip()

    merged["mapped_row"] = (merged["approach_id"] != "") & (merged["approach_name"] != "")
    merged["has_any_annotation"] = merged[["approach_id", "approach_name", "corridor_chain", "movement_notes"]].ne("").any(axis=1)

    coverage = (
        merged.groupby("site_no", as_index=False)
        .agg(
            expected_detectors=("detector_no", "size"),
            rows_present=("_merge", lambda values: int((pd.Series(values) == "both").sum())),
            mapped_detectors=("mapped_row", "sum"),
            annotated_rows=("has_any_annotation", "sum"),
        )
        .sort_values("site_no")
        .reset_index(drop=True)
    )
    coverage["started"] = coverage["annotated_rows"] > 0
    coverage["fully_mapped"] = coverage["mapped_detectors"] == coverage["expected_detectors"]
    coverage["complete_rows_present"] = coverage["rows_present"] == coverage["expected_detectors"]
    coverage["ready_for_build"] = coverage["fully_mapped"] & coverage["complete_rows_present"]

    if (merged["has_any_annotation"] & ~merged["mapped_row"]).any():
        bad_rows = merged.loc[merged["has_any_annotation"] & ~merged["mapped_row"], ["site_no", "detector_no"]]
        raise ValueError(
            "Started mapping rows must include both approach_id and approach_name; incomplete rows found for "
            f"{bad_rows.drop_duplicates().to_dict('records')}"
        )

    if (coverage["started"] & ~coverage["ready_for_build"]).any():
        bad_sites = coverage.loc[coverage["started"] & ~coverage["ready_for_build"], "site_no"].tolist()
        raise ValueError(f"Started sites must be fully mapped before validation passes; incomplete sites: {bad_sites}")

    if not allow_partial_sites and not coverage["ready_for_build"].all():
        remaining = coverage.loc[~coverage["ready_for_build"], "site_no"].tolist()
        raise ValueError(f"All selected corridor sites must be fully mapped before build; remaining sites: {remaining}")

    return merged, coverage


def build_approach_timeseries(
    long_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    corridor_nodes: pd.DataFrame,
    *,
    allow_partial_sites: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validated, coverage = validate_approach_mapping(
        mapping_df=mapping_df,
        long_df=long_df,
        corridor_nodes=corridor_nodes,
        allow_partial_sites=allow_partial_sites,
    )
    ready_sites = coverage.loc[coverage["ready_for_build"], "site_no"].tolist()
    usable = validated[validated["site_no"].isin(ready_sites) & validated["mapped_row"]].copy()
    if usable.empty:
        raise ValueError("No fully mapped approach rows are available to build approach-level time series.")

    mapping_for_merge = usable.rename(columns={"site_no": "NB_SCATS_SITE", "detector_no": "NB_DETECTOR"})
    merged = long_df.merge(
        mapping_for_merge[
            ["NB_SCATS_SITE", "NB_DETECTOR", "approach_id", "approach_name", "corridor_chain", "movement_notes"]
        ],
        on=["NB_SCATS_SITE", "NB_DETECTOR"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No detector rows matched the validated mapping file.")

    approach_ts = (
        merged.groupby(
            ["NB_SCATS_SITE", "approach_id", "approach_name", "corridor_chain", "timestamp_end"],
            as_index=False,
        )["volume"]
        .sum()
        .rename(columns={"NB_SCATS_SITE": "SITE_NO", "volume": "approach_volume"})
    )
    approach_ts["node_id"] = approach_ts["SITE_NO"].astype(str) + ":" + approach_ts["approach_id"].astype(str)
    return approach_ts, usable, coverage


def build_approach_graph(
    approach_ts: pd.DataFrame,
    corridor_nodes: pd.DataFrame,
    *,
    structured: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    node_table = (
        approach_ts[["SITE_NO", "approach_id", "approach_name", "corridor_chain", "node_id"]]
        .drop_duplicates()
        .merge(corridor_nodes[["SITE_NO", "corridor_order", "SITE_NAME", "LATITUDE", "LONGITUDE"]], on="SITE_NO", how="left")
        .sort_values(["corridor_order", "corridor_chain", "approach_id"])
        .reset_index(drop=True)
    )
    node_table = node_table[node_table["corridor_chain"].isin({"fwd", "rev"})].copy()
    if node_table.empty:
        raise ValueError("No mainline approach nodes found; fill corridor_chain with fwd/rev for the desired approaches.")

    per_site_chain = node_table.groupby(["SITE_NO", "corridor_chain"]).size().reset_index(name="count")
    collisions = per_site_chain[per_site_chain["count"] > 1]
    if not collisions.empty:
        raise ValueError(
            "Stage-1 approach graph expects at most one active approach per site and corridor_chain; "
            f"conflicts found for {collisions[['SITE_NO', 'corridor_chain']].to_dict('records')}"
        )

    node_table = node_table.reset_index(drop=True)
    node_table["graph_index"] = np.arange(len(node_table))
    index_lookup = dict(zip(node_table["node_id"], node_table["graph_index"]))

    adj = np.zeros((len(node_table), len(node_table)), dtype=float)
    for chain in ["fwd", "rev"]:
        chain_nodes = node_table[node_table["corridor_chain"] == chain].sort_values("corridor_order").reset_index(drop=True)
        if len(chain_nodes) <= 1:
            continue
        for index in range(len(chain_nodes) - 1):
            src_row = chain_nodes.iloc[index]
            dst_row = chain_nodes.iloc[index + 1]
            src = index_lookup[src_row["node_id"]]
            dst = index_lookup[dst_row["node_id"]]
            distance = haversine_m(src_row["LATITUDE"], src_row["LONGITUDE"], dst_row["LATITUDE"], dst_row["LONGITUDE"])
            weight = 1.0 / max(distance, 1.0)
            if chain == "fwd":
                adj[dst, src] = weight
            else:
                adj[src, dst] = weight

        if structured and len(chain_nodes) > 2:
            for index in range(len(chain_nodes) - 2):
                src_row = chain_nodes.iloc[index]
                mid_row = chain_nodes.iloc[index + 1]
                dst_row = chain_nodes.iloc[index + 2]
                src = index_lookup[src_row["node_id"]]
                dst = index_lookup[dst_row["node_id"]]
                hop_distance = (
                    haversine_m(src_row["LATITUDE"], src_row["LONGITUDE"], mid_row["LATITUDE"], mid_row["LONGITUDE"])
                    + haversine_m(mid_row["LATITUDE"], mid_row["LONGITUDE"], dst_row["LATITUDE"], dst_row["LONGITUDE"])
                )
                skip_weight = 0.35 / max(hop_distance, 1.0)
                if chain == "fwd":
                    adj[dst, src] = max(adj[dst, src], skip_weight)
                else:
                    adj[src, dst] = max(adj[src, dst], skip_weight)

    if structured:
        site_pairs = (
            node_table.groupby("SITE_NO")["graph_index"]
            .apply(list)
            .to_dict()
        )
        for _, indices in site_pairs.items():
            if len(indices) < 2:
                continue
            for src in indices:
                for dst in indices:
                    if src != dst:
                        adj[dst, src] = max(adj[dst, src], 0.2)

    np.fill_diagonal(adj, 1.0)
    adj_norm = normalize_adjacency(adj, mode="row")
    return adj_norm, node_table


# ----------------------------
# Dataset and model
# ----------------------------


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        array_2d: np.ndarray,
        time_feat_2d: np.ndarray,
        seq_len: int,
        pred_len: int,
        node_static_2d: np.ndarray | None = None,
    ):
        sample_count = array_2d.shape[0] - seq_len - pred_len + 1
        if sample_count <= 0:
            raise ValueError("Not enough time steps to create sliding windows with the requested seq_len and pred_len.")

        x_traffic = []
        x_time = []
        y = []
        for index in range(sample_count):
            x_seq = array_2d[index : index + seq_len]
            y_seq = array_2d[index + seq_len : index + seq_len + pred_len]
            tf_seq = time_feat_2d[index : index + seq_len]
            x_traffic.append(x_seq[:, :, None])
            x_time.append(tf_seq)
            y.append(y_seq)

        self.x_traffic = torch.tensor(np.stack(x_traffic), dtype=torch.float32)
        self.x_time = torch.tensor(np.stack(x_time), dtype=torch.float32)
        self.y = torch.tensor(np.stack(y), dtype=torch.float32)
        if node_static_2d is None:
            node_static_2d = np.zeros((array_2d.shape[1], 0), dtype=np.float32)
        self.x_static = torch.tensor(node_static_2d, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x_traffic[idx], self.x_time[idx], self.x_static, self.y[idx]


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adj_norm: np.ndarray):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_norm, dtype=torch.float32))
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,bjc->bic", self.A, x)
        return self.linear(x)


class DiffusionGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adj_norm: np.ndarray):
        super().__init__()
        base = np.asarray(adj_norm, dtype=np.float32)
        backward = normalize_adjacency(base.T.copy(), mode="row")
        two_hop = base @ base
        np.fill_diagonal(two_hop, 0.0)
        two_hop = normalize_adjacency(two_hop + np.eye(two_hop.shape[0], dtype=np.float32), mode="row")
        self.register_buffer("A_fwd", torch.tensor(base, dtype=torch.float32))
        self.register_buffer("A_bwd", torch.tensor(backward, dtype=torch.float32))
        self.register_buffer("A_2hop", torch.tensor(two_hop, dtype=torch.float32))
        self.linear = nn.Linear(in_channels * 4, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fwd = torch.einsum("ij,bjc->bic", self.A_fwd, x)
        x_bwd = torch.einsum("ij,bjc->bic", self.A_bwd, x)
        x_2hop = torch.einsum("ij,bjc->bic", self.A_2hop, x)
        features = torch.cat([x, x_fwd, x_bwd, x_2hop], dim=-1)
        return self.linear(features)


class GCNForecastModel(nn.Module):
    def __init__(
        self,
        variant: str,
        num_nodes: int,
        pred_len: int,
        adj_norm: np.ndarray,
        time_feat_dim: int,
        static_feat_dim: int = 0,
        hidden_channels: int = 16,
        gru_hidden: int = 64,
    ):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unsupported model variant: {variant}")
        self.variant = variant
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.static_feat_dim = static_feat_dim
        if variant == "diffusion_residual":
            self.spatial = DiffusionGraphConv(1, hidden_channels, adj_norm)
        else:
            self.spatial = GraphConv(1, hidden_channels, adj_norm)
        input_size = num_nodes * hidden_channels + (time_feat_dim if variant == "graph_time" else 0) + (
            num_nodes * static_feat_dim
        )
        self.gru = nn.GRU(input_size=input_size, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, num_nodes * pred_len)

    def forward(self, x_traffic: torch.Tensor, x_time: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        sequence = []
        batch_size = x_traffic.size(0)
        static_flat = x_static.reshape(batch_size, -1) if x_static.numel() > 0 else None
        for step in range(x_traffic.size(1)):
            spatial = F.relu(self.spatial(x_traffic[:, step]))
            spatial_flat = spatial.reshape(batch_size, -1)
            if self.variant == "graph_time":
                step_input = torch.cat([spatial_flat, x_time[:, step]], dim=-1)
            else:
                step_input = spatial_flat
            if static_flat is not None:
                step_input = torch.cat([step_input, static_flat], dim=-1)
            sequence.append(step_input)
        seq_tensor = torch.stack(sequence, dim=1)
        out, _ = self.gru(seq_tensor)
        pred = self.head(out[:, -1]).reshape(batch_size, self.pred_len, self.num_nodes)
        if self.variant == "diffusion_residual":
            baseline = x_traffic[:, -1, :, 0].unsqueeze(1).repeat(1, self.pred_len, 1)
            pred = baseline + pred
        return pred


class GRUOnlyForecastModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        pred_len: int,
        static_feat_dim: int = 0,
        gru_hidden: int = 64,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.static_feat_dim = static_feat_dim
        self.gru = nn.GRU(input_size=num_nodes + num_nodes * static_feat_dim, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, num_nodes * pred_len)

    def forward(self, x_traffic: torch.Tensor, x_time: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x_seq = x_traffic[..., 0]
        if x_static.numel() > 0:
            static_flat = x_static.reshape(x_seq.size(0), -1).unsqueeze(1).repeat(1, x_seq.size(1), 1)
            x_seq = torch.cat([x_seq, static_flat], dim=-1)
        out, _ = self.gru(x_seq)
        pred = self.head(out[:, -1]).reshape(x_seq.size(0), self.pred_len, self.num_nodes)
        return pred


class STGCNForecastModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        pred_len: int,
        adj_norm: np.ndarray,
        static_feat_dim: int = 0,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.static_feat_dim = static_feat_dim
        self.register_buffer("A", torch.tensor(adj_norm, dtype=torch.float32))
        self.temporal1 = nn.Conv2d(1, hidden_channels, kernel_size=(1, 3), padding=(0, 1))
        self.temporal2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1))
        self.layer_norm = nn.LayerNorm([num_nodes, hidden_channels])
        self.head = nn.Linear(num_nodes * hidden_channels + num_nodes * static_feat_dim, num_nodes * pred_len)

    def forward(self, x_traffic: torch.Tensor, x_time: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x = x_traffic[..., 0].permute(0, 2, 1).unsqueeze(1)
        x = F.relu(self.temporal1(x))
        x = torch.einsum("ij,bcjt->bcit", self.A, x)
        x = F.relu(self.temporal2(x))
        x = x[..., -1].permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.reshape(x.size(0), -1)
        if x_static.numel() > 0:
            x = torch.cat([x, x_static.reshape(x_static.size(0), -1)], dim=-1)
        pred = self.head(x).reshape(x.size(0), self.pred_len, self.num_nodes)
        return pred


class DCRNNLikeCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, adj_norm: np.ndarray):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gate_conv = DiffusionGraphConv(in_channels + hidden_channels, hidden_channels * 2, adj_norm)
        self.cand_conv = DiffusionGraphConv(in_channels + hidden_channels, hidden_channels, adj_norm)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_conv(torch.cat([x, h], dim=-1)))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=-1)
        candidate = torch.tanh(self.cand_conv(torch.cat([x, reset_gate * h], dim=-1)))
        return update_gate * h + (1.0 - update_gate) * candidate


class DCRNNLikeForecastModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        pred_len: int,
        adj_norm: np.ndarray,
        hidden_channels: int = 32,
        residual: bool = True,
        static_feat_dim: int = 0,
        fusion_mode: str = "concat",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.static_feat_dim = static_feat_dim
        self.fusion_mode = fusion_mode
        self.cell = DCRNNLikeCell(1, hidden_channels, adj_norm)
        if fusion_mode == "gated" and static_feat_dim > 0:
            self.static_proj = nn.Linear(static_feat_dim, hidden_channels)
            self.static_gate = nn.Linear(hidden_channels * 2, hidden_channels)
            head_in_dim = num_nodes * hidden_channels
        else:
            self.static_proj = None
            self.static_gate = None
            head_in_dim = num_nodes * hidden_channels + num_nodes * static_feat_dim
        self.head = nn.Linear(head_in_dim, num_nodes * pred_len)

    def forward(self, x_traffic: torch.Tensor, x_time: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_nodes, _ = x_traffic.shape
        h = torch.zeros(batch_size, num_nodes, self.hidden_channels, device=x_traffic.device)
        for step in range(x_traffic.size(1)):
            h = self.cell(x_traffic[:, step], h)
        if self.fusion_mode == "gated" and x_static.numel() > 0 and self.static_proj is not None:
            static_embed = torch.tanh(self.static_proj(x_static))
            gate = torch.sigmoid(self.static_gate(torch.cat([h, static_embed], dim=-1)))
            fused = h + gate * static_embed
            head_input = fused.reshape(batch_size, -1)
        else:
            head_input = h.reshape(batch_size, -1)
        if self.fusion_mode != "gated" and x_static.numel() > 0:
            head_input = torch.cat([head_input, x_static.reshape(batch_size, -1)], dim=-1)
        pred_delta = self.head(head_input).reshape(batch_size, self.pred_len, self.num_nodes)
        if self.residual:
            baseline = x_traffic[:, -1, :, 0].unsqueeze(1).repeat(1, self.pred_len, 1)
            return baseline + pred_delta
        return pred_delta


def build_model(
    *,
    variant: str,
    num_nodes: int,
    pred_len: int,
    adj_norm: np.ndarray,
    time_feat_dim: int,
    static_feat_dim: int = 0,
) -> nn.Module:
    if variant in {"graph_only", "graph_time", "diffusion_residual"}:
        return GCNForecastModel(
            variant=variant,
            num_nodes=num_nodes,
            pred_len=pred_len,
            adj_norm=adj_norm,
            time_feat_dim=time_feat_dim,
            static_feat_dim=static_feat_dim,
        )
    if variant == "gru_only":
        return GRUOnlyForecastModel(num_nodes=num_nodes, pred_len=pred_len, static_feat_dim=static_feat_dim)
    if variant == "stgcn":
        return STGCNForecastModel(num_nodes=num_nodes, pred_len=pred_len, adj_norm=adj_norm, static_feat_dim=static_feat_dim)
    if variant in {"dcrnn_like", "dcrnn_like_static"}:
        return DCRNNLikeForecastModel(
            num_nodes=num_nodes,
            pred_len=pred_len,
            adj_norm=adj_norm,
            static_feat_dim=static_feat_dim,
        )
    if variant == "dcrnn_like_static_gated":
        return DCRNNLikeForecastModel(
            num_nodes=num_nodes,
            pred_len=pred_len,
            adj_norm=adj_norm,
            static_feat_dim=static_feat_dim,
            fusion_mode="gated",
        )
    if variant == "dcrnn_like_nores":
        return DCRNNLikeForecastModel(
            num_nodes=num_nodes,
            pred_len=pred_len,
            adj_norm=adj_norm,
            residual=False,
            static_feat_dim=static_feat_dim,
        )
    raise ValueError(f"Unsupported model variant: {variant}")


# ----------------------------
# Training and evaluation
# ----------------------------


def train_val_test_split(matrix: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1):
    total = len(matrix)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return matrix.iloc[:train_end], matrix.iloc[train_end:val_end], matrix.iloc[val_end:]


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    return scaler


def transform_df(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df.values).astype(np.float32)


def inverse_transform(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1])
    inv = scaler.inverse_transform(flat)
    return inv.reshape(arr.shape).astype(np.float32)


def evaluate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    denom = np.clip(np.abs(yt), 1.0, None)
    mape = float(np.mean(np.abs(yt - yp) / denom) * 100.0)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_pct": mape}


def build_time_feature_array(index: pd.DatetimeIndex, variant: str) -> np.ndarray:
    if variant == "graph_time":
        return make_time_features(index)
    return np.zeros((len(index), 0), dtype=np.float32)


def build_historical_average_lookup(history_df: pd.DataFrame) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, np.ndarray], np.ndarray]:
    hist = history_df.copy()
    index = pd.DatetimeIndex(hist.index)
    hist["dow"] = index.dayofweek
    hist["slot"] = index.hour * 4 + (index.minute // 15)

    by_dow_slot = {
        key: group.drop(columns=["dow", "slot"]).mean(axis=0).to_numpy(dtype=np.float32)
        for key, group in hist.groupby(["dow", "slot"])
    }
    by_slot = {
        int(key): group.drop(columns=["dow", "slot"]).mean(axis=0).to_numpy(dtype=np.float32)
        for key, group in hist.groupby("slot")
    }
    global_mean = hist.drop(columns=["dow", "slot"]).mean(axis=0).to_numpy(dtype=np.float32)
    return by_dow_slot, by_slot, global_mean


def historical_average_predict(
    history_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    pred_shape: tuple[int, int, int],
) -> np.ndarray:
    by_dow_slot, by_slot, global_mean = build_historical_average_lookup(history_df)
    sample_count, pred_len, num_nodes = pred_shape
    expected_steps = sample_count * pred_len
    if len(target_index) != expected_steps:
        raise ValueError("Target index length does not match prediction shape for historical-average baseline.")

    preds = np.zeros(pred_shape, dtype=np.float32)
    timestamps = list(pd.DatetimeIndex(target_index))
    cursor = 0
    for sample_idx in range(sample_count):
        for step_idx in range(pred_len):
            ts = timestamps[cursor]
            key = (int(ts.dayofweek), int(ts.hour * 4 + (ts.minute // 15)))
            slot_key = key[1]
            value = by_dow_slot.get(key)
            if value is None:
                value = by_slot.get(slot_key, global_mean)
            preds[sample_idx, step_idx, :] = value[:num_nodes]
            cursor += 1
    return preds


def run_training(
    matrix: pd.DataFrame,
    adj_norm: np.ndarray,
    out_dir: Path,
    *,
    model_variant: str,
    node_static_features: np.ndarray | None = None,
    seq_len: int = 12,
    pred_len: int = 1,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, object]:
    set_global_seed(seed)
    train_df, val_df, test_df = train_val_test_split(matrix)
    scaler = fit_scaler(train_df)

    train_arr = transform_df(train_df, scaler)
    val_input_df = pd.concat([train_df.tail(seq_len), val_df])
    val_arr = transform_df(val_input_df, scaler)
    test_input_df = pd.concat([pd.concat([train_df, val_df]).tail(seq_len), test_df])
    test_arr = transform_df(test_input_df, scaler)

    train_tf = build_time_feature_array(train_df.index, model_variant)
    val_tf = build_time_feature_array(val_input_df.index, model_variant)
    test_tf = build_time_feature_array(test_input_df.index, model_variant)

    train_ds = SlidingWindowDataset(train_arr, train_tf, seq_len, pred_len, node_static_2d=node_static_features)
    val_ds = SlidingWindowDataset(val_arr, val_tf, seq_len, pred_len, node_static_2d=node_static_features)
    test_ds = SlidingWindowDataset(test_arr, test_tf, seq_len, pred_len, node_static_2d=node_static_features)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        variant=model_variant,
        num_nodes=matrix.shape[1],
        pred_len=pred_len,
        adj_norm=adj_norm,
        time_feat_dim=train_ds.x_time.shape[-1],
        static_feat_dim=train_ds.x_static.shape[-1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_traffic, x_time, x_static, y_batch in train_loader:
            x_traffic = x_traffic.to(device)
            x_time = x_time.to(device)
            x_static = x_static.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_traffic, x_time, x_static)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_traffic, x_time, x_static, y_batch in val_loader:
                pred = model(x_traffic.to(device), x_time.to(device), x_static.to(device))
                val_losses.append(criterion(pred, y_batch.to(device)).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    preds_scaled = []
    trues_scaled = []
    with torch.no_grad():
        for x_traffic, x_time, x_static, y_batch in test_loader:
            pred = model(x_traffic.to(device), x_time.to(device), x_static.to(device)).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(y_batch.numpy())

    y_pred_scaled = np.concatenate(preds_scaled, axis=0)
    y_true_scaled = np.concatenate(trues_scaled, axis=0)
    y_pred = inverse_transform(y_pred_scaled, scaler)
    y_true = inverse_transform(y_true_scaled, scaler)

    x_test = test_ds.x_traffic.numpy().squeeze(-1)
    pers_scaled = np.repeat(x_test[:, -1:, :], pred_len, axis=1)
    y_persistence = inverse_transform(pers_scaled, scaler)
    target_timestamps = []
    test_index = pd.DatetimeIndex(test_input_df.index)
    for index in range(len(test_ds)):
        target_timestamps.extend(test_index[index + seq_len : index + seq_len + pred_len].tolist())
    y_historical_average = historical_average_predict(
        history_df=pd.concat([train_df, val_df]),
        target_index=pd.DatetimeIndex(target_timestamps),
        pred_shape=y_true.shape,
    )

    metrics_model = evaluate_arrays(y_true, y_pred)
    metrics_persistence = evaluate_arrays(y_true, y_persistence)
    metrics_historical_average = evaluate_arrays(y_true, y_historical_average)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": metrics_model,
                "persistence": metrics_persistence,
                "historical_average": metrics_historical_average,
            },
            handle,
            indent=2,
        )
    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_pred.npy", y_pred)
    np.save(out_dir / "y_persistence.npy", y_persistence)
    np.save(out_dir / "y_historical_average.npy", y_historical_average)
    torch.save(best_state, out_dir / "gcn_gru_state_dict.pt")

    return {
        "metrics": {
            "model": metrics_model,
            "persistence": metrics_persistence,
            "historical_average": metrics_historical_average,
        },
        "shapes": {
            "y_true": list(y_true.shape),
            "y_pred": list(y_pred.shape),
            "y_persistence": list(y_persistence.shape),
            "y_historical_average": list(y_historical_average.shape),
        },
    }


# ----------------------------
# Run metadata helpers
# ----------------------------


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "run"


def build_run_name(
    *,
    level: str,
    variant: str,
    corridor_name: str,
    seq_len: int,
    pred_len: int,
    epochs: int,
    use_osmnx: bool,
) -> str:
    return "__".join(
        [
            sanitize_slug(level),
            sanitize_slug(variant),
            f"corridor-{sanitize_slug(corridor_name)}",
            f"sl{seq_len}",
            f"pl{pred_len}",
            f"e{epochs}",
            f"osmnx{int(use_osmnx)}",
        ]
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Melbourne SCATS corridor baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional explicit run directory override")
    parser.add_argument("--signals-url", type=str, default=SIGNALS_CSV_URL)
    parser.add_argument("--volume-url", type=str, default=VOLUME_JAN_2026_URL)
    parser.add_argument("--seq-len", type=int, default=12, help="History steps (12 = past 3 hours)")
    parser.add_argument("--pred-len", type=int, default=1, help="Forecast horizon in 15-minute steps")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sites", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--variant", choices=sorted(MODEL_VARIANTS), default="graph_only")
    parser.add_argument("--level", choices=sorted(GRAPH_LEVELS), default="site")
    parser.add_argument("--mapping-csv", type=Path, default=None, help="Required when --level approach")
    parser.add_argument("--use-osmnx", action="store_true", help="Use OSMnx route distances for site-level graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.level == "approach" and args.mapping_csv is None:
        raise ValueError("--mapping-csv is required when --level approach")

    data_dir = args.data_dir
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    signals_path = download_file(args.signals_url, raw_dir / "victorian_traffic_signals.csv")
    volume_path = download_file(args.volume_url, raw_dir / "traffic_signal_volume_data.zip")

    print("Loading signals...")
    signals = read_signals_csv(signals_path)
    print(f"Signals rows: {len(signals):,}")

    long_path = interim_dir / "volume_long.parquet"
    site_ts_path = interim_dir / "site_timeseries.parquet"
    site_summary_path = interim_dir / "site_summary.csv"

    if long_path.exists() and site_ts_path.exists() and site_summary_path.exists():
        print("Loading cached reshaped traffic artifacts...")
        long_df = pd.read_parquet(long_path)
        site_ts = pd.read_parquet(site_ts_path)
        site_summary = pd.read_csv(site_summary_path)
        if "timestamp_end" in long_df.columns:
            long_df["timestamp_end"] = pd.to_datetime(long_df["timestamp_end"])
        if "timestamp_end" in site_ts.columns:
            site_ts["timestamp_end"] = pd.to_datetime(site_ts["timestamp_end"])
    else:
        print("Loading volume zip (this may take a while)...")
        volume_df = read_volume_zip(volume_path)
        print(f"Volume raw rows: {len(volume_df):,}")

        print("Reshaping 15-minute counts...")
        long_df = reshape_volume_long(volume_df)
        site_ts = aggregate_site_timeseries(long_df)
        site_summary = summarize_site_activity(site_ts)

        long_df.to_parquet(long_path, index=False)
        site_ts.to_parquet(site_ts_path, index=False)
        site_summary.to_csv(site_summary_path, index=False)

    print("Selecting representative corridor from data distribution...")
    corridor = choose_corridor(signals, site_summary, min_sites=args.min_sites)

    run_dir = args.out_dir
    if run_dir is None:
        run_name = build_run_name(
            level=args.level,
            variant=args.variant,
            corridor_name=corridor.road_name,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            epochs=args.epochs,
            use_osmnx=args.use_osmnx,
        )
        run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    corridor.sites.to_csv(run_dir / "selected_corridor_sites.csv", index=False)
    corridor_choice = {"road_name": corridor.road_name, "score": corridor.score, "reason": corridor.reason}
    write_json(run_dir / "corridor_choice.json", corridor_choice)

    print(f"Selected corridor: {corridor.road_name}")
    print(corridor.reason)

    structured_graph = is_structured_variant(args.variant)

    print("Building site-level corridor graph...")
    site_adj_norm, corridor_nodes = build_corridor_graph(
        corridor.sites,
        use_osmnx=args.use_osmnx,
        structured=structured_graph,
    )
    pd.DataFrame(site_adj_norm, index=corridor_nodes["SITE_NO"], columns=corridor_nodes["SITE_NO"]).to_csv(
        run_dir / "adjacency_matrix_site.csv"
    )
    corridor_nodes.to_csv(run_dir / "corridor_nodes.csv", index=False)

    print("Preparing site-level matrix...")
    site_matrix = build_site_matrix(site_ts, corridor_nodes)
    site_matrix.to_csv(run_dir / "corridor_site_matrix.csv")

    template_path = generate_approach_mapping_template(long_df, corridor_nodes, run_dir / "approach_mapping_template.csv")
    detector_template = build_detector_template(long_df, corridor_nodes)
    template_summary = {
        "n_sites": int(detector_template["site_no"].nunique()),
        "n_detectors": int(len(detector_template)),
    }
    write_json(run_dir / "approach_template_summary.json", template_summary)

    matrix = site_matrix
    adjacency = site_adj_norm
    node_table_name = "corridor_nodes.csv"
    value_matrix_name = "corridor_site_matrix.csv"
    mapping_summary: dict[str, object] | None = None

    if args.level == "approach":
        print("Validating mapping and building approach-level artifacts...")
        mapping_df = pd.read_csv(args.mapping_csv)
        approach_ts, validated_mapping, coverage = build_approach_timeseries(
            long_df=long_df,
            mapping_df=mapping_df,
            corridor_nodes=corridor_nodes,
            allow_partial_sites=False,
        )
        coverage.to_csv(run_dir / "approach_mapping_coverage.csv", index=False)
        validated_mapping.to_csv(run_dir / "validated_approach_mapping.csv", index=False)
        approach_ts.to_csv(run_dir / "approach_timeseries.csv", index=False)

        adjacency, approach_nodes = build_approach_graph(
            approach_ts,
            corridor_nodes,
            structured=structured_graph,
        )
        approach_nodes.to_csv(run_dir / "approach_nodes.csv", index=False)
        matrix = build_generic_matrix(approach_ts, approach_nodes, node_key_col="node_id", value_col="approach_volume")
        matrix.to_csv(run_dir / "corridor_approach_matrix.csv")
        pd.DataFrame(adjacency, index=approach_nodes["node_id"], columns=approach_nodes["node_id"]).to_csv(
            run_dir / "adjacency_matrix_approach.csv"
        )

        node_table_name = "approach_nodes.csv"
        value_matrix_name = "corridor_approach_matrix.csv"
        mapping_summary = {
            "mapping_csv": str(args.mapping_csv),
            "ready_sites": coverage.loc[coverage["ready_for_build"], "site_no"].astype(int).tolist(),
            "coverage_csv": "approach_mapping_coverage.csv",
        }

    run_config: dict[str, object] = {
        "level": args.level,
        "variant": args.variant,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "seed": args.seed,
        "use_osmnx": bool(args.use_osmnx),
        "structured_graph": bool(structured_graph),
        "skip_train": bool(args.skip_train),
        "selected_corridor": corridor_choice,
        "template_summary": template_summary,
        "artifacts": {
            "corridor_nodes_csv": "corridor_nodes.csv",
            "approach_mapping_template_csv": template_path.name,
            "node_table_csv": node_table_name,
            "matrix_csv": value_matrix_name,
        },
    }
    if mapping_summary is not None:
        run_config["approach_mapping"] = mapping_summary
    write_json(run_dir / "run_config.json", run_config)

    if args.skip_train:
        print("Skipped training.")
        return

    print(f"Training {args.level}-level {args.variant} baseline...")
    training_summary = run_training(
        matrix=matrix,
        adj_norm=adjacency,
        out_dir=run_dir,
        model_variant=args.variant,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    run_config["training_summary"] = training_summary
    write_json(run_dir / "run_config.json", run_config)
    print(json.dumps(training_summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
