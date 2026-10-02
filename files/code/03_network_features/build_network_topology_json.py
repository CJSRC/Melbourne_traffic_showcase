from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build site-year network topology JSON files.")
    parser.add_argument("--study-area-csv", type=Path, required=True)
    parser.add_argument("--adjacency-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--max-two-hop", type=int, default=6)
    parser.add_argument("--compact-schema", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def round_float(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(args.study_area_csv)
    sites["SITE_NO"] = pd.to_numeric(sites["SITE_NO"], errors="coerce").astype(int)
    sites = sites.sort_values(["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    site_lookup = sites.set_index("SITE_NO")

    adjacency = pd.read_csv(args.adjacency_csv, index_col=0)
    adjacency.index = adjacency.index.astype(int)
    adjacency.columns = adjacency.columns.astype(int)

    manifest = pd.read_csv(args.manifest_csv)
    manifest["SITE_NO"] = pd.to_numeric(manifest["SITE_NO"], errors="coerce").astype(int)
    manifest["year"] = pd.to_numeric(manifest["year"], errors="coerce").astype(int)
    if args.limit_rows is not None:
        manifest = manifest.head(args.limit_rows).copy()

    neighbor_map: dict[int, list[dict[str, object]]] = {}
    two_hop_map: dict[int, list[int]] = {}
    for site_no in adjacency.index:
        weights = adjacency.loc[site_no]
        weights = weights[weights > 0]
        weights = weights.drop(index=site_no, errors="ignore")
        sorted_neighbors = weights.sort_values(ascending=False)
        neighbors: list[dict[str, object]] = []
        for rank, (neighbor_id, weight) in enumerate(sorted_neighbors.items(), start=1):
            if neighbor_id not in site_lookup.index:
                continue
            neighbor_row = site_lookup.loc[neighbor_id]
            if args.compact_schema:
                neighbors.append(
                    {
                        "r": int(rank),
                        "id": int(neighbor_id),
                        "w": round_float(weight),
                        "lat": round_float(neighbor_row["LATITUDE"], 5),
                        "lon": round_float(neighbor_row["LONGITUDE"], 5),
                    }
                )
            else:
                neighbors.append(
                    {
                        "distance_rank": rank,
                        "site_id": int(neighbor_id),
                        "site_name": str(neighbor_row["SITE_NAME"]),
                        "edge_weight": float(weight),
                        "latitude": float(neighbor_row["LATITUDE"]),
                        "longitude": float(neighbor_row["LONGITUDE"]),
                    }
                )
            if len(neighbors) >= args.max_neighbors:
                break
        neighbor_map[int(site_no)] = neighbors

        neighbor_id_key = "id" if args.compact_schema else "site_id"
        first_hop_ids = [int(item[neighbor_id_key]) for item in neighbors]
        candidate_scores: dict[int, float] = {}
        for neighbor_id in first_hop_ids:
            neighbor_weights = adjacency.loc[int(neighbor_id)]
            neighbor_weights = neighbor_weights[neighbor_weights > 0]
            neighbor_weights = neighbor_weights.drop(index=[site_no, *first_hop_ids], errors="ignore")
            for hop2_id, hop2_weight in neighbor_weights.items():
                candidate_scores[int(hop2_id)] = max(candidate_scores.get(int(hop2_id), 0.0), float(hop2_weight))
        ranked_two_hop = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        two_hop_map[int(site_no)] = [site_id for site_id, _ in ranked_two_hop[: args.max_two_hop]]

    index_rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        site_no = int(record["SITE_NO"])
        site_row = site_lookup.loc[site_no]
        if args.compact_schema:
            payload = {
                "sid": site_no,
                "yr": int(record["year"]),
                "lat": round_float(site_row["LATITUDE"], 5),
                "lon": round_float(site_row["LONGITUDE"], 5),
                "deg": int(len(neighbor_map.get(site_no, []))),
                "nbrs": neighbor_map.get(site_no, []),
                "hop2": two_hop_map.get(site_no, []),
                "hop2n": int(len(two_hop_map.get(site_no, []))),
            }
        else:
            payload = {
                "site_id": site_no,
                "site_name": str(site_row["SITE_NAME"]),
                "municipality": str(site_row.get("MUNICIPALITY", "")),
                "latitude": float(site_row["LATITUDE"]),
                "longitude": float(site_row["LONGITUDE"]),
                "year": int(record["year"]),
                "dataset_split": str(record.get("dataset_split", "")),
                "graph_degree": int(len(neighbor_map.get(site_no, []))),
                "neighbors": neighbor_map.get(site_no, []),
                "two_hop_neighbor_ids": two_hop_map.get(site_no, []),
                "two_hop_neighbor_count": int(len(two_hop_map.get(site_no, []))),
            }
        rel_path = Path(f"year_{int(record['year'])}") / "json" / f"site_{site_no}.json"
        abs_path = out_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        index_rows.append(
            {
                "SITE_NO": site_no,
                "year": int(record["year"]),
                "dataset_split": str(record.get("dataset_split", "")),
                "json_path": str(rel_path),
                "json_type": "network_topology",
            }
        )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(out_dir / "network_topology_json_index.csv", index=False)
    summary = {
        "study_area_csv": str(args.study_area_csv),
        "adjacency_csv": str(args.adjacency_csv),
        "manifest_csv": str(args.manifest_csv),
        "row_count": int(len(index_df)),
        "site_count": int(index_df["SITE_NO"].nunique()),
        "year_count": int(index_df["year"].nunique()),
        "max_neighbors": int(args.max_neighbors),
        "max_two_hop": int(args.max_two_hop),
        "compact_schema": bool(args.compact_schema),
    }
    (out_dir / "network_topology_json_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_dir / "network_topology_json_index.csv")


if __name__ == "__main__":
    main()
