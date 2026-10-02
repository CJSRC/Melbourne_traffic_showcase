"""Derive the 371-junction runlist from the settled one, changing only the data it points at.

The settled series is g_p500_tr_ln_*: 500 m frame, projection fitted on the training sites over
2014-2019 plus 2022, blend alpha searched, anchor 2022, validation 2023, test 2024, epochs 18.
runlist_settled_e40.txt is that same series with epochs 40 and nothing else moved, which was
checked token by token, so it is a safe base to derive from at either epoch count.

Six substitutions, all of them data paths, plus the run name and the epoch count. Nothing about
the model, the split, the anchor or the blend is touched, and the token difference is printed so
that can be checked rather than taken on trust. Three of this project's confounds came from
arguments retyped by hand; none of them survived a printed diff.

    python make_371_runlist.py --epochs 18 > runlist_a371.txt
"""

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "runlist_settled_e40.txt"
N = "/data/gpfs/projects/punim2970/xhe13561/nearmap_v1"

# every one of these is a file the run reads; none of them is a model setting
PATHS = {
    f"{N}/data/subsets/area_sites_attr.csv":
        f"{N}/clue446/area_sites_trainable.csv",
    f"{N}/data/subsets/area_adjacency_attr.csv":
        f"{N}/clue446/area_adjacency_trainable.csv",
    f"{N}/data/attribute_features/slot_wide_2014_2026_with2019.parquet":
        f"{N}/clue446/slot_wide_clue446.parquet",
    f"{N}/features500/pooled_500_k22_tr.csv":
        f"{N}/features500_371/pooled_500_k22_tr371.csv",
    f"{N}/data/attribute_features/site_year_attributes_sq500m.csv":
        f"{N}/clue446/attribute_features/site_year_attributes_sq500m.csv",
    # the route 2 network table is per site-year like the rest; the 143-site one covers only
    # 79 of the 371, and a junction with no row is given a zero vector rather than an error
    f"{N}/route2_embeddings/net.csv":
        f"{N}/clue446/route2_embeddings/net.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18,
                        help="18 matches g_p500_tr_ln_*; 40 matches the e40 series")
    args = parser.parse_args()

    lines = [l.rstrip("\n") for l in SRC.read_text().splitlines() if l.strip()]
    out, proof = [], []
    for line in lines:
        name, rest = line.split("\t", 1)
        if not name.startswith("e40_"):
            raise SystemExit(f"unexpected run name in the source runlist: {name}")
        new_name = "a371_" + name[len("e40_"):]
        new_rest = rest
        for old, new in PATHS.items():
            new_rest = new_rest.replace(old, new)
        new_rest = new_rest.replace("--epochs 40", f"--epochs {args.epochs}")
        for old in PATHS:
            if old in new_rest:
                raise SystemExit(f"a path survived substitution in {new_name}: {old}")
        out.append(f"{new_name}\t{new_rest}")
        if len(proof) < 2:
            a, b = set(rest.split()), set(new_rest.split())
            proof.append((name, new_name, sorted(a - b), sorted(b - a)))

    for name, new_name, gone, added in proof:
        print(f"# {name} -> {new_name}", file=sys.stderr)
        for g in gone:
            print(f"#   - {g}", file=sys.stderr)
        for a in added:
            print(f"#   + {a}", file=sys.stderr)
    print(f"# {len(out)} runs at epochs {args.epochs}", file=sys.stderr)
    for line in out:
        print(line)


if __name__ == "__main__":
    main()
