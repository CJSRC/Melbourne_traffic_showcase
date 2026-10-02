"""Compare retail at x5 and x10 with office at x2, per junction changed and per point of share.

Reads the per-junction means written on Spartan by ch5_extract.py (the settled sensitivity
runs) and lu_retail_extra.py collect (the new retail doses, with their own base), and prints
the responses the check is about. Nothing here feeds the chapter yet.

    python docs/report/scripts/lu_retail_extra_report.py
"""

import pathlib

import numpy as np
import pandas as pd
from scipy import stats

REPORT = pathlib.Path(__file__).resolve().parents[1]
DATA = REPORT / "data"
ALPHA = 0.60
ARMS = ["r1attr", "r2attr"]
COLUMN = {"office": "floorshare_office", "retail": "floorshare_retail__shop"}


def responses(frame: pd.DataFrame, arm: str, dose: str, window: str) -> pd.DataFrame:
    """Seeds x junctions."""
    x = frame[(frame.arm == arm) & (frame.window == window)]
    x = x.pivot_table(index=["seed", "SITE_NO"], columns="dose", values="mean_pred")
    return ((x[dose] - x["base"]) / ALPHA).unstack("SITE_NO")


def main() -> None:
    old = pd.read_csv(DATA / "ch5/site_response.csv")
    new = pd.read_csv(DATA / "ch5/site_response_retail_extra.csv")
    attrs = pd.read_csv(DATA / "attributes.csv")
    a24 = attrs[attrs.year == 2024].set_index("SITE_NO")
    sites = sorted(set(pd.read_csv(DATA / "sites81.csv").SITE_NO))
    rows = []
    cases = [("office", "x2_swap", old), ("office", "x2", old), ("retail", "x2_swap", old),
             ("retail", "x2", old), ("retail", "x5_swap", new), ("retail", "x5", new),
             ("retail", "x10_swap", new), ("retail", "x10", new)]
    for arm in ARMS:
        for use, tag, frame in cases:
            share = a24[COLUMN[use]].reindex(sites)
            live = share.index[share > 0]
            m = float(tag.split("_")[0][1:])
            s = share[live]
            # how far the treatment moves the use's share (shares sum to one per junction)
            if tag.endswith("swap"):
                after = np.minimum(s * m, 1.0)
            else:
                after = s * m / (1 + s * (m - 1))
            dpp = float((after - s).mean() * 100)
            for window in ("all", "am"):
                r = responses(frame, arm, f"{use}_{tag}", window)
                every = r[sites].mean(axis=1)             # the chapter's definition: all 81
                changed = r[live].mean(axis=1)            # only the junctions changed
                rows.append({"arm": arm, "treatment": f"{use}_{tag}", "window": window,
                             "junctions": len(live), "share_change_pp": round(dpp, 1),
                             "all81": round(every.mean(), 2),
                             "p_all81": round(stats.ttest_1samp(every, 0)[1], 4),
                             "changed_only": round(changed.mean(), 2),
                             "per_pp": round(changed.mean() / dpp, 3),
                             "seeds_positive": int((every > 0).sum())})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for window in ("all", "am"):
        print(f"\n  window: {window}")
        print(out[out.window == window].drop(columns="window").to_string(index=False))
    out.to_csv(DATA / "ch5/retail_extra_summary.csv", index=False)


if __name__ == "__main__":
    main()
