"""The land-use response split by road class, in windows taken from the data.

Two corrections the supervisor asked for, and one he did not.

  main against local   "four vehicles per fifteen minutes" averaged a trunk road carrying 1200
                       vehicles at the morning peak with a laneway carrying 15. Splitting them
                       was expected to lift the main-road figure; the class gradient behind the
                       split says whether it does, and whether the line was drawn in the right
                       place (OSM tertiary in the CBD covers Collins Street and Little Bourke
                       Street alike).

  the peak windows     the figures used 06:30-08:30 and 16:30-18:30, chosen by habit. The
                       recorded peaks sit at 08:30 and 17:15, so the morning window stopped
                       almost exactly where the peak began.

  the weekend          nobody asked, but the weekend has no commuter peaks at all -- it has one
                       midday hump. Testing "is retail more sensitive at the weekend?" inside a
                       weekday commuter window could only ever have answered no.

Windows are found in recorded flow rather than hard-coded, so they cannot drift from the data
again. Responses are divided by the 0.60 blend weight throughout, so they are the model's own.

    python outputs/experiment_ledger_v1/lu_roadclass_report.py
"""

import pathlib

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRATCH = pathlib.Path("C:/Users/LIANGZ~1/AppData/Local/Temp/claude/D--Melbourne/"
                       "6730435c-09fe-491f-9366-7890a4f73ed3/scratchpad")
# all 81 junctions of the settled model, not only the 79 that are also in the 371
# trainable list: the two missing ones are trunk roads carrying 734 and 1205 vehicles
# per slot, and dropping them moved the base profile by about 20 vehicles
PROFILE = SCRATCH / "lu_profile_all81.csv"
FIGURES = pathlib.Path(__file__).resolve().parents[1] / "phase2_offnadir_masking_v1"
ALPHA = 0.60
SEEDS = [42, 73, 128, 2024, 7]
ARMS = [("r1attr", "Route 1 + attributes"), ("r2attr", "Route 2 + attributes")]
USES = [("office", "office"), ("resi", "residential"), ("retail", "retail")]
DOSE_TAGS = ["x0.5", "x1.5", "x2", "x3", "x2_swap"]
CLASSES = ["trunk", "primary", "secondary", "tertiary", "unclassified", "service"]
OLD_WINDOWS = {"weekday": [("AM", 26, 35), ("PM", 66, 75), ("evening", 76, 89)],
               "weekend": [("AM", 26, 35), ("PM", 66, 75)]}

# reference palette, light mode; validated all-pairs for these three slots
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
USE_COLOUR = {"office": "#eb6834", "resi": "#2a78d6", "retail": "#1baf7a"}
CLASS_RAMP = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]


def peak_windows(d: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Five slots centred on each peak of the recorded flow, over every site."""
    base = d[(d.dose == "base") & (d.arm == "r1attr") & (d.road_group == "all")]
    out = {}
    for daytype, spans in (("weekday", {"AM": (24, 44), "PM": (56, 80)}),
                           ("weekend", {"midday": (40, 64)})):
        g = base[base.daytype == daytype].groupby("slot_idx").mean_true.mean()
        g = g.reindex(range(96))
        for label, (lo, hi) in spans.items():
            peak = int(g[lo:hi].idxmax())
            out[f"{daytype} {label}"] = (peak - 2, peak + 3)
    return out


def clock(slot: int) -> str:
    return f"{slot // 4:02d}:{slot % 4 * 15:02d}"


def response(d, dose, arm, group, daytype, window) -> np.ndarray:
    """Per-seed response in vehicles per slot."""
    lo, hi = window
    sel = d[(d.arm == arm) & (d.road_group == group) & (d.daytype == daytype) &
            (d.slot_idx >= lo) & (d.slot_idx < hi) & d.dose.isin(["base", dose])]
    out = []
    for seed in SEEDS:
        s = sel[sel.seed == seed]
        b = s[s.dose == "base"].set_index("slot_idx").mean_pred
        t = s[s.dose == dose].set_index("slot_idx").mean_pred
        common = b.index.intersection(t.index)
        out.append(float((t[common] - b[common]).mean() / ALPHA) if len(common) else np.nan)
    return np.array(out)


def base_flow(d, group, daytype, window) -> float:
    lo, hi = window
    s = d[(d.dose == "base") & (d.arm == "r1attr") & (d.seed == 42) &
          (d.road_group == group) & (d.daytype == daytype) &
          (d.slot_idx >= lo) & (d.slot_idx < hi)]
    return float(s.mean_true.mean()) if len(s) else np.nan


def site_count(d, group) -> int:
    s = d[(d.dose == "base") & (d.arm == "r1attr") & (d.seed == 42) &
          (d.road_group == group) & (d.daytype == "weekday")]
    # rows per slot over 2024's weekdays; the "all" group anchors the per-site figure
    per_slot = s.n.max()
    everyone = d[(d.dose == "base") & (d.arm == "r1attr") & (d.seed == 42) &
                 (d.road_group == "all") & (d.daytype == "weekday")].n.max()
    return int(round(per_slot / everyone * 81)) if per_slot else 0


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_windows_and_classes(d, windows) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2),
                             gridspec_kw={"height_ratios": [1, 1.05]})
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Where the land-use response lives: peak windows and road class",
                 fontsize=15, fontweight="bold", color=INK, x=0.04, ha="left", y=0.985)
    fig.text(0.04, 0.948, "Settled model, 81 junctions, test year 2024. Office swap arm "
             "(office share doubled, floorspace fixed). Response divided by the 0.60 blend "
             "weight. Five seeds.", fontsize=10, color=INK2, ha="left")

    hours = np.arange(96) / 4
    base = d[(d.dose == "base") & (d.arm == "r1attr") & (d.road_group == "all")]
    for ax, daytype in zip(axes[0], ("weekday", "weekend")):
        style(ax)
        g = base[base.daytype == daytype].groupby("slot_idx").mean_true.mean().reindex(range(96))
        ax.plot(hours, g.to_numpy(), color=INK, linewidth=2)
        top = float(g.max()) * 1.12
        for label, lo, hi in OLD_WINDOWS[daytype]:
            ax.axvspan(lo / 4, hi / 4, ymin=0, ymax=0.5, facecolor="none", edgecolor=MUTED,
                       hatch="///", linewidth=0)
        for key, (lo, hi) in windows.items():
            if key.startswith(daytype):
                ax.axvspan(lo / 4, hi / 4, color=USE_COLOUR["office"], alpha=0.16, linewidth=0)
                peak = int(g[lo:hi].idxmax())
                ax.annotate(f"{key.split()[1]} peak {clock(peak)}\nwindow "
                            f"{clock(lo)}–{clock(hi)}", (peak / 4, g[peak]),
                            textcoords="offset points", xytext=(0, 12), ha="center",
                            fontsize=9, color=INK)
        ax.set_ylim(0, top)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
        ax.set_ylabel("recorded flow (vehicles per 15 min)", color=INK2)
        ax.set_title(f"{daytype}: recorded flow over all junctions", loc="left",
                     fontsize=12, color=INK, pad=8)
    fig.legend(handles=[
        Patch(facecolor=USE_COLOUR["office"], alpha=0.16, label="window found in the data"),
        Patch(facecolor="none", edgecolor=MUTED, hatch="///", label="window used before"),
    ], frameon=False, fontsize=9.5, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.965))

    # bottom left: recorded flow by class, bottom right: response by class
    window = windows["weekday AM"]
    classes = [c for c in CLASSES if site_count(d, c) > 0]
    y = np.arange(len(classes))[::-1]
    labels = [f"{c}  ({site_count(d, c)})" for c in classes]

    ax = axes[1, 0]
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    flows = [base_flow(d, c, "weekday", window) for c in classes]
    ax.barh(y, flows, height=0.62, color=CLASS_RAMP[:len(classes)], zorder=2)
    for yi, f in zip(y, flows):
        ax.text(f + 12, yi, f"{f:.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel("recorded flow, weekday AM peak (vehicles per 15 min)", color=INK2)
    ax.set_title("the arterials carry the traffic ...", loc="left", fontsize=12,
                 color=INK, pad=8)
    ax.set_xlim(0, max(flows) * 1.12)

    ax = axes[1, 1]
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    reach, shares = {}, {}
    for offset, (arm, arm_label), marker in zip((0.17, -0.17), ARMS, ("o", "D")):
        for yi, c in zip(y, classes):
            r = response(d, "office_x2_swap", arm, c, "weekday", window)
            ok = r[~np.isnan(r)]
            ax.plot([ok.min(), ok.max()], [yi + offset] * 2, color=USE_COLOUR["office"],
                    linewidth=2, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.scatter(ok.mean(), yi + offset, marker=marker, s=60, zorder=3,
                       color=USE_COLOUR["office"], edgecolor=SURFACE, linewidth=2)
            reach[yi] = max(reach.get(yi, -np.inf), ok.max())
            shares.setdefault(yi, []).append(ok.mean() / base_flow(d, c, "weekday", window) * 100)
    for yi in y:
        ax.annotate(f"{shares[yi][0]:.2f}% / {shares[yi][1]:.2f}% of its flow",
                    (reach[yi], yi), textcoords="offset points", xytext=(10, 0),
                    fontsize=8.5, color=INK2, va="center")
    ax.set_xlim(right=max(reach.values()) + 3.4)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel("response to the office swap, weekday AM peak (vehicles per 15 min)",
                  color=INK2)
    ax.set_title("... but the response sits on the smaller roads", loc="left",
                 fontsize=12, color=INK, pad=8)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=USE_COLOUR["office"], lw=0, markersize=7,
               label="Route 1 + attributes"),
        Line2D([], [], marker="D", color=USE_COLOUR["office"], lw=0, markersize=6,
               label="Route 2 + attributes"),
        Line2D([], [], color=USE_COLOUR["office"], lw=2, alpha=0.45, label="range over seeds"),
    ], frameon=False, fontsize=9, loc="upper right", title="labels: Route 1 / Route 2",
       title_fontsize=8.5, alignment="left")
    fig.text(0.51, 0.015, "Brackets give the number of junctions in each OSM class. "
             "Unclassified and service rest on three and two junctions.",
             fontsize=9, color=MUTED, ha="left")

    fig.tight_layout(rect=(0, 0.03, 1, 0.935))
    path = FIGURES / "landuse_roadclass_gradient.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_profile_shape(d) -> pathlib.Path:
    """Why the arterials do not respond: they are already full for most of the day.

    Two readings fit "trunk roads barely move": their traffic is passing through and does not
    care what is built beside it, or they are at capacity and cannot grow. The shape of the
    day separates them. A road with spare capacity has a peak; a road at capacity spreads,
    because the demand above capacity has nowhere to go but earlier and later. Each class is
    divided by its own daily mean so the comparison is of shape, not size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("The arterials run near their peak for half the day; the smaller roads do not",
                 fontsize=14, fontweight="bold", color=INK, x=0.04, ha="left", y=0.98)
    fig.text(0.04, 0.925, "Weekday recorded flow in the test year, each road class divided by its "
             "own daily mean. A spread peak is what a road at capacity looks like, though "
             "flow alone cannot prove capacity is the cause.", fontsize=10, color=INK2, ha="left")

    base = d[(d.dose == "base") & (d.arm == "r1attr") & (d.seed == 42) &
             (d.daytype == "weekday")]
    hours = np.arange(96) / 4
    ax = axes[0]
    style(ax)
    rows = []
    for colour, cls in zip(CLASS_RAMP, CLASSES):
        g = base[base.road_group == cls].sort_values("slot_idx")
        if g.empty:
            continue
        v = g.mean_true.to_numpy()
        ax.plot(hours, v / v.mean(), color=colour, linewidth=2,
                label=f"{cls}  ({site_count(d, cls)})")
        rows.append((cls, colour, v.mean(), float((v >= 0.9 * v.max()).sum()) * 0.25))
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_ylabel("flow relative to that class's daily mean", color=INK2)
    ax.set_title("shape of the weekday", loc="left", fontsize=12, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)

    ax = axes[1]
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[3] for r in rows], height=0.6, color=[r[1] for r in rows], zorder=2)
    for yi, r in zip(y, rows):
        ax.text(r[3] + 0.2, yi, f"{r[3]:.2f} h", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlabel("hours a day within 90% of that class's own peak", color=INK2)
    ax.set_title("how much of the day sits at the peak", loc="left", fontsize=12,
                 color=INK, pad=8)
    ax.set_xlim(0, max(r[3] for r in rows) * 1.25)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = FIGURES / "landuse_roadclass_shape.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_main_local(d, windows) -> pathlib.Path:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4), sharey="row")
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Does excluding local roads lift the response?  Swap arms, all three uses",
                 fontsize=15, fontweight="bold", color=INK, x=0.04, ha="left", y=0.985)
    fig.text(0.04, 0.945, "Main = OSM trunk, primary, secondary. Local = tertiary and below. "
             "Bars are the mean over five seeds, dots the seeds. p from a paired test, "
             "main against local.", fontsize=10, color=INK2, ha="left")
    width = 0.36
    for row, (arm, arm_label) in enumerate(ARMS):
        for col, (key, window) in enumerate(windows.items()):
            ax = axes[row, col]
            style(ax)
            daytype = key.split()[0]
            for i, (use, use_label) in enumerate(USES):
                m = response(d, f"{use}_x2_swap", arm, "main", daytype, window)
                l = response(d, f"{use}_x2_swap", arm, "local", daytype, window)
                colour = USE_COLOUR[use]
                for offset, values, solid in ((-width / 2, m, True), (width / 2, l, False)):
                    x = i + offset
                    # the 2px surface gap between neighbouring bars comes from the edge
                    ax.bar(x, np.nanmean(values), width, color=colour,
                           alpha=1.0 if solid else 0.38, hatch=None if solid else "////",
                           edgecolor=SURFACE, linewidth=2, zorder=2)
                    ax.scatter([x] * len(values), values, s=14, color=INK, zorder=4,
                               linewidths=0, alpha=0.7)
                p = float(stats.ttest_rel(m, l)[1])
                top = np.nanmax(np.concatenate([m, l]))
                ax.text(i, top + 0.35, f"p={p:.3f}" + (" *" if p < 0.05 else ""),
                        ha="center", fontsize=8.5,
                        color=INK if p < 0.05 else MUTED,
                        fontweight="bold" if p < 0.05 else "normal")
            ax.axhline(0, color=AXIS, linewidth=1, zorder=3)
            ax.margins(y=0.12)
            ax.set_xticks(range(len(USES)))
            ax.set_xticklabels([u[1] for u in USES], color=INK)
            lo, hi = window
            ax.set_title(f"{arm_label}  ·  {key} {clock(lo)}–{clock(hi)}", loc="left",
                         fontsize=11, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel("response (vehicles per 15 min)", color=INK2)
    fig.legend(handles=[
        Patch(facecolor=INK2, label="main roads"),
        Patch(facecolor=INK2, alpha=0.38, hatch="////", label="local roads"),
        Line2D([], [], marker="o", color=INK, lw=0, markersize=4, label="seed"),
    ], frameon=False, fontsize=9.5, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = FIGURES / "landuse_main_vs_local.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    d = pd.read_csv(PROFILE)
    doses = set(d.dose)
    missing = [f"{u}_{t}" for u, _ in USES for t in DOSE_TAGS if f"{u}_{t}" not in doses]
    counts = d[d.road_group == "all"].groupby("dose").seed.nunique()
    short = counts[counts < len(SEEDS)]
    print(f"  {len(d)} rows, {len(doses)} doses" +
          (f"; MISSING {missing}" if missing else "") +
          (f"; fewer than five seeds: {dict(short)}" if len(short) else ""))

    windows = peak_windows(d)
    print("\n  windows found in recorded flow (five slots centred on the peak)")
    for key, (lo, hi) in windows.items():
        print(f"    {key:<16}{clock(lo)} - {clock(hi)}")

    print("\n  recorded flow in each window (vehicles per 15 min)")
    print(f"  {'group':<14}" + "".join(f"{k:>17}" for k in windows) + f"{'sites':>8}")
    for group in ["all", "main", "local"] + CLASSES:
        print(f"  {group:<14}" + "".join(
            f"{base_flow(d, group, k.split()[0], w):>17.0f}" for k, w in windows.items())
              + f"{site_count(d, group):>8}")

    print("\n  swap arms: response (vehicles per 15 min) and share of the base flow there")
    for use, use_label in USES:
        print(f"\n  {use_label}")
        print(f"  {'arm':<9}{'group':<9}" + "".join(f"{k:>24}" for k in windows))
        for arm, _ in ARMS:
            for group in ("all", "main", "local"):
                cells = []
                for key, w in windows.items():
                    r = response(d, f"{use}_x2_swap", arm, group, key.split()[0], w)
                    p = float(stats.ttest_1samp(r, 0.0)[1])
                    flow = base_flow(d, group, key.split()[0], w)
                    cells.append(f"{np.mean(r):>+9.2f} ({np.mean(r) / flow * 100:>5.2f}%)"
                                 f"{'*' if p < 0.05 else ' '}")
                print(f"  {arm:<9}{group:<9}" + "".join(f"{c:>24}" for c in cells))

    print("\n  main minus local over every dose, paired by seed")
    tally = {"main moves more *": 0, "local moves more *": 0, "no difference": 0}
    print(f"  {'use':<13}{'arm':<9}" + "".join(f"{k:>18}" for k in windows) +
          "   (main-local per dose: x0.5 x1.5 x2 x3 swap)")
    for use, use_label in USES:
        for arm, _ in ARMS:
            parts = []
            for key, w in windows.items():
                marks = []
                for tag in DOSE_TAGS:
                    m = response(d, f"{use}_{tag}", arm, "main", key.split()[0], w)
                    l = response(d, f"{use}_{tag}", arm, "local", key.split()[0], w)
                    p = float(stats.ttest_rel(m, l)[1])
                    # x0.5 takes office away, so its responses are negative; compare how
                    # strongly each group moves, not which number is larger
                    diff = (np.mean(m) - np.mean(l)) * (-1 if tag == "x0.5" else 1)
                    if p < 0.05:
                        tally["main moves more *" if diff > 0 else "local moves more *"] += 1
                        marks.append("M" if diff > 0 else "L")
                    else:
                        tally["no difference"] += 1
                        marks.append(".")
                parts.append("".join(marks))
            print(f"  {use_label:<13}{arm:<9}" + "".join(f"{p:>18}" for p in parts))
    print("  M = main moves significantly more, L = local moves significantly more, . = no difference")
    total = sum(tally.values())
    print("  " + "   ".join(f"{k}: {v}/{total}" for k, v in tally.items()))

    print()
    for path in (figure_windows_and_classes(d, windows), figure_main_local(d, windows),
                 figure_profile_shape(d)):
        print(f"  -> {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
