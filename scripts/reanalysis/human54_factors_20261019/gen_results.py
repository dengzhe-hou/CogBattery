"""Render saved parallel-analysis and factor-stability results, without fitting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the manuscript's existing print typography, font registration and
# editable SVG settings. Its save function is not used: output stays here.
from scripts.reanalysis.human54_layers_20261019.paper_figures import h54_style as style
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "results/reanalysis/human54_factors_20261019"
CASES = ["primary_valid_only", "primary_valid_only_centroids", "invalid_wrong", "invalid_wrong_centroids"]
LABELS = {
    "primary_valid_only": "Valid responses: 53 models",
    "primary_valid_only_centroids": "Valid responses: 23 family centroids",
    "invalid_wrong": "Invalid as wrong: 54 models",
    "invalid_wrong_centroids": "Invalid as wrong: 23 family centroids",
}
PC_COLOR = style.MODEL
FA_COLOR = style.CONTRAST
CORR_CMAP = LinearSegmentedColormap.from_list("h54_factor_correlation", [style.MODEL, "#FFFFFF", style.CONTRAST])


def read_json(path):
    return json.loads(Path(path).read_text())


def save(fig, output, stem, rows):
    """Exact physical width; PDF, editable SVG and 600 dpi PNG plus data."""
    output.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(output / f"{stem}.{suffix}", dpi=600)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f"{stem}_plotted_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)


def label_panel(ax, letter):
    ax.text(-.12, 1.06, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom")


def parallel_figure(source, output, manifest):
    fig, axes = plt.subplots(2, 2, figsize=(180 * style.MM, 140 * style.MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.15, top=.86, hspace=.40, wspace=.17)
    rows = []
    for letter, case, ax in zip("abcd", CASES, axes.ravel()):
        pa = read_json(source / case / "parallel.json")
        rank = np.arange(1, pa["tasks"] + 1)
        for kind, color in [("pc", PC_COLOR), ("fa", FA_COLOR)]:
            values = np.asarray(pa[kind]["eigenvalues"], dtype=float)
            threshold = np.asarray(pa[kind]["threshold95"], dtype=float)
            ax.plot(rank, values, color=color, marker="o", markersize=2.5, linewidth=1.1)
            ax.plot(rank, threshold, color=color, linestyle="--", linewidth=1.1)
            rows.extend(dict(case=case, panel=letter, kind=kind, rank=int(j), eigenvalue=float(value),
                             threshold95=float(cutoff), retained=pa[kind]["retained"])
                        for j, value, cutoff in zip(rank, values, threshold))
        ax.axhline(0, color=".65", linewidth=.6, zorder=0)
        ax.set_title(LABELS[case], loc="left", fontsize=8, pad=20)
        ax.text(0, 1.015, f"Retained: PC {pa['pc']['retained']}; common factors {pa['fa']['retained']}",
                transform=ax.transAxes, fontsize=7, va="bottom")
        label_panel(ax, letter)
        ax.set_xlim(.7, pa["tasks"] + .3)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13])
        counts = pa["null_common_factor_fit_counts"]
        ax.text(.98, .96,
                f"FA null valid {pa['n_valid_fa_null']}/{pa['draws']}\n"
                f"Boundary {counts['boundary']}; improper {pa['draws'] - counts['proper']}",
                ha="right", va="top", transform=ax.transAxes, fontsize=6)
        rows.append(dict(case=case, panel=letter, kind="diagnostic", null_valid=pa["n_valid_fa_null"],
                         null_failed=pa["n_failed_fa_null"], null_boundary=counts["boundary"],
                         null_improper=pa["draws"] - counts["proper"]))
    for ax in axes[:, 0]:
        ax.set_ylabel("Eigenvalue")
    for ax in axes[-1]:
        ax.set_xlabel("Ordered component / common-factor rank")
    handles = [Line2D([], [], color=PC_COLOR, marker="o", label="PC observed"),
               Line2D([], [], color=PC_COLOR, linestyle="--", label="PC 95% permutation line"),
               Line2D([], [], color=FA_COLOR, marker="o", label="Common factor observed"),
               Line2D([], [], color=FA_COLOR, linestyle="--", label="Common factor 95% permutation line")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995), ncol=2,
               frameon=False, columnspacing=1.2, handlelength=2.5)
    fig.text(.08, .045, f"{manifest['pa_draws']:,} independent column permutations; separate PC and common-factor reference spectra.\n"
             "Common-factor diagonals use one-factor MINRES communalities; all 13 ranks are shown.", fontsize=7)
    if manifest["synthetic"]:
        fig.text(.99, .005, "SYNTHETIC DATA", ha="right", fontsize=7)
    save(fig, output, "fig_h54_parallel_analysis", rows)


def factor_figure(source, output, manifest, k):
    case = "primary_valid_only"
    frame = pd.read_csv(source / case / f"efa_{k}" / "pattern.csv", index_col=0)
    tasks = frame.index.tolist()
    point = frame.to_numpy(dtype=float)
    fit = read_json(source / case / f"efa_{k}" / "fit.json")
    stability = read_json(source / case / "stability.json")["dimensions"][str(k)]
    intervals = stability["summaries"]["converged_finite"]["intervals"]
    counts = stability["counts"]
    low = np.asarray(intervals["pattern"]["lower"], dtype=float)
    high = np.asarray(intervals["pattern"]["upper"], dtype=float)
    factor_rows = 1 if k <= 4 else 2
    factor_cols = int(np.ceil(k / factor_rows))
    height_mm = 177 if factor_rows == 1 else 255
    fig = plt.figure(figsize=(180 * style.MM, height_mm * style.MM))
    grid = fig.add_gridspec(factor_rows + 1, factor_cols,
                           left=.105, right=.97, bottom=.18, top=.895,
                           height_ratios=[1.65] * factor_rows + [1], hspace=.5, wspace=.23)
    ymin, ymax = -0.6, len(tasks) - .4
    xmin = min(-.5, float(np.nanmin(low)) - .08, float(np.min(point)) - .08)
    xmax = max(1.05, float(np.nanmax(high)) + .08, float(np.max(point)) + .08)
    rows = []
    for factor in range(k):
        ax = fig.add_subplot(grid[factor // factor_cols, factor % factor_cols])
        y = np.arange(len(tasks))
        ax.hlines(y, low[:, factor], high[:, factor], color=style.MODEL, alpha=.45, linewidth=1.1)
        ax.plot(point[:, factor], y, "o", color=style.MODEL, markersize=3)
        ax.axvline(0, color=".6", linewidth=.7, zorder=0)
        ax.set_ylim(ymax, ymin)
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"F{factor + 1}", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(tasks if factor % factor_cols == 0 else [""] * len(tasks))
        ax.set_xlabel("Pattern loading")
        ax.set_xticks([value for value in [-1, -.5, 0, .5, 1, 1.5] if xmin < value < xmax])
        if factor == 0:
            label_panel(ax, "a")
        rows.extend(dict(panel="a", factors=k, task=task, factor=factor + 1, point=point[j, factor],
                         lower=low[j, factor], upper=high[j, factor],
                         median=intervals["pattern"]["median"][j][factor]) for j, task in enumerate(tasks))
    bottom = grid[-1, :].subgridspec(1, 2, width_ratios=[1, 1.4], wspace=.45)
    phi_ax = fig.add_subplot(bottom[0])
    phi = np.asarray(fit["phi"])
    image = phi_ax.imshow(phi, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="equal")
    for i in range(k):
        for j in range(k):
            phi_ax.text(j, i, f"{phi[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(phi[i, j]) > .72 else "black")
            rows.append(dict(panel="b", factors=k, factor=i + 1, other_factor=j + 1, point=phi[i, j],
                             lower=intervals["phi"]["lower"][i][j],
                             median=intervals["phi"]["median"][i][j],
                             upper=intervals["phi"]["upper"][i][j]))
    phi_ax.set_xticks(range(k), [f"F{i + 1}" for i in range(k)])
    phi_ax.set_yticks(range(k), [f"F{i + 1}" for i in range(k)])
    phi_ax.set_title("Observed factor correlations", fontsize=8, pad=12)
    label_panel(phi_ax, "b")
    fig.colorbar(image, ax=phi_ax, fraction=.05, pad=.07, ticks=[-1, 0, 1])
    congruence_ax = fig.add_subplot(bottom[1])
    c = intervals["congruence"]
    cm, cl, cu = np.asarray(c["median"]), np.asarray(c["lower"]), np.asarray(c["upper"])
    congruence_ax.errorbar(cm, np.arange(k), xerr=[cm - cl, cu - cm], fmt="o", color=style.MODEL,
                         elinewidth=1.1, capsize=2, markersize=3)
    congruence_ax.set_yticks(np.arange(k), [f"F{i + 1}" for i in range(k)])
    congruence_ax.set_ylim(k - .5, -.5)
    congruence_ax.set_xlim(0, 1.02)
    congruence_ax.set_xticks([0, .5, 1])
    congruence_ax.set_title("Alignment to observed factors", fontsize=8, pad=12)
    congruence_ax.set_xlabel("Tucker congruence (bootstrap median, 95% CI)")
    label_panel(congruence_ax, "c")
    rows.extend(dict(panel="c", factors=k, factor=i + 1, median=cm[i], lower=cl[i], upper=cu[i]) for i in range(k))
    rows.append(dict(panel="diagnostic", factors=k, **counts))
    fig.suptitle(f"{k}-factor MINRES / oblimin solution: 53 models, 23 families", fontsize=9, y=.982)
    fig.legend(handles=[Line2D([], [], color=style.MODEL, marker="o", linestyle="none", label="Observed loading"),
                        Line2D([], [], color=style.MODEL, alpha=.45, linewidth=1.1, label="95% family-bootstrap interval")],
               loc="upper center", bbox_to_anchor=(.54, .956), ncol=2, frameon=False)
    summary = (f"Observed fit: RMSR {fit['rmsr']:.3f}; converged {fit['converged']}; proper {fit['proper']}; boundary {fit['boundary']}.\n"
               f"Bootstrap: {counts['converged_finite']}/{counts['attempted']} converged and finite; {counts['failed']} failed; "
               f"{counts['finite_boundary']} boundary; {counts['finite_improper']} improper.\n"
               "Intervals condition on this fixed factor count and converged, finite resamples; boundary/improper fits are retained.\n"
               "Factors align by column permutation and signs only. Boundary and improper counts can overlap.")
    fig.text(.08, .055, summary, fontsize=6.5, linespacing=1.5)
    if manifest["synthetic"]:
        fig.text(.99, .005, "SYNTHETIC DATA", ha="right", fontsize=7)
    save(fig, output, f"fig_h54_factors_k{k}", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    source = args.input
    manifest = read_json(source / "run_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only completed saved analyses may be rendered")
    output = source / "figures"
    parallel_figure(source, output, manifest)
    for k in manifest["efa_candidate_dimensions"]:
        factor_figure(source, output, manifest, k)
    readme = """# Saved-results figures

Generated by `scripts/reanalysis/human54_factors_20261019/gen_results.py`.
This renderer reads saved results only and does not fit or select models.

All figures use the existing human54 Arial style, at 180 mm width, with
editable SVG text, vector PDF and 600 dpi PNG. Every figure has a CSV ledger.

Parallel-analysis figure: all 13 eigenvalue ranks are shown for all four
configurations. Blue denotes principal components and orange denotes common
factors throughout; dashed lines are their separate 95th-percentile
permutation references. Common-factor diagonals are the saved one-factor
MINRES communalities. Null-fit failures, boundary and improper counts are
reported, with no replacement draws hidden from the plot.

Factor figures: every candidate factor count is rendered for the primary
53-model matrix. Panel a shows observed oblimin pattern loadings for all 13
tasks and 95% whole-family bootstrap percentile intervals. Panel b shows
observed factor correlations (all correlation intervals remain in the CSV).
Panel c shows median Tucker congruence and its 95% interval after permutation
and sign alignment only. Intervals condition on the specified fixed factor
count and on converged, finite fits, including boundary and improper fits.
The figure reports all attempted, failed, boundary and improper draw counts;
boundary and improper counts can overlap. No factor names are imposed.

Reproduction:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.reanalysis.human54_factors_20261019.gen_results --input PATH`
"""
    (output / "README.md").write_text(readme)
    print(output)


if __name__ == "__main__":
    main()
