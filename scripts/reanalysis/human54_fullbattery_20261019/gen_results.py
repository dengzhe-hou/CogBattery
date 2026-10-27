"""Plot saved full-battery outputs without recomputing any statistics.

Run from the project root with:
python -m scripts.reanalysis.human54_fullbattery_20261019.gen_results
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Import the project's established human54 typography, not its paper save helper.
from scripts.reanalysis.human54_layers_20261019.paper_figures.h54_style import MM
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "results/reanalysis/human54_fullbattery_20261019"
OUT = DATA / "figures"
CONFIGURATIONS = [
    ("primary_valid_only", "Primary", "#0072B2", "o"),
    ("invalid_wrong", "Invalid as wrong", "#D55E00", "s"),
    ("wag_highlow_sensitivity", "WAG high−low", "#009E73", "^"),
]
TASK_CODES = {
    "digit_span": "DS", "n_back": "NB", "operation_span": "OS",
    "stroop": "ST", "flanker": "FL", "go_nogo": "GN",
    "cvlt_word_list": "CVLT", "drm_false_memory": "DRM", "source_monitoring": "SM",
    "false_belief": "FB", "epitome_tom": "EP",
    "confidence_calibration": "CAL", "post_decision_wagering": "WAG",
}


def _read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _panel(ax, letter, title):
    ax.text(-.16, 1.07, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom")
    ax.set_title(title, fontsize=8, loc="left", pad=8)


def main():
    summaries = {row["name"]: row for row in _read_csv(DATA / "summary.csv")}
    pcas = {name: json.loads((DATA / name / "pca.json").read_text())
            for name, _, _, _ in CONFIGURATIONS}
    primary = pcas["primary_valid_only"]
    tasks = primary["tasks"]
    if len(tasks) != 13 or any(pca["tasks"] != tasks for pca in pcas.values()):
        raise ValueError("The saved configurations must contain the same 13 ordered tasks")
    target_rows = {TASK_CODES[row["target"]]: row for row in
                   _read_csv(DATA / "primary_valid_only/prediction_per_target.csv")}
    if set(target_rows) != set(tasks):
        raise ValueError("Prediction target labels do not match the saved PCA task labels")
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(180 * MM, 150 * MM))
    axes = [fig.add_axes(rect) for rect in [
        [.10, .59, .35, .31], [.61, .59, .35, .31],
        [.10, .12, .35, .31], [.61, .12, .35, .31],
    ]]
    a, b, c, d = axes
    plotted = []

    _panel(a, "a", "Variance across all 13 PCs")
    x = np.arange(1, 14)
    for name, label, color, marker in CONFIGURATIONS:
        shares = np.asarray(pcas[name]["variance_shares"]) * 100
        a.plot(x, shares, color=color, marker=marker, markersize=3.2, label=label)
        for pc, share in zip(x, shares):
            plotted.append(dict(panel="a", configuration=name, item=f"PC{pc}",
                                value=float(share), unit="variance percent"))
    a.set(xlabel="Principal component", ylabel="Variance explained (%)",
          xticks=[1, 3, 5, 7, 9, 11, 13], xlim=(.7, 13.3), ylim=(0, 46))
    a.set_yticks([0, 10, 20, 30, 40])

    _panel(b, "b", "Primary PC1 correlation loadings")
    y = np.arange(len(tasks))
    loadings = np.asarray(primary["correlation_loadings"])[:, 0]
    b.axvline(0, color="#888888", linewidth=.7, zorder=1)
    b.barh(y, loadings, color="#0072B2", height=.65, zorder=2)
    b.set(yticks=y, yticklabels=tasks, xlabel="Correlation with PC1", xlim=(-.3, 1.0))
    b.set_xticks([0, .5, 1])
    b.invert_yaxis()
    for task, value in zip(tasks, loadings):
        plotted.append(dict(panel="b", configuration="primary_valid_only", item=task,
                            value=float(value), unit="correlation loading"))

    _panel(c, "c", "Gain over general-axis predictor")
    config_labels = []
    for row_index, (name, label, color, marker) in enumerate(CONFIGURATIONS):
        row = summaries[name]
        point = float(row["group_gain"]) * 100
        low, high = (float(row[field]) * 100
                     for field in ("group_gain_lower", "group_gain_upper"))
        c.plot([low, high], [row_index, row_index], color=color, linewidth=1.2)
        c.plot([low, high], [row_index, row_index], color=color, marker="|",
               linestyle="none", markersize=5)
        c.plot(point, row_index, color=color, marker=marker, linestyle="none", markersize=4)
        short_label = {"primary_valid_only": "Primary",
                       "invalid_wrong": "Invalid as\nwrong",
                       "wag_highlow_sensitivity": "WAG\nhigh−low"}[name]
        config_labels.append(f"{short_label}\n{row['models']} / {row['families']}")
        plotted.append(dict(panel="c", configuration=name, item="relative RMSE gain",
                            value=point, lower=low, upper=high, unit="percent"))
    c.axvline(0, color="#888888", linewidth=.7, zorder=1)
    c.set(yticks=range(3), yticklabels=config_labels, xlabel="Relative RMSE gain (%)",
          xlim=(-15, 10), ylim=(2.5, -.5), xticks=[-10, 0, 10])
    c.tick_params(axis="y", length=0)
    c.text(0, -.30, "Labels: models / families; bars: 95% CI", transform=c.transAxes,
           fontsize=7, va="top")

    _panel(d, "d", "Primary target gains (no CI)")
    target_gain = np.array([float(target_rows[task]["relative_rmse_gain"]) * 100 for task in tasks])
    d.axvline(0, color="#888888", linewidth=.7, zorder=1)
    d.barh(y, target_gain, color="#0072B2", height=.65, zorder=2)
    d.set(yticks=y, yticklabels=tasks, xlabel="Relative RMSE gain (%)",
          xlim=(-12, 45), xticks=[-10, 0, 20, 40])
    d.invert_yaxis()
    for task, value in zip(tasks, target_gain):
        plotted.append(dict(panel="d", configuration="primary_valid_only", item=task,
                            value=float(value), unit="percent; no CI"))

    fig.legend(handles=[Line2D([], [], color=color, marker=marker, markersize=3.3,
                             label=f"{label} ({summaries[name]['models']} / {summaries[name]['families']})")
                        for name, label, color, marker in CONFIGURATIONS],
               loc="upper center", bbox_to_anchor=(.53, .995), ncol=3,
               frameon=False, handlelength=1.5, columnspacing=1.7, handletextpad=.5)
    fig.canvas.draw()
    stem = "fig_h54_fullbattery"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=600)
    with (OUT / f"{stem}_plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["panel", "configuration", "item", "value",
                                                   "lower", "upper", "unit"])
        writer.writeheader()
        writer.writerows(plotted)
    plt.close(fig)
    (OUT / "README.md").write_text(
        "# Full-battery human54 results figure\n\n"
        "The figure displays saved post-hoc results; this script does not fit PCA, "
        "prediction models, or bootstrap samples. Reproduce with "
        "`python -m scripts.reanalysis.human54_fullbattery_20261019.gen_results`.\n\n"
        "`fig_h54_fullbattery.pdf`, `.svg`, and `.png` are 180 mm wide and 150 mm high. "
        "PDF/SVG are vector formats, SVG text stays editable, and PNG is 600 dpi. "
        "The established human54 style supplies Arial 7 pt text, axis labels 7.5 pt, "
        "8 pt panel titles, and 8.5 pt bold panel letters.\n\n"
        "a, Full variance spectrum for all 13 task endpoints under all three configurations. "
        "Primary valid-only uses 53 complete models in 23 merged families; invalid-as-wrong "
        "uses 54 models in 23 families; the WAG high-minus-low sensitivity uses 16 complete "
        "models in 9 families. WAG high-minus-low is defined for 17 models before complete-case "
        "filtering. The primary and invalid-as-wrong configurations use WAG type-2 AUROC. "
        "No scree confidence intervals are drawn.\n\n"
        "b, Primary 53-model PC1 correlation loadings, calculated in the saved PCA analysis "
        "as each unit eigenvector coefficient times the square root of its eigenvalue. "
        "These are correlation loadings, not unit eigenvector coefficients. PC1 sign follows "
        "the saved positive-sum convention. No loading confidence intervals are drawn.\n\n"
        "c, Relative RMSE gain from adding a same-group peer task predictor to the general-only "
        "baseline, whose general-axis predictor is PC1 of the other 12 tasks. Gain equals "
        "1 - RMSE(group-augmented) / RMSE(general-only baseline); "
        "positive values indicate lower prediction error. All three configurations are shown. "
        "Out-of-family prediction uses leave-one-family-out fitting. The reported RMSE gives "
        "equal weight to family-by-target cells. Horizontal intervals are the existing 95% "
        "percentile CIs from 20,000 paired whole-family resamples of fixed out-of-fold errors, "
        "without model refitting. They do not represent model-level resampling.\n\n"
        "d, The same relative gain, separately for all 13 targets in the primary configuration. "
        "These are point estimates only; no per-target confidence intervals were computed or "
        "drawn. All task orderings match the saved PCA matrix rather than effect size.\n\n"
        "DS, digit span; NB, n-back; OS, operation span; ST, Stroop; FL, flanker; GN, go/no-go; "
        "CVLT, word-list learning; DRM, false memory; SM, source memory; FB, false belief; "
        "EP, EPITOME theory of mind; CAL, confidence calibration; WAG, post-decision wagering.\n\n"
        "Inputs are `../summary.csv`, each configuration's `pca.json`, and "
        "`../primary_valid_only/prediction_per_target.csv`. The plotted-values CSV lists every "
        "displayed point and CI endpoint without recalculating its underlying statistic.\n"
    )
    print(f"Saved {len(plotted)} plotted values and PDF/SVG/600-dpi PNG to {OUT}")


if __name__ == "__main__":
    main()
