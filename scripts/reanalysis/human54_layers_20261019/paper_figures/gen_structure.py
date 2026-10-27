#!/usr/bin/env python3
"""Display all primary SRO and Ball human-minus-model structure contrasts."""
from h54_style import *


def main():
    rows = list(csv.DictReader((DATA / "layer2/summary.csv").open()))
    data = {r["analysis"]: r for r in rows if r["panel"] == "all54"
            and r["family_mapping"] == "merged" and r["rule"] == "valid_only"}
    fig, axes = plt.subplots(2, 1, figsize=(180 * MM, 111 * MM),
                             gridspec_kw={"height_ratios": [1, 3.45]}, sharex=True)
    fig.subplots_adjust(left=.285, right=.97, bottom=.12, top=.945, hspace=.48)
    used = []

    def forest(ax, keys, labels, tag, positions=None):
        selected = [data[k] for k in keys]
        used.extend(dict(r, figure_panel=tag) for r in selected)
        y = np.arange(len(keys)) if positions is None else np.array(positions)
        point = np.array([float(r["delta_difference"]) for r in selected])
        lo = np.array([float(r["delta_difference_lower"]) for r in selected])
        hi = np.array([float(r["delta_difference_upper"]) for r in selected])
        assert np.all(lo <= point) and np.all(point <= hi)
        ax.errorbar(point, y, xerr=[point-lo, hi-point], fmt="o", color=CONTRAST,
                    markersize=3.7, elinewidth=1.1, capsize=2, capthick=.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.axvline(0, color=".5", linewidth=.7, linestyle="--", zorder=0)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines["left"].set_visible(False)
        ax.text(-.37, 1.08, tag, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="bottom")
        ax.set_xlim(-.35, .90)
        ax.set_xticks([-.25, 0, .25, .5, .75])

    forest(axes[0], ["sro_valid_only_all54_merged", "sro_no_neutral_valid_only_all54_merged"],
           ["SRO: all FL conditions", "SRO: without neutral FL"], "a")
    axes[0].set_ylim(1.5, -.5)
    keys, labels, positions = [], [], []
    for group, display in (("all", "Pooled"), ("no_warning", "No warning"), ("warning", "Warning")):
        for endpoint, representation in (("native", "Native"), ("accuracy_analyzed56", "Accuracy (56)"),
                                         ("balanced_accuracy", "Balanced accuracy")):
            keys.append(f"ball_{group}_{endpoint}_valid_only_all54_merged")
            labels.append(f"{display}: {representation}")
        start = {"all": 0, "no_warning": 3.7, "warning": 7.4}[group]
        positions.extend([start, start+1, start+2])
    forest(axes[1], keys, labels, "b", positions)
    axes[1].set_ylim(10, -.6)
    axes[1].set_xlabel("Human − model grouping contrast, δ difference (95% CI)", labelpad=7)
    save(fig, "fig_h54_structure", used)


if __name__ == "__main__":
    main()
