#!/usr/bin/env python3
"""Plot precomputed DS, NB, ST and FL condition means and intervals."""
from h54_style import *


def main():
    data, used = panel_data(), []
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 99 * MM))
    fig.subplots_adjust(left=.075, right=.985, bottom=.145, top=.865,
                        wspace=.40, hspace=.77)
    for index, direction in enumerate(("forward", "backward")):
        ax = axes[0, index]
        rows = select(data, "digit_span", [f"{direction}_length_{i}" for i in range(3, 10)],
                      chr(97 + index), used)
        compare(ax, rows, x=range(3, 10), connected=True)
        ax.set_xticks(range(3, 10))
        ax.set_xlabel(f"DS {direction}: list length")
        ax.set_ylabel("Sequence accuracy")
        proportion_axis(ax)
    for ax, load, panel in ((axes[0, 2], 2, "c"), (axes[1, 0], 3, "d")):
        rows = select(data, "n_back", [f"load_{load}_{s}" for s in ("target", "lure", "control")], panel, used)
        compare(ax, rows)
        ax.set_xticks(range(3), ["Target", "Lure", "Control"])
        ax.set_xlabel(f"NB {load}-back: trial class")
        ax.set_ylabel("MATCH rate")
        proportion_axis(ax)
    for ax, task, name, conds, panel in (
        (axes[1, 1], "stroop", "ST", ("congruent", "incongruent"), "e"),
        (axes[1, 2], "flanker", "FL", ("congruent", "incongruent", "neutral"), "f"),
    ):
        rows = select(data, task, [f"{c}_accuracy" for c in conds], panel, used)
        compare(ax, rows)
        ax.set_xticks(range(len(conds)), [dict(congruent="Cong.", incongruent="Incong.", neutral="Neutral")[c] for c in conds])
        ax.set_xlabel(f"{name}: condition")
        ax.set_ylabel("Accuracy")
        proportion_axis(ax)
    for ax, tag in zip(axes.flat, "abcdef"):
        letter(ax, tag)
    legend(fig)
    save(fig, "fig_h54_behavior", used)


if __name__ == "__main__":
    main()
