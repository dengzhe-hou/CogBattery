#!/usr/bin/env python3
"""Plot existing CVLT, DRM, calibration and wagering endpoint estimates."""
from h54_style import *


def main():
    data, used = panel_data(), []
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 103 * MM))
    fig.subplots_adjust(left=.075, right=.985, bottom=.155, top=.865,
                        wspace=.40, hspace=.82)
    ax = axes[0, 0]
    rows = select(data, "cvlt_word_list", [f"learning_{i}" for i in range(1, 6)], "a", used)
    compare(ax, rows, x=range(1, 6), connected=True)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 16.6)
    ax.set_yticks([0, 8, 16])
    ax.set_xlabel("CVLT: learning trial")
    ax.set_ylabel("Words recalled")
    for ax, condition, label, panel in (
        (axes[0, 1], "no_warning", "no warning", "b"),
        (axes[0, 2], "warning", "warning", "c"),
    ):
        rows = select(data, "drm_false_memory", [f"{condition}|{s}" for s in (
            "hit_rate", "lure_false_alarm_rate", "new_false_alarm_rate")], panel, used)
        compare(ax, rows)
        ax.set_xticks(range(3), ["Old", "Lure", "New"])
        ax.set_xlabel(f"DRM: {label}")
        ax.set_ylabel("OLD response rate")
        proportion_axis(ax)
    ax = axes[1, 0]
    rows = select(data, "confidence_calibration", ["accuracy", "type2_auc_fixed_direction"], "d", used)
    compare(ax, rows)
    ax.set_xticks(range(2), ["Accuracy", "Type-2\nAUROC"])
    ax.set_xlabel("CAL: performance / monitoring")
    ax.set_ylabel("Proportion / AUROC")
    proportion_axis(ax)
    ax = axes[1, 1]
    rows = select(data, "confidence_calibration", ["official_bias"], "e", used)
    compare(ax, rows)
    ax.set_xticks([0], ["Confidence bias"])
    ax.set_xlabel("CAL: source-defined bias")
    ax.set_ylabel("Bias (original scale)")
    ax.set_ylim(-.05, .35)
    ax.set_yticks([0, .1, .2, .3])
    ax.axhline(0, color=".65", linewidth=.6, linestyle="--", zorder=0)
    ax = axes[1, 2]
    rows = select(data, "post_decision_wagering", ["accuracy", "high_minus_low_accuracy"], "f", used)
    compare(ax, rows)
    ax.set_xticks(range(2), [f"Accuracy\n({rows[0]['models']}/54)", f"High − low\n({rows[1]['models']}/54)"])
    ax.set_xlabel("WAG: endpoint (models)")
    ax.set_ylabel("Accuracy or difference")
    ax.set_ylim(-.15, 1.04)
    ax.set_yticks([0, .5, 1])
    ax.axhline(0, color=".65", linewidth=.6, linestyle="--", zorder=0)
    for ax, tag in zip(axes.flat, "abcdef"):
        letter(ax, tag)
    legend(fig)
    save(fig, "fig_h54_memory_metacognition", used)


if __name__ == "__main__":
    main()
