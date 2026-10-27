#!/usr/bin/env python3
"""Plot saved human54 parallel analysis, EFA and family-held-out prediction.

No model responses are read and no estimates, intervals or fits are recomputed.
The plot deliberately distinguishes EFA from the separate prediction models.
"""
import json

from matplotlib.colors import LinearSegmentedColormap

from h54_style import CONTRAST, HUMAN, MM, MODEL, OUT, ROOT, csv, np, plt


def main():
    factor = ROOT / "results/reanalysis/human54_factors_20261019"
    full = ROOT / "results/reanalysis/human54_fullbattery_20261019"
    pa = json.loads((factor / "primary_valid_only/parallel.json").read_text())
    stability = json.loads((factor / "primary_valid_only/stability.json").read_text())
    prediction = json.loads((full / "summary.json").read_text())["configurations"]
    loadings = list(csv.DictReader((factor / "primary_valid_only/efa_2/pattern.csv").open()))
    used = []
    fig = plt.figure(figsize=(180 * MM, 74 * MM))
    a = fig.add_axes([.058, .23, .205, .635])
    b = fig.add_axes([.337, .23, .100, .635])
    c = fig.add_axes([.520, .23, .172, .635])
    d = fig.add_axes([.785, .23, .195, .635])
    for ax, tag, title in [(a, "a", "Parallel analysis"),
                           (b, "b", "Pattern loadings"),
                           (c, "c", "Loading stability"),
                           (d, "d", "Cross-family prediction")]:
        ax.text(0, 1.09, tag, transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", va="bottom")
        ax.set_title(title, fontsize=7, pad=5)

    rank = np.arange(1, 14)
    observed = np.array(pa["fa"]["eigenvalues"])
    threshold = np.array(pa["fa"]["threshold95"])
    assert len(observed) == len(threshold) == 13
    assert pa["fa"]["retained"] == 2
    a.plot(rank, observed, color=MODEL, marker="o", markersize=2.6,
           linewidth=1.1, label="Observed")
    a.plot(rank, threshold, color=HUMAN, linestyle="--", linewidth=1,
           label="95% null")
    a.axhline(0, color=".75", linewidth=.5, zorder=0)
    a.set_xlim(.6, 13.4)
    a.set_ylim(-.85, 5.05)
    a.set_xticks([1, 4, 7, 10, 13])
    a.set_yticks([0, 2, 4])
    a.set_xlabel("Rank")
    a.set_ylabel("Eigenvalue", labelpad=2)
    a.legend(loc="upper right", frameon=False, fontsize=6,
             handlelength=1.3, handletextpad=.3, borderpad=.1)
    for j, (obs, null) in enumerate(zip(observed, threshold), start=1):
        used.append({"panel": "a", "configuration": "primary_valid_only",
                     "measure": "common_factor_parallel_analysis", "item": j,
                     "estimate": obs, "upper": null, "n": 5000,
                     "source": "primary_valid_only/parallel.json"})

    matrix = np.array([[float(r["F1"]), float(r["F2"])] for r in loadings])
    assert matrix.shape == (13, 2) and np.all(np.abs(matrix) <= 1)
    cmap = LinearSegmentedColormap.from_list("h54_pattern", [CONTRAST, "#FFFFFF", MODEL])
    im = b.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    b.set_xticks([0, 1], ["F1", "F2"])
    b.set_yticks(np.arange(13), [r["task"] for r in loadings])
    b.tick_params(axis="both", length=0, pad=3)
    for spine in b.spines.values():
        spine.set_visible(False)
    for i, row in enumerate(loadings):
        for j, key in enumerate(("F1", "F2")):
            value = matrix[i, j]
            label = f"{value:.2f}".replace("0.", ".").replace("-", "−")
            b.text(j, i, label, ha="center", va="center", fontsize=5.7,
                   color="white" if abs(value) > .7 else HUMAN)
            used.append({"panel": "b", "configuration": "primary_valid_only",
                         "measure": "pattern_loading", "item": row["task"],
                         "factor": key, "estimate": value, "n": 53,
                         "source": "primary_valid_only/efa_2/pattern.csv"})
    cax = fig.add_axes([.337, .115, .100, .025])
    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
    colorbar.outline.set_linewidth(.5)

    summary = stability["dimensions"]["2"]["summaries"]["converged_finite"]
    intervals = summary["intervals"]["congruence"]
    assert summary["n"] == 2000
    for j, (name, color) in enumerate((("F1", MODEL), ("F2", CONTRAST))):
        point, low, high = (intervals[k][j] for k in ("median", "lower", "upper"))
        assert low <= point <= high
        y = 1-j
        c.errorbar(point, y, xerr=[[point-low], [high-point]],
                   fmt="o", color=color, elinewidth=1.1, capsize=2, capthick=.7)
        c.text(.04, y + .20, f"{name}: {point:.3f} [{low:.3f}, {high:.3f}]",
               ha="left", va="bottom", fontsize=6)
        used.append({"panel": "c", "configuration": "primary_valid_only",
                     "measure": "fixed_k2_tucker_congruence", "item": name,
                     "estimate": point, "lower": low, "upper": high,
                     "n": summary["n"], "source": "primary_valid_only/stability.json"})
    c.set_xlim(0, 1.06)
    c.set_ylim(-.65, 1.65)
    c.set_xticks([0, .5, 1], ["0", ".5", "1"])
    c.set_yticks([])
    c.spines["left"].set_visible(False)
    c.set_xlabel("Tucker congruence")

    configs = [("primary_valid_only", "Primary (53)"),
               ("invalid_wrong", "Invalid=wrong (54)"),
               ("wag_highlow_sensitivity", "WAG contrast (16)")]
    for j, (key, label) in enumerate(configs):
        row = prediction[key]
        point, low, high = (100 * row[k] for k in
                            ("group_gain", "group_gain_lower", "group_gain_upper"))
        assert low <= point <= high
        y = 2-j
        d.errorbar(point, y, xerr=[[point-low], [high-point]],
                   fmt="o", color=MODEL, elinewidth=1.1, capsize=2, capthick=.7)
        d.text(-14.5, y + .19, label, ha="left", va="bottom", fontsize=6.5)
        used.append({"panel": "d", "configuration": key,
                     "measure": "relative_rmse_gain_percent", "item": "five_group_information",
                     "estimate": point, "lower": low, "upper": high,
                     "n": row["models"], "families": row["families"],
                     "source": "human54_fullbattery_20261019/summary.json"})
    d.axvline(0, color=".6", linestyle="--", linewidth=.7, zorder=0)
    d.set_xlim(-15, 10)
    d.set_ylim(-.55, 2.55)
    d.set_xticks([-10, 0, 10])
    d.set_yticks([])
    d.spines["left"].set_visible(False)
    d.set_xlabel("RMSE gain (%)")

    fig.canvas.draw()
    stem = "fig_h54_latent_summary"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600)
    fields = list(dict.fromkeys(k for r in used for k in r))
    with (OUT / f"{stem}_plotted_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(used)
    plt.close(fig)


if __name__ == "__main__":
    main()
