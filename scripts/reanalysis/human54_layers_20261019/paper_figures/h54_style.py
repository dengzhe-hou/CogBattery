"""Shared, print-size style for the human54 manuscript figures.

Arial registration/export settings follow the project's
sync_branches_20261010_v2/figures.py:arial_rc. That script also installs an old
analysis write guard, so it cannot be imported as a standalone style module.
The current requested 7 pt print typography supersedes its old 9 pt floor.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "results/reanalysis/human54_layers_20261019"
OUT = ROOT / "paper-overleaf/figures_h54"
OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
HUMAN = "#222222"
MODEL = "#0072B2"
CONTRAST = "#A85400"

for font in (Path.home() / ".local/share/fonts/arial").glob("*.TTF"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": .6,
    "lines.linewidth": 1.1, "lines.markersize": 3.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 130, "savefig.dpi": 600,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial", "axes.unicode_minus": True,
})


def panel_data():
    rows = list(csv.DictReader((DATA / "layer1/panel_differences.csv").open()))
    return {(r["task"], r["endpoint"]): r for r in rows
            if r["panel"] == "all54" and r["family_mapping"] == "merged"
            and r["rule"] == "valid_only" and r["human_cohort"] == "primary"}


def select(data, task, endpoints, panel, used):
    rows = [data[(task, key)] for key in endpoints]
    used.extend(dict(r, figure_panel=panel) for r in rows)
    return rows


def compare(ax, rows, x=None, connected=False):
    """Display existing model CIs: difference CI plus fixed human reference."""
    x = np.arange(len(rows), dtype=float) if x is None else np.array(x, dtype=float)
    human = np.array([float(r["human_mean"]) for r in rows])
    model = np.array([float(r["model_mean"]) for r in rows])
    low = np.array([float(r["lower"]) for r in rows]) + human
    high = np.array([float(r["upper"]) for r in rows]) + human
    # Direct check catches a wrong CI column/scale before it is drawn.
    assert np.all(low <= model) and np.all(model <= high)
    delta = 0 if connected else .09
    line = "-" if connected else "none"
    ax.plot(x - delta, human, color=HUMAN, marker="o", linestyle=line,
            markerfacecolor="white", markeredgewidth=.9, zorder=4)
    ax.errorbar(x + delta, model, yerr=[model - low, high - model],
                color=MODEL, marker="s", linestyle=line, elinewidth=1,
                capsize=1.8, capthick=.7, zorder=3)
    if len(x) > 1:
        ax.set_xlim(x[0] - .35, x[-1] + .35)
    else:
        ax.set_xlim(-.6, .6)


def letter(ax, value):
    ax.text(-.20, 1.04, value, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom")


def proportion_axis(ax):
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, .5, 1])


def legend(fig):
    fig.legend(handles=[
        Line2D([], [], color=HUMAN, marker="o", markerfacecolor="white",
               markeredgewidth=.9, linestyle="none", label="Human mean"),
        Line2D([], [], color=MODEL, marker="s", linestyle="none",
               label="Model mean and 95% CI"),
    ], loc="upper center", bbox_to_anchor=(.53, .995), ncol=2,
       frameon=False, handletextpad=.5, columnspacing=2)


def save(fig, stem, rows):
    """Retain exact 180 mm width; no tight-bbox resizing or text outlining."""
    fig.canvas.draw()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / f"{stem}_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)
