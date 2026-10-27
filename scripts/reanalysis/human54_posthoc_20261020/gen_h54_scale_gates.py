"""Spec E6: headline endpoints against log10 parameters, with comprehension-gate status.

Reads saved outputs only. Writes PDF, SVG and 600 dpi PNG to
results/reanalysis/human54_posthoc_20261020_gates/figures/ (not into the paper).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from scripts.reanalysis.human54_layers_20261019.paper_figures.h54_style import HUMAN, MODEL, MM, plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/reanalysis/human54_posthoc_20261020_gates'
PANELS = [
    ('NB_control_fa', 'n_back', 'load_2_control', '2-back control false alarms'),
    ('ST_interference', 'stroop', 'error_interference', 'Stroop error interference'),
    ('CAL_auroc', 'confidence_calibration', 'type2_auc_fixed_direction', 'Calibration type-2 AUROC'),
    ('WAG_constant', 'post_decision_wagering', 'high_wager_rate', 'High-wager rate'),
]


def main():
    e = pd.read_csv(ROOT / 'results/reanalysis/human54_layers_20261019/layer1/endpoints.csv', low_memory=False)
    e = e[(e.rule == 'valid_only') & (e.included == True)]
    paired = pd.read_csv(ROOT / 'results/reanalysis/human54_layers_20261019/layer1/model_human_endpoints.csv', low_memory=False)
    paired = paired[(paired.rule == 'valid_only') & (paired.human_cohort == 'primary')]
    gates = pd.read_csv(OUT / 'gates.csv').set_index(['model', 'task']).passed
    params = pd.read_csv(OUT / 'model_parameters.csv', index_col='model').params_b
    native = pd.read_csv(ROOT / 'results/reanalysis/human54_layers_20261019/coverage/models.csv', index_col='model').native_thinking
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 52 * MM), constrained_layout=True)
    rows = []
    for ax, (key, task, endpoint, title), letter in zip(axes, PANELS, 'abcd'):
        d = e[(e.task == task) & (e.endpoint == endpoint)].set_index('model').value.astype(float)
        x = np.log10(params[d.index].to_numpy())
        y = d.to_numpy()
        passed = np.array([bool(gates[(m, task)]) for m in d.index])
        thinking = native[d.index].to_numpy(bool)
        ax.scatter(x[passed & ~thinking], y[passed & ~thinking], s=9, color=MODEL, linewidths=0)
        ax.scatter(x[~passed & ~thinking], y[~passed & ~thinking], s=9, facecolors='white', edgecolors=MODEL, linewidths=.6)
        ax.scatter(x[thinking], y[thinking], s=13, marker='^', facecolors=np.where(passed[thinking], MODEL, 'white'),
                   edgecolors=MODEL, linewidths=.6)
        # Non-robust LOWESS (it=0): the robust pass down-weights the single 141B model and bends
        # the tail (verification wf_f60bc7c0-35b). No curve for the 0/1-clustered wager rate.
        if key != 'WAG_constant':
            smooth = lowess(y, x, frac=.6, it=0, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], color=MODEL, lw=1.1)
        human = paired[(paired.task == task) & (paired.endpoint == endpoint)].human_mean
        if len(human):
            ax.axhline(float(human.iloc[0]), color=HUMAN, lw=1, ls='--')
        ax.set_xticks(np.log10([.5, 2, 8, 32, 128]))
        ax.set_xticklabels(['0.5', '2', '8', '32', '128'])
        ax.set_xlabel('Parameters (billions, log scale)')
        ax.set_title(title, fontsize=7)
        ax.text(-.2, 1.08, letter, transform=ax.transAxes, fontsize=8.5, fontweight='bold')
        rows += [dict(panel=letter, model=m, log10_params=xi, value=yi, gate=bool(pi), native_thinking=bool(ti))
                 for m, xi, yi, pi, ti in zip(d.index, x, y, passed, thinking)]
    axes[0].set_ylabel('Model value')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='none', color=MODEL, ms=3.3, label='passes comprehension gate'),
               Line2D([], [], marker='o', ls='none', markerfacecolor='white', markeredgecolor=MODEL, ms=3.3, label='fails gate'),
               Line2D([], [], marker='^', ls='none', color=MODEL, ms=3.8, label='native thinking (filled/open = pass/fail)'),
               Line2D([], [], color=MODEL, lw=1.1, label='LOWESS'),
               Line2D([], [], color=HUMAN, lw=1, ls='--', label='human mean')]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, frameon=False, fontsize=6.5)
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(figures / f'fig_h54_scale_gates.{ext}')
    pd.DataFrame(rows).to_csv(figures / 'fig_h54_scale_gates_plotted_values.csv', index=False)


if __name__ == '__main__':
    main()
