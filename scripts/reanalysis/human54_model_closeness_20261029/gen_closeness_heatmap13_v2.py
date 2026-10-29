"""Paper Figure 3 (v2, with --paper written to the Overleaf figures folder as fig_h54_closeness.pdf) (2026-10-29): binned absolute distance from the human mean in human SDs,
tasks grouped with full names, right bars = share of models over 1 SD, bottom = each model's
median distance across tasks. Same z values as gen_closeness_heatmap13.py."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/reanalysis/human54_layers_20261019/paper_figures'))
import h54_style as style  # noqa: E402  (fonts and sizes)

OUT = ROOT / 'results/reanalysis/human54_model_closeness_20261029/thirteen'
GROUPS = [('Working memory', [('DS', 'Digit span'), ('NB', 'N-back'), ('OS', 'Operation span')]),
          ('Cognitive control', [('ST', 'Stroop'), ('FL', 'Flanker'), ('GN', 'Go/no-go')]),
          ('Episodic memory', [('CVLT', 'List learning'), ('DRM', 'False memory'), ('SM', 'Source memory')]),
          ('Social reasoning', [('FB', 'False belief'), ('EP', 'Mindreading')]),
          ('Metacognition', [('CAL', 'Calibration'), ('WAG', 'Wagering')])]
APPROX = {'SM', 'EP', 'WAG'}
CODES = [c for _, ts in GROUPS for c, _ in ts]
NAMES = {c: n for _, ts in GROUPS for c, n in ts}

d = pd.read_csv(OUT / 'model_task_z.csv')
order = d.groupby('model').params_b.first().sort_values().index.tolist()
z = d.pivot(index='task', columns='model', values='z').loc[CODES, order]
a = z.abs()
bins = [0, .5, 1, 2, np.inf]
colors = ['#f2f2f2', '#b8d4ea', '#4a90c8', '#0b3c6b']
CONST = '#f2c99b'  # constant wagers: no distance, so outside the blue scale
cmap = ListedColormap(colors)
cmap.set_bad(CONST)
norm = BoundaryNorm(bins[:-1] + [1e9], cmap.N)

# Designed at the text width (160.5 mm) so that every font prints at its nominal size.
fig = plt.figure(figsize=(160.5 * style.MM, 66 * style.MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1, .085], height_ratios=[1, .24], wspace=.02, hspace=.06,
                      left=.285, right=.985, top=.915, bottom=.27)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.ma.masked_invalid(a.to_numpy()), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
row = 0
for gi, (gname, ts) in enumerate(GROUPS):
    if gi:
        ax.axhline(row - .5, color='white', lw=2.2)
    ax.annotate(gname, xy=(0, row + (len(ts) - 1) / 2), xycoords=('axes fraction', 'data'), xytext=(-76, 0),
                textcoords='offset points', ha='right', va='center', fontsize=6, fontweight='bold')
    row += len(ts)
ax.set_yticks(range(len(CODES)), [f'{NAMES[c]} ({c}){" †" if c in APPROX else ""}' for c in CODES], fontsize=6)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.tick_params(axis='y', length=0)
for s in ax.spines.values():
    s.set_visible(False)

# Right: share of models more than 1 SD away (WAG among the 17 models with a defined value).
axr = fig.add_subplot(gs[0, 1], sharey=ax)
share = (a > 1).sum(axis=1) / a.notna().sum(axis=1)
axr.barh(range(len(CODES)), share.to_numpy(), color=style.MODEL, height=.62)
axr.axvline(.7, color='#888888', lw=.6, ls=(0, (2, 2)))
axr.set_xlim(0, 1)
axr.set_xticks([0, .7, 1], ['0', '.7', '1'], fontsize=5)
axr.set_title('Share >1 SD', fontsize=5.5, pad=2)
axr.tick_params(axis='y', left=False, labelleft=False)
axr.tick_params(axis='x', labelsize=5)
for s in ('top', 'right', 'left'):
    axr.spines[s].set_visible(False)

# Bottom: each model's median distance across its defined tasks.
axb = fig.add_subplot(gs[1, 0], sharex=ax)
med = a.median(axis=0)
axb.plot(range(len(order)), med.to_numpy(), color=style.MODEL, lw=.9, marker='o', ms=1.8)
axb.annotate('Median distance (SD)', xy=(0, .5), xycoords='axes fraction', xytext=(-14, 0),
             textcoords='offset points', ha='right', va='center', fontsize=6)
axb.tick_params(axis='y', labelsize=5)
axb.set_ylim(0, max(3.0, float(med.max()) * 1.05))
axb.set_xticks(range(len(order)), order, rotation=90, fontsize=5)
axb.set_xlim(-.5, len(order) - .5)
# Size landmarks: first model at or above 1B, 10B, and 100B served parameters.
size = d.groupby('model').params_b.first().loc[order].to_numpy()
for cut, lab in ((1, '1B'), (10, '10B'), (100, '100B')):
    i = int(np.argmax(size >= cut))
    axb.axvline(i - .5, color='#888888', lw=.6, ls=(0, (2, 2)))
    right = i > len(order) - 4  # keep the last label inside the panel
    axb.text(i - .7 if right else i - .3, axb.get_ylim()[1], f'$\\geq${lab}', fontsize=5.5, va='top',
             ha='right' if right else 'left', color='#555555')
for s in ('top', 'right'):
    axb.spines[s].set_visible(False)

handles = [Patch(color=c, label=l) for c, l in zip(colors, ['< .5 SD', '.5–1 SD', '1–2 SD', '> 2 SD'])]
handles.append(Patch(color=CONST, label='constant wager'))
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.28, 1.0), ncol=5, fontsize=5.5, frameon=False,
           title=None, handlelength=1.2, columnspacing=1.2)
fig.text(.278, .972, 'Distance from human mean', fontsize=6, fontweight='bold', va='center', ha='right')
fig.text(.905, .972, '† approximate SD', fontsize=5.5, va='center', ha='right')
for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'fig_h54_closeness_v2.{ext}', **kw)
if '--paper' in sys.argv:
    fig.savefig(ROOT / 'paper-rev-20261021/figures_h54/fig_h54_closeness.pdf')
