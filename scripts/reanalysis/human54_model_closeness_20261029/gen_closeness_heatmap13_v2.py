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
          ('Metacognition', [('CAL', 'Confidence'), ('WAG', 'Wagering')])]
APPROX = {'SM', 'EP', 'WAG'}
CODES = [c for _, ts in GROUPS for c, _ in ts]
NAMES = {c: n for _, ts in GROUPS for c, n in ts}

d = pd.read_csv(OUT / 'model_task_z.csv')
order = d.groupby('model').params_b.first().sort_values().index.tolist()
z = d.pivot(index='task', columns='model', values='z').loc[CODES, order]
a = z.abs()
bins = [0, .5, 1, 2, np.inf]
colors = ['#f1f1f1', '#a6cbe3', '#3f88c5', '#0b3c6b']
cmap = ListedColormap(colors)
cmap.set_bad('#d9d2c3')
norm = BoundaryNorm(bins[:-1] + [1e9], cmap.N)

fig = plt.figure(figsize=(180 * style.MM, 70 * style.MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1, .085], height_ratios=[1, .24], wspace=.02, hspace=.06,
                      left=.285, right=.985, top=.91, bottom=.225)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.ma.masked_invalid(a.to_numpy()), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
row = 0
for gi, (gname, ts) in enumerate(GROUPS):
    if gi:
        ax.axhline(row - .5, color='white', lw=2.2)
    ax.annotate(gname, xy=(0, row + (len(ts) - 1) / 2), xycoords=('axes fraction', 'data'), xytext=(-76, 0),
                textcoords='offset points', ha='right', va='center', fontsize=6, fontweight='bold')
    row += len(ts)
ax.set_yticks(range(len(CODES)), [f'{NAMES[c]} ({c}){"$^\\dagger$" if c in APPROX else ""}' for c in CODES], fontsize=6)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.tick_params(axis='y', length=0)
for s in ax.spines.values():
    s.set_visible(False)

# Right: share of models more than 1 SD away (WAG among the 17 models with a defined value).
axr = fig.add_subplot(gs[0, 1], sharey=ax)
share = (a > 1).sum(axis=1) / a.notna().sum(axis=1)
axr.barh(range(len(CODES)), share.to_numpy(), color='#0b3c6b', height=.62)
axr.axvline(.7, color='#888888', lw=.6, ls=(0, (2, 2)))
axr.set_xlim(0, 1)
axr.set_xticks([0, .7, 1], ['0', '.7', '1'], fontsize=5)
axr.set_title('>1 SD', fontsize=6, pad=2)
axr.tick_params(axis='y', left=False, labelleft=False)
axr.tick_params(axis='x', labelsize=5)
for s in ('top', 'right', 'left'):
    axr.spines[s].set_visible(False)

# Bottom: each model's median distance across its defined tasks.
axb = fig.add_subplot(gs[1, 0], sharex=ax)
med = a.median(axis=0)
axb.plot(range(len(order)), med.to_numpy(), color='#0b3c6b', lw=.9, marker='o', ms=1.8)
axb.set_ylabel('Median\ndistance', fontsize=6)
axb.set_ylim(0, max(3.0, float(med.max()) * 1.05))
axb.set_xticks(range(len(order)), order, rotation=90, fontsize=5)
axb.set_xlim(-.5, len(order) - .5)
for s in ('top', 'right'):
    axb.spines[s].set_visible(False)

handles = [Patch(color=c, label=l) for c, l in zip(colors, ['< .5 SD', '.5–1', '1–2', '> 2'])]
handles.append(Patch(color='#d9d2c3', label='constant wager'))
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.285, 1.0), ncol=5, fontsize=5.5, frameon=False,
           title=None, handlelength=1.2, columnspacing=1.2)
fig.text(.985, .972, 'Distance from human mean, in human SD.  $^\\dagger$ approximate SD', fontsize=5.5, va='center', ha='right')
for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'fig_h54_closeness_v2.{ext}', **kw)
if '--paper' in sys.argv:
    fig.savefig(OUT / 'fig_h54_closeness.pdf')
