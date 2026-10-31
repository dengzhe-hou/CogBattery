"""Paper Figure 3 (2026-10-31): binned absolute distance from the human mean, 13 tasks by 54 models,
tasks grouped with full names, right bars = share of models over 1 SD, bottom = each model's
median distance across tasks. Same z values as gen_closeness_heatmap13.py."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/reanalysis/human54_layers_20261019/paper_figures'))
import h54_style as style  # noqa: E402  (fonts and sizes)

OUT = ROOT / 'results/reanalysis/human54_model_closeness_20261029/thirteen'
DATA = OUT / 'model_task_z.csv'
GROUPS = [
    ('Working memory', 'Working\nmemory', [('DS', 'Digit span'), ('NB', 'N-back'), ('OS', 'Operation span')]),
    ('Cognitive control', 'Cognitive\ncontrol', [('ST', 'Stroop'), ('FL', 'Flanker'), ('GN', 'Go/no-go')]),
    ('Episodic memory', 'Episodic\nmemory', [('CVLT', 'CVLT-II'), ('DRM', 'DRM'), ('SM', 'Source memory')]),
    ('Social reasoning', 'Social\nreasoning', [('FB', 'False belief'), ('EP', 'Mindreading')]),
    ('Metacognition', 'Metacognition', [('CAL', 'Calibration'), ('WAG', 'Wagering')]),
]
APPROX = {'SM', 'EP', 'WAG'}
CODES = [c for _, _, tasks in GROUPS for c, _ in tasks]
NAMES = {c: name for _, _, tasks in GROUPS for c, name in tasks}
d = pd.read_csv(DATA)
order = d.groupby('model').params_b.first().sort_values().index.tolist()
z = d.pivot(index='task', columns='model', values='z').loc[CODES, order]
a = z.abs()
share = (a > 1).sum(axis=1) / a.notna().sum(axis=1)
med = a.median(axis=0)
assert a.shape == (13, 54)
assert a.notna().sum(axis=1).loc['WAG'] == 17
assert a.isna().sum(axis=1).loc['WAG'] == 37

colors = ['#f2f2f2', '#b8d4ea', '#4a90c8', '#0b3c6b']
CONST = '#f2c99b'
cmap = ListedColormap(colors)
cmap.set_bad(CONST)
norm = BoundaryNorm([0, .5, 1, 2, 1e9], cmap.N)
W_MM, H_MM = 160.5, 75
fig = plt.figure(figsize=(W_MM * style.MM, H_MM * style.MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1, .16], height_ratios=[1, .30],
                      wspace=.03, hspace=.075, left=.21, right=.99, top=.895, bottom=.30)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.ma.masked_invalid(a.to_numpy()), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
ax.set_yticks(range(len(CODES)), [NAMES[c] + (' †' if c in APPROX else '') for c in CODES], fontsize=6)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.tick_params(axis='y', length=0, pad=3)
for spine in ax.spines.values():
    spine.set_visible(False)

# A narrow group column, then the task names. Brackets make membership explicit.
trans = blended_transform_factory(fig.transFigure, ax.transData)
row = 0
for i, (name, wrapped, tasks) in enumerate(GROUPS):
    if i:
        ax.axhline(row - .5, color='white', lw=2.2)
    lo, hi = row - .29, row + len(tasks) - .71
    x0, x1 = 14.5 / W_MM, 15.4 / W_MM
    ax.plot([x1, x0, x0, x1], [lo, lo, hi, hi], transform=trans,
            color='#777777', lw=.6, clip_on=False, solid_capstyle='butt')
    ax.text(7.0 / W_MM, row + (len(tasks) - 1) / 2, wrapped,
            transform=trans, ha='center', va='center', fontsize=5.5,
            fontweight='bold', linespacing=1.0, clip_on=False)
    row += len(tasks)

axr = fig.add_subplot(gs[0, 1], sharey=ax)
axr.barh(range(len(CODES)), share.to_numpy(), color=style.MODEL, height=.68)
axr.axvline(.7, color='#888888', lw=.6, ls=(0, (2, 2)))
axr.set_xlim(0, 1)
axr.set_xticks([0, .7, 1], ['0', '.7', '1'], fontsize=6.5)
axr.set_title('Share >1 SD', fontsize=7, pad=3)
axr.tick_params(axis='y', left=False, labelleft=False)
axr.tick_params(axis='x', labelsize=6.5)
for spine in ('top', 'right', 'left'):
    axr.spines[spine].set_visible(False)

axb = fig.add_subplot(gs[1, 0], sharex=ax)
axb.plot(range(len(order)), med.to_numpy(), color=style.MODEL, lw=1.0, marker='o', ms=2.0)
axb.annotate('Median distance\n(SD)', xy=(0, .5), xycoords='axes fraction', xytext=(-17, 0),
             textcoords='offset points', ha='right', va='center', fontsize=6.5, linespacing=1.05)
axb.tick_params(axis='y', labelsize=6.5)
axb.set_ylim(0, max(3.0, float(med.max()) * 1.05))
axb.set_xticks(range(len(order)), order, rotation=90, fontsize=5.5)
axb.set_xlim(-.5, len(order) - .5)
size = d.groupby('model').params_b.first().loc[order].to_numpy()
for cut, label in ((1, '1B'), (10, '10B'), (100, '100B')):
    i = int(np.argmax(size >= cut))
    axb.axvline(i - .5, color='#888888', lw=.6, ls=(0, (2, 2)))
    right = i > len(order) - 4
    axb.text(i - .7 if right else i - .3, axb.get_ylim()[1], f'≥{label}',
             fontsize=6, va='top', ha='right' if right else 'left', color='#555555')
for spine in ('top', 'right'):
    axb.spines[spine].set_visible(False)

handles = [Patch(color=color, label=label) for color, label in zip(colors, ['< .5 SD', '.5–1 SD', '1–2 SD', '> 2 SD'])]
handles.append(Patch(color=CONST, label='constant wager'))
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.215, .995), ncol=5,
           fontsize=6, frameon=False, handlelength=1.2, columnspacing=1.05)
fig.text(.195, .968, 'Distance from\nhuman mean', fontsize=6.5, fontweight='bold',
         va='center', ha='right', linespacing=1.0)
fig.text(.995, .968, '† approximate SD', fontsize=6, va='center', ha='right')
for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'fig_h54_closeness_v2.{ext}', **kw)
