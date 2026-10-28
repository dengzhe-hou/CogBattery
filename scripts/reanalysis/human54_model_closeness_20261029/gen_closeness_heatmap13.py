"""Paper Figure 3 (with --paper: no dots or counts, written to the Overleaf figures folder).
Task x model heatmap (13 key endpoints x 54 models, double-column width) of
z = (model - human mean) / human SD. Dots mark |z| <= .5. Grey WAG cells are constant wagerers
(no high-minus-low value, counted as not close). Columns sorted by served parameters."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/reanalysis/human54_layers_20261019/paper_figures'))
import h54_style as style  # noqa: E402  (sets fonts and sizes)

OUT = ROOT / 'results/reanalysis/human54_model_closeness_20261029/thirteen'
CODES = ['DS', 'NB', 'OS', 'ST', 'FL', 'GN', 'CVLT', 'DRM', 'SM', 'FB', 'EP', 'CAL', 'WAG']

PAPER = '--paper' in sys.argv
d = pd.read_csv(OUT / 'model_task_z.csv')
order = d.groupby('model').params_b.first().sort_values().index.tolist()
z = d.pivot(index='task', columns='model', values='z').loc[CODES, order]
close = z.abs() <= .5
cmap = plt.get_cmap('PuOr_r').copy()
cmap.set_bad('#c8c8c8')
fig, ax = plt.subplots(figsize=(180 * style.MM, (56 if PAPER else 72) * style.MM))
im = ax.imshow(np.ma.masked_invalid(z.clip(-3, 3).to_numpy()), aspect='auto', cmap=cmap, norm=TwoSlopeNorm(0, -3, 3))
yy, xx = np.nonzero(close.to_numpy())
if not PAPER:
    ax.scatter(xx, yy, s=3, c='black', marker='o', linewidths=0)
counts = close.sum(axis=0)
labels = order if PAPER else [f'{m} ({c})' for m, c in zip(order, counts)]
ax.set_xticks(range(len(order)), labels, rotation=90, fontsize=5)
ax.set_yticks(range(len(CODES)), CODES, fontsize=6)
if not PAPER:
    ax.set_xlabel('Model, by served parameters (tasks within .5 human SD)')
cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01, ticks=[-3, -1.5, 0, 1.5, 3])
cb.set_label('Model minus human (SD)')
fig.tight_layout()
dest, stem = (OUT, 'fig_h54_closeness') if PAPER else (OUT, 'fig_task_model_closeness')
for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600})):
    fig.savefig(dest / f'{stem}.{ext}', bbox_inches='tight', pad_inches=.02, **kw)
