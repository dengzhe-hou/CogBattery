"""Model x task heatmap of z = (model - human mean) / human SD for the 10 key endpoints with
person-level human data. Dots mark |z| <= .5 (close). Rows sorted by served parameters."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/reanalysis/human54_layers_20261019/paper_figures'))
import h54_style as style  # noqa: E402,F401  (sets fonts and sizes)

OUT = ROOT / 'results/reanalysis/human54_model_closeness_20261029'
CODES = ['DS', 'NB', 'OS', 'ST', 'FL', 'GN', 'CVLT', 'DRM', 'FB', 'CAL']

d = pd.read_csv(OUT / 'model_task_z.csv')
order = d.groupby('model').params_b.first().sort_values().index.tolist()
z = d.pivot(index='model', columns='task', values='z').loc[order, CODES]
close = z.abs() <= .5
fig, ax = plt.subplots(figsize=(88 * style.MM, 150 * style.MM))
im = ax.imshow(z.clip(-3, 3).to_numpy(), aspect='auto', cmap='PuOr_r', norm=TwoSlopeNorm(0, -3, 3))
yy, xx = np.nonzero(close.to_numpy())
ax.scatter(xx, yy, s=4, c='black', marker='o', linewidths=0)
ax.set_xticks(range(len(CODES)), CODES, rotation=90)
counts = close.sum(axis=1)
ax.set_yticks(range(len(order)), [f'{m} ({c})' for m, c in zip(order, counts)], fontsize=5)
ax.set_xlabel('Task (key endpoint)')
cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.02, ticks=[-3, -1.5, 0, 1.5, 3])
cb.set_label('Model minus human, in human SD (clipped at 3)')
ax.set_title('Rows by size. Dot: within .5 human SD. (n) tasks close', fontsize=6)
fig.tight_layout()
for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'fig_model_task_closeness.{ext}', **kw)
