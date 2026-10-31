"""Relation between model size and distance from humans behind Figure 3 (2026-10-31).

Post hoc. Distance is |z| from thirteen/model_task_z.csv. Each model's distance is its median across
tasks, as in the figure's bottom panel. Reports the Spearman correlation with log size, a 95% interval
from 5,000 whole-family resamples, and per-task correlations and medians below and above 27B.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / 'results/reanalysis/human54_model_closeness_20261029/thirteen'
LARGE_B = 27
DRAWS = 5000
SEED = 20261031


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main():
    z = pd.read_csv(DIR / 'model_task_z.csv').dropna(subset=['z']).assign(dist=lambda d: d.z.abs())
    z['log_size'] = np.log10(z.params_b)
    per_model = z.groupby(['model', 'family', 'params_b', 'log_size'], as_index=False).dist.median()
    rho = spearman(per_model.log_size.to_numpy(), per_model.dist.to_numpy())
    rng = np.random.default_rng(SEED)
    families = per_model.family.unique()
    groups = {f: per_model[per_model.family == f] for f in families}
    draws = []
    for _ in range(DRAWS):
        sample = pd.concat([groups[f] for f in rng.choice(families, len(families))])
        draws.append(spearman(sample.log_size.to_numpy(), sample.dist.to_numpy()))
    large = per_model.params_b >= LARGE_B
    rows = []
    for task, t in z.groupby('task'):
        big = t.params_b >= LARGE_B
        rows.append(dict(task=task, n=len(t), spearman=spearman(t.log_size.to_numpy(), t.dist.to_numpy()),
                         median_below=t.dist[~big].median(), median_large=t.dist[big].median(),
                         share_over_1sd_below=(t.dist[~big] > 1).mean(), share_over_1sd_large=(t.dist[big] > 1).mean()))
    pd.DataFrame(rows).to_csv(DIR / 'size_relation_by_task.csv', index=False)
    out = dict(models=len(per_model), spearman=rho, interval=[float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))],
               median_below=per_model.dist[~large].median(), median_large=per_model.dist[large].median(),
               models_large=int(large.sum()), draws=DRAWS, seed=SEED)
    (DIR / 'size_relation.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2)); print(pd.DataFrame(rows).round(3).to_string())


if __name__ == '__main__':
    main()
