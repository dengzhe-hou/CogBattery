"""Sensitivity of human-paired intervals to human sampling error (third review, 2026-10-22). CPU only.

The primary intervals resample whole model families with the human mean fixed. Here each bootstrap draw
also draws the human mean from Normal(mean, SE) using the saved human-side SE. With SE set to zero the
script must reproduce panel_differences.csv exactly (checked before writing output).
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_layers_20261019.analysis_statistics import cluster_mean, interval

ROOT = Path(__file__).resolve().parents[3]
L = ROOT / 'results/reanalysis/human54_layers_20261019'
SEED_HUMAN = 20261022


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'results/reanalysis/human54_posthoc_20261020_humanse/human_se_sensitivity.csv')
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    pd_ = pd.read_csv(L / 'layer1/panel_differences.csv')
    pd_ = pd_[(pd_.rule == 'valid_only') & (pd_.human_cohort == 'primary') & (pd_.panel == 'all54') &
              (pd_.family_mapping == 'merged') & (pd_.statistic == 'model_minus_human')]
    m = pd.read_csv(L / 'layer1/model_human_endpoints.csv', low_memory=False)
    m = m[(m.rule == 'valid_only') & (m.human_cohort == 'primary') & (m.included == True) & m.panel54]
    refs = json.loads((L / 'human_side/references.json').read_text())['endpoints']

    def human_se(task, endpoint, cohort):
        cands = [x for x in refs if x['task'] == task and x['endpoint'] == endpoint and x.get('cohort', 'primary') == cohort]
        cands.sort(key=lambda x: {'valid_only': 0, 'both': 1}.get(x['rule'], 2))
        for x in cands:
            if x.get('se') is not None:
                return float(x['se']), 'published'
            v = np.array([y for y in x.get('values') or [] if y is not None], float)
            if len(v) > 1:
                return float(v.std(ddof=1) / np.sqrt(len(v))), f'sd/sqrt(n), n={len(v)}'
        return np.nan, ''
    rows = []
    for _, r in pd_.iterrows():
        g = m[(m.task == r.task) & (m.endpoint == r.endpoint)]
        stats, boot = cluster_mean(g.value - r.human_mean, g.family_merged)
        assert abs(stats['lower'] - r.lower) < 1e-9 and abs(stats['upper'] - r.upper) < 1e-9, (r.task, r.endpoint)
        se, source = human_se(r.task, r.endpoint, r.human_cohort)
        row = dict(task=r.task, endpoint=r.endpoint, difference=stats['mean'], lower=r.lower, upper=r.upper,
                   human_se=se, se_source=source)
        if np.isfinite(se):
            z = np.random.default_rng(SEED_HUMAN).standard_normal(len(boot))
            iv = interval(boot - se * z)
            row.update(lower_with_human_se=iv['lower'], upper_with_human_se=iv['upper'])
        rows.append(row)
    out = pd.DataFrame(rows)
    out['excludes_zero'] = (out.lower > 0) | (out.upper < 0)
    out['excludes_zero_with_human_se'] = (out.lower_with_human_se > 0) | (out.upper_with_human_se < 0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    s = out.dropna(subset=['human_se'])
    print(len(out), 'endpoints;', len(s), 'with human SE;', int(s.excludes_zero.sum()), 'exclude zero ->', int(s.excludes_zero_with_human_se.sum()))


if __name__ == '__main__':
    main()
