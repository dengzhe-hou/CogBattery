"""Per-model closeness to humans on the key endpoints (2026-10-29). CPU only, reads saved results.

Exploratory analysis added after the formal results. For each model and each of the 10 key
endpoints with a person-level human distribution, z = (model - human mean) / human SD, and a
model counts as close on that task when |z| <= 0.5 (primary, set before computing). Thresholds
.25 and 1 are sensitivities. SM, EP, and WAG have no person-level human values and are left out.
Family intervals use 5,000 whole-family resamples (seed 20261029).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.reanalysis.human54_posthoc_20261020.gen_core_table import ROWS
from scripts.reanalysis.human54_layers_20261019.analyze import refs_for_rule

OUT = ROOT / 'results/reanalysis/human54_model_closeness_20261029'
THRESH, SENS, DRAWS, SEED = .5, (.25, 1.0), 5000, 20261029
CODES = ['DS', 'NB', 'OS', 'ST', 'FL', 'GN', 'CVLT', 'DRM', 'FB', 'CAL']


def family_boot(df, stat, rng):
    fams = df.family.unique()
    groups = {f: df[df.family == f] for f in fams}
    out = []
    for _ in range(DRAWS):
        pick = rng.choice(fams, len(fams), replace=True)
        out.append(stat(pd.concat([groups[f] for f in pick])))
    return np.nanpercentile(out, [2.5, 97.5]).tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = json.loads((ROOT / 'results/reanalysis/human54_layers_20261019/human_side/references.json').read_text())['endpoints']
    refs = refs_for_rule(refs, 'invalid_wrong')
    p = pd.read_csv(ROOT / 'results/reanalysis/human54_layers_20261019/layer1/model_human_endpoints.csv', low_memory=False)
    p = p[(p.rule == 'invalid_wrong') & (p.human_cohort == 'primary') & (p.included == True)]
    params = pd.read_csv(ROOT / 'results/reanalysis/human54_posthoc_20261020_gates/model_parameters.csv', index_col='model').params_b
    rows, human = [], {}
    for code, task, endpoint, _ in ROWS:
        if code not in CODES:
            continue
        ref = [r for r in refs if r['task'] == task and r['endpoint'] == endpoint and r.get('cohort', 'primary') == 'primary']
        assert len(ref) == 1, (code, len(ref))
        v = np.array([x for x in ref[0]['values'] if x is not None], float)
        mu, sd = float(ref[0]['value']), float(v.std(ddof=1))
        human[code] = dict(mean=mu, sd=sd, n=len(v))
        g = p[(p.task == task) & (p.endpoint == endpoint)]
        assert np.allclose(g.human_mean, mu)
        for r in g.itertuples():
            rows.append(dict(model=r.model, family=r.family_merged, native=bool(r.native_thinking), task=code,
                             value=r.value, z=(r.value - mu) / sd, pct=r.human_percentile_mid_distribution))
    d = pd.DataFrame(rows).dropna(subset=['z'])
    d['params_b'] = d.model.map(params)
    assert d.params_b.notna().all()
    d.to_csv(OUT / 'model_task_z.csv', index=False)
    rng = np.random.default_rng(SEED)
    res = {'threshold_sd': THRESH, 'human': human, 'note': __doc__}
    for t in (THRESH,) + SENS:
        d['close'] = d.z.abs() <= t
        pm = d.groupby('model').agg(close=('close', 'sum'), tasks=('close', 'size'), family=('family', 'first'),
                                    native=('native', 'first'), params_b=('params_b', 'first'))
        pt = d.groupby('task').agg(close=('close', 'sum'), models=('close', 'size'), median_z=('z', 'median'))
        rho = lambda x: x[['close', 'params_b']].assign(params_b=np.log(x.params_b)).corr('spearman').iloc[0, 1]
        big = lambda x: x[x.params_b >= 27].close.mean() - x[x.params_b < 27].close.mean()
        res[f'thr_{t}'] = dict(
            per_model_close=pm.close.value_counts().sort_index().to_dict(),
            max_close=int(pm.close.max()), best_models=pm[pm.close == pm.close.max()].index.tolist(),
            mean_close=float(pm.close.mean()), tasks_per_model=pm.tasks.value_counts().to_dict(),
            per_task=pt.to_dict('index'),
            spearman_logsize=[float(rho(pm))] + family_boot(pm, rho, rng),
            ge27b_minus_lt27b=[float(big(pm))] + family_boot(pm, big, rng),
            native_mean=float(pm[pm.native].close.mean()), direct_mean=float(pm[~pm.native].close.mean()))
        if t == THRESH:
            pm.sort_values('params_b').to_csv(OUT / 'per_model_close.csv')
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k.startswith('thr')}, indent=1, default=float))


if __name__ == '__main__':
    main()
