"""Per-model closeness to humans on the key endpoints (2026-10-29). CPU only, reads saved results.

Exploratory analysis added after the formal results. For each model and each of the 13 key
endpoints, z = (model - human mean) / human SD, and a model counts as close on that task when
|z| <= 0.5 (primary, set before computing). Thresholds .25 and 1 are sensitivities.
Human SD is the SD of person-level endpoint values, except
  SM  between-person contrast: pooled within-condition SD of person source accuracy (3 conditions);
  EP  SD of per-person OLS slopes over mental-state levels 1-7 (persons aligned, see human_se_complete);
  WAG high-minus-low accuracy, the only human wager endpoint (key endpoint has no human value):
      SD = published SE x sqrt(40); models with a constant wager have no value and count as not close.
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

OUT = ROOT / 'results/reanalysis/human54_model_closeness_20261029/thirteen'
THRESH, SENS, DRAWS, SEED = .5, (.25, 1.0), 5000, 20261029
CODES = ['DS', 'NB', 'OS', 'ST', 'FL', 'GN', 'CVLT', 'DRM', 'SM', 'FB', 'EP', 'CAL', 'WAG']


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
    p = p[(p.rule == 'invalid_wrong') & (p.human_cohort == 'primary')]
    params = pd.read_csv(ROOT / 'results/reanalysis/human54_posthoc_20261020_gates/model_parameters.csv', index_col='model').params_b
    rows, human = [], {}
    def ref_of(task, endpoint):
        m = [r for r in refs if r['task'] == task and r['endpoint'] == endpoint and r.get('cohort', 'primary') == 'primary']
        assert len(m) == 1, (task, endpoint, len(m))
        return m[0]

    def person(task, endpoint):
        return np.array([x for x in ref_of(task, endpoint)['values'] if x is not None], float)

    for code, task, endpoint, _ in ROWS:
        if code not in CODES:
            continue
        if code == 'WAG':
            endpoint = 'high_minus_low_accuracy'
        ref = ref_of(task, endpoint)
        mu = float(ref['value'])
        if code == 'SM':
            groups = [person(task, e) for e in ('indirect_accuracy', 'direct_agent_last_accuracy', 'direct_agent_first_accuracy')]
            sd = float(np.sqrt(sum((len(x) - 1) * x.var(ddof=1) for x in groups) / sum(len(x) - 1 for x in groups)))
            n = sum(len(x) for x in groups)
        elif code == 'EP':
            levels = np.array([person(task, f'mental|{k}_accuracy') for k in range(1, 8)]).T
            slopes = np.polyfit(np.arange(1, 8), levels.T, 1)[0]
            sd, n = float(slopes.std(ddof=1)), len(slopes)
        elif code == 'WAG':
            sd, n = float(ref['se'] * np.sqrt(ref['n'])), ref['n']
        else:
            v = person(task, endpoint)
            sd, n = float(v.std(ddof=1)), len(v)
        human[code] = dict(mean=mu, sd=sd, n=n, endpoint=endpoint)
        g = p[(p.task == task) & (p.endpoint == endpoint)]
        assert len(g) == 54 and np.allclose(g.human_mean, mu)
        if code != 'WAG':
            g = g[g.included == True]
        for r in g.itertuples():
            rows.append(dict(model=r.model, family=r.family_merged, native=bool(r.native_thinking), task=code,
                             value=r.value, z=(r.value - mu) / sd, pct=r.human_percentile_mid_distribution))
    d = pd.DataFrame(rows)
    d = d[d.z.notna() | (d.task == 'WAG')]   # constant wagerers stay, as not close
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
