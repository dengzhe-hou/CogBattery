"""Pairwise human-model task correlations in two extra joint human cohorts (2026-10-29).

CNP (130 healthy controls): CVLT-II trials 1-5 total, WMS digit span total, SCWT Stroop.
Bradford et al. 2020 (adults >= 18): false-belief appropriateness, OSpan partial, Stroop RT effect.
Each cohort has one task per paradigm group, so delta is undefined; we compare each pair's
Pearson r between humans (bootstrap over people) and 54 models (bootstrap over 23 merged
families). Human and model scores are oriented so that higher = better / less interference.
Matches are task-family level only.
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
# Raw human sources: OpenNeuro ds000030 (CNP phenotype) and OSF pw7h6 (Bradford et al. 2020).
RAW = Path(os.environ.get('COGBATTERY_RAW', ROOT / 'data/raw'))
FB = ROOT / 'results/reanalysis/human54_fullbattery_20261019/invalid_wrong'
ENDPOINTS = ROOT / 'results/reanalysis/human54_layers_20261019/layer1/model_human_endpoints.csv'
OUT = ROOT / 'results/reanalysis/human54_extra_cohorts_20261029'
B = 5000


def model_scores():
    X = pd.read_csv(FB / 'matrix.csv', index_col='model')
    meta = pd.read_csv(FB / 'metadata.csv', index_col='model')
    e = pd.read_csv(ENDPOINTS, low_memory=False)
    e = e[e.rule == 'invalid_wrong'].pivot_table(index='model', columns=['task', 'endpoint'], values='value')
    m = pd.DataFrame(index=X.index)
    m['CVLT'] = X.cvlt_word_list
    m['DS'] = X.digit_span
    m['OS'] = X.operation_span
    m['ST_acc'] = X.stroop
    m['ST_int'] = e[('stroop', 'incongruent_accuracy')] - e[('stroop', 'congruent_accuracy')]
    m['FB'] = e[('false_belief', 'session|FB_accuracy')]
    m['FB_main'] = X.false_belief
    assert m.notna().all().all() and len(m) == 54
    return m, meta.family_merged.loc[m.index]


def cnp():
    P = RAW / 'ucla_cnp_ds000030'
    rd = lambda f: pd.read_csv(P / 'phenotype' / f, sep='\t', na_values='n/a').set_index('participant_id')
    p = pd.read_csv(P / 'participants.tsv', sep='\t').set_index('participant_id')
    c, w, s = rd('cvlt.tsv'), rd('wms.tsv'), rd('stroop.tsv')
    h = pd.DataFrame({'CVLT': c.cvlt_totcor, 'DS': w.ds_totalraw,
                      'ST_acc': (s.scwt_acccon + s.scwt_accinc) / 2,
                      'ST_int': s.scwt_accinc - s.scwt_acccon})
    return h.loc[p.index[p.diagnosis == 'CONTROL']]


def bradford(age_partial=False):
    P = RAW / 'bradford2020_fb_n400'
    b = pd.read_csv(P / 'BehaviouralData.csv', encoding='utf-8-sig').set_index('File')
    c = pd.read_csv(P / 'Correlations.csv', encoding='utf-8-sig').set_index('PIN')
    h = pd.DataFrame({'FB': b.FBIC_Appropriateness - b.FBC_Appropriateness,  # ratings: 5 = very inappropriate
                      'OS': c.OSpan_Partial, 'ST_int': -c.StroopLog_Scale, 'age': c.Age})
    h = h[h.age >= 18]
    if age_partial:  # residualize each measure on age and age^2 (complete cases per measure)
        for col in ['FB', 'OS', 'ST_int']:
            ok = h[col].notna()
            A = np.column_stack([np.ones(ok.sum()), h.age[ok], h.age[ok] ** 2])
            beta = np.linalg.lstsq(A, h.loc[ok, col].to_numpy(float), rcond=None)[0]
            h.loc[ok, col] = h.loc[ok, col] - A @ beta
    return h.drop(columns='age')


def r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan


def pair_stats(h, m, fam, x, y, rng):
    hx, hy = h[x].to_numpy(float), h[y].to_numpy(float)
    ok = np.isfinite(hx) & np.isfinite(hy)
    hx, hy = hx[ok], hy[ok]
    mx, my = m[x].to_numpy(float), m[y].to_numpy(float)
    groups = [np.flatnonzero(fam.to_numpy() == f) for f in fam.unique()]
    hb = np.array([r(hx[i], hy[i]) for i in (rng.integers(0, len(hx), len(hx)) for _ in range(B))])
    mb = []
    for _ in range(B):
        i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        mb.append(r(mx[i], my[i]))
    mb = np.array(mb)
    d = hb - mb
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    rh, rm = r(hx, hy), r(mx, my)
    return dict(pair=f'{x}-{y}', n_human=int(ok.sum()), r_human=rh, r_human_ci=ci(hb),
                r_model=rm, r_model_ci=ci(mb), diff=rh - rm, diff_ci=ci(d))


def main():
    rng = np.random.default_rng(20261029)
    m, fam = model_scores()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'cnp_primary': (cnp(), ['CVLT', 'DS', 'ST_acc']),
            'cnp_stroop_interference': (cnp(), ['CVLT', 'DS', 'ST_int']),
            'bradford_primary': (bradford(), ['FB', 'OS', 'ST_int']),
            'bradford_age_partial': (bradford(age_partial=True), ['FB', 'OS', 'ST_int'])}
    res, rows = {}, []
    for name, (h, tasks) in plan.items():
        res[name] = dict(n_people=int(len(h)), pairs=[pair_stats(h, m, fam, x, y, rng)
                                                        for x, y in itertools.combinations(tasks, 2)])
        rows += [{'analysis': name, **p} for p in res[name]['pairs']]
    # Model FB sensitivity: battery main accuracy instead of false-belief-trial accuracy.
    m2 = m.assign(FB=m.FB_main)
    res['bradford_model_fb_main'] = dict(pairs=[pair_stats(bradford(), m2, fam, x, y, rng)
                                                for x, y in itertools.combinations(['FB', 'OS', 'ST_int'], 2)])
    rows += [{'analysis': 'bradford_model_fb_main', **p} for p in res['bradford_model_fb_main']['pairs']]
    (OUT / 'summary.json').write_text(json.dumps(res, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(OUT / 'pairs.csv', index=False)
    print(pd.DataFrame(rows)[['analysis', 'pair', 'n_human', 'r_human', 'r_model', 'diff', 'diff_ci']].to_string())


if __name__ == '__main__':
    main()
