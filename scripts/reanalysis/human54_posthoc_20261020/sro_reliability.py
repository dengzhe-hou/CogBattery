"""SRO five-task reliability on both sides and the disattenuated grouping contrast (spec C4).

Human side: SRO raw trials with the same task QC and score definitions as the frozen
human reference (human54_layers_20261019/human.py build_sro, valid_only). Halves
alternate trials within person x cell (digit span), person x condition (Stroop,
flanker, go/no-go) and blocks within person x load (n-back). Digit-span halves use
the available cells with the fixed weights, because many cells hold one to three
trials; the full score keeps the frozen all-cell rule and must reproduce the archived
345 x 5 matrix. Model side: structure_sro per SRO trajectory (n-back: alternate
blocks) from halves.py --kind structure_sro; the full score must reproduce the
layer-two SRO scores for all 54 models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reanalysis.sro_structure_20261015 import build_human as SRO
from scripts.reanalysis.human54_layers_20261019.human import DS_WEIGHTS
from scripts.reanalysis.human54_fullbattery_20261019.pca import family_bootstrap_counts
from .reliability import reliability

ROOT = Path(__file__).resolve().parents[3]
TASKS = ['digit_span', 'n_back', 'stroop', 'flanker', 'go_nogo']
GROUPS = np.array(['WM', 'WM', 'CC', 'CC', 'CC'])
LAYER2 = ROOT / 'results/reanalysis/human54_layers_20261019/layer2/sro_valid_only_all54_merged'
REFERENCES = ROOT / 'results/reanalysis/human54_layers_20261019/human_side/references.json'
EXPECTED_DELTA = dict(human=0.17103566711896587, model=-0.021960203165081904)


def screened():
    out = {}
    for task in SRO.TASKS:
        d = SRO.load(task)
        flags = SRO.qc(task, d)
        out[task] = d[d.worker_id.isin(flags.index[flags.passed])].copy()
    return out


def human_scores(scr, half=None):
    index = pd.Index(sorted(set().union(*(set(d.worker_id) for d in scr.values()))))
    out = pd.DataFrame(index=index)
    keep = (lambda d, keys: d) if half is None else (lambda d, keys: d[d.groupby(keys).cumcount() % 2 == half])
    ds = scr['digit_span'].copy()
    ds['direction'] = ds.condition.replace({'reverse': 'backward'})
    ds['cell'] = ds.direction + '|' + ds.num_digits.astype(int).astype(str)
    ds = keep(ds[ds.rt != -1], ['worker_id', 'cell'])
    cells = [f'{d}|{n}' for d in ('forward', 'backward') for n in range(3, 10)]
    out['digit_span'] = SRO.weighted(ds.groupby(['worker_id', 'cell']).correct.mean().unstack('cell').reindex(columns=cells),
                                     DS_WEIGHTS, half is None)
    nb = scr['adaptive_n_back']
    nb = nb[(nb.exp_stage == 'adaptive') & nb.load.isin([2, 3]) & (nb.rt != -1)]
    blocks = nb.groupby(['worker_id', 'load', 'block_num']).correct.mean().reset_index()
    blocks = keep(blocks, ['worker_id', 'load'])
    out['n_back'] = SRO.weighted(blocks.groupby(['worker_id', 'load']).correct.mean().unstack('load'), {2: 1, 3: 1}, True)
    for source, task, condition, weights in [
            ('stroop', 'stroop', 'condition', {'congruent': 1, 'incongruent': 1}),
            ('attention_network_task', 'flanker', 'flanker_type', {'congruent': 1, 'incongruent': 1, 'neutral': 1}),
            ('go_nogo', 'go_nogo', 'condition', {'go': 945, 'nogo': 105})]:
        d = scr[source]
        d = d[d.exp_stage != 'practice']
        if task != 'go_nogo':
            d = d[d.rt != -1]
        d = keep(d, ['worker_id', condition])
        out[task] = SRO.weighted(d.groupby(['worker_id', condition]).correct.mean().unstack(condition), weights, True)
    return out[TASKS]


def _contrast(R):
    row, col = np.triu_indices(len(TASKS), 1)
    within = GROUPS[row] == GROUPS[col]
    return R[row, col][within].mean() - R[row, col][~within].mean()


def side_statistics(X, forms):
    """forms: task -> (n, k) array aligned with X rows."""
    R = np.corrcoef(X, rowvar=False)
    rel = np.array([reliability(forms[t])['rel'] for t in TASKS])
    out = dict(delta=_contrast(R), rel=rel, delta_star=np.nan)
    if np.all(rel > 0):
        out['delta_star'] = _contrast(R / np.sqrt(np.outer(rel, rel)))
    return out


def bootstrap(X, forms, weights):
    stats = np.full((len(weights), 2 + len(TASKS)), np.nan)
    for i, w in enumerate(weights):
        idx = np.repeat(np.arange(len(X)), w)
        if np.any(X[idx].std(axis=0) == 0):
            continue
        s = side_statistics(X[idx], {t: f[idx] for t, f in forms.items()})
        stats[i] = [s['delta'], s['delta_star'], *s['rel']]
    return stats


def run(model_halves, output, draws=5000, seed=20261018):
    streams = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(14)]
    scr = screened()
    full = human_scores(scr).dropna()
    archived = next(m for m in json.loads(REFERENCES.read_text())['matrices'] if m['name'] == 'sro' and m['rule'] == 'valid_only')
    if not np.allclose(full.to_numpy(), np.asarray(archived['values'], float), rtol=1e-12, atol=1e-12):
        raise ValueError('Human full scores differ from the archived SRO matrix')
    halves = [human_scores(scr, h).reindex(full.index) for h in (0, 1)]
    human_forms = {t: np.column_stack([halves[0][t], halves[1][t]]) for t in TASKS}
    Xh = full.to_numpy()

    mh = pd.read_csv(model_halves)
    llm = pd.read_csv(LAYER2 / 'llm_scores.csv', index_col='model')
    models = list(llm.index)
    got = mh[mh.form == 'full'].pivot(index='model', columns='task', values='value').reindex(index=models, columns=TASKS)
    if not np.allclose(got.to_numpy(), llm[TASKS].to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError('Model full structure_sro scores differ from layer-two SRO scores')
    model_forms = {t: mh[(mh.task == t) & (mh.form != 'full')].pivot(index='model', columns='form', values='value')
                   .reindex(models).to_numpy() for t in TASKS}
    Xm = llm[TASKS].to_numpy()

    point = {'human': side_statistics(Xh, human_forms), 'model': side_statistics(Xm, model_forms)}
    for side in ('human', 'model'):
        if abs(point[side]['delta'] - EXPECTED_DELTA[side]) > 1e-10:
            raise ValueError(f'{side} delta does not reproduce the layer-two value')
    rng = np.random.default_rng(streams[12])
    hw = rng.multinomial(len(Xh), np.full(len(Xh), 1 / len(Xh)), size=draws)
    _, codes, counts = family_bootstrap_counts(llm.family.tolist(), draws, streams[13])
    hb, mb = bootstrap(Xh, human_forms, hw), bootstrap(Xm, model_forms, counts[:, codes])
    q = lambda v: dict(lower=float(np.nanquantile(v, .025)), upper=float(np.nanquantile(v, .975)), n_valid=int(np.isfinite(v).sum()))
    result = dict(n_humans=len(Xh), n_models=len(Xm), n_families=int(llm.family.nunique()), draws=draws,
                  seeds=dict(human=streams[12], model=streams[13]),
                  human_half_rule='alternate trials within person x cell/condition; n-back alternate blocks within load; digit-span halves use available cells',
                  model_forms={t: sorted(mh[(mh.task == t) & (mh.form != 'full')].form.unique()) for t in TASKS})
    for side, b in (('human', hb), ('model', mb)):
        p = point[side]
        result[side] = dict(delta=float(p['delta']), delta_ci=q(b[:, 0]), delta_star=float(p['delta_star']), delta_star_ci=q(b[:, 1]),
                            reliability={t: dict(rel=float(r), ci=q(b[:, 2 + j]),
                                                 n=reliability((human_forms if side == 'human' else model_forms)[t])['n'])
                                         for j, (t, r) in enumerate(zip(TASKS, p['rel']))})
    result['difference'] = dict(delta=float(point['human']['delta'] - point['model']['delta']), delta_ci=q(hb[:, 0] - mb[:, 0]),
                                delta_star=float(point['human']['delta_star'] - point['model']['delta_star']),
                                delta_star_ci=q(hb[:, 1] - mb[:, 1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-halves', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.model_halves, a.output), indent=2))


if __name__ == '__main__':
    main()
