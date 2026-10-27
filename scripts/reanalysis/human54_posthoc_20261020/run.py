"""Post hoc analyses (spec review-stage/human54_posthoc_20261020/SPEC.md).

Reads saved derived tables and the half-score table written by halves.py.
CPU only, single-threaded maths, no model calls.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from datetime import datetime
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_fullbattery_20261019.pca import analyze_pca
from . import data as D
from . import lofo as L
from . import reliability as REL
from . import scale as S

SEED = 20261018
# Children 0-5 are the full-battery seeds (0 = primary PCA, 1 = primary prediction);
# post hoc analyses take later children so no stream is reused for a new purpose.
_STREAMS = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(SEED).spawn(12)]
SEEDS = dict(pca=_STREAMS[0], prediction=_STREAMS[1], scale_corr=_STREAMS[6], scale_structure=_STREAMS[7],
             scale_pa=_STREAMS[8], bias=_STREAMS[9], reliability=_STREAMS[10])
EXPECTED_MAIN = dict(rmse_g=1.746102915718687, rmse_group=1.652661184884706, gain=0.053514446366708546)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float) + '\n')


def prediction_summary(cells, *, draws, extra=()):
    pairs = {'five_group_vs_pc1': ('se_group', 'se_g')}
    for name in extra:
        pairs[f'{name}_vs_pc1'] = ('se_' + name, 'se_g')
        pairs[f'five_group_vs_{name}'] = ('se_group', 'se_' + name)
    out = dict(rmse_pc1=L.rmse(cells, 'se_g'), rmse_five_group=L.rmse(cells, 'se_group'),
               rmse_intercept=L.rmse(cells, 'se_intercept'), gain_five_group=L.gain(cells),
               n_cells=int(len(cells)), n_families=int(cells.family.nunique()))
    for name in extra:
        out['rmse_' + name] = L.rmse(cells, 'se_' + name)
        out[f'gain_{name}_vs_pc1'] = L.gain(cells, 'se_' + name, 'se_g')
        out[f'gain_five_group_vs_{name}'] = L.gain(cells, 'se_group', 'se_' + name)
    out['bootstrap'] = L.paired_bootstrap(cells, pairs, draws=draws, seed=SEEDS['prediction'])
    return out


def per_target(cells):
    rows = []
    net = (cells.se_g - cells.se_group).sum()
    for t, sub in cells.groupby('target', sort=False):
        rows.append(dict(target=D.ABBR[t], rmse_pc1=L.rmse(sub, 'se_g'), rmse_five_group=L.rmse(sub, 'se_group'),
                         gain=L.gain(sub), share_of_net_improvement=float((sub.se_g - sub.se_group).sum() / net)))
    return pd.DataFrame(rows)


def run_prediction(X, families, partitions, *, draws, full_catalog=True, two_dim=False):
    tasks = list(X.columns)
    theory = L.theory_peers(tasks)
    peer_sets = {t: (L.peer_catalog(tasks, t) if full_catalog else theory[t]) for t in tasks}
    rows, diagnostics = L.fold_predictions(X, families, peer_sets, two_dim=two_dim)
    cells = L.cell_table(rows, {t: theory[t][0] for t in tasks})
    summary = prediction_summary(cells, draws=draws, extra=('pc12', 'fa2') if two_dim else ())
    if full_catalog:
        gains, rmse_g, n_cells = L.partition_reference(rows, tasks, partitions)
        k = L.theory_partition_index(partitions, tasks)
        if not np.isclose(gains[k], summary['gain_five_group'], atol=1e-12):
            raise ValueError('Partition engine disagrees with the cell table for the five groups')
        summary['partition_reference'] = dict(
            n_partitions=int(len(gains)), five_group_gain=float(gains[k]),
            rank_from_top=int(np.sum(gains > gains[k]) + 1),
            share_at_least_five_group=float(np.mean(gains >= gains[k])),
            quantiles={str(q): float(np.quantile(gains, q)) for q in (.05, .25, .5, .75, .95)},
            max=float(gains.max()), min=float(gains.min()), share_positive=float(np.mean(gains > 0)))
        return summary, cells, diagnostics, gains
    return summary, cells, diagnostics, None


def run(output, halves_path, *, pca_draws=5000, prediction_draws=20000, pa_draws=5000):
    if output.exists():
        raise FileExistsError(f'{output} exists; use a new output directory')
    start = time.monotonic()
    X, meta = D.load_matrix()
    families = D.family_maps(meta)
    merged = families['merged23']
    partitions = L.enumerate_partitions(13, (3, 3, 3, 2, 2))
    manifest = dict(started_at=datetime.now().astimezone().isoformat(), seed=SEED, seeds=SEEDS,
                    spec='review-stage/human54_posthoc_20261020/SPEC.md', post_hoc=True,
                    halves=str(halves_path), hostname=os.uname().nodename, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    draws=dict(pca=pca_draws, prediction=prediction_draws, parallel_analysis=pa_draws))

    # B0-B4 main.
    main, cells, diagnostics, gains = run_prediction(X, merged, partitions, draws=prediction_draws, two_dim=True)
    for key, value in EXPECTED_MAIN.items():
        got = dict(rmse_g=main['rmse_pc1'], rmse_group=main['rmse_five_group'], gain=main['gain_five_group'])[key]
        if abs(got - value) > 1e-4:
            raise ValueError(f'B0 reproduction failed: {key} {got} != {value}')
    pred = output / 'prediction'
    write_json(pred / 'main.json', main)
    cells.to_csv(pred / 'main_cells.csv', index=False)
    per_target(cells).to_csv(pred / 'main_per_target.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(pred / 'main_fa2_fold_diagnostics.csv', index=False)
    np.save(pred / 'main_partition_gains.npy', gains)

    sensitivities = {}
    for label, cols in (('drop_WAG', ['post_decision_wagering']),
                        ('drop_CAL_WAG', ['confidence_calibration', 'post_decision_wagering'])):
        s, c, _, _ = run_prediction(X.drop(columns=cols), merged, None, draws=prediction_draws, full_catalog=False)
        s['peerless_targets'] = [D.ABBR[t] for t, p in L.theory_peers(list(X.drop(columns=cols).columns)).items() if not p[0]]
        sensitivities[label] = s
        c.to_csv(pred / f'{label}_cells.csv', index=False)
    for label, cols in (('aggregate_without_WAG', ['post_decision_wagering']),
                        ('aggregate_without_CAL_WAG', ['confidence_calibration', 'post_decision_wagering'])):
        kept = cells[~cells.target.isin(cols)]
        sensitivities[label] = prediction_summary(kept, draws=prediction_draws)
    panels = {'no_deepseek_r': [m for m in X.index if merged[m] != 'deepseek-r'],
              'direct51': [m for m in X.index if not meta.native_thinking[m]]}
    family_settings = {'raw28': families['raw28'], 'lineage18': families['lineage18'],
                       'lineage17_aya': families['lineage17_aya']}
    partition_settings = {'main': main['partition_reference']}
    for label, models in panels.items():
        s, c, _, g = run_prediction(X.loc[models], merged, partitions, draws=prediction_draws)
        s['n_models'] = len(models)
        sensitivities[label] = s
        partition_settings[label] = s['partition_reference']
        c.to_csv(pred / f'{label}_cells.csv', index=False)
        np.save(pred / f'{label}_partition_gains.npy', g)
    for label, fam in family_settings.items():
        s, c, _, g = run_prediction(X, fam, partitions, draws=prediction_draws)
        sensitivities['families_' + label] = s
        partition_settings['families_' + label] = s['partition_reference']
        c.to_csv(pred / f'families_{label}_cells.csv', index=False)
        np.save(pred / f'families_{label}_partition_gains.npy', g)
    write_json(pred / 'sensitivities.json', sensitivities)
    write_json(pred / 'partition_reference.json', partition_settings)

    # D. PCA and grouping contrast under other family definitions and the 51-model panel.
    structure = {}
    codes = [D.ABBR[t] for t in X.columns]
    for label, fam in families.items():
        result, _ = analyze_pca(X.to_numpy(), [fam[m] for m in X.index], codes, seed=SEEDS['pca'], draws=pca_draws)
        structure[label] = dict(n_families=result['n_families'], pc1_share=result['variance_shares'][0],
                                structure=result['structure'], intervals=result['bootstrap']['intervals'],
                                n_valid=result['bootstrap']['n_valid'])
    direct = panels['direct51']
    result, _ = analyze_pca(X.loc[direct].to_numpy(), [merged[m] for m in direct], codes, seed=SEEDS['pca'], draws=pca_draws)
    structure['direct51_merged'] = dict(n_families=result['n_families'], pc1_share=result['variance_shares'][0],
                                        structure=result['structure'], intervals=result['bootstrap']['intervals'],
                                        n_valid=result['bootstrap']['n_valid'])
    if abs(structure['merged23']['pc1_share'] - 0.4055) > 5e-4:
        raise ValueError('Primary PC1 share does not reproduce the full-battery result')
    write_json(output / 'families' / 'structure.json', structure)

    # A. Scale.
    params = D.parameter_table(list(X.index))
    params.to_csv(output / 'inputs_model_parameters.csv')
    fam_list = [merged[m] for m in X.index]
    pc1 = pd.read_csv(D.FULLBATTERY / 'primary_valid_only/pca_scores.csv', index_col='model').PC1.reindex(X.index)
    factors = D.factor_scores(X)
    scale = {}
    for basis in ('params_b', 'active_params_b'):
        logp = np.log10(params[basis].to_numpy())
        scale[basis] = {name: dict(**S.correlation(score.to_numpy(), logp, fam_list, draws=pca_draws, seed=SEEDS['scale_corr']),
                                   within_family=S.within_family(score.to_numpy(), logp, fam_list))
                        for name, score in (('PC1', pc1), ('F1', factors.F1), ('F2', factors.F2))}
    scale['structure_params_b'] = S.scale_removed_structure(
        X.to_numpy(), np.log10(params.params_b.to_numpy()), fam_list, draws=pca_draws,
        seed=SEEDS['scale_structure'], pa_draws=pa_draws, pa_seed=SEEDS['scale_pa'])
    scale['structure_params_b']['tasks'] = codes
    bias = D.bias_indices(list(X.index))
    scale['f2_bias'] = {name: S.correlation(factors.F2.to_numpy(), bias[name].to_numpy(), fam_list,
                                            draws=pca_draws, seed=SEEDS['bias'])
                        for name in bias.columns}
    constant = bias.wag_constant_bet == 1
    scale['f2_by_constant_bet'] = dict(constant_n=int(constant.sum()), constant_mean_f2=float(factors.F2[constant].mean()),
                                       varied_n=int((~constant).sum()), varied_mean_f2=float(factors.F2[~constant].mean()))
    write_json(output / 'scale' / 'scale.json', scale)
    pd.concat([pc1.rename('PC1'), factors, params, bias], axis=1).to_csv(output / 'scale' / 'model_scores.csv')

    # C. Reliability.
    halves = pd.read_csv(halves_path)
    full = halves[halves.form == 'full'].set_index(['model', 'task']).value
    for t in X.columns:
        if t in ('drm_false_memory', 'cvlt_word_list'):
            continue
        got = full.reindex([(m, t) for m in X.index]).to_numpy()
        if not np.allclose(got, X[t].to_numpy(), rtol=1e-12, atol=1e-12):
            raise ValueError(f'Full-set half table differs from the matrix for {t}')
    summary, per_task_rel, R_star = REL.analyze(X, halves, fam_list, D.ABBR, draws=pca_draws, seed=SEEDS['reliability'])
    write_json(output / 'reliability' / 'summary.json', summary)
    per_task_rel.to_csv(output / 'reliability' / 'per_task.csv', index=False)
    if R_star is not None:
        R_star.to_csv(output / 'reliability' / 'disattenuated_correlations.csv')

    manifest.update(status='complete', finished_at=datetime.now().astimezone().isoformat(),
                    elapsed_seconds=time.monotonic() - start,
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(output / 'run_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--halves', type=Path, required=True)
    parser.add_argument('--pca-draws', type=int, default=5000)
    parser.add_argument('--prediction-draws', type=int, default=20000)
    parser.add_argument('--pa-draws', type=int, default=5000)
    a = parser.parse_args()
    print(json.dumps(run(a.output, a.halves, pca_draws=a.pca_draws, prediction_draws=a.prediction_draws,
                         pa_draws=a.pa_draws), indent=2))


if __name__ == '__main__':
    main()
