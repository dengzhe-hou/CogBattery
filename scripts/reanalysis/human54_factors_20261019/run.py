"""CPU analysis of saved human54 matrices; never reads or executes model calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from .core import correlation_matrix, fit_minres
from .parallel import parallel_analysis
from .stability import bootstrap_factors


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / 'results/reanalysis/human54_fullbattery_20261019'
TASKS = ['DS', 'NB', 'OS', 'ST', 'FL', 'GN', 'CVLT', 'DRM', 'SM', 'FB', 'EP', 'CAL', 'WAG']
FULL_TASKS = ['digit_span', 'n_back', 'operation_span', 'stroop', 'flanker', 'go_nogo',
              'cvlt_word_list', 'drm_false_memory', 'source_monitoring', 'false_belief',
              'epitome_tom', 'confidence_calibration', 'post_decision_wagering']


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def with_centroids(matrix: pd.DataFrame, families: pd.Series):
    if not matrix.index.equals(families.index):
        raise ValueError('Matrix and family rows must align exactly')
    centroids = matrix.groupby(families, sort=True).mean()
    labels = pd.Series(centroids.index.to_numpy(), index=centroids.index, name='family_merged')
    return centroids, labels


def load_cases(input_dir: Path):
    cases = {}
    for name in ['primary_valid_only', 'invalid_wrong']:
        source = input_dir / name
        matrix = pd.read_csv(source / 'matrix.csv', index_col='model')
        metadata = pd.read_csv(source / 'metadata.csv', index_col='model')
        if matrix.columns.tolist() != FULL_TASKS:
            raise ValueError('Unexpected task order: inputs must be the approved 13 columns')
        families = metadata.loc[matrix.index, 'family_merged']
        matrix.columns = TASKS
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError('Incomplete input matrix; no imputation is allowed')
        cases[name] = (matrix, families)
        cases[name + '_centroids'] = with_centroids(matrix, families)
    return cases


def synthetic_cases(seed=317):
    """Correlated three-factor synthetic models nested in 23 families."""
    rng = np.random.default_rng(seed)
    phi = np.array([[1., .35, .2], [.35, 1., .25], [.2, .25, 1.]])
    loading = np.zeros((13, 3))
    for j in range(13):
        loading[j, min(j // 5, 2)] = .8
    family_ids = np.repeat(np.arange(23), [3] * 7 + [2] * 16)
    family_ids = np.r_[family_ids, 0]
    latent = rng.multivariate_normal(np.zeros(3), phi, 23)
    values = (latent[family_ids] + rng.normal(0, .2, (54, 3))) @ loading.T
    values += rng.normal(0, .5, (54, 13))
    # Heterogeneous task units exercise averaging before standardization.
    values = values * np.linspace(.2, 8., 13) + np.arange(13)
    ids = [f'synthetic_model_{i:02d}' for i in range(54)]
    all_matrix = pd.DataFrame(values, index=pd.Index(ids, name='model'), columns=TASKS)
    all_families = pd.Series([f'family_{i:02d}' for i in family_ids],
                             index=all_matrix.index, name='family_merged')
    cases = {}
    for name, n in [('primary_valid_only', 53), ('invalid_wrong', 54)]:
        matrix, families = all_matrix.iloc[:n].copy(), all_families.iloc[:n].copy()
        cases[name] = (matrix, families)
        cases[name + '_centroids'] = with_centroids(matrix, families)
    return cases


def candidate_dimensions(pa_summaries, p):
    supported = sorted({int(s['fa']['retained']) for s in pa_summaries.values()
                        if s['fa']['retained'] is not None})
    candidates = sorted({j for k in supported for j in (k-1, k, k+1)
                         if j >= 1 and ((p-j)**2-p-j)/2 >= 0})
    return supported, candidates


def save_fit(path, fit, tasks):
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / 'fit.json', fit)
    k = np.asarray(fit['pattern']).shape[1]
    factors = [f'F{i+1}' for i in range(k)]
    for key in ['pattern', 'structure', 'unrotated']:
        pd.DataFrame(fit[key], index=tasks, columns=factors).to_csv(path / f'{key}.csv', index_label='task')
    pd.DataFrame(fit['phi'], index=factors, columns=factors).to_csv(path / 'factor_correlations.csv', index_label='factor')
    pd.DataFrame({'task': tasks, 'communality': fit['communalities'],
                  'uniqueness': fit['uniqueness']}).to_csv(path / 'communality.csv', index=False)
    pd.DataFrame(fit['residual'], index=tasks, columns=tasks).to_csv(path / 'residuals.csv', index_label='task')


def execute(cases, output, pa_draws=5000, bootstrap_draws=2000, seed=20261019, synthetic=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already contains results: {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seeds = np.random.SeedSequence(seed).spawn(2 * len(cases))
    manifest = {'status': 'running', 'synthetic': synthetic, 'seed': seed,
                'pa_draws': pa_draws, 'bootstrap_draws': bootstrap_draws,
                'model_calls': 0, 'formal_responses_read': 0, 'device': 'CPU',
                'threads': {v: os.environ.get(v) for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']},
                'numpy_version': np.__version__, 'cases': {}}
    write_json(output / 'run_manifest.json', manifest)
    summaries = {}
    for i, (name, (matrix, families)) in enumerate(cases.items()):
        case_dir = output / name
        case_dir.mkdir()
        matrix.to_csv(case_dir / 'matrix.csv', index_label='model_or_family')
        families.to_frame().to_csv(case_dir / 'metadata.csv', index_label='model_or_family')
        pa_seed = int(seeds[2*i].generate_state(1)[0])
        bs_seed = int(seeds[2*i+1].generate_state(1)[0])
        manifest['cases'][name] = {'n': len(matrix), 'families': families.nunique(),
                                 'pa_seed': pa_seed, 'bootstrap_seed': bs_seed}
        print(f'{name}: PA {len(matrix)} rows, {families.nunique()} families', flush=True)
        pa, arrays = parallel_analysis(matrix.to_numpy(), draws=pa_draws, seed=pa_seed,
            progress=lambda done, total: print(f'{name}: PA {done}/{total}', flush=True))
        write_json(case_dir / 'parallel.json', pa)
        np.savez_compressed(case_dir / 'parallel_draws.npz', **arrays)
        summaries[name] = pa
        write_json(output / 'run_manifest.json', manifest)
        print(f"{name}: retained PC={pa['pc']['retained']}, FA={pa['fa']['retained']}", flush=True)
    suggested, candidates = candidate_dimensions(summaries, len(TASKS))
    manifest.update(fa_suggested_dimensions=suggested, efa_candidate_dimensions=candidates)
    write_json(output / 'run_manifest.json', manifest)
    print(f'FA suggested={suggested}; EFA candidates={candidates}', flush=True)
    overview = []
    for name, (matrix, families) in cases.items():
        case_dir = output / name
        R = correlation_matrix(matrix.to_numpy())
        pd.DataFrame(R, index=TASKS, columns=TASKS).to_csv(case_dir / 'correlations.csv', index_label='task')
        fits = {}
        for k in candidates:
            fit = fit_minres(R, k)
            fits[k] = fit
            save_fit(case_dir / f'efa_{k}', fit, TASKS)
            overview.append(dict(case=name, n=len(matrix), n_families=families.nunique(), factors=k,
                                 rmsr=fit['rmsr'], converged=fit['converged'], proper=fit['proper'],
                                 boundary=fit['boundary']))
            print(f"{name}: {k} factors, RMSR={fit['rmsr']:.5f}, converged={fit['converged']}, "
                  f"proper={fit['proper']}, boundary={fit['boundary']}", flush=True)
        if fits:
            boot, arrays = bootstrap_factors(matrix.to_numpy(), families.tolist(), fits,
                draws=bootstrap_draws, seed=manifest['cases'][name]['bootstrap_seed'],
                progress=lambda event: print(f"{name}: bootstrap {event['draw']}/{event['draws']}", flush=True))
            write_json(case_dir / 'stability.json', boot)
            np.savez_compressed(case_dir / 'stability_draws.npz', **arrays)
    pd.DataFrame(overview).to_csv(output / 'fit_summary.csv', index=False)
    pa_table = []
    for name, pa in summaries.items():
        for kind in ['pc', 'fa']:
            values = pa[kind]['eigenvalues'] or [None] * len(TASKS)
            thresholds = pa[kind]['threshold95'] or [None] * len(TASKS)
            for j, (value, threshold) in enumerate(zip(values, thresholds)):
                pa_table.append(dict(case=name, kind=kind, rank=j+1, eigenvalue=value,
                                     threshold95=threshold, retained=pa[kind]['retained']))
    pd.DataFrame(pa_table).to_csv(output / 'parallel_summary.csv', index=False)
    manifest.update(status='complete', elapsed_seconds=time.perf_counter()-started,
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(output / 'run_manifest.json', manifest)
    print(json.dumps(json_ready(manifest), ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--pa-draws', type=int, default=5000)
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args()
    cases = synthetic_cases() if args.synthetic else load_cases(args.input)
    execute(cases, args.output, args.pa_draws, args.bootstrap_draws, args.seed, args.synthetic)


if __name__ == '__main__':
    main()
