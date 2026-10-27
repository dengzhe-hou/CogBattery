"""Run the approved post-hoc full-battery analysis on saved endpoint tables."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import resource
import socket
import time

import numpy as np
import pandas as pd

from .matrix import TASKS, TASK_ABBREVIATIONS, build_matrices
from .pca import analyze_pca
from .prediction import analyze_prediction

ROOT = Path(__file__).resolve().parents[3]
SEED = 20261018
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run_analysis(rows, output, *, synthetic, pca_draws=5000, prediction_draws=20000,
                 source='in-memory synthetic fixture'):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
        synthetic=synthetic, post_hoc=True, source=source, seed=SEED,
        spec=str(ROOT/'review-stage/human54_fullbattery_20261019/SPEC.md'),
        threads={name: os.environ.get(name) for name in THREADS}, host=socket.gethostname(),
        model_calls=0, formal_response_record_reads=0, formal_record_rescoring=False,
        pca_draws=pca_draws, prediction_draws=prediction_draws,
        input_rows=len(rows), analysis_configurations={})
    save_json(output/'run_manifest.json', manifest)
    matrices = build_matrices(rows)
    for rule, values in matrices.items():
        folder = output/'inputs'/rule
        folder.mkdir(parents=True)
        for name in ['matrix', 'all_candidates', 'metadata', 'endpoint_ledger', 'wag_sensitivity']:
            values[name].to_csv(folder/f'{name}.csv', index=name != 'endpoint_ledger')
        save_json(folder/'exclusions.json', values['exclusions'])
    primary = matrices['valid_only']
    wag = primary['wag_sensitivity']
    wag_frame = primary['matrix'].copy()
    wag_frame['post_decision_wagering'] = wag['value'].reindex(wag_frame.index)
    wag_frame = wag_frame.dropna()
    configurations = [
        ('primary_valid_only', primary['matrix'], primary['metadata']),
        ('invalid_wrong', matrices['invalid_wrong']['matrix'], matrices['invalid_wrong']['metadata']),
        ('wag_highlow_sensitivity', wag_frame, primary['metadata'].reindex(wag_frame.index)),
    ]
    seeds = [int(stream.generate_state(1)[0]) for stream in np.random.SeedSequence(SEED).spawn(6)]
    all_summaries = {}
    summary_rows = []
    for i, (name, frame, metadata) in enumerate(configurations):
        print(json.dumps(dict(stage='configuration_start', name=name, models=len(frame),
            families=int(metadata.family_merged.nunique()), synthetic=synthetic)), flush=True)
        folder = output/name
        folder.mkdir()
        frame.to_csv(folder/'matrix.csv')
        metadata.to_csv(folder/'metadata.csv')
        codes = [TASK_ABBREVIATIONS[t] for t in frame.columns]
        families = metadata.family_merged.tolist()
        pca, pca_arrays = analyze_pca(frame.to_numpy(), families, codes,
                                    seed=seeds[2*i], draws=pca_draws)
        pca['task_names'] = list(frame.columns)
        pca['model_names'] = list(frame.index)
        save_json(folder/'pca.json', pca)
        np.savez_compressed(folder/'pca_bootstrap.npz', **pca_arrays)
        pd.DataFrame(pca['correlation_matrix'], index=codes, columns=codes).to_csv(folder/'correlations.csv')
        pd.DataFrame(pca['correlation_loadings'], index=codes,
            columns=[f'PC{j+1}' for j in range(len(codes))]).to_csv(folder/'correlation_loadings.csv')
        pd.DataFrame(pca['scores'], index=frame.index,
            columns=[f'PC{j+1}' for j in range(len(codes))]).to_csv(folder/'pca_scores.csv')
        prediction, artifacts = analyze_prediction(frame, metadata.family_merged.to_dict(),
            seed=seeds[2*i+1], draws=prediction_draws)
        save_json(folder/'prediction.json', prediction)
        for key in ['predictions', 'cells', 'per_target', 'per_family']:
            artifacts[key].to_csv(folder/f'prediction_{key}.csv', index=False)
        save_json(folder/'prediction_fold_trace.json', artifacts['fold_trace'])
        np.savez_compressed(folder/'prediction_bootstrap.npz', **artifacts['bootstrap_draws'])
        ci = pca['bootstrap']['intervals']['pc1_share']
        config_summary = dict(name=name, models=len(frame), families=len(set(families)),
            pca_seed=seeds[2*i], prediction_seed=seeds[2*i+1],
            pc1_share=pca['variance_shares'][0], pc1_lower=ci['lower'], pc1_upper=ci['upper'],
            pc2_share=pca['variance_shares'][1], pc3_share=pca['variance_shares'][2],
            pca_bootstrap_valid=pca['bootstrap']['n_valid'],
            pca_bootstrap_invalid=pca['bootstrap']['n_invalid'], **pca['structure'],
            rmse_g=prediction['rmse_g'], rmse_group=prediction['rmse_g_group'],
            rmse_intercept=prediction['rmse_intercept'],
            group_gain=prediction['relative_rmse_gain'],
            group_gain_lower=prediction['bootstrap']['ci95'][0],
            group_gain_upper=prediction['bootstrap']['ci95'][1],
            n_positive_targets=prediction['n_positive_targets'],
            n_positive_families=prediction['n_positive_families'],
            n_prediction_rows=prediction['n_prediction_rows'],
            n_family_target_cells=prediction['n_family_target_cells'])
        all_summaries[name] = {'pca': pca, 'prediction': prediction, 'summary': config_summary}
        summary_rows.append(config_summary)
        manifest['analysis_configurations'][name] = config_summary
        save_json(output/'run_manifest.json', manifest)
        print(json.dumps(dict(stage='configuration_complete', **config_summary)), flush=True)
    pd.DataFrame(summary_rows).to_csv(output/'summary.csv', index=False)
    save_json(output/'summary.json', dict(synthetic=synthetic, post_hoc=True,
        configurations={name: value['summary'] for name, value in all_summaries.items()}))
    manifest.update(status='complete', completed_at=datetime.now().astimezone().isoformat(),
        wall_seconds=time.perf_counter()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save_json(output/'run_manifest.json', manifest)
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--pca-draws', type=int, default=5000)
    parser.add_argument('--prediction-draws', type=int, default=20000)
    args = parser.parse_args()
    for name in THREADS:
        if os.environ.get(name) != '1':
            raise ValueError(f'Run single-thread CPU analysis with {name}=1')
    if args.synthetic:
        if args.input:
            raise ValueError('Synthetic mode generates its own fixture and cannot read a formal input')
        from .synthetic import make_synthetic_rows
        rows, source = make_synthetic_rows(), 'generated synthetic endpoint-schema fixture'
    else:
        if not args.input:
            raise ValueError('An explicit saved endpoint table is required')
        with args.input.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        source = str(args.input.resolve())
    run_analysis(rows, args.output, synthetic=args.synthetic, pca_draws=args.pca_draws,
                 prediction_draws=args.prediction_draws, source=source)


if __name__ == '__main__':
    main()
