"""Inputs for the post hoc analyses: saved matrices, family maps, parameter counts, bias indices."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FULLBATTERY = ROOT / 'results/reanalysis/human54_fullbattery_20261019'
FACTORS = ROOT / 'results/reanalysis/human54_factors_20261019'
ENDPOINTS = ROOT / 'results/reanalysis/human54_layers_20261019/layer1/endpoints.csv'
FORMAL = ROOT / 'results/human54_unified_20261018/formal'
OLLAMA = Path.home() / '.ollama/models'
ABBR = dict(digit_span='DS', n_back='NB', operation_span='OS', stroop='ST', flanker='FL', go_nogo='GN',
            cvlt_word_list='CVLT', drm_false_memory='DRM', source_monitoring='SM', false_belief='FB',
            epitome_tom='EP', confidence_calibration='CAL', post_decision_wagering='WAG')

# Reviewer-listed base-model lineages (META_REVIEW.md, sixth item): R1 distills are
# Qwen2.5 checkpoints; zephyr, openchat, starling and SOLAR start from Mistral 7B weights.
LINEAGE = {'deepseek-r': 'qwen', 'zephyr': 'mistral', 'openchat': 'mistral',
           'starling': 'mistral', 'solar': 'mistral'}
# Aya 23 is built on Cohere's Command base models (spec section D, 17-family variant).
LINEAGE_AYA = {**LINEAGE, 'aya': 'command-r'}
# Mistral AI's published active parameters per token (billions).
MOE_ACTIVE_B = {'mixtral:8x7b': 12.9, 'mixtral:8x22b': 39.0}


def load_matrix(rule='primary_valid_only'):
    folder = FULLBATTERY / rule
    X = pd.read_csv(folder / 'matrix.csv', index_col='model')
    meta = pd.read_csv(folder / 'metadata.csv', index_col='model')
    if list(X.index) != list(meta.index):
        raise ValueError('Matrix and metadata rows differ')
    return X, meta


def family_maps(meta):
    merged = meta.family_merged
    return {'merged23': merged.to_dict(), 'raw28': meta.family_raw.to_dict(),
            'lineage18': merged.replace(LINEAGE).to_dict(),
            'lineage17_aya': merged.replace(LINEAGE_AYA).to_dict()}


def _billions(text):
    number, unit = re.fullmatch(r'\s*([0-9.]+)\s*([KMBT])\s*', text).groups()
    return float(number) * {'K': 1e-6, 'M': 1e-3, 'B': 1., 'T': 1e3}[unit]


def parameter_table(models):
    """Parameters of the served weights. llama.cpp load log when present, else ollama config."""
    rows = []
    for model in models:
        folder = FORMAL / model.replace(':', '__')
        found = None
        for log in sorted(folder.glob('ollama_*.log')):
            match = re.search(r'model params\s*=\s*([0-9.]+ [KMBT])', log.read_text(errors='replace'))
            if match:
                found = (_billions(match.group(1)), 'llama.cpp load log ' + log.name)
                break
        if found is None:
            name, tag = model.split(':')
            manifest = OLLAMA / 'manifests/registry.ollama.ai/library' / name / tag
            config = json.loads(manifest.read_text())['config']['digest'].replace(':', '-')
            model_type = json.loads((OLLAMA / 'blobs' / config).read_text())['model_type']
            found = (_billions(model_type[:-1] + ' ' + model_type[-1]), 'ollama config model_type ' + model_type)
        rows.append(dict(model=model, params_b=found[0], source=found[1],
                         active_params_b=MOE_ACTIVE_B.get(model, found[0])))
    return pd.DataFrame(rows).set_index('model')


def bias_indices(models):
    e = pd.read_csv(ENDPOINTS, low_memory=False)
    e = e[e.rule == 'valid_only'].set_index(['model', 'task', 'endpoint']).value
    get = lambda task, name: e.reindex([(m, task, name) for m in models]).to_numpy(dtype=float)
    high = get('post_decision_wagering', 'high_wager_rate')
    return pd.DataFrame({
        'nb_control_fa': (get('n_back', 'load_2_control') + get('n_back', 'load_3_control')) / 2,
        'wag_constant_bet': np.where(np.isfinite(high), np.isin(high, (0., 1.)).astype(float), np.nan),
        'wag_high_wager_rate': high,
        'nb_lure_fa': (get('n_back', 'load_2_lure') + get('n_back', 'load_3_lure')) / 2,
        'gn_false_alarm': get('go_nogo', 'false_alarm_rate'),
        'drm_new_fa': (get('drm_false_memory', 'no_warning|new_false_alarm_rate')
                       + get('drm_false_memory', 'warning|new_false_alarm_rate')) / 2,
        'cvlt_recognition_fa': get('cvlt_word_list', 'recognition_false_alarm_rate'),
    }, index=models)


def factor_scores(X):
    """Regression-method scores of the saved primary two-factor solution."""
    folder = FACTORS / 'primary_valid_only'
    pattern = pd.read_csv(folder / 'efa_2/pattern.csv', index_col='task')
    phi = pd.read_csv(folder / 'efa_2/factor_correlations.csv', index_col=0).to_numpy(dtype=float)
    saved_R = pd.read_csv(folder / 'correlations.csv', index_col=0).to_numpy(dtype=float)
    if list(pattern.index) != [ABBR[t] for t in X.columns]:
        raise ValueError('Factor pattern task order differs from the matrix')
    Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
    R = Z.T @ Z / len(Z)
    if not np.allclose(R, saved_R, atol=1e-10):
        raise ValueError('Matrix correlations differ from the saved factor-analysis input')
    W = np.linalg.solve(R, pattern.to_numpy() @ phi)
    return pd.DataFrame(Z @ W, index=X.index, columns=['F1', 'F2'])
