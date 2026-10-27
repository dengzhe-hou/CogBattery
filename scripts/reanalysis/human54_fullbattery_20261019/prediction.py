"""Complete-case family-held-out prediction using the existing G0 protocol.

Pure CPU functions. The imported feature/fit routines never read old results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.experiments.profile_validity_20260720.family_heldout_utility import (
    GROUPS, RIDGE_ALPHA, build_features_from_frames, heldout_predictions,
)


def paired_bootstrap(cells, *, draws=20_000, seed=20261018):
    """Resample entire held-out family error blocks; fits are held fixed."""
    columns = ['se_g', 'se_group', 'se_intercept']
    families = sorted(cells.family.unique())
    sums = cells.groupby('family')[columns].sum().reindex(families).to_numpy()
    counts = cells.groupby('family').size().reindex(families).to_numpy()
    sampled = np.random.default_rng(seed).integers(0, len(families),
                                                  size=(draws, len(families)))
    rmse = np.sqrt(sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)[:, None])
    gain = 1 - rmse[:, 1] / rmse[:, 0]
    ci = np.quantile(gain, [.025, .975])
    return {
        'draws': draws, 'seed': seed, 'ci95': ci.tolist(),
        'scope': 'paired whole-family blocks of fixed out-of-fold errors; no refitting',
    }, {'sampled_family_indices': sampled, 'rmse_g': rmse[:, 0],
        'rmse_group': rmse[:, 1], 'rmse_intercept': rmse[:, 2], 'relative_gain': gain}


def _metrics(cells):
    g, group, intercept = [float(np.sqrt(cells[c].mean()))
                           for c in ['se_g', 'se_group', 'se_intercept']]
    return {'rmse_g': g, 'rmse_g_group': group, 'rmse_intercept': intercept,
            'relative_rmse_gain': 1 - group / g,
            'general_gain_vs_intercept': 1 - g / intercept}


def analyze_prediction(frame, family_by_model, *, seed=20261018, draws=20_000):
    """Fit on training family centroids, evaluate held-out checkpoints by cell."""
    expected = {task for tasks in GROUPS.values() for task in tasks}
    if set(frame.columns) != expected or not frame.index.is_unique:
        raise ValueError('Prediction requires one column per approved task and unique models')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Prediction requires the approved complete-case matrix; no imputation')
    families = {model: str(family_by_model[model]) for model in frame.index}
    predictions = heldout_predictions(frame, families, {
        task: group for group, tasks in GROUPS.items() for task in tasks})
    predictions['pred_intercept_z'] = 0.
    predictions['se_g'] = (predictions.truth_z - predictions.pred_g_z)**2
    predictions['se_group'] = (predictions.truth_z - predictions.pred_g_group_z)**2
    predictions['se_intercept'] = predictions.truth_z**2
    for short in ['g', 'g_group', 'intercept']:
        predictions[f'pred_{short}_raw'] = (predictions[f'pred_{short}_z'] *
            predictions.train_outcome_sd + predictions.train_outcome_mean)
    predictions['truth_raw'] = (predictions.truth_z * predictions.train_outcome_sd +
                                 predictions.train_outcome_mean)
    columns = ['se_g', 'se_group', 'se_intercept']
    cells = predictions.groupby(['family', 'target'], as_index=False)[columns].mean()
    counts = predictions.groupby(['family', 'target']).size().rename('n_models')
    cells = cells.join(counts, on=['family', 'target'])
    per_target = pd.DataFrame([dict(target=t, **_metrics(sub))
        for t, sub in cells.groupby('target', sort=True)])
    per_family = pd.DataFrame([dict(family=f, **_metrics(sub))
        for f, sub in cells.groupby('family', sort=True)])
    bootstrap, arrays = paired_bootstrap(cells, draws=draws, seed=seed)
    summary = dict(_metrics(cells), n_models=len(frame), n_families=len(set(families.values())),
        n_prediction_rows=len(predictions), n_family_target_cells=len(cells),
        n_positive_targets=int((per_target.relative_rmse_gain > 0).sum()),
        n_positive_families=int((per_family.relative_rmse_gain > 0).sum()),
        bootstrap=bootstrap, ridge_alpha=RIDGE_ALPHA, fitting_unit='training family centroid',
        scoring_unit='equally weighted family-by-target cell',
        target_scale='training-family centroid mean and population SD',
        baseline='PC1 of the other 12 tasks',
        augmented='baseline plus mean of same-group peer task z scores, residualized on PC1',
        groups={g: sorted(ts) for g, ts in GROUPS.items()},
        tuning='none; inherited fixed ridge penalty', imputation='none')
    trace = []
    for family in sorted(set(families.values())):
        training = frame.loc[[m for m in frame.index if families[m] != family]]
        centroids = training.groupby(pd.Series(families).reindex(training.index)).mean()
        trace.append({'heldout_family': family, 'training_families': list(centroids.index),
            'test_models': [m for m in frame.index if families[m] == family],
            'training_target_means': centroids.mean().to_dict(),
            'training_target_sds': centroids.std(ddof=0).to_dict()})
    return summary, {'predictions': predictions, 'cells': cells, 'per_target': per_target,
        'per_family': per_family, 'bootstrap_draws': arrays, 'fold_trace': trace}
