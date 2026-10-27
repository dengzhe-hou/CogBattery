"""Model-side reliability and disattenuated grouping contrast (spec section C)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_fullbattery_20261019.pca import TASK_GROUPS, family_bootstrap_counts


def forms_matrix(halves, task, models):
    sub = halves[(halves.task == task) & (halves.form != 'full')]
    wide = sub.pivot(index='model', columns='form', values='value')
    return wide.reindex(models)


def reliability(F):
    """Spearman-Brown of the half correlation (two forms) or Cronbach alpha (three or more)."""
    F = np.asarray(F, float)
    F = F[np.isfinite(F).all(axis=1)]
    k = F.shape[1]
    if len(F) < 3 or np.any(F.std(axis=0) == 0):
        return dict(rel=np.nan, mean_r=np.nan, n=len(F), k=k)
    R = np.corrcoef(F, rowvar=False)
    mean_r = R[np.triu_indices(k, 1)].mean()
    if k == 2:
        rel = 2 * mean_r / (1 + mean_r)
    else:
        total = F.sum(axis=1)
        rel = k / (k - 1) * (1 - F.var(axis=0, ddof=1).sum() / total.var(ddof=1))
    return dict(rel=float(rel), mean_r=float(mean_r), n=int(len(F)), k=int(k))


def _structure(R, groups):
    row, col = np.triu_indices(len(groups), 1)
    within = groups[row] == groups[col]
    pairs = R[row, col]
    return pairs[within].mean(), pairs[~within].mean()


def analyze(matrix, halves, families, abbreviations, *, draws, seed):
    tasks = list(matrix.columns)
    models = list(matrix.index)
    forms = {t: forms_matrix(halves, t, models) for t in tasks}
    groups = np.array([TASK_GROUPS[abbreviations[t]] for t in tasks])
    X = matrix.to_numpy(float)

    def compute(idx):
        Z = X[idx]
        R = np.corrcoef(Z, rowvar=False)
        rel = np.array([reliability(forms[t].to_numpy()[idx])['rel'] for t in tasks])
        w, b = _structure(R, groups)
        out = dict(rel=rel, within=w, between=b, delta=w - b)
        if np.all(rel > 0):
            Rs = R / np.sqrt(np.outer(rel, rel))
            np.fill_diagonal(Rs, 1.)
            ws, bs = _structure(Rs, groups)
            out.update(within_star=ws, between_star=bs, delta_star=ws - bs,
                       n_above_one=int(np.sum(np.abs(Rs[np.triu_indices(len(tasks), 1)]) > 1)), R_star=Rs)
        return out

    point = compute(np.arange(len(models)))
    per_task = []
    for t, rel in zip(tasks, point['rel']):
        detail = reliability(forms[t].to_numpy())
        per_task.append(dict(task=abbreviations[t], forms=list(forms[t].columns), **detail))
    _, codes, counts = family_bootstrap_counts(families, draws, seed)
    keys = ('within', 'between', 'delta', 'within_star', 'between_star', 'delta_star')
    boot = {k: np.full(draws, np.nan) for k in keys}
    boot_rel = np.full((draws, len(tasks)), np.nan)
    undefined = 0
    for i, c in enumerate(counts):
        idx = np.repeat(np.arange(len(models)), c[codes])
        if np.any(X[idx].std(axis=0) == 0):
            undefined += 1
            continue
        r = compute(idx)
        boot_rel[i] = r['rel']
        for k in keys:
            if k in r:
                boot[k][i] = r[k]
    interval = lambda v: dict(lower=float(np.nanquantile(v, .025)), upper=float(np.nanquantile(v, .975)),
                              n_valid=int(np.isfinite(v).sum()))
    for row, j in zip(per_task, range(len(tasks))):
        row['rel_ci'] = interval(boot_rel[:, j])
    summary = {k: float(point[k]) for k in keys if k in point}
    summary.update(n_above_one=point.get('n_above_one'),
                   intervals={k: interval(boot[k]) for k in keys},
                   draws=draws, seed=int(seed), draws_with_constant_task=undefined,
                   draws_without_positive_reliability=int(np.sum(~np.isfinite(boot['delta_star']))) - undefined)
    R_star = pd.DataFrame(point['R_star'], index=[abbreviations[t] for t in tasks],
                          columns=[abbreviations[t] for t in tasks]) if 'R_star' in point else None
    return summary, pd.DataFrame(per_task), R_star
