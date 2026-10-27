"""Family-held-out prediction for arbitrary task groupings (post hoc, spec section B).

Same protocol as profile_validity_20260720/family_heldout_utility.py (G0): training
family centroids, z scores from training centroids, PC1 of the other tasks as the
general axis, ridge alpha 1e-6 with intercept, held-out models scored by
family-by-target cell. Differences: any peer set per target (including none,
which leaves the general-axis prediction), all peer sets of size one or two
evaluated per fold for the partition reference, and two-dimensional baselines.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from scripts.experiments.profile_validity_20260720.family_heldout_utility import GROUPS, RIDGE_ALPHA
from scripts.reanalysis.human54_factors_20261019.core import fit_minres

FIVE_GROUPS = {task: group for group, tasks in GROUPS.items() for task in tasks}


def ridge(X, y, alpha=RIDGE_ALPHA):
    """sklearn Ridge(fit_intercept=True) closed form: centre, then solve."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc = X - xm
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return beta, ym - xm @ beta


def _pcs(train_z, test_z, k):
    mean = train_z.mean(axis=0)
    _, _, vt = np.linalg.svd(train_z - mean, full_matrices=False)
    g_train, g_test = (train_z - mean) @ vt[:k].T, (test_z - mean) @ vt[:k].T
    # PC1 orientation as in G0: higher PC1 means higher average performance.
    if np.corrcoef(g_train[:, 0], train_z.mean(axis=1))[0, 1] < 0:
        g_train[:, 0] *= -1
        g_test[:, 0] *= -1
    return g_train, g_test


def _factor_scores(train_z, test_z):
    """Two-factor MINRES regression scores. Oblique rotation does not change the
    span of regression scores, so the unrotated solution gives the same predictions."""
    n = len(train_z)
    R = train_z.T @ train_z / n
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.)
    fit = fit_minres(R, 2, rotate=False)
    W = np.linalg.solve(R, fit['pattern'])
    status = dict(converged=fit['converged'], proper=fit['proper'], boundary=fit['boundary'],
                  heywood=[int(i) for i in fit['diagnostics']['heywood_variables']])
    return train_z @ W, test_z @ W, status


def peer_catalog(tasks, target, sizes=(1, 2)):
    others = [t for t in tasks if t != target]
    return [frozenset(c) for s in sizes for c in combinations(others, s)]


def fold_predictions(frame, family_by_model, peer_sets, *, two_dim=True):
    """Return per-model predictions and fold diagnostics.

    ``peer_sets[target]`` lists frozensets of predictor task names. An empty set
    means "no same-group information": the augmented prediction equals PC1 only.
    """
    tasks = list(frame.columns)
    families = sorted(set(family_by_model[m] for m in frame.index))
    fam = pd.Series({m: family_by_model[m] for m in frame.index})
    rows, diagnostics = [], []
    for target in tasks:
        predictors = [t for t in tasks if t != target]
        for heldout in families:
            test_models = [m for m in frame.index if fam[m] == heldout]
            train = frame.loc[fam != heldout]
            centroids = train.groupby(fam.reindex(train.index)).mean()
            if len(centroids) != len(families) - 1:
                raise ValueError('Expected one centroid per training family')
            test = frame.loc[test_models]
            X_train = centroids[predictors].to_numpy(dtype=float)
            X_test = test[predictors].to_numpy(dtype=float)
            mean, sd = X_train.mean(axis=0), X_train.std(axis=0, ddof=0)
            if not np.all(sd > 1e-10):
                raise ValueError(f'Zero-variance predictor for {heldout}/{target}')
            train_z, test_z = (X_train - mean) / sd, (X_test - mean) / sd
            y_raw = centroids[target].to_numpy(dtype=float)
            y_mean, y_sd = y_raw.mean(), y_raw.std(ddof=0)
            if not y_sd > 1e-10:
                raise ValueError(f'Zero-variance outcome for {heldout}/{target}')
            y_train = (y_raw - y_mean) / y_sd
            truth = (test[target].to_numpy(dtype=float) - y_mean) / y_sd
            g2_train, g2_test = _pcs(train_z, test_z, 2)
            g_train, g_test = g2_train[:, :1], g2_test[:, :1]
            beta, b0 = ridge(g_train, y_train)
            pred_g = g_test @ beta + b0
            out = dict(target=target, family=heldout, models=test_models, truth=truth,
                       pred_g=pred_g, pred_intercept=np.zeros_like(truth), peer={})
            design_train = np.column_stack([np.ones(len(g_train)), g_train[:, 0]])
            design_test = np.column_stack([np.ones(len(g_test)), g_test[:, 0]])
            for peers in peer_sets[target]:
                if not peers:
                    out['peer'][peers] = pred_g
                    continue
                idx = [predictors.index(p) for p in peers]
                peer_train, peer_test = train_z[:, idx].mean(axis=1), test_z[:, idx].mean(axis=1)
                coef, *_ = np.linalg.lstsq(design_train, peer_train, rcond=None)
                Xa_train = np.column_stack([g_train[:, 0], peer_train - design_train @ coef])
                Xa_test = np.column_stack([g_test[:, 0], peer_test - design_test @ coef])
                beta, b0 = ridge(Xa_train, y_train)
                out['peer'][peers] = Xa_test @ beta + b0
            if two_dim:
                beta, b0 = ridge(g2_train, y_train)
                out['pred_pc12'] = g2_test @ beta + b0
                f_train, f_test, status = _factor_scores(train_z, test_z)
                beta, b0 = ridge(f_train, y_train)
                out['pred_fa2'] = f_test @ beta + b0
                diagnostics.append(dict(target=target, heldout_family=heldout, **status))
            rows.append(out)
    return rows, diagnostics


def theory_peers(tasks, groups=FIVE_GROUPS):
    """Same-group peers present in ``tasks``; empty when a target has none."""
    return {t: [frozenset(p for p in tasks if p != t and groups[p] == groups[t])] for t in tasks}


def cell_table(rows, peer_of_target, extra=('pc12', 'fa2')):
    """Family-by-target cells: mean squared error over held-out models."""
    out = []
    for r in rows:
        cell = dict(family=r['family'], target=r['target'], n_models=len(r['models']),
                    se_g=float(np.mean((r['truth'] - r['pred_g'])**2)),
                    se_group=float(np.mean((r['truth'] - r['peer'][peer_of_target[r['target']]])**2)),
                    se_intercept=float(np.mean(r['truth']**2)))
        for name in extra:
            if 'pred_' + name in r:
                cell['se_' + name] = float(np.mean((r['truth'] - r['pred_' + name])**2))
        out.append(cell)
    return pd.DataFrame(out)


def rmse(cells, column):
    return float(np.sqrt(cells[column].mean()))


def gain(cells, numerator='se_group', denominator='se_g'):
    return 1 - rmse(cells, numerator) / rmse(cells, denominator)


def paired_bootstrap(cells, pairs, *, draws, seed):
    """Whole-family blocks of fixed out-of-fold cell errors (as fullbattery prediction.py).

    ``pairs`` maps a label to (numerator column, denominator column);
    returns label -> dict(ci95, draws). The same sampled families serve every pair.
    """
    families = sorted(cells.family.unique())
    columns = sorted({c for pair in pairs.values() for c in pair})
    sums = cells.groupby('family')[columns].sum().reindex(families).to_numpy()
    counts = cells.groupby('family').size().reindex(families).to_numpy()
    sampled = np.random.default_rng(seed).integers(0, len(families), size=(draws, len(families)))
    rmse_draws = np.sqrt(sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)[:, None])
    out = {}
    for label, (num, den) in pairs.items():
        g = 1 - rmse_draws[:, columns.index(num)] / rmse_draws[:, columns.index(den)]
        out[label] = dict(ci95=np.quantile(g, [.025, .975]).tolist(), draws=draws, seed=int(seed))
    return out


def enumerate_partitions(n_items, sizes):
    """All partitions of range(n_items) into unlabeled blocks with the given sizes.

    Each block is opened by the smallest unassigned item, so each partition
    appears once. Returns a list of tuples of blocks.
    """
    out = []

    def recurse(remaining, sizes_left, blocks):
        if not remaining:
            out.append(tuple(blocks))
            return
        first, rest = remaining[0], remaining[1:]
        for size in sorted(set(sizes_left)):
            left = list(sizes_left)
            left.remove(size)
            for others in combinations(rest, size - 1):
                taken = set(others)
                recurse([x for x in rest if x not in taken], left, blocks + [(first, *others)])

    if sum(sizes) != n_items:
        raise ValueError('Block sizes must sum to the number of items')
    recurse(list(range(n_items)), list(sizes), [])
    return out


def partition_reference(rows, tasks, partitions):
    """Gain of every partition from the per-fold peer-set predictions.

    Returns (gains array, rmse_g, n_cells). Requires each target's rows to hold
    predictions for every peer set the partitions use.
    """
    tasks = list(tasks)
    index = {t: {} for t in tasks}
    totals = {t: [] for t in tasks}
    se_g_total, n_cells = 0., 0
    by_target = {t: [r for r in rows if r['target'] == t] for t in tasks}
    for t in tasks:
        catalog = list(by_target[t][0]['peer'])
        for k, peers in enumerate(catalog):
            index[t][peers] = k
            totals[t].append(sum(float(np.mean((r['truth'] - r['peer'][peers])**2)) for r in by_target[t]))
        se_g_total += sum(float(np.mean((r['truth'] - r['pred_g'])**2)) for r in by_target[t])
        n_cells += len(by_target[t])
    codes = np.empty((len(partitions), len(tasks)), dtype=np.int16)
    for i, blocks in enumerate(partitions):
        for block in blocks:
            for member in block:
                peers = frozenset(tasks[j] for j in block if j != member)
                codes[i, member] = index[tasks[member]][peers]
    S = [np.asarray(totals[t]) for t in tasks]
    total = sum(S[j][codes[:, j]] for j in range(len(tasks)))
    rmse_g = np.sqrt(se_g_total / n_cells)
    return 1 - np.sqrt(total / n_cells) / rmse_g, float(rmse_g), n_cells


def theory_partition_index(partitions, tasks, groups=FIVE_GROUPS):
    tasks = list(tasks)
    target = {frozenset(tasks.index(t) for t in tasks if groups[t] == g) for g in set(groups[t] for t in tasks)}
    hits = [i for i, blocks in enumerate(partitions) if {frozenset(b) for b in blocks} == target]
    if len(hits) != 1:
        raise ValueError('Theory partition must appear exactly once')
    return hits[0]
