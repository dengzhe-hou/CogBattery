"""CPU-only PCA of the approved, one-endpoint-per-task human54 matrix.

The caller constructs complete-case or invalid-as-wrong matrices. This module
does not select endpoints, remove columns, impute observations, or read calls.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


TASK_GROUPS = {
    "DS": "WM", "NB": "WM", "OS": "WM",
    "ST": "CC", "FL": "CC", "GN": "CC",
    "CVLT": "EM", "DRM": "EM", "SM": "EM",
    "FB": "ToM", "EP": "ToM", "CAL": "Meta", "WAG": "Meta",
}
_SD_RELATIVE_TOLERANCE = 100 * np.finfo(float).eps


class ConstantFeatureError(ValueError):
    """At least one requested task cannot be standardized in this sample."""


def _standardize(X: np.ndarray, weights: np.ndarray):
    """Population-SD standardization with whole-model multiplicities."""
    n = weights.sum()
    mean = weights @ X / n
    centered = X - mean
    sd = np.sqrt(weights @ (centered * centered) / n)
    # A nearly constant column at machine precision cannot yield a stable
    # correlation. Do not silently drop a requested task in point or bootstrap.
    threshold = _SD_RELATIVE_TOLERANCE * np.maximum(1., np.max(np.abs(X), axis=0))
    bad = np.flatnonzero(sd <= threshold)
    if len(bad):
        raise ConstantFeatureError(f"Constant or nearly constant task columns: {bad.tolist()}")
    Z = centered / sd
    corr = (Z.T * weights) @ Z / n
    corr = (corr + corr.T) / 2
    return mean, sd, Z, corr


def _decompose(corr: np.ndarray):
    eigenvalues, vectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.)
    vectors = vectors[:, order]
    # This is covariance of standardized columns, so the total is task count.
    shares = eigenvalues / eigenvalues.sum()
    loadings = vectors * np.sqrt(eigenvalues)
    for pc in range(len(eigenvalues)):
        if pc == 0:
            direction = loadings[:, pc].sum()
            if abs(direction) <= 100 * np.finfo(float).eps:
                direction = loadings[np.argmax(np.abs(loadings[:, pc])), pc]
        else:
            direction = loadings[np.argmax(np.abs(loadings[:, pc])), pc]
        if direction < 0:
            vectors[:, pc] *= -1
            loadings[:, pc] *= -1
    return eigenvalues, shares, vectors, loadings


def _pair_masks(tasks: Sequence[str]):
    row, col = np.triu_indices(len(tasks), 1)
    if all(task in TASK_GROUPS for task in tasks):
        groups = np.array([TASK_GROUPS[task] for task in tasks])
        within = groups[row] == groups[col]
        if within.any() and (~within).any():
            return row, col, within
    return row, col, None


def _structure(corr, masks):
    row, col, within = masks
    pairs = corr[row, col]
    values = {"mean_pair_correlation": float(pairs.mean())}
    if within is not None:
        values.update(
            within_group_mean=float(pairs[within].mean()),
            between_group_mean=float(pairs[~within].mean()),
            group_delta=float(pairs[within].mean() - pairs[~within].mean()),
        )
    return values


def _interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "lower": float(np.quantile(finite, .025)) if len(finite) else None,
        "upper": float(np.quantile(finite, .975)) if len(finite) else None,
        "n_valid": int(len(finite)),
    }


def family_bootstrap_counts(families: Sequence[str], draws=5000, seed=20261018):
    """Return sorted names, row codes, and whole-family draw multiplicities.

    Multiplicity ``counts[b, codes[i]]`` applies to model i. Thus drawing the
    same family twice retains every member of that family twice.
    """
    labels = np.asarray(families, dtype=str)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Families must be a nonempty one-dimensional sequence")
    if not isinstance(draws, (int, np.integer)) or draws < 0:
        raise ValueError("draws must be a nonnegative integer")
    names, codes = np.unique(labels, return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(
        len(names), np.full(len(names), 1 / len(names)), size=draws
    )
    return names.tolist(), codes, counts


def analyze_pca(X: np.ndarray, families: Sequence[str], tasks: Sequence[str],
                seed=20261018, draws=5000):
    """Return ``(JSON-serializable summary, arrays_for_np_savez)``.

    Rows receive equal weight in the point estimate; columns are standardized
    by population SD. Eigenvectors are unit vectors, correlation loadings are
    eigenvectors times sqrt(eigenvalue), and scores are Z @ eigenvectors.
    Bootstrap draws resample whole families and restandardize on every draw.
    Constant or nearly constant bootstrap columns invalidate that draw; a
    constant point-estimate column raises ConstantFeatureError.
    """
    X = np.asarray(X, dtype=float)
    tasks = list(tasks)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("PCA requires at least two rows and two tasks")
    if len(families) != X.shape[0] or len(tasks) != X.shape[1]:
        raise ValueError("Matrix rows/families and columns/tasks must align")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Task columns must have unique names")
    if not np.isfinite(X).all():
        raise ValueError("PCA requires a complete finite matrix; no imputation is performed")
    names, family_codes, counts = family_bootstrap_counts(families, draws, seed)
    mean, sd, Z, corr = _standardize(X, np.ones(len(X)))
    eigenvalues, shares, vectors, loadings = _decompose(corr)
    scores = Z @ vectors
    masks = _pair_masks(tasks)
    structure = _structure(corr, masks)
    statistics = {"pc1_share": np.full(draws, np.nan)}
    statistics.update({key: np.full(draws, np.nan) for key in structure})
    for draw, family_counts in enumerate(counts):
        weights = family_counts[family_codes]
        try:
            _, _, _, boot_corr = _standardize(X, weights)
        except ConstantFeatureError:
            continue
        _, boot_shares, _, _ = _decompose(boot_corr)
        statistics["pc1_share"][draw] = boot_shares[0]
        for key, value in _structure(boot_corr, masks).items():
            statistics[key][draw] = value
    valid = np.isfinite(statistics["pc1_share"])
    result = {
        "models": int(len(X)), "tasks": tasks, "families": names,
        "n_families": len(names), "weighting": "equal model weight",
        "standardization": "column mean and population SD (ddof=0)",
        "means": mean.tolist(), "population_sd": sd.tolist(),
        "correlation_matrix": corr.tolist(),
        "eigenvalues": eigenvalues.tolist(), "variance_shares": shares.tolist(),
        "eigenvectors": vectors.tolist(),
        "correlation_loadings": loadings.tolist(), "scores": scores.tolist(),
        "pc1_sign": "sum of correlation loadings positive; largest absolute loading positive when sum is numerically zero",
        "structure": structure,
        "group_assignment": {task: TASK_GROUPS[task] for task in tasks if task in TASK_GROUPS},
        "bootstrap": {
            "unit": "whole model family, retain all members and repeat whole family",
            "restandardize_each_draw": True,
            "draws": int(draws), "seed": int(seed),
            "n_valid": int(valid.sum()), "n_invalid": int((~valid).sum()),
            "invalid_reason": "constant or nearly constant task column",
            "intervals": {key: _interval(value) for key, value in statistics.items()},
        },
    }
    arrays = {"family_counts": counts, "family_codes": family_codes}
    arrays.update(statistics)
    return result, arrays
