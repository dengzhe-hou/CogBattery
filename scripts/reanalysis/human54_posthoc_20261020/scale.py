"""Scale analyses (spec section A): scores vs log parameters, scale-removed structure, F2 vs bias."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from scripts.reanalysis.human54_factors_20261019.parallel import permute_columns, retained_dimensions
from scripts.reanalysis.human54_fullbattery_20261019.pca import family_bootstrap_counts


def _interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(lower=float(np.quantile(finite, .025)), upper=float(np.quantile(finite, .975)),
                n_valid=int(len(finite)), n_undefined=int(len(values) - len(finite)))


def _corrs(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return pearsonr(x, y)[0], spearmanr(x, y)[0]


def correlation(x, y, families, *, draws, seed):
    """Pearson and Spearman with whole-family bootstrap; scores held fixed."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    _, codes, counts = family_bootstrap_counts(families, draws, seed)
    boot = np.array([_corrs(np.repeat(x, c[codes]), np.repeat(y, c[codes])) for c in counts])
    pearson, spearman = _corrs(x, y)
    return dict(n=int(len(x)), pearson=float(pearson), spearman=float(spearman),
                pearson_ci=_interval(boot[:, 0]), spearman_ci=_interval(boot[:, 1]),
                bootstrap=dict(draws=draws, seed=int(seed), unit='whole family', scores='held fixed'))


def within_family(x, y, families):
    """Correlation after removing family means; families with at least two models."""
    x, y, families = np.asarray(x, float), np.asarray(y, float), np.asarray(families)
    keep = np.array([np.sum(families == f) >= 2 for f in families])
    xs, ys, fs = x[keep], y[keep], families[keep]
    for f in np.unique(fs):
        xs[fs == f] -= xs[fs == f].mean()
        ys[fs == f] -= ys[fs == f].mean()
    return dict(n_models=int(keep.sum()), n_families=int(len(np.unique(fs))),
                pearson=float(pearsonr(xs, ys)[0]), spearman=float(spearmanr(xs, ys)[0]))


def _standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)


def residualize(X, covariate):
    """OLS residuals of each z-scored column on [1, covariate]."""
    Z = _standardize(np.asarray(X, float))
    design = np.column_stack([np.ones(len(Z)), covariate])
    coef, *_ = np.linalg.lstsq(design, Z, rcond=None)
    return Z - design @ coef, coef[1]


def _shares(E):
    Z = _standardize(E)
    R = Z.T @ Z / len(Z)
    values = np.clip(np.linalg.eigvalsh((R + R.T) / 2)[::-1], 0, None)
    return values, values / values.sum()


def scale_removed_structure(X, log_params, families, *, draws, seed, pa_draws, pa_seed):
    X, log_params = np.asarray(X, float), np.asarray(log_params, float)
    out = {}
    for label, matrix in (('original', _standardize(X)), ('scale_removed', residualize(X, log_params)[0])):
        eigenvalues, shares = _shares(matrix)
        rng = np.random.default_rng(pa_seed)
        null = np.array([_shares(permute_columns(matrix, rng))[0] for _ in range(pa_draws)])
        threshold = np.quantile(null, .95, axis=0)
        retained, crossings = retained_dimensions(eigenvalues, threshold)
        out[label] = dict(eigenvalues=eigenvalues.tolist(), variance_shares=shares.tolist(),
                          pa_threshold95=threshold.tolist(), pa_retained=int(retained),
                          pa_crossings=crossings.tolist(), pa_draws=pa_draws, pa_seed=int(pa_seed))
    _, slopes = residualize(X, log_params)
    out['task_slope_on_log10_params_z'] = slopes.tolist()
    _, codes, counts = family_bootstrap_counts(families, draws, seed)
    boot = np.full((draws, 2), np.nan)
    for i, c in enumerate(counts):
        idx = np.repeat(np.arange(len(X)), c[codes])
        try:
            Xi, li = X[idx], log_params[idx]
            if np.any(Xi.std(axis=0) == 0) or li.std() == 0:
                continue
            boot[i, 0] = _shares(_standardize(Xi))[1][0]
            boot[i, 1] = _shares(residualize(Xi, li)[0])[1][0]
        except np.linalg.LinAlgError:
            continue
    out['pc1_share_ci'] = dict(original=_interval(boot[:, 0]), scale_removed=_interval(boot[:, 1]),
                               draws=draws, seed=int(seed), unit='whole family; restandardize and refit each draw')
    return out
