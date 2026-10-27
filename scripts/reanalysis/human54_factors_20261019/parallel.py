"""Permutation parallel analysis of task correlations, using CPU arrays only.

Columns are independently permuted without replacement, so bounded scores and
ties have exactly the same marginal distributions in every null draw. The PCA
and common-factor reference spectra use the same permutations. The latter uses
one-factor, unrotated MINRES communalities for both observed and null matrices
(the ``SMC=False`` convention), with no clipping of reduced-matrix eigenvalues.
"""
from __future__ import annotations

import numpy as np

from .core import correlation_matrix, fit_minres


def permute_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently reorder each column, preserving every observed value."""
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def retained_dimensions(observed: np.ndarray, threshold: np.ndarray) -> tuple[int, np.ndarray]:
    """Keep the leading run of positive eigenvalues above their rank threshold.

    All rankwise crossings are returned, including later crossings that cannot
    restart the leading run. Equality with the 95% threshold does not pass.
    """
    observed = np.asarray(observed, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    crossings = observed > threshold
    eligible = crossings & (observed > 0)
    failures = np.flatnonzero(~eligible)
    count = int(failures[0]) if len(failures) else len(observed)
    return count, crossings


def _factor_spectrum(R: np.ndarray) -> tuple[np.ndarray, dict]:
    fit = fit_minres(R, 1, rotate=False)
    communalities = np.asarray(fit["communalities"], dtype=float)
    finite = bool(np.isfinite(communalities).all())
    status = {
        "converged": bool(fit["converged"]),
        "proper": bool(fit["proper"]),
        "boundary": bool(fit["boundary"]),
        "finite_communalities": finite,
    }
    status["valid_spectrum"] = status["converged"] and finite
    if not status["valid_spectrum"]:
        return np.full(len(R), np.nan), status
    reduced = R.copy()
    np.fill_diagonal(reduced, communalities)
    return np.linalg.eigvalsh(reduced)[::-1], status


def _comparison(observed: np.ndarray, null: np.ndarray, point_valid: bool) -> dict:
    valid = np.isfinite(null).all(axis=1)
    n_valid = int(valid.sum())
    if n_valid:
        threshold = np.quantile(null[valid], .95, axis=0, method="linear")
        null_mean = null[valid].mean(axis=0)
    else:
        threshold = null_mean = np.full(null.shape[1], np.nan)
    if point_valid and n_valid:
        retained, crossings = retained_dimensions(observed, threshold)
    else:
        retained, crossings = None, None
    return {
        "eigenvalues": observed.tolist() if point_valid else None,
        "null_mean": null_mean.tolist() if n_valid else None,
        "threshold95": threshold.tolist() if n_valid else None,
        "exceeds": crossings.tolist() if crossings is not None else None,
        "retained": retained,
        "null_draws_valid": n_valid,
        "null_draws_invalid": int(len(null) - n_valid),
        "threshold_conditional_on_valid_null_fits": bool(n_valid != len(null)),
        "status": "complete" if point_valid and n_valid == len(null) else "partial",
    }


def parallel_analysis(X, draws=5000, seed=20261019, progress=None):
    """Return JSON-ready summaries and complete observed/null arrays.

    ``progress(completed, total)`` is called after every 250 null draws and at
    completion. A converged boundary solution retains its defined spectrum and
    is counted explicitly. Failed optimizations retain NaN spectra, never get
    replacement draws, and make the resulting reference line conditional on
    valid fits. Inputs are complete matrices supplied by the caller; this
    module does not select rows, impute scores, or read any experiment files.
    """
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    X = np.asarray(X, dtype=float)
    R = correlation_matrix(X)
    p = R.shape[0]
    observed_pc = np.linalg.eigvalsh(R)[::-1]
    observed_fa, point_status = _factor_spectrum(R)
    null_pc = np.full((draws, p), np.nan)
    null_fa = np.full((draws, p), np.nan)
    null_status = {
        key: np.zeros(draws, dtype=bool)
        for key in ("converged", "proper", "boundary", "finite_communalities", "valid_spectrum")
    }
    rng = np.random.default_rng(seed)
    for i in range(draws):
        null_R = correlation_matrix(permute_columns(X, rng))
        null_pc[i] = np.linalg.eigvalsh(null_R)[::-1]
        null_fa[i], status = _factor_spectrum(null_R)
        for key, value in status.items():
            null_status[key][i] = value
        if progress is not None and ((i + 1) % 250 == 0 or i + 1 == draws):
            progress(i + 1, draws)
    pc = _comparison(observed_pc, null_pc, True)
    fa = _comparison(observed_fa, null_fa, point_status["valid_spectrum"])
    summary = {
        "rows": int(len(X)),
        "tasks": int(p),
        "draws": int(draws),
        "seed": int(seed),
        "null_scheme": "independent within-column permutations without replacement",
        "reference_quantile": .95,
        "quantile_method": "linear",
        "retention_rule": "leading consecutive positive eigenvalues strictly above rankwise 95% null line",
        "common_factor_diagonal": "one-factor unrotated MINRES communalities (SMC=False), observed and every null matrix",
        "common_factor_negative_eigenvalues_clipped": False,
        "boundary_solutions_retained_if_converged": True,
        "failed_null_fits_replaced": False,
        "pc": pc,
        "fa": fa,
        "n_valid_fa_null": fa["null_draws_valid"],
        "n_failed_fa_null": fa["null_draws_invalid"],
        "observed_common_factor_fit": point_status,
        "null_common_factor_fit_counts": {key: int(values.sum()) for key, values in null_status.items()},
        "status": "complete" if pc["status"] == fa["status"] == "complete" else "partial",
    }
    arrays = {
        "observed_correlation": R,
        "observed_pc_eigenvalues": observed_pc,
        "observed_fa_eigenvalues": observed_fa,
        "null_pc_eigenvalues": null_pc,
        "null_fa_eigenvalues": null_fa,
        **{f"null_fa_{key}": value for key, value in null_status.items()},
    }
    return summary, arrays
