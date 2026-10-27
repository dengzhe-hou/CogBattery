"""CPU-only minimum-residual factor extraction and oblimin rotation.

The extraction objective is the sum of squared *off-diagonal* residuals,
not PCA and not a likelihood objective.  Its bounded uniqueness
parameterization follows psych::fa's minres/OLS spectral construction:
https://github.com/cran/psych/blob/master/R/fa.R (fit.residuals, FAout.wls).
Unlike psych's approximate minres gradient, the derivative below is the
analytic derivative of that actual off-diagonal spectral objective.

Rotation uses statsmodels' gradient-projection implementation and retains
the full optimization trace.  No smoothing or correlation repair is used.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
    GPA,
    oblimin_objective,
)


UNIQUENESS_BOUNDS = (0.005, 1.0)
EXTRACTION_OPTIONS = {"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7}
ROTATION_MAX_ITER = 1001
ROTATION_TOL = 1e-5


class ConstantColumnError(ValueError):
    """A requested correlation has one or more constant columns."""


def correlation_matrix(X: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlations, optionally using nonnegative row frequencies.

    Frequency weighting is exactly equivalent to duplicating each row by
    its integer weight.  The shared covariance divisor cancels in R.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite observations-by-variables matrix")
    if weights is None:
        weights = np.ones(X.shape[0])
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (X.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and have one value per row")
    if np.any(weights < 0) or np.count_nonzero(weights) < 2:
        raise ValueError("at least two rows must have positive weights")
    weights = weights / weights.sum()
    centered = X - np.sum(X * weights[:, None], axis=0)
    covariance = (centered * weights[:, None]).T @ centered
    sd = np.sqrt(np.diag(covariance))
    scale = np.max(np.abs(X[weights > 0]), axis=0)
    constants = sd <= np.finfo(float).eps * 8 * scale
    constants |= sd == 0
    if np.any(constants):
        raise ConstantColumnError(f"constant columns: {np.flatnonzero(constants).tolist()}")
    R = covariance / np.outer(sd, sd)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return R


def _spectral_model(psi: np.ndarray, R: np.ndarray, k: int):
    reduced = R - np.diag(psi)
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    selected = np.zeros(len(psi), dtype=bool)
    selected[-k:] = True
    kept = np.where(selected, np.maximum(eigenvalues, 0.0), 0.0)
    loadings = eigenvectors[:, -k:][:, ::-1] * np.sqrt(kept[-k:][::-1])
    return loadings, eigenvalues, eigenvectors, kept, selected


def _minres_objective_gradient(psi: np.ndarray, R: np.ndarray, k: int):
    """Return lower-triangle residual SS and its exact uniqueness gradient.

    For M=f(R-diag(psi)), the Fréchet derivative of the spectral map f
    uses divided differences in the eigensystem.  Multiplying it by the
    off-diagonal residual gives d objective / d reduced correlation.
    """
    loadings, values, vectors, kept, selected = _spectral_model(psi, R, k)
    residual = loadings @ loadings.T - R
    np.fill_diagonal(residual, 0.0)
    objective = float(np.sum(residual * residual) / 2)
    value_difference = values[:, None] - values[None, :]
    kept_difference = kept[:, None] - kept[None, :]
    divided = np.zeros_like(value_difference)
    distinct = np.abs(value_difference) > 1e-12
    np.divide(kept_difference, value_difference, out=divided, where=distinct)
    derivative = (selected & (values > 0)).astype(float)
    # Equal eigenvalues wholly inside/outside the retained subspace have
    # the corresponding derivative. A tie exactly at the truncation is
    # nondifferentiable; its neutral average is explicit, not a PCA switch.
    tied = ~distinct
    divided[tied] = ((derivative[:, None] + derivative[None, :]) / 2)[tied]
    transformed = vectors.T @ residual @ vectors
    derivative_matrix = vectors @ (divided * transformed) @ vectors.T
    return objective, -np.diag(derivative_matrix).copy()


def _canonicalize(pattern: np.ndarray, phi: np.ndarray):
    # This only identifies exchangeable columns; no substantive labels are
    # imposed and oblique squared-column sums are not variance partitions.
    order = np.argsort(-np.sum(pattern * pattern, axis=0), kind="stable")
    pattern = pattern[:, order]
    phi = phi[np.ix_(order, order)]
    signs = np.sign(pattern[np.argmax(np.abs(pattern), axis=0), np.arange(pattern.shape[1])])
    signs[signs == 0] = 1
    return pattern * signs, phi * np.outer(signs, signs), order, signs


def fit_minres(R: np.ndarray, k: int, rotate: bool = True) -> dict:
    """Fit a bounded MINRES solution and optionally oblimin (gamma=0).

    Nonconvergence and boundary/improper solutions are returned and flagged;
    they are never silently replaced by another number of factors. Invalid
    inputs fail with ValueError. Communalities and uniqueness are from the
    reconstructed common covariance, not simply the bounded optimizer psi.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or not np.all(np.isfinite(R)):
        raise ValueError("R must be a finite square correlation matrix")
    p = R.shape[0]
    if not 1 <= k < p or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer from 1 to p-1")
    if not np.allclose(R, R.T, atol=1e-10, rtol=0) or not np.allclose(np.diag(R), 1, atol=1e-10, rtol=0):
        raise ValueError("R must be symmetric with a unit diagonal")
    minimum_eigenvalue = float(np.linalg.eigvalsh(R)[0])
    if minimum_eigenvalue < -1e-8:
        raise ValueError("R is not positive semidefinite; no smoothing is performed")
    optimized = minimize(
        _minres_objective_gradient,
        np.full(p, 0.5),
        args=(R, int(k)),
        jac=True,
        method="L-BFGS-B",
        bounds=[UNIQUENESS_BOUNDS] * p,
        options=EXTRACTION_OPTIONS,
    )
    unrotated, eigenvalues, _, _, _ = _spectral_model(optimized.x, R, k)
    pattern = unrotated.copy()
    phi = np.eye(k)
    rotation_trace = []
    rotation_converged = True
    rotation_gradient_norm = 0.0
    if rotate and k > 1:
        pattern, phi, _, rotation_trace = GPA(
            unrotated,
            vgQ=lambda L: oblimin_objective(L=L, gamma=0, rotation_method="oblique"),
            rotation_method="oblique",
            max_tries=ROTATION_MAX_ITER,
            tol=ROTATION_TOL,
        )
        rotation_gradient_norm = float(10 ** rotation_trace[-1][2])
        rotation_converged = rotation_gradient_norm < ROTATION_TOL
    pattern, phi, factor_order, factor_signs = _canonicalize(pattern, phi)
    structure = pattern @ phi
    common = pattern @ phi @ pattern.T
    communalities = np.diag(common).copy()
    uniqueness = 1 - communalities
    residual = R - common
    off_diagonal = residual[np.tril_indices(p, -1)]
    lower, upper = UNIQUENESS_BOUNDS
    lower_boundary = np.flatnonzero(optimized.x <= lower + 1e-6)
    upper_boundary = np.flatnonzero(optimized.x >= upper - 1e-6)
    projected_gradient = optimized.jac.copy()
    projected_gradient[(optimized.x <= lower + 1e-6) & (projected_gradient > 0)] = 0
    projected_gradient[(optimized.x >= upper - 1e-6) & (projected_gradient < 0)] = 0
    # Properness is separate from convergence and includes Heywood cases.
    proper = bool(
        np.all(np.isfinite(pattern))
        and np.all(uniqueness >= -1e-7)
        and np.all(communalities >= -1e-7)
        and np.linalg.eigvalsh(phi)[0] > 1e-8
    )
    return {
        "pattern": pattern,
        "phi": phi,
        "structure": structure,
        "unrotated": unrotated,
        "communalities": communalities,
        "uniqueness": uniqueness,
        "residual": residual,
        "rmsr": float(np.sqrt(np.mean(off_diagonal * off_diagonal))),
        "converged": bool(optimized.success and rotation_converged),
        "proper": proper,
        "boundary": bool(len(lower_boundary) or len(upper_boundary)),
        "diagnostics": {
            "optimizer": "L-BFGS-B",
            "objective": float(optimized.fun),
            "objective_definition": "sum of squared lower-triangle correlation residuals",
            "initial_uniqueness": 0.5,
            "uniqueness_bounds": list(UNIQUENESS_BOUNDS),
            "optimizer_options": dict(EXTRACTION_OPTIONS),
            "fitted_bounded_uniqueness": optimized.x,
            "optimizer_success": bool(optimized.success),
            "optimizer_message": str(optimized.message),
            "optimizer_iterations": int(optimized.nit),
            "optimizer_evaluations": int(optimized.nfev),
            "projected_gradient_max": float(np.max(np.abs(projected_gradient))),
            "lower_boundary_variables": lower_boundary,
            "upper_boundary_variables": upper_boundary,
            "heywood_variables": np.flatnonzero(uniqueness < -1e-7),
            "minimum_input_eigenvalue": minimum_eigenvalue,
            "retained_reduced_eigenvalues": eigenvalues[-k:][::-1],
            "rotation": "oblimin_gamma_0" if rotate and k > 1 else "none",
            "rotation_max_iter": ROTATION_MAX_ITER,
            "rotation_tolerance": ROTATION_TOL,
            "rotation_converged": bool(rotation_converged),
            "rotation_gradient_norm": rotation_gradient_norm,
            "rotation_trace": rotation_trace,
            "canonical_factor_order": factor_order,
            "canonical_factor_signs": factor_signs,
            "rotation_reconstruction_max_error": float(np.max(np.abs(common - unrotated @ unrotated.T))),
        },
    }
