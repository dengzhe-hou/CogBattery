"""Numerical and generative checks performed before using formal matrices."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.optimize._numdiff import approx_derivative

from scripts.reanalysis.human54_factors_20261019 import core


def model_correlation(pattern, phi):
    R = pattern @ phi @ pattern.T
    np.fill_diagonal(R, 1.0)
    return R


def align_to_truth(pattern, phi, truth):
    numerator = truth.T @ pattern
    similarity = numerator / np.outer(np.linalg.norm(truth, axis=0), np.linalg.norm(pattern, axis=0))
    row, col = linear_sum_assignment(-np.abs(similarity))
    assert np.array_equal(row, np.arange(truth.shape[1]))
    signs = np.sign(similarity[row, col])
    return pattern[:, col] * signs, phi[np.ix_(col, col)] * np.outer(signs, signs)


def three_factor_model():
    pattern = np.zeros((9, 3))
    pattern[:3, 0] = [.7, .8, .65]
    pattern[3:6, 1] = [.75, .85, .7]
    pattern[6:, 2] = [.65, .75, .8]
    phi = np.array([[1, .3, .2], [.3, 1, .4], [.2, .4, 1]])
    return pattern, phi


def test_weighted_correlation_matches_literal_row_replication():
    X = np.random.default_rng(71).normal(size=(8, 5))
    counts = np.array([0, 3, 1, 2, 1, 0, 4, 1])
    repeated = np.repeat(X, counts, axis=0)
    np.testing.assert_allclose(core.correlation_matrix(X, counts), np.corrcoef(repeated, rowvar=False), atol=1e-14)


def test_column_scaling_and_location_do_not_change_correlations():
    X = np.random.default_rng(45).normal(size=(40, 4))
    np.testing.assert_allclose(core.correlation_matrix(X * [.001, 2, 3, 100] + [1, 2, 3, 4]), core.correlation_matrix(X), atol=2e-14)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_exact_objective_gradient_against_central_finite_differences(k):
    R = np.corrcoef(np.random.default_rng(23).normal(size=(100, 13)), rowvar=False)
    psi = np.linspace(.2, .8, 13)
    _, gradient = core._minres_objective_gradient(psi, R, k)
    numeric = approx_derivative(lambda x: core._minres_objective_gradient(x, R, k)[0], psi, method="3-point").ravel()
    np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-6)


def test_known_one_factor_recovers_common_variance_and_unique_variance():
    pattern = np.array([.8, .7, .6, .5, .75, .65])[:, None]
    R = model_correlation(pattern, np.eye(1))
    result = core.fit_minres(R, 1)
    np.testing.assert_allclose(result["pattern"], pattern, atol=2e-5)
    np.testing.assert_allclose(result["communalities"], pattern[:, 0] ** 2, atol=2e-5)
    np.testing.assert_allclose(result["uniqueness"], 1 - pattern[:, 0] ** 2, atol=2e-5)
    assert result["rmsr"] < 1e-6
    assert result["converged"] and result["proper"] and not result["boundary"]
    # PCA would allocate total variance; this solution leaves genuine uniqueness.
    assert result["communalities"].sum() < np.linalg.eigvalsh(R)[-1]


def test_correlated_three_factor_loadings_and_factor_correlations_recover():
    truth, truth_phi = three_factor_model()
    result = core.fit_minres(model_correlation(truth, truth_phi), 3)
    aligned, aligned_phi = align_to_truth(result["pattern"], result["phi"], truth)
    np.testing.assert_allclose(aligned, truth, atol=3e-5)
    np.testing.assert_allclose(aligned_phi, truth_phi, atol=3e-5)
    assert result["rmsr"] < 1e-6
    assert result["converged"] and result["proper"]
    assert result["diagnostics"]["rotation_trace"]
    assert result["diagnostics"]["rotation_gradient_norm"] < core.ROTATION_TOL


def test_sampling_noise_still_recovers_the_two_correlated_factors():
    rng = np.random.default_rng(818)
    truth = np.zeros((8, 2))
    truth[:4, 0] = [.8, .75, .7, .65]
    truth[4:, 1] = [.75, .7, .8, .65]
    phi = np.array([[1, .35], [.35, 1]])
    X = rng.multivariate_normal(np.zeros(8), model_correlation(truth, phi), size=6000)
    result = core.fit_minres(core.correlation_matrix(X), 2)
    aligned, aligned_phi = align_to_truth(result["pattern"], result["phi"], truth)
    np.testing.assert_allclose(aligned, truth, atol=.035)
    np.testing.assert_allclose(aligned_phi, phi, atol=.04)
    assert result["converged"] and result["proper"]


def test_rotation_preserves_common_covariance_and_structure_identity():
    truth, phi = three_factor_model()
    R = model_correlation(truth, phi)
    result = core.fit_minres(R, 3)
    np.testing.assert_allclose(result["pattern"] @ result["phi"] @ result["pattern"].T, result["unrotated"] @ result["unrotated"].T, atol=1e-12)
    np.testing.assert_allclose(result["structure"], result["pattern"] @ result["phi"], atol=1e-14)
    # Underfitting the known three-factor matrix cannot fit as well.
    assert core.fit_minres(R, 1)["rmsr"] > .1


def test_boundary_solution_is_reported_without_erasing_it():
    R = np.full((5, 5), .9999)
    np.fill_diagonal(R, 1)
    result = core.fit_minres(R, 1)
    assert result["boundary"] and result["converged"]
    assert len(result["diagnostics"]["lower_boundary_variables"]) == 5
    assert result["rmsr"] > 0


def test_actual_heywood_communality_is_reported_despite_bounded_optimizer():
    R = model_correlation(np.array([1.1, .5, .4])[:, None], np.eye(1))
    assert np.linalg.eigvalsh(R)[0] > 0
    result = core.fit_minres(R, 1)
    assert result["converged"] and result["boundary"] and not result["proper"]
    assert result["communalities"][0] > 1
    assert result["uniqueness"][0] < 0
    np.testing.assert_array_equal(result["diagnostics"]["heywood_variables"], [0])


def test_extraction_nonconvergence_is_preserved(monkeypatch):
    monkeypatch.setattr(core, "EXTRACTION_OPTIONS", {"maxiter": 0, "ftol": 1e-10, "gtol": 1e-7})
    truth, phi = three_factor_model()
    result = core.fit_minres(model_correlation(truth, phi), 2, rotate=False)
    assert not result["converged"]
    assert not result["diagnostics"]["optimizer_success"]


def test_rotation_nonconvergence_is_preserved(monkeypatch):
    monkeypatch.setattr(core, "ROTATION_MAX_ITER", 1)
    truth, phi = three_factor_model()
    result = core.fit_minres(model_correlation(truth, phi), 3)
    assert not result["converged"]
    assert result["diagnostics"]["optimizer_success"]
    assert not result["diagnostics"]["rotation_converged"]


def test_constant_column_and_weighted_constant_column_fail():
    X = np.array([[1, 2], [1, 3], [2, 3]], dtype=float)
    with pytest.raises(core.ConstantColumnError):
        core.correlation_matrix(X, np.array([1, 1, 0]))
    with pytest.raises(core.ConstantColumnError):
        core.correlation_matrix(np.ones((10, 4)))


@pytest.mark.parametrize("R", [np.array([[1, 2], [2, 1]]), np.array([[1, .2], [.1, 1]]), np.array([[1, np.nan], [np.nan, 1]])])
def test_invalid_correlation_fails_without_smoothing(R):
    with pytest.raises(ValueError):
        core.fit_minres(R, 1)


def test_nonfinite_observations_fail_without_imputation():
    X = np.random.default_rng(1).normal(size=(10, 3))
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        core.correlation_matrix(X)
