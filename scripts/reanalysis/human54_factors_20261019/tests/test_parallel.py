"""Parallel analysis synthetic checks; no human54 results or calls are read."""
import json

import numpy as np
import pytest
from scipy.linalg import hadamard

from scripts.reanalysis.human54_factors_20261019 import parallel as pa


def test_permutation_preserves_tied_and_continuous_marginals():
    X = np.column_stack((np.repeat([0., .5, 1.], 8), np.arange(24.), np.tile([1., 2.], 12)))
    permuted = pa.permute_columns(X, np.random.default_rng(717))
    np.testing.assert_array_equal(np.sort(permuted, axis=0), np.sort(X, axis=0))
    assert not np.array_equal(permuted[:, 1], X[:, 1])
    # Permutations are column-specific, not a single shared row permutation.
    source_order = permuted[:, 1].astype(int)
    assert not np.array_equal(permuted[:, 0], X[source_order, 0])


def test_retention_is_leading_positive_run_not_all_crossings():
    count, crossing = pa.retained_dimensions([4., 2., .5, .4, -.1], [3., 1., .6, .3, -.2])
    assert count == 2
    assert crossing.tolist() == [True, True, False, True, True]
    count, crossing = pa.retained_dimensions([0., -.1], [-.2, -.2])
    assert count == 0
    assert crossing.tolist() == [True, True]
    assert pa.retained_dimensions([2., 1.], [2., .8])[0] == 0


def test_zero_factor_known_identity_correlation():
    # Non-intercept Hadamard columns give exactly orthogonal, centered data.
    X = hadamard(256)[:, 1:14].astype(float)
    result, arrays = pa.parallel_analysis(X, draws=40, seed=53)
    np.testing.assert_allclose(arrays["observed_correlation"], np.eye(13), atol=1e-14)
    np.testing.assert_allclose(result["pc"]["eigenvalues"], np.ones(13), atol=1e-14)
    assert result["pc"]["retained"] == 0
    assert result["fa"]["retained"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("factors,seed", [(1, 310), (3, 352)])
def test_recovers_clear_one_and_three_factor_synthetic_structure(factors, seed):
    rng = np.random.default_rng(seed)
    n, p = 600, 12
    latent = rng.normal(size=(n, factors))
    assignments = np.repeat(np.arange(factors), p // factors)
    X = .9 * latent[:, assignments] + np.sqrt(1 - .9 ** 2) * rng.normal(size=(n, p))
    result, arrays = pa.parallel_analysis(X, draws=50, seed=577)
    assert result["pc"]["retained"] == factors
    assert result["fa"]["retained"] == factors
    assert arrays["null_pc_eigenvalues"].shape == (50, p)
    assert arrays["null_fa_eigenvalues"].shape == (50, p)
    assert result["draws"] == 50


def test_reference_line_uses_rankwise_95_percentile_and_same_seed():
    rng = np.random.default_rng(171)
    X = rng.normal(size=(150, 5))
    progress = []
    first, arrays = pa.parallel_analysis(X, draws=12, seed=44, progress=lambda n, total: progress.append((n, total)))
    second, second_arrays = pa.parallel_analysis(X, draws=12, seed=44)
    for key in ["null_pc_eigenvalues", "null_fa_eigenvalues"]:
        np.testing.assert_array_equal(arrays[key], second_arrays[key])
    np.testing.assert_allclose(first["pc"]["threshold95"], np.quantile(arrays["null_pc_eigenvalues"], .95, axis=0))
    valid = arrays["null_fa_valid_spectrum"]
    np.testing.assert_allclose(first["fa"]["threshold95"], np.quantile(arrays["null_fa_eigenvalues"][valid], .95, axis=0))
    assert progress == [(12, 12)]
    assert first == second


def test_factor_spectrum_replaces_diagonal_and_preserves_negative_eigenvalues():
    rng = np.random.default_rng(230)
    g = rng.normal(size=180)
    X = .75 * g[:, None] + rng.normal(size=(180, 6)) * np.sqrt(1 - .75 ** 2)
    R = pa.correlation_matrix(X)
    fit = pa.fit_minres(R, 1, rotate=False)
    reduced = R.copy()
    np.fill_diagonal(reduced, fit["communalities"])
    spectrum, status = pa._factor_spectrum(R)
    np.testing.assert_allclose(spectrum, np.linalg.eigvalsh(reduced)[::-1])
    assert status["valid_spectrum"]
    assert spectrum[-1] < 0
    assert spectrum.sum() == pytest.approx(np.sum(fit["communalities"]))
    assert not np.allclose(spectrum, np.linalg.eigvalsh(R)[::-1])


def test_failed_null_fit_remains_missing_without_replacement(monkeypatch):
    original = pa.fit_minres
    calls = 0

    def injected_failure(R, k, rotate=False):
        nonlocal calls
        calls += 1
        fit = original(R, k, rotate=rotate)
        if calls == 3:  # Observed fit then the second null fit.
            fit["converged"] = False
        return fit

    monkeypatch.setattr(pa, "fit_minres", injected_failure)
    result, arrays = pa.parallel_analysis(np.random.default_rng(108).normal(size=(120, 5)), draws=8)
    assert calls == 9
    assert result["n_failed_fa_null"] == 1
    assert result["n_valid_fa_null"] == 7
    assert np.isnan(arrays["null_fa_eigenvalues"][1]).all()
    assert result["fa"]["threshold_conditional_on_valid_null_fits"]
    assert result["status"] == "partial"
    assert result["pc"]["status"] == "complete"
    json.dumps(result, allow_nan=False)


def test_failed_point_fa_leaves_pc_available(monkeypatch):
    original = pa.fit_minres
    calls = 0

    def injected_failure(R, k, rotate=False):
        nonlocal calls
        calls += 1
        fit = original(R, k, rotate=rotate)
        if calls == 1:
            fit["converged"] = False
        return fit

    monkeypatch.setattr(pa, "fit_minres", injected_failure)
    result, arrays = pa.parallel_analysis(np.random.default_rng(116).normal(size=(100, 5)), draws=8)
    assert result["fa"]["retained"] is None
    assert result["fa"]["eigenvalues"] is None
    assert result["pc"]["retained"] is not None
    assert np.isnan(arrays["observed_fa_eigenvalues"]).all()
    json.dumps(result, allow_nan=False)
