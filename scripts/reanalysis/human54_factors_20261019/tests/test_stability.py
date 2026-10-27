"""Synthetic-only bootstrap and factor-identity checks; no formal inputs."""
import copy
import json

import numpy as np
import pytest

from scripts.reanalysis.human54_factors_20261019 import stability
from scripts.reanalysis.human54_factors_20261019.core import correlation_matrix, fit_minres


def _reference(p=4, k=1):
    pattern = np.full((p, k), .5)
    phi = np.eye(k)
    common = pattern @ phi @ pattern.T
    return {
        "pattern": pattern, "phi": phi, "structure": pattern @ phi,
        "communalities": np.diag(common), "uniqueness": 1 - np.diag(common),
        "rmsr": .1, "converged": True, "proper": True, "boundary": False,
        "diagnostics": {
            "optimizer_success": True, "rotation_converged": True,
            "lower_boundary_variables": np.array([], dtype=int),
            "upper_boundary_variables": np.array([], dtype=int),
            "heywood_variables": np.array([], dtype=int),
        },
    }


def test_signed_permutation_recovers_factor_identity_and_common_covariance():
    reference = np.array([[.8, .1, 0], [.7, 0, .1], [0, .8, .1],
                          [.1, .7, 0], [0, .1, .8], [.1, 0, .7]])
    phi = np.array([[1, .3, -.2], [.3, 1, .1], [-.2, .1, 1]])
    permutation, signs = [2, 0, 1], np.array([-1, 1, -1])
    candidate = reference[:, permutation] * signs
    candidate_phi = phi[np.ix_(permutation, permutation)] * np.outer(signs, signs)
    matched = stability.align_signed_permutation(candidate, candidate_phi, reference)
    np.testing.assert_allclose(matched["pattern"], reference)
    np.testing.assert_allclose(matched["phi"], phi)
    np.testing.assert_allclose(matched["structure"], reference @ phi)
    np.testing.assert_allclose(matched["congruence"], 1)
    np.testing.assert_allclose(candidate @ candidate_phi @ candidate.T,
                               matched["pattern"] @ matched["phi"] @ matched["pattern"].T)


def test_alignment_does_not_apply_continuous_procrustes_rotation():
    reference = np.array([[1., 0], [1., 0], [0, 1.], [0, 1.]])
    angle = .3
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    candidate = reference @ rotation
    matched = stability.align_signed_permutation(candidate, np.eye(2), reference)
    assert not np.allclose(matched["pattern"], reference)
    np.testing.assert_allclose(matched["congruence"], np.cos(angle))
    np.testing.assert_allclose(np.sort(np.abs(matched["pattern"]), axis=1),
                               np.sort(np.abs(candidate), axis=1))


def test_family_multiplicity_retains_members_and_matches_expanded_rows(monkeypatch):
    rng = np.random.default_rng(62)
    X = rng.normal(size=(8, 4))
    families = ["a", "a", "b", "b", "b", "c", "c", "c"]
    captured = []

    def fitting(R, k, rotate=True):
        captured.append(R.copy())
        return _reference(k=k)

    monkeypatch.setattr(stability, "fit_minres", fitting)
    result, arrays = stability.bootstrap_factors(
        X, families, {1: _reference(), 2: _reference(k=2)}, draws=30, seed=5)
    assert result["same_family_draws_for_all_dimensions"]
    weights = arrays["family_counts"][:, arrays["family_codes"]]
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    np.testing.assert_array_equal(weights[:, 2], weights[:, 4])
    assert (weights > 1).any()
    assert (arrays["family_counts"].sum(axis=1) == 3).all()
    np.testing.assert_array_equal(arrays["sample_sizes"], weights.sum(axis=1))
    assert len(captured) == 60
    for draw in range(30):
        expanded = np.repeat(X, weights[draw], axis=0)
        direct = np.corrcoef(expanded, rowvar=False)
        np.testing.assert_allclose(captured[2 * draw], direct, atol=1e-14)
        np.testing.assert_allclose(captured[2 * draw + 1], direct, atol=1e-14)


def test_constant_bootstrap_samples_remain_visible_in_denominator(monkeypatch):
    # Sampling one family twice makes columns 0 and 2 constant, even though
    # it retains two genuinely different rows and repeats both of them.
    X = np.array([[0, 0, 0, 0], [0, 1, 0, 1],
                  [1, 0, 1, 0], [1, 1, 1, 1]], dtype=float)
    monkeypatch.setattr(stability, "fit_minres", lambda *args, **kwargs: _reference())
    summary, arrays = stability.bootstrap_factors(
        X, ["a", "a", "b", "b"], {1: _reference()}, draws=100, seed=6)
    entry = summary["dimensions"]["1"]
    failed = (arrays["family_counts"] == 0).any(axis=1)
    assert 0 < failed.sum() < 100
    assert entry["counts"]["failed"] == failed.sum()
    assert entry["counts"]["converged_finite"] + entry["counts"]["failed"] == 100
    assert entry["summaries"]["converged_finite"]["fraction_of_all_draws"] == pytest.approx((~failed).mean())
    assert np.isnan(arrays["k1_pattern"][failed]).all()
    assert all(reason.startswith("constant_column:") for reason in arrays["k1_failure_reason"][failed])
    json.dumps(summary, allow_nan=False)


def test_single_distinct_row_draw_is_reported_without_abort(monkeypatch):
    X = np.array([[0, 0, 1, 0], [1, 2, 0, 1]], dtype=float)
    monkeypatch.setattr(stability, "fit_minres", lambda *args, **kwargs: _reference())
    summary, arrays = stability.bootstrap_factors(
        X, ["a", "b"], {1: _reference()}, draws=30, seed=2)
    assert summary["dimensions"]["1"]["failure_reasons"]["fewer_than_two_distinct_rows"] > 0
    assert len(arrays["k1_failure_reason"]) == 30


def test_boundary_improper_and_nonconverged_fits_are_not_silently_filtered(monkeypatch):
    fits = []
    for scale in [1., 2., 3., 4.]:
        fit = _reference()
        fit["pattern"] *= scale
        fit["structure"] = fit["pattern"].copy()
        fits.append(fit)
    fits[1]["boundary"] = True
    fits[1]["diagnostics"]["lower_boundary_variables"] = np.array([0])
    fits[2]["proper"] = False
    fits[2]["diagnostics"]["heywood_variables"] = np.array([1])
    fits[3]["converged"] = False
    fits[3]["diagnostics"]["rotation_converged"] = False
    iterator = iter(fits)
    monkeypatch.setattr(stability, "fit_minres", lambda *args, **kwargs: next(iterator))
    # An improper point reference is also retained and explicitly labeled.
    reference = copy.deepcopy(_reference())
    reference["proper"] = False
    reference["boundary"] = True
    X = np.random.default_rng(10).normal(size=(20, 4))
    summary, arrays = stability.bootstrap_factors(
        X, ["a"] * len(X), {1: reference}, draws=4)
    entry = summary["dimensions"]["1"]
    assert entry["reference"]["proper"] is False
    assert entry["reference"]["boundary"] is True
    assert entry["counts"]["finite"] == 4
    assert entry["counts"]["converged_finite"] == 3
    assert entry["counts"]["proper_nonboundary"] == 1
    assert entry["counts"]["finite_boundary"] == 1
    assert entry["counts"]["finite_improper"] == 1
    assert entry["counts"]["rotation_not_converged"] == 1
    assert entry["counts"]["failed"] == 1
    assert entry["failure_reasons"] == {"not_converged": 1}
    assert entry["diagnostic_counts_by_variable"]["heywood_variables"] == [0, 1, 0, 0]
    assert entry["summaries"]["converged_finite"]["intervals"]["pattern"]["upper"][0][0] > 1
    assert entry["summaries"]["proper_nonboundary"]["intervals"]["pattern"]["upper"][0][0] == .5
    assert arrays["k1_pattern"][3, 0, 0] == 2
    json.dumps(summary, allow_nan=False)


def test_correlated_two_factor_synthetic_recovery_and_bootstrap():
    rng = np.random.default_rng(20261019)
    pattern = np.zeros((10, 2))
    pattern[:5, 0] = [.8, .75, .7, .75, .8]
    pattern[5:, 1] = [.8, .75, .7, .75, .8]
    phi = np.array([[1., .35], [.35, 1.]])
    factors = rng.multivariate_normal([0, 0], phi, size=600)
    noise = rng.normal(size=(600, 10)) * np.sqrt(1 - np.diag(pattern @ phi @ pattern.T))
    X = factors @ pattern.T + noise
    fit = fit_minres(correlation_matrix(X), 2)
    matched = stability.align_signed_permutation(fit["pattern"], fit["phi"], pattern)
    assert np.min(matched["congruence"]) > .99
    assert matched["phi"][0, 1] == pytest.approx(.35, abs=.1)
    assert fit["proper"] and fit["converged"] and not fit["boundary"]
    summary, arrays = stability.bootstrap_factors(
        X, [str(i // 30) for i in range(600)], {2: fit}, draws=20, seed=30)
    entry = summary["dimensions"]["2"]
    assert entry["counts"]["converged_finite"] == 20
    assert entry["counts"]["proper_nonboundary"] == 20
    assert np.min(entry["summaries"]["converged_finite"]["intervals"]["congruence"]["lower"]) > .98
    assert np.max(np.abs(np.median(arrays["k2_pattern"], axis=0) - fit["pattern"])) < .06
    json.dumps(summary, allow_nan=False)
