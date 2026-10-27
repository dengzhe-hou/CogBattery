"""Synthetic PCA checks; no formal human54 inputs are read."""
import json

import numpy as np
import pytest

from scripts.reanalysis.human54_fullbattery_20261019.pca import (
    ConstantFeatureError, analyze_pca, family_bootstrap_counts,
)


def test_known_rank_one_population_standardization():
    x = np.arange(-3., 4.)
    X = np.column_stack((x, 3 * x + 2, .2 * x - 7))
    result, draws = analyze_pca(X, ["one"] * len(x), ["a", "b", "c"], draws=20)
    np.testing.assert_allclose(result["population_sd"], X.std(axis=0, ddof=0))
    np.testing.assert_allclose(result["eigenvalues"], [3, 0, 0], atol=1e-14)
    np.testing.assert_allclose(result["variance_shares"], [1, 0, 0], atol=1e-14)
    np.testing.assert_allclose(np.asarray(result["correlation_loadings"])[:, 0], 1)
    assert result["bootstrap"]["n_valid"] == 20
    np.testing.assert_allclose(draws["pc1_share"], 1)
    json.dumps(result, allow_nan=False)


def test_orthogonal_columns_have_known_equal_spectrum():
    # Hadamard columns are centered and mutually orthogonal.
    X = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    result, _ = analyze_pca(X, ["one"] * 4, ["a", "b", "c"], draws=0)
    np.testing.assert_allclose(result["correlation_matrix"], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(result["eigenvalues"], np.ones(3), atol=1e-15)
    np.testing.assert_allclose(result["variance_shares"], np.full(3, 1/3), atol=1e-15)


def test_eigenvectors_correlation_loadings_and_scores_are_distinct():
    rng = np.random.default_rng(4)
    a = rng.normal(size=80)
    X = np.column_stack((a, .7 * a + rng.normal(size=80) * .3, rng.normal(size=80)))
    result, _ = analyze_pca(X, ["one"] * len(X), ["DS", "NB", "ST"], draws=0)
    vectors = np.asarray(result["eigenvectors"])
    loadings = np.asarray(result["correlation_loadings"])
    scores = np.asarray(result["scores"])
    eigenvalues = np.asarray(result["eigenvalues"])
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(loadings, vectors * np.sqrt(eigenvalues), atol=1e-14)
    np.testing.assert_allclose(np.corrcoef(X.T, scores.T)[:3, 3:], loadings, atol=1e-14)
    np.testing.assert_allclose(scores.var(axis=0, ddof=0), eigenvalues, atol=1e-14)
    assert not np.allclose(vectors, loadings)
    assert loadings[:, 0].sum() > 0


def test_family_draws_keep_all_members_and_repeat_family():
    names, codes, counts = family_bootstrap_counts(["b", "a", "a", "c"], draws=300, seed=3)
    assert names == ["a", "b", "c"]
    weights = counts[:, codes]
    np.testing.assert_array_equal(weights[:, 1], weights[:, 2])
    assert (counts.sum(axis=1) == 3).all()
    assert (weights[:, 1] > 1).any()
    # Direct expansion and multiplicity weighting yield identical bootstrap PCA.
    X = np.array([[0, 2, -1], [1, -1, 2], [3, 1, 0], [5, 0, 3]], float)
    result, draws = analyze_pca(X, ["b", "a", "a", "c"], ["a", "b", "c"], draws=300, seed=3)
    np.testing.assert_array_equal(draws["family_counts"], counts)
    for draw in [0, 1, 17, 99]:
        expanded = np.repeat(X, weights[draw], axis=0)
        if (expanded.std(axis=0) == 0).any():
            assert np.isnan(draws["pc1_share"][draw])
        else:
            direct, _ = analyze_pca(expanded, ["one"] * len(expanded), ["a", "b", "c"], draws=0)
            assert draws["pc1_share"][draw] == pytest.approx(direct["variance_shares"][0])
    assert result["models"] == 4


def test_translation_and_positive_rescaling_invariance():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 5))
    families = [str(i // 3) for i in range(len(X))]
    tasks = ["DS", "NB", "ST", "FL", "CVLT"]
    first, first_draws = analyze_pca(X, families, tasks, draws=80)
    transformed, transformed_draws = analyze_pca(X * [2, .3, 4, 1, 20] + [7, -2, 5, 30, 1],
                                                families, tasks, draws=80)
    for key in ["correlation_matrix", "variance_shares", "correlation_loadings", "scores"]:
        np.testing.assert_allclose(first[key], transformed[key], atol=1e-12)
    np.testing.assert_allclose(first_draws["pc1_share"], transformed_draws["pc1_share"], atol=1e-12)


def test_constant_requested_task_errors_and_constant_bootstrap_is_counted():
    with pytest.raises(ConstantFeatureError, match="columns: \\[1\\]"):
        analyze_pca([[0, 1], [1, 1]], ["a", "b"], ["DS", "ST"], draws=10)
    result, draws = analyze_pca([[0, 0], [1, 2]], ["a", "b"], ["DS", "ST"], draws=100, seed=4)
    assert result["bootstrap"]["n_invalid"] > 0
    assert result["bootstrap"]["n_valid"] + result["bootstrap"]["n_invalid"] == 100
    assert result["bootstrap"]["n_invalid"] == np.isnan(draws["pc1_share"]).sum()


def test_five_group_delta_uses_each_task_pair_once():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(60, 13))
    tasks = ["DS", "NB", "OS", "ST", "FL", "GN", "CVLT", "DRM", "SM", "FB", "EP", "CAL", "WAG"]
    result, _ = analyze_pca(X, ["one"] * len(X), tasks, draws=10)
    corr = np.corrcoef(X.T)
    groups = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    within, between = [], []
    for i in range(13):
        for j in range(i + 1, 13):
            (within if groups[i] == groups[j] else between).append(corr[i, j])
    assert len(within) == 11 and len(between) == 67
    assert result["structure"]["group_delta"] == pytest.approx(np.mean(within) - np.mean(between))
    assert result["structure"]["mean_pair_correlation"] == pytest.approx(corr[np.triu_indices(13, 1)].mean())


def test_incomplete_matrix_is_not_imputed():
    with pytest.raises(ValueError, match="complete finite matrix"):
        analyze_pca([[0, 1], [1, np.nan]], ["a", "b"], ["DS", "ST"], draws=0)
