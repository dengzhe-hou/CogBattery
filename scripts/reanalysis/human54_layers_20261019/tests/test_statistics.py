import numpy as np
import pytest

from scripts.reanalysis.human54_layers_20261019.analysis_statistics import (
    cluster_mean, direction_counts, profile_metrics, profile_bootstrap, stratified_weights)


def test_model_equal_point_and_entire_family_sampling():
    result, draws = cluster_mean([0., 0., 1.], ['a', 'a', 'b'], draws=500)
    assert result['mean'] == pytest.approx(1/3)
    assert result['family_equal_mean'] == .5
    assert set(draws) == {0., 1/3, 1.}


def test_profile_minimum_cells_constant_and_missing():
    mae, r = profile_metrics([0, 1, 0, 1, 0], [1, 0, 1, 0, 1])
    assert mae[0] == 1 and np.isnan(r[0])
    mae, r = profile_metrics([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0])
    assert mae[0] == 1 and r[0] == -1
    assert np.isnan(profile_metrics([0, 1]*3, [1]*6)[1][0])
    assert np.isnan(profile_metrics([0, 1]*3, [0, 1, 0, 1, 0, np.nan])[0][0])


def test_stratified_resampling_retains_sample_size():
    weights = stratified_weights(['a', 'a', 'b', 'b', 'b'], 100)
    assert (weights[:, :2].sum(axis=1) == 2).all()
    assert (weights[:, 2:].sum(axis=1) == 3).all()


def test_paired_identical_block_profiles_remain_identical():
    numerator = np.array([[0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0]], float)
    denominator = np.array([[1, 1, 1, 0, 0, 0]]*2 + [[0, 0, 0, 1, 1, 1]]*2, float)
    result = profile_bootstrap(numerator, denominator, ['2', '2', '3', '3'], None,
                               numerator, denominator, draws=100)
    assert result['mae'] == 0
    assert result['mae_ci']['lower'] == result['mae_ci']['upper'] == 0
    assert result['r'] == pytest.approx(1.)


def test_direction_uses_family_means_and_tie_tolerance():
    counts = direction_counts([1, -1, -1, 1e-13, np.nan], ['a', 'a', 'b', 'c', 'd'])
    assert counts['models'] == dict(positive=1, negative=2, tie=1, total=4)
    assert counts['families'] == dict(positive=0, negative=1, tie=2, total=3)
