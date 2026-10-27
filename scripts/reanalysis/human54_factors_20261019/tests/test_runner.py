import numpy as np
import pandas as pd

from scripts.reanalysis.human54_factors_20261019.run import (
    with_centroids, candidate_dimensions, synthetic_cases, json_ready,
)


def test_centroids_equal_family_not_equal_model():
    X = pd.DataFrame([[0., 2.], [2., 4.], [10., 12.]], index=['a', 'b', 'c'])
    fam = pd.Series(['f1', 'f1', 'f2'], index=X.index)
    centers, labels = with_centroids(X, fam)
    np.testing.assert_allclose(centers, [[1., 3.], [10., 12.]])
    assert labels.tolist() == ['f1', 'f2']
    assert not np.allclose(centers.mean(), X.mean())


def test_candidate_dimensions_use_fa_and_neighbors_not_pc():
    summaries = {'a': {'fa': {'retained': 2}, 'pc': {'retained': 5}},
                 'b': {'fa': {'retained': 0}}, 'c': {'fa': {'retained': None}}}
    assert candidate_dimensions(summaries, 13) == ([0, 2], [1, 2, 3])
    assert candidate_dimensions({'a': {'fa': {'retained': 8}}}, 13) == ([8], [7, 8])


def test_synthetic_interface_has_approved_sample_layout():
    cases = synthetic_cases()
    assert [len(x[0]) for x in cases.values()] == [53, 23, 54, 23]
    for X, fam in cases.values():
        assert X.shape[1] == 13 and fam.nunique() == 23
        assert X.index.equals(fam.index)
        assert np.isfinite(X.to_numpy()).all()


def test_json_preserves_failure_as_null_and_native_values():
    assert json_ready({'x': np.array([1., np.nan]), 'ok': np.bool_(False)}) == {
        'x': [1., None], 'ok': False}
