"""Synthetic checks for the post hoc analyses. No formal data are read."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from scripts.reanalysis.human54_factors_20261019.core import fit_minres
from scripts.reanalysis.human54_fullbattery_20261019.matrix import TASKS
from scripts.reanalysis.human54_posthoc_20261020 import halves as H
from scripts.reanalysis.human54_posthoc_20261020 import lofo as L
from scripts.reanalysis.human54_posthoc_20261020 import reliability as REL
from scripts.reanalysis.human54_posthoc_20261020.data import ABBR
from scripts.reanalysis.human54_posthoc_20261020.run import run_prediction


def synthetic_matrix(n_models=40, n_families=10, seed=0):
    rng = np.random.default_rng(seed)
    general = rng.normal(size=n_models)
    X = general[:, None] * rng.uniform(.3, .9, len(TASKS)) + rng.normal(size=(n_models, len(TASKS)))
    models = [f'm{i:02d}' for i in range(n_models)]
    families = {m: f'f{i % n_families}' for i, m in enumerate(models)}
    return pd.DataFrame(X, index=models, columns=list(TASKS)), families


def test_ridge_matches_sklearn():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(22, 2)), rng.normal(size=22)
    beta, b0 = L.ridge(X, y)
    ref = Ridge(alpha=L.RIDGE_ALPHA, fit_intercept=True).fit(X, y)
    assert np.allclose(beta, ref.coef_, atol=1e-10) and np.isclose(b0, ref.intercept_, atol=1e-10)


@pytest.mark.parametrize('n,sizes,count', [(4, (2, 2), 3), (6, (3, 3), 10), (5, (3, 2), 10),
                                           (7, (3, 2, 2), 105), (13, (3, 3, 3, 2, 2), 600600)])
def test_partition_counts(n, sizes, count):
    parts = L.enumerate_partitions(n, sizes)
    assert len(parts) == count
    assert len({frozenset(frozenset(b) for b in p) for p in parts}) == count


def test_partition_engine_matches_cell_table():
    X, fam = synthetic_matrix()
    parts = L.enumerate_partitions(13, (3, 3, 3, 2, 2))
    summary, cells, _, gains = run_prediction(X, fam, parts, draws=200)   # raises on disagreement
    assert gains.shape == (600600,)
    assert summary['partition_reference']['n_partitions'] == 600600


def test_empty_peer_set_equals_general_axis():
    X, fam = synthetic_matrix()
    X = X.drop(columns=['post_decision_wagering'])
    peers = L.theory_peers(list(X.columns))
    assert peers['confidence_calibration'] == [frozenset()]
    rows, _ = L.fold_predictions(X, fam, peers, two_dim=False)
    cal = [r for r in rows if r['target'] == 'confidence_calibration']
    assert all(np.array_equal(r['peer'][frozenset()], r['pred_g']) for r in cal)


def test_factor_score_span_is_rotation_invariant():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(60, 2))
    Z = F @ rng.uniform(.4, .9, (2, 8)) + .6 * rng.normal(size=(60, 8))
    Z = (Z - Z.mean(0)) / Z.std(0)
    R = Z.T @ Z / len(Z)
    y = rng.normal(size=60)
    preds = []
    for rotate in (False, True):
        fit = fit_minres(R, 2, rotate=rotate)
        scores = Z @ np.linalg.solve(R, fit['pattern'] @ fit['phi'])
        beta, b0 = L.ridge(scores, y)
        preds.append(scores @ beta + b0)
    assert np.allclose(preds[0], preds[1], atol=1e-6)


def test_reliability_formulas():
    F = np.array([[1., 2.], [2., 2.5], [3., 4.], [4., 3.5]])
    r = np.corrcoef(F, rowvar=False)[0, 1]
    assert np.isclose(REL.reliability(F)['rel'], 2 * r / (1 + r))
    G = np.array([[1., 2., 2.], [2., 2.5, 3.], [3., 4., 3.], [4., 3.5, 5.], [2., 1., 2.]])
    k, total = 3, G.sum(1)
    alpha = k / (k - 1) * (1 - G.var(0, ddof=1).sum() / total.var(ddof=1))
    assert np.isclose(REL.reliability(G)['rel'], alpha)


def test_disattenuation_with_identical_forms_leaves_delta():
    X, fam = synthetic_matrix()
    rows = [dict(model=m, task=t, form=f, value=X.loc[m, t], n_rows='') for m in X.index for t in X.columns
            for f in ('a', 'b')]
    halves = pd.DataFrame(rows)
    summary, per_task, _ = REL.analyze(X, halves, [fam[m] for m in X.index], ABBR, draws=50, seed=1)
    assert np.allclose(per_task.rel, 1.)
    assert np.isclose(summary['delta_star'], summary['delta'])


def test_nb_halves_alternate_within_load():
    rows = [dict(episode_id=f'nb_compact_sro_nb_L{load}_s{s:03d}_b{b:02d}', step=dict(meta=dict(load=str(load))))
            for load in (2, 3) for s, b in zip(range(40), np.random.default_rng(0).integers(1, 30, 40))]
    label = H.form_of('n_back', rows)
    counts = pd.Series([(r['step']['meta']['load'], label(r)) for r in rows]).value_counts()
    assert set(counts) == {20}
