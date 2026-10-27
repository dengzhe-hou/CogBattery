"""Synthetic checks of fold isolation, family weights, and paired uncertainty.

Each check protects a concrete analysis property: held-out information must not
fit transformations, released checkpoint counts must not weight families, and
the vectorized bootstrap must reproduce literal whole-family sampling.  No
formal response or derived result is read by this module.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.reanalysis.human54_fullbattery_20261019.prediction import (
    GROUPS, analyze_prediction, build_features_from_frames,
)


COLUMNS = [task for tasks in GROUPS.values() for task in sorted(tasks)]
LABELS = {task: group for group, tasks in GROUPS.items() for task in tasks}


def synthetic_panel(*, families=14, members=2, group_signal=.8, seed=21):
    rng = np.random.default_rng(seed)
    general = rng.normal(size=families)
    specific = {group: rng.normal(size=families) for group in GROUPS}
    rows, names, mapping = [], [], {}
    for family in range(families):
        for member in range(members):
            name = f"synthetic_f{family:02d}_m{member}"
            names.append(name)
            mapping[name] = f"synthetic_f{family:02d}"
            rows.append([
                general[family] + group_signal * specific[LABELS[task]][family]
                + rng.normal(scale=.06)
                for task in COLUMNS
            ])
    return pd.DataFrame(rows, index=names, columns=COLUMNS), mapping


def test_each_fold_contains_only_other_families_and_uses_their_centroids():
    frame, mapping = synthetic_panel()
    summary, artifacts = analyze_prediction(frame, mapping, draws=50, seed=71)
    assert summary["n_prediction_rows"] == len(frame) * 13
    assert summary["n_family_target_cells"] == 14 * 13
    assert len(artifacts["fold_trace"]) == 14
    for trace in artifacts["fold_trace"]:
        heldout = trace["heldout_family"]
        train = frame.loc[[m for m in frame.index if mapping[m] != heldout]]
        centroids = train.groupby(pd.Series(mapping).reindex(train.index)).mean()
        assert heldout not in trace["training_families"]
        assert set(trace["training_families"]) == set(centroids.index)
        assert set(trace["test_models"]) == {
            model for model, family in mapping.items() if family == heldout
        }
        np.testing.assert_allclose(
            [trace["training_target_means"][t] for t in COLUMNS],
            centroids[COLUMNS].mean().to_numpy(),
        )
        np.testing.assert_allclose(
            [trace["training_target_sds"][t] for t in COLUMNS],
            centroids[COLUMNS].std(ddof=0).to_numpy(),
        )


def test_heldout_target_cannot_fit_its_prediction_or_standardization():
    frame, mapping = synthetic_panel()
    family, target = "synthetic_f03", COLUMNS[0]
    heldout = [m for m in frame.index if mapping[m] == family]
    changed = frame.copy()
    changed.loc[heldout, target] += 10_000.
    _, before = analyze_prediction(frame, mapping, draws=10)
    _, after = analyze_prediction(changed, mapping, draws=10)
    columns = ["pred_g_z", "pred_g_group_z", "pred_intercept_z",
               "train_outcome_mean", "train_outcome_sd"]
    mask = lambda table: table.loc[
        (table.family == family) & (table.target == target)
    ].sort_values("model")
    old, new = mask(before["predictions"]), mask(after["predictions"])
    np.testing.assert_allclose(old[columns], new[columns], rtol=0, atol=1e-12)
    assert (np.abs(old.truth_z.to_numpy() - new.truth_z.to_numpy()) > 1_000).all()


def test_extreme_heldout_predictors_cannot_fit_training_transformations():
    frame, mapping = synthetic_panel()
    family, target = "synthetic_f03", COLUMNS[0]
    train = frame.loc[[m for m in frame.index if mapping[m] != family]]
    train = train.groupby(pd.Series(mapping).reindex(train.index)).mean()
    test = frame.loc[[m for m in frame.index if mapping[m] == family]]
    extreme = test.copy()
    predictors = [t for t in COLUMNS if t != target]
    extreme.loc[:, predictors] = extreme[predictors] * 1e8 + 1e10
    gtr, gte, xtr, xte = build_features_from_frames(train, test, target, LABELS)
    changed = build_features_from_frames(train, extreme, target, LABELS)
    np.testing.assert_allclose(gtr, changed[0], rtol=0, atol=0)
    np.testing.assert_allclose(xtr, changed[2], rtol=0, atol=0)
    assert not np.allclose(gte, changed[1])
    assert not np.allclose(xte, changed[3])


def test_identical_family_member_copies_do_not_reweight_fit_or_metrics():
    frame, mapping = synthetic_panel(members=1)
    summary, artifacts = analyze_prediction(frame, mapping, draws=50, seed=71)
    original = frame.index[0]
    copies = pd.DataFrame(
        [frame.loc[original].to_numpy()] * 17,
        index=[f"synthetic_copy_{i}" for i in range(17)], columns=COLUMNS,
    )
    expanded = pd.concat([frame, copies])
    expanded_mapping = dict(mapping, **{m: mapping[original] for m in copies.index})
    duplicate_summary, duplicate_artifacts = analyze_prediction(
        expanded, expanded_mapping, draws=50, seed=71,
    )
    columns = ["model", "target", "pred_g_z", "pred_g_group_z", "truth_z"]
    original_predictions = artifacts["predictions"][columns].sort_values(["model", "target"])
    shared_predictions = duplicate_artifacts["predictions"].loc[
        duplicate_artifacts["predictions"].model.isin(frame.index), columns
    ].sort_values(["model", "target"])
    np.testing.assert_allclose(original_predictions.iloc[:, 2:], shared_predictions.iloc[:, 2:],
                               rtol=1e-11, atol=1e-11)
    for key in ["rmse_g", "rmse_g_group", "rmse_intercept", "relative_rmse_gain"]:
        assert duplicate_summary[key] == pytest.approx(summary[key], abs=1e-11)
    np.testing.assert_allclose(
        duplicate_artifacts["bootstrap_draws"]["relative_gain"],
        artifacts["bootstrap_draws"]["relative_gain"], atol=1e-11,
    )
    assert duplicate_summary["n_family_target_cells"] == summary["n_family_target_cells"]


def test_group_signal_is_recovered_without_forcing_a_gain_for_pure_general_data():
    frame, mapping = synthetic_panel(families=36, group_signal=1.2, seed=19)
    summary, _ = analyze_prediction(frame, mapping, draws=100, seed=71)
    assert summary["relative_rmse_gain"] > .2
    assert summary["bootstrap"]["ci95"][0] > 0

    # A rank-one panel has no peer information left after PC1 residualization.
    general = np.linspace(-2., 2., 16)
    pure = pd.DataFrame(
        general[:, None] * np.linspace(.5, 2., 13)[None, :] + np.arange(13)[None, :],
        index=[f"synthetic_general_{i}" for i in range(16)], columns=COLUMNS,
    )
    pure_mapping = {m: m for m in pure.index}
    general_summary, _ = analyze_prediction(pure, pure_mapping, draws=50, seed=71)
    assert general_summary["relative_rmse_gain"] == pytest.approx(0., abs=1e-6)
    np.testing.assert_allclose(general_summary["bootstrap"]["ci95"], [0., 0.], atol=1e-6)


def test_vectorized_bootstrap_equals_literal_paired_family_block_sampling():
    frame, mapping = synthetic_panel()
    summary, artifacts = analyze_prediction(frame, mapping, draws=50, seed=71)
    cells, draws = artifacts["cells"], artifacts["bootstrap_draws"]
    families = sorted(cells.family.unique())
    expected_indices = np.random.default_rng(71).integers(0, len(families), (50, len(families)))
    np.testing.assert_array_equal(draws["sampled_family_indices"], expected_indices)
    reference = []
    for selected in expected_indices:
        selected_cells = pd.concat([cells.loc[cells.family == families[i]] for i in selected])
        g, group, intercept = np.sqrt(
            selected_cells[["se_g", "se_group", "se_intercept"]].mean().to_numpy()
        )
        reference.append((g, group, intercept, 1 - group / g))
    reference = np.asarray(reference)
    for column, key in enumerate(["rmse_g", "rmse_group", "rmse_intercept", "relative_gain"]):
        np.testing.assert_allclose(draws[key], reference[:, column], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(summary["bootstrap"]["ci95"],
                               np.quantile(reference[:, 3], [.025, .975]))
