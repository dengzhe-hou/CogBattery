"""Synthetic inputs only; tests never access recorded endpoint values."""
import math

import numpy as np
import pytest

from scripts.reanalysis.human54_fullbattery_20261019.matrix import (
    DIRECT, MEANS, RULES, TASKS, build_matrices, wager_auc_from_endpoints,
)


def fixture_rows(models=("model-a",)):
    rows = []
    for model in models:
        for rule in RULES:
            for task in TASKS:
                if task in DIRECT:
                    endpoints = DIRECT[task][1]
                elif task in MEANS:
                    endpoints = MEANS[task]
                else:
                    endpoints = ("accuracy", "high_wager_rate", "wager_1_accuracy", "wager_2_accuracy", "high_minus_low_accuracy")
                for endpoint in endpoints:
                    value = {"accuracy": .5, "high_wager_rate": .5, "wager_1_accuracy": .25,
                             "wager_2_accuracy": .75, "high_minus_low_accuracy": .5}.get(endpoint, .7)
                    rows.append(dict(task=task, model=model, rule=rule, endpoint=endpoint,
                                     value=value, coverage=1., n_valid=100, n_planned=100, n_scored=100,
                                     family_raw="raw-a", family_merged="family-a", native_thinking=False,
                                     panel54=True, panel19=True, panel51=True))
    return rows


def find(rows, task, endpoint, model="model-a", rule="valid_only"):
    return next(r for r in rows if (r["task"], r["endpoint"], r["model"], r["rule"]) == (task, endpoint, model, rule))


def test_complete_matrix_order_and_metadata_with_csv_strings():
    rows = fixture_rows()
    for row in rows:
        row.update(value=str(row["value"]), coverage=str(row["coverage"]), native_thinking="False")
    result = build_matrices(rows)
    for rule in RULES:
        assert result[rule]["matrix"].shape == (1, 13)
        assert list(result[rule]["matrix"].columns) == list(TASKS)
        assert result[rule]["metadata"].loc["model-a", "family_merged"] == "family-a"
        assert not result[rule]["metadata"].loc["model-a", "native_thinking"]
        assert not result[rule]["exclusions"]


def test_unequal_load_and_warning_sample_sizes_are_equal_weight():
    rows = fixture_rows()
    for task, endpoints in (("n_back", MEANS["n_back"]), ("drm_false_memory", MEANS["drm_false_memory"])):
        find(rows, task, endpoints[0]).update(value=.2, n_valid=10, n_planned=10, n_scored=10)
        find(rows, task, endpoints[1]).update(value=.8, n_valid=90, n_planned=90, n_scored=90)
    matrix = build_matrices(rows)["valid_only"]["matrix"]
    assert matrix.loc["model-a", "n_back"] == .5
    assert matrix.loc["model-a", "drm_false_memory"] == .5


def test_cvlt_source_coverage_excludes_whole_model_without_imputation():
    rows = fixture_rows(("model-a", "model-b"))
    for rule in RULES:
        find(rows, "cvlt_word_list", "learning_total", rule=rule).update(value=80, coverage=.4, n_valid=2, n_planned=5)
        find(rows, "source_monitoring", "structure_ball", rule=rule).update(coverage=91/144, n_valid=91, n_planned=144)
    result = build_matrices(rows)
    assert list(result["valid_only"]["matrix"].index) == ["model-b"]
    assert math.isnan(result["valid_only"]["all_candidates"].loc["model-a", "cvlt_word_list"])
    assert math.isnan(result["valid_only"]["all_candidates"].loc["model-a", "source_monitoring"])
    assert result["invalid_wrong"]["matrix"].shape == (2, 13)
    assert len(result["valid_only"]["exclusions"]) == 2


def test_threshold_is_inclusive_and_aggregate_coverage_does_not_mask_condition():
    rows = fixture_rows()
    find(rows, "cvlt_word_list", "learning_total")["coverage"] = .8
    assert build_matrices(rows)["valid_only"]["matrix"].shape == (1, 13)
    find(rows, "flanker", "neutral_accuracy")["coverage"] = .79
    result = build_matrices(rows)["valid_only"]
    assert result["matrix"].empty
    assert math.isnan(result["all_candidates"].loc["model-a", "flanker"])
    assert "neutral_accuracy:coverage_below_0.8" in result["exclusions"][0]["reasons"]


@pytest.mark.parametrize("missing_task,endpoint", [("epitome_tom", "mental|4_accuracy"), ("n_back", "load_3_dprime_episode_mean")])
def test_missing_cell_never_renormalizes_remaining_cells(missing_task, endpoint):
    rows = fixture_rows()
    rows.remove(find(rows, missing_task, endpoint))
    result = build_matrices(rows)["valid_only"]
    assert result["matrix"].empty
    assert math.isnan(result["all_candidates"].loc["model-a", missing_task])


def test_seven_epitome_levels_have_equal_weight():
    rows = fixture_rows()
    for level in range(1, 8):
        find(rows, "epitome_tom", f"mental|{level}_accuracy").update(value=level/8, n_scored=level*10)
    assert build_matrices(rows)["valid_only"]["matrix"].loc["model-a", "epitome_tom"] == .5


def test_wager_auc_from_unequal_correctness_classes_matches_pairwise_definition():
    # low: 2 correct, 6 wrong; high: 3 correct, 1 wrong.
    result = wager_auc_from_endpoints(5/12, 4/12, 2/8, 3/4, 12)
    correct = [1, 1, 2, 2, 2]
    wrong = [1] * 6 + [2]
    expected = np.mean([float(a > b) + .5 * float(a == b) for a in correct for b in wrong])
    assert result["auc"] == pytest.approx(expected)
    assert result["auc"] == pytest.approx(25.5/35)
    assert result["correct_low"] == pytest.approx(2)
    assert result["wrong_high"] == pytest.approx(1)


@pytest.mark.parametrize("high_rate,low_accuracy,high_accuracy", [(0., 3/7, None), (1., None, 3/7)])
def test_constant_wager_is_naturally_point_five(high_rate, low_accuracy, high_accuracy):
    assert wager_auc_from_endpoints(3/7, high_rate, low_accuracy, high_accuracy, 7)["auc"] == pytest.approx(.5)
    rows = fixture_rows()
    for endpoint, value in (("accuracy", 3/7), ("high_wager_rate", high_rate), ("wager_1_accuracy", low_accuracy), ("wager_2_accuracy", high_accuracy)):
        find(rows, "post_decision_wagering", endpoint).update(value=value, n_scored=7)
    assert build_matrices(rows)["valid_only"]["matrix"].loc["model-a", "post_decision_wagering"] == pytest.approx(.5)


@pytest.mark.parametrize("accuracy", [0., 1.])
def test_one_correctness_class_is_undefined_and_not_filled(accuracy):
    assert math.isnan(wager_auc_from_endpoints(accuracy, .5, accuracy, accuracy, 10)["auc"])
    rows = fixture_rows()
    for endpoint in ("accuracy", "wager_1_accuracy", "wager_2_accuracy"):
        find(rows, "post_decision_wagering", endpoint)["value"] = accuracy
    assert build_matrices(rows)["valid_only"]["matrix"].empty


def test_inconsistent_joint_summaries_fail_instead_of_silently_changing_accuracy():
    with pytest.raises(ValueError, match="disagree with overall accuracy"):
        wager_auc_from_endpoints(.9, .5, .25, .75, 100)


def test_original_wager_high_low_endpoint_remains_in_sensitivity():
    result = build_matrices(fixture_rows())["valid_only"]
    assert result["wag_sensitivity"].loc["model-a", "value"] == .5
    assert result["wag_sensitivity"].loc["model-a", "defined"]
    derived = result["endpoint_ledger"].query("endpoint == 'type2_auc_fixed_direction_derived'").iloc[0]
    assert derived["value"] == .75
    assert derived["correct_low"] == 12.5
    assert derived["wrong_high"] == 12.5
