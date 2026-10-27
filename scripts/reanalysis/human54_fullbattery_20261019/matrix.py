"""Build the approved post-hoc 13-task matrices from frozen endpoint tables.

This module never reads formal response records or fits statistical models.
All task columns point toward better task performance or metacognitive resolution.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


RULES = ("valid_only", "invalid_wrong")
TASKS = (
    "digit_span", "n_back", "operation_span", "stroop", "flanker", "go_nogo",
    "cvlt_word_list", "drm_false_memory", "source_monitoring", "false_belief", "epitome_tom",
    "confidence_calibration", "post_decision_wagering",
)
TASK_ABBREVIATIONS = dict(zip(TASKS, ("DS", "NB", "OS", "ST", "FL", "GN", "CVLT", "DRM", "SM", "FB", "EP", "CAL", "WAG")))

# The endpoint whose value is used, followed by endpoints whose coverage is
# required. Conditions explicitly averaged by the approved specification each
# retain their own coverage gate; aggregate coverage cannot mask a missing cell.
DIRECT = {
    "digit_span": ("position_accuracy", ("position_accuracy",)),
    "operation_span": ("position_accuracy", ("position_accuracy",)),
    "stroop": ("structure_sro", ("structure_sro", "congruent_accuracy", "incongruent_accuracy")),
    "flanker": ("structure_sro", ("structure_sro", "congruent_accuracy", "incongruent_accuracy", "neutral_accuracy")),
    "go_nogo": ("dprime", ("dprime",)),
    "cvlt_word_list": ("learning_total", ("learning_total",)),
    "source_monitoring": ("structure_ball", ("structure_ball",)),
    "false_belief": ("session|main_accuracy", ("session|main_accuracy",)),
    "confidence_calibration": ("type2_auc_fixed_direction", ("type2_auc_fixed_direction",)),
}
MEANS = {
    "n_back": ("load_2_dprime_episode_mean", "load_3_dprime_episode_mean"),
    "drm_false_memory": ("no_warning|balanced_accuracy", "warning|balanced_accuracy"),
    "epitome_tom": tuple(f"mental|{level}_accuracy" for level in range(1, 8)),
}
WAG_COMPONENTS = ("accuracy", "high_wager_rate", "wager_1_accuracy", "wager_2_accuracy")
METADATA_FIELDS = ("family_raw", "family_merged", "native_thinking", "panel54", "panel19", "panel51")


def _number(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def _metadata_value(value):
    if value in ("True", "False"):
        return value == "True"
    return value


def wager_auc_from_endpoints(accuracy, high_rate, low_accuracy, high_accuracy, n_scored):
    """Reconstruct binary-wager AUROC without revisiting trial-level responses.

    A correct answer wins a pair when its wager is higher than the incorrect
    answer's wager. Ties count one half. Single-wager models therefore have .5
    AUROC when both correctness classes exist; all-correct/all-wrong is undefined.
    """
    accuracy, high_rate, low_accuracy, high_accuracy, n_scored = (
        _number(x) for x in (accuracy, high_rate, low_accuracy, high_accuracy, n_scored)
    )
    if not all(math.isfinite(x) for x in (accuracy, high_rate, n_scored)) or n_scored <= 0:
        return dict(auc=float("nan"), reason="missing_wager_joint_inputs")
    if not 0 <= high_rate <= 1 or not 0 <= accuracy <= 1:
        raise ValueError("Wager rates must lie between zero and one")
    high_n = n_scored * high_rate
    low_n = n_scored - high_n
    if (low_n > 0 and not math.isfinite(low_accuracy)) or (high_n > 0 and not math.isfinite(high_accuracy)):
        return dict(auc=float("nan"), reason="missing_nonempty_wager_class")
    correct_low = low_n * low_accuracy if low_n > 0 else 0.0
    correct_high = high_n * high_accuracy if high_n > 0 else 0.0
    wrong_low, wrong_high = low_n - correct_low, high_n - correct_high
    correct_n = correct_low + correct_high
    wrong_n = wrong_low + wrong_high
    if not math.isclose(correct_n, n_scored * accuracy, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Wager conditional accuracies disagree with overall accuracy")
    out = dict(correct_low=correct_low, wrong_low=wrong_low,
               correct_high=correct_high, wrong_high=wrong_high,
               n_correct=correct_n, n_wrong=wrong_n, n_scored=n_scored)
    if correct_n == 0 or wrong_n == 0:
        return dict(out, auc=float("nan"), reason="one_correctness_class")
    numerator = correct_high * wrong_low + 0.5 * (correct_low * wrong_low + correct_high * wrong_high)
    return dict(out, auc=numerator / (correct_n * wrong_n), reason="")


def build_matrices(rows: list[dict]) -> dict:
    """Return rule -> matrix, all_candidates, metadata, ledger, exclusions.

    ``matrix`` contains complete cases only, ordered by model name, with the 13
    columns in ``TASKS`` order. ``all_candidates`` preserves missing task values.
    The metadata index exactly matches ``matrix``. ``endpoint_ledger`` records
    every constituent, coverage decision, and reconstructed WAG joint counts.
    ``wag_sensitivity`` preserves the original high-minus-low endpoint.
    """
    lookup = {}
    model_metadata = {}
    models_by_rule = {rule: set() for rule in RULES}
    for row in rows:
        if row["rule"] not in RULES:
            continue
        key = (row["rule"], row["model"], row["task"], row["endpoint"])
        if key in lookup:
            raise ValueError(f"Duplicate endpoint: {key}")
        lookup[key] = row
        models_by_rule[row["rule"]].add(row["model"])
        metadata = {field: _metadata_value(row.get(field)) for field in METADATA_FIELDS}
        model_metadata.setdefault(row["model"], metadata)
        if model_metadata[row["model"]] != metadata:
            raise ValueError(f"Inconsistent model metadata: {row['model']}")

    results = {}
    for rule in RULES:
        candidate_rows, ledger, exclusions, wager_sensitivity = [], [], [], []
        for model in sorted(models_by_rule[rule]):
            candidate = {"model": model}
            for task in TASKS:
                value_endpoint, component_names = (DIRECT[task] if task in DIRECT else
                                                    (None, MEANS[task]) if task in MEANS else
                                                    (None, WAG_COMPONENTS))
                components, reasons = {}, []
                for endpoint in component_names:
                    source = lookup.get((rule, model, task, endpoint))
                    value = _number(source["value"]) if source else float("nan")
                    coverage = _number(source["coverage"]) if source else float("nan")
                    gate_pass = source is not None and (rule != "valid_only" or coverage >= 0.8)
                    # Empty wager classes have undefined class accuracy by design.
                    must_be_defined = not (task == "post_decision_wagering" and endpoint in ("wager_1_accuracy", "wager_2_accuracy"))
                    eligible = gate_pass and (math.isfinite(value) or not must_be_defined)
                    components[endpoint] = source
                    record = dict(rule=rule, model=model, task=task, endpoint=endpoint,
                                  value=value, coverage=coverage, gate_pass=gate_pass,
                                  required_defined=must_be_defined, eligible=eligible,
                                  n_valid=source.get("n_valid") if source else None,
                                  n_planned=source.get("n_planned") if source else None,
                                  n_scored=source.get("n_scored") if source else None,
                                  role="value" if endpoint == value_endpoint or value_endpoint is None else "coverage_gate")
                    ledger.append(record)
                    if source is None:
                        reasons.append(f"{endpoint}:missing_endpoint")
                    elif rule == "valid_only" and not gate_pass:
                        reasons.append(f"{endpoint}:coverage_below_0.8")
                    elif must_be_defined and not math.isfinite(value):
                        reasons.append(f"{endpoint}:undefined")

                task_value = float("nan")
                if not reasons:
                    if value_endpoint is not None:
                        task_value = _number(components[value_endpoint]["value"])
                    elif task in MEANS:
                        task_value = float(np.mean([_number(components[name]["value"]) for name in component_names]))
                    else:
                        joint = wager_auc_from_endpoints(
                            components["accuracy"]["value"], components["high_wager_rate"]["value"],
                            components["wager_1_accuracy"]["value"], components["wager_2_accuracy"]["value"],
                            components["accuracy"]["n_scored"])
                        task_value = joint.pop("auc")
                        reason = joint.pop("reason")
                        ledger.append(dict(rule=rule, model=model, task=task, endpoint="type2_auc_fixed_direction_derived",
                                           value=task_value, coverage=min(_number(r["coverage"]) for r in components.values()),
                                           gate_pass=True, required_defined=True, eligible=math.isfinite(task_value),
                                           role="derived_value", **joint))
                        if reason:
                            reasons.append(reason)
                candidate[task] = task_value
                if reasons:
                    exclusions.append(dict(rule=rule, model=model, task=task, reasons=";".join(reasons)))
            candidate_rows.append(candidate)
            sensitivity = lookup.get((rule, model, "post_decision_wagering", "high_minus_low_accuracy"))
            wager_sensitivity.append(dict(model=model,
                value=_number(sensitivity["value"]) if sensitivity else float("nan"),
                coverage=_number(sensitivity["coverage"]) if sensitivity else float("nan"),
                defined=sensitivity is not None and math.isfinite(_number(sensitivity["value"]))))

        all_candidates = pd.DataFrame(candidate_rows, columns=("model", *TASKS)).set_index("model")
        complete = all_candidates.notna().all(axis=1)
        matrix = all_candidates.loc[complete].copy()
        metadata = pd.DataFrame([model_metadata[model] for model in matrix.index], index=matrix.index, columns=METADATA_FIELDS)
        results[rule] = dict(matrix=matrix, all_candidates=all_candidates, metadata=metadata,
                             endpoint_ledger=pd.DataFrame(ledger), exclusions=exclusions,
                             wag_sensitivity=pd.DataFrame(wager_sensitivity).set_index("model"))
    return results
