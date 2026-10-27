"""Descriptive item fit for the repaired P9 Strachan data, with frozen scoring.

The 14 main entries are two versions of seven stories. Human participants take
one list each; uncertainty resamples whole participants within their original
list while keeping all item entries and model responses fixed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
INPUT = PROJECT / "results/phi3_cache_fix_20261013/p9_analysis"
SPEC = PROJECT / "scripts/experiments/same_stimulus_20261011/SPEC.json"
CONDITIONS = ("session", "isolated", "session_no_system", "isolated_no_system")


@dataclass
class HumanList:
    item_ids: list[str]
    participant_ids: list[str]
    scores: np.ndarray  # participant x item; a participant only takes one list


@dataclass
class HumanReference:
    items: list[dict]
    lists: dict[str, HumanList]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def human_reference(rows: list[dict]) -> HumanReference:
    """Recover the author-coded participant clusters without inferring scores."""
    score_columns = [key for key in rows[0] if key.startswith("score")]
    reference, strata, seen_participants = [], {}, set()
    for list_name in ("A", "B"):
        selected = sorted(
            (row for row in rows if row["task"] == f"False Belief {list_name}"),
            key=lambda row: int(row["item"]),
        )
        if not selected:
            raise ValueError(f"Missing human False Belief {list_name} list")
        columns = []
        for column in score_columns:
            missing = [row[column] in ("", "NA", "NaN", "nan") for row in selected]
            if all(missing):
                continue
            if any(missing):
                raise ValueError(f"Incomplete participant {column} in list {list_name}")
            columns.append(column)
        if not columns or seen_participants.intersection(columns):
            raise ValueError("Human list participants must be present and disjoint")
        seen_participants.update(columns)
        scores = np.asarray([[float(row[col]) for row in selected] for col in columns])
        if not np.isin(scores, [0, 1]).all():
            raise ValueError("Human author-coded scores must be 0 or 1")
        item_ids = [f"{list_name}{int(row['item'])}" for row in selected]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError(f"Duplicate human item in list {list_name}")
        strata[list_name] = HumanList(item_ids, columns, scores)
        for index, row in enumerate(selected):
            n = len(columns)
            correct = int(scores[:, index].sum())
            reference.append({
                "item_id": item_ids[index], "list": list_name,
                "position": int(row["item"]), "story_id": int(row["item"]),
                "state": row["trial_state"], "source": row["source"],
                "main_analysis": row["trial_state"] in ("F", "T"),
                "human_n": n, "human_correct": correct, "human_rate": correct / n,
            })
    return HumanReference(reference, strata)


def bootstrap_indices(reference: HumanReference, draws: int, seed: int) -> dict[str, np.ndarray]:
    """Draw the original number of participants independently within each list."""
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    return {name: rng.integers(0, len(group.participant_ids),
                             size=(draws, len(group.participant_ids)))
            for name, group in reference.lists.items()}


def bootstrap_human_rates(reference: HumanReference, draws: int, seed: int) -> np.ndarray:
    indices = bootstrap_indices(reference, draws, seed)
    # Each row of indices selects one complete participant, preserving dependence
    # among all the item responses from that participant's list.
    return np.concatenate([group.scores[indices[name]].mean(axis=1)
                           for name, group in reference.lists.items()], axis=1)


def fit_metrics(model: np.ndarray, human: np.ndarray) -> dict:
    """Signed Pearson r, absolute error, and model-minus-human mean error."""
    model, human = np.asarray(model, dtype=float), np.asarray(human, dtype=float)
    if model.ndim != 1 or model.shape != human.shape or len(model) < 2:
        raise ValueError("Fit needs aligned one-dimensional vectors of at least two items")
    if not np.isfinite(model).all() or not np.isfinite(human).all():
        raise ValueError("Fit vectors must contain finite values")
    model_constant = bool(np.ptp(model) == 0)
    human_constant = bool(np.ptp(human) == 0)
    reason = ("constant_model_and_human" if model_constant and human_constant else
              "constant_model" if model_constant else "constant_human" if human_constant else "")
    r = None if reason else float(np.corrcoef(model, human)[0, 1])
    difference = model - human
    return {"signed_r": r, "r_undefined_reason": reason,
            "mae": float(np.abs(difference).mean()),
            "mean_model_minus_human": float(difference.mean())}


def fit_with_intervals(model: np.ndarray, human: np.ndarray, human_draws: np.ndarray) -> dict:
    result = fit_metrics(model, human)
    difference = np.asarray(model)[None, :] - human_draws
    centered_model = np.asarray(model, dtype=float) - np.mean(model)
    centered_human = human_draws - human_draws.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(centered_model ** 2) * np.sum(centered_human ** 2, axis=1))
    r_draws = np.full(len(human_draws), np.nan)
    defined = denominator > 0
    r_draws[defined] = (centered_human[defined] @ centered_model) / denominator[defined]
    samples = {"signed_r": np.clip(r_draws, -1, 1),
               "mae": np.abs(difference).mean(axis=1),
               "mean_model_minus_human": difference.mean(axis=1)}
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        interval = np.quantile(finite, [0.025, 0.975]) if len(finite) else (None, None)
        result[f"{name}_ci_low"] = None if interval[0] is None else float(interval[0])
        result[f"{name}_ci_high"] = None if interval[1] is None else float(interval[1])
        result[f"{name}_valid_draws"] = len(finite)
        result[f"{name}_undefined_draws"] = len(values) - len(finite)
    return result


def validate_human_alignment(rows: list[dict], reference: HumanReference) -> None:
    """Check the original CSV counts/rates against author-coded participant data."""
    by_item = {row["item_id"]: row for row in reference.items}
    for row in rows:
        expected = by_item[row["item_id"]]
        if int(row["human_n"]) != expected["human_n"]:
            raise ValueError(f"Human sample-size mismatch: {row['item_id']}")
        if "human_correct" in row and int(row["human_correct"]) != expected["human_correct"]:
            raise ValueError(f"Human correct-count mismatch: {row['item_id']}")
        # Existing CSV decimal serialization differs from exact integer division
        # by roundoff only; counts above remain exact.
        if not np.isclose(float(row["human_rate"]), expected["human_rate"], rtol=0, atol=1e-15):
            raise ValueError(f"Human rate mismatch: {row['item_id']}")
        if row["state"] != expected["state"]:
            raise ValueError(f"Human state mismatch: {row['item_id']}")
        if row["main_analysis"] != str(expected["main_analysis"]):
            raise ValueError(f"Human main/catch mismatch: {row['item_id']}")
        if "list" in row and (row["list"] != expected["list"] or
                              int(row["position"]) != expected["position"]):
            raise ValueError(f"Human list/position mismatch: {row['item_id']}")


def validate_inputs(model_rows: list[dict], item_rows: list[dict], summary: dict,
                    reference: HumanReference) -> list[str]:
    """Fail on a mismatched frozen input instead of silently reducing the panel."""
    all_models = list(dict.fromkeys(row["model"] for row in model_rows))
    if len(all_models) != 20 or summary["strachan_models"] != 20:
        raise ValueError("The repaired formal Strachan panel must contain 20 models")
    expected_items = {f"{list_name}{number}" for list_name in ("A", "B") for number in range(1, 10)}
    if {row["item_id"] for row in reference.items} != expected_items:
        raise ValueError("Human reference must include the original A/B lists, nine entries each")
    main = [row for row in reference.items if row["main_analysis"]]
    if len(main) != 14 or sum(row["state"] == "F" for row in main) != 7:
        raise ValueError("Expected 14 main entries: seven false-belief and seven true-belief")
    observed_keys = [(row["model"], row["condition"], row["item_id"]) for row in model_rows]
    expected_keys = {(model, condition, item) for model in all_models
                     for condition in CONDITIONS for item in expected_items}
    if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
        raise ValueError("Incomplete or duplicate formal model x condition x item data")
    aggregate_keys = [(row["condition"], row["item_id"]) for row in item_rows]
    if len(aggregate_keys) != 72 or set(aggregate_keys) != {
            (condition, item) for condition in CONDITIONS for item in expected_items}:
        raise ValueError("Incomplete or duplicate aggregate condition x item data")
    for rows in (model_rows, item_rows):
        validate_human_alignment(rows, reference)
        for row in rows:
            if row["condition_role"] != summary["strachan"]["roles"][row["condition"]]:
                raise ValueError("A frozen condition role disagrees with P9 summary")
    model_totals = {}
    for row in model_rows:
        if row["llm_correct"] not in ("0", "1"):
            raise ValueError("Frozen model scores must be 0 or 1")
        key = row["condition"], row["item_id"]
        model_totals[key] = model_totals.get(key, 0) + int(row["llm_correct"])
    for row in item_rows:
        total = model_totals[row["condition"], row["item_id"]]
        if int(row["llm_models"]) != 20 or int(row["llm_models_correct"]) != total:
            raise ValueError("P9 individual and aggregate model scores disagree")
        if not np.isclose(float(row["llm_rate_models"]), total / 20, rtol=0, atol=1e-15):
            raise ValueError("P9 aggregate model rate disagrees with individual scores")
    human_summary = summary["strachan"]["human"]
    counts = {name: len(group.participant_ids) for name, group in reference.lists.items()}
    if counts != {"A": 23, "B": 26} or counts != human_summary["n_subjects"]:
        raise ValueError("Expected the original 23 + 26 independent human participants")
    for state in ("F", "T"):
        selected = [row for row in main if row["state"] == state]
        correct = sum(row["human_correct"] for row in selected)
        n = sum(row["human_n"] for row in selected)
        expected = human_summary["totals"][state]
        if (correct, n) != (expected["correct"], expected["n"]):
            raise ValueError(f"Original human {state} totals disagree with P9 summary")
    return all_models


def run(output: Path, models: list[str] | None = None, draws: int = 2000,
        seed: int = 20261018) -> dict:
    """Write CPU-only descriptive item fits; no model inference or rescoring."""
    output = Path(output)
    spec = json.loads(SPEC.read_text())
    human_source = Path(spec["materials"]["strachan"]["source_human_scores"])
    reference = human_reference(read_csv(human_source))
    model_rows = read_csv(INPUT / "strachan_by_model_item.csv")
    item_rows = read_csv(INPUT / "strachan_by_item.csv")
    summary = json.loads((INPUT / "summary.json").read_text())
    all_models = validate_inputs(model_rows, item_rows, summary, reference)
    selected_models = all_models if models is None else list(models)
    if (not selected_models or len(set(selected_models)) != len(selected_models)
            or set(selected_models) - set(all_models)):
        raise ValueError("Requested models must be unique members of the formal 20-model panel")
    human_draws = bootstrap_human_rates(reference, draws, seed)
    main_indices = [i for i, row in enumerate(reference.items) if row["main_analysis"]]
    main_ids = [reference.items[i]["item_id"] for i in main_indices]
    human = np.asarray([reference.items[i]["human_rate"] for i in main_indices])
    main_draws = human_draws[:, main_indices]
    lookup = {(row["model"], row["condition"], row["item_id"]): row for row in model_rows}
    fits = []
    for model in selected_models:
        for condition in CONDITIONS:
            matched = [lookup[model, condition, item] for item in main_ids]
            model_values = np.asarray([int(row["llm_correct"]) for row in matched])
            fits.append({
                "model": model, "condition": condition,
                "condition_role": matched[0]["condition_role"],
                "family_merged": matched[0]["family_merged"],
                "strachan_run": matched[0]["strachan_run"],
                "think_false_flag": matched[0]["think_false_flag"],
                "n_items": 14, "n_story_templates": 7, "human_independent_participants": 49,
                "model_mean_accuracy": float(model_values.mean()),
                "human_mean_item_accuracy": float(human.mean()),
                **fit_with_intervals(model_values, human, main_draws),
            })
    human_lookup = {row["item_id"]: row for row in reference.items}
    raw = []
    for row in model_rows:
        if row["model"] in selected_models:
            original_human = human_lookup[row["item_id"]]
            difference = int(row["llm_correct"]) - original_human["human_rate"]
            raw.append({**row, "story_id": original_human["story_id"],
                        "fit_included": original_human["main_analysis"],
                        "absolute_error": abs(difference), "model_minus_human": difference})
    all_rates = np.asarray([row["human_rate"] for row in reference.items])
    human_metadata = {
        "source": str(human_source), "independent_participants": 49,
        "participants_by_list": {name: len(group.participant_ids) for name, group in reference.lists.items()},
        "participant_columns_by_list": {name: group.participant_ids for name, group in reference.lists.items()},
        "main_item_entries": 14, "main_story_templates": 7, "catch_item_entries": 4,
        "main_item_ids": main_ids,
        "main_items_at_ceiling": int(np.sum(human == 1)),
        "main_item_accuracy_range": [float(human.min()), float(human.max())],
        "main_item_accuracy_sd": float(human.std(ddof=1)),
        "item_n": {row["item_id"]: row["human_n"] for row in reference.items},
        "items": [{**row, "human_rate_ci_low": float(np.quantile(human_draws[:, i], 0.025)),
                   "human_rate_ci_high": float(np.quantile(human_draws[:, i], 0.975))}
                  for i, row in enumerate(reference.items)],
        "point_rates_match_original_author_scores": bool(np.array_equal(
            all_rates, np.concatenate([group.scores.mean(axis=0) for group in reference.lists.values()]))),
        "design_note": "A and B are separate participant groups (23 + 26 = 49). Each participant answered nine entries from one list, including seven main entries. The 14 main entries are A/B variants of seven story templates, not 14 independent stories. No participant answered all 18 entries.",
    }
    report = {
        "input_directory": str(INPUT), "spec": str(SPEC), "output_directory": str(output),
        "input_models": all_models, "selected_models": selected_models,
        "n_models": len(selected_models), "n_fit_rows": len(fits), "n_model_item_rows": len(raw),
        "conditions": {condition: summary["strachan"]["roles"][condition] for condition in CONDITIONS},
        "fit_items": 14, "catch_items": 4, "bootstrap_draws": draws, "seed": seed,
        "bootstrap_unit": "Whole participant, independently stratified by A/B list; original list sample sizes retained",
        "interval": "95% percentile; descriptive human-sampling uncertainty only, conditional on fixed 14 entries and frozen model scores",
        "undefined_r": "NA for a constant model or human vector. r intervals use defined bootstrap draws only; valid and undefined draw counts are reported. No imputation as zero.",
        "metric_note": "Equal weight per main item. signed_r retains direction; mae and mean_model_minus_human use probability units. Correlation is not a measure of absolute accuracy. Catch entries remain descriptive in model_item.csv and never enter fit.csv.",
        "scoring_note": "Original repaired P9 llm_correct values are used directly, with no raw-response rescoring. Human author-coded scores reproduce P9 per-item counts and rates. All four original condition roles are preserved.",
        "inference_note": "No p-values, significance tests, item bootstrap, model bootstrap, split-half analysis, model calls, or GPU computation.",
        "outputs": {name: str(output / name) for name in ("model_item.csv", "fit.csv", "human_variability.json", "summary.json")},
        "repair_provenance": summary.get("repair_provenance"),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "model_item.csv", raw)
    write_csv(output / "fit.csv", fits)
    for name, content in (("human_variability.json", human_metadata), ("summary.json", report)):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report
