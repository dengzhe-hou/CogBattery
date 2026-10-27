"""CPU-only legal-choice mapping and own-response branch regressions."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017.choice_constraints import choice_options
from scripts.experiments.human13_20261017.memory import _prompt_for


def item(kind, **meta):
    return {"kind": kind, "meta": meta}


@pytest.mark.parametrize("kind,expected", [
    ("n_back_choice", ["MATCH", "NO MATCH"]),
    ("ospan_math", ["YES", "NO"]),
    ("stroop_choice", ["RED", "GREEN", "BLUE"]),
    ("flanker_choice", ["LEFT", "RIGHT"]),
    ("go_nogo_choice", ["PRESS", "WITHHOLD"]),
    ("cvlt_recognition", ["YES", "NO"]),
    ("cvlt_forced", ["1", "2"]),
    ("drm_rating", [str(i) for i in range(1, 8)]),
    ("drm_recognition", ["OLD", "NEW"]),
    ("tom_answer", ["A", "B"]),
    ("cal_answer", ["s", "l"]),
    ("confidence", [f"{i}%" for i in range(0, 101, 10)]),
    ("wag_answer", ["YES", "NO"]),
    ("wager", ["1", "2"]),
])
def test_closed_choice_instructions_have_string_options_without_gold(kind, expected):
    assert choice_options(item(kind)) == expected
    changed_key = {"kind": kind, "gold": "ALTERED", "meta": {"target": "ALTERED", "source_math": {"answer": "ALTERED"}}}
    assert choice_options(changed_key) == expected


@pytest.mark.parametrize("kind", ["digit_recall", "ospan_recall", "cvlt_recall", "cvlt_recall_followup",
                                  "wag_training", "fb_answer", "ready", "instruction"])
def test_free_responses_and_ready_are_not_constrained(kind):
    assert choice_options(item(kind)) is None


def test_source_choice_preserves_both_explicitly_allowed_representations():
    step = item("source_choice", options=["Jake", "Tammy", "Olivia", "Mike"])
    expected = ["Jake", "Tammy", "Olivia", "Mike", "1", "2", "3", "4"]
    assert choice_options(step) == expected
    step["gold"] = "Tammy"
    assert choice_options(step) == expected
    step["gold"] = "Jake"
    assert choice_options(step) == expected


def test_actual_integer_calibration_options_do_not_replace_response_keys():
    # Actual cal_p0_q17 options are integers, but the prompt requests s/l.
    step = item("cal_answer", left_option=7, right_option=8, target=8)
    assert choice_options(step) == ["s", "l"]
    assert step["meta"]["left_option"] == 7 and step["meta"]["right_option"] == 8


def own_recognition(parsed, valid=True):
    return [{"step": {"id": "recognition_01", "gold": not parsed}, "response": "OWN RESPONSE",
             "score": {"valid": valid, "parsed": parsed, "is_correct": False}}]


@pytest.mark.parametrize("parsed,valid,expected", [
    (True, True, ["FAMILIAR", "REMEMBER", "GUESS"]),
    (False, True, ["UNFAMILIAR", "RECALL_TO_REJECT", "WOULD_REMEMBER"]),
    (True, False, ["UNCLASSIFIABLE"]),
])
def test_drm_branch_uses_own_valid_judgement_and_matches_actual_renderer(parsed, valid, expected):
    step = item("drm_report", recognition_id="recognition_01", probe_class="old", target=True)
    step["gold"] = not parsed
    history = own_recognition(parsed, valid)
    prompt = _prompt_for(step, history)
    assert choice_options(step, history=history) == expected
    assert choice_options(step, current_prompt=prompt) == expected
    assert choice_options(step, history=history, current_prompt=prompt) == expected
    step["gold"] = parsed
    step["meta"]["probe_class"] = "newassoc"
    history[0]["step"]["gold"] = parsed
    history[0]["score"]["is_correct"] = True
    assert choice_options(step, history=history) == expected


def test_drm_ignores_other_recognitions_and_old_ledger_branch_instructions():
    step = item("drm_report", recognition_id="recognition_01")
    history = own_recognition(False)
    history += [{"step": {"id": "recognition_02"}, "score": {"valid": True, "parsed": True}}]
    old_prompt = _prompt_for(step, own_recognition(True))
    current = _prompt_for(step, history)
    ledger = old_prompt + "\nCurrent request:\n" + current
    assert choice_options(step, current_prompt=ledger) == ["UNFAMILIAR", "RECALL_TO_REJECT", "WOULD_REMEMBER"]
    assert choice_options(step, history=[]) == ["UNCLASSIFIABLE"]


def test_drm_missing_or_conflicting_branch_information_fails_before_inference():
    step = item("drm_report", recognition_id="recognition_01")
    with pytest.raises(ValueError, match="require own-response"):
        choice_options(step)
    with pytest.raises(ValueError, match="actual rendered"):
        choice_options(step, current_prompt="Classify your immediately preceding OLD/NEW judgement.")
    with pytest.raises(ValueError, match="different branches"):
        choice_options(step, history=own_recognition(True), current_prompt=_prompt_for(step, own_recognition(False)))


def test_actual_full_materials_have_no_unmapped_kinds_and_do_not_require_answer_keys():
    root = Path(__file__).resolve().parents[4]
    # Copied from the repair package; the source there is the r6 full manifest (read-only).
    full = json.loads((root / "results/human13_20261017/prepared_54_drm_sentence/manifest.json").read_text())
    counts = Counter()
    constrained = 0
    for entry in full["tasks"]:
        bundle = json.loads(Path(entry["material_file"]).read_text())
        for episode in bundle["episodes"]:
            for original in episode["steps"]:
                step = deepcopy(original)
                step.pop("gold", None)
                step["meta"].pop("target", None)
                if step["kind"] == "drm_report":
                    options = choice_options(step, history=[])
                else:
                    options = choice_options(step)
                counts[step["kind"]] += 1
                if options is not None:
                    constrained += 1
                    assert options and all(isinstance(option, str) for option in options)
    assert len(full["tasks"]) == 13
    assert sum(counts.values()) == 6789 and len(counts) == 22
    assert constrained == 5536


def test_unknown_kind_is_not_silently_left_unconstrained():
    with pytest.raises(ValueError, match="Unmapped"):
        choice_options(item("unmapped_fixture_kind"))
