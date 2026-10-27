"""Deterministic checks of the real 570-step WAG presentation, no model calls."""
from copy import deepcopy
from pathlib import Path
import re

import pytest

from scripts.experiments.human13_20261017 import compact_wag as C
from scripts.experiments.human13_20261017 import tom_mc


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def bundle():
    return tom_mc._wag({"raw_root": str(ROOT.parent / "data/raw")})


def response(step, n):
    # Deliberately non-gold, whitespace/multiline replies must remain unchanged.
    return f"  observed-{n}\nraw reply\t  "


def completed(bundle):
    histories = {}
    n = 0
    for episode in bundle["episodes"]:
        history = list(histories[episode["parent"]]) if episode.get("parent") else []
        for step in episode["steps"]:
            yield episode, step, history
            history = history + [{"step": step, "response": response(step, n)}]
            n += 1
        histories[episode["id"]] = history


def test_every_material_response_and_order_preserved(bundle):
    seen = 0
    for episode, step, history in completed(bundle):
        rendered = C.messages(episode, step, history)
        assert len(rendered) == 1 and rendered[0]["role"] == "user"
        text = rendered[0]["content"]
        # Recover every past response exactly, including whitespace and newlines.
        positions = [text.index(row["response"]) for row in history]
        assert positions == sorted(positions) and len(set(positions)) == len(positions)
        stimuli = re.findall(r"(?:Study \d+|Test \d+|Current training string|Current test string): ([MTVRX]+)", text)
        expected = []
        for row in history + [{"step": step}]:
            past = row["step"]
            if past["kind"] == "wag_training":
                expected.append(past["prompt"].split("\n\n")[-1])
            elif past["kind"] == "wag_answer":
                expected.append(re.search(r"Test string: ([MTVRX]+)", past["prompt"])[1])
        assert stimuli == expected
        assert C.TRAINING_RULE in text
        if episode["stage"] == "test":
            source_instructions = episode["steps"][0]["prompt"].split("Test string:")[0]
            assert source_instructions in text
            assert C.ANSWER_RULE in text
            wager_active = step["kind"] == "wager" or any(
                row["step"]["kind"] == "wager" for row in history)
            assert (C.WAGER_RULE in text) == wager_active
        seen += 1
    assert seen == 570


def test_gold_scores_and_metadata_never_rendered(bundle):
    episode, step, history = list(completed(bundle))[-1]
    episode, step, history = deepcopy((episode, step, history))
    for item in [step] + episode["steps"] + [row["step"] for row in history]:
        item["gold"] = "NEVER_REVEAL_GOLD"
        item["meta"] = {"source_grammar": "NEVER_REVEAL_META"}
    for row in history:
        row["score"] = {"is_correct": "NEVER_REVEAL_SCORE"}
    text = C.messages(episode, step, history)[0]["content"]
    assert "NEVER_REVEAL" not in text


def test_training_never_sees_future_test_or_incentive_instructions(bundle):
    seen = 0
    for episode, step, history in completed(bundle):
        if episode["stage"] != "training":
            continue
        text = C.messages(episode, step, history)[0]["content"]
        for future in ("complex set of rules", "Exactly half", "sweets", "wager", "balance",
                       "Classification rule", "Test string"):
            assert future not in text
        seen += 1
    assert seen == 90


def test_first_classification_precedes_full_wager_operation_instruction(bundle):
    checked = 0
    for episode, step, history in completed(bundle):
        if episode["stage"] != "test" or step not in episode["steps"][:2]:
            continue
        text = C.messages(episode, step, history)[0]["content"]
        original_first_rules = episode["steps"][0]["prompt"].split("Test string:")[0]
        assert original_first_rules in text
        if step["kind"] == "wag_answer":
            assert C.WAGER_RULE not in text
            assert "Do not change your answer" not in text
            assert "Respond 1 or 2 only" not in text
        else:
            assert step["kind"] == "wager"
            assert C.WAGER_RULE in text
            assert history[-1]["response"] in text
        checked += 1
    assert checked == 8  # First answer and wager in all four grammar/order branches.


def test_branch_history_does_not_expose_sibling_or_future(bundle):
    for episode, step, history in completed(bundle):
        if episode["stage"] != "test" or step != episode["steps"][0]:
            continue
        text = C.messages(episode, step, history)[0]["content"]
        assert len(history) == 45
        assert "Test 1:" not in text
        assert text.count("Current test string:") == 1
        assert all(row["step"]["id"].startswith(episode["parent"]) for row in history)
        corrupted = deepcopy(history)
        corrupted[-1]["step"]["id"] = "wag_test_sibling_01__answer"
        with pytest.raises(ValueError, match="every parent and own-branch"):
            C.messages(episode, step, corrupted)


def test_missing_training_or_past_answer_is_not_silently_omitted(bundle):
    episode, step, history = list(completed(bundle))[-1]
    with pytest.raises(ValueError, match="every parent and own-branch"):
        C.messages(episode, step, history[1:])


def test_input_materials_and_actual_history_are_not_mutated(bundle):
    episode, step, history = list(completed(bundle))[-1]
    before = deepcopy((episode, step, history))
    C.messages(episode, step, history)
    assert (episode, step, history) == before


def test_actual_long_response_remains_untruncated(bundle):
    episode, step, history = list(completed(bundle))[-1]
    history = deepcopy(history)
    history[-1]["response"] = "long actual reply " * 4000
    text = C.messages(episode, step, history)[0]["content"]
    assert history[-1]["response"] in text
