"""Stage-boundary and lossless-history regressions using prepared real materials.

These checks render every step locally; they never construct a model client.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from scripts.experiments.human13_20261017 import (
    compact_cvlt, compact_drm, compact_wag, memory,
)


ROOT = Path(__file__).resolve().parents[4]
PREPARED = ROOT / "results/human13_20261017"


def episodes(task):
    folder = "prepared_54_cvlt" if task == "cvlt_word_list" else "prepared_54_public_versions"
    return json.loads((PREPARED / folder / "materials" / f"{task}.json").read_text())["episodes"]


def raw_record(step, number):
    return {"step": step, "response": f"  original-{number}\n**raw reply**\t ",
            "score": {"valid": True, "parsed": "NORMALIZED_SENTINEL"}}


def current_section(text, has_history):
    assert text.count("Current phase:") == 1
    if has_history:
        assert text.startswith("Past history (read-only record).")
        past, current = text.split("End of past history.\n", 1)
        assert "do not set the response format for this turn" in past
        assert current.startswith("Current phase:")
        return current
    assert "Past history" not in text
    return text


def test_cvlt_every_step_has_one_current_phase_and_keeps_every_raw_pair():
    episode = episodes("cvlt_word_list")[0]
    original = deepcopy(episode)
    history = []
    for index, step in enumerate(episode["steps"]):
        text = compact_cvlt.messages(episode, step, history)[-1]["content"]
        current = current_section(text, bool(history))
        assert current.startswith("Current phase: " + compact_cvlt.PHASES[step["kind"]] + ".")
        assert current.endswith("Current request:\n" + step["prompt"])
        # Past Ready/recognition formats must not carry over into a recall request.
        for kind, rule in compact_cvlt.FORMAT_RULES.items():
            assert (rule in current) == (kind == step["kind"])
        cursor = 0
        for row in compact_cvlt.history_ledger(history):
            literal = row["body"] + "\t" + row["response"]
            position = text.index(literal, cursor)
            cursor = position + len(literal)
            if row["prefix"]:
                assert row["prefix"] in text
            assert row["format_rule"] in text
        assert "NORMALIZED_SENTINEL" not in text
        history.append(raw_record(step, index))
    assert len(history) == 176 and episode == original


def test_drm_every_step_keeps_all_items_raw_replies_and_original_definitions():
    checked = 0
    for episode in episodes("drm_false_memory"):
        original = deepcopy(episode)
        history = []
        for index, step in enumerate(episode["steps"]):
            text = compact_drm.messages(episode, step, history)[-1]["content"]
            current = current_section(text, bool(history))
            assert current.endswith("Current request:\n" + memory._prompt_for(step, history))
            if step["kind"] == "drm_rating":
                assert current.startswith("Current phase: study meaning rating.")
                assert memory.DRM_TEST_INSTRUCTIONS not in text
            elif step["kind"] == "drm_recognition":
                assert current.startswith("Current phase: recognition, the first of two separate turns.")
                assert "classify after OLD or NEW apply to the next turn" in current
                assert "In this turn, reply only OLD or NEW; do not include a follow-up label." in current
                # Keep the human definitions/examples intact, including the original After rules.
                assert memory.DRM_TEST_INSTRUCTIONS in text
                assert "You answered OLD." not in current and "You answered NEW." not in current
            else:
                assert current.startswith("Current phase: judgement follow-up, the second of two separate turns.")
                assert "Reply only OLD or NEW." not in current
            # Decode the displayed rows, so whitespace/newlines are checked as raw data.
            displayed = [json.loads(line) for line in text.splitlines() if line.startswith('["')]
            expected = []
            for record in history:
                past = record["step"]
                if past["kind"] == "drm_rating":
                    expected.append([past["meta"]["material"], record["response"]])
                elif past["kind"] == "drm_recognition":
                    expected.append([past["meta"]["word"], record["response"]])
                else:
                    expected[-1].append(record["response"])
            assert displayed == expected
            assert "NORMALIZED_SENTINEL" not in text
            record = raw_record(step, index)
            if step["kind"] == "drm_recognition":
                record["response"] = "OLD" if (index // 2) % 2 else "NEW"
                record["score"] = memory.score_step(step, record["response"])
            history.append(record)
            checked += 1
        assert episode == original
    assert checked == 448


@pytest.mark.parametrize("response,expected", [
    ("OLD", "You answered OLD."),
    ("NEW", "You answered NEW."),
    ("neither choice", "Reply only UNCLASSIFIABLE."),
])
def test_drm_followup_uses_actual_preceding_judgement(response, expected):
    episode = episodes("drm_false_memory")[0]
    history = [raw_record(step, i) for i, step in enumerate(episode["steps"][:64])]
    recognition = episode["steps"][64]
    history.append({"step": recognition, "response": response,
                    "score": memory.score_step(recognition, response)})
    text = compact_drm.messages(episode, episode["steps"][65], history)[-1]["content"]
    current = current_section(text, True)
    assert expected in current
    assert json.loads(compact_drm._json_row(compact_drm.history_ledger(history)["tests"][-1]))[1] == response


def test_wag_every_branch_keeps_parent_history_and_only_current_response_format():
    histories = {}
    checked = 0
    for episode in episodes("post_decision_wagering"):
        original = deepcopy(episode)
        history = list(histories[episode["parent"]]) if episode.get("parent") else []
        for step in episode["steps"]:
            text = compact_wag.messages(episode, step, history)[-1]["content"]
            current = current_section(text, bool(history))
            rules = {"wag_training": compact_wag.TRAINING_RULE,
                     "wag_answer": compact_wag.ANSWER_RULE,
                     "wager": compact_wag.WAGER_RULE}
            for kind, rule in rules.items():
                assert (rule in current) == (kind == step["kind"])
            if step["kind"] == "wag_answer":
                assert "The wager is collected in the following turn." in current
            # Rule text remains visible at its original stage, including the human test overview.
            if episode["stage"] == "test":
                overview = episode["steps"][0]["prompt"].split("Test string:", 1)[0]
                assert overview in text
            for row in history:
                assert rules[row["step"]["kind"]] in text
            stimuli = re.findall(r"(?:Study \d+|Test \d+|Current training string|Current test string): ([MTVRX]+)", text)
            expected = []
            for item in history + [{"step": step}]:
                past = item["step"]
                if past["kind"] == "wag_training":
                    expected.append(past["prompt"].split("\n\n")[-1])
                elif past["kind"] == "wag_answer":
                    expected.append(re.search(r"Test string: ([MTVRX]+)", past["prompt"])[1])
            assert stimuli == expected
            cursor = 0
            for row in history:
                position = text.index(row["response"], cursor)
                cursor = position + len(row["response"])
            assert not re.search(r"(?m)^[RAW]:", text)
            assert "NORMALIZED_SENTINEL" not in text
            history.append(raw_record(step, checked))
            checked += 1
        histories[episode["id"]] = history
        assert episode == original
    assert checked == 570
