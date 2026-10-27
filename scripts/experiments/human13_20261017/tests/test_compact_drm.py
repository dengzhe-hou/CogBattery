"""Lossless-candidate checks on actual prepared sentences, with fixture replies."""
import copy
import json
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017 import compact_drm, memory

ROOT = Path(__file__).resolve().parents[4]
MATERIAL = ROOT / "results/human13_20261017/prepared_54_public_versions/materials/drm_false_memory.json"


@pytest.fixture(scope="module")
def episodes():
    bundle = json.loads(MATERIAL.read_text())
    assert bundle["status"] == "blocked"  # Sentence preview does not complete CITY.
    assert len(bundle["episodes"]) == 2
    return bundle["episodes"]


def rec(step, response):
    return {"step": step, "response": response, "score": memory.score_step(step, response)}


def reply(step, old=True):
    return {"drm_rating": "4", "drm_recognition": "OLD" if old else "NEW",
            "drm_report": "FAMILIAR" if old else "RECALL_TO_REJECT"}[step["kind"]]


@pytest.mark.parametrize("old", [False, True])
def test_all_prepared_steps_preserve_order_material_answers_and_current_prompt(episodes, old):
    for episode in episodes:
        history = []
        for step in episode["steps"]:
            messages = compact_drm.messages(episode, step, history)
            assert messages[0] == {"role": "system", "content": episode["system"]}
            assert len(messages) == 2
            assert messages[-1]["content"].endswith("Current request:\n" + memory._prompt_for(step, history))
            ledger = compact_drm.history_ledger(history)
            study = [r for r in history if r["step"]["kind"] == "drm_rating"]
            assert ledger["study"] == [[r["step"]["meta"]["material"], r["response"]] for r in study]
            recognition = [r for r in history if r["step"]["kind"] == "drm_recognition"]
            assert [row[:2] for row in ledger["tests"]] == [[r["step"]["meta"]["word"], r["response"]] for r in recognition]
            reports = [r for r in history if r["step"]["kind"] == "drm_report"]
            assert [row[2] for row in ledger["tests"] if len(row) == 3] == [r["response"] for r in reports]
            # Every serialized row is exactly decodable, no token-based truncation.
            assert [json.loads(compact_drm._json_row(row)) for row in ledger["study"]] == ledger["study"]
            assert [json.loads(compact_drm._json_row(row)) for row in ledger["tests"]] == ledger["tests"]
            history.append(rec(step, reply(step, old)))
        assert len(history) == 224
        ledger = compact_drm.history_ledger(history)
        assert len(ledger["study"]) == 64 and len(ledger["tests"]) == 80
        assert all(len(row) == 3 for row in ledger["tests"])


def test_conditions_block_boundaries_and_definition_timing(episodes):
    for episode in episodes:
        history = []
        first_instruction = episode["steps"][0]["prompt"].split("\nStudy block 1.\n", 1)[0]
        warning = "Such related but unstudied words will appear at test; avoid calling them OLD."
        for index, step in enumerate(episode["steps"]):
            text = compact_drm.messages(episode, step, history)[-1]["content"]
            assert text.count(first_instruction) == 1
            assert (warning in text) == ("_no_warning" not in episode["condition"])
            assert (memory.DRM_TEST_INSTRUCTIONS in text) == (index >= 64)
            if index == 64:
                assert all(f"Study block {block}." in text for block in range(1, 9))
            history.append(rec(step, reply(step)))


def test_invalid_long_and_multiline_responses_remain_literal_data(episodes):
    episode = episodes[0]
    history = [rec(s, reply(s)) for s in episode["steps"][:64]]
    long_response = 'invalid\n"a quoted answer"\\' + "verylong " * 500
    history[0] = rec(history[0]["step"], long_response)
    recognition, report = episode["steps"][64:66]
    history.append(rec(recognition, "not OLD or NEW\n[ambiguous]"))
    text = compact_drm.messages(episode, report, history)[-1]["content"]
    assert "Reply only UNCLASSIFIABLE." in text
    ledger = compact_drm.history_ledger(history)
    assert ledger["study"][0][1] == long_response
    assert json.loads(compact_drm._json_row(ledger["study"][0]))[1] == long_response
    assert ledger["tests"][0][1] == "not OLD or NEW\n[ambiguous]"
    assert len(text) > len(long_response)


def test_no_gold_hidden_classes_or_future_probes_used(episodes):
    episode = copy.deepcopy(episodes[0])
    history = [rec(s, reply(s)) for s in episode["steps"][:66]]
    current = episode["steps"][66]
    before = compact_drm.messages(episode, current, history)
    for record in history:
        record["step"]["gold"] = "GOLD_SENTINEL"
        for field in ("theme", "probe_class", "source_sennum", "source_wordnum", "primary56"):
            record["step"]["meta"][field] = "HIDDEN_SENTINEL"
        record["score"]["is_correct"] = "CORRECT_SENTINEL"
    for future in episode["steps"][67:]:
        future["prompt"] = "FUTURE_PROMPT_SENTINEL"
        future["meta"]["word"] = "FUTURE_WORD_SENTINEL"
    assert compact_drm.messages(episode, current, history) == before
    assert "SENTINEL" not in str(before)


def test_scoring_is_unchanged_and_formal_builder_is_unmodified():
    assert compact_drm.score_step is memory.score_step
    assert memory.messages is not compact_drm.messages


def test_two_sourced_word_length_candidates_stay_blocked_and_keep_non_city(tmp_path):
    from scripts.experiments.human13_20261017.prepare_drm_length_candidates import prepare
    from scripts.experiments.human13_20261017.run_r6 import check_authorized
    outputs = prepare(tmp_path)
    candidates = {key: json.loads(Path(path).read_text()) for key, path in outputs.items()}
    assert set(candidates) == {"conv", "div"}
    a, b = candidates["conv"], candidates["div"]
    for candidate in candidates.values():
        assert candidate["status"] == "blocked" and "CITY version remains unverified" in candidate["blockers"][0]
        assert len(candidate["provenance"]["models"]) == 54
        assert "tinyllama:1.1b" not in candidate["provenance"]["models"]
        assert candidate["provenance"]["inference_authorized"] is False
        assert {e["condition"] for e in candidate["episodes"]} == {"words_warning", "words_no_warning"}
        assert all(len(e["steps"]) == 224 for e in candidate["episodes"])
        with pytest.raises(RuntimeError, match="Material/protocol unresolved"):
            check_authorized({"authorization": {"inference_authorized": True}, "models": ["test"],
                              "tasks": ["drm_false_memory"]}, "drm_false_memory", "test", candidate)
    for ae, be in zip(a["episodes"], b["episodes"]):
        assert [s for s in ae["steps"] if s["meta"].get("theme") != "CITY"] == [s for s in be["steps"] if s["meta"].get("theme") != "CITY"]
        ac = {s["meta"]["material"] for s in ae["steps"] if s["meta"].get("theme") == "CITY"}
        bc = {s["meta"]["material"] for s in be["steps"] if s["meta"].get("theme") == "CITY"}
        assert ac - bc == {"VILLAGE", "SUBURB"}
        assert bc - ac == {"BIG", "CROWDED"}
        # Both complete word candidates also remain lossless through the formatter.
        for episode in (ae, be):
            history = [rec(s, reply(s)) for s in episode["steps"]]
            ledger = compact_drm.history_ledger(history)
            assert len(ledger["study"]) == 64 and len(ledger["tests"]) == 80
            assert {row[0] for row in ledger["study"]} == {s["meta"]["material"] for s in episode["steps"] if s["kind"] == "drm_rating"}
