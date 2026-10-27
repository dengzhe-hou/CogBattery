"""Offline checks on material linkage, prompt leakage, and native score denominators."""
from collections import Counter
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017 import wm_cc


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def bundles():
    return wm_cc.build({"root": str(ROOT), "raw_root": str(ROOT.parent / "data/raw"),
                        "seed": 20261017, "materials_dir": "/tmp/human13_unused_test_output"})


def record(ep, step, response):
    return {"episode_id": ep["id"], "task": ep["task"], "condition": ep["condition"],
            "step": step, "response": response, "score": wm_cc.score_step(step, response)}


def content(ep, step, history):
    return "\n".join(m["content"] for m in wm_cc.messages(ep, step, history))


def test_selected_materials_counts_and_common_people(bundles):
    counts = {task: sum(len(ep["steps"]) for ep in b["episodes"]) for task, b in bundles.items()}
    assert counts == {"digit_span": 606, "n_back": 1800, "operation_span": 621,
                      "stroop": 360, "flanker": 504, "go_nogo": 1080}
    assert all(b["status"] == "ready" for b in bundles.values())
    for task in ("digit_span", "stroop", "flanker", "go_nogo"):
        assert bundles[task]["provenance"]["participant_ids"] == ["s143", "s037", "s490"]
    go = bundles["go_nogo"]["episodes"]
    assert {ep["steps"][0]["meta"]["go_color"] for ep in go} == {"BLUE", "ORANGE"}
    for task, count in (("stroop", 120), ("flanker", 168), ("go_nogo", 360)):
        assert all(len(ep["steps"]) == count for ep in bundles[task]["episodes"])


def test_ds_reversal_words_and_echo(bundles):
    ep = bundles["digit_span"]["episodes"][0]
    recalls = [s for s in ep["steps"] if s["kind"] == "digit_recall"]
    assert Counter(s["meta"]["direction"] for s in recalls) == {"forward": 14, "backward": 14}
    backward = next(s for s in recalls if s["meta"]["direction"] == "backward")
    original = __import__("json").loads(backward["meta"]["source_row"]["sequence"])
    assert backward["gold"] == original[::-1]
    step = {"kind": "digit_recall", "gold": [5, 3, 0], "meta": {}}
    words = wm_cc.score_step(step, "Five three zero")
    assert words["is_correct"] and not words["strict_original_is_correct"]
    assert not wm_cc.score_step(step, "Input 0 3 5; answer 5 3 0")["is_correct"]
    assert wm_cc.score_step(step, "The answer is 5 3 0.")["is_correct"]
    assert not wm_cc.score_step(step, "unknown")["valid"]


def test_ds_unit_reset_keeps_feedback_not_old_material(bundles):
    ep = bundles["digit_span"]["episodes"][0]
    recall_index = next(i for i, s in enumerate(ep["steps"]) if s["kind"] == "digit_recall")
    history = [record(ep, s, "Ready") for s in ep["steps"][:recall_index]]
    recalled = ep["steps"][recall_index]
    history.append(record(ep, recalled, " ".join(map(str, recalled["gold"]))))
    messages = wm_cc.messages(ep, ep["steps"][recall_index + 1], history)
    assert len(messages) == 2
    assert messages[-1]["content"].startswith("Previous response: Correct.\n")
    assert "Recall this sequence" not in messages[-1]["content"]


def test_nback_preserves_full_prefix_and_compacts_responses(bundles):
    b = bundles["n_back"]
    assert b["provenance"]["new_condition"]
    assert not b["provenance"]["archived_p9_data_reused_as_new"]
    ep = b["episodes"][0]
    assert len(ep["steps"]) == len(ep["source_episode"]["sequence"])
    history = [record(ep, s, "NO MATCH\n" + "verbose response " * 50) for s in ep["steps"][:20]]
    text = content(ep, ep["steps"][20], history)
    assert "1:" in text and "20:" in text and "21:" in text
    assert "22:" not in text
    assert "verbose response" not in text
    assert text.count("NO MATCH") >= 20
    assert "human_correct" not in text and "human_response" not in text
    assert ep["steps"][20]["meta"]["prefix"] == ep["source_episode"]["sequence"][:21]
    for step in ep["steps"]:
        assert step["gold"] == ("MATCH" if step["meta"]["condition"] == "target" else "NO MATCH")


@pytest.mark.parametrize("task", ["stroop", "flanker", "go_nogo"])
def test_cc_feedback_from_model_only_in_practice(task, bundles):
    ep = bundles[task]["episodes"][0]
    first, second = ep["steps"][:2]
    first = {**first, "meta": {**first["meta"], "human_response": "SECRET_HUMAN_ANSWER"}}
    practice_history = [record(ep, first, first["gold"])]
    assert "Previous response: Correct." in content(ep, second, practice_history)
    assert "SECRET_HUMAN_ANSWER" not in content(ep, second, practice_history)
    formal = [s for s in ep["steps"] if s["meta"]["stage"] == "test"]
    text = content(ep, formal[1], [record(ep, formal[0], formal[0]["gold"])])
    assert "Previous response:" not in text


def test_ant_neutral_and_all_cue_factors_retained(bundles):
    ep = bundles["flanker"]["episodes"][0]
    steps = [s for s in ep["steps"] if s["meta"]["stage"] == "test"]
    cells = Counter((s["meta"]["cue"], s["meta"]["location"], s["meta"]["condition"],
                     s["meta"]["target_direction"]) for s in steps)
    assert len(cells) == 48 and set(cells.values()) == {3}
    neutral = next(s for s in steps if s["meta"]["condition"] == "neutral")
    assert "Array: - - " in neutral["prompt"]
    assert " - -\n" in neutral["prompt"]


def test_go_nogo_no_output_is_not_withholding_and_window_15(bundles):
    ep = bundles["go_nogo"]["episodes"][0]
    nogo = next(s for s in ep["steps"] if s["meta"]["condition"] == "nogo")
    scored = wm_cc.score_step(nogo, "")
    assert not scored["valid"] and not scored["is_correct"]
    assert wm_cc.score_step(nogo, "WITHHOLD")["is_correct"]
    history = [record(ep, s, s["gold"]) for s in ep["steps"][:30]]
    text = content(ep, ep["steps"][30], history)
    assert "16:" in text and "30:" in text
    assert "\n15:" not in text
    assert "Previous response:" not in text


def test_ospan_full_versions_disjoint_math_and_counts(bundles):
    episodes = bundles["operation_span"]["episodes"]
    formal_materials = []
    for ep in episodes:
        assert len(ep["steps"]) == 207
        formal = [s for s in ep["steps"] if s["meta"]["stage"] == "test"]
        practice = [s for s in ep["steps"] if s["meta"]["stage"] == "practice"]
        assert Counter(s["kind"] for s in formal) == {"ospan_math": 75, "ready": 75, "ospan_recall": 15}
        assert len(practice) == 42
        recalls = [s for s in formal if s["kind"] == "ospan_recall"]
        assert Counter(s["meta"]["length"] for s in recalls) == {3: 3, 4: 3, 5: 3, 6: 3, 7: 3}
        assert all(len(set(s["gold"])) == len(s["gold"]) for s in recalls)
        formal_ids = {s["meta"]["source_math"]["source_line"] for s in formal if s["kind"] == "ospan_math"}
        practice_ids = {s["meta"]["source_math"]["source_line"] for s in practice if s["kind"] == "ospan_math"}
        assert len(formal_ids) == 75 and len(practice_ids) == 21
        assert not (formal_ids & practice_ids)
        formal_materials.append(tuple(tuple(s["gold"]) for s in recalls))
    assert len(set(formal_materials)) == 3


def test_ospan_position_and_malformed_scoring():
    step = {"kind": "ospan_recall", "gold": ["F", "H", "J"], "meta": {}}
    partial = wm_cc.score_step(step, "F _ J")
    assert partial["valid"] and partial["recall_correct_positions"] == 2
    assert not partial["is_correct"] and partial["positions"] == 3
    assert wm_cc.score_step(step, "F, H, J")["is_correct"]
    assert not wm_cc.score_step(step, "F H J K")["is_correct"]
    invalid = wm_cc.score_step(step, "The letters are F H J")
    assert not invalid["valid"] and invalid["recall_correct_positions"] == 0


def test_native_signal_denominators_and_ospan_partial_credit():
    ep = {"id": "fixture", "task": "go_nogo", "condition": "test"}
    records = []
    for condition, gold, response in [("go", "PRESS", "PRESS"), ("go", "PRESS", ""),
                                      ("nogo", "WITHHOLD", "PRESS"), ("nogo", "WITHHOLD", "WITHHOLD"),
                                      ("nogo", "WITHHOLD", "")]:
        step = {"kind": "go_nogo_choice", "gold": gold, "meta": {"condition": condition, "stage": "test"}}
        records.append(record(ep, step, response))
    signal = wm_cc.summarize(records)["go_nogo"]["signal"]
    assert signal["hit"]["all_trials"] == {"k": 1, "n": 2, "rate": .5}
    assert signal["hit"]["valid_response"] == {"k": 1, "n": 1, "rate": 1}
    assert signal["false_alarm"]["all_trials"] == {"k": 1, "n": 3, "rate": 1/3}
    assert signal["correct_rejection_all"]["k"] == 1
    assert signal["balanced_accuracy_all"] == pytest.approx((.5 + 1/3) / 2)
    ep = {"id": "os_fixture", "task": "operation_span", "condition": "test"}
    step = {"kind": "ospan_recall", "gold": ["F", "H", "J"], "meta": {"stage": "test"}}
    result = wm_cc.summarize([record(ep, step, "F _ J")])["operation_span"]
    assert result["position_accuracy"] == {"k": 2, "n": 3, "rate": 2/3}


def test_nback_retains_existing_per_block_dprime_definition():
    ep = {"id": "nb_fixture", "task": "n_back", "condition": "compact_full_prefix"}
    rows = []
    for i, kind in enumerate(["target"] * 5 + ["lure"] * 2 + ["control"] * 3):
        gold = "MATCH" if kind == "target" else "NO MATCH"
        step = {"kind": "n_back_choice", "gold": gold,
                "meta": {"stage": "test", "condition": kind, "trial": i + 1, "load": 2}}
        rows.append(record(ep, step, gold))
    result = wm_cc.summarize(rows)["n_back"]
    p9 = result["p9_endpoint_definitions_by_load"]["load_2"]
    assert p9["dprime_episodes_used"] == 1
    assert p9["dprime_episode_mean"] == pytest.approx(wm_cc.loglinear_dprime(5, 5, 0, 5))
    assert p9["R1"]["lure"] == {"k": 0, "n": 2, "rate": 0}
    assert result["signal_by_load"]["load_2"]["balanced_accuracy_all"] == 1


def test_missing_materials_blocked_without_substitution(tmp_path):
    output = wm_cc.build({"root": str(tmp_path), "raw_root": str(tmp_path), "seed": 20261017})
    assert set(output) == set(wm_cc.TASKS)
    assert all(b["status"] == "blocked" and b["blockers"] and not b["episodes"] for b in output.values())
