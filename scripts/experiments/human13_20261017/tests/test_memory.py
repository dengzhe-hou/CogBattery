"""Offline tests only. CVLT words below are synthetic fixtures, not research data."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps/public_human_structure_20261015"))
from scripts.experiments.human13_20261017 import memory

RAW = Path("data/raw")


def config(tmp_path):
    return {"root": str(ROOT), "raw_root": str(RAW), "materials_dir": str(tmp_path), "seed": 20261017}


def synthetic_cvlt():
    """Deliberately artificial names and evidence, never placed in real materials."""
    a = [{"word": f"fixture_a{i:02d}", "category": f"category_{i//4}"} for i in range(16)]
    b = [{"word": f"fixture_b{i:02d}", "category": f"category_{i//4}"} for i in range(16)]
    return {"version": "CVLT-II", "form": "Standard", "fixture_only": True,
            "source": {"reference": "SYNTHETIC UNIT TEST, not research material", "material_path": "synthetic://fixture",
                       "use_authorization": "SYNTHETIC UNIT TEST ONLY"},
            "protocol_reference": "SYNTHETIC UNIT TEST ONLY",
            "lists": {"A": a, "B": b}, "cue_order": [f"category_{i}" for i in range(4)],
            "recognition": [{"word": x["word"], "is_old": True} for x in a] +
                           [{"word": f"fixture_distractor{i:02d}", "is_old": False} for i in range(32)],
            "prompts": {"study": "Read {word}.", "recall_a": "Recall A.", "recall_b": "Recall B.",
                        "short_free": "Recall A.", "long_free": "Recall A.",
                        "cued": "Recall A words in {category}.", "recognition": "Did A contain {word}?"},
            "gap_steps": {"before_long": [], "before_forced_choice": []}, "forced_choice": [], "extra_steps": []}


def record(episode, step, response):
    return {"episode_id": episode["id"], "task": episode["task"], "condition": episode["condition"],
            "step": step, "response": response, "score": memory.score_step(step, response)}


def test_cvlt_missing_materials_is_blocked_not_fabricated(tmp_path):
    out = memory.build_cvlt(config(tmp_path))
    assert out["status"] == "blocked"
    assert out["episodes"] == []
    assert "schema" in out["provenance"]


def test_cvlt_full_order_counts_and_optional_steps(tmp_path):
    source = synthetic_cvlt()
    path = tmp_path / "cvlt_standard.json"
    path.write_text(json.dumps(source))
    episode = memory.build_cvlt(config(tmp_path))["episodes"][0]
    steps = episode["steps"]
    assert len(steps) == 160
    assert sum(s["kind"] == "ready" for s in steps) == 96
    assert sum(s["kind"] == "cvlt_recall" for s in steps) == 16
    assert sum(s["kind"] == "cvlt_recognition" for s in steps) == 48
    assert steps[16]["id"] == "A1_recall"
    assert steps[84]["id"] == "A5_recall"
    assert steps[101]["id"] == "B_recall"
    assert steps[102]["id"] == "short_free"
    assert steps[107]["id"] == "long_free"
    assert steps[112]["id"] == "recognition_01"
    # Extra instructions and contextual intervals are explicit outputs, not free.
    source["extra_steps"] = [{"before": "A1", "prompt": "Extra instruction"}]
    source["gap_steps"]["before_long"] = [{"prompt": "Context interval task"}]
    source["forced_choice"] = [{"choices": [x["word"], "fixture_foil"], "answer": 1} for x in source["lists"]["A"]]
    path.write_text(json.dumps(source))
    assert len(memory.build_cvlt(config(tmp_path))["episodes"][0]["steps"]) == 178


def test_cvlt_materials_reject_bad_recognition_and_missing_provenance(tmp_path):
    path = tmp_path / "cvlt_standard.json"
    source = synthetic_cvlt(); source["recognition"][0]["is_old"] = False
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="16 A words"):
        memory.build_cvlt(config(tmp_path))
    source = synthetic_cvlt(); source["source"].pop("use_authorization")
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="use_authorization"):
        memory.build_cvlt(config(tmp_path))


def test_cvlt_recall_scores_dedup_intrusions_and_cued_category():
    step = memory._step("recall", "", "cvlt_recall", ["Apple", "Pear"], all_a_words=["Apple", "Pear", "Plane"])
    result = memory.score_step(step, '["apple", "APPLE", "Plane", "imagined"]')
    assert result["valid"]
    assert result["correct_count"] == 1 and result["repetitions"] == 1
    assert result["intrusions"] == ["plane", "imagined"]
    assert result["out_of_category_a_words"] == ["plane"]
    assert memory.score_step(step, "[]")["proportion"] == 0
    assert not memory.score_step(step, "I recall Apple")["valid"]
    assert not memory.score_step(step, '["Apple", 1]')["valid"]
    assert memory.score_step(step, '["Pear", "Apple"]')["is_correct"]


def test_actual_drm_blocked_but_preserves_two_sentence_warnings(tmp_path):
    out = memory.build_drm(config(tmp_path))
    assert out["status"] == "blocked"
    assert "CITY" in " ".join(out["blockers"])
    assert {e["condition"] for e in out["episodes"]} == {"sentences_warning", "sentences_no_warning"}
    for episode in out["episodes"]:
        assert len(episode["steps"]) == 224
        probes = [s for s in episode["steps"] if s["kind"] == "drm_recognition"]
        assert sum(s["meta"]["primary56"] for s in probes) == 56
        assert sum(s["gold"] for s in probes) == 40
        assert len({s["meta"]["source_wordnum"] for s in probes}) == 80
    a, b = out["episodes"]
    assert [s["meta"].get("material") for s in a["steps"]] == [s["meta"].get("material") for s in b["steps"]]
    assert [s["meta"].get("word") for s in a["steps"]] == [s["meta"].get("word") for s in b["steps"]]


def test_drm_full_conditions_with_explicit_test_only_city_mapping(tmp_path):
    # Tests the eight-code application path, not a claim to have recovered CITY.
    path = tmp_path / "drm_city_mapping.json"
    path.write_text(json.dumps({"source_reference": "SYNTHETIC TEST ONLY, NOT VALID RESEARCH MATERIAL",
                                "mapping": {str(i): f"fixture_city_word_{i}" for i in range(1, 9)}}))
    out = memory.build_drm(config(tmp_path))
    assert len(out["episodes"]) == 4
    assert sum(len(e["steps"]) for e in out["episodes"]) == 896
    words = [s for e in out["episodes"] if e["condition"] == "words_warning" for s in e["steps"] if s["kind"] == "drm_rating"]
    assert len(words) == 64
    assert {s["meta"]["material"] for s in words if s["meta"]["theme"] == "CITY"} == {f"fixture_city_word_{i}" for i in range(1,9)}


def test_drm_accepts_sourced_eight_word_set_without_original_codes(tmp_path):
    # Test-only material, deliberately never written to the formal material folder.
    words = ["TOWN", "STATE", "COUNTRY", "SUBWAY", "CAPITAL",
             "fixture_unverified_6", "fixture_unverified_7", "fixture_unverified_8"]
    path = tmp_path / "drm_city_words.json"
    path.write_text(json.dumps({"source_reference": "SYNTHETIC UNIT TEST ONLY, NOT SOURCE CONFIRMATION", "words": words}))
    out = memory.build_drm(config(tmp_path))
    assert out["status"] == "ready" and len(out["episodes"]) == 4
    assert out["provenance"]["city_material"]["mode"] == "verified_word_set_new_ids"
    assert "city_mapping" not in out["provenance"]
    episode = next(e for e in out["episodes"] if e["condition"] == "words_warning")
    city = [s for s in episode["steps"] if s["kind"] == "drm_rating" and s["meta"]["theme"] == "CITY"]
    assert {s["meta"]["material"] for s in city} == set(words)
    assert {s["meta"]["prepared_item_id"] for s in city} == {f"city_word_{i:02d}" for i in range(1, 9)}
    assert all(s["meta"]["source_sennum"] is None for s in city)
    assert sum(len(e["steps"]) for e in out["episodes"]) == 896
    # A word set has no original order: permuting its JSON list cannot alter the prepared order.
    path.write_text(json.dumps({"source_reference": "SYNTHETIC UNIT TEST ONLY", "words": words[::-1]}))
    again = memory.build_drm(config(tmp_path))
    assert [[s["prompt"] for s in e["steps"]] for e in out["episodes"]] == [[s["prompt"] for s in e["steps"]] for e in again["episodes"]]


def test_drm_word_set_requires_eight_distinct_words_and_source(tmp_path):
    path = tmp_path / "drm_city_words.json"
    path.write_text(json.dumps({"words": [f"test_word_{i}" for i in range(8)]}))
    with pytest.raises(ValueError, match="source"):
        memory.build_drm(config(tmp_path))
    path.write_text(json.dumps({"source_reference": "SYNTHETIC TEST ONLY", "words": ["same"] * 8}))
    with pytest.raises(ValueError, match="eight distinct"):
        memory.build_drm(config(tmp_path))


def test_drm_followup_routes_on_response_not_gold(tmp_path):
    e = memory.build_drm(config(tmp_path))["episodes"][0]
    recognition, followup = e["steps"][64:66]
    for response, expected, forbidden in [("OLD", "FAMILIAR, REMEMBER, or GUESS", "UNFAMILIAR"),
                                           ("NEW", "UNFAMILIAR, RECALL_TO_REJECT", "FAMILIAR, REMEMBER")]:
        history = [record(e, recognition, response)]
        message = memory.messages(e, followup, history)[-1]["content"]
        assert expected in message and forbidden not in message
    history = [record(e, recognition, "unclear")]
    assert "UNCLASSIFIABLE" in memory.messages(e, followup, history)[-1]["content"]
    assert not memory.score_step(followup, "UNCLASSIFIABLE")["valid"]


def test_actual_sm_covers_full_bank_per_story_in_two_balanced_forms(tmp_path):
    out = memory.build_source_monitoring(config(tmp_path))
    assert out["status"] == "ready"
    assert len(out["episodes"]) == 6
    assert sum(len(e["steps"]) for e in out["episodes"]) == 168
    assert out["provenance"]["story_words"] == {"indirect": 984, "direct_agent_last": 936, "direct_agent_first": 935}
    by_story = {}
    for e in out["episodes"]:
        questions = e["steps"][4:]
        assert len(questions) == 24
        for theme in range(1,13):
            assert sum(s["meta"]["theme"] == theme for s in questions) == 2
        variant = questions[0]["meta"]["story_variant"]
        by_story.setdefault(variant, []).append({s["id"] for s in questions})
        for s in questions:
            if s["meta"]["source_row"] == 20: assert s["gold"] == "Olivia"
    for a, b in by_story.values():
        assert not a & b and len(a | b) == 48


def test_source_score_and_summary_keep_wrong_invalid_and_confusion(tmp_path):
    e = memory.build_source_monitoring(config(tmp_path))["episodes"][0]
    s = e["steps"][4]
    response = str(s["meta"]["options"].index(s["gold"]) + 1)
    assert memory.score_step(s, response)["is_correct"]
    assert memory.score_step(s, s["gold"].upper() + ".")["is_correct"]
    invalid = record(e,s,"Jake or Mike")
    correct = record(e,s,response)
    out = memory.summarize([correct, invalid])[0]
    assert out["questions"] == 2 and out["valid_questions"] == 1 and out["invalid_questions"] == 1
    assert out["accuracy_valid"] == 1
    assert out["confusion"][s["gold"]][s["gold"]] == 1


def test_drm_summary_denominators_and_hautus():
    e = memory._episode("test", "drm_false_memory", "synthetic", [])
    rows = []
    for probe, responses in [("old", ["OLD", "OLD"]), ("CL", ["OLD", "NEW"]), ("newunrelated", ["NEW", "NEW", "invalid"])]:
        for i, response in enumerate(responses):
            step = memory._step(f"{probe}_{i}", "", "drm_recognition", probe == "old", probe_class=probe)
            rows.append(record(e, step, response))
    out = memory.summarize(rows)[0]
    assert out["recognition"]["newunrelated"]["attempted"] == 3
    assert out["recognition"]["newunrelated"]["valid"] == 2
    assert out["recognition"]["newunrelated"]["invalid"] == 1
    assert out["hautus_valid_response_denominator"]["delta_dprime"] > 0
    assert not out["hautus_valid_response_denominator"]["all_primary_answers_valid"]


def test_cvlt_summary_reports_complete_endpoints_and_does_not_zero_missing(tmp_path):
    (tmp_path / "cvlt_standard.json").write_text(json.dumps(synthetic_cvlt()))
    episode = memory.build_cvlt(config(tmp_path))["episodes"][0]
    rows = []
    for step in episode["steps"]:
        if step["kind"] == "ready": response = "Ready"
        elif step["kind"] == "cvlt_recall": response = json.dumps(step["gold"])
        else: response = "YES" if step["gold"] else "NO"
        rows.append(record(episode, step, response))
    summary = memory.summarize(rows)[0]
    assert summary["learning_total_correct_out_of_80"] == 80
    assert all(x["correct"] == 16 for x in summary["recall_endpoints"].values())
    assert summary["recognition"]["True"]["old_responses"] == 16
    assert summary["recognition"]["False"]["old_responses"] == 0
    rows[16] = record(episode, episode["steps"][16], "not an answer array")
    assert memory.summarize(rows)[0]["learning_total_correct_out_of_80"] is None
