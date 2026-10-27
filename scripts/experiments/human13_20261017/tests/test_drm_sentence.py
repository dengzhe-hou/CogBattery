"""Actual Ball sentence materials and scripted answers; no model calls."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import NormalDist
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps/public_human_structure_20261015"))
from scripts.experiments.human13_20261017 import memory
from scripts.experiments.human13_20261017.score import score_records, validate_records

RAW = Path("data/raw")


def config(tmp_path):
    return {"raw_root": str(RAW), "materials_dir": str(tmp_path), "seed": 20261017,
            "drm_variant": "sentence"}


def records_for(episode):
    """Known rates: hit=30/40, lure FA=6/8, unrelated FA=1/8."""
    seen = {"old": 0, "CL": 0, "newunrelated": 0, "newassoc": 0, "newtheme": 0}
    old_counts = {"old": 30, "CL": 6, "newunrelated": 1, "newassoc": 20, "newtheme": 4}
    rows = []
    for step in episode["steps"]:
        if step["kind"] != "drm_recognition":
            continue
        group = step["meta"]["probe_class"]
        response = "OLD" if seen[group] < old_counts[group] else "NEW"
        seen[group] += 1
        rows.append({"model": "FIXTURE", "task": episode["task"], "episode_id": episode["id"],
                     "condition": episode["condition"], "step": deepcopy(step), "response": response,
                     "finish_reason": "stop", "protocol_valid": True})
    return rows


def test_sentence_ready_never_reads_city_and_preserves_full_source(tmp_path, monkeypatch):
    def forbidden_city_read(_config):
        pytest.fail("Sentence-only preparation must not read or choose a CITY word candidate")
    monkeypatch.setattr(memory, "_load_drm_city", forbidden_city_read)
    bundle = memory.build_drm(config(tmp_path))
    assert bundle["status"] == "ready" and not bundle["blockers"]
    assert bundle["provenance"]["complete_planned_steps_per_model"] == 448
    deferred = bundle["provenance"]["deferred_word_variant"]
    assert deferred["status"] == "blocked_pending_source_verification"
    assert deferred["planned_steps_per_model"] == 448 and not deferred["city_material_loaded"]
    assert {ep["condition"] for ep in bundle["episodes"]} == {"sentences_no_warning", "sentences_warning"}
    study, probes = memory._read_drm(Path(bundle["provenance"]["sentence_material_file"]))
    for ep in bundle["episodes"]:
        assert len(ep["steps"]) == 224
        learning = [step for step in ep["steps"] if step["kind"] == "drm_rating"]
        test = [step for step in ep["steps"] if step["kind"] == "drm_recognition"]
        reports = [step for step in ep["steps"] if step["kind"] == "drm_report"]
        assert len(learning) == 64 and len(test) == len(reports) == 80
        assert {s["meta"]["material"] for s in learning} == {row["sen"] for row in study}
        assert {s["meta"]["source_wordnum"]: (s["meta"]["word"], s["gold"]) for s in test} == {
            int(row["wordnum"]): (row["word"], bool(row["correct"])) for row in probes}
        assert sum(s["meta"]["primary56"] for s in test) == 56
        assert {s["meta"]["recognition_id"] for s in reports} == {s["id"] for s in test}


def test_legacy_full_variant_stays_blocked_without_city(tmp_path):
    cfg = config(tmp_path)
    del cfg["drm_variant"]
    bundle = memory.build_drm(cfg)
    assert bundle["status"] == "blocked"
    assert "CITY" in " ".join(bundle["blockers"])
    assert bundle["provenance"]["complete_planned_steps_per_model"] == 896
    with pytest.raises(ValueError, match="Unsupported DRM variant"):
        memory.build_drm({**cfg, "drm_variant": "guessed_words"})


def test_sentence_endpoints_match_human_definitions_without_extra24_or_pooling(tmp_path):
    episodes = memory.build_drm(config(tmp_path))["episodes"]
    rows = [row for ep in episodes for row in records_for(ep)]
    summaries = score_records(rows)[0]["metrics"]
    assert len(summaries) == 2
    assert {s["warning"] for s in summaries} == {"no_warning", "warning"}
    z = NormalDist().inv_cdf
    for summary in summaries:
        assert summary["variant"] == "sentences" and summary["complete_primary56"]
        endpoints = summary["structure_endpoints"]
        assert endpoints["drm_sentence_native"] == pytest.approx(-.625)
        assert endpoints["drm_sentence_accuracy_analyzed56"] == pytest.approx(39 / 56)
        assert endpoints["drm_sentence_balanced_accuracy"] == pytest.approx(.625)
        assert endpoints["drm_sentence_delta_dprime"] == pytest.approx(z(6.5 / 9) - z(1.5 / 9))
        assert summary["hautus_valid_response_denominator"]["delta_dprime"] == endpoints["drm_sentence_delta_dprime"]
        assert not any("word" in key for key in endpoints)


@pytest.mark.parametrize("change", ["missing", "invalid"])
def test_sentence_incomplete_or_invalid_primary_probe_has_no_structure_endpoint(tmp_path, change):
    episode = memory.build_drm(config(tmp_path))["episodes"][0]
    rows = records_for(episode)
    index = next(i for i, row in enumerate(rows) if row["step"]["meta"]["probe_class"] == "CL")
    if change == "missing":
        rows.pop(index)
    else:
        rows[index]["response"] = "Maybe"
    summary = score_records(rows)[0]["metrics"][0]
    assert not summary["complete_primary56"]
    assert all(value is None for value in summary["structure_endpoints"].values())
    assert summary["recognition"]["CL"]["valid"] == 7


def test_sentence_manifest_rejects_old_word_variant(tmp_path):
    bundle = memory.build_drm(config(tmp_path))
    material = tmp_path / "sentence_material.json"
    material.write_text(json.dumps(bundle))
    manifest = {"spec": {"models": ["FIXTURE"]},
                "tasks": [{"task": "drm_false_memory", "material_file": str(material)}]}
    row = records_for(bundle["episodes"][0])[0]
    validate_records([row], manifest)
    old_word = deepcopy(row)
    old_word.update(episode_id="drm_words_no_warning", condition="words_no_warning")
    old_word["step"]["meta"]["variant"] = "words"
    with pytest.raises(ValueError, match="does not match manifest"):
        score_records([old_word], manifest)


def test_sentence_scorer_rejects_mixed_variant_or_warning(tmp_path):
    episode = memory.build_drm(config(tmp_path))["episodes"][0]
    rows = records_for(episode)
    rows[0]["step"]["meta"]["variant"] = "words"
    with pytest.raises(ValueError, match="one stimulus variant"):
        score_records(rows)
    rows = records_for(episode)
    rows[0]["step"]["meta"]["warned"] = True
    with pytest.raises(ValueError, match="one stimulus variant"):
        score_records(rows)
