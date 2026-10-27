#!/usr/bin/env python3
"""Prepare the existing Ball human reference for sentence-only DRM comparisons.

This reuses the audited participant export and checks it against the three source
SAV tables. It neither averages word/sentence DRM nor calculates correlations,
significance tests, or model results. Every source row and export column stays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.reanalysis.public_human_structure_20261015 import build_ball

DEFAULT_SCORES = ROOT / "results/reanalysis/public_human_structure_20261015/inputs/ball_scores.csv"
DEFAULT_RAW = Path("data/raw/ball2022_memory")
IDENTIFIERS = ["participant_id", "Subject", "experiment", "warning"]
STRUCTURE_COLUMNS = {
    "operation_span": "operation_span",
    "drm_false_memory": "drm_sentence_native",
    "source_monitoring": "source_monitoring",
}
COMPLETE_SETS = {
    "three_paradigms": list(STRUCTURE_COLUMNS.values()),
    "four_indicators": ["operation_span", "drm_sentence_native", "source_gender_accuracy", "source_picture_accuracy"],
    "five_human_indicators": ["operation_span", "reading_span", "drm_sentence_native", "source_gender_accuracy", "source_picture_accuracy"],
}


def prepare_reference(scores_path=DEFAULT_SCORES, raw=DEFAULT_RAW):
    scores_path, raw = Path(scores_path), Path(raw)
    result = pd.read_csv(scores_path)
    if result.duplicated(build_ball.KEYS).any() or result[IDENTIFIERS].isna().any().any():
        raise ValueError("Ball reference requires unique, nonmissing participant/condition keys")
    keyed = result.set_index(build_ball.KEYS).sort_index()
    raw_metadata = {}
    for filename in build_ball.SOURCES:
        source, meta = build_ball.read_table(raw / filename)
        source = source.sort_index()
        if not source.index.equals(keyed.index):
            raise ValueError(f"Participant keys differ from {filename}")
        for column in source:
            np.testing.assert_allclose(keyed[column], source[column], rtol=0, atol=1e-14,
                                       equal_nan=True, err_msg=f"Source field differs: {filename}:{column}")
        raw_metadata[filename] = meta.column_names_to_labels
    expected_warning = result["experiment"].map({1: "no_warning", 2: "warning"})
    if expected_warning.isna().any() or not expected_warning.equals(result["warning"]):
        raise ValueError("Warning labels disagree with the author's experiment coding")

    hit, lure, new = (result[c] for c in ("DRMs_Hit", "DRMs_FA_Lure", "DRMs_FA_New"))
    z = NormalDist().inv_cdf
    corrected = {}
    for name, rate, n in (("hit", hit, 40), ("lure", lure, 8), ("new", new, 8)):
        values = rate.dropna()
        if not values.between(0, 1).all():
            raise ValueError(f"Sentence {name} rates outside [0, 1]")
        np.testing.assert_allclose(values * n, np.rint(values * n), rtol=0, atol=1e-12)
        corrected[name] = rate.map(lambda p: z((p * n + .5) / (n + 1)) if pd.notna(p) else np.nan)
    derived = {
        "drm_sentence_hit_rate": hit,
        "drm_sentence_lure_fa": lure,
        "drm_sentence_unrelated_fa": new,
        "drm_sentence_native": new - lure,
        "drm_sentence_accuracy_analyzed56": (40 * hit + 8 * (1 - lure) + 8 * (1 - new)) / 56,
        "drm_sentence_balanced_accuracy": (hit + 1 - lure + 1 - new) / 3,
        "drm_sentence_delta_dprime": corrected["lure"] - corrected["new"],
        "DRMs_Dprime_New": corrected["hit"] - corrected["new"],
        "DRMs_Dprime_Lure": corrected["hit"] - corrected["lure"],
        "DRMs_Delta_Dprime": corrected["lure"] - corrected["new"],
        "DRMs": corrected["lure"] - corrected["new"],
    }
    max_error = 0.0
    for column, expected in derived.items():
        np.testing.assert_allclose(result[column], expected, rtol=0, atol=1e-12, equal_nan=True,
                                   err_msg=f"Sentence endpoint differs: {column}")
        max_error = max(max_error, float((result[column] - expected).abs().max()))
    for name, columns in COMPLETE_SETS.items():
        result[f"complete_sentence_{name}"] = result[columns].notna().all(axis=1)

    coverage = []
    for condition, subset in [("all", result), *list(result.groupby("warning", sort=True))]:
        coverage.append({
            "condition": condition,
            "rows": len(subset),
            "sentence_endpoints_present": int(subset["drm_sentence_native"].notna().sum()),
            **{f"complete_{name}": int(subset[f"complete_sentence_{name}"].sum()) for name in COMPLETE_SETS},
        })
    metadata = {
        "status": "offline_reference_prepared_no_structure_analysis_or_model_results",
        "study": "Ball et al. (2022), Memory & Cognition 50:751–764",
        "doi": "10.3758/s13421-021-01243-5",
        "input_scores": str(scores_path),
        "input_exporter": str(ROOT / "scripts/reanalysis/public_human_structure_20261015/build_ball.py"),
        "raw_directory": str(raw),
        "source_files": build_ball.SOURCES,
        "linkage": "Subject + experiment; exact same participants across all source tables; retain every row",
        "variant": "sentences",
        "condition_column": "warning",
        "condition_values": {"1": "no_warning", "2": "warning"},
        "condition_rule": "Keep warning and no_warning distinguishable; no cross-condition averaging performed",
        "structure_columns": STRUCTURE_COLUMNS,
        "task_alignment": {
            "drm_false_memory": "Same Ball sentence/probe material; seeded model ordering, text interface and accessible history differ from human administration",
            "operation_span": "Model items generated under Ball's task rules; not a replay of the original human arithmetic/letter sequences or order",
            "source_monitoring": "Current model task uses Eerland stories; Ball human SMg/SMp measure gender/picture source memory. This is a task-family counterpart, not identical material",
            "overall": "Sentence DRM alignment does not make the complete three-paradigm comparison a same-stimulus experiment",
        },
        "complete_sets": COMPLETE_SETS,
        "model_summary_mapping": {
            name: f"structure_endpoints.{name}" for name in (
                "drm_sentence_native", "drm_sentence_accuracy_analyzed56",
                "drm_sentence_balanced_accuracy", "drm_sentence_delta_dprime")
        },
        "score_definitions": {
            "drm_sentence_hit_rate": "DRMs_Hit: OLD responses among 40 studied-word probes",
            "drm_sentence_lure_fa": "DRMs_FA_Lure: OLD responses among 8 unstudied critical-lure probes",
            "drm_sentence_unrelated_fa": "DRMs_FA_New: OLD responses among 8 unrelated-new probes",
            "drm_sentence_native": "FA_New - FA_Lure; higher means less lure-specific false recognition, not greater general ability",
            "drm_sentence_accuracy_analyzed56": "(40*hit + 8*(1-FA_Lure) + 8*(1-FA_New))/56; not all 80 recognition probes",
            "drm_sentence_balanced_accuracy": "(hit + (1-FA_Lure) + (1-FA_New))/3; classes equally weighted",
            "drm_sentence_delta_dprime": "Author DRMs = dprime_new - dprime_lure = z(corrected FA_Lure) - z(corrected FA_New); higher means MORE lure-specific false recognition",
            "hautus_correction": "(OLD_count+0.5)/(probe_count+1), applied separately to 40 old, 8 lure, 8 unrelated probes",
        },
        "verification": {"all_original_fields_match_raw_sav": True,
                         "all_sentence_endpoints_match_rate_reconstruction": True,
                         "max_absolute_endpoint_error": max_error},
        "coverage": coverage,
        "original_column_labels": raw_metadata,
        "limits": [
            "The legacy drm column is retained for provenance only: it averages word and sentence versions and is NOT selected in structure_columns.",
            "Word-version measurement remains pending CITY source verification; no word material is imputed or removed from the overall plan.",
            "Three counterpart paradigms cover working memory and episodic memory only, not all 13 paradigms or the full five-group structure.",
            "Four indicators separate the two human source-memory measures; five_human_indicators additionally includes reading span, not a new fourteenth CogArena paradigm or an approved extra model task.",
            "Source-memory and span columns preserve Ball's human tasks; existing CogArena or other-cohort model tasks are not automatically same-stimulus matches.",
            "The exported human rates cover 56 of 80 DRM recognition probes; the remaining 24 responses cannot be reconstructed from these participant summaries.",
            "Text presentation, accessible model history, seeded orders and paraphrased instructions retain differences from Ball's human administration.",
            "Completeness columns describe availability only. No participant filtering, correlation, hypothesis test or new conclusion is performed.",
        ],
    }
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result, metadata = prepare_reference(args.scores, args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "ball_sentence_scores.csv", index=False, na_rep="")
    (args.out / "ball_sentence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "coverage": metadata["coverage"],
                      "verification": metadata["verification"]}))


if __name__ == "__main__":
    main()
