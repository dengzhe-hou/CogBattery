#!/usr/bin/env python3
"""Extract Ball et al. (2022) human scores; no correlation/effect analysis.

Input: public OSF participant-level SAV files, preserved in shared raw storage.
The three default paradigm columns are fixed before structural analysis:
operation_span = partial-credit ordered recall; drm = negative lure-minus-new
false-alarm contrast; source_monitoring = two source-task accuracies' mean.
Both versions are required for each two-version average. All original columns,
all participants and both experimental conditions remain in the output.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEPS = ROOT / ".deps/public_human_structure_20261015"
if importlib.util.find_spec("pyreadstat") is None:
    sys.path.insert(0, str(DEPS))
import pyreadstat

DEFAULT_RAW = Path("data/raw/ball2022_memory")
KEYS = ["Subject", "experiment"]
SOURCES = {
    "section_1_2.sav": "https://osf.io/download/kvdg4/",
    "table_1_variables.sav": "https://osf.io/download/npz3v/",
    "table_2_hautus.sav": "https://osf.io/download/r43vf/",
}


def read_table(path):
    frame, meta = pyreadstat.read_sav(path, apply_value_formats=False)
    assert not frame[KEYS].isna().any().any(), f"Missing participant key: {path}"
    assert not frame.duplicated(KEYS).any(), f"Duplicate participant key: {path}"
    for key in KEYS:
        assert (frame[key] == frame[key].astype(int)).all(), key
        frame[key] = frame[key].astype(int)
    return frame.set_index(KEYS), meta


def mean_complete(frame, columns):
    return frame[columns].mean(axis=1, skipna=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    main_scores, main_meta = read_table(args.raw / "section_1_2.sav")
    rates, rates_meta = read_table(args.raw / "table_1_variables.sav")
    sdt, sdt_meta = read_table(args.raw / "table_2_hautus.sav")
    frames = [main_scores, rates, sdt]
    for frame in frames[1:]:
        assert set(frame.index) == set(main_scores.index), "Participant sets differ"
    # Author tables repeat four working/source-memory scores. Verify by keys;
    # never rely on file row order or merge different participants by position.
    repeated = [c for c in main_scores if c in rates]
    for column in repeated:
        left = main_scores[column].sort_index().to_numpy()
        right = rates[column].sort_index().to_numpy()
        assert np.array_equal(left, right, equal_nan=True), column
    combined = main_scores.join(rates.drop(columns=repeated), validate="one_to_one")
    combined = combined.join(sdt, validate="one_to_one")
    for version in ("DRMw", "DRMs", "MCw", "MCs"):
        assert np.allclose(
            combined[version], combined[f"{version}_Delta_Dprime"],
            equal_nan=True,
        ), f"Published delta-dprime fields disagree: {version}"
    for column in rates:
        present = rates[column].dropna()
        assert present.between(0, 1).all(), f"Rate outside [0,1]: {column}"

    # Retain every source variable exactly as read; derived columns have explicit
    # names. Subject is globally unique in this release, but condition is still
    # part of the linkage key.
    result = combined.reset_index()
    result.insert(0, "participant_id", [
        f"ball2022_e{exp}_s{subject}"
        for subject, exp in zip(result["Subject"], result["experiment"])
    ])
    result.insert(1, "warning", result["experiment"].map({1: "no_warning", 2: "warning"}))
    assert result["warning"].notna().all(), "Unrecognized experiment condition"
    result["operation_span"] = result["SPANo"]
    result["reading_span"] = result["SPANr"]
    result["source_gender_accuracy"] = result["SMg"]
    result["source_picture_accuracy"] = result["SMp"]
    result["source_monitoring"] = mean_complete(result, ["SMg", "SMp"])

    for prefix, label in (("DRMw", "word"), ("DRMs", "sentence")):
        hit = result[f"{prefix}_Hit"]
        fa_lure = result[f"{prefix}_FA_Lure"]
        fa_new = result[f"{prefix}_FA_New"]
        result[f"drm_{label}_hit_rate"] = hit
        result[f"drm_{label}_lure_fa"] = fa_lure
        result[f"drm_{label}_unrelated_fa"] = fa_new
        result[f"drm_{label}_lure_correct_rejection"] = 1 - fa_lure
        result[f"drm_{label}_unrelated_correct_rejection"] = 1 - fa_new
        result[f"drm_{label}_native"] = -(fa_lure - fa_new)
        # Only 56 of 80 probes enter the author's reported rates. The remaining
        # 24 unrelated/theme-word probes cannot be reconstructed from this table.
        result[f"drm_{label}_accuracy_analyzed56"] = (
            40 * hit + 8 * (1 - fa_lure) + 8 * (1 - fa_new)
        ) / 56
        result[f"drm_{label}_balanced_accuracy"] = (hit + (1 - fa_lure) + (1 - fa_new)) / 3
        result[f"drm_{label}_delta_dprime"] = result[prefix]

    for target, suffix in (
        ("drm", "native"),
        ("drm_accuracy_analyzed56", "accuracy_analyzed56"),
        ("drm_balanced_accuracy", "balanced_accuracy"),
        ("drm_lure_correct_rejection", "lure_correct_rejection"),
        ("drm_unrelated_correct_rejection", "unrelated_correct_rejection"),
        ("drm_hit_rate", "hit_rate"),
        ("drm_lure_fa", "lure_fa"),
        ("drm_unrelated_fa", "unrelated_fa"),
        ("drm_delta_dprime", "delta_dprime"),
    ):
        result[target] = mean_complete(result, [f"drm_word_{suffix}", f"drm_sentence_{suffix}"])

    result.to_csv(args.out / "ball_scores.csv", index=False, na_rep="")
    default_columns = ["operation_span", "drm", "source_monitoring"]
    coverage = []
    for label, subset in [("all", result), *list(result.groupby("warning", sort=True))]:
        coverage.append({
            "condition": label,
            "rows": int(len(subset)),
            "complete_default_three": int(subset[default_columns].notna().all(axis=1).sum()),
            "nonmissing": {c: int(subset[c].notna().sum()) for c in default_columns},
        })
    metadata = {
        "study": "Ball et al. 2022, Memory & Cognition 50:751–764",
        "doi": "10.3758/s13421-021-01243-5",
        "raw_directory": str(args.raw),
        "source_files": SOURCES,
        "linkage": "Exact one-to-one join on Subject + experiment; all keys retained",
        "condition_values": {"1": "no warning", "2": "warning before study about critical lures"},
        "score_definitions": {
            "operation_span": "SPANo: proportion of 75 letters recalled in correct serial position, higher=better",
            "drm": "-mean(word FA_lure-FA_new, sentence FA_lure-FA_new); higher=less lure-specific false recognition, not overall ability",
            "source_monitoring": "mean(SMg,SMp): overall new/old+source identification accuracy, higher=better",
            "drm_accuracy_analyzed56": "Mean word/sentence accuracy on author-analyzed 40 old + 8 lure + 8 unrelated probes; not total 80-probe accuracy",
            "drm_balanced_accuracy": "Mean word/sentence average of hit, lure correct-rejection and unrelated correct-rejection rates",
            "drm_delta_dprime": "Author Delta dprime = dprime_new-dprime_lure; higher=larger lure vs unrelated false-alarm contrast in z units; NOT accuracy",
        },
        "aggregation": "Two versions equally weighted only when both are observed; missing otherwise; original components retained",
        "coverage": coverage,
        "original_column_labels": {
            "section_1_2.sav": main_meta.column_names_to_labels,
            "table_1_variables.sav": rates_meta.column_names_to_labels,
            "table_2_hautus.sav": sdt_meta.column_names_to_labels,
        },
        "original_value_labels": main_meta.variable_value_labels,
        "limits": [
            "No trial-level RT, missing-response, or reliability reconstruction from these participant summaries",
            "Source accuracy includes old/new decisions and source decisions; not source accuracy conditional on recognized old items",
            "DRM total 80-probe accuracy cannot be reconstructed: 24 unreported probes per version",
            "Word/sentence DRM and gender/picture source variants differ from CogArena stimuli and source interfaces",
            "Two groups have different DRM warning instructions and must remain identified in analysis",
            "This 3-paradigm subset has one WM indicator and two EM indicators; it cannot test the complete five-group structure",
        ],
    }
    (args.out / "ball_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.out / "ball_scores.csv"), "coverage": coverage}))


if __name__ == "__main__":
    main()
