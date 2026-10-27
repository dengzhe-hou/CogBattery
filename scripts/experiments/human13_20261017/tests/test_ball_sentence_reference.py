"""Deterministic checks against the cached public human reference; no inference."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.experiments.human13_20261017 import ball_sentence_reference as reference


def test_preserves_original_rows_conditions_and_columns():
    result, metadata = reference.prepare_reference()
    original = pd.read_csv(reference.DEFAULT_SCORES)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert metadata["structure_columns"] == {
        "operation_span": "operation_span", "drm_false_memory": "drm_sentence_native",
        "source_monitoring": "source_monitoring",
    }
    assert metadata["coverage"] == [
        {"condition": "all", "rows": 446, "sentence_endpoints_present": 437,
         "complete_three_paradigms": 409, "complete_four_indicators": 409, "complete_five_human_indicators": 406},
        {"condition": "no_warning", "rows": 245, "sentence_endpoints_present": 240,
         "complete_three_paradigms": 236, "complete_four_indicators": 236, "complete_five_human_indicators": 235},
        {"condition": "warning", "rows": 201, "sentence_endpoints_present": 197,
         "complete_three_paradigms": 173, "complete_four_indicators": 173, "complete_five_human_indicators": 171},
    ]


def test_sentence_endpoints_are_not_legacy_word_sentence_averages():
    result, metadata = reference.prepare_reference()
    observed = result.dropna(subset=["drm_sentence_native", "drm"])
    assert (observed.drm_sentence_native != observed.drm).any()
    np.testing.assert_allclose(result.drm_sentence_native,
                               result.DRMs_FA_New - result.DRMs_FA_Lure, equal_nan=True)
    np.testing.assert_allclose(result.drm_sentence_accuracy_analyzed56,
                              (40*result.DRMs_Hit + 8*(1-result.DRMs_FA_Lure) + 8*(1-result.DRMs_FA_New))/56,
                              equal_nan=True)
    finite = result.dropna(subset=["drm_sentence_native", "drm_sentence_delta_dprime"])
    # Larger lure false alarms lower the native endpoint but increase the author's delta.
    assert ((finite.drm_sentence_native * finite.drm_sentence_delta_dprime) <= 0).all()
    assert metadata["verification"]["max_absolute_endpoint_error"] < 1e-12
