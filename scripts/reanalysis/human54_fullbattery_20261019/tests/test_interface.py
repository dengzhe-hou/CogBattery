"""A synthetic run through matrix construction, all configurations, and outputs."""
import json

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_fullbattery_20261019.run import run_analysis
from scripts.reanalysis.human54_fullbattery_20261019.synthetic import make_synthetic_rows


def test_complete_synthetic_interface_and_all_configurations(tmp_path):
    output = tmp_path/'synthetic'
    results = run_analysis(make_synthetic_rows(), output, synthetic=True,
                           pca_draws=10, prediction_draws=20)
    assert {name: value['summary']['models'] for name, value in results.items()} == {
        'primary_valid_only': 53, 'invalid_wrong': 54, 'wag_highlow_sensitivity': 17}
    golden = ['digit_span', 'n_back', 'operation_span', 'stroop', 'flanker', 'go_nogo',
        'cvlt_word_list', 'drm_false_memory', 'source_monitoring', 'false_belief',
        'epitome_tom', 'confidence_calibration', 'post_decision_wagering']
    for name, value in results.items():
        frame = pd.read_csv(output/name/'matrix.csv', index_col=0)
        assert list(frame.columns) == golden
        assert np.isfinite(frame.to_numpy()).all()
        assert value['summary']['n_prediction_rows'] == len(frame)*13
        assert value['summary']['n_family_target_cells'] == value['summary']['families']*13
        assert value['pca']['bootstrap']['draws'] == 10
        assert value['prediction']['bootstrap']['draws'] == 20
        assert 'group_delta' in value['pca']['structure']
        predictions = pd.read_csv(output/name/'prediction_predictions.csv')
        assert len(predictions) == len(frame)*13 and not predictions.isna().any().any()
    manifest = json.loads((output/'run_manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['synthetic'] is True
    assert manifest['formal_response_record_reads'] == manifest['model_calls'] == 0
    assert (output/'summary.csv').is_file()
