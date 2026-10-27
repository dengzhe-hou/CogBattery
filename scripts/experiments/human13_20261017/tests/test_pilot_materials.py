"""Pilot selection and synthetic-history separation, without model calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017 import pilot_materials
from scripts.experiments.human13_20261017.common import ROOT, TASKS, validate_task


SOURCE = ROOT / 'results/human13_20261017/prepared_54_drm_sentence/manifest.json'


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    source = json.loads(SOURCE.read_text())
    paths = [SOURCE, *[Path(entry['material_file']) for entry in source['tasks']]]
    original = {path: path.read_text() for path in paths}
    output = tmp_path_factory.mktemp('technical_pilot')
    manifest = pilot_materials.prepare_pilot(SOURCE, output)
    assert all(path.read_text() == text for path, text in original.items())
    return source, manifest, output


def test_exact_panel_settings_and_pilot_only_authorization(prepared):
    source, manifest, output = prepared
    assert manifest['scope'] == 'technical_pilot'
    assert manifest['spec']['models'] == source['spec']['models']
    assert manifest['spec']['tasks'] == source['spec']['tasks']
    assert len(manifest['spec']['models']) == 54
    for field in ('inference', 'backends', 'presentation', 'drm_variant', 'drm_word_pending'):
        assert manifest['spec'][field] == source['spec'][field]
    auth = manifest['spec']['authorization']
    assert auth['inference_authorized'] and auth['pilot_authorized']
    assert not auth['formal_authorized']
    assert source['spec']['authorization']['inference_authorized'] is False
    assert manifest['spec']['reuse']['false_belief'] == {}
    assert source['spec']['reuse']['false_belief']
    assert manifest['research_results_eligible'] is False
    assert json.loads((output / 'manifest.json').read_text()) == manifest


def test_fixed_186_call_prefixes_preserve_full_original_step_objects(prepared):
    source, manifest, _ = prepared
    source_entries = {row['task']: row for row in source['tasks']}
    assert [row['task'] for row in manifest['tasks']] == list(TASKS)
    total = 0
    for row in manifest['tasks']:
        task = row['task']
        original = json.loads(Path(source_entries[task]['material_file']).read_text())
        pilot = json.loads(Path(row['material_file']).read_text())
        count = validate_task(task, pilot)
        assert count == pilot_materials.PREFIX_CALLS[task]
        assert row['new_models'] == 54
        assert row['constructed_new_calls'] == 54 * count
        total += count
        for actual, expected in zip(pilot['episodes'], original['episodes']):
            assert actual['steps'] == expected['steps'][:len(actual['steps'])]
            assert {key: value for key, value in actual.items() if key != 'steps'} == {
                key: value for key, value in expected.items() if key != 'steps'}
    assert total == 186
    wager = json.loads(Path(manifest['tasks'][-1]['material_file']).read_text())
    assert [len(episode['steps']) for episode in wager['episodes']] == [45, 2]
    assert wager['episodes'][1]['parent'] == wager['episodes'][0]['id']


def test_three_separate_engineering_probes_use_audited_history_targets(prepared):
    _, manifest, output = prepared
    assert len(manifest['engineering_probes']) == 3
    assert manifest['engineering_probes'] == json.loads((output / 'probes.json').read_text())
    expected = {
        'cvlt_word_list': ('cvlt_standard', 'recognition_48', 175, 3345),
        'drm_false_memory': ('drm_sentences_warning', 'report_80', 223, 3661),
        'post_decision_wagering': ('wag_test_B_forward', 'wag_test_B_forward_q60__wager', 164, 2651),
    }
    for probe in manifest['engineering_probes']:
        assert probe['synthetic'] is True
        assert probe['research_results_eligible'] is False
        assert probe['scope'] == 'synthetic_history_engineering_probe'
        assert (probe['episode_id'], probe['step']['id'], probe['history_records'],
                probe['fixture_selection']['audited_peak_budget_tokens']) == expected[probe['task']]
        assert probe['presentation'] == 'compact_history_v1'
        assert probe['messages'][-1]['role'] == 'user'
        source_bundle = json.loads(Path(probe['source_material_file']).read_text())
        episode = next(row for row in source_bundle['episodes'] if row['id'] == probe['episode_id'])
        assert probe['episode'] == episode
        assert probe['step'] == episode['steps'][-1]
    assert manifest['expected_calls']['total_all_models'] == 10206
    assert manifest['model_calls_executed'] == manifest['slurm_jobs_submitted'] == 0


def test_refuses_overwriting_full_materials():
    with pytest.raises(ValueError, match='separate directory'):
        pilot_materials.prepare_pilot(SOURCE, SOURCE.parent)


def test_incomplete_parent_cannot_satisfy_wager_prefix():
    source = json.loads(SOURCE.read_text())
    entry = next(row for row in source['tasks'] if row['task'] == 'post_decision_wagering')
    bundle = json.loads(Path(entry['material_file']).read_text())
    bad = deepcopy(bundle)
    bad['episodes'] = bad['episodes'][1:]
    with pytest.raises(ValueError, match='Parent must precede'):
        pilot_materials.prefix_bundle('post_decision_wagering', bad)
