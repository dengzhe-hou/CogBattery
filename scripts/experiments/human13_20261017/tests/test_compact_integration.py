"""Check adopted serialization against inspected candidates without any model calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017 import compact_drm, compact_wag, memory
from scripts.experiments.human13_20261017.audit_compact_contexts import format_reply, SCENARIOS
from scripts.experiments.human13_20261017.prepare import prepare
from scripts.experiments.human13_20261017.registry import module_for
from scripts.experiments.human13_20261017.score import validate_records


ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'results/human13_20261017/prepared_54_public_versions'


@pytest.fixture(scope='module')
def manifests(tmp_path_factory):
    historical_files = [OLD / 'manifest.json', *sorted((OLD / 'materials').glob('*.json'))]
    before = {path: path.read_text() for path in historical_files}
    current = prepare(tmp_path_factory.mktemp('compact_integration'))
    assert all(path.read_text() == content for path, content in before.items())
    return json.loads(before[OLD / 'manifest.json']), current


def bundle(manifest, task):
    entry = next(row for row in manifest['tasks'] if row['task'] == task)
    return json.loads(Path(entry['material_file']).read_text())


def test_all_material_steps_and_panel_stay_unchanged(manifests):
    old, current = manifests
    # r7 (2026-10-18) deliberately changed inference/backends/FB reuse; the material
    # builder itself must stay unchanged, which the per-task checks below verify.
    assert current['spec']['revision'] == 'formal_r7_unified_20261018'
    assert current['spec']['drm_variant'] == 'sentence'
    assert current['spec']['models'] == old['spec']['models']
    assert len(current['spec']['models']) == 54
    assert set(current['spec']['reuse']['false_belief']) == set(old['spec']['reuse']['false_belief']) - {'deepseek-r1:7b', 'deepseek-r1:14b'}
    live = json.loads((ROOT / 'scripts/experiments/human13_20261017/SPEC.json').read_text())
    assert current['spec']['inference'] == live['inference']
    assert 'authorization' in current['spec']  # flag values are the user's (USER_DECISIONS.md); refusal tested on a closed copy
    assert current['spec']['presentation'] == {
        'cvlt_word_list': 'compact_history_v1',
        'drm_false_memory': 'compact_history_v1',
        'post_decision_wagering': 'compact_history_v1',
    }
    assert current['model_calls_executed'] == current['slurm_jobs_submitted'] == 0
    for entry in current['tasks']:
        task = entry['task']
        previous, active = bundle(old, task), bundle(current, task)
        if task == 'cvlt_word_list':
            # Explicitly authorized source integration replaces the old empty
            # CVLT preview; the other twelve task materials stay unchanged.
            assert previous['status'] == 'blocked' and not previous['episodes']
            assert active['status'] == 'ready' and not active['blockers']
            assert entry['constructed_calls_per_model'] == 176
            assert len(active['episodes']) == 1
            assert entry['presentation'] == 'compact_history_v1'
            continue
        if task == 'drm_false_memory':
            # The user authorized these complete sentence conditions separately;
            # unresolved word material no longer blocks sentence preparation.
            assert previous['status'] == 'blocked'
            assert active['status'] == 'ready' and not active['blockers']
            assert entry['constructed_calls_per_model'] == 448
        else:
            assert active['status'] == previous['status']
            assert active['blockers'] == previous['blockers']
        adopted = task in current['spec']['presentation']
        assert entry['presentation'] == ('compact_history_v1' if adopted else None)
        assert len(active['episodes']) == len(previous['episodes'])
        for episode, earlier in zip(active['episodes'], previous['episodes']):
            assert episode.get('presentation') == entry['presentation']
            assert {k: v for k, v in episode.items() if k != 'presentation'} == earlier
        provenance = deepcopy(active['provenance'])
        if adopted:
            assert provenance.pop('presentation')['version'] == 'compact_history_v1'
        # Missing-source input paths follow the new output folder, not new data.
        rebased = json.loads(json.dumps(provenance).replace(
            current['config']['materials_dir'], old['config']['materials_dir']))
        if task != 'drm_false_memory':
            assert rebased == previous['provenance']


@pytest.mark.parametrize('task,candidate', [
    ('drm_false_memory', compact_drm),
    ('post_decision_wagering', compact_wag),
])
@pytest.mark.parametrize('scenario', SCENARIOS)
def test_official_dispatch_equals_inspected_candidate_at_every_step(manifests, task, candidate, scenario):
    old, current = manifests
    previous = bundle(old, task)
    active = bundle(current, task)
    module = module_for(task)
    histories = {}
    checked = 0
    for earlier, episode in zip(previous['episodes'], active['episodes']):
        history = list(histories[episode['parent']]) if episode.get('parent') else []
        for step in episode['steps']:
            # The audited candidate received the historical, unmarked episode.
            expected = candidate.messages(earlier, step, history)
            assert module.messages(episode, step, history) == expected
            # The same module still renders historical snapshots in their old form.
            legacy = [{'role': 'system', 'content': earlier['system']}] if earlier.get('system') else []
            for index, row in enumerate(history):
                prompt = memory._prompt_for(row['step'], history[:index]) if task == 'drm_false_memory' else row['step']['prompt']
                response = row['response'] if task == 'drm_false_memory' else row['response'].strip()
                legacy.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}])
            prompt = memory._prompt_for(step, history) if task == 'drm_false_memory' else step['prompt']
            legacy.append({'role': 'user', 'content': prompt})
            assert module.messages(earlier, step, history) == legacy
            response = format_reply(step, scenario)
            history.append({'step': step, 'response': response, 'score': module.score_step(step, response)})
            checked += 1
        histories[episode['id']] = history
    assert checked == (448 if task == 'drm_false_memory' else 570)


@pytest.mark.parametrize('task', ['drm_false_memory', 'post_decision_wagering'])
def test_score_rejects_mixing_presentations_and_accepts_legacy_fb(manifests, task):
    old, current = manifests
    episode = bundle(current, task)['episodes'][0]
    row = {'model': current['spec']['models'][0], 'task': task,
           'episode_id': episode['id'], 'condition': episode['condition'],
           'step': episode['steps'][0], 'presentation': episode['presentation']}
    validate_records([row], current)
    with pytest.raises(ValueError, match='materials/presentation'):
        validate_records([row], old)
    legacy = {k: v for k, v in row.items() if k != 'presentation'}
    validate_records([legacy], old)
    with pytest.raises(ValueError, match='materials/presentation'):
        validate_records([legacy], current)
    fb = bundle(current, 'false_belief')['episodes'][0]
    validate_records([{'model': current['spec']['models'][0], 'task': 'false_belief',
                       'episode_id': fb['id'], 'condition': fb['condition'],
                       'step': fb['steps'][0]}], current)


def test_actual_entry_stays_closed_before_real_client_import(manifests, monkeypatch, tmp_path):
    import sys
    from scripts.experiments.human13_20261017 import run
    _, current = manifests
    manifest_path = Path(current['tasks'][0]['material_file']).parents[1] / 'manifest.json'
    monkeypatch.setattr(sys, 'argv', ['run', '--manifest', str(manifest_path),
                                    '--task', 'post_decision_wagering', '--model', current['spec']['models'][0],
                                    '--output', str(tmp_path / 'must_not_exist')])
    closed = _closed_copy(run.live_spec())
    monkeypatch.setattr(run, 'live_spec', lambda: closed)
    with pytest.raises(RuntimeError, match='Full evaluation is not authorized'):
        run.main()
    assert not (tmp_path / 'must_not_exist').exists()


def test_cvlt_compact_keeps_r4_material_and_all_176_steps(manifests):
    _, current = manifests
    r4 = json.loads((ROOT / 'results/human13_20261017/prepared_54_cvlt/manifest.json').read_text())
    old, active = bundle(r4, 'cvlt_word_list'), bundle(current, 'cvlt_word_list')
    assert current['spec']['models'] == r4['spec']['models']
    assert current['spec']['tasks'] == r4['spec']['tasks']
    assert old['status'] == active['status'] == 'ready'
    assert old['blockers'] == active['blockers'] == []
    assert len(old['episodes']) == len(active['episodes']) == 1
    assert len(active['episodes'][0]['steps']) == 176
    assert {k: v for k, v in active['episodes'][0].items() if k != 'presentation'} == old['episodes'][0]
    provenance = deepcopy(active['provenance'])
    assert provenance.pop('presentation')['version'] == 'compact_history_v1'
    rebased = json.loads(json.dumps(provenance).replace(
        current['config']['materials_dir'], r4['config']['materials_dir']))
    assert rebased == old['provenance']


def test_cvlt_dispatch_and_scoring_keep_presentations_separate(manifests):
    from scripts.experiments.human13_20261017 import compact_cvlt
    from scripts.experiments.human13_20261017.preflight import fixture_reply
    _, current = manifests
    r4 = json.loads((ROOT / 'results/human13_20261017/prepared_54_cvlt/manifest.json').read_text())
    old = bundle(r4, 'cvlt_word_list')['episodes'][0]
    episode = bundle(current, 'cvlt_word_list')['episodes'][0]
    history = []
    for step in episode['steps']:
        assert memory.messages(episode, step, history) == compact_cvlt.messages(episode, step, history)
        legacy = memory.messages(old, step, history)
        assert len(legacy) == 2 * len(history) + 2
        assert [m['content'] for m in legacy[2:-1:2]] == [r['response'] for r in history]
        response = fixture_reply(step)
        history.append({'step': step, 'response': response, 'score': memory.score_step(step, response)})
    row = {'model': current['spec']['models'][0], 'task': 'cvlt_word_list',
           'episode_id': episode['id'], 'condition': episode['condition'],
           'step': episode['steps'][0], 'presentation': episode['presentation']}
    validate_records([row], current)
    with pytest.raises(ValueError, match='materials/presentation'):
        validate_records([row], r4)
    legacy = {k: v for k, v in row.items() if k != 'presentation'}
    validate_records([legacy], r4)
    with pytest.raises(ValueError, match='materials/presentation'):
        validate_records([legacy], current)


def _closed_copy(spec):
    """A copy with every run flag closed: refusal logic is tested independently of the live
    authorization, which the parent opens after the user's decisions (USER_DECISIONS.md)."""
    from copy import deepcopy
    closed = deepcopy(spec)
    for flag in ('inference_authorized', 'formal_authorized', 'pilot_authorized'):
        closed['authorization'][flag] = False
    return closed
