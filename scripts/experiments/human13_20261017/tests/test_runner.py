"""Offline regressions for actual shared-history, resume and no-inference requirements."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.experiments.human13_20261017.common import default_messages, validate_task
from scripts.experiments.human13_20261017.run_r6 import check_authorized, run_episodes


class Counter:
    def count(self, text):
        return len(text.split())


class FixtureClient:
    def __init__(self):
        self.requests = []
        self.serving_checked = False

    def serving(self):
        self.serving_checked = True

    def call(self, payload, identity):
        assert self.serving_checked
        self.requests.append((identity, payload))
        return {'response': identity, 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2}, 'latency_seconds': 0}


def episode(identity, parent=None):
    return {'id': identity, 'task': 'fixture', 'condition': identity,
            'system': 'Follow the instructions.', 'history_mode': 'full', 'parent': parent,
            'steps': [{'id': 'a', 'prompt': 'Present item A.', 'kind': 'fixture',
                       'gold': 'SECRET_NOT_FOR_PROMPT', 'meta': {}}]}


def module():
    return SimpleNamespace(messages=default_messages,
                           score_step=lambda step, response: {'parsed': response, 'valid': True,
                                                               'is_correct': response == step['gold']})


def spec():
    return {'inference': {'temperature': 0, 'max_tokens': 16, 'reasoning_effort': 'none',
                         'context_tokens': 2048, 'template_margin_tokens': 128,
                         'per_message_margin_tokens': 8}}


def test_parent_history_shared_without_sibling_leakage(tmp_path):
    bundle = {'status': 'ready', 'blockers': [],
              'episodes': [episode('train'), episode('forward', 'train'), episode('reverse', 'train')]}
    client = FixtureClient()
    result = run_episodes('fixture', bundle, module(), spec(), tmp_path, client, Counter())
    assert result['logical_calls'] == 3
    assert len(client.requests) == 3
    reverse = client.requests[-1][1]['messages']
    assert any(m['content'] == 'train__a' for m in reverse)
    assert not any('forward__a' in m['content'] for m in reverse)
    assert not any('SECRET_NOT_FOR_PROMPT' in m['content'] for _, req in client.requests for m in req['messages'])
    resumed = FixtureClient()
    again = run_episodes('fixture', bundle, module(), spec(), tmp_path, resumed, Counter())
    assert again['new_calls'] == 0 and not resumed.requests


def test_changed_material_cannot_reuse_old_output(tmp_path):
    bundle = {'status': 'ready', 'blockers': [], 'episodes': [episode('train')]}
    run_episodes('fixture', bundle, module(), spec(), tmp_path, FixtureClient(), Counter())
    changed = deepcopy(bundle)
    changed['episodes'][0]['steps'][0]['prompt'] = 'A different stimulus.'
    with pytest.raises(RuntimeError, match='materials.json differs'):
        run_episodes('fixture', changed, module(), spec(), tmp_path, FixtureClient(), Counter())


def test_presentation_is_recorded_and_checked_on_resume(tmp_path):
    ep = episode('train')
    ep['presentation'] = 'compact_history_v1'
    bundle = {'status': 'ready', 'blockers': [], 'episodes': [ep]}
    run_episodes('fixture', bundle, module(), spec(), tmp_path, FixtureClient(), Counter())
    record_file = tmp_path / 'calls/train__a.json'
    record = json.loads(record_file.read_text())
    assert record['presentation'] == 'compact_history_v1'
    del record['presentation']
    record_file.write_text(json.dumps(record))
    client = FixtureClient()
    with pytest.raises(RuntimeError, match='Resumed request/material differs'):
        run_episodes('fixture', bundle, module(), spec(), tmp_path, client, Counter())
    assert not client.requests and not client.serving_checked


def test_source_block_or_no_authorization_prevents_inference():
    s = {'authorization': {'inference_authorized': False}, 'tasks': ['fixture'], 'models': ['m']}
    with pytest.raises(RuntimeError, match='Preparation only'):
        check_authorized(s, 'fixture', 'm', {'status': 'ready'})
    s['authorization']['inference_authorized'] = True
    with pytest.raises(RuntimeError, match='CITY mapping absent'):
        check_authorized(s, 'fixture', 'm', {'status': 'blocked', 'blockers': ['CITY mapping absent']})


def test_context_failure_sends_no_request(tmp_path):
    bundle = {'status': 'ready', 'blockers': [], 'episodes': [episode('long')]}
    bundle['episodes'][0]['steps'][0]['prompt'] = 'word ' * 2000
    client = FixtureClient()
    with pytest.raises(RuntimeError, match='no history deleted'):
        run_episodes('fixture', bundle, module(), spec(), tmp_path, client, Counter())
    assert not client.requests


def test_late_overflow_is_detected_before_first_call(tmp_path):
    ep = episode('later_long')
    ep['steps'].append({**deepcopy(ep['steps'][0]), 'id': 'b', 'prompt': 'word ' * 2000})
    bundle = {'status': 'ready', 'blockers': [], 'episodes': [ep]}
    client = FixtureClient()
    with pytest.raises(RuntimeError, match='scripted precheck'):
        run_episodes('fixture', bundle, module(), spec(), tmp_path, client, Counter())
    assert not client.requests and not client.serving_checked


def test_serving_mismatch_generates_no_response(tmp_path):
    class WrongServing(FixtureClient):
        def serving(self):
            raise RuntimeError('Wrong served context')
    client = WrongServing()
    bundle = {'status': 'ready', 'blockers': [], 'episodes': [episode('first')]}
    with pytest.raises(RuntimeError, match='Wrong served context'):
        run_episodes('fixture', bundle, module(), spec(), tmp_path, client, Counter())
    assert not client.requests


def test_offline_score_rejects_wrong_manifest_material_and_model(tmp_path):
    from scripts.experiments.human13_20261017.common import write_json
    from scripts.experiments.human13_20261017.score import validate_records
    ep = episode('e')
    material = tmp_path / 'fixture.json'
    write_json(material, {'episodes': [ep]})
    manifest = {'spec': {'models': ['m']}, 'tasks': [{'task': 'fixture', 'material_file': str(material)}]}
    row = {'model': 'm', 'task': 'fixture', 'episode_id': 'e', 'condition': ep['condition'],
           'step': deepcopy(ep['steps'][0])}
    validate_records([row], manifest)
    changed = deepcopy(row)
    changed['step']['gold'] = 'DIFFERENT_KEY'
    with pytest.raises(ValueError, match='does not match manifest'):
        validate_records([changed], manifest)
    with pytest.raises(ValueError, match='outside the manifest panel'):
        validate_records([{**row, 'model': 'wrong_model'}], manifest)


def test_branch_parent_must_precede_child():
    with pytest.raises(ValueError, match='Parent must precede'):
        validate_task('fixture', {'status': 'ready', 'blockers': [],
                                 'episodes': [episode('child', 'missing')]})


def test_all13_preparation_and_offline_score_interfaces(tmp_path):
    # Actual source manifests, scripted format fixtures only; no Client or tokenizer.
    from scripts.experiments.human13_20261017.prepare import prepare
    from scripts.experiments.human13_20261017.preflight import fixture_reply
    from scripts.experiments.human13_20261017.score import score_records
    manifest = prepare(tmp_path)
    assert len(manifest['tasks']) == 13
    assert len(manifest['spec']['models']) == 54
    assert 'tinyllama:1.1b' not in manifest['spec']['models']
    # r7: R1 7B/14B removed from FB reuse (old think=false requests differ in bytes).
    assert len(manifest['spec']['reuse']['false_belief']) == 17
    assert manifest['model_calls_executed'] == 0
    assert 'authorization' in manifest['spec']  # flag values are the user's; refusal tested on closed copies
    rows = []
    constructed = set()
    for entry in manifest['tasks']:
        bundle = json.loads(open(entry['material_file']).read())
        for ep in bundle['episodes']:
            for step in ep['steps']:
                constructed.add(ep['task'])
                rows.append({'model': 'SCRIPTED_FORMAT_FIXTURE', 'task': ep['task'],
                             'episode_id': ep['id'], 'condition': ep['condition'],
                             'step': step, 'response': fixture_reply(step)})
    summaries = score_records(rows)
    assert {r['task'] for r in summaries} == constructed
    assert len(rows) == sum(t['constructed_calls_per_model'] for t in manifest['tasks'])
    json.dumps(summaries, allow_nan=False)
