"""Offline pilot checks: no backend, model, network or Slurm calls."""
from copy import deepcopy
import json
from pathlib import Path
import urllib.error

import pytest

from scripts.experiments.human13_20261017.common import write_json
from scripts.experiments.human13_20261017.pilot_materials import prepare_pilot
from scripts.experiments.human13_20261017.pilot_run import (
    audit_output, diagnostic, known_failure, run_pilot, validate_pilot,
)
from scripts.experiments.human13_20261017.registry import module_for

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'results/human13_20261017/prepared_54_drm_sentence/manifest.json'
MODEL = 'qwen3:4b'


class Counter:
    metadata = {'scope': 'test_only_count_fixture'}

    def count(self, text):
        return 1


class Client:
    def __init__(self):
        self.serving_checked = False
        self.actual_calls = 0
        self.output = None
        self.requests = []

    def serving(self):
        self.serving_checked = True

    def call(self, payload, identity):
        self.actual_calls += 1
        self.requests.append((self.output, identity, deepcopy(payload)))
        return {'response': '1', 'raw_content': '1', 'finish_reason': 'stop',
                'protocol_valid': True, 'latency_seconds': 0.1,
                'usage': {'prompt_tokens': 10, 'completion_tokens': 1},
                'raw_response': {'message': {'content': '1'}, 'done_reason': 'stop',
                                 'prompt_eval_count': 10, 'eval_count': 1,
                                 'load_duration': 1000000000},
                'wire_request': {'think': False}}


@pytest.fixture
def manifest(tmp_path):
    return prepare_pilot(SOURCE, tmp_path / 'materials')


def test_pilot_rejects_formal_scope_or_changed_inference(manifest):
    assert validate_pilot(manifest, MODEL)['authorization']['formal_authorized'] is False
    changed = deepcopy(manifest)
    changed['scope'] = 'formal'
    with pytest.raises(RuntimeError, match='formal evaluation is forbidden'):
        validate_pilot(changed, MODEL)
    changed = deepcopy(manifest)
    changed['spec']['inference']['native_think'] = True
    with pytest.raises(ValueError, match='inference settings'):
        validate_pilot(changed, MODEL)
    changed = deepcopy(manifest)
    changed['spec']['authorization']['formal_authorized'] = True
    with pytest.raises(RuntimeError, match='formal evaluation is forbidden'):
        validate_pilot(changed, MODEL)


def test_exact_calls_separate_synthetic_and_resume(manifest, tmp_path):
    client = Client()
    result = run_pilot(manifest, MODEL, tmp_path / 'run', client, Counter(), module_for)
    assert result['status'] == 'complete'
    assert result['real_prefix_committed_calls'] == 186
    assert result['synthetic_engineering_committed_calls'] == 3
    assert client.actual_calls == 189
    assert result['effects_computed'] is False
    assert result['warmup']['counts_as_logical_pilot_call'] is False
    assert len(list((tmp_path / 'run/engineering_probes').glob('*/calls/*.json'))) == 3
    assert all(row['audit']['reasoning_channel_responses'] == 0 for row in result['tasks'])
    assert all(req['reasoning_effort'] == 'none' and req['temperature'] == 0
               for _, _, req in client.requests)
    resumed = Client()
    again = run_pilot(manifest, MODEL, tmp_path / 'run', resumed, Counter(), module_for)
    assert again['status'] == 'complete'
    assert again['real_prefix_committed_calls'] == 186
    assert again['actual_transport_calls_this_invocation'] == 0
    assert resumed.actual_calls == 0


def test_http400_stops_model_after_one_attempt(manifest, tmp_path):
    class BadRequest(Client):
        def call(self, payload, identity):
            self.actual_calls += 1
            raise urllib.error.HTTPError('fixture', 400, 'unsupported think', None, None)
    client = BadRequest()
    result = run_pilot(manifest, MODEL, tmp_path / 'http400', client, Counter(), module_for)
    assert result['status'] == 'incomplete'
    assert result['tasks'][0]['failure_kind'] == 'http_error'
    assert all(row['status'] == 'not_attempted_model_stopped' for row in result['tasks'][1:])
    assert all(row['status'] == 'not_attempted_model_stopped' for row in result['engineering_probes'])
    assert client.actual_calls == 1


def test_reasoning_failure_preserves_diagnostic_and_stops(manifest, tmp_path):
    class Thinking(Client):
        def call(self, payload, identity):
            self.actual_calls += 1
            write_json(self.output / 'received' / f'{identity}.json', {
                'request': payload, 'wire_request': {'think': False},
                'response_status': 'final_invalid_response',
                'raw_response': {'message': {'thinking': 'Some reasoning', 'content': ''},
                                 'eval_count': payload['max_tokens'], 'prompt_eval_count': 70,
                                 'done_reason': 'length'},
                'error': 'Unexpected reasoning channel with reasoning_effort=none'})
            raise RuntimeError('Unexpected reasoning channel with reasoning_effort=none')
    client = Thinking()
    result = run_pilot(manifest, MODEL, tmp_path / 'thinking', client, Counter(), module_for)
    assert result['status'] == 'incomplete'
    audit = result['tasks'][0]['audit']
    assert audit['reasoning_channel_responses'] == 1
    assert audit['uncommitted_received_responses'] == 1
    assert audit['empty_content_responses'] == 1
    assert audit['completion_reaches_budget_responses'] == 1
    assert audit['rows'][0]['wire_think'] is False
    assert client.actual_calls == 1


def test_context_failure_keeps_other_tasks(manifest, tmp_path):
    # Patch only the counter used by one actual task; no model/data reduction.
    class OnceLong(Counter):
        def __init__(self):
            self.first = True
        def count(self, text):
            if self.first:
                self.first = False
                return 10000
            return 1
    client = Client()
    result = run_pilot(manifest, MODEL, tmp_path / 'overflow', client, OnceLong(), module_for)
    assert result['status'] == 'incomplete'
    assert result['tasks'][0]['failure_kind'] == 'context_blocked'
    assert all(row['status'] == 'complete' for row in result['tasks'][1:])
    assert len(result['tasks']) == 13


def test_unexpected_programming_error_is_not_swallowed(manifest, tmp_path):
    class Broken(Client):
        def call(self, payload, identity):
            raise RuntimeError('unexpected broken implementation')
    with pytest.raises(RuntimeError, match='unexpected broken implementation'):
        run_pilot(manifest, MODEL, tmp_path / 'bug', Broken(), Counter(), module_for)
    assert known_failure(ValueError('bad code')) is None


def test_empty_length_and_inline_think_are_reported_not_reinterpreted():
    record = {'request': {'max_tokens': 32}, 'response': '', 'score': {'valid': False},
              'raw_response': {'message': {'content': '<think>analysis</think>'},
                               'prompt_eval_count': 11, 'eval_count': 32, 'done_reason': 'length'}}
    d = diagnostic(record, Path('record.json'), True)
    assert d['inline_think_markup'] is True
    assert d['reasoning_channel_present'] is False
    assert d['length_finish'] is True
    assert d['extracted_response_empty'] is True
    assert d['parsed_valid'] is False


def test_retry_usage_counts_every_preserved_attempt(tmp_path):
    for index, tokens in enumerate((10, 12), 1):
        write_json(tmp_path / 'received_attempts' / 'x' / f'attempt_{index}.json', {
            'request': {'max_tokens': 32},
            'raw_response': {'message': {'content': 'answer'},
                             'prompt_eval_count': 40, 'eval_count': tokens, 'done_reason': 'stop'},
            'attempt': {'attempt': index, 'elapsed_seconds': 0.5}})
    result = audit_output(tmp_path)
    assert result['transport_attempt_records'] == 2
    assert result['prompt_tokens_across_all_preserved_attempts'] == 80
    assert result['completion_tokens_across_all_preserved_attempts'] == 22
    assert result['transport_seconds_across_all_preserved_attempts'] == 1
