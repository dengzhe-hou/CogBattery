"""r7 runtime behaviour with synthetic clients: no repeats, no revival, one transport, refusal.

Hand-written fixtures; the real Client is exercised with urllib patched, so no
server, model or network is touched.
"""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

from scripts.experiments.human13_20261017 import client as client_module
from scripts.experiments.human13_20261017 import run as run_module
from scripts.experiments.human13_20261017.client import Client, TransportFailure, parse_reply
from scripts.experiments.human13_20261017.run import NotAuthorized, authorize, pilot_limits, run_task
from scripts.experiments.human13_20261017.tests.test_cache import Counter, module, spec, stroop_task

HERE = Path(__file__).resolve().parents[1]
SPEC = json.loads((HERE / 'SPEC.json').read_text())


def ok_raw(content='"RED"', done_reason='stop'):
    return {'message': {'role': 'assistant', 'content': content}, 'done': True, 'done_reason': done_reason,
            'prompt_eval_count': 20, 'eval_count': 2}


class FakeClient:
    """Scripted replies; records every transport. Mirrors Client.send's return shape."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.serving_checked = False
        self.backend = {'path': '/b/ollama', 'version': '0.18.0'}
        self.model_digest = 'd' * 64
        self.transport_starts = 0

    def serving(self, path):
        self.serving_checked = True

    def send(self, wire, identity, folder):
        self.transport_starts += 1
        self.sent.append((identity, deepcopy(wire)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {**parse_reply(wire, reply), 'raw_response': reply, 'wire_request': wire,
                'latency_seconds': 0.01, 'transport_attempts': 1}


def run(tmp_path, client, n=3, **kwargs):
    return run_task('m:1b', 'stroop', stroop_task(n), kwargs.pop('spec', spec()), tmp_path, module=module(),
                    counter=kwargs.pop('counter', Counter()), cache=None, client_factory=lambda: client, **kwargs)


def test_full_run_and_pilot_are_refused_by_the_r7_spec():
    # The live SPEC's authorization flags change when the parent authorizes the pilot or the formal run
    # (USER_DECISIONS.md); the refusal logic is tested on a copy with every run flag closed.
    closed = deepcopy(SPEC)
    for flag in ('inference_authorized', 'formal_authorized', 'pilot_authorized'):
        closed['authorization'][flag] = False
    with pytest.raises(NotAuthorized, match='Full evaluation is not authorized'):
        authorize(closed, 'formal')
    with pytest.raises(NotAuthorized, match='pilot is not authorized'):
        authorize(closed, 'pilot')
    partial = deepcopy(closed)
    partial['authorization']['inference_authorized'] = True  # formal also needs formal_authorized
    with pytest.raises(NotAuthorized):
        authorize(partial, 'formal')


def test_cli_refuses_before_any_output_cache_or_client(tmp_path, monkeypatch):
    from scripts.experiments.human13_20261017.prepare_r7 import prepare_r7
    manifest = prepare_r7(tmp_path / 'prepared')
    out = tmp_path / 'formal'
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(sys, 'argv', ['run', '--manifest', str(tmp_path / 'prepared/manifest.json'),
                                      '--model', 'qwen2.5:7b', '--task', 'stroop', '--output', str(out)])
    monkeypatch.setattr(run_module, 'expected_identity', lambda *a: (_ for _ in ()).throw(AssertionError('too late')))
    closed = deepcopy(run_module.live_spec())
    for flag in ('inference_authorized', 'formal_authorized', 'pilot_authorized'):
        closed['authorization'][flag] = False
    monkeypatch.setattr(run_module, 'live_spec', lambda: closed)
    with pytest.raises(NotAuthorized):
        run_module.main()
    assert not out.exists() and manifest['planned_logical_slots'] == 54 * 6789


def test_valid_calls_are_never_repeated_on_resume(tmp_path):
    client = FakeClient([ok_raw(), ok_raw('"GREEN"'), ok_raw('"BLUE"')])
    first = run(tmp_path, client)
    assert first['new_calls'] == 3 and first['technical_valid'] == 3 and len(client.sent) == 3
    again = FakeClient([])
    second = run(tmp_path, again)
    assert again.sent == [] and second['resumed_in_output'] == 3 and second['new_calls'] == 0


def test_failed_records_are_kept_and_never_revived(tmp_path):
    client = FakeClient([ok_raw(), ok_raw('"GRE', 'length'), ok_raw('"BLUE"')])
    first = run(tmp_path, client)
    assert first['technical_invalid'] == 1 and first['invalid_reasons'] == {'length_finish': 1, 'invalid_format_json': 1}
    again = FakeClient([])
    second = run(tmp_path, again)
    assert again.sent == [] and second['technical_invalid'] == 1
    saved = json.loads((tmp_path / 'calls/ep__s2.json').read_text())
    assert saved['technical_valid'] is False and saved['score']['valid'] is False


def test_invalid_reply_stays_in_history_and_the_chain_continues(tmp_path):
    client = FakeClient([ok_raw(), ok_raw('"GRE', 'length'), ok_raw('"BLUE"')])
    result = run(tmp_path, client)
    assert result['status'] == 'complete' and result['committed_or_reusable'] == 3
    third = client.sent[2][1]['messages']
    assert {'role': 'assistant', 'content': '"GRE'} in third  # raw final text, unchanged
    assert result['valid_coverage'] == pytest.approx(2 / 3)


def test_no_reply_stops_the_chain_without_any_retry(tmp_path):
    client = FakeClient([ok_raw(), TransportFailure('URLError: refused')])
    result = run(tmp_path, client)
    assert len(client.sent) == 2  # one attempt for step 2, nothing for step 3
    assert result['not_run'] == {'no_reply_transport_failure': 2} and result['status'] == 'no_reply_transport_failure'
    assert json.loads((tmp_path / 'chain_stop.json').read_text())['identity'] == 'ep__s2'


def test_context_overflow_stops_without_deleting_history(tmp_path):
    class Growing:
        def count(self, text, special=False):
            return 5000 if 'item 3' in text else 10
    client = FakeClient([ok_raw(), ok_raw()])
    result = run(tmp_path, client, counter=Growing())
    assert len(client.sent) == 2 and result['not_run'] == {'context_overflow': 1}


def test_time_guard_and_new_call_limit_send_nothing_more(tmp_path):
    client = FakeClient([])
    result = run(tmp_path / 'a', client, deadline=0)
    assert client.sent == [] and result['status'] == 'time_checkpoint' and result['not_run_total'] == 3
    client = FakeClient([ok_raw()])
    result = run(tmp_path / 'b', client, max_new_calls=1)
    assert len(client.sent) == 1 and result['not_run'] == {'checkpoint_new_call_limit': 2}


def test_inspect_only_sends_and_writes_nothing(tmp_path):
    client = FakeClient([])
    result = run_task('m:1b', 'stroop', stroop_task(3), spec(), tmp_path / 'out', module=module(), counter=Counter(),
                      cache=None, client_factory=lambda: client, inspect_only=True)
    assert client.sent == [] and not (tmp_path / 'out').exists()
    assert result['not_run'] == {'needs_new_call': 1, 'depends_on_missing_reply': 2}


def make_client(monkeypatch, tmp_path, attempts=1):
    monkeypatch.setenv('OLLAMA_HOST', '127.0.0.1:1')
    view = {'inference': {**spec()['inference'], 'max_transport_attempts': attempts}}
    c = Client('m:1b', view, backend_path='/b/ollama', backend_version='0.18.0', model_digest='d' * 64,
               server_log=tmp_path / 'log')
    c.serving_checked = True
    return c


def test_real_client_sends_once_and_keeps_raw_evidence(monkeypatch, tmp_path):
    calls = []

    def refused(request, timeout):
        calls.append(request.data)
        raise urllib.error.URLError('refused')
    monkeypatch.setattr(client_module.urllib.request, 'urlopen', refused)
    c = make_client(monkeypatch, tmp_path)
    wire = {'model': 'm:1b', 'messages': [{'role': 'user', 'content': 'x'}], 'stream': False, 'think': False,
            'options': {'temperature': 0, 'num_predict': 16, 'top_p': 1.0, 'num_ctx': 4096, 'num_thread': 1}}
    with pytest.raises(TransportFailure):
        c.send(wire, 'ep__s1', tmp_path)
    assert len(calls) == 1 and calls[0] == json.dumps(wire, ensure_ascii=False).encode()
    assert (tmp_path / 'dispatch/ep__s1.json').exists() and (tmp_path / 'received/ep__s1.json').exists()
    with pytest.raises(client_module.PriorAttemptPreserved):
        c.send(wire, 'ep__s1', tmp_path)
    assert len(calls) == 1
    # The runner also refuses to resend after a crash: it stops at the preserved dispatch.
    result = run(tmp_path, FakeClient([]), n=1)
    assert result['not_run'] == {'prior_attempt_preserved': 1}


def test_real_client_success_path(monkeypatch, tmp_path):
    def reply(request, timeout):
        assert timeout == 300
        return io.BytesIO(json.dumps(ok_raw('"RED"')).encode())
    monkeypatch.setattr(client_module.urllib.request, 'urlopen', reply)
    c = make_client(monkeypatch, tmp_path)
    wire = {'model': 'm:1b', 'messages': [{'role': 'user', 'content': 'x'}], 'stream': False, 'think': False,
            'options': {'temperature': 0, 'num_predict': 16, 'top_p': 1.0, 'num_ctx': 4096, 'num_thread': 1},
            'format': {'type': 'string', 'enum': ['RED']}}
    result = c.send(wire, 'ep__s1', tmp_path)
    assert result['response'] == 'RED' and result['technical_valid'] and result['transport_attempts'] == 1
    assert c.transport_starts == 1


def test_more_than_one_transport_attempt_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match='exactly once'):
        make_client(monkeypatch, tmp_path, attempts=3)


def budget_raw(t=40, f=3, origin='natural', content='"RED"'):
    phase = {'total_budget': 8192, 'thinking_budget': 8165, 'thinking_eval_count': t,
             'injected_control_tokens': 1 if origin == 'forced' else 0, 'final_eval_count': f,
             'budget_used': t + f + (1 if origin == 'forced' else 0), 'boundary_origin': origin,
             'thinking_done_reason': 'length' if origin == 'forced' else 'stop',
             'thinking_matched_stop': None if origin == 'forced' else '</think>',
             'injected_control_text': '</think>' if origin == 'forced' else None,
             'final_budget': 8192 - t - (1 if origin == 'forced' else 0), 'initial_prompt': 'P', 'raw_thinking': 'T',
             'final_prompt': 'PT</think>', 'thinking_prompt_eval_count': 20, 'final_prompt_eval_count': 25}
    return {'message': {'role': 'assistant', 'content': content, 'thinking': 'T'}, 'done': True,
            'done_reason': 'stop', 'prompt_eval_count': 20, 'eval_count': t + f, 'phase_accounting': phase}


def test_budgeted_native_replies_use_the_staged_accounting_check():
    wire = {'model': 'r:7b', 'messages': [{'role': 'user', 'content': 'x'}], 'stream': False, 'think': True,
            'format': {'type': 'string', 'enum': ['RED']}, 'thinking_budget': 8165,
            'options': {'temperature': 0, 'num_predict': 8192, 'top_p': 1.0, 'num_ctx': 12288, 'num_thread': 1}}
    assert parse_reply(wire, budget_raw())['technical_valid'] is True
    forced = parse_reply(wire, budget_raw(t=8165, f=3, origin='forced'))
    assert forced['technical_valid'] and forced['budget_verdict']['injected_control_tokens'] == 1
    with pytest.raises(client_module.HardResponseError, match='wrong applied budgets'):
        bad = budget_raw()
        bad['phase_accounting']['thinking_budget'] = 8160
        parse_reply(wire, bad)
    trailing = parse_reply(wire, budget_raw(content='"RE'))
    assert trailing['technical_valid'] is False and 'invalid_format_json' in trailing['protocol_errors']


def test_protocol_identity_ignores_authorization_and_the_reserve_choice_only():
    from scripts.experiments.human13_20261017.run import protocol_spec
    changed = deepcopy(SPEC)
    changed['authorization']['pilot_authorized'] = True
    changed['inference']['answer_reserve']['selected_variant'] = 'grammar_ws_w22'
    changed['inference']['answer_reserve']['status'] = 'decided'
    changed['inference']['answer_reserve']['leaning_not_decided'] = 'grammar_ws_w4'
    assert protocol_spec(changed) == protocol_spec(SPEC)
    reserve_values = deepcopy(changed)
    first = next(iter(reserve_values['inference']['answer_reserve']['values'].values()))
    model = next(iter(first))
    first[model][next(iter(first[model]))] += 1
    assert protocol_spec(reserve_values) != protocol_spec(SPEC)
    changed['inference']['native_thinking_budget']['post_decision_wagering'] = 8000
    assert protocol_spec(changed) != protocol_spec(SPEC)


PLAN = SPEC['authorization']['pilot_plan']


def plan_spec(root, authorized=False):
    view = deepcopy(SPEC)
    view['authorization']['pilot_plan'] = {**PLAN, 'output_root': str(root)}  # ROOT / absolute = absolute
    view['authorization']['pilot_authorized'] = authorized
    return view


def test_pilot_plan_is_prewritten_small_and_skips_the_pending_capped_cells():
    caps = PLAN['max_new_calls']
    assert PLAN['max_new_calls_total'] == sum(sum(v.values()) for v in caps.values()) == 647
    assert set(caps) <= set(SPEC['models']) and all(set(v) <= set(SPEC['tasks']) for v in caps.values())
    assert PLAN['output_root'] == 'results/human54_unified_20261018/pilot'
    for model in SPEC['inference']['native_think_models']:  # proposal 1 pending
        assert set(caps[model]) == {'operation_span', 'post_decision_wagering', 'false_belief'}
    assert set(caps['qwen2.5:7b']) == set(caps['phi3:3.8b']) == set(SPEC['tasks'])


def test_pilot_limits_refuse_anything_outside_the_plan(tmp_path):
    root = tmp_path / 'pilot'
    view = plan_spec(root)
    with pytest.raises(NotAuthorized, match='outside the pilot plan'):
        pilot_limits(view, 'llama3.1:8b', None, root)
    with pytest.raises(NotAuthorized, match='outside the pilot plan'):
        pilot_limits(view, 'qwen3:4b', ['stroop'], root)
    with pytest.raises(NotAuthorized, match='writes only to'):
        pilot_limits(view, 'qwen3:4b', None, tmp_path / 'elsewhere')
    no_plan = deepcopy(view)
    no_plan['authorization'].pop('pilot_plan')
    with pytest.raises(NotAuthorized, match='No pre-written pilot plan'):
        pilot_limits(no_plan, 'qwen3:4b', None, root)
    limits = pilot_limits(view, 'qwen3:4b', None, root)
    assert list(limits) == ['operation_span', 'false_belief', 'post_decision_wagering']  # SPEC order
    assert limits == {'operation_span': 16, 'false_belief': 3, 'post_decision_wagering': 4}


def test_pilot_cap_counts_new_calls_across_resumes(tmp_path):
    root = tmp_path / 'pilot'
    calls = root / 'qwen3__4b' / 'false_belief' / 'calls'
    calls.mkdir(parents=True)
    for name, origin in (('a', 'new'), ('b', 'new'), ('c', 'reused')):
        (calls / f'{name}.json').write_text(json.dumps({'execution_origin': origin}))
    assert pilot_limits(plan_spec(root), 'qwen3:4b', ['false_belief'], root) == {'false_belief': 1}
    for name in ('d', 'e'):
        (calls / f'{name}.json').write_text(json.dumps({'execution_origin': 'new'}))
    assert pilot_limits(plan_spec(root), 'qwen3:4b', ['false_belief'], root) == {'false_belief': 0}


def test_cli_pilot_stays_inside_the_plan_before_any_output_cache_or_client(tmp_path, monkeypatch):
    from scripts.experiments.human13_20261017.prepare_r7 import prepare_r7
    prepare_r7(tmp_path / 'prepared')
    root = tmp_path / 'pilot'
    monkeypatch.setattr(run_module, 'live_spec', lambda: plan_spec(root, authorized=True))
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(run_module, 'expected_identity',
                        lambda *a: (_ for _ in ()).throw(AssertionError('passed the pilot gate')))
    base = ['run', '--manifest', str(tmp_path / 'prepared/manifest.json'), '--scope', 'pilot']
    for extra, error, match in ((['--model', 'llama3.1:8b', '--output', str(root)], NotAuthorized, 'outside'),
                                (['--model', 'qwen3:4b', '--task', 'stroop', '--output', str(root)], NotAuthorized, 'outside'),
                                (['--model', 'qwen3:4b', '--output', str(tmp_path / 'x')], NotAuthorized, 'writes only'),
                                (['--model', 'qwen3:4b', '--output', str(root)], AssertionError, 'passed the pilot gate')):
        monkeypatch.setattr(sys, 'argv', base + extra)
        with pytest.raises(error, match=match):
            run_module.main()
    assert not root.exists() and not (tmp_path / 'x').exists()
