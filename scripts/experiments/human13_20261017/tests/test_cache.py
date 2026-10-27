"""Exact-input cache: what is reused, what is not, in which order (hand-written records only)."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments.human13_20261017 import cache as C
from scripts.experiments.human13_20261017.common import default_messages
from scripts.experiments.human13_20261017.run import run_task

JSON1 = C.PRIVATE_BINARIES['0.18.0-cogarena-budget-json1']
BUDGET1 = C.PRIVATE_BINARIES['0.18.0-cogarena-budget1']
DIGEST = 'a' * 64
SLOT = ('ep', 's1')  # the default source path below is /x/calls/ep__s1.json


def direct_wire(content='Trial: RED. Reply.', stop=True, num_predict=16):
    wire = {'model': 'm:1b', 'messages': [{'role': 'user', 'content': content}], 'stream': False, 'think': False,
            'options': {'temperature': 0, 'num_predict': num_predict, 'top_p': 1.0, 'num_ctx': 4096, 'num_thread': 1},
            'format': {'type': 'string', 'enum': ['RED', 'GREEN', 'BLUE']}}
    if stop:
        wire['options']['stop'] = ['\n']
    return wire


def native_wire(thinking_budget=8165):
    wire = {'model': 'r:7b', 'messages': [{'role': 'user', 'content': 'Trial: RED. Reply.'}], 'stream': False,
            'think': True, 'format': {'type': 'string', 'enum': ['RED', 'GREEN', 'BLUE']},
            'thinking_budget': thinking_budget,
            'options': {'temperature': 0, 'num_predict': 8192, 'top_p': 1.0, 'num_ctx': 12288, 'num_thread': 1}}
    if thinking_budget is None:
        wire.pop('thinking_budget')
    return wire


def raw(content='"RED"', done_reason='stop', eval_count=3):
    return {'message': {'role': 'assistant', 'content': content}, 'done': True, 'done_reason': done_reason,
            'prompt_eval_count': 20, 'eval_count': eval_count}


def source(wire, reply, *, path='/x/calls/ep__s1.json', priority=0, backend=C.CVLT, version='0.18.0',
           digest=DIGEST, evidence='serving_file', row_extra=None):
    row = {'model': wire['model'], 'wire_request': wire, 'raw_response': reply, 'response': 'RED', **(row_extra or {})}
    identity = {'backend_path': backend, 'backend_version': version, 'model_digest': digest, 'job': '1',
                'evidence': evidence}
    return C.Source(Path(path), row, 'test_scope', priority, identity)


def index(*sources, backend=C.CVLT, version='0.18.0', model='m:1b'):
    return C.CacheIndex(model, backend, version, DIGEST, list(sources))


def test_exact_bytes_are_reused():
    wire = direct_wire()
    hits = index(source(deepcopy(wire), raw())).candidates(wire, SLOT)
    assert [kind for _, kind in hits] == ['exact_bytes']


@pytest.mark.parametrize('change', ['num_predict', 'format', 'think', 'messages', 'num_ctx'])
def test_any_other_request_difference_is_not_reused(change):
    wire, old = direct_wire(), direct_wire()
    if change == 'num_predict':
        old['options']['num_predict'] = 8  # an old non-binding cap no longer counts as compatible
    elif change == 'format':
        old.pop('format')  # unconstrained legacy record
    elif change == 'think':
        old['think'] = True
    elif change == 'messages':
        old['messages'][0]['content'] += ' '
    else:
        old['options']['num_ctx'] = 12288
    assert index(source(old, raw())).candidates(wire, SLOT) == []


def test_native_records_without_the_new_thinking_cap_are_not_reused():
    new = native_wire(8165)
    old_8192 = native_wire(None)  # the old natural-closure 8192 protocol
    other_cap = native_wire(8164)
    cache = index(source(old_8192, raw()), source(other_cap, raw(), path='/x/calls/b.json'),
                  backend=JSON1, version='0.18.0-cogarena-budget-json1', model='r:7b')
    assert cache.candidates(new, SLOT) == []


def test_backend_digest_and_identity_must_match():
    wire = native_wire()
    budget1 = source(deepcopy(wire), raw(), backend=BUDGET1, version='0.18.0-cogarena-budget1')
    other_file = source(deepcopy(wire), raw(), backend=JSON1, version='0.18.0-cogarena-budget-json1',
                        digest='b' * 64, path='/x/calls/c.json')
    unknown = source(deepcopy(wire), raw(), backend=None, version=None, digest=None, evidence='missing',
                     path='/x/calls/d.json')
    cache = index(budget1, other_file, unknown, backend=JSON1, version='0.18.0-cogarena-budget-json1', model='r:7b')
    assert cache.candidates(wire, SLOT) == []
    assert cache.rejections == {'backend_differs': 1, 'model_digest_differs': 1, 'source_identity_unverified': 1}


def test_non_terminal_source_reply_is_not_reused():
    wire = direct_wire()
    reply = raw()
    reply['done'] = False
    cache = index(source(deepcopy(wire), reply))
    assert cache.candidates(wire, SLOT) == [] and cache.rejections['source_reply_not_terminal'] == 1


def test_only_relaxation_is_a_stop_absent_from_the_source_reply():
    wire = direct_wire(stop=True)
    no_stop = direct_wire(stop=False)
    assert [k for _, k in index(source(deepcopy(no_stop), raw('"RED"'))).candidates(wire, SLOT)] == ['stop_absent_in_reply']
    cache = index(source(deepcopy(no_stop), raw('"RED"\n')))
    assert cache.candidates(wire, SLOT) == [] and cache.rejections['stop_string_present_in_source_reply'] == 1
    # The relaxation never applies the other way round.
    assert index(source(deepcopy(wire), raw())).candidates(no_stop, SLOT) == []


def test_only_same_slot_sources_are_accepted_then_the_earliest_root():
    wire = direct_wire()
    later_same = source(deepcopy(wire), raw('"RED"'), path='/late/calls/ep__s1.json', priority=5,
                        row_extra={'episode_id': 'ep', 'step': {'id': 's1'}})
    early_other = source(deepcopy(wire), raw('"BLUE"'), path='/early/calls/other__s1.json', priority=0,
                         row_extra={'episode_id': 'other', 'step': {'id': 's1'}})
    cache = index(early_other, later_same)
    hits = cache.candidates(wire, ('ep', 's1'))
    # Same-slot rule (user decision 2026-10-18): the other trial's record is not acceptable at all.
    assert [s.path for s, _ in hits] == [Path('/late/calls/ep__s1.json')]
    assert cache.rejections['source_made_for_another_slot'] == 1
    assert index(early_other, later_same).candidates(wire, ('third', 's1')) == []
    earlier_same = source(deepcopy(wire), raw('"GREEN"'), path='/early/calls/ep__s1.json', priority=0,
                          row_extra={'episode_id': 'ep', 'step': {'id': 's1'}})
    hits = index(later_same, earlier_same).candidates(wire, ('ep', 's1'))
    assert [s.path for s, _ in hits] == [Path('/early/calls/ep__s1.json'), Path('/late/calls/ep__s1.json')]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_scan_indexes_origin_records_only_and_reads_identity_from_serving_files(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    wire = direct_wire()
    write(root / 'm__1b/stroop/calls/a.json', {'model': 'm:1b', 'wire_request': wire, 'raw_response': raw(),
                                               'execution_origin': 'new', 'slurm_job_id': '7'})
    write(root / 'm__1b/stroop/calls/b.json', {'model': 'm:1b', 'wire_request': wire, 'raw_response': raw(),
                                               'execution_origin': 'reused', 'slurm_job_id': '8'})
    write(root / 'm__1b/stroop/calls/c.json', {'model': 'm:1b', 'wire_request': wire, 'raw_response': raw()})
    write(root / 'm__1b/stroop/serving.json', {'slurm_job_id': '7', 'ollama_version': 'ollama version is 0.18.0\n',
                                               'native_ps': {'models': [{'name': 'm:1b', 'digest': DIGEST}]}})
    write(root / 'other__2b/stroop/calls/z.json', {'model': 'other:2b'})  # never read
    found, stats = C.scan(['m:1b'], roots=(), extra_roots=[(root, 'unit', 'human13')])
    paths = sorted(s.path.name for s in found['m:1b'])
    assert paths == ['a.json', 'c.json']
    by = {s.path.name: s.identity for s in found['m:1b']}
    assert by['a.json']['backend_path'] == C.CVLT and by['a.json']['model_digest'] == DIGEST
    # c.json has no job id and the folder serving file names job 7 for this model.
    assert by['c.json']['job'] == '7' and by['c.json']['evidence'] == 'serving_file'
    assert stats[('unit', 'skipped_reused_copy')] == 1


def test_backend_rules_follow_the_recorded_run_specs():
    assert C.backend_for('cvlt_for_all', 'phi3:14b', '0.18.0') == C.CVLT  # same_stimulus job_r2
    assert C.backend_for('cachefix_for_all', 'phi3:14b', '0.18.0') == C.CACHEFIX
    assert C.backend_for('human13', 'phi3:3.8b', '0.18.0') == C.CACHEFIX
    assert C.backend_for('human13', 'qwen2.5:7b', '0.18.0') == C.CVLT
    assert C.backend_for('human13', 'qwen3:4b', '0.18.0-cogarena-constrained1').endswith(
        'human13_constrained_20261017/backend_support/backend/bin/ollama')
    assert C.backend_for('cvlt_for_all', 'x:1b', '0.18.0-cogarena-budget1') is None


class Counter:
    def count(self, text, special=False):
        return len(text.split())


def stroop_task(n=2):
    steps = [{'id': f's{i}', 'prompt': f'Ink colour of item {i}?', 'kind': 'stroop_choice', 'gold': 'RED',
              'meta': {}, 'max_tokens': 16} for i in range(1, n + 1)]
    return {'status': 'ready', 'blockers': [], 'episodes': [
        {'id': 'ep', 'task': 'stroop', 'condition': 'c', 'history_mode': 'full', 'steps': steps}]}


def module():
    def score(step, response):
        return {'parsed': response, 'valid': response in ('RED', 'GREEN', 'BLUE'), 'is_correct': response == 'RED'}
    return SimpleNamespace(messages=default_messages, score_step=score)


def spec():
    return {'revision': 'test', 'inference': {
        'temperature': 0, 'native_top_p': 1.0, 'context_tokens': 4096, 'contexts': {}, 'native_think_models': [],
        'template_margin_tokens': 128, 'per_message_margin_tokens': 8, 'server_threads': 1,
        'task_total_override': {}, 'native_total_output_tokens': 8192, 'native_thinking_budget': {},
        'native_reserve_tasks': [], 'request_timeout_seconds': 300, 'max_transport_attempts': 1}}


def wire_for(step_prompts, model='m:1b'):
    messages = []
    for i, (prompt, answer) in enumerate(step_prompts):
        messages.append({'role': 'user', 'content': prompt})
        if answer is not None:
            messages.append({'role': 'assistant', 'content': answer})
    wire = {'model': model, 'messages': messages, 'stream': False, 'think': False,
            'options': {'temperature': 0, 'num_predict': 16, 'top_p': 1.0, 'num_ctx': 4096, 'num_thread': 1,
                        'stop': ['\n']},
            'format': {'type': 'string', 'enum': ['RED', 'GREEN', 'BLUE']}}
    return wire


class NoClient:
    def __call__(self):
        raise AssertionError('a cached step must not create a client or send a request')


def test_reused_records_become_formal_with_provenance_and_invalid_stays_invalid(tmp_path):
    first = wire_for([('Ink colour of item 1?', None)])
    # The prior reply to step 1 was cut at the cap: invalid, and it must stay invalid.
    first_raw = raw('"RE', done_reason='length', eval_count=16)
    second = wire_for([('Ink colour of item 1?', '"RE'), ('Ink colour of item 2?', None)])
    cache = index(source(first, first_raw, path='/p/calls/ep__s1.json'),
                  source(second, raw('"GREEN"'), path='/p/calls/ep__s2.json'))
    result = run_task('m:1b', 'stroop', stroop_task(), spec(), tmp_path, module=module(), counter=Counter(),
                      cache=cache, client_factory=NoClient(), scope='formal')
    assert result['new_calls'] == 0 and result['reused_from_cache'] == 2
    assert result['reused_valid'] == 1 and result['reused_invalid_kept_invalid'] == 1
    assert result['reused_slots'] == ['ep__s1', 'ep__s2']  # the pilot criterion compares these per slot
    one = json.loads((tmp_path / 'calls/ep__s1.json').read_text())
    two = json.loads((tmp_path / 'calls/ep__s2.json').read_text())
    assert one['technical_valid'] is False and one['score']['valid'] is False
    assert one['protocol_errors'] == ['length_finish', 'invalid_format_json']
    assert one['execution_origin'] == 'reused' and one['formal_record'] is True
    assert one['research_results_eligible'] is True and one['reuse']['match_kind'] == 'exact_bytes'
    assert one['reuse']['source_path'] == '/p/calls/ep__s1.json' and one['reuse']['source_backend_path'] == C.CVLT
    assert two['technical_valid'] is True and two['response'] == 'GREEN'
    # Decision (a): the invalid final text stayed in the history of step 2.
    assert two['desired_wire_request']['messages'][1] == {'role': 'assistant', 'content': '"RE'}


def test_legacy_false_belief_records_are_matched_to_their_own_trial_by_their_fields():
    # same_stimulus job_r2 / phi3 p9 rows: no episode or step id; isolated orders share one folder.
    wire = direct_wire()
    legacy = {'paradigm': 'strachan_fb', 'condition': 'isolated', 'list': 'A', 'item_id': 'A1'}
    isolated = source(deepcopy(wire), raw(), path='/job_r2/m__1b/calls/fb_isolated__A1.json', row_extra=legacy)
    assert C.row_slot(isolated.row) == ('fb_isolated_A', 'fb_isolated_A__A1')
    assert index(isolated).candidates(wire, ('fb_isolated_A', 'fb_isolated_A__A1'))
    assert index(isolated).candidates(wire, ('fb_isolated_B', 'fb_isolated_B__A1')) == []
    assert index(isolated).candidates(wire, ('fb_session_A', 'fb_session_A__A1')) == []
    no_system = {**legacy, 'condition': 'isolated_no_system', 'list': 'B', 'item_id': 'B3'}
    other = source(deepcopy(wire), raw(), path='/job_r2/m__1b/calls/fb_isolated_no_system__B3.json',
                   row_extra=no_system)
    assert index(other).candidates(wire, ('fb_isolated_no_system_B', 'fb_isolated_no_system_B__B3'))
    session = {**legacy, 'condition': 'session', 'list': 'B', 'item_id': 'B4'}
    in_session = source(deepcopy(wire), raw(), path='/job_r2/m__1b/calls/fb_session_B__B4.json', row_extra=session)
    assert index(in_session).candidates(wire, ('fb_session_B', 'fb_session_B__B4'))
