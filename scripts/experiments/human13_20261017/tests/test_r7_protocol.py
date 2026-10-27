"""r7 request construction: modes, staged budgets, formats, stop rules, wager text, bytes.

Offline only: read-only r6 materials mapped in memory, hand-written histories from
the scripted fixture replies, and a few saved prior records compared byte for byte.
No model, server or Slurm job is involved.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017 import budgets, compact_wag, protocol
from scripts.experiments.human13_20261017.constraints import (
    ARRAY_PROMPT, ARRAY_SCHEMA, OLD_PROMPT, final_stop, response_schema)
from scripts.experiments.human13_20261017.prepare_r7 import map_bundle
from scripts.experiments.human13_20261017.preflight import fixture_reply
from scripts.experiments.human13_20261017.registry import module_for
from scripts.experiments.human13_20261017.scoring import wrap_module

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
SPEC = json.loads((HERE / 'SPEC.json').read_text())
R6 = ROOT / 'results/human13_20261017/prepared_54_drm_sentence/manifest.json'
NATIVE = ('qwen3:4b', 'deepseek-r1:7b', 'deepseek-r1:14b')
DIRECT = 'qwen2.5:7b'
JSON1 = str(ROOT / 'scripts/experiments/json_array_stop_20261018/backend_support/backend/bin/ollama')
CVLT = str(ROOT / 'scripts/experiments/cvlt_context_controls_20261007/backend/bin/ollama')
CACHEFIX = str(ROOT / 'scripts/experiments/phi3_cache_fix_20261013/build_support/backend_cachefix_p9/bin/ollama')
BUDGET_ORDER = ['model', 'messages', 'stream', 'think', 'format', 'thinking_budget', 'options']
OPTION_ORDER = ['temperature', 'num_predict', 'top_p', 'num_ctx', 'num_thread']


class WordCounter:
    """Stand-in for the vocabulary counter; request bytes do not depend on it."""
    def count(self, text, special=False):
        return len(text.split())


_BUNDLES = {}


def r7_bundle(task):
    if task not in _BUNDLES:
        manifest = json.loads(R6.read_text())
        entry = next(t for t in manifest['tasks'] if t['task'] == task)
        _BUNDLES[task] = map_bundle(json.loads(Path(entry['material_file']).read_text()),
                                    SPEC['inference']['direct_step_budget_map'])
    return _BUNDLES[task]


def with_variant(variant):
    spec = deepcopy(SPEC)
    spec['inference']['answer_reserve']['selected_variant'] = variant
    return spec


def walk(task, model, spec=SPEC, stop=None):
    """Yield (episode, step, history, request) with scripted fixture replies as history."""
    module = wrap_module(module_for(task))
    histories = {}
    for episode in r7_bundle(task)['episodes']:
        history = list(histories[episode['parent']]) if episode.get('parent') else []
        for step in episode['steps']:
            req = protocol.build_request(model, task, episode, step, history, spec, module, WordCounter())
            yield episode, step, list(history), req
            if req.context['history_pairs_dropped']:
                history = history[req.context['history_pairs_dropped']:]
            reply = fixture_reply(step)
            history.append({'step': step, 'response': reply, 'score': module.score_step(step, reply)})
        histories[episode['id']] = history


def first(task, model, kind=None, spec=SPEC):
    for episode, step, history, req in walk(task, model, spec):
        if kind is None or step['kind'] == kind:
            return episode, step, history, req
    raise AssertionError(f'no {kind} step in {task}')


def test_mode_context_and_backend_per_model():
    assert len(SPEC['models']) == 54 and set(NATIVE) <= set(SPEC['models'])
    assert SPEC['inference']['max_transport_attempts'] == 1
    for model in SPEC['models']:
        inf = protocol.model_inference(SPEC, model)
        backend = SPEC['backends'].get(model, SPEC['backends']['default'])
        if model in NATIVE:
            assert inf['native_think'] is True and inf['context_tokens'] == 12288
            assert backend == JSON1 and SPEC['backend_versions'][backend] == '0.18.0-cogarena-budget-json1'
        else:
            assert inf['native_think'] is False and inf['context_tokens'] == 4096
            assert backend == (CACHEFIX if model.startswith('phi3:') else CVLT)
            assert SPEC['backend_versions'][backend] == '0.18.0'


@pytest.mark.parametrize('model', NATIVE)
def test_native_staged_budgets(model):
    _, _, _, ready = first('operation_span', model, 'ready')
    assert list(ready.wire) == BUDGET_ORDER and list(ready.wire['options']) == OPTION_ORDER
    assert ready.wire['think'] is True and ready.wire['options']['num_ctx'] == 12288
    assert (ready.wire['options']['num_predict'], ready.wire['thinking_budget']) == (128, 96)
    _, _, _, recall = first('operation_span', model, 'ospan_recall')
    assert (recall.wire['options']['num_predict'], recall.wire['thinking_budget']) == (128, 96)
    assert recall.wire['format'] == ARRAY_SCHEMA and recall.wire['messages'][-1]['content'] == ARRAY_PROMPT
    _, _, _, training = first('post_decision_wagering', model, 'wag_training')
    assert (training.wire['options']['num_predict'], training.wire['thinking_budget']) == (8192, 8160)
    assert training.wire['format'] == {'type': 'string', 'pattern': '^[A-Z]+$'}  # no Qwen legacy exception
    _, _, _, fb = first('false_belief', model)
    assert list(fb.wire) == ['model', 'messages', 'stream', 'think', 'options']
    assert fb.wire['think'] is True and fb.wire['options']['num_predict'] == 8192
    assert 'format' not in fb.wire and 'thinking_budget' not in fb.wire and 'stop' not in fb.wire['options']
    for wire in (ready.wire, recall.wire, training.wire, fb.wire):
        assert 'stop' not in wire['options']


@pytest.mark.parametrize('model', NATIVE)
def test_reserve_caps_are_pending_until_the_user_selects_a_variant(model):
    # The live SPEC carries the user's choice (grammar_ws_w22, USER_DECISIONS.md); the pending gate is
    # tested on a copy without a selected variant.
    assert SPEC['inference']['answer_reserve']['selected_variant'] == 'grammar_ws_w22'
    with pytest.raises(budgets.PendingUserDecision):
        first('stroop', model, spec=with_variant(None))
    for variant, table in SPEC['inference']['answer_reserve']['values'].items():
        for task in SPEC['inference']['native_reserve_tasks']:
            _, _, _, req = first(task, model, spec=with_variant(variant))
            assert req.wire['options']['num_predict'] == 8192
            assert req.wire['thinking_budget'] == 8192 - table[model][task]
            assert list(req.wire) == BUDGET_ORDER


def test_direct_budgets_formats_and_stop_rules():
    for kind in ('ready', 'ospan_math', 'ospan_recall'):
        _, _, _, req = first('operation_span', DIRECT, kind)
        assert req.wire['options']['num_predict'] == 128 and 'thinking_budget' not in req.wire
        assert req.wire['think'] is False and req.wire['options']['num_ctx'] == 4096
        if kind == 'ospan_recall':
            assert req.wire['format'] == ARRAY_SCHEMA and 'stop' not in req.wire['options']
            assert req.wire['messages'][-1]['content'] == ARRAY_PROMPT
        else:
            assert req.wire['options']['stop'] == ['\n']
    _, step, _, stroop = first('stroop', DIRECT)
    assert step['max_tokens'] == 16 and stroop.wire['options']['num_predict'] == 16
    assert stroop.wire['format'] == {'type': 'string', 'enum': ['RED', 'GREEN', 'BLUE']}
    assert list(stroop.wire) == ['model', 'messages', 'stream', 'think', 'options', 'format']
    assert list(stroop.wire['options']) == OPTION_ORDER + ['stop']
    _, _, _, recall = first('cvlt_word_list', DIRECT, 'cvlt_recall')
    assert recall.wire['options']['num_predict'] == 512 and recall.wire['format']['type'] == 'array'
    assert 'stop' not in recall.wire['options']
    _, _, _, fb = first('false_belief', DIRECT)
    assert fb.wire['options']['num_predict'] == 1024 and 'format' not in fb.wire and 'stop' not in fb.wire['options']
    assert final_stop({'type': 'string', 'enum': ['A']}, True) == [] and final_stop(None, False) == []


@pytest.mark.parametrize('model', [DIRECT, 'deepseek-r1:7b'])
def test_every_request_in_all_13_paradigms_obeys_the_contract(model):
    spec = with_variant('grammar_ws_w22')
    native = model in NATIVE
    for task in SPEC['tasks']:
        n = 0
        for episode, step, history, req in walk(task, model, spec):
            n += 1
            wire = req.wire
            assert wire['messages'][-1]['role'] == 'user' and wire['model'] == model
            assert body_ok(req)
            if 'thinking_budget' in wire:
                assert native and 'format' in wire and task != 'false_belief'
            elif native:
                assert task == 'false_belief'
            stop = wire['options'].get('stop')
            if stop:
                assert not native and wire['format']['type'] == 'string' and stop == ['\n']
            if task == 'operation_span':
                assert wire['options']['num_predict'] == 128
            for message in wire['messages'][:-1]:
                assert message['content'] != ARRAY_PROMPT  # history keeps the old instruction
        assert n == len([s for e in r7_bundle(task)['episodes'] for s in e['steps']])


def body_ok(req):
    return req.body == json.dumps(req.wire, ensure_ascii=False)


def test_wager_revision_applies_to_every_model_and_only_to_the_current_turn():
    module = module_for('post_decision_wagering')
    for model in (DIRECT, 'qwen3:4b'):
        seen = set()
        for episode, step, history, req in walk('post_decision_wagering', model, SPEC):
            original = module.messages(episode, step, history)[0]['content']
            revised = req.wire['messages'][0]['content']
            if step['kind'] == 'wager':
                assert revised == original
            elif not history:
                phase, target, rule = original.split('\n')
                assert revised == '\n'.join((phase, rule, target))  # training step 1, no history block
            else:
                head = original.split(protocol.HISTORY_END)[0]
                assert revised.startswith(head + protocol.HISTORY_END) and revised != original
                if step['kind'] == 'wag_answer':
                    assert protocol.CLASSIFICATION_CLARIFICATION in revised
            seen.add((step['kind'], bool(history)))
            if len(seen) == 4:
                break
        assert {('wag_training', False), ('wag_training', True), ('wag_answer', True), ('wager', True)} <= seen


def test_schemas_equal_the_format_repair_source_functions():
    from scripts.experiments.human13_format_repair_20261017.constraints import response_schema as source_schema
    for task in SPEC['tasks']:
        for episode, step, history, req in walk(task, DIRECT):
            current = req.wire['messages'][-1]['content']
            if task == 'operation_span' and step['kind'] == 'ospan_recall':
                continue
            if task == 'post_decision_wagering':
                current = None
            assert response_schema(step, history=history, current_prompt=current) == \
                source_schema(step, history=history, current_prompt=current)


def _source(path):
    return json.loads((ROOT / path).read_text())


def test_saved_prior_requests_rebuild_byte_identical():
    # same_stimulus job_r2 false belief (direct, no format, no stop)
    row = _source('results/same_stimulus_20261011/formal/job_r2/qwen2.5__7b/calls/fb_session_A__A1.json')
    _, _, _, req = first('false_belief', DIRECT)
    assert req.body == json.dumps(row['wire_request'], ensure_ascii=False)
    # uniform OSPAN prefix, direct Ready with stop
    row = _source('results/human13_ospan_uniform_20261017/qwen2.5__7b/calls/ospan_v1__ospan_v1__0001.json')
    _, _, _, req = first('operation_span', DIRECT)
    assert req.body == json.dumps(row['wire_request'], ensure_ascii=False)
    # choice-wave DRM rating: identical except the later-added direct stop rule
    row = _source('results/human13_choice_wave_20261017/runs/qwen2.5__7b/drm_false_memory/calls/'
                  'drm_sentences_no_warning__study_01.json')
    _, _, _, req = first('drm_false_memory', DIRECT)
    relaxed = deepcopy(req.wire)
    assert relaxed['options'].pop('stop') == ['\n']
    assert json.dumps(relaxed, ensure_ascii=False) == json.dumps(row['wire_request'], ensure_ascii=False)


def test_scoring_equals_the_format_repair_source_on_saved_records():
    from scripts.experiments.human13_format_repair_20261017.scoring import score_record as source_score
    from scripts.experiments.human13_repair_20261017.registry import module_for as source_module
    from scripts.experiments.human13_20261017.scoring import score_record
    n = 0
    for model in ('zephyr__7b', 'qwen3__4b', 'internlm2__7b'):
        for task in ('post_decision_wagering', 'operation_span', 'cvlt_word_list', 'drm_false_memory'):
            folder = ROOT / 'results/human13_format_repair_20261017/runs' / model / task / 'calls'
            for path in sorted(folder.glob('*.json')):
                row = json.loads(path.read_text())
                assert score_record(module_for(task), row) == source_score(source_module(task), row)
                n += 1
    assert n > 100
