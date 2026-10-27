"""Lossless CVLT compact presentation checks, original 176-step source only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017 import compact_cvlt as C, memory
from scripts.experiments.human13_20261017.audit_cvlt_contexts import SCENARIOS

ROOT = Path(__file__).resolve().parents[4]
MATERIAL = ROOT / 'results/human13_20261017/prepared_54_cvlt/materials/cvlt_word_list.json'


@pytest.fixture(scope='module')
def episode():
    bundle = json.loads(MATERIAL.read_text())
    assert bundle['status'] == 'ready'
    assert len(bundle['episodes']) == 1
    episode = bundle['episodes'][0]
    assert len(episode['steps']) == 176
    return episode


def record(step, response):
    return {'step': step, 'response': response, 'score': memory.score_step(step, response)}


def reconstruct(row):
    content = ('Word: ' if row['word_wrapper'] else '') + row['body']
    if row['prefix']:
        content = row['prefix'] + '\n' + content
    return content + '\n' + row['format_rule']


@pytest.mark.parametrize('scenario', list(SCENARIOS))
def test_all_history_prompts_answers_and_current_requests_are_preserved(episode, scenario):
    reply = SCENARIOS[scenario]
    history = []
    for step in episode['steps']:
        before = deepcopy((episode, step, history))
        rendered = C.messages(episode, step, history)
        assert rendered[0] == {'role': 'system', 'content': episode['system']}
        assert len(rendered) == 2
        assert rendered[-1]['content'].endswith('Current request:\n' + step['prompt'])
        ledger = C.history_ledger(history)
        assert [reconstruct(row) for row in ledger] == [row['step']['prompt'] for row in history]
        assert [row['response'] for row in ledger] == [row['response'] for row in history]
        cursor = 0
        for row in ledger:
            # Check the exact visible body/response pair and its chronological position.
            literal = row['body'] + '\t' + row['response']
            position = rendered[-1]['content'].index(literal, cursor)
            cursor = position + len(literal)
        assert (episode, step, history) == before
        history.append(record(step, reply(step)))
    assert len(history) == 176


def test_instruction_stage_timing_and_no_future_cues(episode):
    history = []
    observed_prefixes, observed_rules = set(), set()
    for step in episode['steps']:
        rendered = C.messages(episode, step, history)[-1]['content']
        split = C.split_prompt(step)
        # Each visible prefix or rule must have actually occurred by this step.
        if split['prefix']:
            observed_prefixes.add(split['prefix'])
        observed_rules.add(split['format_rule'])
        for prefix in observed_prefixes:
            assert prefix in rendered
        for rule in observed_rules:
            assert rule in rendered
        for future in episode['steps'][len(history) + 1:]:
            future_split = C.split_prompt(future)
            if future_split['prefix'] and future_split['prefix'] not in observed_prefixes:
                assert future_split['prefix'] not in rendered
        if step['id'] == 'A1_study_01':
            assert 'violin' not in rendered and 'wallet' not in rendered
            assert 'furniture' not in rendered and 'YES or NO' not in rendered
            assert 'Anything else?' not in rendered
        if step['id'] == 'short_free':
            assert 'furniture' not in rendered and 'ways of traveling' not in rendered
        if step['id'] == 'short_cued_1':
            assert 'furniture' in rendered and 'ways of traveling' not in rendered
        history.append(record(step, '[]' if step['kind'].startswith('cvlt_recall') else 'Ready'))


def test_gold_scores_metadata_and_future_steps_never_rendered(episode):
    episode = deepcopy(episode)
    position = 40
    history = [record(s, f'  raw-{i}\nanswer\t ') for i, s in enumerate(episode['steps'][:position])]
    step = episode['steps'][position]
    before = C.messages(episode, step, history)
    for item in [step] + [row['step'] for row in history]:
        item['gold'] = 'GOLD_SENTINEL'
        item['meta'] = {'all_a_words': ['SECRET_SENTINEL'], 'phase': 'HIDDEN_SENTINEL'}
    for row in history:
        row['score'] = {'valid': True, 'parsed': 'PARSED_SENTINEL'}
    for future in episode['steps'][position + 1:]:
        future['prompt'] = 'FUTURE_SENTINEL'
        future['gold'] = 'FUTURE_GOLD_SENTINEL'
    assert C.messages(episode, step, history) == before
    assert 'SENTINEL' not in str(before)


def test_invalid_multiline_and_long_answers_are_not_normalized_or_trimmed(episode):
    position = 20
    history = [record(s, 'Ready') for s in episode['steps'][:position]]
    responses = ['', '  INVALID\n"quoted" \\ escaped\t ', '["truck", "truck"]']
    for row, response in zip(history, responses):
        row['response'] = response
    # A reachable long response on a 256-token recall step, not on a 24-token Ready step.
    recall_row = next(row for row in history if row['step']['kind'] == 'cvlt_recall')
    long_response = 'additional recalled words ' * 30
    assert recall_row['step']['max_tokens'] == 256
    recall_row['response'] = long_response
    rendered = C.messages(episode, episode['steps'][position], history)[-1]['content']
    cursor = 0
    for row in C.history_ledger(history):
        literal = row['body'] + '\t' + row['response']
        position = rendered.index(literal, cursor)
        cursor = position + len(literal)
    assert '\t["truck", "truck"]' in rendered
    assert '\t  INVALID\n"quoted" \\ escaped\t ' in rendered
    assert long_response in rendered


def test_missing_reordered_or_future_history_is_rejected(episode):
    step = episode['steps'][3]
    history = [record(s, 'Ready') for s in episode['steps'][:3]]
    for bad in (history[1:], history[::-1], history + [record(episode['steps'][4], 'Ready')]):
        with pytest.raises(ValueError, match='every preceding step'):
            C.messages(episode, step, bad)


def test_presentation_does_not_mutate_any_source_step_or_score(episode):
    source = deepcopy(episode)
    history = []
    scores_before = []
    for step in episode['steps']:
        response = SCENARIOS['target_arrays_both_replies'](step)
        scores_before.append(memory.score_step(step, response))
        C.messages(episode, step, history)
        history.append(record(step, response))
    assert episode == source
    assert [row['score'] for row in history] == scores_before
