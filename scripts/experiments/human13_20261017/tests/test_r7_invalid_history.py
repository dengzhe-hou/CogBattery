"""Decision (a) on the real r7 materials: how a technically invalid reply appears in later turns.

Pins SPEC inference.failure_policy.invalid_reply_presentation (verifier round 2): the
record keeps the reply, but each paradigm's established presentation decides what later
turns show. Scripted replies only; no model, no server. A word count stands in for the
tokenizer (the scripted histories stay far below 4096).
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from scripts.experiments.human13_20261017.client import parse_reply
from scripts.experiments.human13_20261017.common import ROOT
from scripts.experiments.human13_20261017.registry import module_for
from scripts.experiments.human13_20261017.run import run_task

HERE = Path(__file__).resolve().parents[1]
SPEC = json.loads((HERE / 'SPEC.json').read_text())
PRESENTATION = SPEC['inference']['failure_policy']['invalid_reply_presentation']
MANIFEST = json.loads((ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json').read_text())
MARK = 'ZQX7 INV'
VERBATIM = set(PRESENTATION['response_text_verbatim'])
LABEL = set(PRESENTATION['label_INVALID_not_text']) - {'status'}
NEVER = set(PRESENTATION['past_replies_never_shown'])


class Words:
    def count(self, text, special=False):
        return len(text.split())


class Scripted:
    """Step `invalid_at` gets a cut-off reply containing MARK; every other step a filler reply."""

    def __init__(self, invalid_at):
        self.invalid_at, self.sent = invalid_at, []
        self.serving_checked, self.transport_starts = True, 0
        self.backend, self.model_digest = {'path': 'x', 'version': 'y'}, 'd'

    def serving(self, path):
        pass

    def send(self, wire, identity, folder):
        self.sent.append((identity, deepcopy(wire)))
        mark = len(self.sent) == self.invalid_at
        raw = {'message': {'role': 'assistant', 'content': '"' + MARK if mark else 'filler reply'},
               'done': True, 'done_reason': 'length' if mark else 'stop', 'prompt_eval_count': 10, 'eval_count': 2}
        return {**parse_reply(wire, raw), 'raw_response': raw, 'wire_request': wire,
                'latency_seconds': 0.0, 'transport_attempts': 1}


def bundle(task):
    entry = next(t for t in MANIFEST['tasks'] if t['task'] == task)
    return json.loads(Path(entry['material_file']).read_text())


def run(task, invalid_at, steps, tmp_path):
    client = Scripted(invalid_at)
    run_task('qwen2.5:7b', task, bundle(task), SPEC, tmp_path, module=module_for(task), counter=Words(),
             cache=None, client_factory=lambda: client, scope='pilot', max_new_calls=steps)
    record = json.loads((tmp_path / 'calls' / f'{client.sent[invalid_at - 1][0]}.json').read_text())
    assert record['technical_valid'] is False and MARK in record['response']  # stored unchanged
    return [json.dumps(wire, ensure_ascii=False) for _, wire in client.sent[invalid_at:]], client.sent


def test_every_paradigm_has_exactly_one_presentation_class():
    assert VERBATIM | LABEL | NEVER == set(SPEC['tasks'])
    assert not (VERBATIM & LABEL or VERBATIM & NEVER or LABEL & NEVER)
    assert LABEL == {'n_back', 'go_nogo'}


@pytest.mark.parametrize('task', SPEC['tasks'])
def test_invalid_reply_presentation_matches_spec(task, tmp_path):
    later, sent = run(task, 1, 12, tmp_path)
    if task in VERBATIM:
        assert MARK in later[0]
    else:
        assert not any(MARK in text for text in later)
    if task in LABEL:
        meta = bundle(task)['episodes'][0]['steps'][0]['meta']
        stimulus = meta['letter'] if task == 'n_back' else meta['color']
        assert f"{meta['trial']}:{stimulus}:INVALID" in json.loads(later[0])['messages'][-1]['content']
    if task in NEVER:
        assert 'INVALID' not in later[0]
        assert not any(m['role'] == 'assistant' for m in json.loads(later[0])['messages'])


def test_recall_replies_end_their_unit_in_digit_and_operation_span():
    """SPEC: unit_context shows only later steps of the same unit; every recall closes its unit."""
    for task, kind in (('digit_span', 'digit_recall'), ('operation_span', 'ospan_recall')):
        for episode in bundle(task)['episodes']:
            steps = episode['steps']
            for i, step in enumerate(steps):
                if step['kind'] == kind:
                    assert all(later['meta'].get('unit') != step['meta'].get('unit') for later in steps[i + 1:])


def test_feedback_steps_treat_an_invalid_reply_as_wrong(tmp_path):
    counts = {}
    for task in SPEC['tasks']:
        steps = [s for e in bundle(task)['episodes'] for s in e['steps']]
        n = sum(bool(s['meta'].get('feedback')) for s in steps)
        if n:
            counts[task] = n
    assert counts == {'digit_span': 84, 'operation_span': 129, 'stroop': 72, 'flanker': 72, 'go_nogo': 30}
    for task, n in counts.items():
        assert f'{task} {n}' in PRESENTATION['feedback_steps']
    for task in counts:
        first = bundle(task)['episodes'][0]['steps']
        index = next(i for i, s in enumerate(first) if s['meta'].get('feedback'))
        later, _ = run(task, index + 1, index + 2, tmp_path / task)
        current = json.loads(later[0])['messages'][-1]['content']
        step = first[index]
        if step['meta']['feedback'] == 'span':
            assert current.startswith(f"Previous recall: 0 of {len(step['gold'])} positions correct.\n")
        else:
            assert current.startswith('Previous response: Incorrect.\n')
        if task == 'go_nogo':
            assert re.search(r'\n1:[A-Z]+:INVALID\n', current)
