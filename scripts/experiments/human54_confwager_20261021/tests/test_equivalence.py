"""Offline checks: the runner rebuilds the formal requests byte for byte, and a condition
changes only what it names. Replies come from the saved formal records; no server."""
import json
from pathlib import Path

import pytest

from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter as TokenCounter
from scripts.experiments.human54_confwager_20261021 import runner as R

ROOT = Path(__file__).resolve().parents[4]
FORMAL = ROOT / 'results/human54_unified_20261018/formal'
MANIFEST = json.loads((ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json').read_text())
SPEC = json.loads((ROOT / 'scripts/experiments/human13_20261017/SPEC.json').read_text())
MODELS = ['qwen2.5:7b', 'gemma2:2b', 'llama3.1:8b']


def bundle(task):
    entry = next(t for t in MANIFEST['tasks'] if t['task'] == task)
    return json.loads(Path(entry['material_file']).read_text())


def formal_sender(model, task):
    replies = {}
    for path in (FORMAL / model.replace(':', '__') / task / 'calls').glob('*.json'):
        record = json.loads(path.read_text())
        replies[json.dumps(record['desired_wire_request'], ensure_ascii=False, sort_keys=True)] = record['raw_response']

    def send(wire):
        key = json.dumps(wire, ensure_ascii=False, sort_keys=True)
        if key not in replies:
            raise AssertionError('Rebuilt request differs from every formal request')
        return replies[key]
    return send, len(replies)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('condition', ['cal_original', 'wag_original'])
def test_original_rebuilds_formal_requests(model, condition, tmp_path):
    task = R.TASK[condition]
    send, n = formal_sender(model, task)
    with TokenCounter(model) as counter:
        counts = R.run(model, condition, bundle(task), SPEC, counter, send, tmp_path)
    assert counts['sent'] == n and counts['not_run'] == 0


def test_logprobs_changes_only_answer_requests(tmp_path):
    model, task = 'qwen2.5:7b', 'confidence_calibration'
    send, _ = formal_sender(model, task)
    sent = []

    def capture(wire):
        sent.append(wire)
        plain = {k: v for k, v in wire.items() if k not in ('logprobs', 'top_logprobs')}
        return send(plain)
    with TokenCounter(model) as counter:
        R.run(model, 'cal_logprobs', bundle(task), SPEC, counter, capture, tmp_path)
    answers = [w for w in sent if w.get('logprobs')]
    assert len(answers) == 80 and all(w['top_logprobs'] == R.TOP_LOGPROBS for w in answers)
    assert len(sent) == 160


def test_wager_labels_change_only_wager_text():
    step = {'kind': 'wager'}
    base = {'messages': [{'role': 'user', 'content': 'Past wager rule: ' + R.WAGER_RULE + '\nx\n' + R.WAGER_PHASE +
                          '\nCurrent request: ' + R.WAGER_RULE}],
            'format': {'type': 'string', 'enum': ['1', '2']}, 'options': {'temperature': 0}}
    for condition, (rule, mapping) in R.LABELS.items():
        out = R.transform(condition, step, base)
        text = out['messages'][0]['content']
        assert R.WAGER_RULE not in text and text.count(rule) == 2 and 'Reply only A or B' in text
        assert out['format']['enum'] == ['A', 'B'] and out['options'] == base['options']
        assert sorted(mapping.values()) == [1, 2]
    assert R.wager_amount('wag_AB', 'A') == 1 and R.wager_amount('wag_BA', 'A') == 2


@pytest.mark.parametrize('condition', ['cal_original', 'wag_original'])
def test_zero_counter_builds_the_same_requests(condition, tmp_path):
    model, task = 'llama2:13b', R.TASK[condition]
    send, n = formal_sender(model, task)
    counts = R.run(model, condition, bundle(task), SPEC, R.ZeroCounter(), send, tmp_path)
    assert counts['sent'] == n and counts['not_run'] == 0
