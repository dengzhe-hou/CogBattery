from types import SimpleNamespace

from scripts.experiments.human13_20261017.common import score_record, step_budget
from scripts.experiments.human13_20261017.run import model_spec


def test_truncated_parse_does_not_count_or_drive_next_action():
    def parse(step, text):
        return {'valid': bool(text), 'parsed': text or None, 'is_correct': bool(text),
                'history_response': text or 'INVALID'}
    module = SimpleNamespace(score_step=parse)
    row = {'step': {}, 'response': 'MATCH', 'finish_reason': 'length', 'protocol_valid': False}
    score = score_record(module, row)
    assert score['parsed_valid'] is True
    assert score['valid'] is False and score['behavior_eligible'] is False
    assert score['history_response'] == 'INVALID'
    row.update(finish_reason='stop', protocol_valid=True)
    assert score_record(module, row)['behavior_eligible'] is True
    row['synthetic'] = True
    result = score_record(module, row)
    assert result['valid'] and result['response_complete']
    assert result['behavior_eligible'] is False


def test_native_budget_is_total_and_mode_is_per_model():
    spec = {'inference': {'contexts': {'r': 12288}, 'context_tokens': 4096,
                         'native_think_models': ['r'], 'native_total_output_tokens': 8192,
                         'max_tokens': 1024, 'native_think': False}}
    native = model_spec(spec, 'r')['inference']
    direct = model_spec(spec, 'd')['inference']
    assert step_budget({'max_tokens': 16}, native) == 8192
    assert step_budget({'max_tokens': 16}, direct) == 16
    assert native['context_tokens'] == 12288 and direct['context_tokens'] == 4096
    assert spec['inference']['native_think'] is False
