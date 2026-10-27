"""User decision 2026-10-18 (「继续」): a native model that ends its turn without opening the format phase
replied; the reply is kept as technically invalid and the chain continues. A reply saved in received/
without a committed record is re-read under the current rules on resume, never resent."""
import json

import pytest

from scripts.experiments.human13_20261017.client import HardResponseError, parse_reply
from scripts.experiments.human13_20261017.tests.test_r7_runner import FakeClient, ok_raw, run

ENUM = {'type': 'string', 'enum': ['Jake', 'Tammy', 'Olivia', 'Mike', '1', '2', '3', '4']}


def budget_wire():
    return {'model': 'deepseek-r1:7b', 'messages': [{'role': 'user', 'content': 'Who said it?'}], 'stream': False,
            'think': True, 'format': ENUM, 'thinking_budget': 8164,
            'options': {'temperature': 0, 'num_predict': 8192, 'top_p': 1.0, 'num_ctx': 12288, 'num_thread': 1}}


def skipped_raw(origin='none', unconstrained='Tammy\n\n'):
    # Shape of results/human54_unified_20261018/formal/deepseek-r1__7b/source_monitoring/received/
    # sm_indirect_formA__q_row72.json (job 26122).
    accounting = {'total_budget': 8192, 'thinking_budget': 8164, 'boundary_origin': origin,
                  'injected_control_tokens': 0, 'budget_used': 4, 'raw_thinking': unconstrained,
                  'final_budget': 0, 'thinking_eval_count': 4, 'final_eval_count': 0,
                  'thinking_prompt_eval_count': 1805, 'final_prompt_eval_count': 0,
                  'thinking_done_reason': 'stop', 'thinking_matched_stop': '</think>',
                  'unconstrained_content': unconstrained}
    return {'model': 'deepseek-r1:7b', 'message': {'role': 'assistant', 'content': ''}, 'done': True,
            'done_reason': 'stop', 'prompt_eval_count': 1805, 'eval_count': 4, 'phase_accounting': accounting}


def test_skipped_thinking_reply_is_invalid_and_shows_the_models_own_words():
    parsed = parse_reply(budget_wire(), skipped_raw())
    assert parsed['technical_valid'] is False
    assert 'skipped_thinking_unconstrained_answer' in parsed['protocol_errors']
    assert 'no_final_phase' in parsed['protocol_errors']
    assert parsed['response'] == 'Tammy'
    assert parsed['budget_verdict']['unconstrained_content'] == 'Tammy\n\n'


def test_text_outside_the_format_phase_after_a_real_boundary_still_breaks_the_contract():
    raw = skipped_raw(origin='natural')
    with pytest.raises(HardResponseError):
        parse_reply(budget_wire(), raw)


def test_a_saved_reply_without_a_record_is_committed_on_resume_without_resending(tmp_path):
    run(tmp_path, FakeClient([ok_raw('"RED"'), ok_raw('"GREEN"'), ok_raw('"RED"')]))
    calls = tmp_path / 'calls'
    two = json.loads((calls / 'ep__s2.json').read_text())
    (tmp_path / 'received').mkdir(exist_ok=True)
    (tmp_path / 'received/ep__s2.json').write_text(json.dumps(
        {'wire_request': two['wire_request'], 'raw_response': two['raw_response'], 'latency_seconds': 0.5,
         'received_at': 't'}))
    (tmp_path / 'dispatch').mkdir(exist_ok=True)
    (tmp_path / 'dispatch/ep__s2.json').write_text(json.dumps({'slurm_job_id': '111'}))
    (tmp_path / 'chain_stop.json').write_text(json.dumps({'identity': 'ep__s2', 'reason': 'hard_response_error'}))
    (tmp_path / 'hard_errors').mkdir()
    (tmp_path / 'hard_errors/ep__s2.json').write_text('{}')
    (calls / 'ep__s2.json').unlink()
    (calls / 'ep__s3.json').unlink()
    client = FakeClient([ok_raw('"RED"')])
    second = run(tmp_path, client)
    assert [identity for identity, _ in client.sent] == ['ep__s3']  # s2 was not resent
    assert second['recovered_from_received'] == 1 and second['new_calls'] == 2 and second['not_run_total'] == 0
    recovered = json.loads((calls / 'ep__s2.json').read_text())
    assert recovered['recovered_from_received'] == 't' and recovered['response'] == 'GREEN'
    assert recovered['recording_slurm_job_id'] == '111'  # the job that sent it, not the one that committed it
    assert 'committed_by_slurm_job_id' in recovered and 'sent_by_slurm_job_id' not in recovered
    assert not (tmp_path / 'chain_stop.json').exists() and (tmp_path / 'chain_stop_resolved__ep__s2.json').exists()
    assert not (tmp_path / 'hard_errors').exists() and (tmp_path / 'hard_errors_resolved/ep__s2.json').exists()


def test_a_saved_reply_that_still_breaks_the_contract_keeps_the_cell_stopped(tmp_path):
    run(tmp_path, FakeClient([ok_raw('"RED"'), ok_raw('"GREEN"'), ok_raw('"RED"')]))
    calls = tmp_path / 'calls'
    two = json.loads((calls / 'ep__s2.json').read_text())
    broken = dict(two['raw_response'], done=False)
    (tmp_path / 'received').mkdir(exist_ok=True)
    (tmp_path / 'received/ep__s2.json').write_text(json.dumps({'wire_request': two['wire_request'],
                                                               'raw_response': broken}))
    (calls / 'ep__s2.json').unlink()
    (calls / 'ep__s3.json').unlink()
    client = FakeClient([])
    second = run(tmp_path, client)
    assert client.sent == [] and second['stop_reason'] == 'prior_attempt_preserved'


def test_history_text_is_cut_at_the_models_own_thinking_marker():
    # q_row21 shape (job 26126): the answer first, then a <think> block until the cap.
    long_tail = 'Mike\n\n<think>\nAlright, let me try to figure this out step by step.\n' + 'x ' * 50
    raw = skipped_raw(unconstrained=long_tail)
    raw['done_reason'] = 'length'
    parsed = parse_reply(budget_wire(), raw)
    assert parsed['response'] == 'Mike' and parsed['technical_valid'] is False
    assert parse_reply(budget_wire(), skipped_raw(unconstrained='Tammy\n\n'))['response'] == 'Tammy'
    assert parse_reply(budget_wire(), skipped_raw(unconstrained='Olivia </think> more'))['response'] == 'Olivia'


def test_a_committed_history_text_from_an_older_rule_is_recommitted_from_received_without_resending(tmp_path):
    from scripts.experiments.human13_20261017.run import SUPERSEDED_DIR
    run(tmp_path, FakeClient([ok_raw('"RED"'), ok_raw('"GREEN"'), ok_raw('"RED"')]))
    calls = tmp_path / 'calls'
    two = json.loads((calls / 'ep__s2.json').read_text())
    (tmp_path / 'received').mkdir(exist_ok=True)
    (tmp_path / 'received/ep__s2.json').write_text(json.dumps(
        {'wire_request': two['wire_request'], 'raw_response': two['raw_response'], 'received_at': 't'}))
    (tmp_path / 'dispatch').mkdir(exist_ok=True)
    (tmp_path / 'dispatch/ep__s2.json').write_text(json.dumps({'slurm_job_id': '111'}))
    # Stored under an older rule: a long history text the current rule would not derive.
    two['protocol_errors'] = ['skipped_thinking_unconstrained_answer']
    two['response'] = 'GREEN and then a very long text the old rule kept'
    (calls / 'ep__s2.json').write_text(json.dumps(two))
    (calls / 'ep__s3.json').unlink()  # its request would follow the corrected history
    client = FakeClient([ok_raw('"RED"')])
    second = run(tmp_path, client)
    assert [identity for identity, _ in client.sent] == ['ep__s3']
    assert second['superseded'] == {'history_text_rule_changed': 1} and second['recovered_from_received'] == 1
    assert json.loads((calls / 'ep__s2.json').read_text())['response'] == 'GREEN'
    note = json.loads((tmp_path / SUPERSEDED_DIR / 'ep__s2__1.note.json').read_text())
    assert note['moved'] == {'calls': 'ep__s2__1.calls.json'}  # dispatch and received stay for the re-commit
    assert (tmp_path / 'received/ep__s2.json').exists()
