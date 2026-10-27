"""Same-slot rule (user decision 2026-10-18, 「不允许」): a record made for another trial is never
reused. Committed copies made under the earlier rule are moved to superseded/ and the slot is
re-made; a later record of the same chain is moved aside only if its rebuilt request changed."""
import json

import pytest

from scripts.experiments.human13_20261017.run import SUPERSEDED_DIR
from scripts.experiments.human13_20261017.tests.test_r7_runner import FakeClient, ok_raw, run


def committed(tmp_path, replies=('"RED"', '"RED"', '"RED"')):
    """Three committed new records of one full-history chain (stroop_task, episode 'ep')."""
    first = run(tmp_path, FakeClient([ok_raw(r) for r in replies]))
    assert first['new_calls'] == 3
    return tmp_path / 'calls'


def make_copy_of_other_slot(tmp_path, identity='ep__s1', origin_episode='other', origin_step='other__s1'):
    """Turn a committed record into a copy of another trial's record (as the earlier rule allowed)."""
    path = tmp_path / 'calls' / f'{identity}.json'
    record = json.loads(path.read_text())
    origin = tmp_path / 'origin' / 'calls' / f'{origin_step}.json'
    origin.parent.mkdir(parents=True, exist_ok=True)
    origin.write_text(json.dumps({'episode_id': origin_episode, 'step': {'id': origin_step},
                                  'wire_request': record['wire_request'], 'raw_response': record['raw_response']}))
    record['execution_origin'] = 'reused'
    record['reuse'] = {'source_path': str(origin), 'source_scope': 'old_run', 'source_step_id': origin_step}
    path.write_text(json.dumps(record))


def test_copy_of_another_slot_is_moved_aside_and_remade_and_identical_reply_keeps_the_rest(tmp_path):
    committed(tmp_path)
    make_copy_of_other_slot(tmp_path)
    client = FakeClient([ok_raw('"RED"')])
    second = run(tmp_path, client)
    assert [identity for identity, _ in client.sent] == ['ep__s1']
    assert second['superseded'] == {'copied_from_another_slot': 1}
    assert second['new_calls'] == 1 and second['resumed_in_output'] == 2
    kept = tmp_path / SUPERSEDED_DIR
    note = json.loads((kept / 'ep__s1__1.note.json').read_text())
    assert note == {'identity': 'ep__s1', 'episode_id': 'ep', 'reason': 'copied_from_another_slot',
                    'moved': {'calls': 'ep__s1__1.calls.json'}}
    assert json.loads((kept / 'ep__s1__1.calls.json').read_text())['execution_origin'] == 'reused'
    assert json.loads((tmp_path / 'calls/ep__s1.json').read_text())['execution_origin'] == 'new'


def test_a_changed_remade_reply_moves_only_the_later_records_whose_request_changed(tmp_path):
    calls = committed(tmp_path)
    # Transport evidence of the original s2 call must move with its record, or the slot would stop.
    for folder in ('dispatch', 'received'):
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / 'ep__s2.json').write_text('{}')
    make_copy_of_other_slot(tmp_path)
    client = FakeClient([ok_raw('"GREEN"'), ok_raw('"RED"'), ok_raw('"RED"')])
    second = run(tmp_path, client)
    assert [identity for identity, _ in client.sent] == ['ep__s1', 'ep__s2', 'ep__s3']
    assert second['superseded'] == {'copied_from_another_slot': 1, 'request_changed_after_earlier_remake': 2}
    assert second['new_calls'] == 3 and second['resumed_in_output'] == 0 and second['not_run_total'] == 0
    kept = tmp_path / SUPERSEDED_DIR
    assert json.loads((kept / 'ep__s2__1.note.json').read_text())['moved'] == {
        'calls': 'ep__s2__1.calls.json', 'dispatch': 'ep__s2__1.dispatch.json', 'received': 'ep__s2__1.received.json'}
    assert not (tmp_path / 'dispatch/ep__s2.json').exists()
    assert '"GREEN"' in json.dumps(json.loads((calls / 'ep__s2.json').read_text())['desired_wire_request'])


def test_superseding_is_remembered_across_resumes(tmp_path):
    committed(tmp_path)
    make_copy_of_other_slot(tmp_path)
    # Run 1 re-makes s1 with a different reply, then stops at the new-call limit before s2.
    first = run(tmp_path, FakeClient([ok_raw('"GREEN"')]), max_new_calls=1)
    assert first['superseded'] == {'copied_from_another_slot': 1, 'request_changed_after_earlier_remake': 1}
    assert first['not_run_total'] == 2
    # Run 2: s1 is now a same-slot record; s3's request changed too, which run 2 knows from superseded/.
    client = FakeClient([ok_raw('"RED"'), ok_raw('"RED"')])
    second = run(tmp_path, client)
    assert [identity for identity, _ in client.sent] == ['ep__s2', 'ep__s3']
    assert second['superseded'] == {'request_changed_after_earlier_remake': 1}
    assert second['resumed_in_output'] == 1 and second['not_run_total'] == 0


def test_a_differing_record_without_an_earlier_remake_still_fails_fast(tmp_path):
    calls = committed(tmp_path)
    record = json.loads((calls / 'ep__s2.json').read_text())
    record['desired_wire_request']['options']['num_predict'] = 999
    (calls / 'ep__s2.json').write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='Committed record differs'):
        run(tmp_path, FakeClient([]))


def test_inspect_only_counts_but_moves_nothing(tmp_path):
    committed(tmp_path)
    make_copy_of_other_slot(tmp_path)
    before = sorted(p.name for p in (tmp_path / 'calls').iterdir())
    result = run(tmp_path, FakeClient([]), inspect_only=True)
    assert result['superseded'] == {'copied_from_another_slot': 1}
    assert not (tmp_path / SUPERSEDED_DIR).exists()
    assert sorted(p.name for p in (tmp_path / 'calls').iterdir()) == before


def test_same_slot_copy_is_resumed_untouched(tmp_path):
    committed(tmp_path)
    make_copy_of_other_slot(tmp_path, origin_episode='ep', origin_step='s1')
    client = FakeClient([])
    second = run(tmp_path, client)
    assert client.sent == [] and second['superseded'] == {} and second['resumed_in_output'] == 3
