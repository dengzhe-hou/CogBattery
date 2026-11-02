"""Leaderboard rules of scripts.release.report_model. No raw data or model records."""
import json

import pytest

from scripts.release.report_model import add_to_board, report_text


TASKS = [dict(code='NB', measure='False alarms', kind='value', human=.1, human_sd=.2, model_mean=.3),
         dict(code='WAG', measure='Same wager', kind='constant', constant_models=37, models=54)]
VALUES = {'NB': .5, 'WAG': True}
NOT_RUN = {'NB': 0, 'WAG': 0}


@pytest.fixture
def board(tmp_path):
    path = tmp_path / 'leaderboard.json'
    path.write_text(json.dumps(dict(tasks=TASKS, models=[dict(model='paper:1b', serving='paper', values=VALUES)])))
    return path


@pytest.fixture
def scored(tmp_path):
    (tmp_path / 'submission_header.json').write_text(json.dumps(dict(header=dict(backend='openai'))))
    return tmp_path


def test_report_shows_difference_and_human_sds():
    text = report_text('m', VALUES, {'NB': 1., 'WAG': 1.}, NOT_RUN, dict(tasks=TASKS))
    assert '+0.400' in text and '+2.00' in text and 'always same wager' in text
    assert 'did not run' not in text


def test_report_flags_unrun_steps():
    text = report_text('m', VALUES, {'NB': .5, 'WAG': 1.}, {'NB': 3, 'WAG': 0}, dict(tasks=TASKS))
    assert '3 planned steps did not run' in text


def test_add_uses_submitted_backend(board, scored):
    add_to_board(board, scored, 'mine:1b', VALUES, NOT_RUN, 'mine', 1.)
    row = json.loads(board.read_text())['models'][-1]
    assert row['serving'] == 'openai' and row['family'] == 'mine' and row['values'] == VALUES


def test_add_requires_replay(board, tmp_path):
    (tmp_path / 'submission_header.json').unlink(missing_ok=True)
    with pytest.raises(SystemExit, match='replayed'):
        add_to_board(board, tmp_path, 'mine:1b', VALUES, NOT_RUN, 'mine', 1.)


def test_add_refuses_incomplete_or_undefined(board, scored):
    with pytest.raises(SystemExit, match='did not run'):
        add_to_board(board, scored, 'mine:1b', VALUES, {'NB': 1, 'WAG': 0}, 'mine', 1.)
    with pytest.raises(SystemExit, match='undefined'):
        add_to_board(board, scored, 'mine:1b', {'NB': None, 'WAG': True}, NOT_RUN, 'mine', 1.)


def test_add_protects_paper_rows_and_existing_submissions(board, scored):
    with pytest.raises(SystemExit, match='paper row'):
        add_to_board(board, scored, 'paper:1b', VALUES, NOT_RUN, 'mine', 1., replace=True)
    add_to_board(board, scored, 'mine:1b', VALUES, NOT_RUN, 'mine', 1.)
    with pytest.raises(SystemExit, match='already'):
        add_to_board(board, scored, 'mine:1b', VALUES, NOT_RUN, 'mine', 1.)
    add_to_board(board, scored, 'mine:1b', {'NB': .6, 'WAG': False}, NOT_RUN, 'mine', 1., replace=True)
    rows = [m for m in json.loads(board.read_text())['models'] if m['model'] == 'mine:1b']
    assert len(rows) == 1 and rows[0]['values']['NB'] == .6
