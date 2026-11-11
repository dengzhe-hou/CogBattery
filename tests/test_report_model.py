"""Report text of scripts.release.report_model. No raw data or model records."""
from scripts.release.report_model import report_text


TASKS = [dict(code='NB', measure='False alarms', kind='value', human=.1, human_sd=.2, model_mean=.3),
         dict(code='WAG', measure='Same wager', kind='constant', constant_models=37, models=54)]
VALUES = {'NB': .5, 'WAG': True}
NOT_RUN = {'NB': 0, 'WAG': 0}


def test_report_shows_difference_and_human_sds():
    text = report_text('m', VALUES, {'NB': 1., 'WAG': 1.}, NOT_RUN, dict(tasks=TASKS))
    assert '+0.400' in text and '+2.00' in text and 'always same wager' in text
    assert 'did not run' not in text


def test_report_flags_unrun_steps():
    text = report_text('m', VALUES, {'NB': .5, 'WAG': 1.}, {'NB': 3, 'WAG': 0}, dict(tasks=TASKS))
    assert '3 planned steps did not run' in text
