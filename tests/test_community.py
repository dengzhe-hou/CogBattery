"""Community leaderboard rules of scripts.release.community. No raw data, materials, or model records."""
import hashlib
import json

import pytest

from scripts.release.community import build, check_unique, make_row, read_metadata

VALUES = {'NB': .5, 'WAG': True}
NOT_RUN = {'NB': 0, 'WAG': 0}
META = {'family': 'mine', 'params_b': 1}


def write_submission(folder, sub_id, meta=META, pack=b'packed'):
    (folder / f'{sub_id}.json').write_text(json.dumps(meta))
    (folder / f'{sub_id}.cogbattery.jsonl.gz').write_bytes(pack)
    return hashlib.sha256(pack).hexdigest()


def test_row_keeps_backend_and_values():
    row = make_row('mine', META, 'abc', 'mine:1b', 'openai', VALUES, NOT_RUN)
    assert row['serving'] == 'openai' and row['family'] == 'mine' and row['values'] == VALUES
    assert row['params_b'] == 1.0 and row['pack_sha256'] == 'abc'


def test_row_refuses_incomplete_or_undefined():
    with pytest.raises(SystemExit, match='did not run'):
        make_row('mine', META, 'abc', 'mine:1b', 'ollama', VALUES, {'NB': 1, 'WAG': 0})
    with pytest.raises(SystemExit, match='undefined'):
        make_row('mine', META, 'abc', 'mine:1b', 'ollama', {'NB': None, 'WAG': True}, NOT_RUN)


def test_same_model_and_backend_twice_is_refused():
    row = make_row('a', META, 'x', 'mine:1b', 'ollama', VALUES, NOT_RUN)
    check_unique([row, {**row, 'id': 'b', 'serving': 'openai'}])
    with pytest.raises(SystemExit, match='both submit'):
        check_unique([row, {**row, 'id': 'b'}])


def test_metadata_needs_family_params_and_pack(tmp_path):
    write_submission(tmp_path, 'ok')
    assert read_metadata(tmp_path / 'ok.json')[1].name == 'ok.cogbattery.jsonl.gz'
    write_submission(tmp_path, 'nofamily', meta={'params_b': 1})
    with pytest.raises(SystemExit, match='needs family'):
        read_metadata(tmp_path / 'nofamily.json')
    write_submission(tmp_path, 'zero', meta={'family': 'x', 'params_b': 0})
    with pytest.raises(SystemExit, match='positive'):
        read_metadata(tmp_path / 'zero.json')
    (tmp_path / 'nopack.json').write_text(json.dumps(META))
    with pytest.raises(SystemExit, match='no nopack.cogbattery.jsonl.gz'):
        read_metadata(tmp_path / 'nopack.json')


def test_unchanged_pack_is_reused_and_changed_pack_needs_replay(tmp_path):
    subs = tmp_path / 'submissions'
    subs.mkdir()
    board, community = tmp_path / 'leaderboard.json', tmp_path / 'community.json'
    board.write_text(json.dumps(dict(version='v', scoring='s', tasks=[], models=[])))
    digest = write_submission(subs, 'mine', meta={'family': 'renamed', 'params_b': 2})
    row = make_row('mine', META, digest, 'mine:1b', 'ollama', VALUES, NOT_RUN)
    community.write_text(json.dumps(dict(models=[row])))
    build(None, subs, board, community)
    kept = json.loads(community.read_text())['models']
    assert kept == [{**row, 'family': 'renamed', 'params_b': 2.0}]
    write_submission(subs, 'mine', pack=b'changed')
    with pytest.raises(SystemExit, match='needs --materials'):
        build(None, subs, board, community)


def test_removed_submission_leaves_the_table(tmp_path):
    subs = tmp_path / 'submissions'
    subs.mkdir()
    board, community = tmp_path / 'leaderboard.json', tmp_path / 'community.json'
    board.write_text(json.dumps(dict(version='v', scoring='s', tasks=[], models=[])))
    row = make_row('gone', META, 'x', 'mine:1b', 'ollama', VALUES, NOT_RUN)
    community.write_text(json.dumps(dict(models=[row])))
    build(None, subs, board, community)
    assert json.loads(community.read_text())['models'] == []
