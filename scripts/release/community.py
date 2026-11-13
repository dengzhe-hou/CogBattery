"""Score the submissions in submissions/ and write the community leaderboard. CPU only, no model calls.

    python -m scripts.release.community --materials results/paper_materials/manifest.json \
        --summary summary.md

Each submission is a packed run (`<id>.cogbattery.jsonl.gz`, from `pack_submission pack`) and a
metadata file (`<id>.json`) with the model's family and served parameters in billions, or null when the
size is not disclosed. Every pack is replayed through the paper's run loop with `pack_submission score`,
so each request must match the one the submitter sent, and the replayed records are scored with the
paper's scorers. Only complete runs with all 13 key measures defined are accepted.

The result goes to docs/data/community.json, kept apart from the paper's leaderboard.json. A pack whose
SHA-256 equals the one recorded there is not replayed again, since replay takes a few minutes per model.
The summary lists each submission's key measures next to the human references, for a pull request check.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts.experiments.human13_20261017.run import slug
from scripts.release.pack_submission import score
from scripts.release.report_model import model_values, report_text

ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = ROOT / 'submissions'
BOARD = ROOT / 'docs/data/leaderboard.json'
COMMUNITY = ROOT / 'docs/data/community.json'
FIELDS = {'family', 'params_b'}


def read_metadata(path):
    meta = json.loads(path.read_text())
    missing = FIELDS - set(meta)
    if missing:
        raise SystemExit(f'{path.name} needs {", ".join(sorted(missing))}.')
    if not isinstance(meta['family'], str) or not meta['family']:
        raise SystemExit(f'{path.name}: family must be a non-empty string.')
    size = meta['params_b']
    if size is not None and (not isinstance(size, (int, float)) or size <= 0):
        raise SystemExit(f'{path.name}: params_b must be a positive number of billions, or null if not disclosed.')
    pack = path.with_name(f'{path.stem}.cogbattery.jsonl.gz')
    if not pack.is_file():
        raise SystemExit(f'{path.name} has no {pack.name} next to it.')
    return meta, pack


def params(meta):
    """Served parameters in billions, or None for a model whose size is not disclosed."""
    return None if meta['params_b'] is None else float(meta['params_b'])


def make_row(sub_id, meta, digest, model, backend, values, not_run):
    missing = sum(not_run.values())
    if missing:
        raise SystemExit(f'{sub_id}: {missing} planned steps did not run, so the submission is incomplete.')
    undefined = [code for code, value in values.items() if value is None]
    if undefined:
        raise SystemExit(f'{sub_id}: a row needs all 13 measures, and these are undefined: {", ".join(undefined)}.')
    return dict(id=sub_id, model=model, family=meta['family'], params_b=params(meta),
                native_thinking=False, serving=backend, pack_sha256=digest, values=values)


def check_unique(rows):
    seen = {}
    for row in rows:
        key = (row['model'], row['serving'])
        if key in seen:
            raise SystemExit(f'{row["id"]} and {seen[key]} both submit {row["model"]} through {row["serving"]}. '
                             'Update the earlier submission instead.')
        seen[key] = row['id']


def score_pack(pack, materials, work):
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        score(pack, materials, tmp)
        folder = next(Path(tmp).iterdir())
        backend = json.loads((folder / 'submission_header.json').read_text())['header']['backend']
        model, values, coverage, not_run = model_values(folder, materials)
    if folder.name != slug(model):
        raise SystemExit(f'{pack.name}: replayed folder {folder.name} does not match model {model}.')
    return model, backend, values, coverage, not_run


def build(materials, submissions=SUBMISSIONS, board=BOARD, community=COMMUNITY, summary=None, work=None):
    reference = json.loads(Path(board).read_text())
    old = {}
    if Path(community).is_file():
        old = {r['id']: r for r in json.loads(Path(community).read_text())['models']}
    rows, notes = [], []
    for path in sorted(Path(submissions).glob('*.json')):
        meta, pack = read_metadata(path)
        digest = hashlib.sha256(pack.read_bytes()).hexdigest()
        cached = old.get(path.stem)
        if cached and cached['pack_sha256'] == digest:
            rows.append({**cached, 'family': meta['family'], 'params_b': params(meta)})
            continue
        if materials is None:
            raise SystemExit(f'{path.stem} is new or changed and needs --materials to be replayed.')
        model, backend, values, coverage, not_run = score_pack(pack, materials, work)
        rows.append(make_row(path.stem, meta, digest, model, backend, values, not_run))
        notes.append(f'### {path.stem}\n\n```\n{report_text(model, values, coverage, not_run, reference)}\n```\n')
    check_unique(rows)
    Path(community).write_text(json.dumps(dict(version=reference['version'], scoring=reference['scoring'],
                                               models=rows), indent=1) + '\n')
    text = (f'Community leaderboard: {len(rows)} submissions, {len(notes)} replayed and scored in this run.\n\n'
            + '\n'.join(notes))
    if summary:
        Path(summary).write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--materials', type=Path, help='manifest.json written by build_paper_materials')
    parser.add_argument('--summary', type=Path, help='Also write the summary to this Markdown file')
    args = parser.parse_args()
    build(args.materials, summary=args.summary)


if __name__ == '__main__':
    main()
