"""Show one model's key endpoints next to the human references, and add a replayed submission to the leaderboard.

    python -m scripts.release.report_model --run results/submissions/qwen2.5__7b \
        --materials results/paper_materials/manifest.json

    python -m scripts.release.report_model --run results/scored/qwen2.5__7b \
        --materials results/paper_materials/manifest.json \
        --add-to docs/data/leaderboard.json --family qwen --params-b 7

The endpoints are the paper's: one per paradigm (Table 4), scored by the paper's scorer with invalid
task answers counted as wrong. Human values, human SDs, and the 54-model means come from the leaderboard
file. A difference from humans shows distance, not quality. `--add-to` accepts only a complete run
that went through `pack_submission score`, whose replay checked every request.
"""
import argparse
import json
from pathlib import Path

from scripts.reanalysis.human54_layers_20261019.model_scoring import analyze_cell
from scripts.reanalysis.human54_posthoc_20261020.gen_core_table import ROWS

ROOT = Path(__file__).resolve().parents[2]
RULE = 'invalid_wrong'


def model_values(run, materials):
    """Key endpoint, endpoint coverage, and unrun steps per paper task code, from one run folder."""
    entries = {t['task']: t for t in json.loads(Path(materials).read_text())['tasks']}
    models, values, coverage, not_run = set(), {}, {}, {}
    for code, task, endpoint, _ in ROWS:
        folder = Path(run) / task
        if task not in entries:
            raise SystemExit(f'{task} is not in {materials}.')
        records = [json.loads(p.read_text()) for p in sorted((folder / 'calls').glob('*.json'))]
        if not records:
            raise SystemExit(f'{folder} has no records; finish or resume the run first.')
        models |= {r['model'] for r in records}
        bundle = json.loads(Path(entries[task]['material_file']).read_text())
        cell = analyze_cell(records[0]['model'], task, bundle, records)
        row = next(e for e in cell['endpoints'] if e['rule'] == RULE and e['endpoint'] == endpoint)
        value = row['value']
        values[code] = value in (0., 1.) if code == 'WAG' and value is not None else value
        coverage[code] = row['coverage']
        not_run[code] = sum(r['category'] == 'D' for r in cell['rows'])
    if len(models) != 1:
        raise SystemExit(f'{run} mixes records from {len(models)} models.')
    return models.pop(), values, coverage, not_run


def report_text(model, values, coverage, not_run, board):
    def cell(x, digits=3):
        return '' if x is None else f'{x:.{digits}f}'
    head = ('Task', 'Key measure', 'Model', 'Human', 'Difference', 'Human SDs', '54-model mean', 'Valid')
    rows = []
    for k in board['tasks']:
        code, value = k['code'], values[k['code']]
        if k['kind'] == 'constant':
            shown = 'undefined' if value is None else 'always same wager' if value else 'varied'
            rows.append((code, k['measure'], shown, '', '', '', f"{k['constant_models']} of {k['models']} constant", cell(coverage[code], 2)))
        elif value is None:
            rows.append((code, k['measure'], 'undefined', cell(k['human']), '', '', cell(k['model_mean']), cell(coverage[code], 2)))
        else:
            gap = value - k['human']
            rows.append((code, k['measure'], cell(value), cell(k['human']), f'{gap:+.3f}', f"{gap / k['human_sd']:+.2f}",
                         cell(k['model_mean']), cell(coverage[code], 2)))
    widths = [max(len(str(r[i])) for r in (head, *rows)) for i in range(len(head))]
    lines = [f'CogBattery key endpoints for {model}', '']
    lines += ['  '.join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in (head, *rows)]
    lines += ['', 'Invalid task answers count as wrong. Valid is the share of planned steps for that measure with a usable answer.',
              'Human SDs is the distance from the human mean in across-person human SDs. It shows distance, not quality.']
    missing = sum(not_run.values())
    if missing:
        lines += [f'{missing} planned steps did not run, so this report is incomplete. Run the same command again to resume.']
    return '\n'.join(lines)


def add_to_board(path, run, model, values, not_run, family, params_b, replace=False):
    header_path = Path(run) / 'submission_header.json'
    if not header_path.is_file():
        raise SystemExit(f'{run} has no submission_header.json. Only replayed submissions join the leaderboard, see pack_submission score.')
    missing = sum(not_run.values())
    if missing:
        raise SystemExit(f'{missing} planned steps did not run, so this submission is incomplete.')
    undefined = [code for code, value in values.items() if value is None]
    if undefined:
        raise SystemExit(f'A leaderboard row needs all 13 measures, and these are undefined: {", ".join(undefined)}.')
    backend = json.loads(header_path.read_text())['header']['backend']
    board = json.loads(Path(path).read_text())
    old = [m for m in board['models'] if m['model'] == model]
    if any(m['serving'] == 'paper' for m in old):
        raise SystemExit(f'{model} is a paper row and is not replaced by a submission.')
    if old and not replace:
        raise SystemExit(f'{model} is already on the leaderboard. Pass --replace to update its row.')
    board['models'] = [m for m in board['models'] if m['model'] != model]
    board['models'].append(dict(model=model, family=family, params_b=params_b, native_thinking=False, serving=backend, values=values))
    Path(path).write_text(json.dumps(board, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='results/submissions/<model> or results/scored/<model>')
    parser.add_argument('--materials', type=Path, required=True, help='manifest.json written by build_paper_materials')
    parser.add_argument('--leaderboard', type=Path, default=ROOT / 'docs/data/leaderboard.json', help='Source of the human references')
    parser.add_argument('--json', type=Path, help='Also write the values to this file')
    parser.add_argument('--add-to', type=Path, help='Leaderboard file that receives this submission')
    parser.add_argument('--family', help='Model family, required with --add-to')
    parser.add_argument('--params-b', type=float, help='Served parameters in billions, required with --add-to')
    parser.add_argument('--replace', action='store_true', help='Update an existing submitted row')
    args = parser.parse_args()
    if args.add_to and (args.family is None or args.params_b is None):
        parser.error('--add-to needs --family and --params-b')
    model, values, coverage, not_run = model_values(args.run, args.materials)
    print(report_text(model, values, coverage, not_run, json.loads(args.leaderboard.read_text())))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(model=model, values=values, coverage=coverage, not_run=not_run), indent=1) + '\n')
    if args.add_to:
        add_to_board(args.add_to, args.run, model, values, not_run, args.family, args.params_b, args.replace)
        print(f'\nAdded {model} to {args.add_to}.')


if __name__ == '__main__':
    main()
