"""Show one model's key endpoints next to the human references.

    python -m scripts.release.report_model --run results/submissions/qwen2.5__7b \
        --materials results/paper_materials/manifest.json

The endpoints are the paper's: one per paradigm (Table 4), scored by the paper's scorer with invalid
task answers counted as wrong. Human values, human SDs, and the 54-model means come from the leaderboard
file. A difference from humans shows distance, not quality. Submissions join the community table
through scripts.release.community.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='results/submissions/<model> or results/scored/<model>')
    parser.add_argument('--materials', type=Path, required=True, help='manifest.json written by build_paper_materials')
    parser.add_argument('--leaderboard', type=Path, default=ROOT / 'docs/data/leaderboard.json', help='Source of the human references')
    parser.add_argument('--json', type=Path, help='Also write the values to this file')
    args = parser.parse_args()
    model, values, coverage, not_run = model_values(args.run, args.materials)
    print(report_text(model, values, coverage, not_run, json.loads(args.leaderboard.read_text())))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(model=model, values=values, coverage=coverage, not_run=not_run), indent=1) + '\n')


if __name__ == '__main__':
    main()
