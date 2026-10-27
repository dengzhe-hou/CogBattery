"""Offline task-native scoring. Produces no overall human-likeness ranking."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .common import write_json
# r7: the format_repair whole-answer parsing plus the completion gate (scoring.py).
from .scoring import score_record


def validate_records(records, manifest):
    """Prevent accidentally scoring another protocol's calls with this manifest."""
    lookup = {}
    for entry in manifest['tasks']:
        bundle = json.loads(Path(entry['material_file']).read_text())
        for episode in bundle['episodes']:
            for step in episode['steps']:
                lookup[(entry['task'], episode['id'], step['id'])] = (
                    episode['condition'], step, episode.get('presentation'))
    for row in records:
        if row['model'] not in manifest['spec']['models']:
            raise ValueError(f'Model is outside the manifest panel: {row["model"]}')
        identity = (row['task'], row['episode_id'], row['step']['id'])
        if identity not in lookup or lookup[identity] != (
                row['condition'], row['step'], row.get('presentation')):
            raise ValueError(f'Record does not match manifest materials/presentation: {identity}')


def score_records(records, manifest=None):
    if manifest is not None:
        validate_records(records, manifest)
    from .registry import module_for
    groups = defaultdict(list)
    identities = set()
    for row in records:
        if row.get('synthetic'):
            raise ValueError('Synthetic engineering probes cannot enter behavioral summaries')
        identity = (row['model'], row['task'], row['episode_id'], row['step']['id'])
        if identity in identities:
            raise ValueError(f'Duplicate model measurement: {identity}')
        identities.add(identity)
        module = module_for(row['task'])
        scored = {**row, 'score': score_record(module, row)}
        groups[(row['model'], row['task'])].append(scored)
    summaries = []
    for (model, task), rows in sorted(groups.items()):
        module = module_for(task)
        summaries.append({'model': model, 'task': task, 'record_count': len(rows),
                          'metrics': module.summarize(rows)})
    return summaries


def valid_coverage(records, manifest):
    """Per model x paradigm: planned steps, committed records, technically valid records (decision a)."""
    planned = {entry['task']: entry.get('steps_per_model', entry.get('constructed_calls_per_model'))
               for entry in manifest['tasks']}
    table = defaultdict(lambda: {'committed': 0, 'technical_valid': 0, 'reused': 0})
    for row in records:
        cell = table[(row['model'], row['task'])]
        cell['committed'] += 1
        cell['technical_valid'] += bool(row.get('technical_valid'))
        cell['reused'] += row.get('execution_origin') == 'reused'
    return [{'model': model, 'task': task, 'planned': planned[task], **cell,
             'valid_coverage': cell['technical_valid'] / planned[task] if planned[task] else None}
            for (model, task), cell in sorted(table.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-reused-fb', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [json.loads(p.read_text()) for p in sorted(args.input.glob('**/calls/*.json'))]
    if args.include_reused_fb:
        from .tom_mc import load_reused_fb
        records.extend(load_reused_fb(manifest['config'], manifest['spec']))
    summaries = score_records(records, manifest)
    write_json(args.output, {'scope': 'offline_scoring_of_provided_responses',
                            'records': len(records), 'summaries': summaries,
                            'valid_coverage': valid_coverage(records, manifest),
                            'note': 'Partial input stays partial; no completed-panel or human-structure claim is inferred.'})
    print(json.dumps({'records': len(records), 'model_task_groups': len(summaries)}))


if __name__ == '__main__':
    main()
