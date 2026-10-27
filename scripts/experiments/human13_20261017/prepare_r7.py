"""Derive r7 materials read-only from the r6 full materials; only max_tokens changes.

Each r6 step max_tokens is mapped once by SPEC inference.direct_step_budget_map
(human13_repair_20261017/pilot_materials.py:132-137). Stimuli, steps, order,
branches and answers are copied unchanged. The r6 files are never modified.
No model, Ollama server or Slurm job is involved.
"""
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import shutil

from .common import HERE, ROOT, TASKS, validate_task, write_json

R6_MANIFEST = ROOT / 'results/human13_20261017/prepared_54_drm_sentence/manifest.json'
DEFAULT_OUTPUT = ROOT / 'results/human54_unified_20261018/prepared_r7'


def map_bundle(bundle, budget_map):
    result = deepcopy(bundle)
    for episode in result['episodes']:
        for step in episode['steps']:
            step['max_tokens'] = budget_map[str(step['max_tokens'])]
    return result


def prepare_r7(output=DEFAULT_OUTPUT, source_manifest=R6_MANIFEST):
    output, source_manifest = Path(output), Path(source_manifest)
    source = json.loads(source_manifest.read_text())
    spec = json.loads((HERE / 'SPEC.json').read_text())
    if list(source['spec']['tasks']) != list(spec['tasks']) or source['spec']['models'] != spec['models']:
        raise ValueError('r6 and r7 must keep the same 54 models and 13 paradigms in order')
    budget_map = spec['inference']['direct_step_budget_map']
    entries = {e['task']: e for e in source['tasks']}
    summary = []
    for task in TASKS:
        bundle = json.loads(Path(entries[task]['material_file']).read_text())
        mapped = map_bundle(bundle, budget_map)
        count = validate_task(task, mapped)
        target = output / 'materials' / f'{task}.json'
        write_json(target, mapped)
        summary.append({'task': task, 'status': mapped['status'], 'episodes': len(mapped['episodes']),
                        'steps_per_model': count, 'models': len(spec['models']),
                        'planned_logical_slots': count * len(spec['models']),
                        'presentation': entries[task].get('presentation'),
                        'source_material_file': entries[task]['material_file'],
                        'material_file': str(target.resolve())})
    shutil.copyfile(Path(source['config']['materials_dir']) / 'cvlt_standard.json',
                    output / 'materials' / 'cvlt_standard.json')
    per_model = sum(row['steps_per_model'] for row in summary)
    manifest = {'study_id': spec['study_id'], 'scope': 'formal_r7_prepared', 'revision': spec['revision'],
                'source_manifest': str(source_manifest), 'spec': spec,
                'config': {**source['config'], 'materials_dir': str((output / 'materials').resolve())},
                'tasks': summary, 'steps_per_model': per_model,
                'planned_logical_slots': per_model * len(spec['models']),
                'human_references': source.get('human_references'),
                'model_calls_executed': 0, 'slurm_jobs_submitted': 0,
                'notes': 'Materials equal r6 except step max_tokens (direct budget map). FB reuse is resolved '
                         'by the exact-input cache, so every model has 72 FB slots here.'}
    write_json(output / 'manifest.json', manifest)
    with (output / 'coverage.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare_r7(args.output)
    print(json.dumps({'steps_per_model': manifest['steps_per_model'],
                      'planned_logical_slots': manifest['planned_logical_slots'],
                      'tasks': {t['task']: t['steps_per_model'] for t in manifest['tasks']}}, indent=1))


if __name__ == '__main__':
    main()
