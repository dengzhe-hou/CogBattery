"""Build source-derived task manifests offline; never starts a model or Slurm job."""
import argparse
import csv
import json
from pathlib import Path
import sys

from .common import HERE, ROOT, RAW, TASKS, validate_task, write_json


def prepare(output, seed=20261017):
    # Reuse the existing local analysis dependency, instead of installing another copy.
    deps = ROOT / '.deps' / 'public_human_structure_20261015'
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    from .registry import MODULES
    from .cvlt_materials import write_materials
    output = Path(output)
    spec = json.loads((HERE / 'SPEC.json').read_text())
    spec['preparation_seed'] = seed
    write_materials(output / 'materials' / 'cvlt_standard.json')
    config = {'root': str(ROOT), 'raw_root': str(RAW), 'seed': seed,
              'materials_dir': str(output / 'materials'),
              'drm_variant': spec['drm_variant']}
    bundles = {}
    for module in MODULES:
        built = module.build(config)
        if set(built) & set(bundles):
            raise ValueError('Multiple builders produced the same task')
        bundles.update(built)
    if set(bundles) != set(TASKS):
        raise ValueError(f'Task coverage differs from all 13: {set(bundles) ^ set(TASKS)}')
    summary = []
    for task in TASKS:
        bundle = bundles[task]
        presentation = spec.get('presentation', {}).get(task)
        if presentation:
            for episode in bundle['episodes']:
                episode['presentation'] = presentation
            bundle['provenance']['presentation'] = {
                'version': presentation,
                'adopted_on': spec['presentation_adoption']['adopted_on'],
                'preserved': 'All material, trial order, original responses, steps and scoring; full history remains accessible.',
                'changed': 'Repeated instructions and past chat-role boundaries become a chronological log.',
            }
        count = validate_task(task, bundle)
        filename = output / 'materials' / f'{task}.json'
        write_json(filename, bundle)
        models_new = len(spec['models']) - (len(spec['reuse']['false_belief']) if task == 'false_belief' else 0)
        summary.append({'task': task, 'status': bundle['status'],
                        'episodes': len(bundle['episodes']), 'constructed_calls_per_model': count,
                        'new_models': models_new, 'constructed_new_calls': count * models_new,
                        'presentation': presentation,
                        'blockers': bundle['blockers'], 'material_file': str(filename.resolve())})
    manifest = {'study_id': spec['study_id'], 'scope': 'preparation_only',
                'status_meaning': 'ready means source/material construction only; context checks, pilot, and inference authorization are separate.',
                'config': config, 'spec': spec, 'tasks': summary,
                'model_calls_executed': 0, 'slurm_jobs_submitted': 0,
                'notes': 'Constructed steps in blocked previews are not runnable or completed measurements.'}
    # Keep the selected human endpoint beside the model materials. Historical
    # exports retain their word/sentence mean; this reference names the sentence
    # columns explicitly and never calculates a new correlation or significance.
    from .ball_sentence_reference import prepare_reference
    human_scores, human_metadata = prepare_reference()
    reference_dir = output / 'ball_sentence_reference'
    reference_dir.mkdir(parents=True, exist_ok=True)
    human_scores.to_csv(reference_dir / 'ball_sentence_scores.csv', index=False, na_rep='')
    write_json(reference_dir / 'ball_sentence_metadata.json', human_metadata)
    manifest['human_references'] = {'drm_false_memory': {
        'variant': 'sentences',
        'scores_file': str((reference_dir / 'ball_sentence_scores.csv').resolve()),
        'metadata_file': str((reference_dir / 'ball_sentence_metadata.json').resolve()),
        'condition_column': 'warning',
    }}
    write_json(output / 'manifest.json', manifest)
    with (output / 'coverage.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow({**row, 'blockers': '; '.join(row['blockers'])})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'human13_20261017' / 'prepared_54_drm_sentence')
    parser.add_argument('--seed', type=int, default=20261017)
    args = parser.parse_args()
    manifest = prepare(args.output, args.seed)
    print(json.dumps({'tasks': manifest['tasks'], 'model_calls_executed': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
