"""Build the paper-version materials (protocol formal_r7_unified_20261018) from the original sources.

    python -m scripts.release.fetch_sources
    python -m scripts.release.build_paper_materials --output results/paper_materials

Sources are read from data/raw/ (or COGBATTERY_RAW). The build runs the same task builders as the
paper, applies the r7 output-token budget map, and compares each task with the digest of the paper's
material file. A mismatch stops the build, because model scores would then not be comparable with
the paper. No model is called.

CVLT needs the licensed CVLT-II Standard lists in
scripts/experiments/human13_20261017/cvlt_standard_source.json. Without that file the other 12 tasks
are built and CVLT is reported as missing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'  # the SRO quality-control module requires single-threaded numerics

from scripts.experiments.human13_20261017.common import HERE, RAW, ROOT, TASKS, validate_task, write_json
from scripts.experiments.human13_20261017.prepare_r7 import map_bundle

PAPER_DIR = Path(__file__).resolve().parent / 'paper_materials'
DIGESTS = PAPER_DIR / 'material_digests.json'
SRO_IDS = PAPER_DIR / 'sro_eligible_ids.json'
CVLT_SOURCE = HERE / 'cvlt_standard_source.json'
SEED = 20261017


def normalized_digest(bundle, root, raw):
    """SHA-256 of a bundle's episodes (prompts, answers, scoring metadata), with machine-specific
    path prefixes replaced. Provenance records local paths and is not compared."""
    text = json.dumps(bundle['episodes'], ensure_ascii=False, sort_keys=True)
    for prefix, token in sorted(((str(raw), '<RAW>'), (str(root), '<ROOT>')), key=lambda x: -len(x[0])):
        text = text.replace(prefix, token)
    return hashlib.sha256(text.encode()).hexdigest()


def build_same_stimulus(directory):
    """Regenerate the n-back blocks and Strachan items that the paper builders read."""
    from scripts.experiments.same_stimulus_20261011 import materials as M
    episodes, human_trials, _, _, _ = M.build_nback()
    items, _, _ = M.build_strachan()
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (('nback_episodes.jsonl', episodes), ('human_nback_trials.jsonl', human_trials),
                       ('strachan_items.jsonl', items)):
        (directory / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def build(output, check=True):
    output = Path(output).resolve()
    spec = json.loads((HERE / 'SPEC.json').read_text())
    spec['preparation_seed'] = SEED
    same_stimulus = output / 'same_stimulus'
    build_same_stimulus(same_stimulus)

    materials = output / 'materials'
    have_cvlt = CVLT_SOURCE.is_file()
    if have_cvlt:
        from scripts.experiments.human13_20261017.cvlt_materials import write_materials
        write_materials(materials / 'cvlt_standard.json')
    config = {'root': str(ROOT), 'raw_root': str(RAW), 'seed': SEED, 'materials_dir': str(materials),
              'drm_variant': spec['drm_variant'], 'sro_ids_file': str(SRO_IDS),
              'same_stimulus_dir': str(same_stimulus), 'fb_spec_file': str(PAPER_DIR / 'same_stimulus_prompts.json'),
              'wag_table_file': str(PAPER_DIR / 'wag_dienes_scott2005_table6.json')}
    from scripts.experiments.human13_20261017.registry import MODULES
    bundles = {}
    for module in MODULES:
        bundles.update(module.build(config))

    expected = json.loads(DIGESTS.read_text())['tasks']
    budget_map = spec['inference']['direct_step_budget_map']
    summary, missing, mismatched = [], [], []
    for task in TASKS:
        if task == 'cvlt_word_list' and not have_cvlt:
            missing.append(task)
            continue
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
        validate_task(task, bundle)
        bundle = map_bundle(bundle, budget_map)
        count = validate_task(task, bundle)
        digest = normalized_digest(bundle, ROOT, RAW)
        if digest != expected[task]:
            mismatched.append(task)
        target = materials / f'{task}.json'
        write_json(target, bundle)
        summary.append({'task': task, 'episodes': len(bundle['episodes']), 'steps_per_model': count,
                        'material_file': str(target), 'digest': digest, 'matches_paper': digest == expected[task]})

    manifest = {'study_id': spec['study_id'], 'scope': 'formal_r7_prepared', 'revision': spec['revision'],
                'spec': spec, 'config': config, 'tasks': summary,
                'steps_per_model': sum(row['steps_per_model'] for row in summary),
                'missing_tasks': missing, 'model_calls_executed': 0}
    write_json(output / 'manifest.json', manifest)
    for task in missing:
        print(f'{task}: not built. Place the licensed CVLT-II Standard lists in {CVLT_SOURCE}.')
    if check and mismatched:
        raise SystemExit(f'Materials differ from the paper for: {", ".join(mismatched)}. '
                         'Check that data/raw/ holds the files listed in scripts/release/paper_materials/sources.json.')
    print(f'Built {len(summary)} of {len(TASKS)} tasks in {materials}. All built tasks match the paper.'
          if not mismatched else f'Built {len(summary)} tasks; {len(mismatched)} differ from the paper.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'paper_materials')
    args = parser.parse_args()
    build(args.output)


if __name__ == '__main__':
    main()
