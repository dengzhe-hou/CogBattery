"""Alternate-form and split-half task scores for the post hoc reliability analysis.

Reads FORMAL/<model>/<task>/calls/*.json read only and applies the frozen layer-one
scorers (human54_layers_20261019.model_scoring) to subsets of the planned rows.
Nothing here replaces an endpoint: the full row set is scored as well and must
reproduce the saved valid_only endpoint exactly, or the run stops.

DRM (warning vs no-warning condition) and CVLT (five learning trials) come from
the saved endpoint table and need no records. Spec: review-stage/
human54_posthoc_20261020/SPEC.md section C1.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import resource
import time

from scripts.reanalysis.human54_layers_20261019.model_scoring import (
    Builder, epitome, fb, metacognition, prepare_records, source_memory, wm_cc,
)
from scripts.reanalysis.human54_fullbattery_20261019.matrix import DIRECT, MEANS, wager_auc_from_endpoints

ROOT = Path(__file__).resolve().parents[3]
RECORD_TASKS = ('digit_span', 'n_back', 'operation_span', 'stroop', 'flanker', 'go_nogo',
                'source_monitoring', 'false_belief', 'epitome_tom',
                'confidence_calibration', 'post_decision_wagering')
SCORERS = {'source_monitoring': source_memory, 'false_belief': fb, 'epitome_tom': epitome,
           'confidence_calibration': metacognition, 'post_decision_wagering': metacognition}


def _episode_form(row):
    return row['episode_id'].rsplit('_', 1)[-1]


def _nb_form(rows):
    """Alternate sorted block ids within each load (20/20 per load)."""
    forms = {}
    for load in sorted({r['step']['meta']['load'] for r in rows}):
        episodes = sorted({r['episode_id'] for r in rows if r['step']['meta']['load'] == load})
        forms.update({ep: 'half' + str(i % 2) for i, ep in enumerate(episodes)})
    return lambda row: forms[row['episode_id']]


def form_of(task, rows):
    """Return a function row -> form label for this task."""
    if task in ('digit_span', 'stroop', 'flanker', 'go_nogo'):
        return _episode_form                                   # s037 / s143 / s490
    if task == 'operation_span':
        return _episode_form                                   # v1 / v2 / v3
    if task == 'n_back':
        return _nb_form(rows)
    if task == 'source_monitoring':
        return _episode_form                                   # formA / formB
    if task == 'false_belief':
        return lambda r: 'list' + str(r['step']['meta']['list'])
    if task == 'post_decision_wagering':
        return lambda r: 'grammar' + r['condition'].removeprefix('training_')[0]
    if task == 'confidence_calibration':
        # Episodes cal_p<position>_q<item>: both positions of an item share a half.
        return lambda r: 'item_' + ('odd' if int(r['episode_id'].rsplit('_q', 1)[1]) % 2 else 'even')
    if task == 'epitome_tom':
        return lambda r: 'story_' + ('odd' if int(r['step']['meta']['story_id']) % 2 else 'even')
    raise ValueError(task)


def score(model, task, rows):
    """Valid-only endpoints of the frozen scorer: name -> (value, n_scored)."""
    b = Builder(model, task, 'valid_only')
    (SCORERS.get(task) or wm_cc)(b, rows)
    return {e['endpoint']: (e['value'], e['n_scored']) for e in b.endpoints}


def _number(value):
    return float('nan') if value is None else float(value)


def matrix_value(task, endpoints):
    """The 13-task matrix definition (fullbattery matrix.py) without the coverage gate."""
    if task in DIRECT:
        return _number(endpoints[DIRECT[task][0]][0])
    if task in MEANS:
        values = [_number(endpoints[name][0]) for name in MEANS[task]]
        return float(sum(values) / len(values))
    if task == 'post_decision_wagering':
        get = lambda name: endpoints[name][0]
        return wager_auc_from_endpoints(get('accuracy'), get('high_wager_rate'), get('wager_1_accuracy'),
                                        get('wager_2_accuracy'), endpoints['accuracy'][1])['auc']
    raise ValueError(task)


def structure_sro_value(task, endpoints):
    """SRO structure endpoint (layer two), used for the SRO human-model comparison."""
    return _number(endpoints['structure_sro'][0])


def cell_halves(model, task, bundle, records, value=matrix_value):
    rows = prepare_records(bundle, records)
    full = score(model, task, rows)
    label = form_of(task, rows)
    forms = {}
    for row in rows:
        forms.setdefault(label(row), []).append(row)
    out = [dict(model=model, task=task, form='full', value=value(task, full), n_rows=len(rows))]
    for form, subset in sorted(forms.items()):
        out.append(dict(model=model, task=task, form=form, value=value(task, score(model, task, subset)),
                        n_rows=len(subset)))
    return out, full


def saved_endpoints(path):
    table = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row['rule'] == 'valid_only':
                table[row['model'], row['task'], row['endpoint']] = row['value']
    return table


def check_full(model, task, full, saved):
    """Every valid-only endpoint of the full row set equals the saved layer-one value."""
    for name, (value, _) in full.items():
        stored = saved.get((model, task, name))
        if stored is None:
            raise ValueError(f'No saved endpoint {model}/{task}/{name}')
        a, b = _number(value), (float('nan') if stored == '' else float(stored))
        if not ((math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f'Full-set rescore differs from saved endpoint: {model}/{task}/{name} {a} != {b}')


def saved_halves(saved, models):
    """DRM conditions and CVLT learning trials from the saved endpoint table."""
    out = []
    for model in models:
        for form in ('no_warning', 'warning'):
            v = saved[model, 'drm_false_memory', form + '|balanced_accuracy']
            out.append(dict(model=model, task='drm_false_memory', form=form,
                            value=float('nan') if v == '' else float(v), n_rows=''))
        for trial in range(1, 6):
            v = saved[model, 'cvlt_word_list', f'learning_{trial}']
            out.append(dict(model=model, task='cvlt_word_list', form=f'trial{trial}',
                            value=float('nan') if v == '' else float(v), n_rows=''))
    return out


SRO_TASKS = ('digit_span', 'n_back', 'stroop', 'flanker', 'go_nogo')


def run(manifest_path, formal, endpoints_path, output, kind='matrix'):
    if output.exists():
        raise FileExistsError(f'{output} exists; write a new output file')
    start = time.monotonic()
    manifest = json.loads(manifest_path.read_text())
    bundles = {e['task']: json.loads(Path(e['material_file']).read_text()) for e in manifest['tasks']}
    saved = saved_endpoints(endpoints_path)
    rows, checked = [], 0
    tasks, value = (RECORD_TASKS, matrix_value) if kind == 'matrix' else (SRO_TASKS, structure_sro_value)
    for model in manifest['spec']['models']:
        for task in tasks:
            folder = formal / model.replace(':', '__') / task
            records = [json.loads(p.read_text()) for p in sorted((folder / 'calls').glob('*.json'))]
            if any(r.get('synthetic') or r['model'] != model or r['task'] != task for r in records):
                raise ValueError(f'Unexpected record in {folder}')
            cell_rows, full = cell_halves(model, task, bundles[task], records, value)
            check_full(model, task, full, saved)
            checked += len(full)
            rows.extend(cell_rows)
        print(json.dumps(dict(model=model, elapsed_seconds=round(time.monotonic() - start, 1))), flush=True)
    if kind == 'matrix':
        rows.extend(saved_halves(saved, manifest['spec']['models']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'task', 'form', 'value', 'n_rows'])
        writer.writeheader()
        writer.writerows(rows)
    status = dict(kind=kind, finished_at=datetime.now().astimezone().isoformat(), rows=len(rows),
                  endpoints_checked_equal=checked, elapsed_seconds=time.monotonic() - start,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  hostname=os.uname().nodename, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    output.with_suffix('.json').write_text(json.dumps(status, indent=2) + '\n')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json')
    parser.add_argument('--formal', type=Path, default=ROOT / 'results/human54_unified_20261018/formal')
    parser.add_argument('--endpoints', type=Path, default=ROOT / 'results/reanalysis/human54_layers_20261019/layer1/endpoints.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=('matrix', 'structure_sro'), default='matrix')
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.formal, args.endpoints, args.output, args.kind), indent=2))


if __name__ == '__main__':
    main()
