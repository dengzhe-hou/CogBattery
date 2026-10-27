"""Prepare the authorized technical pilot, separately from full-study materials."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .common import ROOT, TASKS, default_messages, validate_task, write_json


PREFIX_CALLS = {
    'digit_span': 4, 'n_back': 22, 'operation_span': 13,
    'stroop': 2, 'flanker': 2, 'go_nogo': 3,
    'cvlt_word_list': 18, 'drm_false_memory': 66,
    'source_monitoring': 5, 'false_belief': 1, 'epitome_tom': 1,
    'confidence_calibration': 2, 'post_decision_wagering': 47,
}
PROBE_AUDITS = {
    'cvlt_word_list': (
        ROOT / 'results/human13_20261017/prepared_54_cvlt_compact/context_audit/context_audit.json',
        'target_arrays_both_replies',
    ),
    'drm_false_memory': (
        ROOT / 'results/human13_20261017/prepared_54_drm_sentence/context_audit/context_audit.json',
        'new_no_high',
    ),
    'post_decision_wagering': (
        ROOT / 'results/human13_20261017/compact_candidate_54/all_models/context_audit.json',
        'new_no_high',
    ),
}


def prefix_bundle(task, bundle):
    """Copy the fixed chronological prefix, including any parent episode."""
    if bundle['status'] != 'ready':
        raise ValueError(f'Pilot materials are unresolved for {task}')
    result = deepcopy(bundle)
    result['episodes'] = []
    remaining = PREFIX_CALLS[task]
    for source_episode in bundle['episodes']:
        episode = deepcopy(source_episode)
        episode['steps'] = episode['steps'][:remaining]
        result['episodes'].append(episode)
        remaining -= len(episode['steps'])
        if not remaining:
            break
    if remaining or validate_task(task, result) != PREFIX_CALLS[task]:
        raise ValueError(f'Cannot construct the specified complete pilot prefix for {task}')
    result['scope'] = 'technical_pilot'
    result['research_results_eligible'] = False
    result['selection'] = {
        'method': 'fixed chronological prefix, selected before any model responses',
        'logical_calls': PREFIX_CALLS[task],
        'history': 'Actual earlier model replies only; no synthetic replies in this bundle.',
    }
    return result


def peak_target(task):
    """Select the largest previously audited compact template across the panel."""
    path, scenario = PROBE_AUDITS[task]
    audit = json.loads(path.read_text())
    candidates = []
    for model in audit['models']:
        checks = model['checks']
        if isinstance(checks, dict):
            checks = [dict(check, scenario=name) for name, check in checks.items()]
        for check in checks:
            if check['scenario'] != scenario or check.get('task', 'cvlt_word_list') != task:
                continue
            peak = check.get('peak_scripted_history') or check['peaks']['compact']
            candidates.append((peak['estimated_tokens_with_margins'], model['model'], peak))
    if not candidates:
        raise ValueError(f'No existing static audit for {task}/{scenario}')
    budget, model, peak = max(candidates, key=lambda row: row[0])
    return {
        'episode_id': peak['episode_id'], 'step_id': peak['step_id'],
        'scenario': scenario, 'audit_file': str(path.resolve()),
        'selection': 'Largest previously audited compact input budget across all 54 models.',
        'audited_peak_model': model, 'audited_peak_budget_tokens': budget,
    }


def engineering_probe(task, bundle, source_file):
    """Render one labeled fixture history using the existing offline audit replies."""
    from .audit_compact_contexts import format_reply
    from .audit_cvlt_contexts import SCENARIOS
    from .registry import module_for

    target = peak_target(task)
    module = module_for(task)
    reply_builder = (SCENARIOS[target['scenario']] if task == 'cvlt_word_list'
                     else lambda step: format_reply(step, target['scenario']))
    histories = {}
    for episode in bundle['episodes']:
        history = list(histories[episode['parent']]) if episode.get('parent') else []
        for step in episode['steps']:
            if (episode['id'], step['id']) == (target['episode_id'], target['step_id']):
                return {
                    'id': f'{task}__synthetic_long_history', 'task': task,
                    'scope': 'synthetic_history_engineering_probe', 'synthetic': True,
                    'research_results_eligible': False,
                    'episode_id': episode['id'], 'condition': episode['condition'],
                    'presentation': episode.get('presentation'),
                    'episode': deepcopy(episode), 'step': deepcopy(step),
                    'messages': getattr(module, 'messages', default_messages)(episode, step, history),
                    'history_records': len(history),
                    'source_material_file': str(Path(source_file).resolve()),
                    'fixture_selection': target,
                    'scenario': target['scenario'],
                    'notice': 'Synthetic earlier replies test input capacity and engineering only; '
                              'they are not model observations or formal study results.',
                }
            response = reply_builder(step)
            score = module.score_step(step, response)
            if not score['valid']:
                raise ValueError(f'Invalid existing audit fixture: {task}/{step["id"]}')
            history.append({'step': step, 'response': response, 'score': score})
        histories[episode['id']] = history
    raise ValueError(f'Previously audited probe is absent from current materials: {task}/{target}')


def prepare_pilot(source_manifest, output):
    source_manifest, output = Path(source_manifest).resolve(), Path(output).resolve()
    if output == source_manifest.parent:
        raise ValueError('Technical pilot must use a separate directory from full-study materials')
    source = json.loads(source_manifest.read_text())
    if set(source['spec']['tasks']) != set(TASKS) or len(source['spec']['models']) != 54:
        raise ValueError('This pilot requires the unchanged 54-model, 13-task panel')
    entries = {entry['task']: entry for entry in source['tasks']}
    bundles = {task: json.loads(Path(entries[task]['material_file']).read_text()) for task in TASKS}
    prefixes = {task: prefix_bundle(task, bundles[task]) for task in TASKS}
    probes = [engineering_probe(task, bundles[task], entries[task]['material_file'])
              for task in PROBE_AUDITS]
    spec = deepcopy(source['spec'])
    spec['revision'] = source['spec']['revision'] + '_technical_pilot_v1'
    spec['authorization'] = {
        'preparation_authorized': True, 'pilot_authorized': True,
        'inference_authorized': True, 'formal_authorized': False,
        'scope': 'technical_pilot',
        'user_instruction': '执行，先用一个代表模型测资源，再按实测安排54款的小规模短测，不要申请多余资源。重点核对Qwen3 4B的思考模式，完整评测先不启动，论文和Overleaf先不改。',
    }
    spec['scope'] = 'technical_pilot'
    spec['inference_status'] = 'Authorized fixed technical prefixes and engineering probes only; full evaluation remains unauthorized.'
    # Technical format/resource checks must not be satisfied by old formal results.
    spec['reuse']['false_belief'] = {}
    spec['technical_pilot'] = {
        'source_manifest': str(source_manifest),
        'real_prefix_calls_per_model': 186, 'synthetic_probe_calls_per_model': 3,
        'total_calls_per_model': 189, 'research_results_eligible': False,
        'execution_order': 'One representative model first; size the remaining panel from its measured resources.',
    }
    summary = []
    for task in TASKS:
        filename = output / 'materials' / f'{task}.json'
        write_json(filename, prefixes[task])
        entry = deepcopy(entries[task])
        entry.update({
            'episodes': len(prefixes[task]['episodes']),
            'constructed_calls_per_model': PREFIX_CALLS[task],
            'new_models': 54, 'constructed_new_calls': PREFIX_CALLS[task] * 54,
            'material_file': str(filename), 'source_material_file': entries[task]['material_file'],
            'scope': 'technical_pilot', 'research_results_eligible': False,
        })
        summary.append(entry)
    manifest = {
        'study_id': source['study_id'], 'scope': 'technical_pilot',
        'source_manifest': str(source_manifest), 'spec': spec,
        'config': {**source['config'], 'materials_dir': str(output / 'materials')},
        'tasks': summary, 'engineering_probes': probes,
        'engineering_probes_file': str(output / 'probes.json'),
        'expected_calls': {
            'real_per_model': 186, 'synthetic_per_model': 3, 'total_per_model': 189,
            'real_all_models': 10044, 'synthetic_all_models': 162, 'total_all_models': 10206,
        },
        'model_calls_executed': 0, 'slurm_jobs_submitted': 0,
        'research_results_eligible': False,
        'notes': 'Separate technical pilot only; all 54 models and 13 paradigms retained. '
                 'Full materials, pending DRM words and inference settings remain unchanged.',
    }
    if sum(entry['constructed_calls_per_model'] for entry in summary) != 186 or len(probes) != 3:
        raise ValueError('Pilot call count must be exactly 186 actual-prefix requests plus 3 engineering probes')
    write_json(output / 'SPEC.json', spec)
    write_json(output / 'probes.json', probes)
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_pilot(args.source_manifest, args.output)
    print(json.dumps({'scope': manifest['scope'], 'expected_calls': manifest['expected_calls'],
                      'model_calls_executed': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
