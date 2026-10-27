"""Offline long-chat/compact comparison; vocabularies only, never inference."""
import argparse
import csv
import json
from pathlib import Path
import time

from .common import ROOT, write_json
from .run import model_spec


SCENARIOS = ('old_yes_low', 'new_no_high')
TASKS = ('drm_false_memory', 'post_decision_wagering')


def format_reply(step, scenario):
    """Legal-format fixtures, independent of scoring gold; not model observations."""
    low = scenario == 'old_yes_low'
    kind = step['kind']
    if kind == 'drm_rating':
        return '4'
    if kind == 'drm_recognition':
        return 'OLD' if low else 'NEW'
    if kind == 'drm_report':
        return 'FAMILIAR' if low else 'RECALL_TO_REJECT'
    if kind == 'wag_training':
        # A copy of the already-visible string, not the classification answer key.
        return step['prompt'].rsplit('\n\n', 1)[1]
    if kind == 'wag_answer':
        return 'YES' if low else 'NO'
    if kind == 'wager':
        return '1' if low else '2'
    raise ValueError(f'Unexpected step kind in two-task format check: {kind}')


def requests_for(bundle, original, compact, scenario):
    histories, requests = {}, []
    for episode in bundle['episodes']:
        # The current manifest now selects compact presentation. This audit's
        # explicit "original" baseline must still use the earlier long chat.
        original_episode = dict(episode)
        original_episode.pop('presentation', None)
        history = list(histories[episode['parent']]) if episode.get('parent') else []
        for step in episode['steps']:
            requests.append({
                'episode_id': episode['id'], 'step_id': step['id'],
                'condition': episode['condition'], 'output_budget': step['max_tokens'],
                'history_records': len(history),
                'original': original.messages(original_episode, step, history),
                'compact': compact.messages(episode, step, history),
            })
            response = format_reply(step, scenario)
            score = original.score_step(step, response)
            if not score['valid']:
                raise ValueError(f'Fixture is not a legal output: {step["id"]}/{response}')
            history.append({'step': step, 'response': response, 'score': score})
        histories[episode['id']] = history
    return requests


def inspect_requests(requests, counter, inf, declared):
    peaks = {variant: None for variant in ('original', 'compact')}
    overflows = {variant: 0 for variant in peaks}
    for request in requests:
        for variant in peaks:
            messages = request[variant]
            raw = sum(counter.count(m['content']) for m in messages)
            need = (raw + inf['template_margin_tokens']
                    + inf['per_message_margin_tokens'] * len(messages)
                    + request['output_budget'])
            row = {key: request[key] for key in
                   ('episode_id', 'step_id', 'condition', 'history_records', 'output_budget')}
            row.update({'raw_tokens': raw, 'message_count': len(messages),
                        'estimated_tokens_with_margins': need})
            if peaks[variant] is None or need > peaks[variant]['estimated_tokens_with_margins']:
                peaks[variant] = row
            overflows[variant] += need > inf['context_tokens']
    if not requests:
        raise ValueError('Empty task is not a valid compact-candidate check')
    need = peaks['compact']['estimated_tokens_with_margins']
    old = peaks['original']['estimated_tokens_with_margins']
    return {'inspected_steps': len(requests), 'peaks': peaks,
            'overflow_steps': overflows,
            'fits_configured_context': need <= inf['context_tokens'],
            'fits_gguf_declared_context': None if declared <= 0 else need <= declared,
            'headroom_configured': inf['context_tokens'] - need,
            'peak_budget_reduction': old - need}


def audit(manifest_path, output, models=None, length_candidates=None, tasks=None):
    from . import compact_drm, compact_wag, memory, tom_mc
    from .audit_contexts import local_limits
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    spec = manifest['spec']
    selected = models or spec['models']
    if any(m not in spec['models'] for m in selected):
        raise ValueError('Only the user-authorized 54-model panel may be inspected')
    modules = {'drm_false_memory': (memory, compact_drm),
               'post_decision_wagering': (tom_mc, compact_wag)}
    prepared, source_states = {}, {}
    selected_tasks = tasks or TASKS
    if any(task not in TASKS for task in selected_tasks):
        raise ValueError('Only DRM and wagering support this compact-format audit')
    for task in selected_tasks:
        row = next(t for t in manifest['tasks'] if t['task'] == task)
        bundle = json.loads(Path(row['material_file']).read_text())
        source_states[task] = {'status': bundle['status'], 'blockers': bundle['blockers'],
                               'episodes': len(bundle['episodes'])}
        original, compact = modules[task]
        for scenario in SCENARIOS:
            prepared[(task, scenario)] = requests_for(bundle, original, compact, scenario)
    # These source-backed alternatives remain blocked. Neither is selected as
    # the missing Ball CITY version; only their possible input length is checked.
    for path in length_candidates or []:
        path = Path(path)
        bundle = json.loads(path.read_text())
        if bundle['status'] != 'blocked' or any(
                e['task'] != 'drm_false_memory' for e in bundle['episodes']):
            raise ValueError('CITY alternatives must remain blocked DRM length-only bundles')
        key = f'{path.parent.name}_{path.stem}'
        if key in source_states:
            raise ValueError(f'Duplicate named length candidate: {key}')
        source_states[key] = {'status': bundle['status'], 'blockers': bundle['blockers'],
                             'episodes': len(bundle['episodes']), 'candidate_file': str(path.resolve())}
        for scenario in SCENARIOS:
            prepared[(key, scenario)] = requests_for(bundle, memory, compact_drm, scenario)
    # Samples deliberately contain only format fixtures, not generated answers.
    samples = []
    for (task, scenario), requests in prepared.items():
        for request in (requests[0], requests[-1]):
            samples.append({'task': task, 'scenario': scenario,
                            'notice': 'Scripted format fixture, no model inference', **request})
    write_json(output / 'message_samples.json', samples)
    results, rows = [], []
    start = time.monotonic()
    for model in selected:
        inf = model_spec(spec, model)['inference']
        model_result = {'model': model, 'model_calls': 0, 'checks': []}
        with Counter(model) as counter:
            model_result['tokenizer'] = counter.metadata
            model_result['limits'] = local_limits(counter)
            declared = model_result['limits']['gguf_declared_context_tokens']
            for (task, scenario), requests in prepared.items():
                check = inspect_requests(requests, counter, inf, declared)
                check.update({'task': task, 'scenario': scenario,
                              'source_status': source_states[task]['status']})
                model_result['checks'].append(check)
                rows.append({'model': model, 'task': task, 'scenario': scenario,
                             'source_status': check['source_status'],
                             'configured_context': inf['context_tokens'],
                             'gguf_declared_context': declared,
                             'original_peak_budget': check['peaks']['original']['estimated_tokens_with_margins'],
                             'compact_peak_budget': check['peaks']['compact']['estimated_tokens_with_margins'],
                             'headroom_configured': check['headroom_configured'],
                             'fits_configured_context': check['fits_configured_context'],
                             'fits_gguf_declared_context': check['fits_gguf_declared_context'],
                             'inspected_steps': check['inspected_steps']})
        write_json(output / 'models' / f'{model.replace(":", "__")}.json', model_result)
        results.append(model_result)
        write_json(output / 'progress.json', {'models_checked': len(results),
                   'models_expected': len(selected), 'last_model': model, 'model_calls': 0})
        print(f'{len(results)}/{len(selected)} {model}: ' +
              ', '.join(f'{c["task"]}/{c["scenario"]}={c["peaks"]["compact"]["estimated_tokens_with_margins"]}'
                        for c in model_result['checks']), flush=True)
    result = {'scope': 'Explicit long-chat/compact comparison, CPU vocabulary-only; no inference',
              'manifest': str(manifest_path.resolve()), 'spec_revision': spec['revision'],
              'manifest_presentations': spec.get('presentation', {}),
              'model_count': len(selected), 'model_calls': 0, 'resource_requests': 0,
              'source_states': source_states, 'scenarios': list(SCENARIOS), 'models': results,
              'elapsed_offline_seconds': time.monotonic() - start,
              'limitations': [
                  'Only legal short-format scripted replies, not real model responses or performance.',
                  'Raw-content tokenization plus inherited template margins, not exact rendered server templates.',
                  'Verbose or invalid real replies are retained without truncation and may still overflow.',
                  'DRM source status and selected variant come from the manifest; separate CITY word alternatives remain length-only and do not resolve the missing source version.',
                  'Source study/item order, scores and output budgets are unchanged; the two explicit presentation serializers are compared.',
                  'Presentation adoption is recorded by the loaded manifest, independently of this no-inference check.',
              ]}
    write_json(output / 'context_audit.json', result)
    with (output / 'context_by_model_task.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'results/human13_20261017/prepared_54_public_versions/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', action='append', help='Inspect selected authorized models; omitted means all 54')
    parser.add_argument('--task', choices=TASKS, action='append',
                        help='Inspect only a changed task; omitted means both compact tasks')
    parser.add_argument('--length-candidate', type=Path, action='append',
                        help='Blocked DRM word alternative to inspect for length only; never selects a source version')
    args = parser.parse_args()
    audit(args.manifest, args.output, args.model, args.length_candidate, args.task)


if __name__ == '__main__':
    main()
