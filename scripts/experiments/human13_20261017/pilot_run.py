"""Technical-only 13-task pilot; formal SPEC and behavior analyses are untouched."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error

from .common import TASKS, validate_task, write_json

R6_SPEC = (Path(__file__).resolve().parents[3] / 'refine-logs/human54_unified_entry_20261018'
           / 'backups_r6_20261018_161458/SPEC_20261018_161458.json')
from .run_r6 import check_authorized, model_spec, run_episodes


def validate_pilot(manifest, model):
    spec = manifest['spec']
    auth = spec['authorization']
    if (manifest.get('scope') != 'technical_pilot'
            or auth.get('inference_authorized') is not True
            or auth.get('pilot_authorized') is not True
            or auth.get('formal_authorized') is not False):
        raise RuntimeError('Only explicitly authorized technical pilots may use this runner; formal evaluation is forbidden.')
    if model not in spec['models']:
        raise ValueError(f'Model outside the authorized pilot panel: {model}')
    if list(spec['tasks']) != list(TASKS) or [e['task'] for e in manifest['tasks']] != list(TASKS):
        raise ValueError('The pilot must retain all 13 tasks in the declared order.')
    if len(spec['models']) != 54 or 'tinyllama:1.1b' in spec['models']:
        raise ValueError('The pilot must retain the authorized 54-model panel.')
    # Frozen tool: compare with the r6 SPEC backup, not the r7 formal SPEC.
    original = json.loads(R6_SPEC.read_text())
    if spec['inference'] != original['inference'] or spec['backends'] != original['backends']:
        raise ValueError('Pilot inference settings or repaired backends differ from the prepared protocol.')
    if spec['models'] != original['models']:
        raise ValueError('Pilot model identities or order differ from the prepared protocol.')
    total = 0
    for entry in manifest['tasks']:
        bundle = json.loads(Path(entry['material_file']).read_text())
        check_authorized(spec, entry['task'], model, bundle)
        total += validate_task(entry['task'], bundle)
    if total != 186 or len(manifest['engineering_probes']) != 3:
        raise ValueError(f'Expected 186 real-prefix calls and 3 engineering probes, found {total} real calls.')
    for probe in manifest['engineering_probes']:
        if probe.get('synthetic') is not True or probe.get('scope') != 'synthetic_history_engineering_probe':
            raise ValueError('Engineering probes must be explicitly marked synthetic.')
    return spec


def validate_serving_environment(spec, model):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Real inference requires an authorized allocated Slurm GPU.')
    backend = spec['backends'].get(model, spec['backends']['default'])
    selected = shutil.which('ollama')
    if not selected or Path(selected).resolve() != Path(backend).resolve():
        raise RuntimeError('Serving binary differs from the recorded repaired backend.')


def known_failure(error):
    """Classify actual runner/client failures; unexpected implementation bugs propagate."""
    if isinstance(error, urllib.error.HTTPError):
        return 'http_error', True
    if isinstance(error, (urllib.error.URLError, TimeoutError)):
        return 'transport_error', True
    if not isinstance(error, RuntimeError):
        return None
    message = str(error)
    if message.startswith(('Unexpected reasoning channel', 'Full GPU serving not established',
                           'Unexpected served context', 'Actual server NumThreads=',
                           'Missing terminal response metadata', 'Response content is not text',
                           'Unsupported finish reason', 'Reported completion tokens are negative',
                           'Reported input tokens exceed served context')):
        return 'serving_or_response_contract_error', True
    if message.startswith(('Complete task exceeds context', 'Complete ', 'Engineering probe exceeds context')):
        return 'context_blocked', False
    if message.startswith('A prior response or attempt is preserved but not committed'):
        return 'preserved_uncommitted_attempt', True
    return None


def diagnostic(record, path, committed):
    raw = record.get('raw_response') or {}
    choices = raw.get('choices') or []
    message = raw.get('message') or (choices[0].get('message', {}) if choices else {})
    content = message.get('content', record.get('raw_content', record.get('response', '')))
    content = content if isinstance(content, str) else ''
    thinking = {key: message.get(key) for key in ('thinking', 'reasoning', 'reasoning_content')
                if message.get(key)}
    usage = record.get('usage') or raw.get('usage') or {}
    prompt = usage.get('prompt_tokens', raw.get('prompt_eval_count'))
    completion = usage.get('completion_tokens', raw.get('eval_count'))
    finish = record.get('finish_reason', raw.get('done_reason'))
    if finish is None and choices:
        finish = choices[0].get('finish_reason')
    request = record.get('request', {})
    budget = request.get('max_tokens')
    response = record.get('response')
    return {'record_file': str(path), 'committed_call': committed,
            'parsed_valid': record.get('score', {}).get('valid') if committed else None,
            'empty_content': not content.strip(),
            'extracted_response_empty': not response.strip() if isinstance(response, str) else None,
            'content_characters': len(content),
            'reasoning_channel_present': bool(thinking),
            'reasoning_channel_characters': {key: len(str(value)) for key, value in thinking.items()},
            'inline_think_markup': '<think>' in content or '</think>' in content,
            'finish_reason': finish, 'length_finish': finish == 'length',
            'prompt_tokens': prompt, 'completion_tokens': completion,
            'requested_max_tokens': budget,
            'completion_reaches_budget': completion >= budget if completion is not None and budget is not None else None,
            'wire_think': record.get('wire_request', {}).get('think'),
            'latency_seconds': record.get('latency_seconds'),
            'server_load_seconds': raw.get('load_duration', 0) / 1e9,
            'server_total_seconds': raw.get('total_duration', 0) / 1e9,
            'response_status': record.get('response_status', 'committed' if committed else None),
            'error': record.get('error')}


def audit_output(output):
    output = Path(output)
    rows = []
    committed = set()
    for path in sorted((output / 'calls').glob('*.json')):
        committed.add(path.name)
        rows.append(diagnostic(json.loads(path.read_text()), path, True))
    for path in sorted((output / 'received').glob('*.json')):
        if path.name not in committed:
            rows.append(diagnostic(json.loads(path.read_text()), path, False))
    attempts = [json.loads(path.read_text())
                for path in sorted((output / 'received_attempts').glob('*/attempt_*.json'))]
    attempt_diagnostics = [diagnostic(record, '', False) for record in attempts]
    summary = {'committed_calls': sum(r['committed_call'] for r in rows),
               'uncommitted_received_responses': sum(not r['committed_call'] for r in rows),
               'parse_valid_calls': sum(r['parsed_valid'] is True for r in rows),
               'parse_invalid_calls': sum(r['parsed_valid'] is False for r in rows),
               'empty_content_responses': sum(r['empty_content'] for r in rows),
               'length_finish_responses': sum(r['length_finish'] for r in rows),
               'reasoning_channel_responses': sum(r['reasoning_channel_present'] for r in rows),
               'inline_think_markup_responses': sum(r['inline_think_markup'] for r in rows),
               'completion_reaches_budget_responses': sum(r['completion_reaches_budget'] is True for r in rows),
               'prompt_tokens_total': sum(r['prompt_tokens'] or 0 for r in rows),
               'completion_tokens_total': sum(r['completion_tokens'] or 0 for r in rows),
               'max_prompt_tokens': max((r['prompt_tokens'] or 0 for r in rows), default=0),
               'max_completion_tokens': max((r['completion_tokens'] or 0 for r in rows), default=0),
               'load_seconds_in_received_responses': sum(r['server_load_seconds'] for r in rows),
               'transport_attempt_records': len(attempts),
               'prompt_tokens_across_all_preserved_attempts': sum(r['prompt_tokens'] or 0 for r in attempt_diagnostics),
               'completion_tokens_across_all_preserved_attempts': sum(r['completion_tokens'] or 0 for r in attempt_diagnostics),
               'transport_seconds_across_all_preserved_attempts': sum(r['attempt']['elapsed_seconds'] for r in attempts),
               'attempt_accounting_note': 'Attempt totals include retries; selected-response totals above do not. HTTP failures may have no token usage.',
               'rows': rows}
    write_json(output / 'technical_audit.json', summary)
    return summary


def run_engineering_probe(model, probe, spec, output, client, counter, module):
    output.mkdir(parents=True, exist_ok=True)
    messages = probe['messages']
    inf = spec['inference']
    budget = probe['step'].get('max_tokens', inf['max_tokens'])
    estimated = sum(counter.count(m['content']) for m in messages) + inf['template_margin_tokens']
    estimated += inf['per_message_margin_tokens'] * len(messages) + budget
    if estimated > inf['context_tokens']:
        raise RuntimeError(f'Engineering probe exceeds context: {estimated} > {inf["context_tokens"]}; no truncation.')
    payload = {'model': model, 'messages': messages, 'temperature': inf['temperature'],
               'max_tokens': budget, 'reasoning_effort': inf['reasoning_effort']}
    identity = 'engineering__' + probe['task'] + '__' + probe['step']['id']
    path = output / 'calls' / f'{identity}.json'
    if path.exists():
        record = json.loads(path.read_text())
        if record['request'] != payload or record['probe'] != probe:
            raise RuntimeError('Resumed engineering probe differs from the prepared synthetic-history probe.')
        return {'new_calls': 0, 'logical_calls': 1, 'status': 'complete'}
    if not client.serving_checked:
        client.serving()
    reply = client.call(payload, identity)
    record = {'model': model, 'task': probe['task'], 'synthetic': True,
              'scope': 'synthetic_history_engineering_probe', 'probe': probe,
              'request': payload, 'step': probe['step'], 'estimated_context_tokens': estimated,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **reply}
    record['score'] = module.score_step(probe['step'], reply['response'])
    write_json(path, record)
    return {'new_calls': 1, 'logical_calls': 1, 'status': 'complete'}


def run_pilot(manifest, model, output, client, counter, module_loader):
    """Injectable core: tests use fake clients; the CLI separately requires Slurm/backend."""
    spec = model_spec(validate_pilot(manifest, model), model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (('pilot_manifest.json', manifest), ('tokenizer.json', counter.metadata)):
        path = output / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f'Existing {name} differs; use a new pilot output.')
        write_json(path, value)
    started = time.perf_counter()
    actual_before = client.actual_calls
    summary = {'scope': 'technical_pilot', 'formal_evaluation': False, 'model': model,
               'effects_computed': False, 'expected_real_prefix_calls': 186,
               'expected_synthetic_engineering_calls': 3, 'tasks': [], 'engineering_probes': [],
               'warmup': {'record_file': str(output / 'warmup.json'),
                          'counts_as_logical_pilot_call': False,
                          'resource_cost_included_in_job_accounting': True},
               'inference_settings': spec['inference']}
    stop_model = None

    def perform(task_output, action):
        nonlocal stop_model
        client.output = task_output
        task_output.mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        try:
            result = action()
        except (RuntimeError, urllib.error.URLError, TimeoutError) as error:
            known = known_failure(error)
            if known is None:
                raise
            kind, stop = known
            result = {'status': 'failed', 'failure_kind': kind,
                      'error_type': type(error).__name__, 'error': str(error)}
            if stop:
                stop_model = result.copy()
            write_json(task_output / 'failure.json', result)
        result['wall_seconds_this_invocation'] = time.perf_counter() - began
        result['audit'] = audit_output(task_output)
        return result

    for entry in manifest['tasks']:
        task = entry['task']
        if stop_model:
            result = {'status': 'not_attempted_model_stopped', 'reason': stop_model['error']}
        else:
            bundle = json.loads(Path(entry['material_file']).read_text())
            task_output = output / task
            result = perform(task_output, lambda: run_episodes(
                model, bundle, module_loader(task), spec, task_output, client, counter))
        summary['tasks'].append({'task': task, **result})
        write_json(output / 'pilot_progress.json', summary)
    for probe in manifest['engineering_probes']:
        if stop_model:
            result = {'status': 'not_attempted_model_stopped', 'reason': stop_model['error']}
        else:
            task_output = output / 'engineering_probes' / probe['task']
            result = perform(task_output, lambda: run_engineering_probe(
                model, probe, spec, task_output, client, counter, module_loader(probe['task'])))
        summary['engineering_probes'].append({'task': probe['task'], 'synthetic': True, **result})
        write_json(output / 'pilot_progress.json', summary)
    all_results = summary['tasks'] + summary['engineering_probes']
    summary['status'] = ('complete' if all(r['status'] == 'complete' for r in all_results) else 'incomplete')
    summary['model_stop_reason'] = stop_model
    summary['real_prefix_committed_calls'] = sum(r.get('audit', {}).get('committed_calls', 0) for r in summary['tasks'])
    summary['synthetic_engineering_committed_calls'] = sum(r.get('audit', {}).get('committed_calls', 0) for r in summary['engineering_probes'])
    summary['actual_transport_calls_this_invocation'] = client.actual_calls - actual_before
    summary['wall_seconds_this_invocation'] = time.perf_counter() - started
    summary['interpretation'] = ('Completion means calls were recorded, not that response format, thinking mode, or behavioral validity passed. '
                                 'Review task technical_audit.json and raw records. Synthetic engineering probes are never research observations. '
                                 'Warmup/model loading is excluded from logical calls and included in Slurm resource measurements.')
    write_json(output / 'pilot_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True, help='Batch root; the model slug is appended.')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    spec = validate_pilot(manifest, args.model)
    validate_serving_environment(spec, args.model)
    from .registry import module_for
    from scripts.experiments.cvlt_context_controls_20261007.run import Client
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter
    output = args.output / args.model.replace(':', '__')
    view = model_spec(spec, args.model)
    client = Client(args.model, view, output / manifest['tasks'][0]['task'])
    with Counter(args.model) as counter:
        result = run_pilot(manifest, args.model, output, client, counter, module_for)
    print(json.dumps({key: result[key] for key in ('model', 'status', 'real_prefix_committed_calls',
                     'synthetic_engineering_committed_calls', 'actual_transport_calls_this_invocation',
                     'wall_seconds_this_invocation', 'model_stop_reason')}, ensure_ascii=False), flush=True)
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
