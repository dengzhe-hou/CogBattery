"""FROZEN r6 runner (copy of run.py before the 2026-10-18 r7 merge).

Only the frozen technical-pilot tools (pilot_run.py) and their tests import it.
The formal r7 entry is run.py. Do not use this module for new inference.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

from .common import default_messages, validate_task, write_json


def check_authorized(spec, task, model, bundle):
    if not spec['authorization']['inference_authorized']:
        raise RuntimeError('Preparation only: user explicitly prohibited experiments and resource requests.')
    if model not in spec['models']:
        raise ValueError(f'Model is outside the authorized panel: {model}')
    if task not in spec['tasks']:
        raise ValueError(f'Task is outside the 13-task scope: {task}')
    if bundle['status'] != 'ready':
        raise RuntimeError(f'Material/protocol unresolved for {task}: ' + '; '.join(bundle['blockers']))


def model_spec(spec, model):
    view = copy.deepcopy(spec)
    view['inference']['context_tokens'] = spec['inference']['contexts'].get(
        model, spec['inference']['context_tokens'])
    return view


def fit_messages(module, episode, step, history, counter, inf):
    builder = getattr(module, 'messages', default_messages)
    messages = builder(episode, step, history)
    raw_tokens = lambda ms: sum(counter.count(m['content']) for m in ms)
    margin = inf['template_margin_tokens']
    per_message = inf['per_message_margin_tokens']
    if episode.get('context_policy') == 'legacy_strachan':
        # The completed 20 used this same input-only rule and persistent pair dropping.
        from scripts.experiments.same_stimulus_20261011.run import fit_session
        head = messages[:1] if messages[0]['role'] == 'system' else []
        body = messages[len(head):-1]
        pairs = [body[i:i+2] for i in range(0, len(body), 2)]
        fitted, raw, dropped = fit_session(head, pairs, messages[-1], raw_tokens,
                                          inf['context_tokens'], margin, per_message,
                                          item_id=step['id'])
        return fitted, {'raw_content_tokens': raw, 'history_pairs_dropped': dropped,
                        'policy': 'unchanged_legacy_strachan_input_rule'}
    raw = raw_tokens(messages)
    output_budget = step.get('max_tokens', inf['max_tokens'])
    estimate = raw + margin + per_message * len(messages) + output_budget
    if estimate > inf['context_tokens']:
        raise RuntimeError(f'Complete {episode["id"]}/{step["id"]} prompt/output does not fit '
                           f'{inf["context_tokens"]} tokens (estimated {estimate}); no history deleted.')
    return messages, {'raw_content_tokens': raw, 'history_pairs_dropped': 0,
                      'policy': 'full_prompt_plus_output_budget'}


def run_episodes(model, bundle, module, spec, output, client, counter):
    """Core exercised only with synthetic clients by offline tests this turn."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    task = bundle['episodes'][0]['task']
    expected = validate_task(task, bundle)
    for name, value in (('SPEC.json', spec), ('materials.json', bundle)):
        target = output / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise RuntimeError(f'Existing {name} differs; use a separate output for a new protocol.')
        write_json(target, value)
    # Reject already-known late overflow before spending any model calls. Actual
    # responses still get checked at each step; scripted replies are not a guarantee.
    from .preflight import inspect_task
    context_check = inspect_task(bundle, module, counter, spec['inference'])
    write_json(output / 'context_preflight.json', context_check)
    legacy = all(e.get('context_policy') == 'legacy_strachan' for e in bundle['episodes'])
    if context_check['first_untrimmed_overflow'] and not legacy:
        write_json(output / 'progress.json', {'task': task, 'model': model,
                   'status': 'context_blocked', 'new_calls': 0})
        raise RuntimeError('Complete task exceeds context in scripted precheck; no history deleted, no calls sent.')
    histories = {}
    records = []
    new_calls = 0
    started = time.perf_counter()
    inf = spec['inference']
    for episode in bundle['episodes']:
        parent = episode.get('parent')
        history = list(histories[parent]) if parent else []
        for step in episode['steps']:
            messages, context = fit_messages(module, episode, step, history, counter, inf)
            dropped = context['history_pairs_dropped']
            if dropped:
                history = history[dropped:]
            identity = episode['id'] + '__' + step['id']
            payload = {'model': model, 'messages': messages,
                       'temperature': inf['temperature'],
                       'max_tokens': step.get('max_tokens', inf['max_tokens']),
                       'reasoning_effort': inf['reasoning_effort']}
            path = output / 'calls' / f'{identity}.json'
            if path.exists():
                record = json.loads(path.read_text())
                if (record['request'] != payload or record['step'] != step
                        or record.get('presentation') != episode.get('presentation')):
                    raise RuntimeError(f'Resumed request/material differs: {identity}')
            else:
                if not client.serving_checked:
                    client.serving()
                reply = client.call(payload, identity)
                record = {'model': model, 'task': episode['task'], 'episode_id': episode['id'],
                          'condition': episode['condition'], 'parent': parent,
                          'presentation': episode.get('presentation'),
                          'step': step, 'request': payload, 'context': context,
                          'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **reply}
                record['score'] = module.score_step(step, reply['response'])
                write_json(path, record)
                new_calls += 1
            # Re-score on resume so the actual past response, never a gold answer, drives history.
            record['score'] = module.score_step(step, record['response'])
            history.append({'step': step, 'response': record['response'], 'score': record['score']})
            records.append(record)
            write_json(output / 'progress.json', {'task': task, 'model': model,
                       'completed_calls': len(records), 'expected_calls': expected,
                       'new_calls': new_calls})
        histories[episode['id']] = history
    if len(records) != expected:
        raise RuntimeError('Completed call count differs from the material manifest')
    summary = {'model': model, 'task': task, 'status': 'complete',
               'logical_calls': len(records), 'new_calls': new_calls,
               'reused_calls': len(records)-new_calls, 'wall_seconds': time.perf_counter()-started}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    spec = manifest['spec']
    # Check the user's no-inference constraint before importing/constructing any real client.
    if not spec['authorization']['inference_authorized']:
        raise RuntimeError('Preparation only: no pilot or formal inference has been authorized.')
    entry = next(t for t in manifest['tasks'] if t['task'] == args.task)
    bundle = json.loads(Path(entry['material_file']).read_text())
    check_authorized(spec, args.task, args.model, bundle)
    if args.task == 'false_belief' and args.model in spec['reuse']['false_belief']:
        from .tom_mc import load_reused_fb
        existing = [r for r in load_reused_fb(manifest['config'], spec) if r['model'] == args.model]
        print(json.dumps({'status': 'reuse_existing', 'model': args.model,
                          'verified_calls': len(existing),
                          'source': spec['reuse']['false_belief'][args.model]}))
        return
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Real inference requires a separately authorized allocated Slurm GPU.')
    backend = spec['backends'].get(args.model, spec['backends']['default'])
    selected = shutil.which('ollama')
    if not selected or Path(selected).resolve() != Path(backend).resolve():
        raise RuntimeError('Serving binary differs from the recorded repaired backend.')
    from .registry import module_for
    from scripts.experiments.cvlt_context_controls_20261007.run import Client
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter
    view = model_spec(spec, args.model)
    output = args.output / args.model.replace(':', '__') / args.task
    client = Client(args.model, view, output)
    with Counter(args.model) as counter:
        result = run_episodes(args.model, bundle, module_for(args.task), view, output, client, counter)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
