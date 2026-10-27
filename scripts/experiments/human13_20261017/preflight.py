"""CPU vocabulary-only prompt inspection with labeled scripted replies, no inference."""
import argparse
import json
from pathlib import Path

from .common import default_messages, write_json, step_budget
from .run import model_spec


def fixture_reply(step):
    """Formatting fixture only. Never a model response or research observation."""
    if step['kind'] in ('cvlt_recall', 'cvlt_recall_followup'):
        return json.dumps(step['gold'])
    if step['kind'] == 'cvlt_recognition':
        return 'YES' if step['gold'] else 'NO'
    if step['kind'] == 'confidence':
        return '50%'
    if step['kind'] == 'wager':
        return '1'
    if step['kind'] == 'drm_rating':
        return '4'
    if step['kind'] == 'drm_report':
        return 'GUESS'
    gold = step['gold']
    if isinstance(gold, bool):
        return 'OLD' if gold else 'NEW'
    if isinstance(gold, list):
        return ' '.join(map(str, gold))
    return str(gold) if gold is not None else 'Ready'


def inspect_task(bundle, module, counter, inf, reply_builder=None):
    reply_builder = reply_builder or fixture_reply
    histories = {}
    peak = None
    first_overflow = None
    n = 0
    for episode in bundle['episodes']:
        history = list(histories[episode['parent']]) if episode.get('parent') else []
        for step in episode['steps']:
            messages = getattr(module, 'messages', default_messages)(episode, step, history)
            raw = sum(counter.count(m['content']) for m in messages)
            budget = raw + inf['template_margin_tokens'] + inf['per_message_margin_tokens'] * len(messages)
            if episode.get('context_policy') != 'legacy_strachan':
                budget += step_budget(step, inf)
            row = {'episode_id': episode['id'], 'step_id': step['id'],
                   'raw_tokens': raw, 'estimated_tokens_with_margins': budget,
                   'context_tokens': inf['context_tokens']}
            if peak is None or budget > peak['estimated_tokens_with_margins']:
                peak = row
            if budget > inf['context_tokens'] and first_overflow is None:
                first_overflow = row
            reply = reply_builder(step)
            history.append({'step': step, 'response': reply, 'score': module.score_step(step, reply)})
            n += 1
        histories[episode['id']] = history
    check_status = ('not_checked_no_materials' if not n else
                    'overflow_detected' if first_overflow else 'no_overflow_in_scripted_check')
    return {'source_status': bundle['status'], 'context_check_status': check_status,
            'inspected_steps': n,
            'peak_scripted_history': peak, 'first_untrimmed_overflow': first_overflow,
            'interpretation': 'Scripted replies only; absence of overflow is not a guarantee for real outputs. '
                              'Legacy Strachan separately retains its authorized pair-dropping rule.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', action='append', help='Repeat to inspect selected tasks; default all constructed tasks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.model not in manifest['spec']['models']:
        raise ValueError('Model is not in the authorized panel')
    from .registry import module_for
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter
    inf = model_spec(manifest['spec'], args.model)['inference']
    result = {'scope': 'CPU_vocabulary_only_scripted_format_check', 'model': args.model,
              'model_calls': 0, 'weights_or_inference_context_loaded': False, 'tasks': {}}
    with Counter(args.model) as counter:
        result['tokenizer'] = counter.metadata
        for task in manifest['tasks']:
            if args.task and task['task'] not in args.task:
                continue
            bundle = json.loads(Path(task['material_file']).read_text())
            result['tasks'][task['task']] = inspect_task(bundle, module_for(task['task']), counter, inf)
    write_json(args.output, result)
    print(json.dumps({k: {'steps': v['inspected_steps'],
                         'context_check_status': v['context_check_status']}
                      for k, v in result['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
