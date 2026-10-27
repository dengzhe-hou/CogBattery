"""Check the complete CVLT transcript for all 54 local vocabularies, no inference."""
import argparse
import csv
import json
from pathlib import Path
import time

from .audit_contexts import local_limits
from .common import write_json
from .preflight import fixture_reply, inspect_task
from .run import model_spec


def empty_recall_reply(step):
    if step['kind'] in ('cvlt_recall', 'cvlt_recall_followup'):
        return '[]'
    return fixture_reply(step)


SCENARIOS = {
    'empty_recall_arrays': empty_recall_reply,
    'target_arrays_both_replies': fixture_reply,
}


def audit(manifest_path, output):
    from . import memory
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    spec = manifest['spec']
    entry = next(row for row in manifest['tasks'] if row['task'] == 'cvlt_word_list')
    bundle = json.loads(Path(entry['material_file']).read_text())
    if bundle['status'] != 'ready':
        raise ValueError('CVLT material must be complete before checking its contexts')
    results, rows = [], []
    start = time.monotonic()
    for model in spec['models']:
        inf = model_spec(spec, model)['inference']
        result = {'model': model, 'model_calls': 0, 'checks': {}}
        with Counter(model) as counter:
            result['tokenizer'] = counter.metadata
            result['limits'] = local_limits(counter)
            declared = result['limits']['gguf_declared_context_tokens']
            for scenario, reply_builder in SCENARIOS.items():
                check = inspect_task(bundle, memory, counter, inf, reply_builder)
                peak = check['peak_scripted_history']
                needed = peak['estimated_tokens_with_margins']
                check['fits_gguf_declared_context'] = None if declared <= 0 else needed <= declared
                result['checks'][scenario] = check
                rows.append({
                    'model': model, 'scenario': scenario,
                    'configured_context': inf['context_tokens'],
                    'gguf_declared_context': declared,
                    'peak_raw_tokens': peak['raw_tokens'], 'peak_budget': needed,
                    'peak_step': peak['step_id'],
                    'first_overflow_step': (check['first_untrimmed_overflow'] or {}).get('step_id'),
                    'fits_configured_context': needed <= inf['context_tokens'],
                    'fits_gguf_declared_context': check['fits_gguf_declared_context'],
                    'headroom_configured': inf['context_tokens'] - needed,
                    'inspected_steps': check['inspected_steps'],
                })
        write_json(output / 'models' / f'{model.replace(":", "__")}.json', result)
        results.append(result)
        write_json(output / 'progress.json', {'models_checked': len(results),
                   'models_expected': len(spec['models']), 'last_model': model, 'model_calls': 0})
        print(f'{len(results)}/{len(spec["models"])} {model}: ' + ', '.join(
            f'{name}={check["peak_scripted_history"]["estimated_tokens_with_margins"]}'
            for name, check in result['checks'].items()), flush=True)
    summary = {
        'manifest': str(manifest_path.resolve()), 'spec_revision': spec['revision'],
        'scope': 'CVLT only, complete retained histories; CPU vocabulary-only tokenization',
        'model_count': len(results), 'model_calls': 0, 'resource_requests': 0,
        'scenarios': {
            'empty_recall_arrays': 'Every recall and follow-up returns []; all other replies are legal fixtures.',
            'target_arrays_both_replies': 'Each recall and follow-up returns its complete target array. Repeating it on the follow-up is intentional; this is a length fixture, not a model result.',
        },
        'models': results, 'elapsed_offline_seconds': time.monotonic() - start,
        'limitations': [
            'No weight tensors, inference context, model service or GPU allocation.',
            'Scripted replies are not observations or upper bounds on arbitrary real outputs.',
            'Raw-content token counts plus inherited template/output margins, not rendered server templates.',
            'GGUF declarations do not demonstrate effective serving limits or long-context performance.',
            'No history removal, presentation changes, context increases or model exclusions.',
        ],
    }
    write_json(output / 'context_audit.json', summary)
    with (output / 'context_by_model.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.output)


if __name__ == '__main__':
    main()
