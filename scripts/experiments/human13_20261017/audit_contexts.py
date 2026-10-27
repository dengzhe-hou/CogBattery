"""Inspect every authorized local model vocabulary/context, without inference."""
import argparse
import csv
import ctypes
import json
from pathlib import Path

from .common import write_json
from .preflight import inspect_task
from .run import model_spec


def meta_value(counter, key):
    # Only short scalar metadata are requested, never token arrays or model tensors.
    buffer = ctypes.create_string_buffer(512)
    size = counter.lib.llama_model_meta_val_str(counter.pointer, key.encode(), buffer, len(buffer))
    if size < 0:
        return None
    if size >= len(buffer):
        raise ValueError(f'Expected short scalar GGUF metadata: {key}')
    return buffer.value.decode()


def local_limits(counter):
    architecture = meta_value(counter, 'general.architecture')
    # vocab_only does not initialize inference hparams: n_ctx_train() reports 0.
    # The context declaration is still available in the original GGUF metadata.
    vocab_only_hparam_context = int(counter.lib.llama_model_n_ctx_train(counter.pointer))
    keys = ('context_length', 'attention.sliding_window', 'rope.scaling.type',
            'rope.scaling.factor', 'rope.scaling.original_context_length', 'rope.freq_base')
    fields = {f'{architecture}.{key}': meta_value(counter, f'{architecture}.{key}') for key in keys}
    train_context = int(fields[f'{architecture}.context_length'] or 0)
    manifest = json.loads(counter.manifest.read_text())
    params = {}
    for layer in manifest['layers']:
        if layer['mediaType'] == 'application/vnd.ollama.image.params':
            path = counter.blob.parent / layer['digest'].replace(':', '-')
            params.update(json.loads(path.read_text()))
    return {'architecture': architecture, 'gguf_declared_context_tokens': train_context,
            'llama_cpp_n_ctx_train_vocab_only': vocab_only_hparam_context,
            'gguf_context_fields': fields, 'ollama_stored_num_ctx': params.get('num_ctx'),
            'interpretation': 'Local GGUF training/context metadata, not a runtime serving or long-context accuracy test.'}


def audit(manifest_path, output):
    from .registry import module_for
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    bundles = {t['task']: json.loads(Path(t['material_file']).read_text()) for t in manifest['tasks']}
    rows, results = [], []
    for model in manifest['spec']['models']:
        inf = model_spec(manifest['spec'], model)['inference']
        result = {'model': model, 'configured_context_tokens': inf['context_tokens'],
                  'scope': 'CPU_vocabulary_only_scripted_format_check', 'model_calls': 0,
                  'weights_or_inference_context_loaded': False, 'tasks': {}}
        try:
            with Counter(model) as counter:
                result['tokenizer'] = counter.metadata
                result['limits'] = local_limits(counter)
                declared = result['limits']['gguf_declared_context_tokens']
                for task, bundle in bundles.items():
                    check = inspect_task(bundle, module_for(task), counter, inf)
                    peak = check['peak_scripted_history']
                    need = peak['estimated_tokens_with_margins'] if peak else None
                    check['fits_gguf_declared_context'] = None if need is None or declared <= 0 else need <= declared
                    result['tasks'][task] = check
                    rows.append({'model': model, 'task': task, 'source_status': bundle['status'],
                                 'configured_context': inf['context_tokens'], 'gguf_declared_context': declared,
                                 'scripted_peak_raw_tokens': peak['raw_tokens'] if peak else None,
                                 'scripted_peak_budget': need, 'context_check_status': check['context_check_status'],
                                 'fits_gguf_declared_context': check['fits_gguf_declared_context'],
                                 'inspected_steps': check['inspected_steps']})
                result['status'] = 'inspected'
        except (RuntimeError, FileNotFoundError, KeyError) as error:
            # A local unsupported/missing model stays an explicit failed inspection.
            # No alternative tokenizer or remote model is silently substituted.
            result['status'] = 'inspection_failed'
            result['error'] = f'{type(error).__name__}: {error}'
        write_json(output / 'models' / (model.replace(':', '__') + '.json'), result)
        results.append(result)
        write_json(output / 'progress.json', {'inspected_models': len(results), 'expected_models': len(manifest['spec']['models']),
                   'model_calls': 0, 'last_model': model, 'last_status': result['status']})
        print(f'{len(results)}/{len(manifest["spec"]["models"])} {model}: {result["status"]}', flush=True)
    summary = {'manifest': str(manifest_path.resolve()), 'spec_revision': manifest['spec']['revision'],
               'scope': 'CPU_vocabulary_only; scripted format histories, no real responses',
               'model_calls': 0, 'resource_requests': 0, 'models': results,
               'limitations': ['CVLT without materials is not checked.',
                   'DRM previews are not the full word/sentence study until its word pool is confirmed.',
                   'GGUF-declared context is not proof of quality or current server configuration.',
                   'Raw-content token counts use configured template margins, not a rendered chat-template measurement.',
                   'Short scripted replies are not a guarantee for real output lengths; runtime checks remain.']}
    write_json(output / 'context_audit.json', summary)
    if rows:
        with (output / 'context_by_model_task.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.output)
    failed = [r['model'] for r in result['models'] if r['status'] != 'inspected']
    if failed:
        raise SystemExit(f'Inspection incomplete for: {failed}; see recorded errors, no inference was run.')


if __name__ == '__main__':
    main()
