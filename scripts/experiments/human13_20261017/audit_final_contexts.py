"""Offline context audit of the final r7 protocol: 54 models x 13 paradigms.

Builds every r7 request with protocol.build_request (ARRAY_PROMPT, revised wager
text, final budgets) and the model's own GGUF vocabulary (vocab-only), checking
input + 128 + 8 per message + the step's total allowance <= served context.
Scenario A: past replies are the scripted legal fixtures (preflight.fixture_reply).
Scenario B (direct models only): every past reply is filler of exactly its step's
output allowance, the longest reply decision (a) can put into the history. For the
native models a final can approach 8192 tokens after an early natural closure, so
B is not meaningful there and is not computed. No model call, no server.
"""
import argparse
import json
from pathlib import Path
import time

from .budgets import is_native, total_tokens
from .common import ROOT, write_json
from .preflight import fixture_reply
from .protocol import ContextOverflow, build_request
from .registry import module_for
from .scoring import wrap_module

DEFAULT_MANIFEST = ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json'
DEFAULT_OUT = ROOT / 'refine-logs/human54_unified_entry_20261018/CONTEXT_AUDIT.json'


class Filler:
    """Text of an exact token count under one vocabulary (calibrated, cached)."""

    def __init__(self, counter):
        self.counter, self.cache = counter, {}

    def __call__(self, n):
        if n not in self.cache:
            words = n
            text = ' '.join(['x'] * words)
            while self.counter.count(text) > n:
                words -= 1
                text = ' '.join(['x'] * words)
            while self.counter.count(text) < n:
                text += ' x'
            self.cache[n] = text
        return self.cache[n]


def walk(model, task, bundle, spec, counter, reply_for):
    module = wrap_module(module_for(task))
    histories, peak, first_overflow, steps, dropped, legacy_peak = {}, None, None, 0, 0, None
    for episode in bundle['episodes']:
        history = list(histories[episode['parent']]) if episode.get('parent') else []
        for step in episode['steps']:
            steps += 1
            try:
                req = build_request(model, task, episode, step, history, spec, module, counter,
                                    reserve_variant='grammar_ws_w22' if is_native(model, spec['inference']) else None)
            except ContextOverflow as error:
                if first_overflow is None:
                    first_overflow = {'episode_id': episode['id'], 'step_id': step['id'], 'error': str(error)}
                req = None
            if req is not None:
                if req.context['history_pairs_dropped']:
                    dropped += req.context['history_pairs_dropped']
                    history = history[req.context['history_pairs_dropped']:]
                estimate = req.context.get('estimated_input_plus_output')
                legacy = req.context.get('estimated_input_plus_output_not_enforced')
                if legacy is not None:
                    legacy_peak = max(legacy_peak or 0, legacy)
                if estimate is not None and (peak is None or estimate > peak['estimated_input_plus_output']):
                    peak = {'episode_id': episode['id'], 'step_id': step['id'],
                            'estimated_input_plus_output': estimate, 'context_tokens': req.context['context_tokens'],
                            'raw_content_tokens': req.context['raw_content_tokens'], 'output_allowance': req.total}
            reply = reply_for(step, task)
            history.append({'step': step, 'response': reply, 'score': module.score_step(step, reply)})
        histories[episode['id']] = history
    return {'steps': steps, 'peak': peak, 'first_overflow': first_overflow,
            'legacy_fb_history_pairs_dropped': dropped,
            'legacy_fb_peak_input_plus_output_not_enforced': legacy_peak}


def audit(manifest_path=DEFAULT_MANIFEST, models=None, tasks=None):
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter as TokenCounter
    manifest = json.loads(Path(manifest_path).read_text())
    spec = manifest['spec']
    models = models or spec['models']
    tasks = tasks or spec['tasks']
    bundles = {t['task']: json.loads(Path(t['material_file']).read_text()) for t in manifest['tasks'] if t['task'] in tasks}
    rows, started = [], time.perf_counter()
    for model in models:
        native = is_native(model, spec['inference'])
        with TokenCounter(model) as counter:
            filler = Filler(counter)
            for task in tasks:
                a = walk(model, task, bundles[task], spec, counter, lambda step, t: fixture_reply(step))
                row = {'model': model, 'task': task, 'native_think': native, 'scenario_A_scripted_legal': a}
                if not native:
                    row['scenario_B_replies_fill_allowance'] = walk(
                        model, task, bundles[task], spec, counter,
                        lambda step, t: filler(total_tokens(t, step, model, spec['inference'])))
                rows.append(row)
        print(json.dumps({'model': model, 'elapsed_s': round(time.perf_counter() - started, 1),
                          'A_overflows': sum(bool(r['scenario_A_scripted_legal']['first_overflow'])
                                             for r in rows if r['model'] == model),
                          'B_overflows': sum(bool(r.get('scenario_B_replies_fill_allowance', {}).get('first_overflow'))
                                             for r in rows if r['model'] == model)}), flush=True)
    return {'scope': 'offline_vocabulary_only_context_audit_final_r7', 'model_calls': 0,
            'rule': 'input tokens + 128 + 8 x messages + step total allowance <= served context (4096 direct, 12288 native); '
                    'false belief keeps its legacy input-only rule with persistent pair dropping',
            'manifest': str(manifest_path), 'spec_revision': spec['revision'], 'rows': rows,
            'elapsed_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--model', action='append')
    parser.add_argument('--task', action='append')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    write_json(args.output, audit(args.manifest, args.model, args.task))


if __name__ == '__main__':
    main()
