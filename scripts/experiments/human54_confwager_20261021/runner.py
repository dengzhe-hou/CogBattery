"""Confidence token-probability and wager-label reruns (approved 2026-10-21; plan
review-stage/rev_20261021/MDX_RERUN_PLAN.md, spec SPEC.md in this directory).

Requests come from the frozen unified entry (human13_20261017.protocol.build_request)
with the same materials, histories, budgets and options. A condition changes only:
- cal_original / wag_original: nothing (same-hardware baseline);
- cal_logprobs: cal_answer requests also ask for token log-probabilities;
- wag_AB / wag_BA: the wager rule names labels A and B for one/two sweets (AB) or
  two/one sweets (BA); wager replies are A or B, and later turns show them verbatim.
Replies are parsed by the frozen human13 client.parse_reply. One request per slot,
temperature zero, as in the formal run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import time
import urllib.request
from zoneinfo import ZoneInfo

from scripts.experiments.human13_20261017.client import HardResponseError, parse_reply
from scripts.experiments.human13_20261017.protocol import build_request
from scripts.experiments.human13_20261017.compact_wag import WAGER_RULE
from scripts.experiments.human13_20261017.registry import module_for

ROOT = Path(__file__).resolve().parents[3]
TASK = {'cal_original': 'confidence_calibration', 'cal_logprobs': 'confidence_calibration',
        'wag_original': 'post_decision_wagering', 'wag_AB': 'post_decision_wagering',
        'wag_BA': 'post_decision_wagering'}
WAGER_PHASE = 'Current phase: wager only. Reply only 1 or 2; do not include history labels.'
LABELS = {
    'wag_AB': ('Wager one or two sweets on the answer you just gave. Do not change your answer. '
               'Respond A to wager one sweet or B to wager two sweets. Respond A or B only.', {'A': 1, 'B': 2}),
    'wag_BA': ('Wager one or two sweets on the answer you just gave. Do not change your answer. '
               'Respond A to wager two sweets or B to wager one sweet. Respond A or B only.', {'A': 2, 'B': 1}),
}
TOP_LOGPROBS = 20
# A reply whose prompt is this close to the served context may have been truncated by the server.
CONTEXT_GUARD_TOKENS = 64


class ZeroCounter:
    """Stand-in when llama_cpp is unavailable: the frozen estimate then never raises
    ContextOverflow, so the server-side guard below checks the actual prompt length instead.
    Formal CAL/WAG direct-answer requests peaked at 3,338 of 4,096 tokens."""
    def count(self, text):
        return 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def transform(condition, step, wire):
    """Apply the condition to a request built by the frozen entry. Returns a new wire."""
    wire = deepcopy(wire)
    if condition == 'cal_logprobs' and step['kind'] == 'cal_answer':
        wire['logprobs'] = True
        wire['top_logprobs'] = TOP_LOGPROBS
    if condition in LABELS:
        rule, _ = LABELS[condition]
        content = wire['messages'][-1]['content']
        if WAGER_RULE in content:
            content = content.replace(WAGER_RULE, rule)
        if step['kind'] == 'wager':
            if content.count(WAGER_PHASE) != 1 or wire['format'] != {'type': 'string', 'enum': ['1', '2']}:
                raise ValueError('Unexpected wager request; inspect before relabelling')
            content = content.replace(WAGER_PHASE, WAGER_PHASE.replace('Reply only 1 or 2', 'Reply only A or B'))
            wire['format'] = {'type': 'string', 'enum': ['A', 'B']}
        wire['messages'][-1]['content'] = content
    return wire


def wager_amount(condition, response):
    if condition in LABELS:
        return LABELS[condition][1].get(response)
    return {'1': 1, '2': 2}.get(response)


def run(model, condition, bundle, spec, counter, send, output, deadline=None):
    """Walk every episode in material order; one request per slot."""
    task = TASK[condition]
    module = module_for(task)
    output.mkdir(parents=True, exist_ok=True)
    histories, stopped, counts = {}, set(), {'sent': 0, 'valid': 0, 'invalid': 0, 'not_run': 0, 'resumed': 0}
    for episode in bundle['episodes']:
        parent = episode.get('parent')
        history = list(histories[parent]) if parent else []
        blocked = parent in stopped
        for step in episode['steps']:
            identity = f"{episode['id']}__{step['id']}"
            path = output / 'calls' / f'{identity}.json'
            if blocked:
                counts['not_run'] += 1
                continue
            req = build_request(model, task, episode, step, history, spec, module, counter)
            wire = transform(condition, step, req.wire)
            if path.exists():
                record = json.loads(path.read_text())
                if record['wire_request'] != wire:
                    raise RuntimeError(f'Saved record differs from the rebuilt request: {path}')
                counts['resumed'] += 1
            else:
                if deadline is not None and time.time() + 300 > deadline:
                    blocked = True
                    counts['not_run'] += 1
                    continue
                started = time.perf_counter()
                raw = send(wire)
                latency = time.perf_counter() - started
                try:
                    parsed = parse_reply(wire, raw)
                    if raw.get('prompt_eval_count', 0) + wire['options']['num_predict'] > wire['options']['num_ctx'] - CONTEXT_GUARD_TOKENS:
                        raise HardResponseError('Prompt within the context guard; the server may have truncated it')
                except HardResponseError as error:
                    (output / 'hard_errors').mkdir(exist_ok=True)
                    (output / 'hard_errors' / f'{identity}.json').write_text(json.dumps(
                        {'wire_request': wire, 'raw_response': raw, 'error': str(error)}, ensure_ascii=False))
                    blocked = True
                    counts['not_run'] += 1
                    continue
                record = {**parsed, 'raw_response': raw, 'wire_request': wire, 'latency_seconds': latency,
                          'model': model, 'task': task, 'condition_variant': condition, 'episode_id': episode['id'],
                          'step': deepcopy(step), 'context': req.context,
                          'received_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                          'host': os.uname().nodename}
                if step['kind'] == 'wager':
                    record['wager_amount'] = wager_amount(condition, record['response'])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False))
                counts['sent'] += 1
            counts['valid' if record['technical_valid'] else 'invalid'] += 1
            history.append({'step': step, 'response': record['response'], 'score': None})
        histories[episode['id']] = history
        if blocked:
            stopped.add(episode['id'])
    (output / 'summary.json').write_text(json.dumps(dict(model=model, condition=condition, **counts), indent=2))
    return counts


def ollama_sender(host, timeout):
    def send(wire):
        body = json.dumps(wire, ensure_ascii=False).encode()
        request = urllib.request.Request(f'http://{host}/api/chat', data=body,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    return send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json')
    parser.add_argument('--model', required=True)
    parser.add_argument('--condition', action='append', required=True, choices=sorted(TASK))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default=os.environ.get('OLLAMA_HOST', '127.0.0.1:11434'))
    parser.add_argument('--deadline-epoch', type=float)
    parser.add_argument('--zero-counter', action='store_true', help='No llama_cpp: rely on the server-side context guard')
    a = parser.parse_args()
    manifest = json.loads(a.manifest.read_text())
    spec = json.loads((ROOT / 'scripts/experiments/human13_20261017/SPEC.json').read_text())
    if spec['models'] and a.model not in spec['models']:
        raise ValueError('Model outside the panel')
    send = ollama_sender(a.host, spec['inference']['request_timeout_seconds'])
    if not a.zero_counter:
        from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter as TokenCounter
    with (ZeroCounter() if a.zero_counter else TokenCounter(a.model)) as counter:
        for condition in a.condition:
            entry = next(t for t in manifest['tasks'] if t['task'] == TASK[condition])
            bundle = json.loads(Path(entry['material_file']).read_text())
            out = a.output / a.model.replace(':', '__') / condition
            print(json.dumps(run(a.model, condition, bundle, spec, counter, send, out, a.deadline_epoch)), flush=True)


if __name__ == '__main__':
    main()
