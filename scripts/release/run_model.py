"""Run one model on the paper-version battery with your own server.

    python -m scripts.release.run_model --materials results/paper_materials/manifest.json \
        --model qwen2.5:7b --backend ollama --base-url http://localhost:11434 \
        --output results/submissions

    python -m scripts.release.run_model --materials results/paper_materials/manifest.json \
        --model my-model --backend openai --base-url http://localhost:8000/v1 --api-key-env MY_KEY \
        --tokenizer org/my-model --output results/submissions

The run uses the paper's requests: temperature 0, one reply per step, the paper's per-step output
limits, JSON answer schemas, and the same history rules. The model is treated as a direct-answer
model (no native thinking). Interrupted runs resume from the records already written.

Comparability. The paper served every model with Ollama 0.18.0 and a 4,096-token context.
`--backend ollama` sends the same request body. `--backend openai` translates it to an
OpenAI-compatible chat request with a JSON-schema response format; the server's chat template,
context handling, and constrained decoding then decide the result, so scores are close to the
paper's protocol but not identical to it. The run manifest records which backend was used.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from scripts.experiments.human13_20261017 import client as paper_client
from scripts.experiments.human13_20261017.common import write_json
from scripts.experiments.human13_20261017.registry import module_for
from scripts.experiments.human13_20261017.run import live_spec, run_task, slug


class TokenCounter:
    """Counts tokens for the context check. A Hugging Face tokenizer counts exactly; without one,
    about four characters per token is assumed."""

    def __init__(self, tokenizer=None):
        self.name = tokenizer or 'approximate: 4 characters per token'
        self._tokenizer = None
        if tokenizer:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def count(self, text):
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return math.ceil(len(text) / 4)


class ServerClient:
    """Sends one request per step to an Ollama or OpenAI-compatible server. Replies are converted to
    the Ollama /api/chat shape so the paper's reply checks and scoring apply unchanged."""

    def __init__(self, model, backend, base_url, api_key, timeout):
        self.model, self.kind, self.timeout = model, backend, timeout
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.backend = {'kind': backend, 'base_url': self.base_url}
        self.model_digest = None
        self.serving_checked = True  # the paper's GPU and digest checks apply to its own servers only
        self.transport_starts = 0

    def _post(self, route, body):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        request = urllib.request.Request(self.base_url + route, data=json.dumps(body, ensure_ascii=False).encode(),
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _openai(self, wire):
        options = wire['options']
        body = {'model': self.model, 'messages': wire['messages'], 'temperature': options['temperature'],
                'top_p': options['top_p'], 'max_tokens': options['num_predict'], 'stream': False}
        if 'format' in wire:
            body['response_format'] = {'type': 'json_schema',
                                       'json_schema': {'name': 'answer', 'schema': wire['format'], 'strict': True}}
        if options.get('stop'):
            body['stop'] = options['stop']
        reply = self._post('/chat/completions', body)
        choice = reply['choices'][0]
        usage = reply.get('usage') or {}
        finish = {'stop': 'stop', 'length': 'length'}.get(choice.get('finish_reason'), choice.get('finish_reason'))
        return {'message': {'role': 'assistant', 'content': choice['message'].get('content') or ''},
                'done': True, 'done_reason': finish,
                'prompt_eval_count': usage.get('prompt_tokens'), 'eval_count': usage.get('completion_tokens'),
                'openai_response': reply}, body

    def send(self, wire, identity, folder):
        folder = Path(folder)
        dispatch, received = folder / 'dispatch' / f'{identity}.json', folder / 'received' / f'{identity}.json'
        if dispatch.exists() or received.exists():
            raise paper_client.PriorAttemptPreserved(f'Prior transport evidence without a committed record: {dispatch}')
        wire = deepcopy(wire)
        wire['model'] = self.model
        write_json(dispatch, {'model': self.model, 'identity': identity, 'wire_request': wire,
                              'started_at': paper_client.now(), 'backend': self.backend})
        self.transport_starts += 1
        started = time.perf_counter()
        try:
            if self.kind == 'ollama':
                raw, sent = self._post('/api/chat', wire), wire
            else:
                raw, sent = self._openai(wire)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, IndexError) as error:
            write_json(received, {'wire_request': wire, 'raw_response': None, 'transport_error': str(error),
                                  'latency_seconds': time.perf_counter() - started})
            raise paper_client.TransportFailure(f'{type(error).__name__}: {error}') from error
        latency = time.perf_counter() - started
        write_json(received, {'wire_request': wire, 'sent_body': sent, 'raw_response': raw,
                              'latency_seconds': latency, 'received_at': paper_client.now()})
        parsed = paper_client.parse_reply(wire, raw)
        return {**parsed, 'raw_response': raw, 'wire_request': wire, 'latency_seconds': latency,
                'transport_attempts': 1}


def run(materials, model, backend, base_url, output, *, api_key=None, tokenizer=None, tasks=None):
    manifest = json.loads(Path(materials).read_text())
    spec = deepcopy(live_spec())
    if model not in spec['models']:
        spec['models'] = spec['models'] + [model]
    if model in spec['inference']['native_think_models']:
        raise SystemExit(f'{model} is a native-thinking model in the paper panel; this runner sends direct-answer requests only.')
    entries = {t['task']: t for t in manifest['tasks']}
    tasks = tasks or [t for t in spec['tasks'] if t in entries]
    counter = TokenCounter(tokenizer)
    root = Path(output) / slug(model)
    write_json(root / 'run_manifest.json', {
        'model': model, 'backend': backend, 'base_url': base_url, 'tokenizer': counter.name,
        'protocol_revision': spec['revision'], 'tasks': tasks, 'missing_tasks': manifest.get('missing_tasks', []),
        'materials_manifest': str(Path(materials).resolve()), 'started_at': paper_client.now(),
        'comparability': 'paper request body' if backend == 'ollama' else 'OpenAI-compatible translation'})
    timeout = spec['inference']['request_timeout_seconds']
    results = []
    for task in tasks:
        bundle = json.loads(Path(entries[task]['material_file']).read_text())
        result = run_task(model, task, bundle, spec, root / task, module=module_for(task), counter=counter,
                          cache=None, client_factory=lambda: ServerClient(model, backend, base_url, api_key, timeout))
        results.append(result)
        print(json.dumps({k: result[k] for k in ('task', 'status', 'planned_steps', 'new_calls', 'resumed_in_output',
                                                 'technical_valid', 'not_run_total', 'stop_reason')}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--materials', type=Path, required=True, help='manifest.json written by build_paper_materials')
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=('ollama', 'openai'), required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--api-key-env', help='Name of the environment variable that holds the API key')
    parser.add_argument('--tokenizer', help='Hugging Face tokenizer for exact context counts')
    parser.add_argument('--task', action='append', help='Repeat to run a subset; default all built tasks')
    parser.add_argument('--output', type=Path, default=Path('results/submissions'))
    args = parser.parse_args()
    api_key = os.environ[args.api_key_env] if args.api_key_env else None
    results = run(args.materials, args.model, args.backend, args.base_url, args.output,
                  api_key=api_key, tokenizer=args.tokenizer, tasks=args.task)
    if any(r['status'] != 'complete' for r in results):
        raise SystemExit(3)


if __name__ == '__main__':
    main()
