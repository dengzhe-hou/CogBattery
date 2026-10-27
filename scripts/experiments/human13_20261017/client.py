"""Unified r7 transport: one attempt, dispatch marker first, raw reply saved first.

Parsing is copied from the verified clients, not redesigned:
- requests without thinking_budget: human13_format_repair_20261017/client.py:143-216
  (single attempt instead of its retry loop; SPEC max_transport_attempts = 1);
- requests with thinking_budget: native_history_20261018/protocol.py:117-159
  (validate_response), with the format_repair answer rule for the history text.
Serving checks follow format_repair client.py:56-83 and json_array_stop run.py:49-73,
plus the binary path and model digest that the r7 cache policy needs.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

from .constraints import decode_final


class TransportFailure(RuntimeError):
    """No HTTP reply (connection error, HTTP error status or timeout). Never resent."""


class HardResponseError(RuntimeError):
    """A reply arrived but breaks the serving contract; the chain stops, nothing is resent."""


class PriorAttemptPreserved(RuntimeError):
    """A dispatch or received file exists without a committed record; never resent."""


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _require(condition, message):
    if not condition:
        raise HardResponseError(message)


def validate_budget_response(wire, raw):
    """native_history_20261018/protocol.py:117-159, unchanged checks."""
    p = raw['phase_accounting']
    b, tmax = wire['options']['num_predict'], wire['thinking_budget']
    t, c, f = p['thinking_eval_count'], p['injected_control_tokens'], p['final_eval_count']
    _require(p['total_budget'] == b and p['thinking_budget'] == tmax, 'wrong applied budgets')
    _require(0 <= t <= tmax and f >= 0 and c in (0, 1), 'invalid phase counts')
    _require(raw['eval_count'] == t + f, 'sampled count must exclude injected control')
    _require(p['budget_used'] == t + c + f <= b, 'combined budget exceeded or misreported')
    origin = p['boundary_origin']
    if origin == 'forced':
        _require(t == tmax and c == 1 and p['thinking_done_reason'] == 'length', 'false forced boundary')
        _require(not p.get('thinking_matched_stop'), 'forced boundary cannot have natural stop')
        _require(p['injected_control_text'] == '</think>', 'unexpected control text')
    elif origin == 'natural':
        _require(c == 0 and p['thinking_done_reason'] == 'stop'
                 and p['thinking_matched_stop'] == '</think>', 'false natural boundary')
        _require(not p.get('injected_control_text'), 'natural boundary charged twice')
    else:
        _require(origin == 'none' and c == 0 and f == 0, 'unknown boundary or hidden final')
    if origin in ('natural', 'forced'):
        _require(p['final_budget'] == b - t - c and f <= p['final_budget'], 'wrong final allowance')
        _require(p['final_prompt'] == p['initial_prompt'] + p.get('raw_thinking', '') + '</think>',
                 'final is not an exact continuation of the same assistant')
    else:
        _require(p['final_budget'] == 0, 'unused final stage has a nonzero budget')
    context = wire['options']['num_ctx']
    _require(raw['prompt_eval_count'] == p['thinking_prompt_eval_count'], 'wrong original input count')
    _require(0 < p['thinking_prompt_eval_count'] <= context
             and 0 <= p['final_prompt_eval_count'] <= context, 'context bounds violated')
    unconstrained = p.get('unconstrained_content') or ''
    # User decision 2026-10-18 (「继续」): a model that ends its turn without opening the format phase
    # (no </think>, text outside the constraint) did reply; the reply is kept as technically invalid
    # and the chain continues. Text outside the format phase after a real boundary still breaks the contract.
    _require(not unconstrained or origin == 'none', 'unconstrained final content before format phase')
    decoded, errors = decode_final(raw.get('message', {}).get('content', ''), wire['format'])
    complete = (raw.get('done') is True and raw.get('done_reason') == 'stop'
                and origin in ('natural', 'forced') and f > 0 and not errors)
    return {'accounting_valid': True, 'technical_valid': complete, 'boundary_origin': origin,
            'natural_thinking_completion': origin == 'natural', 'decoded_final': decoded,
            'format_errors': errors, 'thinking_tokens': t, 'injected_control_tokens': c,
            'final_tokens': f, 'sampled_tokens': t + f, 'budget_used': t + c + f,
            'total_budget': b, 'done_reason': raw.get('done_reason'), 'unconstrained_content': unconstrained}


def parse_reply(wire, raw):
    """Turn one raw /api/chat reply into the committed fields. Raises HardResponseError."""
    try:
        message = raw['message']
        native_think = wire['think']
        reasoning = {k: message[k] for k in ('thinking', 'reasoning', 'reasoning_content') if message.get(k)}
        _require(not (reasoning and not native_think), 'Unexpected reasoning channel with native_think=false')
        _require(all(isinstance(v, str) for v in reasoning.values()), 'Response reasoning channel is not text')
        content = message.get('content', '')
        _require(isinstance(content, str), 'Response content is not text')
        finish = raw.get('done_reason')
        _require(finish in ('stop', 'length'), f'Unsupported or missing finish reason: {finish!r}')
        _require(raw.get('done') is True, 'Missing terminal response metadata: done')
        prompt_tokens, completion_tokens = raw.get('prompt_eval_count'), raw.get('eval_count')
        _require(isinstance(prompt_tokens, int) and prompt_tokens > 0, 'Missing prompt_eval_count')
        _require(isinstance(completion_tokens, int) and completion_tokens >= 0, 'Missing eval_count')
        _require(prompt_tokens <= wire['options']['num_ctx'], 'Reported input tokens exceed served context')
        verdict = None
        if 'thinking_budget' in wire:
            verdict = validate_budget_response(wire, raw)
        else:
            _require(not (native_think and 'format' in wire),
                     'r7 never sends a native formatted request without thinking_budget')
    except (KeyError, TypeError, AttributeError) as error:
        raise HardResponseError(f'Malformed reply: {type(error).__name__}: {error}') from error
    # History text: format_repair client.py:204-211 (decoded value when it decodes,
    # otherwise the retained final text; empty when there is no final).
    errors = []
    if finish != 'stop':
        errors.append('length_finish')
    if not content.strip():
        errors.append('empty_final')
    answer = content.strip()
    if 'format' in wire and content.strip():
        answer, format_errors = decode_final(content, wire['format'])
        errors.extend(format_errors)
    if verdict is not None:
        if verdict['boundary_origin'] == 'none' or verdict['final_tokens'] <= 0:
            errors.append('no_final_phase')
        if verdict['unconstrained_content']:
            # Later turns show the answer the model wrote first: the text before its own <think> or
            # </think> marker (user decisions 2026-10-18, 「继续」 then 「截断」); the raw reply is kept whole.
            errors.append('skipped_thinking_unconstrained_answer')
            if not content.strip():
                answer = re.split(r'</?think>', verdict['unconstrained_content'], maxsplit=1)[0].strip()
        if verdict['technical_valid'] != (not errors):
            raise HardResponseError('Budget verdict and protocol errors disagree')
    return {'response': answer, 'raw_content': content, 'finish_reason': finish,
            'thinking': message.get('thinking', ''), 'reasoning_channels': reasoning,
            'protocol_valid': not errors, 'protocol_errors': errors, 'technical_valid': not errors,
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                      'total_tokens': prompt_tokens + completion_tokens},
            'budget_verdict': verdict,
            'prompt_plus_completion_exceeds_context': prompt_tokens + completion_tokens > wire['options']['num_ctx']}


class Client:
    """One model's server. All real transport goes through send()."""

    def __init__(self, model, spec_view, *, backend_path, backend_version, model_digest, server_log):
        inf = spec_view['inference']
        if inf['max_transport_attempts'] != 1:
            raise ValueError('r7 sends each logical request exactly once (max_transport_attempts must be 1)')
        self.model, self.inf = model, inf
        self.backend = {'path': backend_path, 'version': backend_version}
        self.model_digest = model_digest
        self.server_log = Path(server_log)
        self.url = 'http://' + os.environ['OLLAMA_HOST']
        self.serving_checked = False
        self.transport_starts = 0

    def _get(self, route):
        with urllib.request.urlopen(self.url + route, timeout=20) as response:
            return json.load(response)

    def serving(self, record_path):
        selected = shutil.which('ollama')
        table = subprocess.check_output(['ollama', 'ps'], text=True)
        native = self._get('/api/ps')
        version = self._get('/api/version')
        row = next((m for m in native.get('models', []) if m['name'] == self.model), None)
        lines = [line.decode('ascii', 'replace') for line in self.server_log.read_bytes().splitlines()
                 if b'load request' in line and b'Operation:commit' in line] if self.server_log.exists() else []
        threads = re.search(r'NumThreads:\s*(\d+)', lines[-1]) if lines else None
        evidence = {'captured_at': now(), 'hostname': socket.gethostname(),
                    'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'which_ollama': selected,
                    'ollama_ps': table, 'native_ps': native, 'version': version,
                    'gpu': subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                           '--format=csv,noheader'], text=True, capture_output=True).stdout,
                    'server_log': str(self.server_log), 'committed_load_request_line': lines[-1] if lines else None,
                    'actual_server_num_threads': int(threads.group(1)) if threads else None,
                    'expected': {'backend': self.backend, 'model_digest': self.model_digest,
                                 'context_tokens': self.inf['context_tokens'],
                                 'server_threads': self.inf['server_threads']}}
        write_json(record_path, evidence)
        problems = []
        if not selected or str(Path(selected).resolve()) != str(Path(self.backend['path']).resolve()):
            problems.append('serving binary path differs')
        if version.get('version') != self.backend['version']:
            problems.append(f"backend version {version.get('version')!r} != {self.backend['version']!r}")
        if row is None or not (row['size_vram'] == row['size'] > 0):
            problems.append('model not fully on GPU')
        elif row['context_length'] != self.inf['context_tokens']:
            problems.append(f"served context {row['context_length']} != {self.inf['context_tokens']}")
        elif row['digest'] != self.model_digest:
            problems.append('served model digest differs from the local manifest')
        if evidence['actual_server_num_threads'] != self.inf['server_threads']:
            problems.append(f"NumThreads={evidence['actual_server_num_threads']}")
        if problems:
            raise HardResponseError('Serving check failed: ' + '; '.join(problems) + f' (see {record_path})')
        self.serving_checked = True

    def send(self, wire, identity, folder):
        """Exactly one HTTP request. Raw evidence is written before any validation."""
        folder = Path(folder)
        dispatch = folder / 'dispatch' / f'{identity}.json'
        received = folder / 'received' / f'{identity}.json'
        if dispatch.exists() or received.exists():
            raise PriorAttemptPreserved(f'Prior transport evidence without a committed record: {dispatch}')
        dispatch.parent.mkdir(parents=True, exist_ok=True)
        with dispatch.open('x') as handle:  # never overwritten, never retried
            json.dump({'model': self.model, 'identity': identity, 'wire_request': wire, 'started_at': now(),
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'backend': self.backend},
                      handle, ensure_ascii=False, indent=2)
        self.transport_starts += 1
        data = json.dumps(wire, ensure_ascii=False).encode()
        request = urllib.request.Request(self.url + '/api/chat', data=data,
                                         headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.inf['request_timeout_seconds']) as response:
                raw = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            write_json(received, {'wire_request': wire, 'raw_response': None, 'transport_error': str(error),
                                  'error_type': type(error).__name__,
                                  'latency_seconds': time.perf_counter() - started})
            raise TransportFailure(f'{type(error).__name__}: {error}') from error
        latency = time.perf_counter() - started
        write_json(received, {'wire_request': wire, 'raw_response': raw, 'latency_seconds': latency,
                              'received_at': now()})
        try:
            parsed = parse_reply(wire, raw)
        except HardResponseError as error:
            write_json(folder / 'hard_errors' / f'{identity}.json', {'error': str(error), 'received': str(received)})
            raise
        return {**parsed, 'raw_response': raw, 'wire_request': wire, 'latency_seconds': latency,
                'transport_attempts': 1}
