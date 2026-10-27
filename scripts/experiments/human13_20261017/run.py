"""r7 unified formal entry: one model x one paradigm, cache first, one transport per request.

Order per step (MERGE_PLAN.md 5.3): build the request from the actual history ->
resume a committed record in this output -> stop on uncommitted prior transport ->
exact-input cache (exact bytes, then stop-absent relaxation; same slot only) -> one new request.
Same-slot rule (user decision 2026-10-18): a record made for another trial is never reused,
even with identical bytes. A committed copy of one is moved to superseded/ and the slot is
re-made; a later record of that chain is moved aside too only if its rebuilt request changed.
A reply that is invalid is committed as invalid, never scored, and the chain
continues (user decision a, USER_DECISIONS.md:9). Its record keeps the raw reply
and the retained final text unchanged (arrays verbatim, other text stripped); later
turns show that text only where the paradigm's established presentation shows past
replies (SPEC inference.failure_policy.invalid_reply_presentation: n-back and go/no-go
list INVALID instead, proposal 3).
No reply, a hard serving error, or a context overflow stops this (model, paradigm)
and marks the rest not run (user decision on proposal 2, USER_DECISIONS.md).
Real transport needs SPEC authorization for the chosen scope, an allocated Slurm
GPU and the SPEC backend; a pilot is limited to SPEC authorization.pilot_plan.
--inspect-only sends nothing and writes nothing here.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from .budgets import PendingUserDecision, is_native
from .cache import committed_copy_is_same_slot
from .common import validate_task, write_json
from .protocol import ContextOverflow, build_request, model_inference
from .scoring import score_record, wrap_module

PROTOCOL_EXCLUDED_KEYS = ('authorization', 'inference_status')


class NotAuthorized(RuntimeError):
    pass


def slug(model):
    return model.replace(':', '__')


def protocol_spec(spec):
    """SPEC without authorization fields: what makes two outputs the same protocol.

    The answer-reserve choice (proposal 1) is left out as well: it only affects the
    native requests of ten paradigms, and every committed record is checked against
    its own rebuilt request bytes, so other outputs can resume after that decision.
    The reserve block's decision notes (status, leaning_not_decided) are prose about that
    choice and are left out too; the reserve values themselves stay in the protocol.
    """
    view = deepcopy({k: v for k, v in spec.items() if k not in PROTOCOL_EXCLUDED_KEYS})
    reserve = view.get('inference', {}).get('answer_reserve')
    if reserve is not None:
        reserve['selected_variant'] = 'checked_per_request'
        for note in ('status', 'leaning_not_decided'):
            reserve.pop(note, None)
    return view


def model_spec(spec, model):
    """Kept for preflight.py: per-model context and thinking mode."""
    view = deepcopy(spec)
    view['inference'] = model_inference(spec, model)
    return view


def authorize(spec, scope):
    auth = spec['authorization']
    if scope == 'formal':
        if not (auth.get('inference_authorized') is True and auth.get('formal_authorized') is True):
            raise NotAuthorized('Full evaluation is not authorized in SPEC.json; no request is sent.')
    elif scope == 'pilot':
        if auth.get('pilot_authorized') is not True:
            raise NotAuthorized('The unified-entry pilot is not authorized in SPEC.json yet; no request is sent.')
    else:
        raise ValueError(f'Unknown scope: {scope}')


def pilot_limits(spec, model, tasks, output_root):
    """Pre-written pilot plan (SPEC authorization.pilot_plan): paradigms and new-call cap per paradigm.

    tasks=None means the planned paradigms of this model in SPEC order. The cap counts
    new transports across resumes: committed new records in the pilot output are
    subtracted. Raises NotAuthorized for anything outside the plan.
    """
    from .common import ROOT
    plan = spec['authorization'].get('pilot_plan')
    if not plan:
        raise NotAuthorized('No pre-written pilot plan in SPEC.json; no request is sent.')
    if Path(output_root).resolve() != (ROOT / plan['output_root']).resolve():
        raise NotAuthorized(f"The pilot writes only to {plan['output_root']}")
    caps = plan['max_new_calls'].get(model)
    if caps is None:
        raise NotAuthorized(f'{model} is outside the pilot plan')
    tasks = [t for t in spec['tasks'] if t in caps] if tasks is None else list(tasks)
    outside = [t for t in tasks if t not in caps]
    if outside:
        raise NotAuthorized(f'{model}: {outside} outside the pilot plan')
    remaining = {}
    for task in tasks:
        calls = Path(output_root) / slug(model) / task / 'calls'
        done = sum(json.loads(p.read_text()).get('execution_origin') == 'new'
                   for p in calls.glob('*.json')) if calls.exists() else 0
        remaining[task] = max(caps[task] - done, 0)
    return remaining


def expected_identity(spec, model):
    from .cache import current_digest
    path = spec['backends'].get(model, spec['backends']['default'])
    return path, spec['backend_versions'][path], current_digest(model)


def materialize(source, match_kind, req, step, module):
    """Turn an accepted prior record into an r7 record; validity comes from its raw reply."""
    from .client import parse_reply
    parsed = parse_reply(req.wire, source.row['raw_response'])
    row = source.row
    return {**parsed, 'raw_response': row['raw_response'], 'wire_request': row['wire_request'],
            'latency_seconds': row.get('latency_seconds'), 'execution_origin': 'reused',
            'r7_backend': {'path': source.identity['backend_path'], 'version': source.identity['backend_version']},
            'r7_model_digest': source.identity['model_digest'],
            'reuse': {'match_kind': match_kind, 'source_path': str(source.path), 'source_scope': source.scope,
                      'source_record_scope': row.get('scope'), 'source_job_id': source.identity['job'],
                      'source_backend_path': source.identity['backend_path'],
                      'source_backend_version': source.identity['backend_version'],
                      'source_model_digest': source.identity['model_digest'],
                      'source_research_results_eligible': row.get('research_results_eligible'),
                      'source_technical_pilot': row.get('technical_pilot'),
                      'source_synthetic': bool(row.get('synthetic')),
                      'source_response': row.get('response'),
                      'response_recomputed_equal': row.get('response') == parsed['response'],
                      'source_step_id': (row.get('step') or {}).get('id')}}


SUPERSEDED_DIR = 'superseded'


def history_text(wire, raw):
    """The history text the current rules derive from a saved raw reply."""
    from .client import parse_reply
    return parse_reply(wire, raw)['response']


def recover_received(received_path, req, client):
    """Commit a reply saved in received/ without a record, if the current rules accept it."""
    from .client import HardResponseError, parse_reply
    saved = json.loads(received_path.read_text())
    raw = saved.get('raw_response')
    if raw is None or saved.get('wire_request') != req.wire:
        return None
    try:
        parsed = parse_reply(req.wire, raw)
    except HardResponseError:
        return None
    backend = client.backend if client is not None else None
    digest = client.model_digest if client is not None else None
    dispatch = received_path.parent.parent / 'dispatch' / received_path.name
    sent_by = json.loads(dispatch.read_text()).get('slurm_job_id') if dispatch.exists() else None
    return {**parsed, 'raw_response': raw, 'wire_request': saved['wire_request'],
            'sent_by_slurm_job_id': sent_by,
            'latency_seconds': saved.get('latency_seconds'), 'transport_attempts': 1,
            'execution_origin': 'new', 'recovered_from_received': saved.get('received_at') or True,
            'r7_backend': backend, 'r7_model_digest': digest}


def resolve_stop_files(output, identity):
    """Move the stop markers of a recovered slot aside (kept, renamed), so they no longer flag the cell."""
    stop = output / 'chain_stop.json'
    if stop.exists() and json.loads(stop.read_text()).get('identity') == identity:
        os.replace(stop, output / f'chain_stop_resolved__{identity}.json')
    hard = output / 'hard_errors' / f'{identity}.json'
    if hard.exists():
        (output / 'hard_errors_resolved').mkdir(exist_ok=True)
        os.replace(hard, output / 'hard_errors_resolved' / f'{identity}.json')
        if not any((output / 'hard_errors').iterdir()):
            (output / 'hard_errors').rmdir()


def supersede(output, identity, reason, episode_id, folders=('calls', 'dispatch', 'received')):
    """Move a committed record (and any transport files of that slot) aside; nothing is deleted.

    Used only for records the 2026-10-18 same-slot decision no longer accepts: a copy of another
    trial's record, and a later record of the same chain whose rebuilt request changed because an
    earlier reply in that chain was re-made.
    """
    kept = output / SUPERSEDED_DIR
    kept.mkdir(parents=True, exist_ok=True)
    n = 1
    while (kept / f'{identity}__{n}.note.json').exists():
        n += 1
    moved = {}
    for folder in folders:
        source = output / folder / f'{identity}.json'
        if source.exists():
            target = kept / f'{identity}__{n}.{folder}.json'
            os.replace(source, target)
            moved[folder] = target.name
    write_json(kept / f'{identity}__{n}.note.json', {'identity': identity, 'episode_id': episode_id,
                                                     'reason': reason, 'moved': moved})


def run_task(model, task, bundle, spec, output, *, module, counter, cache=None, client_factory=None,
             inspect_only=False, scope='formal', max_new_calls=None, deadline=None, reserve_variant=None,
             explain=None):
    """Core loop; offline tests drive it with synthetic clients and caches."""
    output = Path(output)
    module = wrap_module(module)
    planned = validate_task(task, bundle)
    if reserve_variant is not None and not inspect_only:
        raise ValueError('A hypothetical answer-reserve variant is allowed only in --inspect-only')
    if not inspect_only:
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (('SPEC_protocol.json', protocol_spec(spec)), ('materials.json', bundle)):
            target = output / name
            if target.exists() and json.loads(target.read_text()) != value:
                raise RuntimeError(f'Existing {name} differs; use a separate output for a new protocol.')
            write_json(target, value)
    counts = Counter()
    not_run = Counter()
    invalid_reasons = Counter()
    first_missing = []
    reused_slots = []
    stop_reason = None
    histories, stopped_episodes = {}, set()
    # Episodes in which a reply was re-made after superseding (read back across resumes): only
    # their later committed records may differ from the rebuilt request, and those are superseded.
    changed_episodes = set()
    if (output / SUPERSEDED_DIR).exists():
        for note in (output / SUPERSEDED_DIR).glob('*.note.json'):
            changed_episodes.add(json.loads(note.read_text())['episode_id'])
    client = None
    timeout = spec['inference']['request_timeout_seconds']
    started = time.perf_counter()

    def finish(status):
        summary = {'model': model, 'task': task, 'scope': scope, 'status': status,
                   'inspect_only': inspect_only, 'planned_steps': planned,
                   'committed_or_reusable': counts['new'] + counts['reused'] + counts['resumed'],
                   'new_calls': counts['new'], 'reused_from_cache': counts['reused'],
                   'resumed_in_output': counts['resumed'],
                   'recovered_from_received': counts['recovered_from_received'],
                   'superseded': {k.removeprefix('superseded_'): v for k, v in counts.items()
                                  if k.startswith('superseded_')},
                   'technical_valid': counts['valid'], 'technical_invalid': counts['invalid'],
                   'reused_valid': counts['reused_valid'], 'reused_invalid_kept_invalid': counts['reused_invalid'],
                   'reused_with_disagreeing_identical_sources': counts['reuse_sources_disagree'],
                   'invalid_reasons': dict(invalid_reasons), 'not_run': dict(not_run),
                   'not_run_total': sum(not_run.values()), 'stop_reason': stop_reason,
                   'first_missing': first_missing,
                   # Pilot criterion (MERGE_PLAN.md 10): every slot inspect lists here must be reused
                   # whenever the run reaches it; the run may reuse more slots than inspect predicted.
                   'reused_slots': reused_slots,
                   'valid_coverage': counts['valid'] / planned if planned else None,
                   'transport_starts': client.transport_starts if client else 0,
                   'wall_seconds': time.perf_counter() - started}
        if summary['committed_or_reusable'] + summary['not_run_total'] != planned:
            raise RuntimeError(f'Coverage accounting error for {model}/{task}: {summary}')
        if not inspect_only:
            write_json(output / 'summary.json', summary)
        return summary

    steps = [(episode, step) for episode in bundle['episodes'] for step in episode['steps']]
    index = 0
    for episode in bundle['episodes']:
        parent = episode.get('parent')
        history = list(histories[parent]) if parent else []
        independent = episode.get('history_mode') == 'independent' and not episode.get('presentation')
        episode_blocked = parent in stopped_episodes
        for step in episode['steps']:
            index += 1
            if stop_reason is not None:
                not_run[stop_reason] += 1
                continue
            if episode_blocked:
                not_run['depends_on_missing_reply'] += 1
                continue
            identity = episode['id'] + '__' + step['id']
            try:
                req = build_request(model, task, episode, step, history, spec, module, counter,
                                    reserve_variant=reserve_variant)
            except PendingUserDecision:
                stop_reason = 'pending_user_decision'
                not_run[stop_reason] += 1
                continue
            except ContextOverflow:
                stop_reason = 'context_overflow'
                not_run[stop_reason] += 1
                continue
            dropped = req.context['history_pairs_dropped']
            if dropped:
                history = history[dropped:]
            path = output / 'calls' / f'{identity}.json'
            record = None
            superseded = None
            resumed = False
            if path.exists():
                record = json.loads(path.read_text())
                differs = (record['desired_wire_request'] != req.wire or record['step'] != step
                           or record.get('presentation') != episode.get('presentation'))
                if (record.get('execution_origin') == 'reused'
                        and not committed_copy_is_same_slot(record, (episode['id'], step['id']))):
                    superseded = 'copied_from_another_slot'
                elif differs and (episode['id'] in changed_episodes or parent in changed_episodes):
                    superseded = 'request_changed_after_earlier_remake'
                elif (not differs and 'skipped_thinking_unconstrained_answer' in record.get('protocol_errors', [])
                      and record['response'] != history_text(req.wire, record['raw_response'])):
                    superseded = 'history_text_rule_changed'  # re-committed from received/, not resent
                elif differs:
                    raise RuntimeError(f'Committed record differs from the rebuilt request: {path}')
                if superseded:
                    counts['superseded_' + superseded] += 1
                    changed_episodes.add(episode['id'])
                    if not inspect_only:
                        supersede(output, identity, superseded, episode['id'],
                                  folders=('calls',) if superseded == 'history_text_rule_changed'
                                  else ('calls', 'dispatch', 'received'))
                    record = None
                else:
                    counts['resumed'] += 1
                    resumed = True
            received_path = output / 'received' / f'{identity}.json'
            if record is None and superseded in (None, 'history_text_rule_changed') and received_path.exists():
                # A reply that arrived but was not committed (it broke an earlier rule) is re-read under
                # the current rules; nothing is resent. If it still breaks the contract, the cell stays stopped.
                if client is None and not inspect_only:
                    client = client_factory()
                record = recover_received(received_path, req, client)
                if record is not None:
                    counts['new'] += 1
                    counts['recovered_from_received'] += 1
                    if not inspect_only:
                        resolve_stop_files(output, identity)
            if record is None and superseded in (None, 'history_text_rule_changed') and (
                    (output / 'dispatch' / f'{identity}.json').exists() or received_path.exists()):
                stop_reason = 'prior_attempt_preserved'
                not_run[stop_reason] += 1
                continue
            if record is None:
                accepted = cache.candidates(req.wire, (episode['id'], step['id'])) if cache is not None else []
                from .client import HardResponseError
                for source, kind in accepted:
                    try:
                        record = materialize(source, kind, req, step, module)
                    except HardResponseError as error:  # breaks the r7 reply contract: not reusable
                        cache.rejections['source_reply_fails_r7_contract'] += 1
                        if explain is not None:
                            explain.setdefault('contract_rejections', []).append([str(source.path), str(error)])
                        continue
                    if len(accepted) > 1:
                        replies = {json.dumps((s.row['raw_response'].get('message') or {}), sort_keys=True,
                                              ensure_ascii=False) for s, _ in accepted}
                        record['reuse']['acceptable_sources'] = len(accepted)
                        record['reuse']['acceptable_sources_disagree'] = len(replies) > 1
                        counts['reuse_sources_disagree'] += len(replies) > 1
                    break
                if record is not None:
                    counts['reused'] += 1
                    reused_slots.append(identity)
                    if explain is not None:
                        key = record['reuse']['source_scope'] + '|' + record['reuse']['match_kind']
                        explain.setdefault('reuse_sources', Counter())[key] += 1
                    counts['reused_valid' if record['technical_valid'] else 'reused_invalid'] += 1
                elif inspect_only:
                    reason = (explain['classify'](req.wire, (episode['id'], step['id']))
                              if explain and 'classify' in explain else None)
                    first_missing.append({'episode_id': episode['id'], 'step_id': step['id'], 'reason': reason})
                    not_run['needs_new_call'] += 1
                    if not independent:
                        episode_blocked = True
                    continue
                else:
                    if max_new_calls is not None and counts['new'] >= max_new_calls:
                        stop_reason = 'checkpoint_new_call_limit'
                        not_run[stop_reason] += 1
                        continue
                    if deadline is not None and time.time() + timeout + 30 > deadline:
                        stop_reason = 'time_checkpoint'
                        not_run[stop_reason] += 1
                        continue
                    from .client import PriorAttemptPreserved, TransportFailure
                    try:
                        if client is None:
                            client = client_factory()
                        if not client.serving_checked:
                            client.serving(output / f"serving_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
                        reply = client.send(req.wire, identity, output)
                    except (TransportFailure, HardResponseError, PriorAttemptPreserved) as error:
                        stop_reason = {TransportFailure: 'no_reply_transport_failure',
                                       HardResponseError: 'hard_response_error',
                                       PriorAttemptPreserved: 'prior_attempt_preserved'}[type(error)]
                        not_run[stop_reason] += 1
                        write_json(output / 'chain_stop.json', {'identity': identity, 'reason': stop_reason,
                                                                 'error': str(error)})
                        continue
                    record = {**reply, 'execution_origin': 'new', 'r7_backend': client.backend,
                              'r7_model_digest': client.model_digest}
                    counts['new'] += 1
            if not resumed:  # new, reused, or recovered from received/: add this run's slot metadata
                record.update({'model': model, 'task': task, 'episode_id': episode['id'],
                               'condition': episode['condition'], 'parent': parent,
                               'presentation': episode.get('presentation'), 'step': deepcopy(step),
                               'desired_wire_request': req.wire, 'context': req.context,
                               'scope': scope, 'formal_record': scope == 'formal',
                               'research_results_eligible': scope == 'formal', 'synthetic': False,
                               'protocol_revision': spec['revision'],
                               'recording_slurm_job_id': os.environ.get('SLURM_JOB_ID')})
                if record.get('recovered_from_received'):
                    # The job that sent and received the reply, as for every other record; the
                    # committing job is kept separately.
                    record['committed_by_slurm_job_id'] = record['recording_slurm_job_id']
                    record['recording_slurm_job_id'] = record.pop('sent_by_slurm_job_id')
            record['score'] = score_record(module, record)
            if not inspect_only and not path.exists():
                write_json(path, record)
            counts['valid' if record['technical_valid'] else 'invalid'] += 1
            for reason in record.get('protocol_errors', []):
                invalid_reasons[reason] += 1
            # Decision (a): the record enters history whether or not it is valid; each paradigm's
            # presentation decides what later turns show (SPEC failure_policy.invalid_reply_presentation).
            history.append({'step': step, 'response': record['response'], 'score': record['score']})
        histories[episode['id']] = history
        if episode_blocked or (stop_reason is not None):
            stopped_episodes.add(episode['id'])
    if index != len(steps):
        raise RuntimeError('Step walk differs from the material manifest')
    status = ('complete' if not not_run else
              'inspected' if inspect_only else
              stop_reason or 'incomplete')
    return finish(status)


def live_spec():
    """The live SPEC.json next to this file (authorization and the reserve choice come from here)."""
    return json.loads((Path(__file__).with_name('SPEC.json')).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', action='append', help='Repeat; default all 13 in SPEC order')
    parser.add_argument('--output', type=Path, required=True, help='Scope root, e.g. results/human54_unified_20261018/formal')
    parser.add_argument('--scope', choices=('formal', 'pilot'), default='formal')
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--max-new-calls', type=int, help='Per task; checkpoint after this many new requests '
                        '(a pilot never exceeds its SPEC pilot_plan cap)')
    parser.add_argument('--deadline-epoch', type=float, help='Do not start a request that could outlive this time')
    parser.add_argument('--server-log', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    live = live_spec()
    if protocol_spec(live) != protocol_spec(manifest['spec']):
        raise RuntimeError('Manifest SPEC differs from the live SPEC.json protocol; re-derive prepared_r7.')
    spec = live  # authorization and the reserve choice are read from the live SPEC
    if args.model not in spec['models']:
        raise ValueError(f'Model outside the 54-model panel: {args.model}')
    if not args.inspect_only:
        authorize(live, args.scope)  # before any client, server or cache work
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise NotAuthorized('Real inference requires an allocated Slurm GPU.')
    # A pilot is bounded by the pre-written plan, also when only inspecting (same task list).
    caps = pilot_limits(spec, args.model, args.task, args.output) if args.scope == 'pilot' else None
    tasks = list(caps) if caps is not None else (args.task or list(spec['tasks']))
    from .cache import CacheIndex, scan, SOURCE_ROOTS
    from .registry import module_for
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter as TokenCounter
    backend, version, digest = expected_identity(spec, args.model)
    extra = []
    pilot_root = args.output.parent / 'pilot'
    if args.scope == 'formal' and pilot_root.exists():
        extra.append((pilot_root, 'unified_pilot', 'r7'))
    sources, _ = scan([args.model], SOURCE_ROOTS, extra)
    cache = CacheIndex(args.model, backend, version, digest, sources[args.model])

    def client_factory():
        from .client import Client
        return Client(args.model, model_spec(spec, args.model), backend_path=backend,
                      backend_version=version, model_digest=digest,
                      server_log=args.server_log or (args.output / slug(args.model) / f"ollama_{os.environ['SLURM_JOB_ID']}.log"))

    results = []
    with TokenCounter(args.model) as counter:
        for task in tasks:
            entry = next(t for t in manifest['tasks'] if t['task'] == task)
            bundle = json.loads(Path(entry['material_file']).read_text())
            out = args.output / slug(args.model) / task
            cap = args.max_new_calls
            if caps is not None:
                cap = caps[task] if cap is None else min(cap, caps[task])
            result = run_task(args.model, task, bundle, spec, out, module=module_for(task), counter=counter,
                              cache=cache, client_factory=client_factory, inspect_only=args.inspect_only,
                              scope=args.scope, max_new_calls=cap, deadline=args.deadline_epoch)
            results.append(result)
            print(json.dumps({k: result[k] for k in ('model', 'task', 'status', 'planned_steps', 'new_calls',
                                                     'reused_from_cache', 'resumed_in_output', 'superseded',
                                                     'technical_valid', 'not_run_total', 'stop_reason')},
                             ensure_ascii=False), flush=True)
    stopped = [r for r in results if r['status'] not in ('complete', 'inspected', 'pending_user_decision',
                                                         'checkpoint_new_call_limit', 'time_checkpoint')]
    if stopped and not args.inspect_only:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
