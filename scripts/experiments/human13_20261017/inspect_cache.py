"""Offline cache audit of the full r7 call list (inspect-only; nothing is sent or written to results).

For every model x paradigm it walks the r7 materials in order with run.run_task in
inspect-only mode: a step is reusable only through cache.CacheIndex (exact bytes +
backend + model digest; stop-absent relaxation). The walk continues with the reused
reply as history; at the first step with no acceptable source the rest of that
episode (and its child branches) depends on a reply that does not exist yet.
Independent single-turn episodes continue. Writes CACHE_AUDIT.json/.md.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from .budgets import is_native
from .cache import CacheIndex, SOURCE_ROOTS, body, current_digest, row_slot, scan
from .common import ROOT, write_json
from .registry import module_for
from .run import expected_identity, protocol_spec, run_task

DEFAULT_MANIFEST = ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json'
DEFAULT_OUT = ROOT / 'refine-logs/human54_unified_entry_20261018'
FORMAL_OUT = ROOT / 'results/human54_unified_20261018/formal'


def _differing_fields(a, b):
    fields = []
    for key in sorted(set(a) | set(b)):
        if key == 'messages':
            continue
        if key == 'options':
            for sub in sorted(set(a.get('options', {})) | set(b.get('options', {}))):
                if a.get('options', {}).get(sub) != b.get('options', {}).get(sub):
                    fields.append('options.' + sub)
        elif a.get(key) != b.get(key):
            fields.append(key)
    return fields


class MissExplainer:
    """Why no prior record matched this slot (diagnostic only; never changes reuse).

    Compares the r7 request with prior records made for the same (episode, step)
    and names the fields that differ; messages are split into the current turn and
    the earlier history.
    """

    def __init__(self, sources, cache):
        self.cache = cache
        self.by_slot = defaultdict(list)
        for source in sources:
            recorded = row_slot(source.row)
            if recorded is not None:
                self.by_slot[recorded].append(source)
            self.by_slot[('stem', source.path.stem)].append(source)

    @staticmethod
    def signature(old, new):
        fields = _differing_fields(old, new)
        a, b = old.get('messages') or [], new['messages']
        if a[-1:] != b[-1:]:
            fields.append('messages.current_turn')
        if a[:-1] != b[:-1]:
            fields.append('messages.history')
        return ','.join(fields) or 'identical_bytes'

    def __call__(self, wire, slot):
        accepted = self.cache.by_body.get(body(wire))
        if accepted:
            # Same bytes: either every source fails the identity checks, or one passes and was
            # made for another slot (same-slot rule, user decision 2026-10-18).
            if any(self.cache._identity_problem(s) is None for s in accepted):
                return 'byte_match_made_for_another_slot'
            return 'byte_match_rejected_by_identity'
        episode_id, step_id = slot
        same = {id(s): s for key in (slot, ('stem', step_id), ('stem', f'{episode_id}__{step_id}'))
                for s in self.by_slot.get(key, [])}
        if not same:
            return 'no_prior_record_for_this_slot'
        diffs = Counter(self.signature(s.row['wire_request'], wire) for s in same.values())
        return 'prior_slot_record_differs_in:' + diffs.most_common(1)[0][0]


def audit(manifest_path=DEFAULT_MANIFEST, models=None, tasks=None, variants=('canonical_w0', 'grammar_ws_w22')):
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter as TokenCounter
    manifest = json.loads(Path(manifest_path).read_text())
    spec = manifest['spec']
    live = json.loads((Path(__file__).with_name('SPEC.json')).read_text())
    if protocol_spec(live) != protocol_spec(spec):
        raise RuntimeError('prepared_r7 manifest SPEC differs from live SPEC.json')
    models = models or spec['models']
    tasks = tasks or spec['tasks']
    bundles = {t['task']: json.loads(Path(t['material_file']).read_text()) for t in manifest['tasks'] if t['task'] in tasks}
    started = time.perf_counter()
    rows, identities, scan_stats = [], {}, Counter()
    for model in models:
        sources, stats = scan([model])
        scan_stats.update(stats)
        backend, version, digest = expected_identity(spec, model)
        table = Counter()
        for source in sources[model]:
            ident = source.identity
            table['|'.join((source.scope, ident['evidence'], str(ident['backend_version']),
                            'backend_ok' if ident['backend_path'] == backend and ident['backend_version'] == version
                            else 'backend_other', 'digest_ok' if ident['model_digest'] == digest else 'digest_other'))] += 1
        identities[model] = {'backend_path': backend, 'backend_version': version, 'model_digest': digest,
                             'native_think': is_native(model, spec['inference']),
                             'origin_records_indexed': len(sources[model]),
                             'source_identity_table': dict(sorted(table.items()))}
        with TokenCounter(model) as counter:
            for task in tasks:
                native_reserve = is_native(model, spec['inference']) and task in spec['inference']['native_reserve_tasks']
                runs = variants if native_reserve else (None,)
                for variant in runs:
                    cache = CacheIndex(model, backend, version, digest, sources[model])
                    explain = {'classify': MissExplainer(sources[model], cache)}
                    result = run_task(model, task, bundles[task], spec, FORMAL_OUT / model.replace(':', '__') / task,
                                      module=module_for(task), counter=counter, cache=cache, inspect_only=True,
                                      reserve_variant=variant, explain=explain)
                    reasons = Counter(m['reason'] for m in result['first_missing'])
                    rows.append({'model': model, 'task': task, 'native_think': identities[model]['native_think'],
                                 'reserve_variant_hypothetical': variant,
                                 'planned': result['planned_steps'],
                                 'resumed_in_formal_output': result['resumed_in_output'],
                                 'reusable_valid': result['reused_valid'],
                                 'reusable_invalid_kept_invalid': result['reused_invalid_kept_invalid'],
                                 'reused_with_disagreeing_identical_sources': result['reused_with_disagreeing_identical_sources'],
                                 'remaining_new_calls_upper_bound': result['planned_steps'] - result['committed_or_reusable'],
                                 'first_missing_steps': len(result['first_missing']),
                                 'depends_on_missing_reply': result['not_run'].get('depends_on_missing_reply', 0),
                                 'blocked_reason': result['stop_reason'],
                                 'byte_match_rejections': dict(cache.rejections),
                                 'first_miss_reasons': dict(reasons),
                                 'first_missing_examples': result['first_missing'][:3],
                                 'reuse_sources': dict(explain.get('reuse_sources', {})),
                                 # Pilot criterion (MERGE_PLAN.md 10): each slot here must be reused when reached.
                                 'predicted_reuse_slots': result['reused_slots'],
                                 'contract_rejections': explain.get('contract_rejections', [])[:3]})
        print(json.dumps({'model': model, 'elapsed_s': round(time.perf_counter() - started, 1),
                          'reusable': sum(r['reusable_valid'] + r['reusable_invalid_kept_invalid']
                                          for r in rows if r['model'] == model and r['reserve_variant_hypothetical'] in (None, variants[-1]))}), flush=True)
    main_rows = [r for r in rows if r['reserve_variant_hypothetical'] in (None, variants[-1])]
    alt_rows = [r for r in rows if r['reserve_variant_hypothetical'] == variants[0]]

    def totals(group):
        out = Counter()
        for r in group:
            for key in ('planned', 'reusable_valid', 'reusable_invalid_kept_invalid', 'remaining_new_calls_upper_bound',
                        'reused_with_disagreeing_identical_sources'):
                out[key] += r[key]
        return dict(out)
    summary = {'all': totals(main_rows),
               'by_task': {t: totals([r for r in main_rows if r['task'] == t]) for t in tasks},
               'by_model': {m: totals([r for r in main_rows if r['model'] == m]) for m in models},
               'native_reserve_tasks_other_variant_reusable': sum(
                   r['reusable_valid'] + r['reusable_invalid_kept_invalid'] for r in alt_rows),
               'rows_note': 'Native rows of the ten capped paradigms appear once per hypothetical reserve variant; '
                            'summary uses ' + variants[-1] + ' (both give the same counts here).'}
    return {'scope': 'offline_inspect_only_cache_audit', 'model_calls': 0, 'slurm_jobs': 0, 'summary': summary,
            'manifest': str(manifest_path), 'spec_revision': spec['revision'],
            'source_roots': [{'root': r, 'scope': s, 'backend_rule': b} for r, s, b in SOURCE_ROOTS],
            'scan_stats': {f'{k[0]}|{k[1]}': v for k, v in sorted(scan_stats.items())},
            'expected_identities': identities, 'rows': rows,
            'elapsed_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--model', action='append')
    parser.add_argument('--task', action='append')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT / 'CACHE_AUDIT.json')
    args = parser.parse_args()
    result = audit(args.manifest, args.model, args.task)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
