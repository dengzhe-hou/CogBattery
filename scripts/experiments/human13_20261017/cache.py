"""Exact-input cache for the r7 entry (user decision b, USER_DECISIONS.md:10).

A prior record is reused only when
1. its stored wire request serializes to the same body bytes as the r7 request
   (json.dumps(wire, ensure_ascii=False)); the one relaxation: the r7 request
   adds options.stop and the source reply content does not contain that stop;
2. its backend binary path and version equal the ones SPEC assigns to the model;
3. the served model digest equals the current local manifest sha256;
4. the raw reply is terminal (done=true with a done_reason).
Only origin records are indexed (execution_origin 'new' or absent); copies that an
earlier run marked 'reused' point to an origin that is indexed itself. Lookups are
outcome-blind: only a record made for the same (episode, step) slot is acceptable
(a different trial with identical bytes gets its own call; user decision 2026-10-18),
then SOURCE_ROOTS order (chronological), then file path (CacheIndex.candidates).
The index lives in memory, keyed by the body string; no hash file is written.
Results are only read here; no result directory is modified.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

from .common import ROOT

RESULTS = ROOT / 'results'
OLLAMA_MANIFESTS = Path.home() / '.ollama/models/manifests/registry.ollama.ai/library'
CVLT = str(ROOT / 'scripts/experiments/cvlt_context_controls_20261007/backend/bin/ollama')
CACHEFIX = str(ROOT / 'scripts/experiments/phi3_cache_fix_20261013/build_support/backend_cachefix_p9/bin/ollama')
PRIVATE_BINARIES = {
    '0.18.0-cogarena-constrained1': str(ROOT / 'scripts/experiments/human13_constrained_20261017/backend_support/backend/bin/ollama'),
    '0.18.0-cogarena-budget1': str(ROOT / 'scripts/experiments/native_budget_20261018/backend_support/backend/bin/ollama'),
    '0.18.0-cogarena-budget-json1': str(ROOT / 'scripts/experiments/json_array_stop_20261018/backend_support/backend/bin/ollama'),
}

# (root, scope label, backend rule). Order = chronological = lookup priority among the
# same-slot sources accepted by CacheIndex.candidates.
# Backend rules come from each run's recorded SPEC (checked 2026-10-18):
# same_stimulus job_r2 SPEC inference.backend_executable = CVLT for every model;
# phi3 p9 SPEC = CACHEFIX; every human13 run SPEC maps default->CVLT, phi3->CACHEFIX,
# native models -> CVLT (r6/repair) or constrained1 (choice/format/uniform), told apart
# by the served version string; native_budget/native_history/json_array_stop hardcode
# their private binary, again told apart by the served version string.
SOURCE_ROOTS = (
    ('results/same_stimulus_20261011/formal/job_r2', 'same_stimulus_formal', 'cvlt_for_all'),
    ('results/phi3_cache_fix_20261013/formal/p9', 'phi3_cache_fix_p9_formal', 'cachefix_for_all'),
    ('results/human13_20261017/technical_pilot/runs', 'human13_technical_pilot', 'human13'),
    ('results/human13_repair_20261017/smoke', 'human13_repair_smoke', 'human13'),
    ('results/human13_constrained_20261017/smoke', 'human13_constrained_smoke', 'human13'),
    ('results/human13_choice_wave_20261017/runs', 'human13_choice_wave_technical', 'human13'),
    ('results/human13_format_repair_20261017/runs', 'human13_format_repair_technical', 'human13'),
    ('results/human13_format_repair_20261017/smoke', 'human13_format_repair_smoke', 'human13'),
    ('results/human13_ospan_array_20261017', 'human13_ospan_array_technical', 'human13'),
    ('results/human13_ospan_finish_20261017', 'human13_ospan_finish_technical', 'human13'),
    ('results/human13_ospan_history_20261017', 'human13_ospan_history_technical', 'human13'),
    ('results/human13_ospan_uniform_20261017', 'human13_ospan_uniform_technical', 'human13'),
    ('results/r1_7b_diagnostic_20261017', 'r1_7b_diagnostic_technical', 'human13'),
    ('results/native_budget_20261018', 'native_budget_technical', 'human13'),
    ('results/native_history_20261018', 'native_history_technical', 'human13'),
    ('results/json_array_stop_20261018', 'json_array_stop_technical', 'human13'),
)
SKIP_DIRS = {'backend_build', 'backend_support', 'received', 'received_attempts', 'dispatch',
             'transport_reservations', 'resources', '__pycache__'}
VERSION_RE = re.compile(r'(\d+\.\d+\.\d+(?:-cogarena-[a-z0-9-]+)?)')
MODEL_DIR_RE = re.compile(r'[a-z0-9.\-]+__[a-z0-9.\-]+')  # e.g. qwen2.5__7b; not ospan_v1__0009


def slug(model):
    return model.replace(':', '__')


def current_digest(model):
    """sha256 of the local manifest file; this is the digest Ollama reports in /api/ps."""
    family, tag = model.split(':', 1)
    return hashlib.sha256((OLLAMA_MANIFESTS / family / tag).read_bytes()).hexdigest()


def body(wire):
    return json.dumps(wire, ensure_ascii=False)


def _walk_calls(root):
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        if Path(current).name == 'calls':
            for name in sorted(files):
                if name.endswith('.json'):
                    yield Path(current) / name


def _version(value):
    if isinstance(value, dict):
        value = value.get('version')
    match = VERSION_RE.search(str(value or ''))
    return match.group(1) if match else None


@lru_cache(maxsize=None)
def serving_evidence(root):
    """(model, job) -> set of (digest, version) from every serving file under root (read once)."""
    evidence = defaultdict(set)
    folder_job = {}
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in files:
            if not (name.startswith('serving') and name.endswith('.json')):
                continue
            path = Path(current) / name
            data = json.loads(path.read_text())
            job = str(data.get('slurm_job_id'))
            ps = data.get('native_ps') or data.get('ps') or {}
            version = _version(data.get('version') or data.get('ollama_version'))
            for row in ps.get('models', []):
                evidence[(row['name'], job)].add((row['digest'], version))
                folder_job.setdefault((str(Path(current)), row['name']), job)
    return evidence, folder_job


def backend_for(rule, model, version):
    if version in PRIVATE_BINARIES and rule == 'human13':
        return PRIVATE_BINARIES[version]
    if version != '0.18.0':
        return None
    if rule == 'cvlt_for_all':
        return CVLT
    if rule == 'cachefix_for_all':
        return CACHEFIX
    if rule == 'human13':
        return CACHEFIX if model.startswith('phi3:') else CVLT
    return None


def _source_identity(path, row, root, rule, evidence, folder_job):
    """Backend path/version and model digest of the run that produced row."""
    if 'r7_backend' in row:  # records written by this entry carry their identity
        return {'backend_path': row['r7_backend']['path'], 'backend_version': row['r7_backend']['version'],
                'model_digest': row.get('r7_model_digest'), 'job': row.get('slurm_job_id'),
                'evidence': 'r7_record_fields'}
    model = row.get('model') or row['wire_request'].get('model')
    job = row.get('slurm_job_id') or row.get('recording_slurm_job_id')
    if job is None:
        # job_r2 and p9 records carry no job id; their model folder's serving file names it.
        for parent in path.parents:
            if (str(parent), model) in folder_job:
                job = folder_job[(str(parent), model)]
                break
            if parent == root:
                break
    found = evidence.get((model, str(job)), set())
    if len(found) != 1:
        return {'backend_path': None, 'backend_version': None, 'model_digest': None, 'job': job,
                'evidence': 'missing' if not found else 'ambiguous'}
    digest, version = next(iter(found))
    return {'backend_path': backend_for(rule, model, version), 'backend_version': version,
            'model_digest': digest, 'job': str(job), 'evidence': 'serving_file'}


def row_slot(row):
    """The (episode, step) a prior record was made for, from its own fields, or None.

    same_stimulus job_r2 and phi3_cache_fix p9 false-belief records carry no episode or step
    id, and their isolated orders sit in one folder (fb_isolated__A1.json); their condition,
    list and item_id name the trial, which r7 calls fb_<condition>_<list>__<item>.
    """
    recorded = (row.get('episode_id'), (row.get('step') or {}).get('id'))
    if all(recorded):
        return recorded
    if row.get('paradigm') == 'strachan_fb' and all(row.get(k) for k in ('condition', 'list', 'item_id')):
        episode = f"fb_{row['condition']}_{row['list']}"
        return episode, f"{episode}__{row['item_id']}"
    return None


class Source:
    __slots__ = ('path', 'row', 'scope', 'priority', 'identity')

    def __init__(self, path, row, scope, priority, identity):
        self.path, self.row, self.scope, self.priority, self.identity = path, row, scope, priority, identity


def scan(models, roots=SOURCE_ROOTS, extra_roots=()):
    """Read every origin record of the given models once. Returns model -> [Source]."""
    wanted = set(models)
    slugs = {slug(m): m for m in wanted}
    out = defaultdict(list)
    stats = Counter()
    all_roots = [(ROOT / r, scope, rule) for r, scope, rule in roots] + list(extra_roots)
    for priority, (root, scope, rule) in enumerate(all_roots):
        root = Path(root)
        if not root.exists():
            continue
        evidence, folder_job = serving_evidence(str(root))
        for path in _walk_calls(root):
            parts = path.relative_to(root).parts
            # Skip folders named for another model without reading their files.
            if any(MODEL_DIR_RE.fullmatch(p) and p not in slugs for p in parts[:-1]):
                continue
            row = json.loads(path.read_text())
            model = row.get('model') or (row.get('wire_request') or {}).get('model')
            if model not in wanted:
                continue
            stats[(scope, 'read')] += 1
            if row.get('execution_origin', 'new') != 'new':
                stats[(scope, 'skipped_reused_copy')] += 1
                continue
            if not isinstance(row.get('wire_request'), dict) or not isinstance(row.get('raw_response'), dict):
                stats[(scope, 'skipped_no_wire_or_raw')] += 1
                continue
            identity = _source_identity(path, row, root, rule, evidence, folder_job)
            out[model].append(Source(path, row, scope, priority, identity))
            stats[(scope, 'origin_indexed')] += 1
    return out, stats


class CacheIndex:
    """Per-model index of origin records keyed by request body string."""

    def __init__(self, model, expected_backend, expected_version, expected_digest, sources):
        self.model = model
        self.expected = {'backend_path': expected_backend, 'backend_version': expected_version,
                         'model_digest': expected_digest}
        self.by_body = defaultdict(list)
        for source in sorted(sources, key=lambda s: (s.priority, str(s.path))):
            self.by_body[body(source.row['wire_request'])].append(source)
        self.rejections = Counter()

    def _identity_problem(self, source):
        ident = source.identity
        if ident['evidence'] not in ('serving_file', 'r7_record_fields') or ident['backend_path'] is None:
            return 'source_identity_unverified'
        if ident['backend_path'] != self.expected['backend_path'] or ident['backend_version'] != self.expected['backend_version']:
            return 'backend_differs'
        if ident['model_digest'] != self.expected['model_digest']:
            return 'model_digest_differs'
        raw = source.row['raw_response']
        if raw.get('done') is not True or not raw.get('done_reason'):
            return 'source_reply_not_terminal'
        return None

    @staticmethod
    def same_slot(source, slot):
        """True when the source record was made for this very (episode, step)."""
        if slot is None:
            return False
        episode_id, step_id = slot
        recorded = row_slot(source.row)
        if recorded is not None:
            return recorded == (episode_id, step_id)
        return source.path.stem in (step_id, f'{episode_id}__{step_id}')

    def candidates(self, wire, slot):
        """Acceptable sources as (source, match_kind): only records made for this very slot.

        User decision 2026-10-18 (USER_DECISIONS.md, 「不允许」): a record made for a different
        trial is never reused, even with identical request bytes; that slot gets its own call.
        Among same-slot sources the earliest root wins, blind to the reply.
        """
        accepted = []
        for item in self._candidates(wire):
            if self.same_slot(item[0], slot):
                accepted.append(item)
            else:
                self.rejections['source_made_for_another_slot'] += 1
        return sorted(accepted, key=lambda item: (item[0].priority, str(item[0].path)))

    def _candidates(self, wire):
        accepted = []
        for source in self.by_body.get(body(wire), []):
            problem = self._identity_problem(source)
            if problem:
                self.rejections[problem] += 1
            else:
                accepted.append((source, 'exact_bytes'))
        stop = wire.get('options', {}).get('stop')
        if stop and not accepted:
            relaxed = deepcopy(wire)
            relaxed['options'].pop('stop')
            for source in self.by_body.get(body(relaxed), []):
                content = (source.row['raw_response'].get('message') or {}).get('content', '')
                if any(s in content for s in stop):
                    self.rejections['stop_string_present_in_source_reply'] += 1
                    continue
                problem = self._identity_problem(source)
                if problem:
                    self.rejections[problem] += 1
                else:
                    accepted.append((source, 'stop_absent_in_reply'))
        return accepted


def committed_copy_is_same_slot(record, slot):
    """For a committed record copied from the cache: was its origin made for this slot?

    Same test as CacheIndex.same_slot, applied to the origin file the record names, so
    records committed under the earlier rule (other slots allowed) can be recognised.
    """
    path = Path(record['reuse']['source_path'])
    row = json.loads(path.read_text())
    return CacheIndex.same_slot(Source(path, row, record['reuse']['source_scope'], 0, {}), slot)
