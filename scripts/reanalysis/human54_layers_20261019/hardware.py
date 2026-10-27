"""Metadata-only hardware and technical-coverage accounting for frozen human54.

Reads committed calls, their serving metadata, and original-source inference logs.
Never evaluates behavioral scores, calls a model, or edits formal records.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / 'results/human54_unified_20261018/formal'
OUT = ROOT / 'results/reanalysis/human54_layers_20261019/hardware'
NATIVE = {'deepseek-r1:7b', 'deepseek-r1:14b', 'qwen3:4b'}
CARDS = ('A100 MIG 3g.40gb', 'A100 MIG 4g.40gb', 'RTX PRO 6000')


def read(path):
    return json.loads(path.read_text())


def card_of(description):
    for label in CARDS:
        if label.removeprefix('A100 ') in description:
            return label
    raise ValueError(f'Unrecognized GPU in runtime log: {description}')


@lru_cache(None)
def log_evidence(path):
    lines = []
    # Older captured server logs contain non-UTF-8 terminal bytes; the ASCII
    # inference lines themselves must remain intact.
    with path.open(errors='replace') as stream:
        for number, line in enumerate(stream, 1):
            if 'inference compute' in line:
                assert '\ufffd' not in line, (path, number)
                match = re.search(r'description="([^"]+)"', line)
                if match:
                    lines.append({'line': number, 'description': match[1], 'text': line.rstrip()})
    cards = {card_of(row['description']) for row in lines}
    if len(cards) != 1:
        raise ValueError(f'Need one runtime card type at {path}: {cards}')
    return {'gpu_type': cards.pop(), 'log_path': str(path.relative_to(ROOT)), 'inference_compute': lines}


@lru_cache(None)
def source_evidence(folder, job):
    """Trace the origin directory, not the later formal cell's serving record."""
    folder = Path(folder)
    for parent in (folder, *folder.parents):
        log = parent / f'ollama_{job}.log'
        if log.exists():
            result = dict(log_evidence(log))
            candidates = list(folder.glob('serving*.json')) + list(parent.glob('serving*.json'))
            candidates += list(parent.glob('serving_check/serving*.json'))
            result['serving_paths'] = sorted({str(p.relative_to(ROOT)) for p in candidates
                if str(read(p).get('slurm_job_id')) == job})
            return result
        if parent == ROOT / 'results':
            break
    raise FileNotFoundError(f'No runtime GPU evidence for {folder}, job {job}')


def backend_name(path):
    if 'json_array_stop' in path:
        return 'json1'
    if 'backend_cachefix_p9' in path:
        return 'cachefix_p9'
    if 'cvlt_context_controls' in path:
        return 'cvlt'
    raise ValueError(f'Unrecognized backend {path}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells, groups = [], []
    totals, reuse_scopes, missing_source_jobs = Counter(), Counter(), Counter()
    by_card = {origin: {card: {'models': set(), 'cells': 0, 'records': 0} for card in CARDS}
               for origin in ('new', 'reused', 'all')}
    model_new_cards = defaultdict(set)
    summaries = sorted(FORMAL.glob('*/*/summary.json'))
    assert len(summaries) == 702, len(summaries)
    for index, path in enumerate(summaries, 1):
        s = read(path)
        folder = path.parent
        model, task = s['model'], s['task']
        counts = Counter()
        for call in sorted((folder / 'calls').glob('*.json')):
            data = read(call)
            # Only these metadata fields are used. Response, thought, and score fields
            # are not inspected or retained.
            origin = data['execution_origin']
            valid = bool(data['technical_valid'])
            backend = backend_name(data['r7_backend']['path'])
            if origin == 'new':
                job = str(data['recording_slurm_job_id'])
                source_folder, scope = str(folder), 'formal_new'
            elif origin == 'reused':
                reuse = data['reuse']
                source = Path(reuse['source_path'])
                job = reuse['source_job_id']
                if job is None:
                    original = read(source)
                    job = original.get('recording_slurm_job_id') or original.get('slurm_job_id')
                    if job is None:
                        raise ValueError(f'Missing source job: {source}')
                    missing_source_jobs[str(job)] += 1
                job = str(job)
                source_folder, scope = str(source.parent.parent), reuse['source_scope']
                reuse_scopes[scope] += 1
            else:
                raise ValueError(f'Unknown execution origin {origin} at {call}')
            counts[(origin, job, source_folder, scope, backend, valid)] += 1
        n = sum(counts.values())
        valid_n = sum(count for key, count in counts.items() if key[-1])
        assert s['status'] == 'complete' and n == s['planned_steps'], (model, task, n, s)
        assert valid_n == s['technical_valid'], (model, task, valid_n, s['technical_valid'])
        card_counts = {origin: Counter() for origin in ('new', 'reused', 'all')}
        for (origin, job, source_folder, scope, backend, valid), count in sorted(counts.items()):
            evidence = source_evidence(source_folder, job)
            card = evidence['gpu_type']
            serving_path = folder / f'serving_{job}.json' if origin == 'new' else None
            if serving_path is not None:
                serving = read(serving_path)
                assert serving['slurm_job_id'] == job
                expected_host = 'c03' if 'A100' in card else 'c04'
                assert serving['hostname'] == expected_host, (serving_path, card)
            groups.append({'model': model, 'task': task, 'origin': origin, 'source_scope': scope,
                'job_id': job, 'mode': 'native' if model in NATIVE else 'direct', 'backend': backend,
                'technical_valid': valid, 'records': count, **evidence,
                'formal_serving_path': str(serving_path.relative_to(ROOT)) if serving_path else None})
            card_counts[origin][card] += count
            card_counts['all'][card] += count
            if origin == 'new':
                model_new_cards[model].add(card)
        cells.append({'model': model, 'task': task, 'mode': 'native' if model in NATIVE else 'direct',
            'planned': s['planned_steps'], 'recorded': n, 'technical_valid': valid_n,
            'technical_invalid': n-valid_n, 'not_run': s['not_run_total'],
            'valid_coverage': valid_n/s['planned_steps'],
            'new_by_gpu': dict(card_counts['new']), 'reused_by_source_gpu': dict(card_counts['reused']),
            'new_gpu_type_count': len(card_counts['new']), 'all_source_gpu_type_count': len(card_counts['all']),
            'backends': sorted({key[-2] for key in counts})})
        for origin in card_counts:
            for card, count in card_counts[origin].items():
                by_card[origin][card]['models'].add(model)
                by_card[origin][card]['cells'] += 1
                by_card[origin][card]['records'] += count
        totals.update(planned=s['planned_steps'], recorded=n, technical_valid=valid_n,
            technical_invalid=n-valid_n, not_run=s['not_run_total'],
            new=sum(card_counts['new'].values()), reused=sum(card_counts['reused'].values()))
        if index % 100 == 0:
            print(f'Hardware metadata: {index}/702 cells', flush=True)
    for origins in by_card.values():
        for result in origins.values():
            result['models'] = sorted(result['models'])
            result['model_count'] = len(result['models'])
    patterns = defaultdict(list)
    for model, cards in sorted(model_new_cards.items()):
        patterns[' + '.join(sorted(cards))].append(model)
    summary = {'formal_root': str(FORMAL.relative_to(ROOT)), 'models': len(model_new_cards),
        'cells': len(cells), 'totals': dict(totals), 'valid_coverage': totals['technical_valid']/totals['planned'],
        'by_gpu': by_card, 'new_gpu_patterns': dict(patterns), 'reuse_scopes': dict(reuse_scopes),
        'reused_source_job_null_resolved_from_origin': dict(missing_source_jobs),
        'new_calls_mixed_gpu_cells': [c for c in cells if c['new_gpu_type_count'] > 1],
        'including_reuse_mixed_gpu_cells': [c for c in cells if c['all_source_gpu_type_count'] > 1]}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUT/'cell_hardware.json').write_text(json.dumps(cells, indent=2)+'\n')
    (OUT/'provenance_groups.json').write_text(json.dumps(groups, indent=2)+'\n')
    with (OUT/'cell_hardware.csv').open('w', newline='') as f:
        fields = ['model','task','mode','planned','recorded','technical_valid','technical_invalid','not_run','valid_coverage','new_gpu_type_count','all_source_gpu_type_count','backends']
        fields += [f'{origin}_{card}' for origin in ('new', 'reused') for card in CARDS]
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for cell in cells:
            row = {k:cell[k] for k in fields if k in cell};row['backends']=';'.join(row['backends'])
            for origin, key in [('new','new_by_gpu'),('reused','reused_by_source_gpu')]:
                row.update({f'{origin}_{card}':cell[key].get(card,0) for card in CARDS})
            writer.writerow(row)
    print(json.dumps({'totals':dict(totals),'new_gpu_patterns':dict(patterns),
        'mixed_new_cells':len(summary['new_calls_mixed_gpu_cells']),
        'mixed_all_cells':len(summary['including_reuse_mixed_gpu_cells']),
        'source_job_null_resolved':dict(missing_source_jobs)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
