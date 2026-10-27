"""Read-only job/result monitoring for the authorized 54-model technical pilot."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from .common import ROOT, write_json

LOG = ROOT / 'refine-logs/human54_pilot_20261017'
RUNS = ROOT / 'results/human13_20261017/technical_pilot/runs'
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}


def cpu_seconds(value):
    fields = value.split(':')
    return sum(float(part) * 60 ** i for i, part in enumerate(reversed(fields)))


def collect():
    schedule = json.loads((LOG / 'active_jobs.json').read_text())
    job_ids = schedule['jobs']
    raw = subprocess.check_output(['sacct', '-j', ','.join(job_ids), '--units=K', '-P',
        '-o', 'JobID,State,ElapsedRaw,TotalCPU,MaxRSS,ReqMem,AllocCPUS'], text=True)
    lines = [line.split('|') for line in raw.strip().splitlines()]
    records = [dict(zip(lines[0], line)) for line in lines[1:]]
    jobs = []
    for job in job_ids:
        main = next((r for r in records if r['JobID'] == job), {})
        batch = next((r for r in records if r['JobID'] == job + '.batch'), {})
        item = {'job_id': job, 'main': main, 'batch': batch}
        if batch.get('MaxRSS') and main.get('ElapsedRaw') and main.get('ReqMem'):
            rss = float(batch['MaxRSS'].rstrip('K'))
            requested = float(main['ReqMem'].rstrip('Knc'))
            elapsed = int(main['ElapsedRaw'])
            cpus = int(main['AllocCPUS'])
            item.update(peak_gib=rss / 1048576,
                cpu_efficiency_percent=100 * cpu_seconds(batch['TotalCPU']) / (elapsed * cpus) if elapsed else None,
                memory_efficiency_percent=100 * rss / requested)
        jobs.append(item)
        if main.get('State', '').split()[0] in TERMINAL:
            write_json(LOG / ('accounting_' + job + '.json'), item)
    spec = json.loads((ROOT / 'scripts/experiments/human13_20261017/SPEC.json').read_text())
    models = []
    for model in spec['models']:
        folder = RUNS / model.replace(':', '__')
        path = folder / 'pilot_summary.json'
        partial = folder / 'pilot_progress.json'
        if path.exists() or partial.exists():
            data = json.loads((path if path.exists() else partial).read_text())
            tasks = data['tasks'] + data['engineering_probes']
            audit = [t.get('audit', {}) for t in tasks]
            models.append({'model': model, 'status': data.get('status', 'running'),
                'recorded_calls': sum(a.get('committed_calls', 0) for a in audit),
                'length_responses': sum(a.get('length_finish_responses', 0) for a in audit),
                'parse_invalid': sum(a.get('parse_invalid_calls', 0) for a in audit),
                'reasoning_channel_responses': sum(a.get('reasoning_channel_responses', 0) for a in audit),
                'failed_tasks': [{'task': t['task'], 'error': t.get('error') or t.get('reason')}
                    for t in tasks if t.get('status') not in ('complete',)],
                'note': 'Recorded calls do not imply valid format or thinking mode; no behavioral effects computed.'})
        else:
            models.append({'model': model, 'status': 'not_started', 'recorded_calls': 0})
    finished = schedule['scheduling_complete'] and all(j['main'].get('State', '').split(' ')[0] in TERMINAL for j in jobs)
    result = {'updated_at': datetime.now().astimezone().isoformat(), 'scope': 'technical_pilot_only',
        'formal_authorized': False, 'all_jobs_terminal': finished,
        'scheduling_complete': schedule['scheduling_complete'],
        'superseded_pending_jobs': schedule['superseded_pending_jobs'],
        'model_count': len(models), 'models_with_all_calls_recorded': sum(m['status'] == 'complete' for m in models),
        'total_recorded_logical_calls': sum(m['recorded_calls'] for m in models),
        'maximum_planned_logical_calls': 10206, 'jobs': jobs, 'models': models,
        'qwen3_4b_verdict': '189/189 responses length-truncated with reasoning in content despite think=false; protocol failed.'}
    write_json(LOG / 'STATUS.json', result)
    (LOG / 'sacct_latest.txt').write_text(raw)
    report = ['# 54款技术短测状态', '', result['updated_at'], '',
        f"已记录全部189次调用的模型：{result['models_with_all_calls_recorded']}/54。",
        f"已记录逻辑调用：{result['total_recorded_logical_calls']}/10206。", '',
        '调用记录完整不等于协议通过；Qwen3 4B已确认思考模式不兼容。此处不计算行为效应，完整评测仍未授权。', '',
        '| 作业 | 状态 | 已用秒 | 峰值GiB |', '|---|---|---:|---:|']
    for job in jobs:
        report.append(f"| {job['job_id']} | {job['main'].get('State', 'unknown')} | {job['main'].get('ElapsedRaw', '')} | {job.get('peak_gib', '')} |")
    report.extend(['', '| 模型 | 状态 | 已记录调用 | length次数 | 格式解析无效 |', '|---|---|---:|---:|---:|'])
    for m in models:
        report.append(f"| {m['model']} | {m['status']} | {m['recorded_calls']} | {m.get('length_responses', '')} | {m.get('parse_invalid', '')} |")
    (LOG / 'STATUS.md').write_text('\n'.join(report) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Poll existing jobs only; never submits or modifies jobs.')
    args = parser.parse_args()
    while True:
        result = collect()
        print(json.dumps({k: result[k] for k in ('updated_at', 'all_jobs_terminal',
            'models_with_all_calls_recorded', 'total_recorded_logical_calls')}), flush=True)
        if not args.watch or result['all_jobs_terminal']:
            break
        time.sleep(45)


if __name__ == '__main__':
    main()
