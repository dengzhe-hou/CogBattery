"""Pack a finished run into one small file for submission, and score a packed submission.

    python -m scripts.release.pack_submission pack --run results/submissions/qwen2.5__7b \
        --out qwen2.5__7b.cogbattery.jsonl.gz
    python -m scripts.release.pack_submission score --pack qwen2.5__7b.cogbattery.jsonl.gz \
        --materials results/paper_materials/manifest.json --output results/scored

`pack` keeps each step's reply and validity fields and the SHA-256 of the request that was sent.
Request bodies and raw server replies are left out, which makes the file a few MB instead of
about 200 MB.

`score` replays the paper's run loop over the packed replies. Every request is rebuilt from the
materials and the earlier replies, and its SHA-256 must equal the packed one, so a run on other
materials or with a changed history is rejected. The replayed records are written in the paper's
record layout and scored with the paper's scorers.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.experiments.human13_20261017 import client as paper_client
from scripts.experiments.human13_20261017.common import write_json
from scripts.experiments.human13_20261017.registry import module_for
from scripts.experiments.human13_20261017.run import live_spec, run_task, slug
from scripts.release.run_model import TokenCounter

KEEP = ('response', 'raw_content', 'finish_reason', 'protocol_valid', 'protocol_errors', 'technical_valid',
        'usage', 'prompt_plus_completion_exceeds_context')


def request_digest(wire):
    return hashlib.sha256(json.dumps(wire, ensure_ascii=False).encode()).hexdigest()


def pack(run, out):
    run, rows = Path(run), []
    meta = json.loads((run / 'run_manifest.json').read_text())
    for task in meta['tasks']:
        summary = run / task / 'summary.json'
        if not summary.is_file():
            raise SystemExit(f'{task} has no summary.json; finish or resume the run first.')
        for path in sorted((run / task / 'calls').glob('*.json')):
            record = json.loads(path.read_text())
            rows.append({'task': task, 'identity': path.stem, **{k: record[k] for k in KEEP},
                         'request_sha256': request_digest(record['desired_wire_request'])})
    with gzip.open(out, 'wt', encoding='utf-8') as handle:
        handle.write(json.dumps({'header': meta, 'summaries': {t: json.loads((run / t / 'summary.json').read_text())
                                                                for t in meta['tasks']}}, ensure_ascii=False) + '\n')
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{len(rows)} steps from {len(meta["tasks"])} tasks written to {out}')


class ReplayClient:
    """Returns packed replies in place of a server, after checking the rebuilt request."""

    def __init__(self, replies, backend):
        self.replies, self.backend = replies, backend
        self.model_digest = None
        self.serving_checked = True
        self.transport_starts = 0

    def send(self, wire, identity, folder):
        row = self.replies.get(identity)
        if row is None:
            raise paper_client.TransportFailure(f'No packed reply for {identity}')
        if request_digest(wire) != row['request_sha256']:
            raise SystemExit(f'Request for {identity} differs from the one the submitter sent.')
        self.transport_starts += 1
        return {**{k: row[k] for k in KEEP}, 'thinking': '', 'reasoning_channels': {}, 'budget_verdict': None,
                'raw_response': None, 'wire_request': wire, 'latency_seconds': None, 'transport_attempts': 1}


def score(pack_path, materials, output):
    with gzip.open(pack_path, 'rt', encoding='utf-8') as handle:
        head = json.loads(handle.readline())
        rows = [json.loads(line) for line in handle if line.strip()]
    meta = head['header']
    model = meta['model']
    spec = live_spec()
    if model not in spec['models']:
        spec['models'] = spec['models'] + [model]
    entries = {t['task']: t for t in json.loads(Path(materials).read_text())['tasks']}
    root = Path(output) / slug(model)
    counter = TokenCounter(None if meta['tokenizer'].startswith('approximate') else meta['tokenizer'])
    results = []
    for task in meta['tasks']:
        replies = {r['identity']: r for r in rows if r['task'] == task}
        bundle = json.loads(Path(entries[task]['material_file']).read_text())
        replay = ReplayClient(replies, {'kind': 'replay', 'submitted_backend': meta['backend']})
        result = run_task(model, task, bundle, spec, root / task, module=module_for(task), counter=counter,
                          cache=None, client_factory=lambda: replay)
        if result['new_calls'] != len(replies):
            raise SystemExit(f'{task}: replayed {result["new_calls"]} of {len(replies)} packed replies.')
        results.append(result)
        print(json.dumps({k: result[k] for k in ('task', 'status', 'planned_steps', 'new_calls',
                                                 'technical_valid', 'not_run_total')}), flush=True)
    write_json(root / 'submission_header.json', head)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack')
    p.add_argument('--run', type=Path, required=True, help='results/submissions/<model>')
    p.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('score')
    s.add_argument('--pack', type=Path, required=True)
    s.add_argument('--materials', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.run, args.out)
    else:
        score(args.pack, args.materials, args.output)


if __name__ == '__main__':
    main()
