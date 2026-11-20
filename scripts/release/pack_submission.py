"""Pack a finished run into one small file for submission, and score a packed submission.

    python -m scripts.release.pack_submission pack --run results/submissions/qwen2.5__7b \
        --out qwen2.5__7b.cogbattery.jsonl.gz
    python -m scripts.release.pack_submission score --pack qwen2.5__7b.cogbattery.jsonl.gz \
        --materials results/paper_materials/manifest.json --output results/scored

`pack` keeps each step's reply and validity fields and the SHA-256 of the request that was sent.
Request bodies and raw server replies are left out, which makes the file a few MB instead of
about 200 MB.

CVLT-II list words are copyrighted, so they never enter a pack. Each one in a CVLT reply is written
as its index in the sorted study list, with its letter case, and `score` restores it from the
materials it builds itself. The local paths of the run (server URL and materials manifest) are also
left out of the header.

`score` replays the paper's run loop over the packed replies. Every request is rebuilt from the
materials and the earlier replies, and its SHA-256 must equal the packed one, so a run on other
materials or with a changed history is rejected. The replayed records are written in the paper's
record layout and scored with the paper's scorers.
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from scripts.experiments.human13_20261017 import client as paper_client
from scripts.experiments.human13_20261017.common import write_json
from scripts.experiments.human13_20261017.registry import module_for
from scripts.experiments.human13_20261017.run import live_spec, run_task, slug
from scripts.release.run_model import TokenCounter

KEEP = ('response', 'raw_content', 'finish_reason', 'protocol_valid', 'protocol_errors', 'technical_valid',
        'usage', 'prompt_plus_completion_exceeds_context')
LOCAL = ('base_url', 'materials_manifest')
CVLT = 'cvlt_word_list'
OPEN, CLOSE = '\u27e6', '\u27e7'
TOKEN = re.compile(OPEN + r'(\d+)(?::([Ul]+))?' + CLOSE)


def request_digest(wire):
    return hashlib.sha256(json.dumps(wire, ensure_ascii=False).encode()).hexdigest()


def cvlt_words(materials):
    """The sorted CVLT study words of the built materials."""
    entries = {t['task']: t for t in json.loads(Path(materials).read_text())['tasks']}
    bundle = json.loads(Path(entries[CVLT]['material_file']).read_text())
    return sorted({s['meta']['material'].lower() for e in bundle['episodes'] for s in e['steps']
                   if s['meta'].get('material')})


def _map_strings(value, change):
    if isinstance(value, str):
        return change(value)
    if isinstance(value, list):
        return [_map_strings(v, change) for v in value]
    if isinstance(value, dict):
        return {k: _map_strings(v, change) for k, v in value.items()}
    return value


def encode_words(value, words):
    """Replace each list word, also before a plural s or es, by its index and letter case."""
    index = {w: i for i, w in enumerate(words)}
    found = re.compile(r'(?<![A-Za-z])(' + '|'.join(sorted(map(re.escape, words), key=len, reverse=True))
                       + r')(?=(?:e?s)?(?![A-Za-z]))', re.IGNORECASE)

    def token(match):
        text = match.group(1)
        case = '' if text.islower() else ':' + ''.join('U' if c.isupper() else 'l' for c in text)
        return f'{OPEN}{index[text.lower()]}{case}{CLOSE}'

    def change(text):
        if OPEN in text or CLOSE in text:
            raise SystemExit(f'A CVLT reply already contains {OPEN} or {CLOSE}, so it cannot be packed.')
        return found.sub(token, text)
    return _map_strings(value, change)


def decode_words(value, words):
    def word(match):
        text, case = words[int(match.group(1))], match.group(2)
        if case is None:
            return text
        if len(case) != len(text):
            raise SystemExit('A packed CVLT word does not fit the materials; build them again.')
        return ''.join(c.upper() if flag == 'U' else c for c, flag in zip(text, case))
    return _map_strings(value, lambda text: TOKEN.sub(word, text))


def pack(run, out):
    run, rows = Path(run), []
    meta = json.loads((run / 'run_manifest.json').read_text())
    words = cvlt_words(meta['materials_manifest']) if CVLT in meta['tasks'] else None
    for task in meta['tasks']:
        summary = run / task / 'summary.json'
        if not summary.is_file():
            raise SystemExit(f'{task} has no summary.json; finish or resume the run first.')
        for path in sorted((run / task / 'calls').glob('*.json')):
            record = json.loads(path.read_text())
            kept = {k: record[k] for k in KEEP}
            if task == CVLT:
                kept = encode_words(kept, words)
            rows.append({'task': task, 'identity': path.stem, **kept,
                         'request_sha256': request_digest(record['desired_wire_request'])})
    header = {k: v for k, v in meta.items() if k not in LOCAL}
    with gzip.open(out, 'wt', encoding='utf-8') as handle:
        handle.write(json.dumps({'header': header, 'summaries': {t: json.loads((run / t / 'summary.json').read_text())
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
    if CVLT in meta['tasks']:
        words = cvlt_words(materials)
        rows = [{**r, **decode_words({k: r[k] for k in KEEP}, words)} if r['task'] == CVLT else r for r in rows]
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
