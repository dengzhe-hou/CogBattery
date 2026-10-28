"""A stand-in model server for a no-cost smoke test of the battery.

    python -m scripts.release.stub_server --port 11555

It answers /api/chat (Ollama) and /v1/chat/completions (OpenAI-compatible) with a reply that fits
each step's JSON answer schema. Choices are picked from the allowed options by a hash of the prompt,
so scores vary but mean nothing. No model is loaded.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

FREE_TEXT = 'basket'


def answer(messages, schema):
    seed = int(hashlib.sha256(messages[-1]['content'].encode()).hexdigest(), 16)
    if schema is None:
        return FREE_TEXT
    if schema.get('type') == 'array':
        item = schema.get('items', {})
        return json.dumps(['A'] if item.get('pattern') == '^[A-Z_]$' else ['apple'])
    if 'enum' in schema:
        return json.dumps(schema['enum'][seed % len(schema['enum'])])
    pattern = schema.get('pattern', '')
    if '0-9' in pattern:
        return json.dumps(' '.join(str((seed >> (4 * i)) % 10) for i in range(3)))
    if '_' in pattern:
        return json.dumps('A B C')
    return json.dumps('ABC')


def tokens(text):
    return max(1, len(text) // 4)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body['messages']
        prompt = sum(tokens(m['content']) for m in messages)
        if self.path.endswith('/api/chat'):
            content = answer(messages, body.get('format'))
            reply = {'model': body['model'], 'message': {'role': 'assistant', 'content': content}, 'done': True,
                     'done_reason': 'stop', 'prompt_eval_count': prompt, 'eval_count': tokens(content)}
        elif self.path.endswith('/chat/completions'):
            schema = (body.get('response_format') or {}).get('json_schema', {}).get('schema')
            content = answer(messages, schema)
            reply = {'choices': [{'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': prompt, 'completion_tokens': tokens(content)}}
        else:
            self.send_error(404)
            return
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(port):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=11555)
    args = parser.parse_args()
    print(f'Stub server on http://127.0.0.1:{args.port}')
    serve(args.port).serve_forever()


if __name__ == '__main__':
    main()
