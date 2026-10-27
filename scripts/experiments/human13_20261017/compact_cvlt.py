"""Lossless CVLT history presentation; materials, responses and steps stay intact.

Only repeated word/response wrappers and chat role boundaries are merged. The
current request stays verbatim, with explicit past/current stage boundaries.
Instructions enter the transcript at their original stage, and no answer keys
or future prompts are read into the history.
"""
FORMAT_RULES = {
    'ready': 'Reply only Ready.',
    'cvlt_recall': 'Return only a JSON array of the words recalled, in recall order. Use [] if none.',
    'cvlt_recall_followup': 'Return only a JSON array of any additional words recalled, in recall order. Use [] if none.',
    'cvlt_recognition': 'Reply only YES or NO.',
}
PHASES = {
    'ready': 'study confirmation',
    'cvlt_recall': 'recall',
    'cvlt_recall_followup': 'additional recall',
    'cvlt_recognition': 'recognition',
}


def split_prompt(step):
    """Split the actual prompt reversibly; never consult gold or metadata."""
    kind = step['kind']
    if kind not in FORMAT_RULES:
        raise ValueError(f'Unsupported CVLT compact step kind: {kind}')
    rule = FORMAT_RULES[kind]
    suffix = '\n' + rule
    prompt = step['prompt']
    if not prompt.endswith(suffix):
        raise ValueError(f'CVLT response-format instruction differs: {step["id"]}')
    content = prompt[:-len(suffix)]
    if kind in ('ready', 'cvlt_recognition'):
        if content.startswith('Word: '):
            prefix, body = '', content[len('Word: '):]
        else:
            prefix, marker, body = content.rpartition('\nWord: ')
            if not marker:
                raise ValueError(f'CVLT word presentation differs: {step["id"]}')
        if not body or '\n' in body:
            raise ValueError(f'CVLT prepared word is not a single line: {step["id"]}')
        word_wrapper = True
    else:
        prefix, body, word_wrapper = '', content, False
    return {'prefix': prefix, 'body': body, 'format_rule': rule,
            'word_wrapper': word_wrapper}


def history_ledger(history):
    """Every past stimulus/instruction and raw reply, in exact original order."""
    return [{**split_prompt(row['step']), 'response': row['response']} for row in history]


def messages(episode, step, history):
    """Serialize complete past history plus the unchanged current request."""
    if episode['task'] != 'cvlt_word_list':
        raise ValueError('This compact presentation is only for CVLT')
    ids = [s['id'] for s in episode['steps']]
    position = ids.index(step['id'])
    if episode.get('parent') or [r['step']['id'] for r in history] != ids[:position]:
        raise ValueError('CVLT history must contain every preceding step, in order')

    parts = []
    if history:
        parts.append('Past history (read-only record). Past instructions applied to their recorded phases; '
                     'they do not set the response format for this turn. All earlier material remains visible. '
                     'Each record: shown word or instruction, a tab, then your exact reply.')
        formats_seen = set()
        previous_kind = None
        for record, row in zip(history, history_ledger(history)):
            kind = record['step']['kind']
            if kind != previous_kind:
                parts.append('Past phase: ' + PHASES[kind] + '.')
                previous_kind = kind
            if row['prefix']:
                parts.append(row['prefix'])
            if row['format_rule'] not in formats_seen:
                parts.append(row['format_rule'])
                formats_seen.add(row['format_rule'])
            # Prepared words/instructions are single-line text. Both columns stay
            # literal; in particular, never re-encode an actual JSON recall reply.
            parts.append(row['body'] + '\t' + row['response'])
        parts.append('End of past history.')
    parts.extend(['Current phase: ' + PHASES[step['kind']] + '.',
                  'Use only the response format in the current request. Do not include history labels.',
                  'Current request:', step['prompt']])
    result = [{'role': 'system', 'content': episode['system']}] if episode.get('system') else []
    result.append({'role': 'user', 'content': '\n'.join(parts)})
    return result
