"""Final-answer formats for the r7 unified entry (copied verbatim, not redesigned).

Sources (read-only, verified there):
- response_schema / final_stop / string decode_final:
  human13_format_repair_20261017/constraints.py:8-67 (interface final_typed_json_v2).
- ARRAY_SCHEMA / array decode: human13_ospan_array_20261017/constraints.py:5-20.
- OLD_PROMPT / ARRAY_PROMPT: human13_ospan_array_20261017/probe.py:21-25.
- composite decoder: human13_ospan_history_20261017/constraints.py:6-7.
No answer keys, target vocabulary or fixed lengths enter any schema.
"""
import json
import re

from .choice_constraints import choice_options


STRING_PATTERNS = {
    'digit_recall': r'^[0-9]( [0-9])*$',
    'ospan_recall': r'^[A-Z_]( [A-Z_])*$',
    'wag_training': r'^[A-Z]+$',
}

ARRAY_SCHEMA = {'type': 'array', 'minItems': 1,
                'items': {'type': 'string', 'pattern': '^[A-Z_]$'}}

OLD_PROMPT = ("Recall this group's letters in order, separated by spaces. "
              'Use _ for a forgotten position. Output only the sequence.')
ARRAY_PROMPT = ("Recall this group's letters in order. Use _ for a forgotten position. "
                'Return only a JSON array. Each element must be a string containing one uppercase '
                'letter or _. Close the array, then stop.')


def response_schema(step, history=None, current_prompt=None):
    options = choice_options(step, history=history, current_prompt=current_prompt)
    if options is not None:
        return {'type': 'string', 'enum': options}
    kind = step['kind']
    if kind == 'ready':
        return {'type': 'string', 'enum': ['Ready']}
    if kind in STRING_PATTERNS:
        return {'type': 'string', 'pattern': STRING_PATTERNS[kind]}
    if kind in ('cvlt_recall', 'cvlt_recall_followup'):
        return {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}
    return None


def final_stop(schema, native_think):
    """Stop direct root strings at a trailing LF, retaining the closing quote.

    The authorized string grammars have no leading LF or literal LF inside a
    JSON string. Arrays may contain structural newlines; native thinking may
    contain newlines before its final phase. Neither receives this stop rule.
    """
    return ['\n'] if not native_think and schema is not None and schema.get('type') == 'string' else []


def decode_string_or_word_array(raw, schema):
    """format_repair decode_final: history text and schema errors, arrays verbatim."""
    if schema is None:
        return raw.strip(), []
    retained = raw if schema['type'] == 'array' else raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return retained, ['invalid_format_json']
    if schema['type'] == 'array':
        if (not isinstance(value, list)
                or any(not isinstance(word, str) or not word.strip() for word in value)):
            return retained, ['invalid_word_array']
        return raw, []
    if schema['type'] != 'string':
        raise ValueError('Unsupported authorized response schema')
    if 'enum' in schema:
        if not isinstance(value, str) or value not in schema['enum']:
            return retained, ['answer_outside_legal_options']
    elif 'pattern' in schema:
        if not isinstance(value, str) or re.fullmatch(schema['pattern'], value) is None:
            return retained, ['answer_outside_string_pattern']
    else:
        raise ValueError('Authorized string response requires enum or pattern')
    return value, []


def decode_ospan_array(raw, schema):
    """OSPAN character array: no length ceiling, no deduplication or trimming."""
    if schema != ARRAY_SCHEMA:
        raise ValueError('Only the fixed OSPAN character-array schema is supported.')
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return raw, ['invalid_format_json']
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or re.fullmatch('[A-Z_]', value) is None
                   for value in values)):
        return raw, ['invalid_ospan_character_array']
    return ' '.join(values), []


def decode_final(raw, schema):
    return decode_ospan_array(raw, schema) if schema == ARRAY_SCHEMA else decode_string_or_word_array(raw, schema)
