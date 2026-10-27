"""Authorized final formats; no answer keys, target vocabulary or fixed lengths."""
import json
import re

from scripts.experiments.human13_repair_20261017.choice_constraints import choice_options


STRING_PATTERNS = {
    'digit_recall': r'^[0-9]( [0-9])*$',
    'ospan_recall': r'^[A-Z_]( [A-Z_])*$',
    'wag_training': r'^[A-Z]+$',
}


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


def decode_final(raw, schema):
    """Return history text and schema errors, keeping array JSON verbatim.

    Empty/length/phase validity is checked by Client independently. A formatted
    but incomplete answer is never made complete by decoding its syntax.
    """
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
