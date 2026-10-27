"""Whole-answer parsing additions and the completion gate (copied, not redesigned).

Source: human13_format_repair_20261017/scoring.py:12-67. Only the imports point
at this entry's copies of the repaired common/memory modules. Mistakes are never
deleted and nothing is corrected against an answer key. Free false-belief
answers keep the published scorer unchanged.
"""
import re

from .common import score_record as _score_record
from .memory import _answer_text


def score_step(module, step, response):
    """Read a whole answer without deleting mistakes or selecting a candidate."""
    original = module.score_step(step, response)
    result = original
    normalization = None
    if not original['valid']:
        text = _answer_text(response)
        if step['kind'] == 'wag_training':
            match = re.fullmatch(r'(?:[Rr]:\s*)?([A-Z]+(?:[\s,]+[A-Z]+)*)[.!]?', text)
            if match:
                sequence = re.sub(r'[\s,]+', '', match.group(1))
                result = module.score_step(step, sequence)
                normalization = 'whole_wag_letter_sequence'
        elif step['kind'] == 'ospan_recall':
            # Each underscore remains one forgotten position, including at either
            # end; do not trim to the expected set size or use the answer key.
            text = text.upper()
            match = re.fullmatch(r'([A-Z_]+)[.!]?', text)
            if match:
                sequence = ' '.join(match.group(1))
                result = module.score_step(step, sequence)
                normalization = 'whole_ospan_character_sequence'
    output = {**result, 'pre_format_repair_score': original}
    if 'strict_original_score' in original:
        output['strict_original_score'] = original['strict_original_score']
    if normalization is not None:
        output['format_normalization'] = normalization
    return output


class _ScoringModule:
    """Use with runners that accept a module with score_step/messages."""
    def __init__(self, module):
        self.original_module = module

    def score_step(self, step, response):
        return score_step(self.original_module, step, response)

    def __getattr__(self, name):
        return getattr(self.original_module, name)


def wrap_module(module):
    return module if isinstance(module, _ScoringModule) else _ScoringModule(module)


def score_record(module, record):
    """Apply the completion gate and retain the original record score."""
    wrapped = wrap_module(module)
    result = _score_record(wrapped, record)
    result['pre_format_repair_record_score'] = _score_record(wrapped.original_module, record)
    return result
