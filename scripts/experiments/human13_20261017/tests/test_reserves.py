"""Answer reserves: recomputed from the vocabularies, equal to SPEC, consistent arithmetic.

Loads only GGUF vocabularies (vocab_only=True, no tensors, no inference context).
"""
import json
from pathlib import Path

import pytest

from scripts.experiments.human13_20261017.derive_reserves import (
    LIMITATIONS, NATIVE_MODELS, RESERVE_TASKS, VARIANTS, derive, legal_answers, R6_MANIFEST)

HERE = Path(__file__).resolve().parents[1]
SPEC = json.loads((HERE / 'SPEC.json').read_text())
RESERVE = SPEC['inference']['answer_reserve']


class FakeCounter:
    def __init__(self, model):
        self.model = model

    def count(self, text, special=False):
        return 1 if special else len(text)

    def close(self):
        pass


def test_formula_on_a_hand_written_counter():
    result = derive(counter_factory=FakeCounter)
    row = next(r for r in result['rows'] if r['task'] == 'epitome_tom' and r['variant'] == 'canonical_w0')
    assert row['longest_answer_json'] == '"A"' and row['answer_json_tokens'] == 3
    assert row['reserve'] == 1 + 3 + 0 + 1 and row['thinking_cap'] == 8192 - 5
    w22 = next(r for r in result['rows'] if r['task'] == 'epitome_tom' and r['variant'] == 'grammar_ws_w22'
               and r['model'] == row['model'])
    assert w22['reserve'] == row['reserve'] + 22


def test_reserve_table_matches_spec_and_the_saved_derivation():
    saved = json.loads((HERE.parents[2] / RESERVE['derivation_file']).read_text())
    assert saved['values'] == RESERVE['values']
    recomputed = derive()  # real vocabularies, vocab-only
    assert recomputed['values'] == RESERVE['values']
    for row in recomputed['rows']:
        assert row['close_tokens'] == 1
        assert row['reserve'] >= row['answer_json_tokens'] + 2
        assert row['thinking_cap'] + row['reserve'] == SPEC['inference']['native_total_output_tokens']
    assert set(RESERVE['values']) == set(VARIANTS)
    for variant in VARIANTS:
        assert set(RESERVE['values'][variant]) == set(NATIVE_MODELS)
        for model in NATIVE_MODELS:
            assert set(RESERVE['values'][variant][model]) == set(RESERVE_TASKS)
    assert RESERVE['selected_variant'] == 'grammar_ws_w22'  # user decision 2026-10-18 (USER_DECISIONS.md)


def test_longest_answers_come_from_legal_formats_not_a_flat_32():
    manifest = json.loads(R6_MANIFEST.read_text())
    entry = next(t for t in manifest['tasks'] if t['task'] == 'cvlt_word_list')
    answers = legal_answers('cvlt_word_list', json.loads(Path(entry['material_file']).read_text()))
    longest = max(answers['cvlt_recall'], key=len)
    assert len(json.loads(longest)) == 16
    assert max(r for m in RESERVE['values']['canonical_w0'].values() for r in m.values()) > 32
    assert set(SPEC['inference']['native_reserve_tasks']) == set(RESERVE_TASKS)
    assert set(SPEC['inference']['native_thinking_budget']) == {'operation_span', 'post_decision_wagering', 'false_belief'}


def test_the_two_derivation_limits_travel_with_proposal_1():
    """Verifier round 2: W counts only the run after the root value; J is a lower bound."""
    saved = json.loads((HERE.parents[2] / RESERVE['derivation_file']).read_text())
    assert RESERVE['limitations'][:2] == LIMITATIONS == saved['limitations']
    assert derive(counter_factory=FakeCounter)['limitations'] == LIMITATIONS
    manifest = json.loads(R6_MANIFEST.read_text())
    entry = next(t for t in manifest['tasks'] if t['task'] == 'cvlt_word_list')
    answers = legal_answers('cvlt_word_list', json.loads(Path(entry['material_file']).read_text()))
    words = max(len(json.loads(text)) for text in answers['cvlt_recall'])
    # "[" space, item space (x n), "," space (x n-1), "]" space: grammar positions of a word array.
    positions = 1 + words + (words - 1) + 1
    assert f'{words}-word CVLT array has {positions} such positions' in LIMITATIONS[0]
    assert 'lower bound' in LIMITATIONS[1]
