"""Derive the native answer reserve per model and paradigm, vocabulary-only, offline.

R = c + J + W + e (MERGE_PLAN.md 3.3):
  c = tokens of the injected '</think>' (counted in the total, native_budget README:7-30);
  J = tokens of the longest legal final answer, canonical JSON json.dumps(value, ensure_ascii=False),
      under this model's own tokenizer. Enumerations: every legal option. Unbounded
      formats (digit recall pattern, CVLT word arrays): the longest complete correct
      answer present in the materials;
  W = trailing whitespace the backend JSON grammar allows after the root value:
      0 (canonical) or 22 ("\\n"{1,2} plus up to 20 spaces/tabs,
      json-schema-to-grammar.cpp:228); which one applies is the user's proposal 1;
  e = 1 end-of-sequence token, counted in final_eval_count.
Thinking cap = 8192 - R. No model is loaded for inference; only GGUF vocabularies.

Limits (LIMITATIONS below, also in SPEC answer_reserve.limitations): W covers one
whitespace run, after the root value; the grammar allows the same run after "[",
after every array item and after every comma, which W does not count. J is the
canonical tokenization, a lower bound on generated final tokens.
"""
import argparse
import json
from pathlib import Path

from .choice_constraints import _DRM_OPTIONS, choice_options
from .common import ROOT, write_json

R6_MANIFEST = ROOT / 'results/human13_20261017/prepared_54_drm_sentence/manifest.json'
NATIVE_MODELS = ('qwen3:4b', 'deepseek-r1:7b', 'deepseek-r1:14b')
RESERVE_TASKS = ('digit_span', 'n_back', 'stroop', 'flanker', 'go_nogo', 'cvlt_word_list',
                 'drm_false_memory', 'source_monitoring', 'epitome_tom', 'confidence_calibration')
VARIANTS = {'canonical_w0': 0, 'grammar_ws_w22': 22}
TOTAL = 8192
LIMITATIONS = [
    "W=22 counts one whitespace run, after the root value. The backend grammar allows the same run (\"\" | \" \" | \"\\n\"{1,2} [ \\t]{0,20}; json-schema-to-grammar.cpp:228) after \"[\", after every array item and after every comma (:932 array rule; every item ends with space: string with minLength :945, pattern :534, enum :844). A 16-word CVLT array has 33 such positions; the canonical single space after each comma is inside J, the rest is not counted. Enumerations and the digit-recall string have one position, after the root value.",
    "J is the token count of the canonical JSON text under the model tokenizer. Constrained sampling may split the same text into more tokens, so J is a lower bound on the generated final tokens.",
]


def legal_answers(task, bundle):
    """kind -> {canonical JSON text: value} for every legal final answer."""
    out = {}
    for episode in bundle['episodes']:
        for step in episode['steps']:
            kind = step['kind']
            if kind == 'ready':
                values = ['Ready']
            elif kind == 'digit_recall':
                values = [' '.join(str(d) for d in step['gold'])]
            elif kind in ('cvlt_recall', 'cvlt_recall_followup'):
                values = [list(step['gold'])]
            elif kind == 'drm_report':
                values = [label for labels in _DRM_OPTIONS.values() for label in labels]
            else:
                values = choice_options(step)
                if values is None:
                    raise ValueError(f'{task}: no legal answer set for {kind}')
            for value in values:
                out.setdefault(kind, {})[json.dumps(value, ensure_ascii=False)] = value
    return out


def derive(manifest_path=R6_MANIFEST, models=NATIVE_MODELS, counter_factory=None):
    if counter_factory is None:
        from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter as counter_factory
    manifest = json.loads(Path(manifest_path).read_text())
    entries = {e['task']: e for e in manifest['tasks']}
    answers = {t: legal_answers(t, json.loads(Path(entries[t]['material_file']).read_text())) for t in RESERVE_TASKS}
    rows, values = [], {v: {} for v in VARIANTS}
    for model in models:
        counter = counter_factory(model)
        try:
            close = counter.count('</think>', special=True)
            if close != 1:
                raise RuntimeError(f'{model}: </think> is {close} tokens; the backend requires one')
            for task in RESERVE_TASKS:
                best = None
                for kind, texts in answers[task].items():
                    for text in texts:
                        n = counter.count(text)
                        if best is None or n > best[0]:
                            best = (n, kind, text)
                j, kind, text = best
                for variant, w in VARIANTS.items():
                    r = close + j + w + 1
                    values[variant].setdefault(model, {})[task] = r
                    rows.append({'model': model, 'task': task, 'variant': variant, 'step_kind': kind,
                                 'longest_answer_json': text, 'close_tokens': close, 'answer_json_tokens': j,
                                 'trailing_whitespace_tokens': w, 'eos_tokens': 1, 'reserve': r,
                                 'thinking_cap': TOTAL - r})
        finally:
            counter.close()
    return {'scope': 'offline_vocabulary_only_reserve_derivation', 'source_manifest': str(manifest_path),
            'formula': 'R = close(</think>) + J(longest legal final, canonical JSON) + W + EOS; thinking cap = 8192 - R',
            'variants': VARIANTS, 'limitations': LIMITATIONS, 'pending_user_decision': 'proposal 1: which variant applies',
            'model_calls': 0, 'values': values, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/human54_unified_20261018/prepared_r7/RESERVE_DERIVATION.json')
    args = parser.parse_args()
    result = derive()
    write_json(args.output, result)
    print(json.dumps(result['values'], indent=1))


if __name__ == '__main__':
    main()
