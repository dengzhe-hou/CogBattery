"""Per-model, per-paradigm token budgets for the r7 entry, read only from SPEC.

Recorded sources (cited in SPEC.json and MERGE_PLAN.md section 3.2):
- direct per-step budgets: r6 material max_tokens mapped once by
  inference.direct_step_budget_map (human13_repair_20261017/pilot_materials.py:132-137),
  applied in prepare_r7.py; the runner reads step['max_tokens'].
- operation_span: 128 per call for all 54 models (human13_ospan_uniform_20261017/protocol.py:19).
- native total 8192 and context 12288 (human13_repair_20261017/SPEC.json:89-103).
- native thinking caps: OSPAN 96, wagering 8160 (native_history_20261018/protocol.py:80-85);
  false belief none (USER_DECISIONS.md:12); the other ten = 8192 - answer reserve
  (USER_DECISIONS.md:6), whose derivation variant is the user's pending proposal 1.
"""


class PendingUserDecision(RuntimeError):
    """A protocol value the user has not decided yet; no request may be built."""


def is_native(model, inference):
    return model in inference['native_think_models']


def context_tokens(model, inference):
    return inference['contexts'].get(model, inference['context_tokens'])


def total_tokens(task, step, model, inference):
    """num_predict: thinking plus final for native models, final only for direct ones."""
    override = inference['task_total_override']
    if task in override:
        return override[task]
    if is_native(model, inference):
        return inference['native_total_output_tokens']
    return step['max_tokens']


def thinking_budget(task, model, inference, reserve_variant=None):
    """None means the request carries no thinking_budget field."""
    if not is_native(model, inference):
        return None
    staged = inference['native_thinking_budget']
    if task in staged:
        return staged[task]
    if task not in inference['native_reserve_tasks']:
        raise ValueError(f'No recorded native thinking budget for {task}')
    reserve = inference['answer_reserve']
    variant = reserve_variant or reserve['selected_variant']
    if variant is None:
        raise PendingUserDecision(
            f'{model}/{task}: answer-reserve variant is pending the user (proposal 1); '
            'no native request with a thinking cap may be built or sent.')
    value = reserve['values'][variant][model][task]
    return inference['native_total_output_tokens'] - value
