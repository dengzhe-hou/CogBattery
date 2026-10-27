"""Build the exact r7 wire request for one step from the actual history.

Nothing here serves a model. Field order is part of the request bytes, so it
follows the two verified builders exactly:
- no thinking_budget (51 direct models; native false belief):
  human13_format_repair_20261017/client.py:95-102;
- thinking_budget (native models, the other twelve paradigms):
  native_history_20261018/protocol.py:87-93.
Body bytes = json.dumps(wire, ensure_ascii=False) (format_repair client.py:103).
"""
from copy import deepcopy
from dataclasses import dataclass
import json

from .budgets import context_tokens, is_native, thinking_budget, total_tokens
from .common import default_messages
from .constraints import ARRAY_PROMPT, ARRAY_SCHEMA, OLD_PROMPT, final_stop, response_schema

# native_budget_20261018/prepare.py:16-20, 34-59 (copied); applied to all 54 models (USER_DECISIONS.md:7).
HISTORY_END = 'End of past history.\n'
CLASSIFICATION_CLARIFICATION = (
    'The earlier training strings remain the material for this classification; '
    'their past reproduction instructions do not set this turn\'s reply format.'
)


class ContextOverflow(RuntimeError):
    """Actual history plus the fixed output allowance does not fit; nothing is trimmed."""


def _revise_current(current, kind):
    if kind == 'wag_training':
        phase, target, rule = current.split('\n')
        if not target.startswith('Current training string: ') or not rule.startswith('Read the following letter string'):
            raise ValueError('Training source no longer has the reviewed instruction order')
        return '\n'.join((phase, rule, target))
    if kind == 'wag_answer':
        phase, material = current.split('\n', 1)
        instructions, target = material.rsplit('Current test string: ', 1)
        if not phase.startswith('Current phase: classification only.'):
            raise ValueError('Classification source no longer has the reviewed phase')
        return (instructions + phase + '\n' + CLASSIFICATION_CLARIFICATION
                + '\nCurrent test string: ' + target)
    raise ValueError('Only the two authorized wager kinds may change prompts')


def revise_wager_messages(messages, kind, history_length):
    """revise_wager_prompt (native_budget prepare.py:34-59) on the message list.

    Added branch (implementation choice, MERGE_PLAN.md 3.5): training step 1 has
    no history block, so the same reordering applies to the whole message.
    Only the current turn changes; the past record stays verbatim.
    """
    result = deepcopy(messages)
    if len(result) != 1 or result[0]['role'] != 'user':
        raise ValueError('Expected the recorded one-user-message wager presentation')
    before = result[0]['content']
    history, separator, current = before.partition(HISTORY_END)
    if not separator:
        if history_length != 0 or kind != 'wag_training':
            raise ValueError('Recorded wager history boundary is missing')
        history, current = '', before
    result[0]['content'] = history + separator + _revise_current(current, kind)
    return result


@dataclass
class Request:
    wire: dict
    body: str
    context: dict
    schema: dict
    total: int
    thinking_budget: int


def model_inference(spec, model):
    """Per-model view of SPEC.inference (context and thinking mode)."""
    inf = deepcopy(spec['inference'])
    inf['context_tokens'] = context_tokens(model, spec['inference'])
    inf['native_think'] = is_native(model, spec['inference'])
    return inf


def _fit(module, episode, step, history, counter, inf, total):
    messages = getattr(module, 'messages', default_messages)(episode, step, history)
    if episode.get('context_policy') == 'legacy_strachan':
        # Unchanged FB input-only rule with persistent pair dropping (run_r6.py:37-47).
        from scripts.experiments.same_stimulus_20261011.run import fit_session
        raw_tokens = lambda ms: sum(counter.count(m['content']) for m in ms)
        head = messages[:1] if messages[0]['role'] == 'system' else []
        body = messages[len(head):-1]
        pairs = [body[i:i+2] for i in range(0, len(body), 2)]
        fitted, raw, dropped = fit_session(head, pairs, messages[-1], raw_tokens, inf['context_tokens'],
                                          inf['template_margin_tokens'], inf['per_message_margin_tokens'],
                                          item_id=step['id'])
        return fitted, {'raw_content_tokens': raw, 'history_pairs_dropped': dropped,
                        'policy': 'unchanged_legacy_strachan_input_rule'}
    return messages, {'history_pairs_dropped': 0, 'policy': 'full_prompt_plus_output_budget'}


def build_request(model, task, episode, step, history, spec, module, counter, *, reserve_variant=None):
    inf = model_inference(spec, model)
    native = inf['native_think']
    total = total_tokens(task, step, model, inf)
    budget = thinking_budget(task, model, inf, reserve_variant)
    # For legacy FB, the caller drops the same number of earliest history pairs
    # persistently after this step (run_r6.py:89-91).
    messages, context = _fit(module, episode, step, history, counter, inf, total)
    schema = response_schema(step, history=history, current_prompt=messages[-1]['content'])
    messages = deepcopy(messages)
    if task == 'operation_span' and step['kind'] == 'ospan_recall':
        # human13_ospan_uniform_20261017/protocol.py:51-56: current turn only.
        if messages[-1]['content'] != OLD_PROMPT:
            raise ValueError('Unexpected OSPAN recall instruction; inspect before replacing it')
        messages[-1]['content'] = ARRAY_PROMPT
        schema = deepcopy(ARRAY_SCHEMA)
    if task == 'post_decision_wagering' and step['kind'] in ('wag_training', 'wag_answer'):
        messages = revise_wager_messages(messages, step['kind'], len(history))
    raw = sum(counter.count(m['content']) for m in messages)
    estimate = (raw + inf['template_margin_tokens'] + inf['per_message_margin_tokens'] * len(messages)
                + total)
    if episode.get('context_policy') == 'legacy_strachan':
        # Reported only: the unchanged FB rule checks input alone (same_stimulus fit_session).
        context.update(estimated_input_plus_output_not_enforced=estimate, context_tokens=inf['context_tokens'])
    else:
        context.update(raw_content_tokens=raw, estimated_input_plus_output=estimate,
                       context_tokens=inf['context_tokens'])
        if estimate > inf['context_tokens']:
            raise ContextOverflow(f'{episode["id"]}/{step["id"]}: estimated {estimate} > '
                                  f'{inf["context_tokens"]}; no history deleted, no call.')
    options = {'temperature': inf['temperature'], 'num_predict': total, 'top_p': inf['native_top_p'],
               'num_ctx': inf['context_tokens'], 'num_thread': inf['server_threads']}
    if budget is None:
        wire = {'model': model, 'messages': messages, 'stream': False, 'think': native, 'options': options}
        if schema is not None:
            wire['format'] = schema
        stop = final_stop(schema, native)
        if stop:
            wire['options']['stop'] = stop
    else:
        if schema is None:
            raise ValueError('The backend accepts thinking_budget only with a final format')
        wire = {'model': model, 'messages': messages, 'stream': False, 'think': True,
                'format': schema, 'thinking_budget': budget, 'options': options}
    return Request(wire=wire, body=json.dumps(wire, ensure_ascii=False), context=context,
                   schema=schema, total=total, thinking_budget=budget)
