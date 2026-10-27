"""Inference disclosures for score-table presentation; no scoring or statistics."""
NATIVE_MODELS = {'deepseek-r1:7b', 'deepseek-r1:14b', 'qwen3:4b'}


def inference_metadata(model, tasks):
    native = model in NATIVE_MODELS
    if not native:
        return dict(native_thinking=False, token_budget='',
                    token_budget_note='Direct answering; per-step budgets vary by protocol (16-1024 tokens across the study).')
    tasks = list(tasks)
    if tasks == ['operation_span']:
        budget = '128'
    elif 'operation_span' not in tasks:
        budget = '8192'
    else:
        other = ', '.join(t for t in tasks if t != 'operation_span')
        budget = f'128 (operation_span); 8192 ({other})'
    return dict(native_thinking=True, token_budget=budget,
                token_budget_note='Per-response total thinking plus final-answer tokens; operation_span uses 128, other tasks 8192.')


def annotate_score_frame(frame, tasks):
    """Return a display copy; numeric score columns are not recomputed."""
    displayed = frame.copy()
    disclosures = [inference_metadata(model, tasks) for model in displayed.index]
    for column in ('native_thinking', 'token_budget', 'token_budget_note'):
        displayed[column] = [row[column] for row in disclosures]
    return displayed
