"""Small shared episode interface. Importing this module never loads a model."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = Path(os.environ.get('COGBATTERY_RAW', ROOT / 'data' / 'raw'))
TASKS = (
    'digit_span', 'n_back', 'operation_span', 'stroop', 'flanker', 'go_nogo',
    'cvlt_word_list', 'drm_false_memory', 'source_monitoring', 'false_belief',
    'epitome_tom', 'confidence_calibration', 'post_decision_wagering',
)


def step_budget(step, inference):
    """Native reasoning uses a total thinking-plus-final allowance."""
    if inference.get('native_think'):
        return inference['native_total_output_tokens']
    return step.get('max_tokens', inference['max_tokens'])


def score_record(module, record):
    """Keep parse diagnostics while excluding incomplete replies from scoring."""
    response = record['response']
    parsed = module.score_step(record['step'], response)
    complete = (record.get('finish_reason') == 'stop'
                and record.get('protocol_valid') is not False
                and bool(response.strip())
                and not record.get('prompt_plus_completion_exceeds_context', False))
    result = parsed if complete else module.score_step(record['step'], '')
    return {**result, 'parsed_valid': parsed['valid'],
            'response_complete': complete,
        'behavior_eligible': bool(complete and result['valid']
                                  and not record.get('synthetic', False)
                                  and record.get('research_results_eligible') is not False),
            'protocol_failure': None if complete else 'incomplete_or_invalid_response'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def default_messages(episode, step, history):
    """Only prompt/response content is sent; gold and scoring metadata stay local."""
    messages = []
    if episode.get('system'):
        messages.append({'role': 'system', 'content': episode['system']})
    mode = episode.get('history_mode', 'full')
    if mode not in ('full', 'independent'):
        raise ValueError(f'Custom message builder required for history_mode={mode}')
    if mode == 'full':
        for record in history:
            messages.extend(({'role': 'user', 'content': record['step']['prompt']},
                             {'role': 'assistant', 'content': record['response']}))
    messages.append({'role': 'user', 'content': step['prompt']})
    return messages


def validate_task(task, bundle):
    """Detect missing/duplicated steps and invalid branch order before inference."""
    if bundle['status'] not in ('ready', 'blocked'):
        raise ValueError(f'Unknown readiness status for {task}')
    if bundle['status'] == 'blocked' and not bundle['blockers']:
        raise ValueError(f'Blocked task has no explanation: {task}')
    ids = set()
    calls = 0
    for episode in bundle['episodes']:
        if episode['task'] != task or episode['id'] in ids:
            raise ValueError(f'Wrong task or repeated episode: {episode["id"]}')
        parent = episode.get('parent')
        if parent and parent not in ids:
            raise ValueError(f'Parent must precede its test branch: {parent}')
        ids.add(episode['id'])
        steps = set()
        if not episode['steps']:
            raise ValueError(f'Empty episode: {episode["id"]}')
        for step in episode['steps']:
            if step['id'] in steps or not isinstance(step['prompt'], str) or not step['prompt']:
                raise ValueError(f'Duplicate/empty step: {episode["id"]}/{step["id"]}')
            for required in ('kind', 'gold', 'meta'):
                if required not in step:
                    raise ValueError(f'Missing {required}: {episode["id"]}/{step["id"]}')
            steps.add(step['id'])
            calls += 1
    if bundle['status'] == 'ready' and not calls:
        raise ValueError(f'Ready task has no executable steps: {task}')
    return calls
