"""WAG chronological log, adopted for episodes marked compact_history_v1.

The repeated instructions/roles become one rule block and a chronological log.
Every presented string and every actual reply remain verbatim.  This changes
presentation, not material, trial order, response collection or scoring.
"""
import re


TRAINING_RULE = ("Read the following letter string and reproduce it exactly. "
                 "Give only the letters.")
ANSWER_RULE = ("Does this string follow the rules of the training strings? "
               "Respond YES or NO only.")
WAGER_RULE = ("Wager one or two sweets on the answer you just gave. Do not change your answer. "
              "Respond 1 or 2 only.")


def _training_string(step):
    prefix = TRAINING_RULE + "\n\n"
    if step["kind"] != "wag_training" or not step["prompt"].startswith(prefix):
        raise ValueError("Compact WAG expects the prepared training instruction unchanged")
    stimulus = step["prompt"][len(prefix):]
    if not re.fullmatch(r"[MTVRX]+", stimulus):
        raise ValueError("Compact WAG expects a literal published letter string")
    return stimulus


def _test_parts(step):
    match = re.fullmatch(r"(.*)Test string: ([MTVRX]+)\n" + re.escape(ANSWER_RULE),
                         step["prompt"], re.DOTALL)
    if step["kind"] != "wag_answer" or not match:
        raise ValueError("Compact WAG expects the prepared test instruction unchanged")
    return match.group(1), match.group(2)


def _expected_history(episode, step):
    ids = [s["id"] for s in episode["steps"]]
    position = ids.index(step["id"])
    parent = episode.get("parent")
    inherited = [f"{parent}_{i:02d}" for i in range(1, 46)] if parent else []
    return inherited + ids[:position]


def messages(episode, step, history):
    """Render one step, preserving complete parent + own-branch history only.

    Response text is inserted without strip, normalization or abbreviation.
    No gold labels, grammar labels, scores or metadata enter the prompt.
    """
    if episode["task"] != "post_decision_wagering":
        raise ValueError("This presentation candidate is only for WAG")
    if [r["step"]["id"] for r in history] != _expected_history(episode, step):
        raise ValueError("WAG history must contain every parent and own-branch step, in order")

    testing = episode["stage"] == "test"
    past_classification = any(row["step"]["kind"] == "wag_answer" for row in history)
    parts = []
    if history:
        parts.extend([
            "Past history (read-only record). Past instructions applied to their recorded phases; "
            "they do not set the response format for this turn. All earlier material remains visible.",
            "Past training rule: " + TRAINING_RULE,
        ])
        if testing and past_classification:
            instructions, _ = _test_parts(episode["steps"][0])
            parts.extend(["Previously shown test instructions:", instructions,
                          "Past classification rule: " + ANSWER_RULE])
        if any(row["step"]["kind"] == "wager" for row in history):
            parts.append("Past wager rule: " + WAGER_RULE)
        parts.append("Chronological record of presented strings and your exact past replies.")

    study_number = test_number = 0
    for row in history:
        past, response = row["step"], row["response"]
        kind = past["kind"]
        if kind == "wag_training":
            study_number += 1
            parts.append(f"Study {study_number}: {_training_string(past)}\nPast reproduction reply:\n{response}")
        elif kind == "wag_answer":
            _, stimulus = _test_parts(past)
            test_number += 1
            parts.append(f"Test {test_number}: {stimulus}\nPast classification reply:\n{response}")
        elif kind == "wager":
            if past["prompt"] != WAGER_RULE:
                raise ValueError("Compact WAG expects the prepared wager instruction unchanged")
            parts.append("Past wager reply:\n" + response)
        else:
            raise ValueError(f"Unsupported WAG step kind: {kind}")

    if history:
        parts.append("End of past history.")
    if step["kind"] == "wag_training":
        parts.append("Current phase: training reproduction. Reply only with the letters; do not include history labels.")
        parts.append(f"Current training string: {_training_string(step)}\n{TRAINING_RULE}")
    elif step["kind"] == "wag_answer":
        parts.append("Current phase: classification only. The wager is collected in the following turn. "
                     "Reply only YES or NO; do not include a wager or history labels.")
        if not past_classification:
            instructions, _ = _test_parts(episode["steps"][0])
            parts.append(instructions)
        _, stimulus = _test_parts(step)
        parts.append(f"Current test string: {stimulus}\n{ANSWER_RULE}")
    elif step["kind"] == "wager":
        if step["prompt"] != WAGER_RULE:
            raise ValueError("Compact WAG expects the prepared wager instruction unchanged")
        parts.append("Current phase: wager only. Reply only 1 or 2; do not include history labels.")
        parts.append("Current request: " + WAGER_RULE)
    else:
        raise ValueError(f"Unsupported current WAG step kind: {step['kind']}")

    result = [{"role": "system", "content": episode["system"]}] if episode.get("system") else []
    result.append({"role": "user", "content": "\n".join(parts)})
    return result
