"""DRM chronological ledger, adopted for episodes marked compact_history_v1.

Only repeated instructions and past chat-role boundaries are merged. Study items,
block order, past probe order, and every actual response remain visible. This does
not simulate human forgetting: the full earlier material is still accessible.
"""
import json

from . import memory

score_step = memory.score_step


def _json_row(values):
    # JSON string escaping is reversible, including invalid/multiline responses.
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def history_ledger(history):
    """Return only already-presented material and actual answers, in task order.

    No gold, correctness, probe class, theme, or future episode step is read.
    A test row's third field exists only after its judgement report was given.
    """
    study, tests, report_instructions = [], [], []
    study_instructions = test_instructions = None
    for index, record in enumerate(history):
        step = record["step"]
        kind, response = step["kind"], record["response"]
        if kind == "drm_rating":
            if study_instructions is None:
                study_instructions = step["prompt"].split("\nStudy block 1.\n", 1)[0]
            study.append([step["meta"]["material"], response])
        elif kind == "drm_recognition":
            if test_instructions is None:
                test_instructions = step["prompt"].split("\nTest word:", 1)[0]
            tests.append([step["meta"]["word"], response])
        elif kind == "drm_report":
            assert index > 0 and history[index - 1]["step"]["id"] == step["meta"]["recognition_id"]
            assert len(tests[-1]) == 2
            tests[-1].append(response)
            prompt = memory._prompt_for(step, history[:index])
            if prompt not in report_instructions:
                report_instructions.append(prompt)
        else:
            raise ValueError(f"DRM compact candidate does not support {kind}")
    return {"study_instructions": study_instructions,
            "test_instructions": test_instructions,
            "report_instructions": report_instructions,
            "study": study, "tests": tests}


def messages(episode, step, history):
    """Message builder selected by memory.messages for marked episodes."""
    if episode["task"] != "drm_false_memory":
        raise ValueError("This candidate is only for DRM")
    ledger = history_ledger(history)
    parts = []
    if history:
        parts.append('Past history (read-only record). Past response instructions applied to their '
                     'recorded phases; they do not set the response format for this turn. '
                     'All earlier material remains visible.')
    if ledger["study"]:
        parts.extend(["Past phase: study meaning rating.", ledger["study_instructions"],
                      "Study history in presentation order. JSON rows: [item, your actual meaning-rating response]."])
        for index, row in enumerate(ledger["study"]):
            if index % 8 == 0:
                parts.append(f"Study block {index // 8 + 1}.")
            parts.append(_json_row(row))
    if ledger["tests"]:
        parts.extend(["Past phase: recognition and completed judgement follow-ups.",
                      ledger["test_instructions"], "Past recognition response format: Reply only OLD or NEW.",
                      "Test history in presentation order. JSON rows: [test word, your actual OLD/NEW response, your actual follow-up response if completed]."])
        parts.extend(_json_row(row) for row in ledger["tests"])
    if ledger["report_instructions"]:
        parts.append("Previously shown follow-up instructions (not the current request):")
        parts.extend(ledger["report_instructions"])
    if history:
        parts.append("End of past history.")
    phase = {
        "drm_rating": "Current phase: study meaning rating. Reply only one integer from 1 to 7.",
        "drm_recognition": (
            "Current phase: recognition, the first of two separate turns. "
            "The general test instructions to classify after OLD or NEW apply to the next turn. "
            "In this turn, reply only OLD or NEW; do not include a follow-up label."
        ),
        "drm_report": (
            "Current phase: judgement follow-up, the second of two separate turns. "
            "Reply only with the label requested below for your preceding response."
        ),
    }[step["kind"]]
    parts.extend([phase, "Do not include history rows or record headings in your reply."])
    parts.extend(["Current request:", memory._prompt_for(step, history)])
    return [{"role": "system", "content": episode["system"]},
            {"role": "user", "content": "\n".join(parts)}]
