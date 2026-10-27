"""Legal final-answer choices from task instructions, never answer keys.

This module does not choose an answer or change task stimuli. Free recall,
reproduction, false-belief free responses and Ready confirmations stay free.
The caller owns the wire schema and reasoning/final-channel handling.
"""

_FIXED_OPTIONS = {
    "n_back_choice": ("MATCH", "NO MATCH"),
    "ospan_math": ("YES", "NO"),
    "stroop_choice": ("RED", "GREEN", "BLUE"),
    "flanker_choice": ("LEFT", "RIGHT"),
    "go_nogo_choice": ("PRESS", "WITHHOLD"),
    "cvlt_recognition": ("YES", "NO"),
    "cvlt_forced": ("1", "2"),  # Supported optional CVLT forced-choice prompt.
    "drm_rating": tuple(str(value) for value in range(1, 8)),
    "drm_recognition": ("OLD", "NEW"),
    "tom_answer": ("A", "B"),
    "cal_answer": ("s", "l"),
    "confidence": tuple(f"{value}%" for value in range(0, 101, 10)),
    "wag_answer": ("YES", "NO"),
    "wager": ("1", "2"),
}

_UNCONSTRAINED = {
    "digit_recall", "ospan_recall", "cvlt_recall", "cvlt_recall_followup",
    "wag_training", "fb_answer", "ready", "instruction",
}

_DRM_OPTIONS = {
    "OLD": ("FAMILIAR", "REMEMBER", "GUESS"),
    "NEW": ("UNFAMILIAR", "RECALL_TO_REJECT", "WOULD_REMEMBER"),
    "UNCLASSIFIABLE": ("UNCLASSIFIABLE",),
}

_DRM_CURRENT_PROMPTS = {
    "You answered OLD. Classify that judgement: FAMILIAR, REMEMBER, or GUESS. "
    "Reply with one label only.": "OLD",
    "You answered NEW. Classify that judgement: UNFAMILIAR, RECALL_TO_REJECT, or WOULD_REMEMBER. "
    "Reply with one label only.": "NEW",
    "The preceding OLD/NEW response could not be classified. Reply only UNCLASSIFIABLE.": "UNCLASSIFIABLE",
}


def _drm_branch_from_history(step, history):
    recognition_id = step["meta"]["recognition_id"]
    previous = next((row for row in reversed(history)
                     if row["step"]["id"] == recognition_id), None)
    score = previous["score"] if previous is not None else {}
    if score.get("valid") and score.get("parsed") is True:
        return "OLD"
    if score.get("valid") and score.get("parsed") is False:
        return "NEW"
    return "UNCLASSIFIABLE"


def _drm_branch_from_prompt(current_prompt):
    # The compact renderer ends with this marker and the current request.
    # Earlier ledger rows or past follow-up instructions cannot select a branch.
    current = current_prompt.rsplit("Current request:\n", 1)[-1].strip()
    if current not in _DRM_CURRENT_PROMPTS:
        raise ValueError("DRM choice constraint requires the actual rendered current follow-up prompt")
    return _DRM_CURRENT_PROMPTS[current]


def choice_options(step, *, history=None, current_prompt=None):
    """Return allowed final strings, or None for an unconstrained response.

    All fields inspected are instruction/schema fields: kind, source-choice
    options and the DRM recognition_id. No gold, target, probe_class, correctness,
    source answer, or displayed-option value determines the correct choice.

    DRM report steps have no stored branch metadata in the prepared materials.
    Pass the model's own scored history (as used by memory._prompt_for), or the
    actual rendered current prompt for a fixed-history engineering probe. Both
    may be supplied to verify that the constraint matches the request sent.
    """
    kind = step["kind"]
    if kind in _FIXED_OPTIONS:
        return list(_FIXED_OPTIONS[kind])
    if kind in _UNCONSTRAINED:
        return None
    if kind == "source_choice":
        options = step["meta"]["options"]
        # The prompt explicitly permits a speaker's name OR an option number.
        # Preserve both representations, in displayed order.
        return [str(option) for option in options] + [str(i) for i in range(1, len(options) + 1)]
    if kind == "drm_report":
        if history is None and current_prompt is None:
            raise ValueError("DRM report choices require own-response history or the rendered current prompt")
        branch = _drm_branch_from_history(step, history) if history is not None else None
        if current_prompt is not None:
            prompt_branch = _drm_branch_from_prompt(current_prompt)
            if branch is not None and branch != prompt_branch:
                raise ValueError("DRM own-response history and rendered prompt select different branches")
            branch = prompt_branch
        return list(_DRM_OPTIONS[branch])
    raise ValueError(f"Unmapped task step kind for choice constraints: {kind}")
