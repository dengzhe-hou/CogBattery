"""Prompt construction for the P9 same-stimulus study.

n-back: the archived CogArena multi-turn format (scripts/run_eval.py run_multiturn_item):
one user message per trial = the overview line(s) plus the last 30 history lines plus
"Trial k: X\\nYour response:", with the archived system prompt.  The only change is one
sentence copied from the SRO task instructions about letter case (the human participants
saw upper- and lower-case letters and were told to ignore case).

Strachan false belief: the item text exactly as stored in the OSF workbook (outer
whitespace stripped), either in one conversation per list (session) or alone (isolated).
"""

SYSTEM_PROMPT = ("You are taking a cognitive evaluation. Follow the instructions precisely. "
                 "Give only the requested answer, nothing else.")

# expfactory adaptive_n_back/experiment.js (commits 82d85b6-7e87c82, 2016), instructions block.
SRO_CASE_SENTENCE = 'The letters will be both lower and upper case. You should ignore the case (so "t" matches "T").'

MAX_CONTEXT_LINES = 30  # run_eval.py --max-context-turns default


def nback_overview(n, length, first, case_sentence=SRO_CASE_SENTENCE):
    """NBackGenerator.generate stimulus_text, with the SRO case sentence before 'First token'."""
    lines = [f"N-Back Task (n={n}).",
             f"You will see a sequence of {length} tokens, presented one at a time.",
             f"For each token, respond MATCH if it is the same as the token presented {n} position(s) earlier, "
             f"otherwise respond NO MATCH."]
    if case_sentence:
        lines.append(case_sentence)
    lines.append(f"First token: {first}")
    return "\n".join(lines)


def nback_prompt(history_lines, i, stimulus, max_lines=MAX_CONTEXT_LINES, drop_pairs=0):
    """Prompt for trial index i (0-based), as in run_eval.run_multiturn_item.

    history_lines[0] is the overview; then 'Trial k: X' / 'Your response: R' pairs.
    drop_pairs > 0 removes that many of the oldest kept trial pairs (context guard only;
    every use is logged by the runner)."""
    if len(history_lines) > max_lines + 1:
        kept = [history_lines[0]] + history_lines[-max_lines:]
    else:
        kept = list(history_lines)
    if drop_pairs:
        body = kept[1:]
        if 2 * drop_pairs > len(body):
            raise ValueError("Cannot drop more history pairs than kept")
        kept = [kept[0]] + body[2 * drop_pairs:]
    return "\n".join(kept) + f"\nTrial {i + 1}: {stimulus}\nYour response:"


def history_pairs_in_prompt(history_lines, max_lines=MAX_CONTEXT_LINES):
    kept = len(history_lines) - 1
    return min(kept, max_lines) // 2


def nback_append(history_lines, i, stimulus, response):
    """run_eval appends the trial line and the stripped response; n-back turns have no feedback."""
    return history_lines + [f"Trial {i + 1}: {stimulus}", f"Your response: {response}"]
