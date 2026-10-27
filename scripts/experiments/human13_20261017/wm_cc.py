"""CPU-only material preparation and scoring for six human-reference tasks.

No client, scheduler, model loader, or network operation is used by this module.
The compact n-back condition is new; archived P9 responses are not relabeled.
"""
from __future__ import annotations

import ast
import csv
import gzip
import json
import operator
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from cogarena.dimensions.working_memory import DigitSpanGenerator, OperationSpanGenerator
from .memory import _answer_text
from scripts.experiments.same_stimulus_20261011.scoring import (
    classify, endpoints as p9_endpoints, loglinear_dprime, nb_code, nb_code_first_line,
)

TASKS = ["digit_span", "n_back", "operation_span", "stroop", "flanker", "go_nogo"]
SYSTEM = "Follow the task instructions. Give only the requested answer, without explanation."
LETTERS = list("FHJKLNPQRSTY")
DIGIT_WORDS = dict(zip("zero one two three four five six seven eight nine".split(), "0123456789"))
DIGIT_RE = re.compile(r"\b(" + "|".join(DIGIT_WORDS) + r")\b", re.I)
COLORS = {"stim1": "ORANGE", "stim2": "BLUE"}
SRO_FILES = {
    "digit_span": "digit_span", "stroop": "stroop",
    "flanker": "attention_network_task", "go_nogo": "go_nogo",
}


def _ready(episodes, provenance):
    return {"status": "ready", "blockers": [], "provenance": provenance, "episodes": episodes}


def _blocked(reason):
    return {"status": "blocked", "blockers": [str(reason)], "provenance": {}, "episodes": []}


def _episode(eid, task, condition, mode):
    return {"id": eid, "task": task, "condition": condition, "system": SYSTEM,
            "history_mode": mode, "steps": []}


def _step(ep, prompt, kind, gold, meta, max_tokens=16):
    step = {"id": f"{ep['id']}__{len(ep['steps']) + 1:04d}", "prompt": prompt,
            "kind": kind, "gold": gold, "meta": {"task": ep["task"], **meta},
            "max_tokens": max_tokens}
    ep["steps"].append(step)
    return step


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _color(row):
    match = re.search(r"id\s*=\s*(stim[12])\b", row["stimulus"])
    if not match:
        raise ValueError(f"Unknown SRO square HTML: {row['stimulus']}")
    return COLORS[match.group(1)]


def _sro_sources(config):
    root, raw = Path(config["root"]), Path(config["raw_root"])
    id_file = root / "results/reanalysis/sro_structure_20261015/analysis/summary.json"
    ids = sorted(json.loads(id_file.read_text())["fixed_accuracy_55"]["human"]["ids"])
    base = raw / "self_regulation_ontology/Complete_02-16-2019/Individual_Measures"
    data = {task: _rows(base / f"{filename}.csv.gz") for task, filename in SRO_FILES.items()}
    go_color = defaultdict(set)
    for row in data["go_nogo"]:
        if row["worker_id"] in ids and row["condition"] == "go":
            go_color[row["worker_id"]].add(_color(row))
    if any(len(go_color[pid]) != 1 for pid in ids):
        raise ValueError("Every eligible SRO participant must have one stable GO-color mapping")
    rng = random.Random(config["seed"])
    selected = [rng.choice([pid for pid in ids if color in go_color[pid]])
                for color in ("BLUE", "ORANGE")]
    selected.append(rng.choice([pid for pid in ids if pid not in selected]))
    chosen = {}
    for task, rows in data.items():
        chosen[task] = {
            pid: sorted((r for r in rows if r["worker_id"] == pid),
                        key=lambda r: float(r["time_elapsed"])) for pid in selected
        }
    provenance = {
        "seed": config["seed"], "participant_ids": selected, "eligible_id_source": str(id_file),
        "selection": "one random ID per GO-color mapping, then one remaining ID; no outcome selection",
        "eligible_n": len(ids), "sources": {t: str(base / f"{f}.csv.gz") for t, f in SRO_FILES.items()},
        "source_row_policy": "Original CSV rows including human responses are retained in step metadata, never in prompts",
        "coverage_limit": "Three complete trajectories do not reproduce the population distribution of materials",
    }
    return chosen, provenance


def _build_ds(rows_by_id, provenance):
    episodes = []
    for pid, rows in rows_by_id.items():
        if len(rows) != 28 or Counter(r["condition"] for r in rows) != {"forward": 14, "reverse": 14}:
            raise ValueError(f"Incomplete digit-span trajectory: {pid}")
        ep = _episode(f"ds_{pid}", "digit_span", "original_serial_digits", "unit_context")
        for index, row in enumerate(rows):
            digits = json.loads(row["sequence"])
            if len(digits) != int(float(row["num_digits"])):
                raise ValueError(f"Digit count mismatch: {row['']}")
            direction = "forward" if row["condition"] == "forward" else "backward"
            meta = {"unit": f"{pid}_{index:02d}", "stage": "test", "worker_id": pid,
                    "direction": direction, "length": len(digits), "source_row": row}
            for j, digit in enumerate(digits):
                _step(ep, f"{direction.capitalize()} digit recall. Memorize the digits in presentation order.\n"
                      f"Digit {j + 1}: {digit}\nReply Ready only.", "ready", "Ready",
                      {**meta, "digit_position": j + 1}, 8)
            gold = digits if direction == "forward" else digits[::-1]
            _step(ep, f"Recall this sequence in {'presentation' if direction == 'forward' else 'reverse'} order. "
                  "Output only the digits separated by spaces.", "digit_recall", gold,
                  {**meta, "feedback": "binary"}, 64)
        episodes.append(ep)
    return _ready(episodes, {**provenance, "practice": "No additional practice in the original DS program",
                            "scoring": "existing DigitSpanGenerator, with declared standalone digit-word normalization"})


def _build_cc(task, rows_by_id, provenance):
    episodes = []
    expected = {"stroop": {"practice": 24, "test": 96},
                "flanker": {"practice": 24, "test": 144},
                "go_nogo": {"practice": 10, "test": 350}}
    for pid, rows in rows_by_id.items():
        if Counter(r["exp_stage"] for r in rows) != expected[task]:
            raise ValueError(f"Incomplete {task} trajectory: {pid}")
        ep = _episode(f"{task}_{pid}", task, "original_text_adaptation",
                      "compact_last15" if task == "go_nogo" else "current_trial_with_practice_feedback")
        go_color = next((_color(r) for r in rows if r.get("condition") == "go"), None)
        for i, row in enumerate(rows):
            meta = {"stage": row["exp_stage"], "worker_id": pid, "trial": i + 1,
                    "source_row": row, "feedback": "binary" if row["exp_stage"] == "practice" else None}
            if task == "stroop":
                gold = row["stim_color"].upper()
                if gold not in ("RED", "GREEN", "BLUE"):
                    raise ValueError(f"Unknown Stroop color: {gold}")
                prompt = ("Report the ink-color label, ignoring the word's meaning.\n"
                          f"Word: {row['stim_word'].upper()}\nInk-color label: {row['stim_color']}\n"
                          "Reply RED, GREEN, or BLUE only.")
                meta.update(condition=row["condition"], stimulus_word=row["stim_word"])
            elif task == "flanker":
                images = re.findall(r"images/([^/ >]+)\.png", row["stimulus"])
                symbols = {"left_arrow": "<", "right_arrow": ">", "no_arrow": "-"}
                if len(images) != 5 or any(x not in symbols for x in images):
                    raise ValueError(f"Unrecognized ANT array: {row['']}")
                array = " ".join(symbols[x] for x in images)
                gold = row["flanker_middle_direction"].upper()
                if symbols[images[2]] != ("<" if gold == "LEFT" else ">"):
                    raise ValueError(f"ANT target mismatch: {row['']}")
                cue = {"nocue": "none", "center": "center", "double": "both upper and lower positions",
                       "spatial": row["flanker_location"]}[row["cue"]]
                prompt = ("Judge the middle arrow's direction.\n"
                          f"Earlier cue: {cue}\nTarget position: {row['flanker_location']}\n"
                          f"Array: {array}\nReply LEFT or RIGHT only.")
                meta.update(condition=row["flanker_type"], cue=row["cue"],
                            location=row["flanker_location"], target_direction=gold)
            else:
                color = _color(row)
                gold = "PRESS" if row["condition"] == "go" else "WITHHOLD"
                if (color == go_color) != (gold == "PRESS"):
                    raise ValueError(f"GO mapping changed within {pid}")
                prompt = (f"For {go_color}, choose PRESS. For the other color, choose WITHHOLD.\n"
                          f"Current square color: {color}\nReply PRESS or WITHHOLD only.")
                meta.update(condition=row["condition"], color=color, go_color=go_color)
            _step(ep, prompt, f"{task}_choice", gold, meta, 8)
        episodes.append(ep)
    return _ready(episodes, {**provenance, "practice_and_test_per_person": expected[task],
                            "adaptation": {"stroop": "color labels replace visual ink color",
                                           "flanker": "ASCII arrows and textual cue/location replace visual ANT",
                                           "go_nogo": "explicit choice replaces motor non-response; last 15 parsed actions"}[task]})


def _build_nb(config):
    base = Path(config["root"]) / "results/same_stimulus_20261011/materials"
    source = base / "nback_episodes.jsonl"
    originals = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    human_path = base / "human_nback_trials.jsonl"
    humans = {x["episode_id"]: x for x in map(json.loads, human_path.read_text().splitlines())}
    if len(originals) != 80 or sum(len(x["sequence"]) for x in originals) != 1800:
        raise ValueError("Frozen P9 n-back materials no longer match 80 blocks / 1800 trials")
    episodes = []
    for original in originals:
        eid, seq, load = original["episode_id"], original["sequence"], original["load"]
        human = humans[eid]
        if len(human["codes"]) != len(seq):
            raise ValueError(f"Human n-back linkage length mismatch: {eid}")
        trial_types = classify(seq, load)
        if [t["kind"] for t in trial_types] != original["trial_kinds"]:
            raise ValueError(f"P9 trial classifications changed: {eid}")
        ep = _episode(f"nb_compact_{eid}", "n_back", "compact_full_prefix", "compact_all_actions")
        ep["source_episode"] = original
        for i, letter in enumerate(seq):
            prefix = " ".join(f"{j + 1}:{s}" for j, s in enumerate(seq[:i + 1]))
            prompt = (f"N-back task, n={load}. Ignore letter case.\n"
                      "Reply MATCH when the current letter equals the letter n positions earlier; "
                      "otherwise reply NO MATCH. If fewer than n earlier letters exist, reply NO MATCH.\n"
                      f"All letters presented so far, in order: {prefix}\n"
                      f"Current trial: {i + 1}; current letter: {letter}\nReply MATCH or NO MATCH only.")
            meta = {"stage": "test", "load": load, "trial": i + 1, "letter": letter,
                    "prefix": seq[:i + 1], "condition": trial_types[i]["kind"],
                    "lm1": trial_types[i]["lm1"], "lp1": trial_types[i]["lp1"],
                    "source": original["source"],
                    "human_response": human["codes"][i], "human_rt": human["rt"][i],
                    "human_correct": human["correct"][i]}
            _step(ep, prompt, "n_back_choice", "MATCH" if trial_types[i]["kind"] == "target" else "NO MATCH", meta)
        episodes.append(ep)
    return _ready(episodes, {"sources": [str(source), str(human_path)], "blocks": 80,
                            "trials_per_model": 1800, "new_condition": True,
                            "archived_p9_models": 19, "archived_p9_data_reused_as_new": False,
                            "practice": 0, "history": "all presented letters and all prior parsed actions; original replies archived"})


def _arithmetic_value(expr):
    """Check the actual arithmetic supplied in the public material file."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError(f"Unsupported arithmetic in public OSPAN pool: {expr}")
    return visit(ast.parse(expr.replace("×", "*"), mode="eval").body)


def _build_os(config):
    source = Path(config["raw_root"]) / "py_span_task_english_20261017/operations.txt"
    pool = []
    for i, line in enumerate(source.read_text().splitlines()):
        if not line.strip():
            continue
        equation, label = line.split("\t")
        left, right = equation.split("=")
        if label not in ("y", "n") or (_arithmetic_value(left.strip()) == int(right)) != (label == "y"):
            raise ValueError(f"Public OSPAN arithmetic key mismatch on line {i + 1}")
        pool.append({"equation": equation, "answer": "YES" if label == "y" else "NO", "source_line": i + 1})
    if len(pool) != 200:
        raise ValueError("Public OSPAN pool must contain the inspected 200 equations")
    episodes = []
    for version in range(3):
        rng = random.Random(config["seed"] + version)
        ep = _episode(f"ospan_v{version + 1}", "operation_span", "standard_reconstruction", "unit_context")
        ep["version"] = version + 1
        math_by_key = {key: [p for p in pool if p["answer"] == key] for key in ("YES", "NO")}
        formal_math = rng.sample(math_by_key["YES"], 38 if version % 2 == 0 else 37)
        formal_math += rng.sample(math_by_key["NO"], 37 if version % 2 == 0 else 38)
        rng.shuffle(formal_math)
        chosen_lines = {m["source_line"] for m in formal_math}
        practice_math = rng.sample([m for m in pool if m["source_line"] not in chosen_lines], 21)

        def add_math(item, meta):
            _step(ep, f"Is this equation correct? {item['equation']}\nReply YES or NO only.",
                  "ospan_math", item["answer"], {**meta, "source_math": item,
                   "feedback": "binary" if meta["stage"] == "practice" else None}, 8)

        def add_group(unit, size, stage, maths=None):
            letters = rng.sample(LETTERS, size)
            meta = {"unit": unit, "stage": stage, "version": version + 1,
                    "length": size, "phase": "storage_practice" if maths is None else "combined"}
            for j, letter in enumerate(letters):
                if maths is not None:
                    add_math(maths[j], meta)
                _step(ep, f"Memorize this letter: {letter}\nReply Ready only.", "ready", "Ready",
                      {**meta, "letter": letter, "position": j + 1}, 8)
            _step(ep, "Recall this group's letters in order, separated by spaces. "
                  "Use _ for a forgotten position. Output only the sequence.", "ospan_recall", letters,
                  {**meta, "feedback": "span"}, 64)

        for i in range(4):
            add_group(f"storage_{i}", 2, "practice")
        for i, item in enumerate(practice_math[:15]):
            add_math(item, {"unit": f"arithmetic_{i}", "stage": "practice", "phase": "math_practice",
                            "version": version + 1})
        for i in range(3):
            add_group(f"combined_practice_{i}", 2, "practice", practice_math[15 + i * 2:17 + i * 2])
        sizes = [3, 4, 5, 6, 7] * 3
        rng.shuffle(sizes)
        cursor = 0
        for i, size in enumerate(sizes):
            add_group(f"formal_{i}", size, "test", formal_math[cursor:cursor + size])
            cursor += size
        if len(ep["steps"]) != 207:
            raise ValueError("OSPAN complete version must contain 165 formal and 42 practice outputs")
        episodes.append(ep)
    return _ready(episodes, {"math_source": str(source), "math_license": "GPL v2 or later",
                            "math_pool_n": len(pool), "letter_pool": LETTERS,
                            "versions": 3, "calls_per_version": 207, "seed": config["seed"],
                            "practice": {"storage": "4 sets of 2 (approved text adaptation, not recovered Ball setting)",
                                         "math": 15, "combined": "3 sets of 2"},
                            "time_calibration": "No human-equivalent deadline is inferred from model runtime",
                            "comparison": "Ball task-specification reconstruction, not original participant-item replay"})


def build(config):
    """Build all six task bundles; missing material is explicit, never fabricated."""
    out = {}
    try:
        rows, provenance = _sro_sources(config)
    except FileNotFoundError as exc:
        for task in SRO_FILES:
            out[task] = _blocked(f"Missing source needed for common SRO trajectory selection: {exc.filename}")
    else:
        out["digit_span"] = _build_ds(rows["digit_span"], provenance)
        for task in ("stroop", "flanker", "go_nogo"):
            out[task] = _build_cc(task, rows[task], provenance)
    for task, fn in (("n_back", _build_nb), ("operation_span", _build_os)):
        try:
            out[task] = fn(config)
        except FileNotFoundError as exc:
            out[task] = _blocked(f"Missing required original material: {exc.filename}")
    return {task: out[task] for task in TASKS}


def _choice(response, options):
    text = str(response).strip().upper().strip('.!"\'`* ')
    return text if text in options else None


def _strict_score_step(step, response):
    kind, gold = step["kind"], step["gold"]
    raw = str(response)
    extra = {}
    if kind == "ready":
        parsed = "Ready" if raw.strip().strip(".!").lower() == "ready" else None
        return {"parsed": parsed, "valid": parsed is not None, "is_correct": None,
                "history_response": parsed if parsed is not None else "INVALID", "protocol_only": True}
    if kind == "digit_recall":
        normalized = DIGIT_RE.sub(lambda m: DIGIT_WORDS[m.group(1).lower()], raw)
        digits = re.findall(r"\d", normalized)
        parsed = [int(x) for x in digits] if digits else None
        task = SimpleNamespace(expected_response=" ".join(map(str, gold)),
                               metadata=SimpleNamespace(parameters={"digit_length": len(gold)}))
        main = DigitSpanGenerator.score(task, normalized)
        old = DigitSpanGenerator.score(task, raw)
        return {"parsed": parsed, "valid": parsed is not None, "is_correct": bool(main["accuracy"]),
                "partial_credit": main["partial_credit"], "strict_original_is_correct": bool(old["accuracy"]),
                "history_response": " ".join(digits) if parsed is not None else "INVALID"}
    if kind == "ospan_recall":
        text = raw.strip().upper().strip(".!")
        # A lone letter or explicitly separated sequence, including skipped positions.
        if re.fullmatch(r"[A-Z_](?:[ ,]+[A-Z_])*", text):
            parsed = re.split(r"[ ,]+", text)
        else:
            parsed = None
        task = SimpleNamespace(metadata=SimpleNamespace(parameters={"turns": [], "set_size": len(gold), "letters": gold}))
        scored = OperationSpanGenerator.score(task, [" ".join(parsed or [])])
        return {"parsed": parsed, "valid": parsed is not None,
                "is_correct": bool(scored["recall_exact_match"]),
                "partial_credit": scored["recall_partial_credit"],
                "recall_correct_positions": int(scored["recall_correct_positions"]), "positions": len(gold),
                "history_response": " ".join(parsed) if parsed is not None else "INVALID"}
    if kind == "n_back_choice":
        code = nb_code_first_line(raw, step["meta"]["trial"])
        parsed = None if code is None else ("MATCH" if code else "NO MATCH")
        extra["archived_prefix_code"] = nb_code(raw)
    else:
        choices = {"ospan_math": ("YES", "NO"), "stroop_choice": ("RED", "GREEN", "BLUE"),
                   "flanker_choice": ("LEFT", "RIGHT"), "go_nogo_choice": ("PRESS", "WITHHOLD")}
        parsed = _choice(raw, choices[kind])
    return {"parsed": parsed, "valid": parsed is not None, "is_correct": parsed == gold,
            "history_response": parsed if parsed is not None else "INVALID", **extra}


def score_step(step, response):
    """Normalize a complete final answer without mining digits from reasoning."""
    original = _strict_score_step(step, response)
    text = _answer_text(response)
    if (step["kind"] == "n_back_choice"
            and re.match(r"^Reply:", str(response).strip(), re.I)
            and _choice(text, ("MATCH", "NO MATCH")) is None):
        # Do not extend the legacy first-line rule to newly stripped explanatory
        # or multiple-answer suffixes. The new wrapper must contain one label.
        text = str(response)
    if step["kind"] == "digit_recall":
        # Retain the already-supported fixed answer prefix, but require the
        # entire remainder to be one digit sequence, not an explanation.
        text = re.sub(r"^The answer is\s+", "", text, flags=re.I)
        if re.fullmatch(r"<[0-9]>(?:\s+<[0-9]>)*[.!]?", text):
            text = " ".join(re.findall(r"<([0-9])>", text))
        elif re.fullmatch(r"Digit\s+[1-9][0-9]*:\s*[0-9](?:\s+Digit\s+[1-9][0-9]*:\s*[0-9])*[.!]?",
                          text, re.I):
            labelled = re.findall(r"Digit\s+([1-9][0-9]*):\s*([0-9])", text, re.I)
            # Position labels must already be exactly 1..n. Never reorder them.
            if [int(label) for label, _ in labelled] == list(range(1, len(labelled) + 1)):
                text = " ".join(value for _, value in labelled)
        normalized = DIGIT_RE.sub(lambda match: DIGIT_WORDS[match.group(1).lower()], text)
        if not re.fullmatch(r"[0-9]+(?:[ ,]+[0-9]+)*[.!]?", normalized):
            normalized = ""
        scored = _strict_score_step(step, normalized)
        scored["strict_original_is_correct"] = original["strict_original_is_correct"]
    else:
        scored = _strict_score_step(step, text)
    if "archived_prefix_code" in original:
        scored["archived_prefix_code"] = original["archived_prefix_code"]
    return {**scored, "strict_original_score": original}


def _feedback(history):
    if not history:
        return ""
    previous = history[-1]
    mode = previous["step"]["meta"].get("feedback")
    if not mode:
        return ""
    score = previous["score"]
    if mode == "span":
        return f"Previous recall: {score['recall_correct_positions']} of {score['positions']} positions correct.\n"
    return "Previous response: " + ("Correct.\n" if score["is_correct"] else "Incorrect.\n")


def messages(episode, step, history):
    """Material/gold metadata never enter the model messages implicitly."""
    mode = episode["history_mode"]
    output = [{"role": "system", "content": episode["system"]}]
    feedback = _feedback(history)
    if mode == "unit_context":
        unit = step["meta"]["unit"]
        for index, past in enumerate(history):
            if past["step"]["meta"].get("unit") == unit:
                output.extend([{"role": "user", "content": _feedback(history[:index]) + past["step"]["prompt"]},
                               {"role": "assistant", "content": past["response"]}])
    elif mode in ("compact_all_actions", "compact_last15"):
        kept = history if mode == "compact_all_actions" else history[-15:]
        lines = []
        for past in kept:
            meta = past["step"]["meta"]
            stimulus = meta["letter"] if episode["task"] == "n_back" else meta["color"]
            lines.append(f"{meta['trial']}:{stimulus}:{past['score']['history_response']}")
        if lines:
            feedback += "Previous stimuli and your actions (not feedback):\n" + "\n".join(lines) + "\n"
    output.append({"role": "user", "content": feedback + step["prompt"]})
    return output


def _fraction(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None}


def _signal_summary(records, signal, positive):
    targets = [r for r in records if r["step"]["meta"]["condition"] == signal]
    others = [r for r in records if r["step"]["meta"]["condition"] != signal]
    result = {}
    for name, group in (("hit", targets), ("false_alarm", others)):
        valid = [r for r in group if r["score"]["valid"]]
        k = sum(r["score"]["parsed"] == positive for r in valid)
        result[name] = {"valid_response": _fraction(k, len(valid)), "all_trials": _fraction(k, len(group))}
    nt, nn = result["hit"]["valid_response"], result["false_alarm"]["valid_response"]
    result["dprime_valid_response"] = loglinear_dprime(nt["k"], nt["n"], nn["k"], nn["n"]) if nt["n"] and nn["n"] else None
    correct_rejections = sum(r["score"]["valid"] and r["score"]["is_correct"] for r in others)
    result["correct_rejection_all"] = _fraction(correct_rejections, len(others))
    result["balanced_accuracy_all"] = ((result["hit"]["all_trials"]["rate"] + correct_rejections / len(others)) / 2
                                        if targets and others else None)
    return result


def summarize(records):
    """Descriptive native endpoints, retaining denominators and protocol failures."""
    output = {}
    for task in TASKS:
        all_rows = [r for r in records if r["task"] == task]
        if not all_rows:
            continue
        rows = [r for r in all_rows if r["step"]["meta"]["stage"] == "test" and r["step"]["kind"] != "ready"]
        item = {"responses": len(all_rows), "formal_scored_responses": len(rows),
                "formal_valid": sum(r["score"]["valid"] for r in rows),
                "accuracy_all": _fraction(sum(r["score"]["is_correct"] is True for r in rows), len(rows)),
                "ready_invalid": sum(r["step"]["kind"] == "ready" and not r["score"]["valid"] for r in all_rows)}
        groups = defaultdict(list)
        for row in rows:
            m = row["step"]["meta"]
            key = (f"{m['direction']}_length_{m['length']}" if task == "digit_span" else
                   f"load_{m['load']}" if task == "n_back" else
                   row["step"]["kind"] if task == "operation_span" else m["condition"])
            groups[key].append(row)
        item["conditions"] = {key: {"accuracy_all": _fraction(sum(r["score"]["is_correct"] is True for r in group), len(group)),
                                     "valid": sum(r["score"]["valid"] for r in group)} for key, group in groups.items()}
        if task == "digit_span":
            item["strict_original_correct"] = sum(r["score"]["strict_original_is_correct"] for r in rows)
        elif task == "operation_span":
            recall = [r for r in rows if r["step"]["kind"] == "ospan_recall"]
            item["position_accuracy"] = _fraction(sum(r["score"]["recall_correct_positions"] for r in recall),
                                                  sum(r["score"]["positions"] for r in recall))
            item["note"] = "Use position_accuracy for Ball correspondence; do not mix arithmetic and recall accuracy as one ability score"
        elif task == "n_back":
            item["signal_by_load"] = {}
            item["p9_endpoint_definitions_by_load"] = {}
            for load, group in groups.items():
                endpoint = _signal_summary(group, "target", "MATCH")
                for kind in ("lure", "control"):
                    subset = [r for r in group if r["step"]["meta"]["condition"] == kind]
                    valid = [r for r in subset if r["score"]["valid"]]
                    matches = sum(r["score"]["parsed"] == "MATCH" for r in valid)
                    endpoint[kind + "_false_alarm"] = {"valid_response": _fraction(matches, len(valid)),
                                                         "all_trials": _fraction(matches, len(subset))}
                item["signal_by_load"][load] = endpoint
                blocks = defaultdict(list)
                for r in group:
                    parsed = r["score"]["parsed"]
                    blocks[r["episode_id"]].append({"kind": r["step"]["meta"]["condition"],
                                                     "code": None if parsed is None else parsed == "MATCH"})
                item["p9_endpoint_definitions_by_load"][load] = p9_endpoints(list(blocks.values()))
            item["note"] = ("signal_by_load dprime is pooled; p9_endpoint_definitions_by_load retains the original "
                            "per-block dprime mean (at least 5 valid target and 5 valid nontarget trials). "
                            "New compact responses remain a separate protocol from archived P9 responses.")
        elif task == "go_nogo":
            item["signal"] = _signal_summary(rows, "go", "PRESS")
        elif task in ("stroop", "flanker"):
            c = item["conditions"].get("congruent", {}).get("accuracy_all", {}).get("rate")
            i = item["conditions"].get("incongruent", {}).get("accuracy_all", {}).get("rate")
            item["error_interference_incongruent_minus_congruent"] = c - i if c is not None and i is not None else None
            if task == "stroop":
                incompatible = [r for r in rows if r["step"]["meta"]["condition"] == "incongruent"]
                item["word_response_on_incongruent"] = _fraction(
                    sum(r["score"]["parsed"] == r["step"]["meta"]["stimulus_word"].upper() for r in incompatible),
                    len(incompatible))
            else:
                cells = defaultdict(list)
                for r in rows:
                    m = r["step"]["meta"]
                    cells[f"{m['cue']}|{m['location']}|{m['condition']}"].append(r)
                item["cue_location_condition_accuracy"] = {
                    key: _fraction(sum(r["score"]["is_correct"] is True for r in group), len(group))
                    for key, group in cells.items()}
            item["note"] = "Error-rate interference only; model generation time is not a human reaction-time endpoint"
        output[task] = item
    return output
