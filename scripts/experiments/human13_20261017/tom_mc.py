"""Offline materials and scoring for human-reference ToM/metacognition tasks.

No inference client, network access or scheduler is imported here.  Source-blocked
modules still expose their material episodes for inspection and synthetic tests.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

TASKS = ["false_belief", "epitome_tom", "confidence_calibration", "post_decision_wagering"]
FB_CONDITIONS = [("session", True, True), ("isolated", False, True),
                 ("session_no_system", True, False), ("isolated_no_system", False, False)]


def _step(identifier, prompt, kind, gold=None, meta=None, max_tokens=32):
    return {"id": identifier, "prompt": prompt, "kind": kind, "gold": gold,
            "meta": meta or {}, "max_tokens": max_tokens}


def _episode(identifier, task, condition, steps, system=None, history_mode="independent", **extra):
    return dict(id=identifier, task=task, condition=condition, system=system,
                history_mode=history_mode, steps=steps, **extra)


def _bundle(episodes, provenance, blockers=()):
    return {"status": "blocked" if blockers else "ready", "blockers": list(blockers),
            "provenance": provenance, "episodes": episodes}


def _fb(config):
    root = Path(config["root"])
    spec_path = Path(config.get("fb_spec_file", root / "scripts/experiments/same_stimulus_20261011/SPEC.json"))
    spec = json.loads(spec_path.read_text())
    material = Path(config.get("same_stimulus_dir", root / "results/same_stimulus_20261011/materials")) / "strachan_items.jsonl"
    items = [json.loads(line) for line in material.read_text().splitlines() if line.strip()]
    if len(items) != 18:
        raise ValueError("Strachan material must contain all 18 A/B items, including catches")
    episodes = []
    for condition, session, with_system in FB_CONDITIONS:
        for lst in ("A", "B"):
            eid = f"fb_{condition}_{lst}"
            steps = []
            for item in sorted((x for x in items if x["list"] == lst), key=lambda x: x["position"]):
                meta = {k: item[k] for k in ("item_id", "list", "story", "state", "main_analysis")}
                meta["scorer_path"] = str(root / "scripts/experiments/same_stimulus_20261011/scoring.py")
                steps.append(_step(f"{eid}__{item['item_id']}", item["text"], "fb_answer",
                                   item["correct_location"], meta, max_tokens=1024))
            episodes.append(_episode(eid, TASKS[0], condition, steps,
                system=spec["prompts"]["system_prompt"] if with_system else None,
                history_mode="full" if session else "independent",
                context_policy="legacy_strachan"))
    return _bundle(episodes, {
        "source": "https://osf.io/fwj6v/", "materials": str(material), "spec": str(spec_path),
        "calls_per_model": 72, "existing_models": [m["model"] for m in spec["model_panel"]],
        "reuse_provenance": str(root / "results/phi3_cache_fix_20261013/p9_analysis/input_provenance.json"),
        "comparison": "Same Strachan items; preserves P9 four conditions and r1a scorer",
        "warnell_gap": "Does not replace Warnell Belief-Desires or its same-person structural comparison",
    })


def _tom(config):
    raw = Path(config["raw_root"]) / "epitome2024_sn7gj/recursive_mindreading"
    with (raw / "stories.csv").open(newline="") as stream:
        stories = list(csv.DictReader(stream))
    with (raw / "questions.csv").open(newline="") as stream:
        questions = list(csv.DictReader(stream))
    human_path = raw / "tom_rm_trial.csv"
    with human_path.open(newline="") as stream:
        human_rows = list(csv.DictReader(stream))
    selected_human = [row for row in human_rows if row["response_type"] == "2AFC"]
    human_n = len({row["participant_id"] for row in selected_human})
    if len(selected_human) != 3315 or human_n != 63:
        raise ValueError("Selected public processed EPITOME 2AFC version must contain 63 participants / 3315 rows")
    rows = [(s, q) for s in stories for q in questions if s["story_id"] == q["story_id"]]
    if len(stories) != 8 or len(questions) != 112 or len(rows) != 224:
        raise ValueError("EPITOME must retain 8 story rows, 112 question rows and 224 crossed judgments")
    # Balance the displayed correct position within each format/type cell.  This
    # counteracts the source's 105 a / 7 b keys without changing either option.
    strata = defaultdict(list)
    for index, (story, question) in enumerate(rows):
        strata[(story["story_format"], question["question_format"], question["question_type"])].append(index)
    rng = random.Random(config.get("seed", 20261017))
    display_a = set()
    for key in sorted(strata):
        indices = strata[key][:]
        rng.shuffle(indices)
        display_a.update(indices[:len(indices) // 2])
    episodes = []
    for index, (story, question) in enumerate(rows):
        correct = question[f"option_{question['correct_option']}"]
        other = question["option_b" if question["correct_option"] == "a" else "option_a"]
        options = [correct, other] if index in display_a else [other, correct]
        gold = "A" if index in display_a else "B"
        eid = (f"rm_s{story['story_id']}_{story['story_format']}_{question['question_format']}_"
               f"{question['question_type']}_L{question['question_level']}")
        text = story["story_text"].replace("\\n", "\n")
        prompt = (f"Read the story.\n\n{text}\n\nWhich statement is consistent with the story?\n"
                  f"A. {options[0]}\nB. {options[1]}\nRespond with A or B only.")
        meta = {k: question[k] for k in ("story_id", "question_format", "question_type")}
        meta.update(story_format=story["story_format"], question_level=int(question["question_level"]),
                    original_correct_option=question["correct_option"], displayed_options=options,
                    response_type="2AFC", protocol="generated_choice_not_PMI")
        episodes.append(_episode(eid, TASKS[1], "generated_2AFC", [
            _step(eid + "__answer", prompt, "tom_answer", gold, meta)]))
    return _bundle(episodes, {
        "source": "https://osf.io/sn7gj/", "paper": "https://aclanthology.org/2024.tacl-1.45/",
        "materials": str(raw), "calls_per_model": 224,
        "human_version": "User-selected public processed 2AFC subset; not a reconstruction of the paper's exclusion chain",
        "human_reference": {
            "path": str(human_path), "source_url": "https://osf.io/download/56ae7/",
            "osf_file_id": "647b9e8ea8dbe9091ecb5b8f", "subset_rule": "response_type == '2AFC'",
            "participants": human_n, "rows": len(selected_human), "cached_total_rows": len(human_rows),
            "additional_filtering": "None; no reapplication of attention, item or RT exclusions",
            "paper_sample_reproduced": False,
        },
        "source_differences": [
            "TACL reports 72 recruited minus 6 excluded; selected cached 2AFC subset has 63 participants / 3315 rows",
            "TACL text says 300 ms but cached analysis script uses 3000 ms; original filtering chain remains unresolved",
            "Human story rereading remains unverified; model story text is visible",
            "Generated A/B choice is an adaptation, not official continuation-PMI scoring",
        ],
        "protocol": "Generated A/B adaptation, not official continuation PMI; retains all stimulus-format cells",
        "scope": "External recursive-mindreading module, not the original four-subtype ProjectToM",
        "warnell_gap": "Warnell original stories/keys remain unavailable; not a prerequisite for the selected EPITOME module",
    })


def _cal(config):
    from openpyxl import load_workbook
    raw = Path(config["raw_root"]) / "mazancieux2020_metacognition"
    book = load_workbook(raw / "Material_EM_SM_EF_tasks.xlsx", read_only=True, data_only=True)
    sheet = list(book["SM task"].values)
    book.close()
    if tuple(sheet[0]) != ("Cue", "Target", "Distractor") or len(sheet[1:]) != 40:
        raise ValueError("Mazancieux SM materials must contain the original 40 Cue/Target/Distractor rows")
    items = list(enumerate(sheet[1:], start=1))
    random.Random(config.get("seed", 20261017)).shuffle(items)
    episodes = []
    for position_version in (0, 1):
        for item_id, (cue, target, distractor) in items:
            target_left = (item_id + position_version) % 2 == 0
            left, right = (target, distractor) if target_left else (distractor, target)
            gold = "s" if target_left else "l"
            eid = f"cal_p{position_version}_q{item_id:02d}"
            meta = {"item_id": item_id, "position_version": position_version, "language": "French",
                    "left_option": left, "right_option": right, "target": target, "pair_id": eid}
            answer = _step(eid + "__answer",
                f"{cue}\n\nÀ gauche : {left}\nÀ droite : {right}\n"
                "Répondez uniquement s pour la réponse à gauche ou l pour la réponse à droite.",
                "cal_answer", gold, meta)
            confidence = _step(eid + "__confidence",
                "Quel est votre niveau de confiance dans la réponse que vous venez de donner ? "
                "Utilisez l’échelle de confiance à 11 niveaux : 0 %, 10 %, 20 %, 30 %, 40 %, "
                "50 %, 60 %, 70 %, 80 %, 90 %, 100 %. 0 % signifie aucune confiance et "
                "100 % une confiance totale. Ne modifiez pas votre réponse. "
                "Répondez uniquement avec le pourcentage.", "confidence", None,
                {"pair_id": eid, "answer_step_id": answer["id"], "scale": "Maz_11_levels",
                 "probability_mapping": "0.5 + percentage/200"})
            episodes.append(_episode(eid, TASKS[2], f"position_{position_version}",
                                     [answer, confidence], history_mode="full"))
    return _bundle(episodes, {
        "source": "https://osf.io/b5ype/", "materials": str(raw), "calls_per_model": 160,
        "human_linkage": "181 humans x 40 items = 7240 unique matched item-response rows",
        "scope": "French semantic-memory 2AFC; both option positions; not source monitoring",
        "adaptation": "Independent item episodes; answer then original 11-level confidence; no feedback",
        "calibration": "Official script maps k=0..10 to 0.50+0.05*k; not k/10 correctness probability",
    })


def read_wag_materials(pdf_path):
    """Recover the literal printed Table 6; preserve the documented B-list discrepancy."""
    import fitz
    training = {"A": [], "B": []}
    tests, labels = [], []
    with fitz.open(pdf_path) as document:
        for page in document:
            text = page.get_text()
            markers = ["Table 6 The training and test strings in order of presentation", "Table 6 (Contd.)"]
            marker = next((m for m in markers if m in text), None)
            if marker is None:
                continue
            text = text.split(marker, 1)[1]
            pairs = re.findall(r"(?m)^(\d+)\. ([MTVRX]+)$", text)
            labels.extend(re.findall(r"(?m)^([AB])$", text))
            while pairs:
                number = int(pairs[0][0])
                if number <= 45:
                    triple, pairs = pairs[:3], pairs[3:]
                    if len(triple) != 3 or [int(row[0]) for row in triple] != [number] * 3:
                        raise ValueError("Cannot recover the three columns of published Dienes Table 6")
                    training["A"].append(triple[0][1])
                    training["B"].append(triple[1][1])
                    tests.append(triple[2][1])
                else:
                    tests.append(pairs.pop(0)[1])
    if [len(training["A"]), len(training["B"]), len(tests), len(labels)] != [45, 45, 60, 60]:
        raise ValueError("Published Dienes Table 6 did not yield 45/45 training and 60 labeled test rows")
    return training, list(zip(tests, labels))


def _wag(config):
    raw = Path(config["raw_root"]) / "postdecision_wagering_text_candidates"
    training, tests = read_wag_materials(raw / "dienes_scott2005.pdf")
    episodes = []
    for grammar in ("A", "B"):
        parent_id = f"wag_train_{grammar}"
        training_steps = []
        for number, letters in enumerate(training[grammar], 1):
            prompt = ("Read the following letter string and reproduce it exactly. "
                      f"Give only the letters.\n\n{letters}")
            training_steps.append(_step(f"{parent_id}_{number:02d}", prompt, "wag_training",
                                        letters, {"grammar": grammar, "position": number}))
        episodes.append(_episode(parent_id, TASKS[3], f"training_{grammar}", training_steps,
                                 history_mode="full", stage="training"))
        for order in ("forward", "reverse"):
            eid = f"wag_test_{grammar}_{order}"
            indexed = list(enumerate(tests, 1))
            if order == "reverse":
                indexed.reverse()
            steps = []
            for position, (source_item, (letters, source_grammar)) in enumerate(indexed, 1):
                pair_id = f"{eid}_q{source_item:02d}"
                instructions = ""
                if position == 1:
                    instructions = (
                        "The letter order in the strings studied earlier followed a complex set of rules. "
                        "Exactly half the test strings follow those same rules. For each string, decide "
                        "whether it follows the rules, then wager one or two sweets on your answer. "
                        "You start with 10 sweets. A correct answer wins the amount wagered; an incorrect "
                        "answer loses that amount. No correctness or balance feedback is given until the "
                        "end. A negative final balance is not collected.\n\n")
                answer = _step(pair_id + "__answer", instructions +
                    f"Test string: {letters}\nDoes this string follow the rules of the training strings? "
                    "Respond YES or NO only.", "wag_answer",
                    "YES" if source_grammar == grammar else "NO",
                    {"pair_id": pair_id, "grammar": grammar, "test_order": order,
                     "source_item": source_item, "position": position, "source_grammar": source_grammar})
                wager = _step(pair_id + "__wager",
                    "Wager one or two sweets on the answer you just gave. Do not change your answer. "
                    "Respond 1 or 2 only.", "wager", None,
                    {"pair_id": pair_id, "answer_step_id": answer["id"], "stakes": [1, 2]})
                steps.extend([answer, wager])
            episodes.append(_episode(eid, TASKS[3], f"{grammar}_{order}", steps,
                history_mode="full", parent=parent_id, stage="test",
                history_visibility="Original training inputs and reproductions remain visible; not hidden-memory replication"))
    return _bundle(episodes, {
        "source": "https://doi.org/10.1016/j.concog.2009.09.009",
        "material_source": "https://doi.org/10.1007/s00426-004-0208-3",
        "materials": str(raw), "calls_per_model": 570, "human_reference": "40 PDW participants; published group summaries",
        "scope": "External AGL post-decision wagering; does not replace existing knowledge-question WAG",
        "material_version": "User-selected literal published Dienes & Scott 2005 Table 6, pp.349-350",
        "exact_original_program_reproduction": False,
        "training_unique_A": len(set(training["A"])), "training_unique_B": len(set(training["B"])),
        "training_test_overlap_A": len(set(training["A"]) & {letters for letters, _ in tests}),
        "training_test_overlap_B": len(set(training["B"]) & {letters for letters, _ in tests}),
        "source_differences": [
            "Published B training table has 16 unique strings although method states 15; literal table retained",
            "Printed Table 6 training-test overlaps retained despite method description of new test strings",
            "Training strings and reproductions remain in model context; human timed blank-screen recall is not reproduced",
            "Human sweets have real consequences; model wagering is instruction-based",
        ],
        "human_structure": "No individual trial table retrieved; cannot estimate cross-task human CAL-WAG structure",
    })


def build(config):
    """Construct all four complete material plans without executing any responses."""
    return dict(zip(TASKS, [_fb(config), _tom(config), _cal(config), _wag(config)]))


@lru_cache(maxsize=2)
def _legacy_scorer(path):
    spec = importlib.util.spec_from_file_location("human13_legacy_p9_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_score_step(step, response):
    """Score an answer without changing the raw response or inventing a missing value."""
    text = str(response).strip()
    kind = step["kind"]
    if kind == "fb_answer":
        meta = step["meta"]
        correct, path, chosen = _legacy_scorer(meta["scorer_path"]).score_fb(
            meta["story"], meta["list"], text, step["prompt"])
        return {"parsed": chosen if chosen != "none" else None, "valid": chosen != "none",
                "is_correct": bool(correct), "path": path, "legacy_score": correct}
    if kind == "confidence":
        match = re.fullmatch(r"(100|[0-9]|[1-9][0-9])\s*%?[.!]?", text)
        value = int(match.group(1)) if match else None
        valid = value is not None and value % 10 == 0
        return {"parsed": value if valid else None, "valid": valid, "is_correct": None,
                "level": value // 10 if valid else None,
                "calibration_probability": .5 + value / 200 if valid else None}
    if kind == "wager":
        match = re.fullmatch(r"([12])[.!]?", text)
        value = int(match.group(1)) if match else None
        return {"parsed": value, "valid": value is not None, "is_correct": None}
    patterns = {"tom_answer": r"([AB])[.!]?", "cal_answer": r"([sl])[.!]?",
                "wag_answer": r"(YES|NO)[.!]?", "wag_training": r"([MTVRX]+)[.!]?"}
    if kind not in patterns:
        raise ValueError(f"Unknown ToM/metacognition step kind: {kind}")
    match = re.fullmatch(patterns[kind], text, re.IGNORECASE)
    parsed = match.group(1) if match else None
    if parsed is not None:
        parsed = parsed.lower() if kind == "cal_answer" else parsed.upper()
    return {"parsed": parsed, "valid": parsed is not None,
            "is_correct": parsed == step["gold"] if parsed is not None else None}


def score_step(step, response):
    """Accept complete unambiguous answer wrappers without repairing content."""
    from .memory import _answer_text, _label_with_option
    original = _strict_score_step(step, response)
    kind = step["kind"]
    # The published legacy free-response scorer remains unchanged.
    if kind == "fb_answer":
        return {**original, "strict_original_score": original.copy()}
    text = _answer_text(response)
    if kind == "tom_answer":
        options = {chr(65 + i): option for i, option in
                   enumerate(step["meta"].get("displayed_options", []))}
        label = _label_with_option(text, options)
        if label is not None:
            text = label
    elif kind == "cal_answer":
        meta = step.get("meta", {})
        options = {label: str(meta[key]) for label, key in
                   (("s", "left_option"), ("l", "right_option")) if key in meta}
        label = _label_with_option(text, options)
        for candidate, option in options.items():
            if (text.rstrip(".!").casefold() == option.casefold()
                    or re.fullmatch(re.escape(candidate) + r"\s*\(" + re.escape(option) + r"\)[.!]?", text, re.I)):
                label = candidate
        if label is not None:
            text = label
    elif kind == "wager":
        match = re.fullmatch(r"(?:W|Wager):\s*([12])[.!]?", text, re.I)
        if match:
            text = match.group(1)
    elif kind == "wag_training":
        # Wrong letters are copying errors, not repaired to the shown stimulus.
        # R: and spaces/commas are the observed complete reproduction wrappers.
        match = re.fullmatch(r"(?:R:\s*)?([A-Z]+|[A-Z](?:[ ,]+[A-Z])+)[.!]?", text, re.I)
        if match:
            parsed = re.sub(r"[ ,]+", "", match.group(1)).upper()
            return {"parsed": parsed, "valid": True, "is_correct": parsed == step["gold"],
                    "strict_original_score": original}
    return {**_strict_score_step(step, text), "strict_original_score": original}


def messages(episode, step, history):
    """Serialize only prompts/responses, never answer keys or scoring metadata."""
    if episode.get('presentation'):
        if episode['task'] != 'post_decision_wagering' or episode['presentation'] != 'compact_history_v1':
            raise ValueError(f"Unsupported ToM/metacognition presentation: {episode['presentation']}")
        from .compact_wag import messages as compact_messages
        return compact_messages(episode, step, history)
    result = [{"role": "system", "content": episode["system"]}] if episode.get("system") else []
    if episode["history_mode"] == "full":
        for row in history:
            result.extend([{"role": "user", "content": row["step"]["prompt"]},
                           {"role": "assistant", "content": row["response"].strip()}])
    elif episode["history_mode"] != "independent":
        raise ValueError(f"Unknown history mode: {episode['history_mode']}")
    result.append({"role": "user", "content": step["prompt"]})
    return result


def load_reused_fb(config, spec=None):
    """Adapt completed P9 records and check the active runner's reuse mapping when supplied."""
    root = Path(config["root"])
    bundle = _fb(config)
    lookup = {(episode["condition"], step["meta"]["item_id"]): (episode, step)
              for episode in bundle["episodes"] for step in episode["steps"]}
    provenance = json.loads(Path(bundle["provenance"]["reuse_provenance"]).read_text())
    records, keys = [], set()
    for folder in provenance["sources"].values():
        source = Path(folder)
        # Existing provenance uses absolute project paths; preserve the relative
        # dataset mapping when the project is mounted under another root.
        if not source.is_relative_to(root):
            source = root / str(source).split("/CogArena/", 1)[1]
        for path in sorted((source / "calls").glob("fb_*.json")):
            old = json.loads(path.read_text())
            if old.get("synthetic") or old.get("resource_pilot"):
                raise ValueError(f"Reuse source contains non-formal FB record: {path}")
            key = (old["model"], old["condition"], old["item_id"])
            if key in keys:
                raise ValueError(f"Duplicate FB record in latest provenance: {key}")
            keys.add(key)
            episode, step = lookup[(old["condition"], old["item_id"])]
            score = score_step(step, old["response"])
            if int(score["is_correct"]) != old["score"]:
                raise ValueError(f"Frozen Strachan score changed during reuse: {path}")
            records.append({"model": old["model"], "family": old["family"],
                "family_merged": old["family_merged"], "episode_id": episode["id"], "task": TASKS[0],
                "condition": episode["condition"], "step": step, "response": old["response"],
                "score": score, "reused": True, "source_file": str(path),
                "source_context_items_dropped": old.get("earlier_items_dropped_for_context", 0)})
    counts = Counter(r["model"] for r in records)
    expected = set(bundle["provenance"]["existing_models"])
    if set(counts) != expected or any(counts[m] != 72 for m in expected):
        raise ValueError("Latest FB provenance must provide exactly 20 models x 72 complete records")
    if spec is not None:
        configured = {model: Path(folder).resolve() for model, folder in spec["reuse"]["false_belief"].items()}
        # r7: the three native-thinking models now send think=true requests, so
        # their old think=false FB records are not the same request (USER_DECISIONS.md:10).
        native = set(spec.get("inference", {}).get("native_think_models", []))
        active_expected = (expected & set(spec["models"])) - native
        if set(configured) != active_expected:
            raise ValueError("FB reuse model set differs from active SPEC")
        # Verify the entire historical twenty-model source above, then return only
        # the explicitly selected active panel (seventeen in r7: no TinyLlama, no native three).
        records = [record for record in records if record["model"] in configured]
        for record in records:
            source = Path(record["source_file"]).parent.parent.resolve()
            if source != configured[record["model"]]:
                raise ValueError(f"FB reuse source differs from active SPEC for {record['model']}: "
                                 f"{source} != {configured[record['model']]}")
    return records


def summarize(records):
    """Task-specific descriptive endpoints for a supplied record set (fixtures or existing data)."""
    groups = defaultdict(list)
    for record in records:
        if record["task"] in TASKS:
            groups[(record.get("model"), record["task"], record["condition"])].append(record)
    summaries = []
    for (model, task, condition), rows in sorted(groups.items(), key=lambda pair: str(pair[0])):
        summary = {"model": model, "task": task, "condition": condition, "n_records": len(rows)}
        answers = {r["step"]["id"]: r for r in rows
                   if r["step"]["kind"] in {"fb_answer", "tom_answer", "cal_answer", "wag_answer"}}
        summary["valid_answers"] = sum(r["score"]["valid"] for r in answers.values())
        summary["answer_count"] = len(answers)
        observed = [r["score"]["is_correct"] for r in answers.values() if r["score"]["is_correct"] is not None]
        summary["invalid_answers"] = len(answers) - summary["valid_answers"]
        summary["valid_answer_rate"] = summary["valid_answers"] / len(answers) if answers else None
        summary["scored_answers"] = len(observed)
        summary["correct_answers"] = sum(observed)
        summary["accuracy_all_attempted"] = sum(observed) / len(answers) if answers else None
        summary["accuracy_on_scored_answers"] = sum(observed) / len(observed) if observed else None
        if task == TASKS[1]:
            cells = defaultdict(list)
            for row in answers.values():
                meta = row["step"]["meta"]
                cells[(meta["story_format"], meta["question_format"], meta["question_type"],
                       meta["question_level"])].append(row)
            summary["by_format_type_level"] = []
            for key, cell in sorted(cells.items()):
                scored = [r["score"]["is_correct"] for r in cell if r["score"]["is_correct"] is not None]
                valid = sum(r["score"]["valid"] for r in cell)
                summary["by_format_type_level"].append({
                    "story_format": key[0], "question_format": key[1], "question_type": key[2], "level": key[3],
                    "n_presented": len(cell), "n_scored": len(scored),
                    "n_valid": valid, "n_invalid": len(cell) - valid, "valid_rate": valid / len(cell),
                    "n_correct": sum(scored), "accuracy_all_attempted": sum(scored) / len(cell),
                    "accuracy_on_scored_answers": sum(scored) / len(scored) if scored else None})
        if task == TASKS[0]:
            for state in ("F", "T"):
                selected = [r for r in answers.values() if r["step"]["meta"]["main_analysis"]
                            and r["step"]["meta"]["state"] == state]
                summary[state + "B_n"] = len(selected)
                summary[state + "B_accuracy"] = (sum(r["score"]["is_correct"] for r in selected) /
                                                   len(selected)) if selected else None
        paired = []
        for row in rows:
            if row["step"]["kind"] not in {"confidence", "wager"} or not row["score"]["valid"]:
                continue
            answer = answers.get(row["step"]["meta"]["answer_step_id"])
            if answer and answer["score"]["valid"] and answer["score"]["is_correct"] is not None:
                paired.append((answer["score"]["is_correct"], row["score"]))
        if task == TASKS[2]:
            summary["paired_answer_confidence_n"] = len(paired)
            summary["confidence_mean_percent"] = (sum(v["parsed"] for _, v in paired) / len(paired)) if paired else None
            summary["official_bias"] = (sum(v["calibration_probability"] - int(a) for a, v in paired) /
                                        len(paired)) if paired else None
            right = [value["parsed"] for correct, value in paired if correct]
            wrong = [value["parsed"] for correct, value in paired if not correct]
            summary["type2_auc_correct_n"] = len(right)
            summary["type2_auc_incorrect_n"] = len(wrong)
            summary["type2_auc_fixed_direction"] = (
                sum((a > b) + .5 * (a == b) for a in right for b in wrong) / (len(right) * len(wrong))
                if right and wrong else None)
        if task == TASKS[3]:
            summary["paired_answer_wager_n"] = len(paired)
            for stake in (1, 2):
                outcomes = [correct for correct, value in paired if value["parsed"] == stake]
                summary[f"wager_{stake}_n"] = len(outcomes)
                summary[f"wager_{stake}_accuracy"] = sum(outcomes) / len(outcomes) if outcomes else None
            high, low = summary["wager_2_accuracy"], summary["wager_1_accuracy"]
            summary["high_minus_low_accuracy"] = high - low if high is not None and low is not None else None
        summaries.append(summary)
    return summaries
