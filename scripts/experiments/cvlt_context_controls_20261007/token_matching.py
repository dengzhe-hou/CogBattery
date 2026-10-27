"""Match real CVLT histories with the E20 within-field x placeholders.

Only local GGUF vocabulary loading and tokenization occur here. No inference
context, tensor loading, synthetic C0 counts, or model service is used.
build() returns (user_text, diagnostics, shared_replacement_units).
"""

import json
import os
from pathlib import Path
import re
import unicodedata

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

CACHE = Path("~/.ollama/models")
UNSPECIFIED_PRE = {"aya:8b", "deepseek-llm:7b", "stablelm2:1.6b"}
CONDITIONS = ("C0", "C1", "C2", "C3")


class Counter:
    """E19's exact local-blob vocab-only counter, without its fixture readers."""

    def __init__(self, model):
        from llama_cpp import llama_cpp as lib
        from llama_cpp import __version__

        family, tag = model.split(":", 1)
        self.model = model
        self.manifest = CACHE / "manifests/registry.ollama.ai/library" / family / tag
        manifest = json.loads(self.manifest.read_text())
        layers = [x for x in manifest["layers"]
                  if x["mediaType"] == "application/vnd.ollama.image.model"]
        if len(layers) != 1:
            raise RuntimeError(f"Expected one local GGUF model layer: {model}")
        self.blob = CACHE / "blobs" / layers[0]["digest"].replace(":", "-")
        if not self.blob.is_file():
            raise FileNotFoundError(self.blob)
        self.lib = lib
        lib.llama_backend_init()
        params = lib.llama_model_default_params()
        params.vocab_only = True
        params.n_gpu_layers = 0
        self.pointer = lib.llama_model_load_from_file(str(self.blob).encode(), params)
        if self.pointer is None:
            raise RuntimeError(f"Unavailable local vocabulary: {model}")
        self.vocab = lib.llama_model_get_vocab(self.pointer)
        if self.vocab is None:
            self.close()
            raise RuntimeError(f"No vocabulary in local GGUF: {model}")
        self._counts = {}
        self._checked_placeholders = set()
        self.metadata = {
            "model": model, "manifest": str(self.manifest), "GGUF_blob": str(self.blob),
            "llama_cpp_python_version": __version__, "vocab_only": True,
            "n_gpu_layers": 0, "inference_context_created": False,
            "pre_tokenizer_unspecified_in_GGUF": model in UNSPECIFIED_PRE,
            "scope": "raw UTF-8 content, add_special=False, parse_special=False; no chat template",
        }

    def count(self, text, *, special=False):
        key = (text, special)
        if key not in self._counts:
            raw = text.encode("utf-8")
            n = self.lib.llama_tokenize(self.vocab, raw, len(raw), None, 0, False, special)
            if n > 0:
                raise RuntimeError(f"Unexpected zero-capacity tokenize result: {n}")
            self._counts[key] = -n
        return self._counts[key]

    def check_placeholder(self, text, targets):
        if {w.lower() for w in re.findall(r"[A-Za-z]+", text)} & targets:
            raise RuntimeError("Placeholder contains an A/B target word")
        if text in self._checked_placeholders:
            return
        # Check replacements only. Source prompts intentionally contain targets.
        for special in (False, True):
            raw = text.encode("utf-8")
            n = self.count(text, special=special)
            array = (self.lib.llama_token * n)()
            got = self.lib.llama_tokenize(self.vocab, raw, len(raw), array, n, False, special)
            if got != n:
                raise RuntimeError("Inconsistent placeholder token count")
            forbidden = self.lib.LLAMA_TOKEN_ATTR_CONTROL | self.lib.LLAMA_TOKEN_ATTR_UNKNOWN
            if any(self.lib.llama_vocab_get_attr(self.vocab, token) & forbidden
                   for token in set(array)):
                raise RuntimeError("Placeholder produces a control/unknown token")
        self._checked_placeholders.add(text)

    def close(self):
        if self.pointer is not None:
            self.lib.llama_model_free(self.pointer)
            self.pointer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def pieces(item, trial_index, replies):
    """E20 field boundaries; only replies before this probe are used."""
    turns = item["parameters"]["turns"]
    if not 0 <= trial_index < len(turns) or len(replies) < trial_index:
        raise ValueError("Invalid probe index or incomplete C0 history")
    chunks = []

    def fixed(text):
        if chunks and chunks[-1]["kind"] == "fixed":
            chunks[-1]["text"] += text
        else:
            chunks.append({"kind": "fixed", "text": text})

    fixed(item["initial_input"])
    params = item["parameters"]
    for index in range(trial_index):
        response = replies[index]
        if not isinstance(response, str):
            raise TypeError("replies must contain the unmodified C0 response strings")
        turn = turns[index]
        stimulus = turn["stimulus"]
        fixed(f"\nTrial {index + 1}: ")
        if turn["type"] in ("learning_trial", "interference_trial"):
            kind = "A_list" if turn["type"] == "learning_trial" else "B_list"
            words = params["primary_list"] if kind == "A_list" else params["interference_list"]
            literal = ", ".join(words)
            if stimulus.count(literal) != 1:
                raise RuntimeError(f"Expected one structured list in {item['task_id']} trial {index + 1}")
            before, after = stimulus.split(literal)
            fixed(before)
            chunks.append({"kind": kind, "text": literal, "trial": index + 1,
                           "unit": f"{item['task_id']}#{index + 1}#{kind}"})
            fixed(after)
        else:
            fixed(stimulus)
        fixed("\nYour response: ")
        chunks.append({"kind": "response", "text": response, "trial": index + 1,
                       "unit": f"{item['task_id']}#{index + 1}#response"})
    fixed(f"\nTrial {trial_index + 1}: {turns[trial_index]['stimulus']}\nYour response:")
    return chunks


def masked(kind, condition):
    return (kind == "response" and condition != "C0") or (
        kind == "A_list" and condition in ("C2", "C3")
    ) or (kind == "B_list" and condition == "C3")


def _boundary_parts(text):
    """Retain actual leading whitespace and trailing whitespace/punctuation."""
    start = len(text) - len(text.lstrip())
    end = len(text)
    while end > start and (text[end - 1].isspace() or
                           unicodedata.category(text[end - 1]).startswith("P")):
        end -= 1
    return text[:start], text[start:end], text[end:]


class Matcher:
    def __init__(self, counter, item):
        self.counter = counter
        self.item = item
        params = item["parameters"]
        self.targets = {str(w).lower() for w in params["primary_list"] + params["interference_list"]}
        self.replacements = {}
        self.last_diagnostics = None

    def _fail(self, details):
        self.last_diagnostics = {"matched": False, "model": self.counter.model,
                                 "task_id": self.item["task_id"], **details}
        raise RuntimeError(json.dumps(self.last_diagnostics, ensure_ascii=False))

    def _fit(self, part, left, original_left, guard):
        count = self.counter.count
        unit, original = part["unit"], part["text"]
        target_end = count(original_left + original)
        target_guard = count(original_left + original + guard)
        if unit not in self.replacements:
            leading, core, suffix = _boundary_parts(original)
            attempts = []
            found = None
            candidates_seen = set()

            def check(candidate, variant, markers):
                nonlocal found
                if candidate in candidates_seen:
                    return False
                candidates_seen.add(candidate)
                end = count(left + candidate)
                after_guard = count(left + candidate + guard)
                attempts.append({"variant": variant, "markers": markers,
                                 "candidate": candidate, "end_delta": end - target_end,
                                 "guard_delta": after_guard - target_guard})
                if end == target_end and after_guard == target_guard:
                    self.counter.check_placeholder(candidate, self.targets)
                    found = candidate
                    return True
                return False

            # Empty output stays empty. Punctuation/whitespace has no lexical
            # response content to replace; preserving it also preserves its boundary.
            if not core:
                check(original, "empty_or_punctuation_only", 0)
            else:
                seed_n = max(1, count(original) - count(leading + suffix))
                # Finite, deterministic x-only alternatives inside this field.
                # No prompt-tail padding and no original-answer fallback.
                for tail in ("x", "xx", "xxx", "xxxx"):
                    for gap in ("", " ", "  "):
                        pending, seen_n = [seed_n], set()
                        for _ in range(12):
                            if not pending:
                                break
                            n = pending.pop(0)
                            if n in seen_n:
                                continue
                            seen_n.add(n)
                            candidate = leading + " ".join(["x"] * max(0, n - 1) + [tail]) + gap + suffix
                            if check(candidate, f"tail={tail!r},gap={gap!r}", n):
                                break
                            end_delta = target_end - count(left + candidate)
                            guard_delta = target_guard - count(left + candidate + guard)
                            for delta in (guard_delta, end_delta, -1, 1):
                                proposed = max(1, n + delta)
                                if proposed not in seen_n and proposed not in pending:
                                    pending.append(proposed)
                        if found is not None:
                            break
                    if found is not None:
                        break
            if found is None:
                self._fail({"reason": "No within-field x placeholder meets both prefix targets",
                            "failed_unit": unit, "source_text": original,
                            "target_end_tokens": target_end, "target_guard_tokens": target_guard,
                            "fit_attempts": attempts})
            self.replacements[unit] = {
                "unit": unit, "kind": part["kind"], "trial": part["trial"],
                "source_text": original, "replacement": found,
                "source_chars": len(original), "replacement_chars": len(found),
                "source_raw_field_tokens": count(original),
                "replacement_raw_field_tokens": count(found), "fit_attempts": attempts,
            }
        value = self.replacements[unit]
        if value["source_text"] != original:
            self._fail({"reason": "C0 history changed for a previously matched unit", "failed_unit": unit})
        candidate = value["replacement"]
        end_delta = count(left + candidate) - target_end
        guard_delta = count(left + candidate + guard) - target_guard
        if end_delta or guard_delta:
            self._fail({"reason": "Shared replacement does not match this prefix/guard",
                        "failed_unit": unit, "replacement": candidate,
                        "end_delta": end_delta, "guard_delta": guard_delta})
        return candidate, {"unit": unit, "end_delta": end_delta, "guard_delta": guard_delta}

    def build(self, trial_index, replies, condition):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        parts = pieces(self.item, trial_index, replies)
        baseline = "".join(part["text"] for part in parts)
        current = original = ""
        anchors, used = [], []
        count = self.counter.count
        for index, part in enumerate(parts):
            if part["kind"] == "fixed" or not masked(part["kind"], condition):
                current += part["text"]
            else:
                guard = parts[index + 1]["text"] if index + 1 < len(parts) and parts[index + 1]["kind"] == "fixed" else ""
                candidate, diagnostic = self._fit(part, current, original, guard)
                current += candidate
                anchors.append(diagnostic)
                used.append(part["unit"])
            original += part["text"]
            if part["kind"] != "fixed" and count(current) != count(original):
                self._fail({"reason": "Historical field-end prefix mismatch", "condition": condition,
                            "trial": trial_index + 1, "failed_unit": part["unit"],
                            "prefix_delta": count(current) - count(original)})
        turn = self.item["parameters"]["turns"][trial_index]
        current_suffix = f"\nTrial {trial_index + 1}: {turn['stimulus']}\nYour response:"
        if not current.startswith(self.item["initial_input"] + "\n") or not current.endswith(current_suffix):
            self._fail({"reason": "Initial/current stimulus changed"})
        delta = count(current) - count(baseline)
        prefix_delta = count(current[:-len(current_suffix)]) - count(baseline[:-len(current_suffix)])
        diagnostic = {
            "model": self.counter.model, "task_id": self.item["task_id"],
            "trial": trial_index + 1, "turn_type": turn["type"], "condition": condition,
            "matched": delta == prefix_delta == 0,
            "C0_user_content_tokens": count(baseline), "new_user_content_tokens": count(current),
            "full_user_content_tokens": count(current),
            "delta_tokens": delta, "current_prefix_count_delta": prefix_delta,
            "history_anchor_count_failures": 0, "history_anchors": anchors,
            "C0_chars": len(baseline), "new_chars": len(current),
            "masked_fields": len(used), "replacement_units": used,
            "pre_unspecified": self.counter.model in UNSPECIFIED_PRE,
            "synthetic_history": False,
            "prefix_note": "Re-tokenized prefix counts, not full-sequence token offsets",
        }
        if delta or prefix_delta:
            self._fail({"reason": "Full prompt/current-prefix token mismatch", **diagnostic})
        if condition == "C0" and current != baseline:
            self._fail({"reason": "C0 changed"})
        self.last_diagnostics = diagnostic
        return current, diagnostic, dict(self.replacements)
