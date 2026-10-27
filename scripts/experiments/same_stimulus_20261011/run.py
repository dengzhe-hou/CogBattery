"""P9 same-stimulus runner: SRO n-back episodes and Strachan 2024 false-belief items.

The HTTP client is the E22 Client (scripts/experiments/cvlt_context_controls_20261007/run.py),
imported unchanged: patched backend executable, native /api/chat, think=false, context 4096,
server thread 1, temperature 0, max_tokens 1024, transport-only retries with every raw attempt
preserved, full-GPU / context / thread serving check.  No old output is read or overwritten.
tinyllama:1.1b, trained at 2048, is requested, checked and input-budgeted at 2048 (SPEC
inference.served_context_20261012, 2026-10-12); the other 19 models stay at 4096.

SPEC r2 (user decision 2026-10-12): every n-back record stores both parses (code = archived prefix
rule, code_first_line = primary first-line rule); Strachan items also run without any system
message (conditions session_no_system and isolated_no_system, formal mode only; pilot 25552 ran
only the two conditions with the system prompt).

SPEC r3 (user decision 2026-10-12 "四项都按你倾向", item 1): tinyllama:1.1b stopped in n-back at 2048
in job 25572; its n-back is reported as unable to complete.  Mode formal_r3_strachan runs only its
Strachan part (4 conditions x 18 items = 72 calls, same rules as r2) into
formal/job_r3_tinyllama_strachan/.  Under r3 check_authorized refuses every other run.

    python -m scripts.experiments.same_stimulus_20261011.run --mode pilot --model deepseek-r1:7b --output DIR
"""
import argparse
import copy
import json
import os
import time
from collections import Counter as Tally
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import protocol as P
from . import scoring as SC

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def served_context(spec, model):
    """Context the backend serves this model at: 4096 (E22), or the GGUF training length when it is
    smaller.  Ollama 0.18.0 cuts num_ctx to that length and `ollama ps` shows the cut value
    (SPEC inference.served_context_20261012: only tinyllama:1.1b, 2048)."""
    inf = spec["inference"]
    return inf["served_context_20261012"]["models"].get(model, inf["context_tokens"])


def model_spec(spec, model):
    """Spec view handed to the client: context_tokens = served_context, so the wire num_ctx and the
    unchanged E22 serving() check use what the backend serves.  Output dirs store the full SPEC."""
    view = copy.deepcopy(spec)
    view["inference"]["context_tokens"] = served_context(spec, model)
    return view


def e22_client(model, spec, output):
    from scripts.experiments.cvlt_context_controls_20261007.run import Client
    return Client(model, spec, output)


def gguf_counter(model):
    from scripts.experiments.cvlt_context_controls_20261007.token_matching import Counter
    return Counter(model)


def load_materials(spec, mode):
    m = spec["materials"]
    eps = [json.loads(l) for l in Path(m["nback"]["path"]).read_text().splitlines()]
    items = [json.loads(l) for l in Path(m["strachan"]["path"]).read_text().splitlines()]
    if len(eps) != m["nback"]["episodes"] or len(items) != 18:
        raise RuntimeError("Materials changed since the specification was frozen")
    for e in eps:
        if e["overview"] != P.nback_overview(e["load"], e["n_trials"], e["sequence"][0]):
            raise RuntimeError(f"Overview text differs from the frozen template: {e['episode_id']}")
        if [k["kind"] for k in SC.classify(e["sequence"], e["load"])] != e["trial_kinds"]:
            raise RuntimeError(f"Trial kinds differ: {e['episode_id']}")
    if mode == "pilot":
        eps = [e for e in eps if e["pilot"]]
        items = [i for i in items if i["pilot"]]
    return eps, items


# (condition id, presentation, system prompt sent).  The first two are the conditions of r0-r1a and
# of pilot 25552; the no-system pair was added at r2 (sensitivity) and runs in formal mode only.
STRACHAN_CONDITIONS = (("session", "session", True), ("isolated", "isolated", True),
                       ("session_no_system", "session", False), ("isolated_no_system", "isolated", False))

# SPEC r3: the one authorized tinyllama:1.1b run, Strachan only (SPEC authorization.formal_r3_authorization).
R3_MODE = "formal_r3_strachan"
MODES = ("pilot", "formal", R3_MODE)
CALLS_KEY = {"pilot": "pilot_per_model", "formal": "per_model", R3_MODE: "r3_tinyllama_strachan"}


def strachan_conditions(mode):
    return STRACHAN_CONDITIONS if mode in ("formal", R3_MODE) else STRACHAN_CONDITIONS[:2]


def runs_nback(mode):
    return mode != R3_MODE


def expected_calls(eps, items, mode):
    nb = sum(e["n_trials"] for e in eps) if runs_nback(mode) else 0
    return nb + len(strachan_conditions(mode)) * len(items)


def check_authorized(spec, mode, model, output):
    """Refuse any run the SPEC does not authorize (pilot: approved models under pilot/; formal: the
    approved revision under formal/; r3: only tinyllama's Strachan part in the approved directory).
    Under r3 every mode other than R3_MODE is refused."""
    auth = spec["authorization"]
    if model not in [m["model"] for m in spec["model_panel"]]:
        raise RuntimeError("Model not in the main-20 panel")
    r3 = auth.get("formal_r3_authorization")
    if r3 and spec["current_revision"] == r3["revision"] and mode != R3_MODE:
        raise RuntimeError(f"SPEC {r3['revision']} authorizes only --mode {R3_MODE} "
                           f"({', '.join(r3['models'])}, Strachan only)")
    if mode == R3_MODE:
        if not r3 or not r3["authorized_to_submit"]:
            raise RuntimeError("r3 Strachan run not authorized")
        if spec["current_revision"] != r3["revision"]:
            raise RuntimeError(f"r3 approval covers SPEC {r3['revision']}, not {spec['current_revision']}")
        if model not in r3["models"]:
            raise RuntimeError(f"r3 approval covers only {r3['models']}")
        if r3["parts"] != ["strachan"] or r3["mode"] != R3_MODE:
            raise RuntimeError("SPEC r3 scope differs from what this mode runs (Strachan only)")
        if output.resolve() != Path(r3["output_dir"]).resolve():
            raise RuntimeError(f"r3 output must be {r3['output_dir']}")
    elif mode == "pilot":
        if not auth["pilot_authorized_to_submit"]:
            raise RuntimeError("Pilot not authorized")
        if model not in spec["pilot"]["models"]:
            raise RuntimeError("Model not in the approved pilot")
        if "/pilot/" not in str(output.resolve()) + "/":
            raise RuntimeError("Pilot output must stay under results/same_stimulus_20261011/pilot/")
    elif mode == "formal":
        if not auth["formal_authorized_to_submit"]:
            raise RuntimeError("Formal run not authorized")
        if spec["current_revision"] != auth["formal_authorization"]["revision"]:
            raise RuntimeError(f"Formal approval covers SPEC {auth['formal_authorization']['revision']}, "
                               f"not {spec['current_revision']}; ask the user again")
        if "/formal/" not in str(output.resolve()) + "/":
            raise RuntimeError("Formal output must stay under results/same_stimulus_20261011/formal/")
    else:
        raise RuntimeError(f"Unknown mode {mode}")


class ContextBudgetError(RuntimeError):
    """The frozen input rule (SPEC inference.input_check) cannot fit a prompt; `failure` is written
    to context_budget_failure.json by the runner."""

    def __init__(self, message, failure):
        super().__init__(message)
        self.failure = failure


def fit_nback(hist, i, stim, n, system, raw_tokens, ctx, margin, episode_id=None):
    """n-back input rule: while raw + margin > ctx, drop the oldest kept history pair; stop if fewer
    than n pairs would remain.  Returns (messages, raw, pairs dropped)."""
    drop = 0
    while True:
        prompt = P.nback_prompt(hist, i, stim, drop_pairs=drop)
        messages = [system, {"role": "user", "content": prompt}]
        raw = raw_tokens(messages)
        if raw + margin <= ctx:
            return messages, raw, drop
        if P.history_pairs_in_prompt(hist) - (drop + 1) < n:
            raise ContextBudgetError("n-back history cannot fit while keeping n trials; no silent truncation",
                                     {"episode_id": episode_id, "trial": i + 1, "raw_content_tokens": raw,
                                      "messages": messages})
        drop += 1


def fit_session(head, pairs, user, raw_tokens, ctx, margin, per_msg, item_id=None):
    """Strachan session rule: while raw + margin + per_msg x messages > ctx, drop the oldest earlier
    item (question and reply) from `pairs` in place; stop if the current item alone does not fit.
    Returns (messages, raw, items dropped by this call)."""
    dropped = 0
    while True:
        messages = head + [m for pr in pairs for m in pr] + [user]
        raw = raw_tokens(messages)
        if raw + margin + per_msg * len(messages) <= ctx:
            return messages, raw, dropped
        if not pairs:
            raise ContextBudgetError("Strachan item cannot fit even alone; no silent truncation",
                                     {"item_id": item_id, "raw_content_tokens": raw, "messages": messages})
        pairs.pop(0)
        dropped += 1


def run_model(model, spec, mode, output, client_factory=e22_client, counter_factory=gguf_counter, synthetic=False):
    """All calls of one model.  A completed call with an identical request is reused."""
    inf = spec["inference"]
    ctx, margin, per_msg = served_context(spec, model), inf["template_margin_tokens"], inf["per_message_margin_tokens"]
    panel = {m["model"]: m for m in spec["model_panel"]}
    eps, items = load_materials(spec, mode)
    if not runs_nback(mode):
        eps = []  # r3: Strachan only
    expected = expected_calls(eps, items, mode)
    if expected != spec["calls"][CALLS_KEY[mode]]["total"]:
        raise RuntimeError(f"Call count {expected} differs from SPEC calls for mode {mode}")
    output.mkdir(parents=True, exist_ok=True)
    copy = output / "SPEC.json"
    if copy.exists() and json.loads(copy.read_text()) != spec:
        raise RuntimeError("Existing output belongs to a different specification")
    write(copy, spec)
    client = client_factory(model, model_spec(spec, model), output)
    start, started_at = time.perf_counter(), now()
    records, new_calls = [], 0
    system = {"role": "system", "content": spec["prompts"]["system_prompt"]}
    with counter_factory(model) as counter:
        meta = getattr(counter, "metadata", {"synthetic_counter": True})
        meta_path = output / "tokenizer.json"
        if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
            raise RuntimeError("Local tokenizer metadata changed; do not mix completed calls")
        write(meta_path, meta)

        def raw_tokens(messages):
            return sum(counter.count(m["content"]) for m in messages)

        def call(identity, messages, base):
            nonlocal new_calls
            path = output / "calls" / f"{identity}.json"
            payload = {"model": model, "messages": messages, "temperature": inf["temperature"],
                       "max_tokens": inf["max_tokens"], "reasoning_effort": inf["reasoning_effort"]}
            if path.exists():
                rec = json.loads(path.read_text())
                if rec["request"] != payload:
                    raise RuntimeError(f"Completed call {identity} no longer matches the same request")
                records.append(rec)
                return rec
            reply = client.call(payload, identity)
            new_calls += 1
            rec = {"model": model, "family": panel[model]["family"], "family_merged": panel[model]["family_merged"],
                   **base, "request": payload, "resource_pilot": mode == "pilot", "synthetic": synthetic,
                   "run_mode": mode, "spec_revision": spec["current_revision"], "captured_at": now(), **reply}
            rec["input_tokens"] = reply["usage"]["prompt_tokens"]
            rec["output_tokens"] = reply["usage"]["completion_tokens"]
            rec["elapsed_seconds"] = reply["latency_seconds"]
            if base["paradigm"] == "n_back_sro":
                rec["code"] = SC.nb_code(reply["response"])  # archived prefix rule (sensitivity)
                rec["code_first_line"] = SC.nb_code_first_line(reply["response"], base["trial"])  # primary
            else:
                s, path_, chosen = SC.score_fb(base["story"], base["list"], reply["response"], base["item_text"])
                rec["score"], rec["score_path"], rec["score_chosen"] = s, path_, chosen
            write(path, rec)
            records.append(rec)
            return rec

        # ---- n-back: one fresh episode per sampled SRO block
        for e in eps:
            n, seq = e["load"], e["sequence"]
            hist = [e["overview"]]
            for i, stim in enumerate(seq):
                try:
                    messages, raw, drop = fit_nback(hist, i, stim, n, system, raw_tokens, ctx, margin, e["episode_id"])
                except ContextBudgetError as err:
                    write(output / "context_budget_failure.json", err.failure)
                    raise
                base ={"paradigm": "n_back_sro", "episode_id": e["episode_id"], "load": n, "trial": i + 1,
                        "stimulus": stim, "trial_kind": e["trial_kinds"][i],
                        "history_pairs_in_prompt": P.history_pairs_in_prompt(hist) - drop,
                        "history_pairs_dropped_for_context": drop, "offline_raw_content_tokens": raw}
                rec = call(f"{e['episode_id']}__t{i + 1:02d}", messages, base)
                hist = P.nback_append(hist, i, stim, rec["response"])
            write(output / "progress.json", {"phase": "n_back", "episode": e["episode_id"],
                                             "completed_calls": len(records), "updated_at": now()})
        print(f"{model} n_back " + ("complete" if runs_nback(mode) else "not run (r3: Strachan only)")
              + f"; calls={len(records)}", flush=True)

        # ---- Strachan, per condition: session (one conversation per list), then isolated items
        for cond, presentation, with_system in strachan_conditions(mode):
            head = [system] if with_system else []

            def fb_base(it, in_context, dropped, raw):
                return {"paradigm": "strachan_fb", "condition": cond, "presentation": presentation,
                        "system_prompt": with_system, "item_id": it["item_id"], "list": it["list"],
                        "position": it["position"], "story": it["story"], "state": it["state"],
                        "main_analysis": it["main_analysis"], "item_text": it["text"],
                        "earlier_items_in_context": in_context, "earlier_items_dropped_for_context": dropped,
                        "offline_raw_content_tokens": raw}
            if presentation == "session":
                for lst in ("A", "B"):
                    seq_items = sorted((i for i in items if i["list"] == lst), key=lambda x: x["position"])
                    pairs, dropped = [], 0
                    for it in seq_items:
                        user = {"role": "user", "content": it["text"]}
                        try:
                            messages, raw, now_dropped = fit_session(head, pairs, user, raw_tokens, ctx, margin,
                                                                     per_msg, it["item_id"])
                        except ContextBudgetError as err:
                            write(output / "context_budget_failure.json", err.failure)
                            raise
                        dropped += now_dropped
                        rec =call(f"fb_{cond}_{lst}__{it['item_id']}", messages, fb_base(it, len(pairs), dropped, raw))
                        # archived call_llm strips outer whitespace before history use
                        pairs.append([user, {"role": "assistant", "content": rec["response"]}])
            else:
                for it in sorted(items, key=lambda x: (x["list"], x["position"])):
                    messages = head + [{"role": "user", "content": it["text"]}]
                    raw = raw_tokens(messages)
                    if raw + margin + per_msg * len(messages) > ctx:
                        raise RuntimeError("Strachan item does not fit the served context")
                    call(f"fb_{cond}__{it['item_id']}", messages, fb_base(it, 0, 0, raw))
        write(output / "progress.json", {"phase": "strachan", "completed_calls": len(records), "updated_at": now()})
        print(f"{model} strachan complete; calls={len(records)}", flush=True)
    if len(records) != expected:
        raise RuntimeError(f"Incomplete authorized call count {len(records)} != {expected}")
    summary = summarize_run(model, spec, mode, records, expected, client, new_calls, started_at,
                            time.perf_counter() - start, synthetic)
    write(output / "summary.json", summary)
    return summary


def summarize_run(model, spec, mode, records, expected, client, new_calls, started_at, wall, synthetic):
    """Resources and format only.  No hit, false-alarm, d' or FB/TB score is summarized here."""
    ctx = served_context(spec, model)
    nb = [r for r in records if r["paradigm"] == "n_back_sro"]
    fb = [r for r in records if r["paradigm"] == "strachan_fb"]

    def cell(rs):
        return {"n": len(rs), "max_input_tokens": max((r["input_tokens"] for r in rs), default=None),
                "max_output_tokens": max((r["output_tokens"] for r in rs), default=None),
                "mean_output_tokens": (sum(r["output_tokens"] for r in rs) / len(rs)) if rs else None,
                "sum_elapsed_seconds": sum(r["elapsed_seconds"] for r in rs),
                "max_elapsed_seconds": max((r["elapsed_seconds"] for r in rs), default=None),
                "length_finish": sum(r["finish_reason"] == "length" for r in rs),
                "blank": sum(not r["response"] for r in rs)}
    load_ns = [r["raw_response"].get("load_duration", 0) for r in records if isinstance(r.get("raw_response"), dict)]
    overhead = [r["input_tokens"] - r["offline_raw_content_tokens"] for r in records]
    status = ("offline_synthetic_check_complete" if synthetic else
              "resource_pilot_complete" if mode == "pilot" else
              "formal_r3_strachan_complete" if mode == R3_MODE else "formal_model_complete")
    return {
        "status": status, "model": model, "mode": mode, "synthetic_only": synthetic,
        "spec_revision": spec["current_revision"], "nback_run": runs_nback(mode),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": started_at, "completed_at": now(),
        "wall_seconds": wall, "logical_calls": len(records), "expected_logical_calls": expected,
        "new_calls_this_invocation": new_calls,
        "http_attempts_this_invocation": getattr(client, "actual_calls", None),
        "sum_elapsed_seconds_all_calls": sum(r["elapsed_seconds"] for r in records),
        "load_duration_seconds_sum": sum(load_ns) / 1e9,
        "max_input_tokens": max(r["input_tokens"] for r in records),
        "served_context_tokens": ctx,
        "all_inputs_within_context": all(r["input_tokens"] <= ctx for r in records),
        "template_overhead_tokens_min_max": [min(overhead), max(overhead)],
        "finish_reasons": dict(Tally(r["finish_reason"] for r in records)),
        "cells": {"n_back_load2": cell([r for r in nb if r["load"] == 2]),
                  "n_back_load3": cell([r for r in nb if r["load"] == 3]),
                  **{f"strachan_{c}": cell([r for r in fb if r["condition"] == c])
                     for c, _, _ in strachan_conditions(mode)}},
        "nback_unparseable": sum(r["code"] is None for r in nb),
        "nback_unparseable_first_line": sum(r["code_first_line"] is None for r in nb), "nback_calls": len(nb),
        "nback_calls_with_history_dropped": sum(r["history_pairs_dropped_for_context"] > 0 for r in nb),
        "strachan_session_calls_with_items_dropped": sum(r["earlier_items_dropped_for_context"] > 0
                                                         for r in fb if r["presentation"] == "session"),
        "strachan_score_paths": dict(Tally(r["score_path"] for r in fb)),
        "pilot_effects_not_reported": mode == "pilot",
        "scores_not_summarized_here": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Inference only inside a Slurm GPU allocation")
    spec = json.loads((HERE / "SPEC.json").read_text())
    check_authorized(spec, args.mode, args.model, args.output)
    result = run_model(args.model, spec, args.mode, args.output / args.model.replace(":", "__"))
    print(json.dumps({k: result[k] for k in ("status", "model", "logical_calls", "wall_seconds",
                                             "max_input_tokens")}), flush=True)


if __name__ == "__main__":
    main()
