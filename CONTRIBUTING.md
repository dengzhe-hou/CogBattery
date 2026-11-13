# Contributing

There are three ways to contribute. Each one is described below, from the lightest to the heaviest.

1. Report a problem with an issue, for example a material that does not build, a scoring rule that looks wrong, or a broken link.
2. Submit a model to the community table. [REPRODUCING.md](REPRODUCING.md) and [submissions/README.md](submissions/README.md) cover this. No code change is needed.
3. Add a new paradigm. The rest of this file describes how.

## Adding a paradigm

A paradigm belongs in CogBattery when a model can do a text version of it and a published study gives human data to compare with. Please open an issue before writing code. A short proposal saves work on both sides, because some tasks turn out to lack usable human data or cannot be adapted to text without changing what they measure.

### What the proposal should contain

- The paradigm and the source study, with a link to the paper.
- Where the human data are and under which license. CogBattery builds every material from the original files on the user's machine, so the data must be downloadable by anyone. Materials that cannot be redistributed, like the CVLT-II word lists, can still be used if a public source lets the build reconstruct them, as `cvlt_from_codebook.py` does.
- One key measure, defined so that it can be computed the same way for humans and models, for example an error rate difference between two conditions. Say whether a higher value is better or only different.
- The human reference for that measure: the mean, the across-person SD, and the number of participants, preferably computed from participant-level records.
- What changes in the text version, for example arrows written as characters, no response deadline, or a visible history. The existing entries in [`scripts/release/paper_materials/tasks/`](scripts/release/paper_materials/tasks/) show the expected level of detail.

### Where the code goes

Each paradigm touches the following places. The Flanker task is a compact example to copy, since its builder, scorer, and key measure are each a few dozen lines.

| Step | File | What to add |
|---|---|---|
| Sources | `scripts/release/paper_materials/sources.json` | Each file's URL, local path, license, size, and SHA-256, so that `fetch_sources` downloads and checks it into `data/raw/`. |
| Task module | `scripts/experiments/human13_20261017/` | A module with `TASKS`, `build(config)`, `messages(episode, step, history)`, `score_step(step, response)`, and `summarize(records)`. `wm_cc.py` holds Flanker. A new module is registered in `registry.py` and its task name is added to `TASKS` in `common.py`. |
| Answer format | `scripts/experiments/human13_20261017/constraints.py` | The answer schema for each new step `kind`, used for constrained decoding on OpenAI-compatible servers. |
| Output budget | `SPEC.json`, `inference.direct_step_budget_map` | The token budget for each new step kind. |
| Key measure | `scripts/reanalysis/human54_layers_20261019/model_scoring.py` and `scripts/reanalysis/human54_posthoc_20261020/gen_core_table.py` | The endpoint computed from scored records in `analyze_cell`, and one row in `ROWS` naming the key measure. |
| Description | `scripts/release/paper_materials/tasks/<task>.json` | The text shown in the task guide on the project page. |
| Tests | `tests/` | At least one test that builds the task from a small fixture and scores a correct and an incorrect reply. |

`build(config)` returns one bundle per task. A bundle has a `status` (`ready`, or `blocked` with a reason when a source file is missing), `provenance`, and a list of episodes. Each episode has an `id`, its `task`, a `condition`, a `system` prompt, a `history_mode`, and steps. Each step has an `id`, a `prompt`, a `kind`, the `gold` answer, `meta`, and `max_tokens`. `validate_task` in `common.py` checks this structure, and the build stops on any error. Never fill a missing source with made-up items. Return a blocked bundle instead.

### Checking your paradigm

1. Build the materials. The build compares every task with a recorded digest in `material_digests.json`. A new task has no recorded digest yet, so add the one the build prints and say in the pull request that it is new.
2. Run the stub server (REPRODUCING.md, step 4) to check that every step gets a reply and that the run finishes.
3. Run at least one real model, and include its key measure and the human reference in the pull request.
4. Run `pytest`.

### How a new paradigm reaches the leaderboard

The paper's leaderboard and the community table compare models on the same 13 measures, and a model's row is accepted only when all of them are defined. A new paradigm therefore does not change those tables directly. Merged paradigms are collected into the next battery version, for which the maintainers run the paper's model panel on the new tasks before adding a column. Until then, the new task can be run and reported with `report_model` like any other.
