<div align="center">

<div id="user-content-toc">
  <ul align="center" type="none">
    <summary>
      <h1>CogBattery: Human-Referenced Evaluation of Cognitive Task Behavior and Structure in Language Models</h1>
    </summary>
  </ul>
</div>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-2F5FB3?style=for-the-badge&logo=python&logoColor=2F5FB3&labelColor=F0F0F0"/></a> &emsp;
<a href="#the-battery"><img src="https://img.shields.io/badge/Paradigms-13-2F5FB3?style=for-the-badge&labelColor=F0F0F0"/></a> &emsp;
<a href="https://dengzhe-hou.github.io/CogBattery/#results"><img src="https://img.shields.io/badge/Models-54-2F5FB3?style=for-the-badge&labelColor=F0F0F0"/></a> &emsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2F5FB3?style=for-the-badge&labelColor=F0F0F0"/></a>

![CogBattery overview](docs/img/overview.webp)

<div id="toc">
  <ul align="center" type="none">
    <summary>
      <h2><a href="https://arxiv.org/abs/2607.24999">Paper</a> &emsp; <a href="https://dengzhe-hou.github.io/CogBattery/">Project page</a></h2>
    </summary>
  </ul>
</div>

</div>

# Overview

CogBattery adapts 13 cognitive-science paradigms to text and compares language models with the human data published for each paradigm. The current paper version evaluates 54 open-weight models from 23 families. The [project page](https://dengzhe-hou.github.io/CogBattery/) has the task guide and per-model results.

Matching the direction of a human effect does not mean matching the errors behind it. Like humans, models respond more to n-back lures than to controls, yet they make far more false alarms on ordinary non-targets. Most models' verbal confidence separates correct from incorrect answers less well than humans' does, while their token probabilities discriminate about as well as human reports. Most models always choose the same wager. Model scores share two dimensions, the first tied to scale, and named cognitive groups predict held-out model families no more reliably than random groups of equal size.

> [!NOTE]
> CogBattery was first released as *CogArena* (arXiv 2607.24999) and was renamed to avoid confusion with an unrelated benchmark of the same name. The Python package, the `cogarena` command, and recorded result paths keep the original name, so existing commands and citations still work.

### Features

- **13 paradigms** in five groups by origin:
  - **Working memory**: digit span, n-back, operation span.
  - **Cognitive control**: Stroop, Flanker, Go/no-go.
  - **Episodic memory**: CVLT, DRM, source memory.
  - **Social reasoning**: false belief, EPITOME recursive mindreading.
  - **Metacognition**: confidence calibration, post-decision wagering.
- **Source-specific human references**, mostly participant-level records, compared by condition and response type.
- **Paper-version runner** for the 54-model comparison, and the arXiv v1 command-line battery for OpenAI-compatible endpoints, hosted APIs, and Hugging Face models.
- **Gymnasium environments** for every paradigm, for custom interactive integrations.

# Quick Start

> [!IMPORTANT]
> The paper's 54-model results come from the paper-version runner (`scripts/experiments/human13_20261017/run.py`), which uses the published source materials and human references. Its setup is in [PAPER_VERSION.md](PAPER_VERSION.md). The `cogarena eval` command below runs the arXiv v1 battery with procedurally generated items, so its scores are not comparable with the paper's tables.

### Installation

```bash
git clone <repository-url> CogBattery
cd CogBattery
pip install -e .            # core (gymnasium)
pip install -e ".[openai]"  # + OpenAI / any OpenAI-compatible endpoint
# extras: ".[anthropic]", ".[google]", ".[huggingface]", ".[all]"
```

Requires Python ≥ 3.10.

### Run the battery

The default text evaluation runs all 13 paradigms. CogBattery handles the three
multi-turn paradigms within the same command.

```bash
# 0. Check all 13 paradigms without calling a model
cogarena eval --dry-run --model test --n 1

# 1. Any OpenAI-compatible endpoint, including Ollama, vLLM, TGI, and LM Studio
cogarena eval --provider local --base-url http://localhost:11434/v1 \
  --model qwen2.5:7b --n 50

# 2. Hosted APIs (set the matching key in your environment)
OPENAI_API_KEY=...    cogarena eval --provider openai    --model gpt-4o-mini
ANTHROPIC_API_KEY=... cogarena eval --provider anthropic --model claude-3-5-sonnet-20241022
GOOGLE_API_KEY=...     cogarena eval --provider google    --model gemini-1.5-pro

cogarena list   # show the paradigms
```

Useful flags include `--n` (items per paradigm, default 50), `--seed`,
`--paradigms stroop flanker ...`, `--temperature`, `--max-tokens`, and
`--output DIR`.

### Load a Hugging Face model directly

No serving layer is required for a Transformers causal language model.

```bash
pip install -e ".[huggingface]"
cogarena eval --provider huggingface \
  --model Qwen/Qwen2.5-7B-Instruct --device auto --dtype bfloat16
```

Use `--device cpu`, `--device cuda`, or `--device cuda:1` when needed. Models
that require custom repository code also need `--trust-remote-code`.

### Bring your own model

Four integration routes are available.

1. Use an **OpenAI-compatible server** for most local and hosted models with
   `--provider local --base-url <url>`.
2. Use a **native SDK** with `--provider {openai,anthropic,google}`.
3. Load a Transformers model directly with `--provider huggingface`.
4. For another interface, subclass the client and override one method.

   ```python
   from cogarena.llm_client import LLMClient
   class MyClient(LLMClient):
       def _dispatch(self, prompt, system_prompt=None, temperature=None,
                     max_tokens=None, images=None):
           return my_model.generate(prompt)   # return the response text
   ```
   Then drive the battery from Python (see `cogarena/cli.py` for a ~30-line example, or use
   `cogarena.scoring.score_static` + the generators in `cogarena.generators`).

# Output

Results are written to `cogarena_results/<model>/<mode>/`.

- `aggregate.json` contains overall accuracy, per-paradigm accuracy, and the five grouping means.
- `details.json` contains the prompt id, paradigm, model response, and full score record for each item.

The console prints a summary table (per-paradigm + grouping + overall).

# The battery

| Grouping | Text paradigms |
|---|---|
| Working Memory | digit span, n-back, operation span |
| Cognitive Control | Stroop, Flanker, Go/No-Go |
| Episodic Memory | DRM false memory, source monitoring, CVLT |
| Social reasoning | false belief, EPITOME |
| Metacognition | confidence calibration, post-decision wagering |

The CLI runs the complete text battery. The same paradigms are also registered
as Gymnasium environments for custom interactive integrations:

```python
import gymnasium as gym
import cogarena.gym_env            # registers the envs
env = gym.make("CogArena/NBack-v0", seed=42)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step("NO MATCH")
print(env.unwrapped.score())
```

# Reproduce arXiv v1

This section covers the arXiv v1 battery (procedurally generated items, 55 models). For the current paper version, see [PAPER_VERSION.md](PAPER_VERSION.md).

**What is in this repo.** The analysis code and the derived, de-identified result artifacts are
committed. These include the observational reanalyses under `results/reanalysis/`, corrected
matrices and inference outputs, the fully crossed intervention study under
`results/causal_selectivity_20260720/`, and the profile-stability and family-held-out diagnostics.
Their manifests bind the inputs, specifications, code, seeds, and outputs used for the reported
numbers, so the results can be inspected without rerunning model inference.
The committed target-versus-baseline sensitivity is inspectable and output-bound by its
manifest. Replaying it from its frozen inputs also requires the formal raw records and frozen
runtime manifest distributed with the separate raw-output bundle.

**What is not in this repo.** The *raw* per-item evaluation outputs (every model's
`details.json` / `aggregate.json`, several hundred MB across the `results/full_eval_*` and
`results/multiturn_*` directories) are not committed, because of their size. The analysis
scripts read from those directories, so re-running an analysis end-to-end needs the raw outputs.
There are two ways to obtain them.

1. **Regenerate** them by running the benchmark (`cogarena eval`, or the `scripts/run_*.sh`
   drivers) over the model set; items are seeded, so generation is deterministic.
2. Use the separately archived raw-output bundle. This large archive is not part
   of the current public release.

Scripts resolve the project root from their own location or from a `COGARENA_ROOT` environment
variable, so they run from any checkout. The reanalysis scripts need the `analysis` extra
(`pip install -e ".[analysis]"`: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`,
`statsmodels`, `cairosvg`, and `PyMuPDF`).
The paper figures are generated by the scripts in `paper/figures/`, including
`generate_all.py`, `fig_manifold.py`, and `generate_causal_selectivity.py`. They consume the
committed derived artifacts (and, where noted, the local raw-output bundle) and write the
tracked PDF figures next to the generators.

# Citation

```bibtex
@article{cogarena2026,
  title   = {CogArena: A Multimethod Evaluation of Cognitive Ability Structure in Large Language Models},
  journal = {arXiv preprint arXiv:2607.24999},
  year    = {2026}
}
```

# License

MIT. See [LICENSE](LICENSE).
