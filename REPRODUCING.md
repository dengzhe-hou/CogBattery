# Running CogBattery on your model

This guide runs the paper version of CogBattery (protocol `formal_r7_unified_20261018`, 13 paradigms) on a model you serve yourself, and prepares a submission for the leaderboard. Everything runs on your machine. Nothing is sent to us until you submit.

## 1. Install

```bash
git clone https://github.com/dengzhe-hou/CogBattery.git
cd CogBattery
pip install -e ".[paper]"
```

Python 3.10 or later.

## 2. Download the sources

```bash
python -m scripts.release.fetch_sources
```

This downloads 16 files (about 19 MB) from OSF, GitHub, and UCLA into `data/raw/`. It checks each file against the SHA-256 recorded in [`scripts/release/paper_materials/sources.json`](scripts/release/paper_materials/sources.json). That file also lists each source and its terms. The materials are built on your machine and are not redistributed here. The one exception is the wagering task, whose 150 letter strings from Table 6 of Dienes and Scott (2005) ship with the repository.

If a download fails, the command prints the URL and the path to save the file under, so you can fetch it by hand.

CVLT uses the CVLT-II Standard word lists, which are Pearson copyright and are not in this repository. The build reads them from a public UCLA codebook that `fetch_sources` downloads, and places them with the word-free structure in [`cvlt_structure.json`](scripts/release/paper_materials/cvlt_structure.json). A licensed copy in `scripts/experiments/human13_20261017/cvlt_standard_source.json` is used instead when present.

## 3. Build the materials

```bash
python -m scripts.release.build_paper_materials
```

This writes `results/paper_materials/`. It compares every task with the digest of the material the paper used ([`material_digests.json`](scripts/release/paper_materials/material_digests.json)) and stops if one differs. A run on different materials would not be comparable with the paper.

## 4. Smoke test (no model, no cost)

```bash
python -m scripts.release.stub_server --port 11555 &
python -m scripts.release.run_model --materials results/paper_materials/manifest.json \
    --model stub:1b --backend ollama --base-url http://127.0.0.1:11555
```

The stub answers every step with a reply that fits its answer format. All 13 tasks finish in seconds. The scores are meaningless.

## 5. Run your model

With Ollama, which the paper used:

```bash
python -m scripts.release.run_model --materials results/paper_materials/manifest.json \
    --model qwen2.5:7b --backend ollama --base-url http://localhost:11434
```

With any OpenAI-compatible server (vLLM, llama.cpp server, TGI, or a hosted API):

```bash
export MY_KEY=...
python -m scripts.release.run_model --materials results/paper_materials/manifest.json \
    --model my-model --backend openai --base-url http://localhost:8000/v1 \
    --api-key-env MY_KEY --tokenizer org/my-model
```

> [!IMPORTANT]
> The paper served every model with Ollama 0.18.0 and a 4,096-token context. `--backend ollama` sends the paper's request body unchanged. `--backend openai` translates it to an OpenAI chat request with a JSON-schema response format, so the server's chat template and constrained decoding decide the result. Those scores are close to the paper's protocol but not identical, and the leaderboard marks them.

- Requests use temperature 0, one reply per step, the paper's output limits and answer schemas, and the same history rules. Your model is treated as a direct-answer model.
- `--tokenizer` gives exact token counts for the context check. Without it, about four characters per token is assumed.
- A full run is 6,789 requests for 13 tasks. If it stops, run the same command again. Finished steps are kept and the run resumes.
- Results go to `results/submissions/<model>/`.
- As a guide to time, Aya 8B averaged about 0.4 seconds per request on the paper's GPUs, or about 47 minutes for a full run. Larger models take longer. A run's records take about 150 MB.

## 6. See your results

```bash
python -m scripts.release.report_model --run results/submissions/qwen2.5__7b \
    --materials results/paper_materials/manifest.json
```

This needs no GPU and no submission. It prints the key measure of each paradigm from Table 4 of the paper, computed by the paper's scorers with invalid task answers counted as wrong, next to the human reference and the mean of the paper's 54 models. It also shows each difference in across-person human standard deviations and the share of valid answers. These differences show distance from humans, not quality. Add `--json report.json` to save the values.

## 7. Pack and submit

```bash
python -m scripts.release.pack_submission pack --run results/submissions/qwen2.5__7b \
    --out qwen2.5__7b.cogbattery.jsonl.gz
```

The packed file holds each step's reply, its validity fields, and the SHA-256 of the request that was sent. It is usually under a few MB. Open a pull request that adds it under `submissions/`, or attach it to an issue, and give the model's family and its served parameters in billions.

We score it with

```bash
python -m scripts.release.pack_submission score --pack qwen2.5__7b.cogbattery.jsonl.gz \
    --materials results/paper_materials/manifest.json --output results/scored
python -m scripts.release.report_model --run results/scored/qwen2.5__7b \
    --materials results/paper_materials/manifest.json \
    --add-to docs/data/leaderboard.json --family qwen --params-b 7
```

The first command replays the paper's run loop over your replies. Each request is rebuilt from the materials and your earlier replies, and must match the packed SHA-256. The second scores the replayed records with the paper's scorers and adds the row to the leaderboard, marked with your backend. It accepts only complete replayed runs with all 13 measures defined, and it does not replace the paper's rows. Scores computed elsewhere are not accepted as is.

## What was checked

- From an empty directory and a clean environment, `fetch_sources` and `build_paper_materials` reproduce all 13 task materials of the paper exactly, CVLT included, without any licensed file.
- Packing the paper's archived records for one model (aya:8b, 6,789 steps) and scoring them rebuilds every request with the same SHA-256 as the paper's and reproduces every step score. `report_model` on the replayed records gives that model's leaderboard row.
- Changing one reply in a packed file is rejected at the next request that shows it.

## Reproducing the paper's analyses

See [PAPER_VERSION.md](PAPER_VERSION.md) for the analysis code, derived tables, and supplementary results. The arXiv v1 battery (procedurally generated items, 55 models) is archived at tag [`v1.0.0-arxiv`](https://github.com/dengzhe-hou/CogBattery/tree/v1.0.0-arxiv).
