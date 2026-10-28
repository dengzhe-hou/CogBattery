# Paper version: 54 models and source-specific human references

This page describes the code for the current paper version of CogBattery. That version evaluates 54 open-weight models on text adaptations of 13 paradigms and compares each one with the human data published for it. The protocol is `formal_r7_unified_20261018`. The rest of this repository is the earlier arXiv v1 battery, which uses procedurally generated items and covers 55 models.

The project site at https://dengzhe-hou.github.io/CogBattery/ shows the task guide and the per-model results.

## What is included

| Directory | Role |
|---|---|
| `scripts/experiments/human13_20261017/` | Task builders, request protocol, Ollama client, response cache, scoring, and the formal runner (`run.py`) |
| `scripts/experiments/human13_repair_20261017/`, `human13_format_repair_20261017/` | Earlier material stages that the r7 materials are derived from (`prepare_r7.py`) |
| `scripts/reanalysis/human54_layers_20261019/` | Endpoint scoring, human references, and the behavioral comparisons with family intervals |
| `scripts/reanalysis/human54_fullbattery_20261019/`, `human54_factors_20261019/` | Score structure: PCA, parallel analysis, factor analysis, and cross-family prediction |
| `scripts/reanalysis/human54_posthoc_20261020/` | Post hoc analyses: minimal-performance gates, scale strata, reliability, human standard-error sensitivity, and the no-WAG structure check |
| `scripts/experiments/human54_confwager_20261021/` | Confidence and wager rerun: token probabilities and A/B wager labels |
| `scripts/release/build_site_data.py` | Builds `docs/data/leaderboard.json` and checks that it reproduces paper Table 3 |
| `results/reanalysis/human54_layers_20261019/layer1/`, `coverage/`, `human54_posthoc_20261020_gates/model_parameters.csv` | Derived per-model endpoints and aggregated human references |

Rebuild the site data and check it against Table 3 with:

```bash
python -m scripts.release.build_site_data --output docs/data/leaderboard.json
```

## What is not included

- **Human source data and task materials.** `python -m scripts.release.fetch_sources` downloads them from the original sources into `data/raw/`, using the folder names below, and `python -m scripts.release.build_paper_materials` builds the materials. Each source keeps its own terms. See [REPRODUCING.md](REPRODUCING.md).
- **CVLT-II Standard word lists.** They are Pearson copyright and are not in this repository. `build_paper_materials` reads them from the public UCLA NNN codebook (downloaded by `fetch_sources`) and joins them with the word-free structure in `scripts/release/paper_materials/cvlt_structure.json`. A licensed copy at `scripts/experiments/human13_20261017/cvlt_standard_source.json` takes precedence when present.
- **Raw model response records.** They are about 12 GB. They are not in this repository.

| Paradigms | Source | Folder under `data/raw/` |
|---|---|---|
| DS, NB, ST, FL, GN | Self-Regulation Ontology (Eisenberg et al., 2019) | `self_regulation_ontology` |
| OS, DRM | Ball et al. (2022) | `ball2022_memory` |
| CVLT | UCLA CNP, OpenNeuro ds000030 (Poldrack et al., 2016) | `ucla_cnp_ds000030` |
| SM | Eerland et al. (2018) | `eerland2018_source_memory` |
| FB | Strachan et al. (2024) | `strachan2024_tom` |
| EP | EPITOME (Jones et al., 2024) | `epitome2024_sn7gj` |
| CAL | Mazancieux et al. (2020) | `mazancieux2020_metacognition` |
| WAG | Dienes and Scott (2005) strings; Dienes and Seth (2010) summaries | see `tom_mc.py` |

## Tests

The tests in these directories need the raw datasets, so continuous integration does not run them. After you set up `data/raw/`, run them locally, for example:

```bash
python -m pytest scripts/experiments/human13_20261017/tests
```

## Serving and administration details

These details complement Appendix A of the paper.

### Serving

- Ollama 0.18.0, native `/api/chat`, temperature 0, top-p 1, one server thread.
- Repeated-token abort is disabled. The run uses the repaired Phi-3 cache build and a native build with a thinking cap and JSON constraints.
- Response records keep the backend and model identities and the full request bodies.
- Default tags use GGUF Q4_K_M for 28 models and Q4_0 for 23, including Aya 8B, which Ollama metadata labels F16. Llama3.2 1B and SmolLM2 1.7B use Q8_0, and SmolLM2 360M uses F16.

### Hardware and records

- New calls used A100 MIG 3g.40gb, A100 MIG 4g.40gb, and RTX PRO 6000. All new calls in one model and task cell used one hardware type.
- Of 366,606 records, 360,358 are new. The other 6,248 reuse pilot or false-belief records with matching episode, step, backend, model, and request bytes.

### N-back block sampling

Each of the 80 human blocks comes from a different participant, 40 per load. Load 3 blocks were drawn first, from the 397 of 504 QC-passed participants who reached load 3. Load 2 blocks were then drawn from the remaining 464 participants.

## Supplementary structure results

These tables and settings complement Section 6 and Appendix A.6 of the paper.

### Factor-analysis settings

- The common-factor parallel analysis uses one-factor MINRES communalities.
- MINRES bounds uniqueness parameters in [.005, 1]. Reported uniquenesses are 1 minus communality and can be negative (Heywood cases).
- Oblimin rotation uses gamma = 0.
- Bootstrap alignment uses Hungarian matching of Tucker congruence with sign flips.

### Thirteen-task PCA and cross-family prediction

| Configuration | Models/families | PC1 share (%) [95% CI] | Mean RMSE | G0 RMSE | G1 RMSE | Gain (%) [95% CI] |
|---|---:|---|---:|---:|---:|---|
| Primary, invalid as wrong | 54/23 | 39.23 [31.78, 46.62] | 1.5898 | 1.5627 | 1.5154 | +3.02 [−5.85, 6.01] |
| Without WAG (12 tasks) | 54/23 | 42.49 [33.20, 50.23] | – | – | – | – |
| Original WAG contrast | 17/9 | 38.85 [35.26, 53.95] | 1.9861 | 1.9722 | 2.0983 | −6.39 [−16.55, −0.94] |

Primary PC2 and PC3 shares are 17.09% and 9.37%. Without WAG, parallel analysis still retains two dimensions (eigenvalues 5.10 and 1.95 against 95% references 2.10 and 1.76, post hoc, prediction not refit). PCA intervals use 5,000 family resamples (WAG contrast 4,956 defined, 44 constant-column failures). Mean predicts the equally weighted mean of training-family target centroids. RMSEs use fold-specific target standardization and equally weighted family and target cells. Gain is relative to G0, and its intervals use 20,000 paired family-error resamples conditional on saved predictions.

### Dimension and fit sensitivity

| Analysis unit | n | Parallel analysis PC/FA | RMSR (k = 1/2/3) | k = 2 point fit | k = 2 improper/boundary |
|---|---:|---|---|---|---|
| Primary models | 54 | 2/2 | 0.143/0.074/0.056 | proper | 37/48 |
| Primary family means | 23 | 2/2 | 0.246/0.099/0.074 | Heywood (NB) | 1005/1135 |

PC and common-factor parallel analyses retain two dimensions throughout (5,000 columnwise permutations, rankwise 95% reference). All point fits converge. Each k = 2 bootstrap has 2,000 converged, finite draws, and the improper and boundary counts overlap. Primary family means give NB uniqueness −.0202, and their k = 3 fit is also improper.

### Split and form reliability [95% CI]

| Task | Battery model reliability (n = 54) | SRO human reliability | SRO model reliability (n = 54) |
|---|---|---|---|
| DS | 0.987 [0.976, 0.992] | 0.338 [0.150, 0.492] | 0.972 [0.954, 0.979] |
| NB | 0.994 [0.962, 0.998] | 0.798 [0.743, 0.845] | 0.998 [0.993, 0.999] |
| OS | 0.993 [0.989, 0.997] | – | – |
| ST | 0.999 [0.999, 1.000] | 0.796 [0.734, 0.842] | 0.999 [0.999, 1.000] |
| FL | 0.999 [0.998, 1.000] | 0.882 [0.798, 0.927] | 0.999 [0.998, 1.000] |
| GN | 0.990 [0.982, 0.995] | 0.797 [0.741, 0.843] | 0.988 [0.895, 0.998] |
| CVLT | 0.962 [0.938, 0.984] | – | – |
| DRM | 0.949 [0.890, 0.976] | – | – |
| SM | 0.978 [0.961, 0.984] | – | – |
| FB | 0.918 [0.822, 0.959] | – | – |
| EP | 0.869 [0.793, 0.935] | – | – |
| CAL | 0.902 [0.608, 0.946] | – | – |
| WAG | 0.515 [−25.321, 0.959] | – | – |

Two forms use Spearman–Brown, and three or more use Cronbach's alpha. SRO human DS and NB use 327 and 287 defined pairs, and other tasks use 345. Battery and SRO endpoints differ. CVLT forms are trials within one session, and DRM forms are warning conditions. WAG estimates are unstable because 39 of 54 models score .5 in both halves, and the half correlation is 0.347 (−0.780 without R1 14B). CVLT gives 45 of 54 identical five-trial scores.
