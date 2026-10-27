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

- **Human source data and task materials.** Download them from the original sources and place them under `data/raw/`, using the folder names below. Each source keeps its own terms.
- **CVLT-II Standard word lists.** They are copyrighted by Pearson and require a qualified purchase or a research license. The CVLT builder expects `scripts/experiments/human13_20261017/cvlt_standard_source.json`, which you must create from your own licensed materials.
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
