# Paper version: 54 models and source-specific human references

This page describes the code for the current paper version of CogBattery. That version evaluates 54 open-weight models on text adaptations of 13 paradigms and compares each one with the human data published for it. The protocol is `formal_r7_unified_20261018`. The rest of this repository is the earlier arXiv v1 battery, which uses procedurally generated items and covers 55 models.

The project site at https://dengzhe-hou.github.io/CogBattery/ shows the task guide and the per-model results.

The current overview figure is available as [PDF](docs/figures/fig1_overview.pdf), [SVG](docs/figures/fig1_overview.svg), and [editable PowerPoint](docs/figures/CogBattery_Fig1_editable.pptx). The task cards are schematic examples.

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

The paper has no appendix. Its former appendix (model and task tables, prompt examples, validity rules, analysis order, and supplementary result tables S1 to S8, plus S9 for the two further cohorts and S10 for the wager-label rerun) is in [cogbattery_supplement.pdf](docs/cogbattery_supplement.pdf). The details below add serving, source, and analysis settings.

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

### Budgets, contexts, and quantization

- Direct-answer contexts are 4,096 tokens, with output limits of 16 to 1,024 tokens by step type.
- Qwen3 4B (served as Qwen3-4B-Thinking-2507) and DeepSeek-R1 7B and 14B (Qwen distills) use native thinking with 12,288-token contexts and 8,192-token limits (96 thinking plus 32 final tokens in OS).
- Budget exhaustion never triggers regeneration. Responses are schema-constrained except FB, and only materials and final answers enter later history.
- Hardware does not determine inclusion, weights, or adjustment.

### Sources and task administration

- DS, ST, FL, and GN replay three public SRO trajectories (Eisenberg et al., 2019). QC retains 522, 521, 519, and 521 humans. DS uses both directions and lengths 3 to 9 without adaptive model spans. Forward minus backward pools each person's or model's trials, so it also reflects trajectory length.
- NB uses one random lure-containing block from each of 80 participants (40 per load), with lag n±1 lures and control non-targets. ST substitutes color labels. FL has three flanker types. GN requires `WITHHOLD`.
- OS reconstructs Ball's rules, not participant item orders (Ball et al., 2022). It has three versions of 15 sets (lengths 3 to 7). Human SPANo is correct positions / 75 for 438 people.
- CVLT uses the CVLT-II Standard lists (Delis et al., 2000) and 130 CNP controls (Poldrack et al., 2016). Delay labels denote stages, not elapsed time.
- DRM uses Ball's sentences, with 40 studied, eight lure, and eight unrelated probes per warning condition (240 and 197 humans). Warning is between-person in humans and paired in models.
- SM uses the Eerland et al. (2018) stories with all 48 questions. Exp. 1 has 299 complete human records, and published exclusions are not reconstructed. Exp. 2 human scoring first used Exp. 1 codes. The corrected scores apply each experiment's author formulas, and model scores are unchanged.
- FB uses the Strachan et al. (2024) stories and 49 humans' author-coded answers. Each condition has seven false-belief, seven true-belief, and four check entries. Two stories have an altered structure that the source paper did not report, each in both lists. Session administration with a system message is primary.
- EP uses EPITOME's processed 2AFC data (Jones et al., 2024), 63 humans and 224 items, with generated A/B answers instead of official PMI scoring.
- CAL uses French semantic-memory 2AFC materials (Mazancieux et al., 2020) with 181 human sets of 40 answer and confidence pairs on the 11-level 0–100% scale of the model prompt. Models answer each item in both answer positions (80 pairs), each pair in its own context.
- WAG uses the grammar strings of Dienes and Scott (2005). The 40-human reference (Dienes and Seth, 2010) reports only accuracy (65%, SE 1.9) and high-minus-low accuracy (10.5 points, SE 2.18). Models make 240 instructed bets without stakes. Each test sequence starts with 10 sweets, a correct answer wins the amount wagered, an incorrect answer loses it, and a negative final balance is not collected. So a high wager has the larger expected gain whenever an answer is more likely right than wrong.

### Behavioral comparisons and uncertainty

- Family labels are analysis clusters, not independent-ancestry claims. Intervals are descriptive and unadjusted. There are no human–model significance or equivalence tests.
- The human-error sensitivity draws the human mean from a normal distribution with its standard error in each family draw. The error is published for WAG and is SD / √N over people elsewhere.
- Within-person NB and EP contrasts use per-person differences. SM quotation and DRM warning contrasts combine group errors. EP order slopes use 5,000 person draws.

### Structure within jointly measured cohorts

- SRO uses accuracies. It weights DS length-3 to 7 cells by the model recalls, averages NB block accuracy within load and then across loads, and weights ST and FL conditions equally.
- Ball's pooled, unwarned, and warned cohorts retain all three DRM scores.
- All ten SRO and three Ball partitions are enumerated. Independent participant and family draws estimate δ_H − δ_M.

### Full-battery model analyses

- WAG reconstructs correct and incorrect answers by high and low counts, awarding ties half credit.
- PCA standardizes columns with population standard deviations.
- Prediction fits standardization, PC1, OLS residualization of the same-group mean, and ridge regression (α = 1e-6) within each training fold, without tuning.
- Parallel analysis (Horn, 1965) uses 5,000 column permutations. Bootstrap factor solutions are aligned by Tucker congruence with sign flips. Failed fits are counted, not replaced.

### Scale, prediction controls, and reliability

- Scale uses served parameter counts. The mixed models of Section 5.6 use REML fits with a random intercept per merged family, a 0/1 native-thinking term, and Wald intervals. Their likelihood-ratio tests compare maximum-likelihood fits with and without log size.
- Within-family correlations demean both variables. Residualized analyses regress standardized tasks on log size.
- F2 tests three primary and four secondary bias indicators (Table 12 of the paper).
- Family maps retain 28 raw labels or merge lineages (18 or 17 families).
- The positive control simulates 100 matrices per group strength with the same family map, a general factor, and one factor per paradigm group. With population δ near the human .16, the mean gain is 3.96%, its interval excludes zero in 26 matrices, and the partition tail is at most .05 in 97. Without group factors these counts are 0 and 7, and at δ = .23 they are 82 and 100.
- Reliability uses the same score on alternate forms (trajectories, versions, blocks, lists, grammars, item or story parity, warning conditions, recall trials), with 5,000 family draws. Disattenuation r_jk / √(q_j q_k) is undefined for nonpositive reliabilities.

### Confidence and wager rerun

- The 51 direct-answer models were rerun on rented A100 GPUs with the same server binary, weights, context, and request builder. All 103,530 calls were valid. Responses matched the formal run on .957 (CAL) and .954 (WAG) of steps.
- Token confidence is the chosen option's renormalized probability at the first generated position (top 20). That position holds the opening quote of the constrained answer, and Ollama returns unconstrained probabilities there, so the most probable option can differ from the written answer.
- The primary analysis keeps models with both options on at least 40 of 80 items. This minimum and the wager-label thresholds were set after the specification.
- Of the 36 rerun models that always chose the same wager in the formal run, 21 choose the same letter on at least 90% of wagers under both A/B mappings, 5 keep the same amount on at least 80%, and 10 do neither. Among the four of them with WAG accuracy of at least .80, mistral-nemo 12B chooses the same letter, gemma2 9B and gemma3 27B keep the low wager, and gemma2 27B does neither (Table S10).

### Protocol and seeds

- Protocol `formal_r7_unified_20261018`, material seed 20261017.
- Analysis seeds 20261018 and 20261019 (factors).

### Structure on the key endpoints (Section 6)

`scripts/reanalysis/human54_keyendpoint_structure_20261029/run.py` repeats the structure analyses on the 12 key endpoints of Table 4 (WAG dropped, each oriented to its structural score). Parallel analysis keeps one dimension, PC1 explains 23.1% and correlates .68 with log size, and the five groups give -2.02% [-5.10, 0.48]% in leave-one-family-out prediction. Added after the formal results.

### Alternative task groupings (Section 6)

`scripts/reanalysis/human54_alt_groupings_20261029/run.py` repeats the leave-one-family-out prediction test with two groupings of the same block sizes: a response-format grouping (DS OS CVLT, NB GN DRM, ST FL SM, FB EP, CAL WAG) and a grouping chosen inside each fold from training-family correlations. Gains are 4.63% [-6.58, 8.52]% and 0.80% [-4.10, 11.68]%, against 3.02% [-5.85, 6.01]% for the predefined groups. Added after the formal results. Like the other analysis scripts here, it reads the derived score matrix from the paper run, which is not part of this repository.

### Distance from humans per model (Figure 3)

`scripts/reanalysis/human54_model_closeness_20261029/run13.py` standardizes each model's key-endpoint difference from the human mean by the across-person human SD (SM pooled within-condition SD, EP SD of per-person slopes, WAG published SE times √40 for high-minus-low accuracy), and `gen_closeness_heatmap13_v2.py --paper` draws Figure 3. `run.py` is the 10-task version without SM, EP, and WAG. This view was added after the formal results. The homepage table uses the same human SDs for its shading.

## Robustness of behavioral results

These checks were removed from the paper and are not in the supplement PDF.

These checks support Section 5 of the paper. Family counts use the 23 merged families and raw counts the 28 raw labels.

**Working memory.** The n-back control false-alarm excess appears in 19 of 23 family means and under every single-family deletion (.226–.269). Lure sensitivity and excess control false alarms coexist in 32 models. Without the 16 models that give one answer on at least 95% of 2-back letters, the excess is .238 (33 of 38 models). The target-minus-lure difference against humans is −.319 [−.382, −.232], and direct-answer models average −.004. At 3-back, 13 of 23 families show the control excess. The operation-span gate trials are part of the endpoint.

**Cognitive control.** The Stroop word-response excess holds in all 23 merged-family means, under every single-family deletion (.318–.367), in 27 of 28 raw families, and in the 51- and 19-model panels (22 of 22 and 8 of 9 families). The Flanker congruent gate shares trials with the contrast.

**Episodic memory.** The model-minus-human difference in the DRM warning effect is −.006 [−.055, .027]. Human warning groups are different people, so this comparison is unpaired. The new-word trials of DRM gate passers are part of the contrast.

**Social reasoning.** Main-item belief accuracy ranges over .778–.877 across session and isolated presentation, with and without a system message. Depending on condition, 16–34 models answer all 14 main items correctly.

**Metacognition.** Lower AUROC holds in all 23 merged-family means, 27 of 28 raw families, under every single-family deletion (−.164 to −.132), in both answer orders, and in the 19- and 51-model panels. In the rerun, mean verbal AUROC on all items is below humans (−.139 [−.186, −.104]) in the 41 models with usable token probabilities. Eight of the nine models of 27B or more are among them and average token AUROC .867. Constant wagers span 36 of 51 direct-answer models.

**Gates.** Disjoint-trial gates swap gate and endpoint roles across versions, blocks, lists, stories, grammars, or item halves. DRM uses the warned condition, and CVLT cannot be split. Gaps remain close to same-trial values, with Flanker interference at .341–.352 (16–17 passers), operation span at .057–.070, and CAL at −.140 to −.061. Held-out n-back gaps among gate passers are .234–.298.

## Supplementary structure results

These tables and settings complement Section 6 of the paper.

### Pairwise correlations in two further cohorts

The paper reports this result in one sentence in Section 6. CNP and Bradford each have one task per paradigm group, so the grouping contrast is undefined. We compare each task pair's Pearson correlation in humans and in the 54 models.

- CNP, 130 healthy controls ([OpenNeuro ds000030](https://openneuro.org/datasets/ds000030), Poldrack et al. 2016). CVLT-II trials 1 to 5 total recall, Wechsler Memory Scale digit span total, and mean Stroop accuracy.
- Bradford et al. (2020, Cortex 126, 265 to 280; [OSF pw7h6](https://osf.io/pw7h6/)), adults aged 18 or older. False-belief appropriateness ratings (belief-inconsistent minus belief-consistent, where higher ratings mean less appropriate), partial operation span (letters recalled in position), and log congruent minus log incongruent Stroop reaction time. Each measure is residualized on age and age squared.
- Model scores are battery CVLT total recall, DS, OS, and ST accuracy, false-belief trial accuracy, and for the Bradford pairs Stroop incongruent minus congruent accuracy. Higher values mean better performance or less interference on both sides.
- Intervals use 5,000 bootstraps over people and over the 23 merged model families. The task matches are at task-family level only.

| Cohort | Pair | n | Human r | Model r | Human minus model [95% CI] |
|---|---|---:|---:|---:|---|
| CNP | CVLT and DS | 130 | .21 | .40 | −.20 [−.47, .17] |
| CNP | CVLT and ST | 130 | .22 | .40 | −.17 [−.42, .14] |
| CNP | DS and ST | 130 | .06 | .44 | **−.38 [−.61, −.13]** |
| Bradford | FB and OS | 252 | .20 | .29 | −.10 [−.39, .22] |
| Bradford | FB and ST | 253 | −.09 | −.12 | .03 [−.36, .44] |
| Bradford | OS and ST | 249 | −.12 | .15 | −.27 [−.49, .04] |

Model correlations exceed human ones in five of six pairs, and the interval excludes zero only for CNP DS and ST. Without age adjustment, the Bradford OS and ST difference is −.36 [−.59, −.05]. With Stroop interference instead of accuracy, all CNP difference intervals include zero. Using battery false-belief main accuracy for models changes the Bradford differences by at most .03. The four joint cohorts cover ten tasks, so CAL, EP, and WAG have no human structural comparison. Added after the formal results. Code is `scripts/reanalysis/human54_extra_cohorts_20261029/run.py`, which reads the raw sources from `$COGBATTERY_RAW`.

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
