"""CPU analysis of the confidence/wager rerun (SPEC.md, written before the run). No model calls.

Endpoints come from the frozen layer scorer (analyze_cell, rule valid_only), applied to the rerun
records. A/B wager replies are mapped to sweets (1/2) before scoring. Token confidence is read at the
first generated position of the cal_logprobs answer requests (SPEC change log 2026-10-21).
Family intervals use the posthoc cluster_mean (whole-family bootstrap, 5,000 draws, human fixed).
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.reanalysis.human54_layers_20261019.model_scoring import analyze_cell, prepare_records
from scripts.reanalysis.human54_layers_20261019.analysis_statistics import cluster_mean
from scripts.experiments.human54_confwager_20261021 import runner as R

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'results/human54_confwager_20261021/mdx_run/records'
FORMAL = ROOT / 'results/human54_unified_20261018/formal'
MANIFEST = ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json'
MODELS = ROOT / 'results/human54_confwager_20261021/mdx_inputs/models.json'
COVERAGE = ROOT / 'results/reanalysis/human54_layers_20261019/coverage/models.csv'
GATES = ROOT / 'results/reanalysis/human54_posthoc_20261020_gates'
LAYER1 = ROOT / 'results/reanalysis/human54_layers_20261019/layer1'
CONDITIONS = ('cal_original', 'cal_logprobs', 'wag_original', 'wag_AB', 'wag_BA')
ENDPOINTS = ('accuracy', 'confidence_mean_percent', 'official_bias', 'type2_auc_fixed_direction',
             'high_wager_rate', 'high_minus_low_accuracy')
LARGE_B = 27.0
OPTIONS = ('s', 'l')


def bundles():
    tasks = json.loads(MANIFEST.read_text())['tasks']
    return {t['task']: json.loads(Path(t['material_file']).read_text())
            for t in tasks if t['task'] in ('confidence_calibration', 'post_decision_wagering')}


def load(directory):
    return [json.loads(p.read_text()) for p in sorted((directory / 'calls').glob('*.json'))]


def to_sweets(condition, record):
    """Map an A/B wager reply to the sweets it stands for; other steps are unchanged."""
    if condition not in R.LABELS or record['step']['kind'] != 'wager' or not record['technical_valid']:
        return record
    label = record['response']
    amount = str(R.wager_amount(condition, label))
    return dict(record, response=amount, raw_content=record['raw_content'].replace(label, amount), wager_label=label)


def endpoints(model, condition, bundle, records):
    # Rerun records name the rerun condition in condition_variant; the scorer wants the episode's
    # material condition, which the formal records carried as `condition`.
    material = {ep['id']: ep['condition'] for ep in bundle['episodes']}
    records = [dict(r, condition=material[r['episode_id']]) for r in records]
    out = analyze_cell(model, R.TASK[condition], bundle, records)
    rows = [e for e in out['endpoints'] if e['rule'] == 'valid_only' and e['endpoint'] in ENDPOINTS]
    cats = pd.Series([r['category'] for r in out['rows']]).value_counts().to_dict()
    return {e['endpoint']: e['value'] for e in rows}, cats


def first_position(record):
    """s/l probabilities at the first generated position, renormalized over the two options."""
    top = record['raw_response']['logprobs'][0]['top_logprobs']
    found = {}
    for t in top:
        key = t['token'].strip()
        if key in OPTIONS:
            found[key] = max(found.get(key, -math.inf), t['logprob'])
    floor = min(t['logprob'] for t in top)
    return found, floor


def token_confidence(model, records, verbal):
    rows = []
    for r in records:
        if r['step']['kind'] != 'cal_answer':
            continue
        found, floor = first_position(r)
        text = r['response'] if r['technical_valid'] else None
        row = dict(model=model, episode_id=r['episode_id'], step_id=r['step']['id'], text=text,
                   correct=(text == r['step']['gold']) if text in OPTIONS else None,
                   n_options_found=len(found), verbal=verbal.get((r['episode_id'], r['step']['id'])))
        if text in OPTIONS:
            lp = {o: found.get(o, floor) for o in OPTIONS}          # floor = upper bound for a missing option
            z = np.logaddexp(lp['s'], lp['l'])
            row['token_conf_bound'] = float(np.exp(lp[text] - z))
            if len(found) == 2:
                row['token_conf'] = row['token_conf_bound']
                row['argmax_disagrees'] = max(OPTIONS, key=lambda o: lp[o]) != text
        rows.append(row)
    return rows


def auroc(conf, correct):
    right = [c for c, y in zip(conf, correct) if y]
    wrong = [c for c, y in zip(conf, correct) if not y]
    if not right or not wrong:
        return None
    return float(np.mean([float(a > b) + .5 * float(a == b) for a in right for b in wrong]))


def verbal_lookup(bundle, records):
    """(episode, answer step) -> verbal confidence as parsed by the frozen scorer (valid replies only)."""
    material = {ep['id']: ep['condition'] for ep in bundle['episodes']}
    rows = prepare_records(bundle, [dict(r, condition=material[r['episode_id']]) for r in records])
    return {(r['episode_id'], r['step']['meta']['answer_step_id']): float(r['score']['parsed'])
            for r in rows if r['step']['kind'] == 'confidence' and r['category'] == 'A'}


def agreement(model, condition, records):
    formal = {(r['episode_id'], r['step']['id']): r['response']
              for r in load(FORMAL / model.replace(':', '__') / R.TASK[condition])}
    same = [formal.get((r['episode_id'], r['step']['id'])) == r['response'] for r in records]
    return float(np.mean(same))


def wager_labels(records):
    labels = [r.get('wager_label') for r in records if r.get('wager_label')]
    return float(np.mean([x == 'A' for x in labels])) if labels else None


def stratum(values, human, families, label, direction=-1):
    v = np.asarray(values, float)
    keep = np.isfinite(v)
    v, f = v[keep], np.asarray(families)[keep]
    if not len(v):
        return dict(stratum=label, n_models=0)
    stats, _ = cluster_mean(v - human, f)
    fam = pd.Series(v, index=f).groupby(level=0).mean()
    return dict(stratum=label, n_models=int(len(v)), n_families=int(len(fam)), model_mean=float(v.mean()),
                human=human, difference=stats['mean'], lower=stats['lower'], upper=stats['upper'],
                models_in_direction=int(np.sum(direction * (v - human) > 0)),
                families_in_direction=int(np.sum(direction * (fam.to_numpy() - human) > 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reanalysis/human54_confwager_20261022')
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(f'{a.output} exists; use a new output directory')
    a.output.mkdir(parents=True)
    spec = json.loads(MODELS.read_text())
    models = [m['model'] for m in spec['models'] + spec['big']]
    B = bundles()
    cover = pd.read_csv(COVERAGE, index_col='model')
    params = pd.read_csv(GATES / 'model_parameters.csv', index_col='model').params_b
    gates = pd.read_csv(GATES / 'gates.csv')
    cal_gate = gates[gates.task == 'confidence_calibration'].set_index('model').passed
    wag_gate = gates[gates.task == 'post_decision_wagering'].set_index('model').passed
    layer = pd.read_csv(LAYER1 / 'endpoints.csv', low_memory=False)
    layer = layer[(layer.rule == 'valid_only') & layer.endpoint.isin(ENDPOINTS)]
    paired = pd.read_csv(LAYER1 / 'model_human_endpoints.csv', low_memory=False)
    human_cal = float(paired[(paired.rule == 'valid_only') & (paired.human_cohort == 'primary') &
                             (paired.task == 'confidence_calibration') &
                             (paired.endpoint == 'type2_auc_fixed_direction')].human_mean.iloc[0])

    ep_rows, tok_rows, cat_rows = [], [], []
    for model in models:
        slug = model.replace(':', '__')
        recs = {c: [to_sweets(c, r) for r in load(RUN / slug / c)] for c in CONDITIONS}
        verbal = verbal_lookup(B['confidence_calibration'], recs['cal_original'])
        for c in CONDITIONS:
            values, cats = endpoints(model, c, B[R.TASK[c]], recs[c])
            row = dict(model=model, condition=c, **values)
            if c in ('cal_original', 'wag_original'):
                row['formal_step_agreement'] = agreement(model, c, recs[c])
            if c in R.LABELS:
                row['pick_A_rate'] = wager_labels(recs[c])
            ep_rows.append(row)
            cat_rows.append(dict(model=model, condition=c, **{f'category_{k}': v for k, v in cats.items()}))
        tok_rows += token_confidence(model, recs['cal_logprobs'], verbal)
        print(model, flush=True)

    ep = pd.DataFrame(ep_rows)
    tok = pd.DataFrame(tok_rows)
    ep.to_csv(a.output / 'endpoints_by_condition.csv', index=False)
    tok.to_csv(a.output / 'token_confidence_items.csv', index=False)
    pd.DataFrame(cat_rows).fillna(0).to_csv(a.output / 'categories.csv', index=False)

    # Per-model CAL token measures.
    per = []
    for model, g in tok.groupby('model', sort=False):
        full = g.dropna(subset=['token_conf'])
        both = full.dropna(subset=['verbal'])
        rho = spearmanr(both.token_conf, both.verbal).statistic if len(both) > 2 and both.verbal.nunique() > 1 and both.token_conf.nunique() > 1 else None
        answered = g[g.text.isin(OPTIONS)]
        per.append(dict(model=model, n_answers=len(answered), n_both_options=len(full),
                        options_missing=int(len(answered) - len(full)),
                        token_auroc=auroc(full.token_conf, full.correct.astype(bool)),
                        token_auroc_bound=auroc(answered.token_conf_bound, answered.correct.astype(bool)),
                        token_conf_mean=float(full.token_conf.mean()) if len(full) else None,
                        spearman_token_verbal=None if rho is None or not np.isfinite(rho) else float(rho),
                        argmax_disagree_rate=float(full.argmax_disagrees.mean()) if len(full) else None))
    per = pd.DataFrame(per).set_index('model')
    wide = ep.pivot(index='model', columns='condition')
    per['verbal_auroc_rerun'] = wide[('type2_auc_fixed_direction', 'cal_original')]
    formal_auc = layer[(layer.task == 'confidence_calibration') & (layer.endpoint == 'type2_auc_fixed_direction')].set_index('model').value
    per['verbal_auroc_formal'] = formal_auc.reindex(per.index)
    per['family'] = cover.family_merged.reindex(per.index)
    per['params_b'] = params.reindex(per.index)
    per['cal_gate'] = cal_gate.reindex(per.index)
    per.to_csv(a.output / 'cal_token_by_model.csv')

    # Human comparison by stratum (SPEC: same strata as posthoc E2).
    strata = []
    for measure in ('token_auroc', 'token_auroc_bound', 'verbal_auroc_rerun', 'verbal_auroc_formal'):
        for label, keep in (('all', per.index == per.index),
                            ('gate', per.cal_gate.fillna(False).astype(bool)),
                            ('large', per.params_b >= LARGE_B)):
            sub = per[keep]
            strata.append(dict(measure=measure, **stratum(sub[measure], human_cal, sub.family, label)))
    pd.DataFrame(strata).to_csv(a.output / 'cal_strata.csv', index=False)

    # Wager: constant wagers per condition and label vs amount following.
    w = pd.DataFrame(index=models)
    for c in ('wag_original', 'wag_AB', 'wag_BA'):
        w[c + '_high_rate'] = wide[('high_wager_rate', c)].reindex(models)
        w[c + '_constant'] = w[c + '_high_rate'].isin([0., 1.])
    w['pick_A_AB'] = wide[('pick_A_rate', 'wag_AB')].reindex(models)
    w['pick_A_BA'] = wide[('pick_A_rate', 'wag_BA')].reindex(models)
    # Label index 0 = picks A equally often whatever A means; 1 = flips with the amount.
    w['amount_following'] = (w.pick_A_AB - w.pick_A_BA).abs()
    w['formal_high_rate'] = layer[(layer.task == 'post_decision_wagering') & (layer.endpoint == 'high_wager_rate')].set_index('model').value.reindex(models)
    w['family'] = cover.family_merged.reindex(models)
    w['params_b'] = params.reindex(models)
    w['wag_gate'] = wag_gate.reindex(models)
    w.to_csv(a.output / 'wager_by_model.csv', index_label='model')

    (a.output / 'run_manifest.json').write_text(json.dumps(dict(
        finished_at=datetime.now().astimezone().isoformat(), spec='scripts/experiments/human54_confwager_20261021/SPEC.md',
        models=len(models), human_cal_auroc=human_cal, large_b=LARGE_B, hostname=os.uname().nodename,
        slurm_job=os.environ.get('SLURM_JOB_ID')), indent=2) + '\n')


if __name__ == '__main__':
    main()
