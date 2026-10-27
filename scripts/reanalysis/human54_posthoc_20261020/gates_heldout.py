"""Comprehension gates on trials disjoint from the endpoint (third review, 2026-10-22). CPU only.

For each record task the planned rows split into the alternate forms of halves.py (versions, blocks,
lists, grammars, item or story parity). For every form f, the gate uses the section E rule on the rows of
f and the core endpoint (gen_core_table ROWS) is scored on all other rows; every form takes the gate
role once. DRM gates on the warned condition (studied versus unrelated new) and reads the unwarned
endpoint. CVLT has no disjoint version: its learning slope includes the gate trial.
The full row set must reproduce the saved valid_only endpoints (halves.check_full), or the run stops.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_layers_20261019.model_scoring import prepare_records
from . import halves as H
from .gates import ALPHA, binomial_gate, sdt_gate, stratum_summary
from .gen_core_table import ROWS

ROOT = Path(__file__).resolve().parents[3]
LAYER1 = ROOT / 'results/reanalysis/human54_layers_20261019/layer1'
CORE = {task: endpoint for _, task, endpoint, _ in ROWS}


def gate(task, s):
    """Section E gate on one form's valid-only endpoints s: name -> (value, n_scored)."""
    v = lambda name: s[name][0]
    n = lambda name: s[name][1]
    if task == 'n_back':
        hits, fa = v('load_2_target'), [v('load_2_lure'), v('load_2_control')]
        ns, nn = n('load_2_target'), [n('load_2_lure'), n('load_2_control')]
        if hits is None or None in fa:
            return False
        return sdt_gate(round(hits * ns), ns, round(fa[0] * nn[0]) + round(fa[1] * nn[1]), sum(nn))['p'] < ALPHA
    if task == 'go_nogo':
        h, f = v('hit_rate'), v('false_alarm_rate')
        if h is None or f is None:
            return False
        return sdt_gate(round(h * n('hit_rate')), n('hit_rate'), round(f * n('false_alarm_rate')), n('false_alarm_rate'))['p'] < ALPHA
    binom = {'flanker': ('congruent_accuracy', .5), 'stroop': ('congruent_accuracy', 1 / 3),
             'source_monitoring': ('structure_ball', .25), 'epitome_tom': ('control|1_accuracy', .5),
             'confidence_calibration': ('accuracy', .5), 'post_decision_wagering': ('accuracy', .5)}
    if task in binom:
        name, chance = binom[task]
        if v(name) is None or not n(name):
            return False
        return binomial_gate(v(name), n(name), chance)['p'] < ALPHA
    threshold = {'digit_span': 'forward_length_3', 'operation_span': 'length_3_position_accuracy',
                 'false_belief': 'session|TB_accuracy'}
    value = v(threshold[task])
    return value is not None and value >= .5


def endpoint_value(task, s):
    value = s[CORE[task]][0]
    if value is None:
        return np.nan
    if task == 'post_decision_wagering':
        return float(value in (0, 1))          # constant wager indicator
    return float(value)


def record_rows(manifest, formal, saved, models):
    bundles = {e['task']: json.loads(Path(e['material_file']).read_text()) for e in manifest['tasks']}
    out = []
    for model in models:
        for task in H.RECORD_TASKS:
            folder = formal / model.replace(':', '__') / task
            records = [json.loads(p.read_text()) for p in sorted((folder / 'calls').glob('*.json'))]
            rows = prepare_records(bundles[task], records)
            H.check_full(model, task, H.score(model, task, rows), saved)
            label = H.form_of(task, rows)
            forms = sorted({label(r) for r in rows})
            for f in forms:
                g = H.score(model, task, [r for r in rows if label(r) == f])
                e = H.score(model, task, [r for r in rows if label(r) != f])
                out.append(dict(model=model, task=task, gate_form=f, passed=bool(gate(task, g)),
                                value=endpoint_value(task, e)))
        print(json.dumps(dict(model=model)), flush=True)
    return out


def drm_rows(models):
    p = pd.read_csv(LAYER1 / 'profiles.csv', low_memory=False)
    p = p[(p.rule == 'valid_only') & (p.task == 'drm_false_memory')]
    e = pd.read_csv(LAYER1 / 'endpoints.csv', low_memory=False)
    e = e[(e.rule == 'valid_only') & (e.task == 'drm_false_memory') & (e.endpoint == CORE['drm_false_memory'])].set_index('model')
    out = []
    for m in models:
        old = p[(p.model == m) & (p.cell == 'warning|old')].iloc[0]
        new = p[(p.model == m) & (p.cell == 'warning|newunrelated')].iloc[0]
        passed = sdt_gate(round(old.value * old.n_scored), old.n_scored, round(new.value * new.n_scored), new.n_scored)['p'] < ALPHA
        out.append(dict(model=m, task='drm_false_memory', gate_form='warning', passed=bool(passed), value=float(e.value[m])))
    return out


def summarize(rows, models_table):
    paired = pd.read_csv(LAYER1 / 'model_human_endpoints.csv', low_memory=False)
    paired = paired[(paired.rule == 'valid_only') & (paired.human_cohort == 'primary')]
    e = pd.read_csv(LAYER1 / 'endpoints.csv', low_memory=False)
    e = e[(e.rule == 'valid_only') & (e.included == True)]
    out = []
    for task, endpoint in CORE.items():
        if task == 'cvlt_word_list':
            continue
        allv = e[(e.task == task) & (e.endpoint == endpoint)].set_index('model').value.astype(float)
        if task == 'post_decision_wagering':
            human, direction = None, None
        else:
            human = float(paired[(paired.task == task) & (paired.endpoint == endpoint)].human_mean.iloc[0])
            direction = 'greater' if allv.mean() > human else 'less'
        sub = rows[rows.task == task]
        for f, g in sub.groupby('gate_form'):
            keep = g[g.passed & np.isfinite(g.value)]
            s = stratum_summary(keep.value.to_numpy(), human, [models_table.family_merged[m] for m in keep.model],
                                direction, 'heldout_gate')
            if task == 'post_decision_wagering':        # values are 0/1 constant indicators
                s = dict(stratum='heldout_gate', n_models=int(len(keep)), constant=int(keep.value.sum()))
            out.append(dict(task=task, endpoint=endpoint, gate_form=f, direction=direction, **s))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=ROOT / 'results/human54_unified_20261018/prepared_r7/manifest.json')
    ap.add_argument('--formal', type=Path, default=ROOT / 'results/human54_unified_20261018/formal')
    ap.add_argument('--output', type=Path, default=ROOT / 'results/reanalysis/human54_posthoc_20261022_gates_heldout')
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    t0 = time.monotonic()
    manifest = json.loads(a.manifest.read_text())
    models_table = pd.read_csv(ROOT / 'results/reanalysis/human54_layers_20261019/coverage/models.csv', index_col='model')
    models = list(manifest['spec']['models'])
    saved = H.saved_endpoints(LAYER1 / 'endpoints.csv')
    rows = pd.DataFrame(record_rows(manifest, a.formal, saved, models) + drm_rows(models))
    rows.to_csv(a.output / 'gate_endpoint_by_form.csv', index=False)
    summarize(rows, models_table).to_csv(a.output / 'heldout_summary.csv', index=False)
    (a.output / 'run_manifest.json').write_text(json.dumps(dict(
        finished_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - t0,
        hostname=os.uname().nodename, slurm_job=os.environ.get('SLURM_JOB_ID'), post_hoc=True,
        note='third review: gates on disjoint trials'), indent=2) + '\n')


if __name__ == '__main__':
    main()
