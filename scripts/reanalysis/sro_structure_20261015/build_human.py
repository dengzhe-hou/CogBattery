"""Build SRO common-task score matrices without computing structural statistics.

CPU-only. Read original trial files and reuse the previously audited task QC.
All 522 participant IDs remain in each matrix; ineligible endpoints are NaN.
Fixed-support scores require every prespecified condition, without imputation or
weight renormalization. Available-support scores are a separate sensitivity.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
RAW = Path('data/raw/self_regulation_ontology/Complete_02-16-2019/Individual_Measures')
TASKS = ['digit_span', 'adaptive_n_back', 'stroop', 'attention_network_task', 'go_nogo']
COLS = ['digit_span', 'n_back', 'stroop', 'flanker', 'go_nogo']
# Copied unchanged from the already audited sro_human.py QC implementation.
QC_THRESH = {
    'digit_span': dict(rt=200, acc=0, missed=.25, response=np.nan),
    'go_nogo': dict(rt=200, acc=.6, missed=1, response=.98),
    'adaptive_n_back': dict(rt=200, acc=.6, missed=.25, response=.95),
    'stroop': dict(rt=200, acc=.6, missed=.25, response=.95),
    'attention_network_task': dict(rt=200, acc=.6, missed=.25, response=.95),
}


def load(task):
    d = pd.read_csv(RAW / f'{task}.csv.gz', low_memory=False)
    d['correct'] = pd.to_numeric(d['correct'].map({True: 1., False: 0., 1.: 1., 0.: 0., 'True': 1., 'False': 0.}))
    assert d['correct'].notna().all(), task
    return d.sort_values(['worker_id', 'time_elapsed'], kind='stable').reset_index(drop=True)


def qc(task, d):
    t = QC_THRESH[task]
    g_all = d.groupby('worker_id')
    resp = d[d.rt != -1]
    g = resp.groupby('worker_id')
    passed_rt = g.rt.median() >= t['rt']
    passed_miss = g_all.rt.agg(lambda x: np.mean(x == -1)) < t['missed']

    def dominant(x):
        vc = pd.Series(x).value_counts()
        return bool(np.any(vc > vc.sum() * t['response']))

    if task == 'go_nogo':
        passed_acc = g_all.correct.mean() >= t['acc']
        passed_resp = ~g_all.key_press.agg(dominant)
    else:
        passed_acc = g.correct.mean() >= t['acc']
        col = 'mouse_click' if 'mouse_click' in d.columns else 'key_press'
        passed_resp = ~g[col].agg(dominant)
    flags = pd.concat([passed_rt.rename('rt'), passed_acc.rename('acc'), passed_miss.rename('missed'),
                       passed_resp.rename('response')], axis=1).reindex(d.worker_id.unique())
    flags = flags.astype('boolean').fillna(False).astype(bool)
    flags['passed'] = flags.all(axis=1)
    return flags


def weighted(table, weights, require_all):
    weights = pd.Series(weights, dtype=float)
    x = table.reindex(columns=weights.index)
    if require_all:
        return x.mul(weights).sum(axis=1, min_count=len(weights)) / weights.sum()
    denominator = x.notna().mul(weights).sum(axis=1)
    return x.mul(weights).sum(axis=1, min_count=1) / denominator.replace(0, np.nan)


def dprime(hits, n_signal, fas, n_noise):
    if n_signal < 5 or n_noise < 5:
        return np.nan
    h = np.clip((hits + .5) / (n_signal + 1), .001, .999)
    f = np.clip((fas + .5) / (n_noise + 1), .001, .999)
    return float(norm.ppf(h) - norm.ppf(f))


def memory_usage():
    """Read this process's Linux VmHWM; retain getrusage separately for provenance."""
    status = Path('/proc/self/status').read_text().splitlines()
    vmhwm = int(next(line for line in status if line.startswith('VmHWM:')).split()[1])
    return dict(max_rss_kib=vmhwm,
                max_rss_source='/proc/self/status:VmHWM',
                getrusage_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def save_table(table, path):
    table.rename_axis('worker_id').to_csv(path)


def build(design, out):
    wall0, cpu0 = time.monotonic(), time.process_time()
    out.mkdir(parents=True, exist_ok=True)
    all_ids = None
    screened, qc_tables, qc_summary = {}, [], {}
    for task in TASKS:
        d = load(task)
        ids = set(d.worker_id)
        all_ids = ids if all_ids is None else all_ids
        assert ids == all_ids, f'participant inventory differs: {task}'
        flags = qc(task, d)
        q = flags.copy()
        q.insert(0, 'task', task)
        qc_tables.append(q)
        qc_summary[task] = dict(n_workers=len(flags), n_pass=int(flags.passed.sum()),
                                **{f'failed_{k}': int((~flags[k]).sum()) for k in flags.columns})
        screened[task] = d[d.worker_id.isin(flags.index[flags.passed])].copy()
    index = pd.Index(sorted(all_ids), name='worker_id')
    fixed = pd.DataFrame(index=index, columns=COLS, dtype=float)
    available = fixed.copy()
    construct = fixed.copy()
    native = fixed.copy()
    extra = pd.DataFrame(index=index)
    coverage = pd.DataFrame(index=index)
    pd.concat(qc_tables).rename_axis('worker_id').reset_index().to_csv(out / 'human_qc_by_task.csv', index=False)

    # Digit span: forward/backward only, common fixed lengths 3--7.
    ds = screened['digit_span']
    assert set(ds.exp_stage) == {'test'}
    ds['L'] = ds.num_digits.astype(int)
    ds['direction'] = ds.condition.map({'forward': 'forward', 'reverse': 'backward'})
    ds['position'] = ds.groupby(['worker_id', 'direction']).cumcount()
    dsc = ds.groupby(['worker_id', 'direction', 'L']).correct.agg(['mean', 'count'])
    dsc.to_csv(out / 'human_digit_span_cells.csv')
    dw = {int(k): v for k, v in design['digit_span'].items()}
    f = dsc['mean'].xs('forward', level='direction').unstack('L').reindex(columns=dw)
    b = dsc['mean'].xs('backward', level='direction').unstack('L').reindex(columns=dw)
    present = f.notna() & b.notna()
    fixed['digit_span'] = (weighted(f, dw, True) + weighted(b, dw, True)) / 2
    available['digit_span'] = (weighted(f.where(present), dw, False) + weighted(b.where(present), dw, False)) / 2
    construct['digit_span'] = fixed['digit_span']
    span = ds[ds.position > 3].groupby(['worker_id', 'direction']).L.mean().unstack('direction')
    native['digit_span'] = span[['forward', 'backward']].mean(axis=1, skipna=False)
    coverage['digit_span_n_common_lengths'] = present.sum(axis=1)
    coverage['digit_span_common_lengths'] = present.apply(lambda r: ','.join(str(k) for k in present.columns if r[k]), axis=1)
    coverage['digit_span_fixed_support'] = present.all(axis=1)
    save_table(span, out / 'human_digit_span_authors_span.csv')

    # Adaptive n-back: all trials for strict accuracy, valid responses for native d-prime.
    nb = screened['adaptive_n_back']
    nb = nb[nb.exp_stage == 'adaptive'].copy()
    nb['load'] = nb.load.astype(int)
    assert (nb.loc[nb.rt == -1, 'correct'] == 0).all()
    blocks = nb.groupby(['worker_id', 'load', 'block_num']).correct.agg(['mean', 'count'])
    blocks.to_csv(out / 'human_n_back_blocks.csv')
    nbc = blocks['mean'].groupby(['worker_id', 'load']).mean().unstack('load')
    nw = {int(k): v for k, v in design['n_back'].items()}
    fixed['n_back'] = weighted(nbc, nw, True)
    available['n_back'] = weighted(nbc, nw, False)
    npres = nbc.reindex(columns=nw).notna()
    coverage['n_back_common_loads'] = npres.apply(lambda r: ','.join(str(k) for k in npres.columns if r[k]), axis=1)
    coverage['n_back_all_observed_loads'] = nbc.notna().apply(lambda r: ','.join(str(k) for k in nbc.columns if r[k]), axis=1)
    coverage['n_back_fixed_support'] = npres.all(axis=1)
    nb_native_rows = []
    for worker, g in nb[(nb.rt != -1) & nb.load.isin(nw)].groupby('worker_id'):
        target = g.correct_response == 37
        match = g.key_press == 37
        hit, ns = int((match & target).sum()), int(target.sum())
        fa, nn = int((match & ~target).sum()), int((~target).sum())
        nb_native_rows.append(dict(worker_id=worker, hits=hit, n_target=ns, false_alarms=fa,
                                   n_nontarget=nn, dprime=dprime(hit, ns, fa, nn)))
    nb_native = pd.DataFrame(nb_native_rows).set_index('worker_id')
    construct['n_back'] = nb_native.dprime
    native['n_back'] = nb.groupby('worker_id').load.mean()
    save_table(nb_native, out / 'human_n_back_native_counts.csv')
    save_table(nbc, out / 'human_n_back_accuracy_by_load.csv')
    extra['n_back_mean_adaptive_load'] = native['n_back']

    cc_counts = {}
    for task, col, cond in [('stroop', 'stroop', 'condition'),
                            ('attention_network_task', 'flanker', 'flanker_type'),
                            ('go_nogo', 'go_nogo', 'condition')]:
        d = screened[task]
        d = d[d.exp_stage != 'practice'].copy()
        if task != 'go_nogo':
            assert (d.loc[d.rt == -1, 'correct'] == 0).all()
        cells = d.groupby(['worker_id', cond]).correct.agg(['mean', 'count'])
        cells.to_csv(out / f'human_{col}_cells.csv')
        values = cells['mean'].unstack(cond)
        fixed[col] = weighted(values, design[col], True)
        available[col] = fixed[col]
        if col == 'flanker':
            extra['flanker_full_accuracy'] = weighted(values, {'congruent': 32, 'incongruent': 34}, True)
        coverage[f'{col}_fixed_support'] = values.reindex(columns=design[col]).notna().all(axis=1)
        cc_counts[col] = dict(condition_trials=d.groupby(cond).size().astype(int).to_dict(),
                              no_key_trials=int((d.rt == -1).sum()))
        if task != 'go_nogo':
            valid = d[d.rt != -1].groupby(['worker_id', cond]).correct.mean().unstack(cond)
            native[col] = valid['incongruent'] - valid['congruent']
            construct[col] = native[col]
        else:
            rows = []
            for worker, g in d.groupby('worker_id'):
                go = g.condition == 'go'
                press = g.key_press == 32
                hit, ns = int((press & go).sum()), int(go.sum())
                fa, nn = int((press & ~go).sum()), int((~go).sum())
                rows.append(dict(worker_id=worker, hits=hit, n_go=ns, false_alarms=fa,
                                  n_nogo=nn, dprime=dprime(hit, ns, fa, nn)))
            gn = pd.DataFrame(rows).set_index('worker_id')
            native[col] = gn.dprime
            construct[col] = native[col]
            save_table(gn, out / 'human_go_nogo_native_counts.csv')

    # Construct-score sensitivity is restricted to exactly the fixed-accuracy cohort.
    construct.loc[~fixed.notna().all(axis=1), :] = np.nan
    save_table(extra, out / 'human_extra_endpoints.csv')
    fixed_full = fixed.copy()
    fixed_full['flanker'] = extra['flanker_full_accuracy']
    available_full = available.copy()
    available_full['flanker'] = extra['flanker_full_accuracy']
    summaries = {}
    for name, matrix in [('fixed_accuracy', fixed), ('available_accuracy', available),
                         ('fixed_construct', construct), ('native_reference', native),
                         ('fixed_accuracy_full_flanker', fixed_full), ('available_accuracy_full_flanker', available_full)]:
        save_table(matrix, out / f'human_{name}.csv')
        coverage[f'{name}_complete'] = matrix.notna().all(axis=1)
        summaries[name] = dict(n_rows=len(matrix), n_complete=int(matrix.notna().all(axis=1).sum()),
                                n_available_by_task=matrix.notna().sum().astype(int).to_dict())
    save_table(coverage, out / 'human_coverage_by_participant.csv')
    summary = dict(source=str(RAW), task_order=COLS, groups=['WM', 'WM', 'CC', 'CC', 'CC'],
                   score_design=design, qc=qc_summary, matrices=summaries, cc_trial_counts=cc_counts,
                   digit_span_common_length_profiles=coverage.digit_span_common_lengths.value_counts().to_dict(),
                   n_back_observed_load_profiles=coverage.n_back_all_observed_loads.value_counts().to_dict(),
                   definitions={
                       'fixed_accuracy': 'Accuracy with all required cells; digit span equal forward/backward, fixed length weights; n-back equal blocks within load, fixed load weights; CC fixed condition weights. Missed responses count incorrect, except go/no-go nonresponse is task-valid.',
                       'available_accuracy': 'Sensitivity only: digit span weights renormalized over lengths shared by both directions, n-back weights renormalized over observed prescribed loads; CC unchanged.',
                       'fixed_construct': 'Sensitivity in the fixed-accuracy complete-case cohort only: digit span fixed accuracy unchanged; n-back pooled loglinear dprime on valid-response trials at loads 1/2/3; Stroop/flanker I-C accuracy on valid responses; go/no-go loglinear dprime on all test trials.',
                       'native_reference': 'Human-only reference: digit span mean presented length after first four trials per direction (then equal direction mean); n-back trial-weighted mean adaptive load across all reached loads; Stroop/flanker I-C accuracy on valid responses; go/no-go loglinear dprime on all test trials. Not an equivalent human-model score scale.',
                       'full_flanker_accuracy': 'Sensitivity using congruent/incongruent accuracy weights 32/34 instead of arrow-only LLM weights 10/11; human ANT remains arrow stimuli.',
                       'dprime': '(hits+0.5)/(signal+1), (FA+0.5)/(noise+1), rates clipped to [.001,.999]; signal and noise each >=5.',
                   },
                   limitations=[
                       'The fixed-support complete cases are a conditional common-support sample, not the full human ability distribution. Adaptive task coverage depends on preceding performance.',
                       'Common task families and score weights do not match stimuli, timing, response opportunities, practice, or material visibility.',
                       'Digit span excludes LLM sequencing tasks because human SRO has only forward and reverse.',
                       'Human ANT flanker accuracy averages cue and target-location conditions; LLM arrow flanker has no matched cue/location protocol. Neutral ANT trials are excluded from the common score.',
                       'Human Stroop is a timed color-word response task; the LLM score retains the approved mixed text-adapted Stroop item set, so matching C/I weights does not equate the measurement instrument.',
                       'Human-native span and mean adaptive n-back load have no equivalent LLM fixed-design endpoints; this reference is human-only.',
                       'Construct n-back dprime pools valid trials across loads 1/2/3; adaptive human load frequencies differ from the fixed LLM episode/turn frequencies.',
                       'Mean adaptive load is explicitly trial weighted; the original authors mean_load DV implementation has not been independently rechecked here.',
                   ], resource=dict(wall_seconds=time.monotonic()-wall0, cpu_seconds=time.process_time()-cpu0,
                                    **memory_usage()))
    with (out / 'human_input_summary.json').open('w') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(dict(matrices=summaries, resource=summary['resource']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'results/reanalysis/sro_structure_20261015/inputs')
    args = p.parse_args()
    build(json.loads(args.design.read_text()), args.out)
