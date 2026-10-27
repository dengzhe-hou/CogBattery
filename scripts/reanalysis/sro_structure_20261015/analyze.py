#!/usr/bin/env python3
"""SRO/LLM five-task score structures. See results/.../SPEC.md before running."""
import argparse
import itertools
import json
import os
from pathlib import Path
import resource
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '1', f'{key}=1 required'

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'results/reanalysis/sro_structure_20261015'
TASKS = ['digit_span', 'n_back', 'stroop', 'flanker', 'go_nogo']
I, J = np.triu_indices(5, 1)
WITHIN = ((I < 2) & (J < 2)) | ((I >= 2) & (J >= 2))
METRICS = ['within', 'between', 'delta', 'pc1', 'wm_correlation', 'cc_mean', 'mean_correlation']
SEED = 20261015


def statistics(corr):
    corr = np.asarray(corr)
    edges = corr[..., I, J]
    within = edges[..., WITHIN].mean(axis=-1)
    between = edges[..., ~WITHIN].mean(axis=-1)
    eig = np.linalg.eigvalsh(corr)
    return np.stack([within, between, within-between, eig[..., -1]/5,
                     corr[..., 0, 1], edges[..., (I >= 2)].mean(axis=-1),
                     edges.mean(axis=-1)], axis=-1)


def bootstrap(x, families, n_boot, seed):
    """Exact cluster resampling via multinomial multiplicities and moments."""
    n = len(x)
    if families is None:
        units = np.eye(n)
    else:
        codes, unique = pd.factorize(families, sort=True)
        units = (np.arange(len(unique))[:, None] == codes[None, :]).astype(float)
    k = len(units)
    rng = np.random.default_rng(seed)
    outer = (x[:, :, None] * x[:, None, :]).reshape(n, 25)
    draws = []
    valid_count = 0
    # Small chunks keep the human bootstrap memory bounded.
    for start in range(0, n_boot, 250):
        b = min(250, n_boot-start)
        weights = rng.multinomial(k, np.full(k, 1/k), size=b) @ units
        total = weights.sum(axis=1)
        mean = weights @ x / total[:, None]
        cov = (weights @ outer / total[:, None]).reshape(b, 5, 5) - mean[:, :, None]*mean[:, None, :]
        var = np.diagonal(cov, axis1=1, axis2=2)
        valid = np.all(var > 1e-15, axis=1)
        corr = np.full((b, 5, 5), np.nan)
        sd = np.sqrt(var[valid])
        corr[valid] = cov[valid] / (sd[:, :, None]*sd[:, None, :])
        # Numerical diagonal values are exactly one; no effect-dependent filtering.
        corr[valid] = np.clip(corr[valid], -1, 1)
        valid_count += int(valid.sum())
        draws.append(corr)
    return np.concatenate(draws), valid_count


def partitions(corr):
    rows = []
    for pair in itertools.combinations(range(5), 2):
        member = np.isin(np.arange(5), pair)
        within = member[I] == member[J]
        delta = float(corr[I, J][within].mean() - corr[I, J][~within].mean())
        rows.append(dict(group_of_two='|'.join(TASKS[j] for j in pair),
                         group_of_three='|'.join(TASKS[j] for j in range(5) if j not in pair),
                         delta=delta, proposed=pair == (0, 1)))
    orig = next(r['delta'] for r in rows if r['proposed'])
    for row in rows:
        row['rank'] = 1 + sum(r['delta'] > row['delta'] + 1e-12 for r in rows)
    return rows, sum(r['delta'] >= orig-1e-12 for r in rows)/len(rows)


def load(name, kind, full_flanker=False):
    frame = pd.read_csv(BASE/'inputs'/name, dtype={kind: str})
    assert not frame[kind].duplicated().any(), name
    if full_flanker:
        frame['flanker'] = frame['flanker_full']
    return frame.set_index(kind)


def complete(frame):
    return frame.dropna(subset=TASKS).copy()


def population(frame, group, run_name, output, n_boot, seed):
    frame = complete(frame)
    x = frame[TASKS].to_numpy(float)
    assert len(x) > 5 and np.isfinite(x).all()
    assert (x.std(axis=0, ddof=1) > 0).all(), (run_name, group, 'constant task')
    families = frame['family'].to_numpy() if group == 'llm' else None
    corr = np.corrcoef(x, rowvar=False)
    point = statistics(corr)
    spearman = np.corrcoef(rankdata(np.round(x, 10), axis=0), rowvar=False)
    boots, n_valid = bootstrap(x, families, n_boot, seed)
    good = np.isfinite(boots).all(axis=(1, 2))
    boot_stats = np.full((n_boot, len(METRICS)), np.nan)
    boot_stats[good] = statistics(boots[good])
    ci = np.nanquantile(boot_stats, [.025, .975], axis=0)
    edge_ci = np.nanquantile(boots[:, I, J], [.025, .975], axis=0)
    part, tail = partitions(corr)
    results = dict(n=len(frame), families=None if families is None else len(set(families)),
                   n_boot=n_boot, n_boot_valid=n_valid, seed=seed,
                   metrics={key: dict(value=float(point[z]), ci95=ci[:, z].tolist()) for z, key in enumerate(METRICS)},
                   eigenvalues=np.linalg.eigvalsh(corr)[::-1].tolist(),
                   spearman_metrics=dict(zip(METRICS, statistics(spearman).tolist())),
                   partition_tail=tail,
                   proposed_rank=next(p['rank'] for p in part if p['proposed']),
                   ids=frame.index.tolist())
    prefix = output/f'{run_name}_{group}'
    frame.to_csv(str(prefix)+'_scores.csv')
    pd.DataFrame(corr, index=TASKS, columns=TASKS).to_csv(str(prefix)+'_pearson.csv')
    pd.DataFrame(spearman, index=TASKS, columns=TASKS).to_csv(str(prefix)+'_spearman.csv')
    pd.DataFrame(part).sort_values('rank').to_csv(str(prefix)+'_partitions.csv', index=False)
    pd.DataFrame([dict(task_a=TASKS[i], task_b=TASKS[j], within_group=bool(w),
                       pearson=float(corr[i, j]), ci_lower=float(edge_ci[0, k]), ci_upper=float(edge_ci[1, k]),
                       spearman=float(spearman[i, j])) for k, (i, j, w) in enumerate(zip(I, J, WITHIN))]).to_csv(str(prefix)+'_edges.csv', index=False)
    desc = pd.DataFrame(dict(n=frame[TASKS].count(), mean=frame[TASKS].mean(), sd=frame[TASKS].std(),
                             min=frame[TASKS].min(), max=frame[TASKS].max(),
                             fraction_zero=(frame[TASKS] == 0).mean(), fraction_one=(frame[TASKS] == 1).mean()))
    desc.to_csv(str(prefix)+'_descriptives.csv', index_label='task')
    np.savez_compressed(str(prefix)+'_bootstrap.npz', metrics=boot_stats, edges=boots[:, I, J])
    return results, boot_stats


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--pilot', action='store_true')
    arg.add_argument('--output', type=Path)
    args = arg.parse_args()
    t0, c0 = time.perf_counter(), time.process_time()
    n_boot = 100 if args.pilot else 5000
    output = args.output or BASE/('pilot' if args.pilot else 'analysis')
    output.mkdir(parents=True, exist_ok=True)
    hfixed = load('human_fixed_accuracy.csv', 'worker_id')
    havail = load('human_available_accuracy.csv', 'worker_id')
    hconstruct = load('human_fixed_construct.csv', 'worker_id')
    hnative = load('human_native_reference.csv', 'worker_id')
    lstrict = load('llm_shared_accuracy.csv', 'model')
    lconstruct = load('llm_shared_construct.csv', 'model')
    primary20 = lstrict.primary20.astype(str).str.lower().isin(['true', '1'])
    primary_names = lstrict.index[primary20]
    assert len(primary_names) == 20 and len(lstrict) == 55
    native_h = complete(hconstruct).index
    native_l = complete(lconstruct).index
    runs = [
        ('fixed_accuracy_55', hfixed, lstrict),
        ('fixed_accuracy_20', hfixed, lstrict.loc[primary_names]),
        ('available_accuracy_55', havail, lstrict),
        ('full_flanker_accuracy_55', load('human_fixed_accuracy_full_flanker.csv', 'worker_id'),
         load('llm_shared_accuracy.csv', 'model', True)),
        ('fixed_construct_observed', hconstruct, lconstruct),
        ('fixed_construct_primary20', hconstruct, lconstruct.loc[primary_names]),
        ('accuracy_native_complete_cases', hfixed.loc[native_h], lstrict.loc[native_l]),
        ('human_native_reference', hnative, None),
    ]
    all_results, table = {}, []
    for k, (name, human, llm) in enumerate(runs):
        h, hb = population(human, 'human', name, output, n_boot, SEED+2*k)
        item = dict(human=h)
        for group, pop in [('human', h)]:
            row = dict(analysis=name, population=group, n=pop['n'], families=pop['families'],
                       partition_tail=pop['partition_tail'], partition_rank=pop['proposed_rank'])
            for metric, value in pop['metrics'].items():
                row.update({metric: value['value'], metric+'_lower': value['ci95'][0], metric+'_upper': value['ci95'][1]})
            table.append(row)
        if llm is not None:
            l, lb = population(llm, 'llm', name, output, n_boot, SEED+2*k+1)
            item['llm'] = l
            row = dict(analysis=name, population='llm', n=l['n'], families=l['families'],
                       partition_tail=l['partition_tail'], partition_rank=l['proposed_rank'])
            for metric, value in l['metrics'].items():
                row.update({metric: value['value'], metric+'_lower': value['ci95'][0], metric+'_upper': value['ci95'][1]})
            table.append(row)
            paired_valid = np.isfinite(hb).all(axis=1) & np.isfinite(lb).all(axis=1)
            differences = hb[paired_valid] - lb[paired_valid]
            item['human_minus_llm'] = dict(n_boot_valid=int(paired_valid.sum()),
                metrics={metric: dict(value=h['metrics'][metric]['value']-l['metrics'][metric]['value'],
                                      ci95=np.quantile(differences[:, z], [.025, .975]).tolist())
                         for z, metric in enumerate(METRICS)})
        all_results[name] = item
    (output/'summary.json').write_text(json.dumps(all_results, indent=2, allow_nan=False)+'\n')
    pd.DataFrame(table).to_csv(output/'summary.csv', index=False)
    # A previous exec reported the same implausible getrusage peak for very
    # different workloads. Retain that reading and separately report VmHWM.
    proc_status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    runtime = dict(pilot=args.pilot, n_boot=n_boot, analyses=len(runs),
                   wall_seconds=time.perf_counter()-t0, cpu_seconds=time.process_time()-c0,
                   max_rss_kib=int(proc_status['VmHWM'].split()[0]),
                   getrusage_maxrss_reported_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   cpu_threads=1, gpu=0, slurm_jobs=0,
                   scope='process runtime after imports, including output serialization')
    (output/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    print(json.dumps(runtime, indent=2))


if __name__ == '__main__':
    main()
