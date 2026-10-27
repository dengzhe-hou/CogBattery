"""CPU-only integration of frozen human54 endpoint and structure tables.

This module never opens model response files. Scoring is a separate, single pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .model_metadata import annotate_score_frame
from .analysis_statistics import (SEED, PANEL_DRAWS, PROFILE_DRAWS, cluster_mean,
                         direction_counts, profile_bootstrap, interval)
from scripts.reanalysis.public_human_structure_20261015 import analyze as ball_stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/reanalysis/human54_layers_20261019'
NATIVE = {'deepseek-r1:7b', 'deepseek-r1:14b', 'qwen3:4b'}
GROUPS = {**ball_stats.GROUPS, 'drm_false_memory': 'EM'}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    ball_stats.save_json(path, value)


def panel_names(models):
    spec = json.loads((ROOT/'scripts/experiments/shortcut_fix_20261011/SPEC.json').read_text())
    primary = set(spec['main20']) - {'tinyllama:1.1b'}
    assert len(primary) == 19 and primary <= set(models)
    return {'all54': set(models), 'primary19': primary, 'direct51': set(models)-NATIVE}


def refs_for_rule(references, rule):
    """Layer1 retains the source endpoint; R2 additionally has its P9 reference.

    Human missing-as-wrong is explicitly paired with model sensitivity for SRO
    layer2 only. It does not silently replace the source reference in layer1.
    """
    # The referenced original SRO cell builder includes missed responses as
    # errors for layer1 DS/ST/FL. Proposal 9 changes layer2 denominators only.
    source_all_trials = {'digit_span', 'stroop', 'flanker'}
    selected = [r for r in references if r['rule'] == 'both' or
                r['rule'] == ('invalid_wrong' if r['task'] in source_all_trials else 'valid_only')]
    if rule == 'nb_r2':
        overrides = {(r['task'], r.get('endpoint', r.get('profile')), r.get('cohort', 'primary')): r
                     for r in references if r['rule'] == 'nb_r2'}
        selected = [overrides.get((r['task'], r.get('endpoint', r.get('profile')), r.get('cohort', 'primary')), r)
                    for r in selected]
    return selected


def endpoint_comparisons(endpoints, refs, families, panels, destination):
    by_model, summary, directions, unmatched = [], [], [], []
    for (task, rule, endpoint), group in endpoints.groupby(['task', 'rule', 'endpoint'], sort=True):
        matching = [r for r in refs_for_rule(refs, rule) if r['task'] == task and r['endpoint'] == endpoint]
        if not matching:
            unmatched.append({'task': task, 'rule': rule, 'endpoint': endpoint, 'status': 'model_only'})
            continue
        for ref in matching:
            human = ref['value']
            if human is None:
                continue
            human_values = np.array([v for v in ref.get('values', []) if v is not None], float)
            cohort = ref.get('cohort', 'primary')
            base = dict(task=task, rule=rule, endpoint=endpoint, human_cohort=cohort,
                        human_mean=human, human_n=ref.get('n'))
            for row in group.to_dict('records'):
                value = row['value']
                finite = pd.notna(value)
                pct = (100*(np.sum(human_values < value)+.5*np.sum(human_values == value))/len(human_values)
                       if finite and len(human_values) else None)
                item = dict(row, **{k: v for k, v in base.items() if k not in row},
                            model_minus_human=float(value-human) if finite else None,
                            human_percentile_mid_distribution=pct,
                            native_thinking=row['model'] in NATIVE)
                if ref.get('se') is not None:
                    item['within_human_mean_1_96se'] = bool(abs(value-human) <= 1.96*ref['se']) if finite else None
                by_model.append(item)
            for panel, names in panels.items():
                selected = group[group.model.isin(names) & group.value.notna() & group.included.eq(True)]
                for family_kind in ('merged', 'raw'):
                    f = [families[family_kind][m] for m in selected.model]
                    if len(selected):
                        stats, _ = cluster_mean(selected.value.to_numpy()-human, f)
                        stats['model_mean'] = float(selected.value.mean())
                    else:
                        stats = dict(mean=None, model_mean=None, models=0, families=0,
                                     family_equal_mean=None, lower=None, upper=None, n_valid=0)
                    item = dict(base, panel=panel, family_mapping=family_kind,
                                family_mapping_role='primary' if family_kind == 'merged' else 'sensitivity',
                                statistic='model_minus_human', **stats,
                                interval='5000 whole-family bootstrap; 95% percentile; human mean fixed',
                                seed=SEED, coverage_threshold=.8 if rule == 'valid_only' else None)
                    if ref.get('se') is not None:
                        item['within_human_mean_1_96se_models'] = int((abs(selected.value-human) <= 1.96*ref['se']).sum())
                    summary.append(item)
                    directions.append(dict(base, panel=panel, family_mapping=family_kind,
                                           contrast='model_minus_human',
                                           **direction_counts(selected.value.to_numpy()-human, f)))
    pd.DataFrame(by_model).to_csv(destination/'model_human_endpoints.csv', index=False)
    pd.DataFrame(summary).to_csv(destination/'panel_differences.csv', index=False)
    save_json(destination/'model_human_direction_counts.json', directions)
    pd.DataFrame(unmatched).to_csv(destination/'model_only_endpoints.csv', index=False)
    return len(by_model), len(summary)


def behavior_directions(endpoints, families, panels, destination):
    """Direction endpoints are explicitly marked by the task scoring module."""
    if 'is_direction' not in endpoints:
        return
    rows = []
    for key, group in endpoints[endpoints.is_direction.eq(True)].groupby(['task', 'rule', 'endpoint'], sort=True):
        for panel, names in panels.items():
            selected = group[group.model.isin(names) & group.value.notna() & group.included.eq(True)]
            for mapping in ('merged', 'raw'):
                rows.append(dict(zip(('task', 'rule', 'endpoint'), key), panel=panel,
                                 family_mapping=mapping,
                                 **direction_counts(selected.value.to_numpy(),
                                                    [families[mapping][m] for m in selected.model])))
    save_json(destination/'behavior_direction_counts.json', rows)
    perfect = []
    fb = endpoints[endpoints.task.eq('false_belief') & endpoints.endpoint.str.endswith('|all_correct')]
    for (rule, endpoint), group in fb.groupby(['rule', 'endpoint'], sort=True):
        for panel, names in panels.items():
            selected = group[group.model.isin(names) & group.value.notna() & group.included.eq(True)]
            perfect.append(dict(rule=rule, endpoint=endpoint, panel=panel,
                                all_correct_models=int(selected.value.eq(1).sum()),
                                included_models=len(selected)))
    pd.DataFrame(perfect).to_csv(destination/'fb_all_correct_counts.csv', index=False)


def structural_population(frame, tasks, kind, family, output, seed):
    """Reuse existing SRO/Ball statistics and exact whole-cluster bootstrap."""
    if len(tasks) == 5:
        from scripts.reanalysis.sro_structure_20261015 import analyze as stat
        metrics = stat.METRICS
        calc = stat.statistics
    else:
        stat = ball_stats
        metrics = stat.METRICS
        _, _, within = stat.mask_for(tasks, GROUPS)
        calc = lambda corr: stat.metrics(corr, within)
    output.mkdir(parents=True, exist_ok=True)
    selected = frame.dropna(subset=tasks)
    displayed_scores = annotate_score_frame(selected, tasks) if kind == 'llm' else selected
    displayed_scores.to_csv(output/f'{kind}_scores.csv')
    x = selected[tasks].to_numpy(float)
    result = dict(n_available=len(frame), n_complete=len(selected), tasks=tasks,
                  n_by_task=frame[tasks].notna().sum().to_dict(), seed=seed,
                  master_seed=SEED, draws=PANEL_DRAWS)
    if len(x) <= len(tasks) or np.any(np.std(x, axis=0) <= 1e-12):
        result.update(status='not_estimable', reason='too few complete cases or constant task',
                      sd=selected[tasks].std().to_dict())
        return result, None
    f = selected['family'].to_numpy() if family else None
    corr = np.corrcoef(x, rowvar=False)
    boots = stat.bootstrap(x, f, PANEL_DRAWS, seed)
    if isinstance(boots, tuple):
        boots = boots[0]
    good = np.isfinite(boots).all(axis=(1, 2))
    bs = np.full((PANEL_DRAWS, len(metrics)), np.nan)
    bs[good] = calc(boots[good])
    point = calc(corr)
    spearman = np.corrcoef(rankdata(np.round(x, 10), axis=0), rowvar=False)
    i, j = np.triu_indices(len(tasks), 1)
    part, tail = (stat.partitions(corr) if len(tasks) == 5 else
                  (stat.partitions(corr, tasks, GROUPS), None))
    if len(tasks) == 3:
        delta = point[metrics.index('delta')]
        tail = sum(p['delta'] >= delta-1e-12 for p in part)/len(part)
        for p in part:
            p['rank'] = 1+sum(q['delta'] > p['delta']+1e-12 for q in part)
    result.update(status='ok', families=len(set(f)) if f is not None else None,
                  n_boot_valid=int(good.sum()), partition_tail=tail,
                  partition_rank=next(p['rank'] for p in part if p['proposed']),
                  metrics={m: {'value': float(point[k]), 'ci95': [interval(bs[:, k])['lower'], interval(bs[:, k])['upper']]}
                           for k, m in enumerate(metrics)},
                  spearman_metrics=dict(zip(metrics, calc(spearman).tolist())),
                  eigenvalues=np.linalg.eigvalsh(corr)[::-1].tolist())
    pd.DataFrame(corr, index=tasks, columns=tasks).to_csv(output/f'{kind}_pearson.csv')
    pd.DataFrame(spearman, index=tasks, columns=tasks).to_csv(output/f'{kind}_spearman.csv')
    pd.DataFrame(part).to_csv(output/f'{kind}_partitions.csv', index=False)
    pd.DataFrame([dict(task_a=tasks[a], task_b=tasks[b], pearson=corr[a, b],
                       spearman=spearman[a, b], **interval(boots[:, a, b]))
                  for a, b in zip(i, j)]).to_csv(output/f'{kind}_edges.csv', index=False)
    selected[tasks].describe().T.to_csv(output/f'{kind}_descriptives.csv')
    np.savez_compressed(output/f'{kind}_bootstrap.npz', metrics=bs, edges=boots[:, i, j])
    return result, bs


def run_structure(model_scores, human_matrices, families, panels, destination):
    results, table, cache = {}, [], {}
    seeds = np.random.SeedSequence(SEED).generate_state(2).tolist()
    expanded = [dict(hm, rule=rule) for hm in human_matrices
                for rule in (('valid_only', 'invalid_wrong') if hm['rule'] == 'both' else (hm['rule'],))]
    for hm in expanded:
        name, rule, tasks = hm['name'], hm['rule'], hm['columns']
        structure, representation = hm['structure'], hm['representation']
        hframe = pd.DataFrame(hm['values'], columns=tasks, dtype=float)
        matching = model_scores[(model_scores.structure == structure) &
                                (model_scores.representation == representation) & (model_scores.rule == rule)]
        if matching.empty:
            raise ValueError(f'No matching model structure: {name}/{rule}/{representation}')
        matching = matching.copy()
        matching.loc[~matching.included.eq(True), 'value'] = np.nan
        mframe = matching.pivot(index='model', columns='task', values='value').reindex(columns=tasks)
        hkey = (name, rule)
        if hkey not in cache:
            archived = destination.parent/'human_side/structures'/f'{name}_{rule}'
            if (archived/'summary.json').exists():
                hs = json.loads((archived/'summary.json').read_text())
                hb = np.load(archived/'human_bootstrap.npz')['metrics'] if hs['status'] == 'ok' else None
                cache[hkey] = (hs, hb)
            else:
                raise FileNotFoundError(f'Human structure must be archived first: {archived}')
        hs, hb = cache[hkey]
        for panel, names in panels.items():
            for mapping in ('merged', 'raw'):
                run_name = f'{name}_{rule}_{panel}_{mapping}'
                frame = mframe.loc[mframe.index.isin(names)].copy()
                frame['family'] = [families[mapping][m] for m in frame.index]
                ls, lb = structural_population(frame, tasks, 'llm', True, destination/run_name, seeds[1])
                item = dict(human=hs, llm=ls, panel=panel, family_mapping=mapping,
                            structure=structure, representation=representation, rule=rule,
                            human_matrix=name, coverage_threshold=.8 if rule == 'valid_only' else None)
                if hb is not None and lb is not None:
                    diff = hb-lb
                    metric_names = list(hs['metrics'])
                    item['human_minus_llm'] = {
                        m: dict(value=hs['metrics'][m]['value']-ls['metrics'][m]['value'],
                                **interval(diff[:, k])) for k, m in enumerate(metric_names)}
                results[run_name] = item
                save_json(destination/run_name/'summary.json', item)
                row = dict(analysis=run_name, panel=panel, family_mapping=mapping,
                           rule=rule, structure=structure, representation=representation)
                for label, pop in [('human', hs), ('llm', ls)]:
                    row[label+'_n'] = pop['n_complete']
                    row[label+'_status'] = pop['status']
                    row[label+'_partition_tail'] = pop.get('partition_tail')
                    for m, v in pop.get('metrics', {}).items():
                        row[label+'_'+m] = v['value']
                        row[label+'_'+m+'_lower'], row[label+'_'+m+'_upper'] = v['ci95']
                if 'human_minus_llm' in item:
                    d = item['human_minus_llm']['delta']
                    row.update(delta_difference=d['value'], delta_difference_lower=d['lower'],
                               delta_difference_upper=d['upper'])
                table.append(row)
    save_json(destination/'summary.json', results)
    pd.DataFrame(table).to_csv(destination/'summary.csv', index=False)
    return len(results)


def profile_comparisons(profiles, refs, blocks, destination):
    results = []
    for (task, rule, name, model), group in profiles.groupby(['task', 'rule', 'profile', 'model'], sort=True):
        matches = [r for r in refs_for_rule(refs, rule) if r['task'] == task and r['profile'] == name]
        if not matches:
            raise ValueError(f'No human profile for {task}/{rule}/{name}')
        if group.cell.duplicated().any():
            raise ValueError(f'Duplicate model profile cell {task}/{name}/{model}')
        group = group.set_index('cell')
        for ref in matches:
            missing = set(ref['cells'])-set(group.index)
            if missing:
                raise ValueError(f'Model profile lacks planned cells: {task}/{name}/{model}: {missing}')
            ordered = group.loc[ref['cells']]
            args = {}
            if task == 'n_back':
                selected = blocks[(blocks.model == model) & (blocks.rule == rule)].copy()
                if set(selected.episode_id) != set(ref['unit_ids']):
                    raise ValueError('Paired n-back human/model episode identities differ')
                selected['cell'] = selected.load.astype(str)+'|'+selected.kind
                n = selected.pivot(index='episode_id', columns='cell', values='match_n')
                d = selected.pivot(index='episode_id', columns='cell', values='denominator')
                # Out-of-load entries are structural zeros in these unit matrices.
                args['model_numerator'] = n.reindex(index=ref['unit_ids'], columns=ref['cells']).fillna(0).to_numpy()
                args['model_denominator'] = d.reindex(index=ref['unit_ids'], columns=ref['cells']).fillna(0).to_numpy()
            calculated = profile_bootstrap(ref['unit_numerators'], ref['unit_denominators'],
                                            ref['unit_strata'], ordered.value.to_numpy(), **args)
            np.testing.assert_allclose(calculated['human_values'], np.asarray(ref['values'], float), equal_nan=True, atol=1e-12)
            np.testing.assert_allclose(calculated['model_values'], ordered.value.to_numpy(), equal_nan=True, atol=1e-12)
            coverage = ordered.n_valid.sum()/ordered.n_planned.sum()
            results.append(dict(task=task, rule=rule, profile=name, model=model,
                                human_cohort=ref.get('cohort', 'primary'),
                                native_thinking=model in NATIVE,
                                coverage=float(coverage),
                                included=bool(rule != 'valid_only' or coverage >= .8),
                                **calculated))
    save_json(destination/'profile_comparisons.json', results)
    flat = []
    for result in results:
        row = {k:v for k,v in result.items() if k not in ('human_values', 'model_values', 'mae_ci', 'r_ci')}
        for metric in ('mae', 'r'):
            row.update({metric+'_'+k:v for k,v in result[metric+'_ci'].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(destination/'profile_comparisons.csv', index=False)
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError(f'{key}=1 required')
    start, cpu = time.monotonic(), time.process_time()
    output = args.output
    h = json.loads((output/'human_side/references.json').read_text())
    families = json.loads((ROOT/'results/reanalysis/aplus_20260718/family_map.json').read_text())
    endpoints = pd.read_csv(output/'layer1/endpoints.csv')
    profiles = pd.read_csv(output/'layer1/profiles.csv', dtype={'cell': str})
    blocks = pd.read_csv(output/'layer1/nb_block_counts.csv')
    structure = pd.read_csv(output/'layer2/structure_scores.csv')
    models = sorted(endpoints.model.unique())
    if len(models) != 54 or set(models) != set(families['merged'])-{'tinyllama:1.1b'}:
        raise ValueError('Expected the complete frozen 54-model panel')
    panels = panel_names(models)
    (output/'layer1').mkdir(exist_ok=True)
    save_json(output/'analysis_inventory.json', dict(master_seed=SEED, panel_draws=PANEL_DRAWS,
        profile_draws=PROFILE_DRAWS, panels={p:sorted(v) for p,v in panels.items()},
        families=families, endpoint_rule='source-native human reference in layer1; paired rules in SRO layer2',
        coverage_threshold=.8, human_percentile='100*(number below + 0.5*number equal)/N',
        intervals='descriptive, uncorrected 95% percentile intervals; no hypothesis tests',
        limitations=['One saved model response per trial, not a model response distribution.',
                     'No combined human-likeness ranking or fictitious cross-cohort correlation.',
                     'Native thinking: 8192 total tokens except OSPAN 128; direct answers 16–1024.',
                     'Fixed runtime invalid-history handling cannot be corrected in analysis.']))
    n_pairs, n_panels = endpoint_comparisons(endpoints, h['endpoints'], families, panels, output/'layer1')
    behavior_directions(endpoints, families, panels, output/'layer1')
    n_profiles = profile_comparisons(profiles, h['profiles'], blocks, output/'layer1')
    print(json.dumps(dict(stage='layer1_complete', endpoint_rows=n_pairs, panel_rows=n_panels, profiles=n_profiles)), flush=True)
    n_structures = run_structure(structure, h['matrices'], families, panels, output/'layer2')
    proc = dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    runtime = dict(wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
                   max_rss_kib=int(proc['VmHWM'].split()[0]), cpu_threads=1, gpu=0, model_calls=0,
                   endpoint_rows=n_pairs, panel_rows=n_panels, profile_rows=n_profiles, structures=n_structures)
    save_json(output/'analysis_runtime.json', runtime)
    print(json.dumps(runtime), flush=True)


if __name__ == '__main__':
    main()
