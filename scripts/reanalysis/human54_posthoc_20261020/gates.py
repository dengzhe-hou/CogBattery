"""Comprehension gates and scale (spec section E). Saved derived tables only; CPU.

E1 gates per task; E2 headline endpoints by stratum; E3 held-out gate versions;
E4 mixed models on log parameters; E5 capability correlations; E7 exploratory
direction counts over all human-paired endpoints. The figure (E6) is drawn by
gen_h54_scale_gates.py from the saved outputs.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scripts.reanalysis.human54_layers_20261019.analysis_statistics import cluster_mean
from scripts.reanalysis.human54_fullbattery_20261019.pca import family_bootstrap_counts
from . import data as D

ROOT = Path(__file__).resolve().parents[3]
LAYER1 = ROOT / 'results/reanalysis/human54_layers_20261019/layer1'
ALPHA = .05
LARGE_B = 27.0
STRICT = .8
# SeedSequence(20261018).spawn(12) child 11 is unused by run.py (children 0, 1, 6-10).
SEED_CAP = int(np.random.SeedSequence(20261018).spawn(12)[11].generate_state(1)[0])

# Headline endpoints: (task, endpoint, direction of the reported human-model gap).
HEADLINES = {
    'NB_control_fa': ('n_back', 'load_2_control', 'greater'),
    'ST_interference': ('stroop', 'error_interference', 'greater'),
    'CAL_auroc': ('confidence_calibration', 'type2_auc_fixed_direction', 'less'),
    'WAG_constant': ('post_decision_wagering', 'high_wager_rate', None),
}


def load():
    e = pd.read_csv(LAYER1 / 'endpoints.csv', low_memory=False)
    e = e[e.rule == 'valid_only']
    p = pd.read_csv(LAYER1 / 'profiles.csv', low_memory=False)
    p = p[p.rule == 'valid_only']
    nb = pd.read_csv(LAYER1 / 'nb_block_counts.csv', low_memory=False)
    nb = nb[nb.rule == 'valid_only']
    paired = pd.read_csv(LAYER1 / 'model_human_endpoints.csv', low_memory=False)
    paired = paired[(paired.rule == 'valid_only') & (paired.human_cohort == 'primary')]
    models = pd.read_csv(ROOT / 'results/reanalysis/human54_layers_20261019/coverage/models.csv', index_col='model')
    return e, p, nb, paired, models


def value(e, model, task, endpoint, field='value'):
    row = e[(e.model == model) & (e.task == task) & (e.endpoint == endpoint)]
    if len(row) != 1:
        raise ValueError(f'Expected one endpoint row for {model}/{task}/{endpoint}, got {len(row)}')
    return float(row[field].iloc[0])


def binomial_gate(acc, n, chance):
    k = int(round(acc * n))
    return dict(measure=acc, n=int(n), k=k, chance=chance,
                p=float(binomtest(k, int(n), chance, alternative='greater').pvalue))


def sdt_gate(hits, n_signal, fas, n_noise):
    table = [[hits, n_signal - hits], [fas, n_noise - fas]]
    hit, fa = hits / n_signal, fas / n_noise
    return dict(measure=(hit + 1 - fa) / 2, hit_rate=hit, fa_rate=fa, n_signal=int(n_signal), n_noise=int(n_noise),
                p=float(fisher_exact(table, alternative='greater')[1]))


def nb_counts(nb, model, load=2, half=None):
    sub = nb[(nb.model == model) & (nb.load == load)]
    if half is not None:
        episodes = sorted(sub.episode_id.unique())
        keep = set(episodes[half::2])
        sub = sub[sub.episode_id.isin(keep)]
    g = sub.groupby('kind')[['match_n', 'denominator']].sum()
    return g


def gates(e, p, nb, models):
    rows = []
    for m in models.index:
        def add(task, rule, info):
            passed = info['p'] < ALPHA if 'p' in info else info['measure'] >= .5
            rows.append(dict(model=m, task=task, rule=rule, passed=bool(passed),
                             strict_passed=bool(info['measure'] >= STRICT), **info))
        g = nb_counts(nb, m)
        add('n_back', 'sdt_fisher', sdt_gate(g.match_n['target'], g.denominator['target'],
                                            g.match_n[['lure', 'control']].sum(), g.denominator[['lure', 'control']].sum()))
        for task, chance in (('flanker', .5), ('stroop', 1 / 3)):
            add(task, 'binomial', binomial_gate(value(e, m, task, 'congruent_accuracy'),
                                                value(e, m, task, 'congruent_accuracy', 'n_scored'), chance))
        hit, fa = value(e, m, 'go_nogo', 'hit_rate'), value(e, m, 'go_nogo', 'false_alarm_rate')
        ng, nn = value(e, m, 'go_nogo', 'hit_rate', 'n_scored'), value(e, m, 'go_nogo', 'false_alarm_rate', 'n_scored')
        add('go_nogo', 'sdt_fisher', sdt_gate(round(hit * ng), ng, round(fa * nn), nn))
        old = p[(p.model == m) & (p.task == 'drm_false_memory') & (p.cell == 'no_warning|old')].iloc[0]
        new = p[(p.model == m) & (p.task == 'drm_false_memory') & (p.cell == 'no_warning|newunrelated')].iloc[0]
        add('drm_false_memory', 'sdt_fisher', sdt_gate(round(old.value * old.n_scored), old.n_scored,
                                                      round(new.value * new.n_scored), new.n_scored))
        add('source_monitoring', 'binomial', binomial_gate(value(e, m, 'source_monitoring', 'structure_ball'),
                                                           value(e, m, 'source_monitoring', 'structure_ball', 'n_scored'), .25))
        add('epitome_tom', 'binomial', binomial_gate(value(e, m, 'epitome_tom', 'control|1_accuracy'),
                                                     value(e, m, 'epitome_tom', 'control|1_accuracy', 'n_scored'), .5))
        for task in ('confidence_calibration', 'post_decision_wagering'):
            add(task, 'binomial', binomial_gate(value(e, m, task, 'accuracy'), value(e, m, task, 'accuracy', 'n_scored'), .5))
        for task, endpoint, scale in (('digit_span', 'forward_length_3', 1), ('operation_span', 'length_3_position_accuracy', 1),
                                      ('cvlt_word_list', 'learning_1', 16), ('false_belief', 'session|TB_accuracy', 1)):
            v = value(e, m, task, endpoint) / scale
            rows.append(dict(model=m, task=task, rule='threshold_0.5', passed=bool(v >= .5), strict_passed=bool(v >= STRICT),
                             measure=v, n=value(e, m, task, endpoint, 'n_scored')))
    return pd.DataFrame(rows)


def stratum_summary(values, human, families, direction, label):
    values = np.asarray(values, float)
    out = dict(stratum=label, n_models=int(len(values)), n_families=int(len(set(families))))
    if len(values) == 0:
        return out
    if direction is None:      # WAG constant wager: proportion only
        const = np.isin(values, (0., 1.))
        fam = pd.Series(const, index=families).groupby(level=0).max()
        return dict(out, constant=int(const.sum()), share=float(const.mean()),
                    families_with_constant=int(fam.sum()))
    stats, _ = cluster_mean(values - human, families)
    sign = 1 if direction == 'greater' else -1
    fam_means = pd.Series(values, index=families).groupby(level=0).mean()
    return dict(out, model_mean=float(values.mean()), human_mean=float(human), difference=stats['mean'],
                lower=stats['lower'], upper=stats['upper'],
                models_in_direction=int(np.sum(sign * (values - human) > 0)),
                families_in_direction=int(np.sum(sign * (fam_means.to_numpy() - human) > 0)))


def headlines(e, paired, models, gate_table, params):
    out = []
    pass_by = gate_table.set_index(['model', 'task'])
    for key, (task, endpoint, direction) in HEADLINES.items():
        rows = e[(e.task == task) & (e.endpoint == endpoint) & (e.included == True)].set_index('model')
        human = float(paired[(paired.task == task) & (paired.endpoint == endpoint)].human_mean.iloc[0]) if direction else None
        for rule in ('passed', 'strict_passed'):
            for label, keep in (('all', lambda m: True),
                                ('gate', lambda m: pass_by.loc[(m, task), rule]),
                                ('large', lambda m: params.params_b[m] >= LARGE_B),
                                ('gate_and_large', lambda m: pass_by.loc[(m, task), rule] and params.params_b[m] >= LARGE_B)):
                if rule == 'strict_passed' and label in ('all', 'large'):
                    continue
                ms = [m for m in rows.index if keep(m)]
                s = stratum_summary(rows.value[ms].astype(float), human, [models.family_merged[m] for m in ms], direction, label)
                out.append(dict(headline=key, gate_rule='p<.05 or >=.5' if rule == 'passed' else '>=.8', **s))
    return pd.DataFrame(out)


def held_out(e, nb, paired, models):
    """Gate and endpoint from disjoint trials (spec E3)."""
    out = []
    human_nb = float(paired[(paired.task == 'n_back') & (paired.endpoint == 'load_2_control')].human_mean.iloc[0])
    vals, fams = [], []
    for m in models.index:
        g0, g1 = nb_counts(nb, m, half=0), nb_counts(nb, m, half=1)
        gate = sdt_gate(g0.match_n['target'], g0.denominator['target'],
                        g0.match_n[['lure', 'control']].sum(), g0.denominator[['lure', 'control']].sum())
        if gate['p'] < ALPHA:
            vals.append(g1.match_n['control'] / g1.denominator['control'])
            fams.append(models.family_merged[m])
    out.append(dict(headline='NB_control_fa', version='gate half0 blocks -> endpoint half1 blocks',
                    **stratum_summary(vals, human_nb, fams, 'greater', 'heldout_gate')))
    human_cal = float(paired[(paired.task == 'confidence_calibration') & (paired.endpoint == 'type2_auc_fixed_direction')].human_mean.iloc[0])
    for gpos, epos in ((0, 1), (1, 0)):
        vals, fams = [], []
        for m in models.index:
            acc = value(e, m, 'confidence_calibration', f'position_{gpos}|accuracy')
            n = value(e, m, 'confidence_calibration', f'position_{gpos}|accuracy', 'n_scored')
            auc = value(e, m, 'confidence_calibration', f'position_{epos}|type2_auc_fixed_direction')
            if binomial_gate(acc, n, .5)['p'] < ALPHA and np.isfinite(auc):
                vals.append(auc)
                fams.append(models.family_merged[m])
        out.append(dict(headline='CAL_auroc', version=f'gate position {gpos} accuracy -> endpoint position {epos} AUROC',
                        **stratum_summary(vals, human_cal, fams, 'less', 'heldout_gate')))
    vals, fams = [], []
    for m in models.index:
        def pooled(grammar, endpoint):
            parts = [(value(e, m, 'post_decision_wagering', f'{grammar}_{d}|{endpoint}'),
                      value(e, m, 'post_decision_wagering', f'{grammar}_{d}|{endpoint}', 'n_scored')) for d in ('forward', 'reverse')]
            return sum(v * n for v, n in parts) / sum(n for _, n in parts), sum(n for _, n in parts)
        acc, n = pooled('A', 'accuracy')
        high, _ = pooled('B', 'high_wager_rate')
        if binomial_gate(acc, n, .5)['p'] < ALPHA:
            vals.append(high)
            fams.append(models.family_merged[m])
    out.append(dict(headline='WAG_constant', version='gate grammar A accuracy -> endpoint grammar B constancy',
                    **stratum_summary(vals, None, fams, None, 'heldout_gate')))
    return pd.DataFrame(out)


def mixed_models(e, models, params):
    out = []
    logp = np.log10(params.params_b)
    mu, sd = logp.mean(), logp.std(ddof=0)
    for key, (task, endpoint, direction) in HEADLINES.items():
        rows = e[(e.task == task) & (e.endpoint == endpoint) & (e.included == True)].set_index('model')
        d = pd.DataFrame({'y': rows.value.astype(float), 'zlogp': (logp[rows.index] - mu) / sd,
                          'native': models.native_thinking[rows.index].astype(int),
                          'family': models.family_merged[rows.index]})
        at = {f'{b}B': (np.log10(b) - mu) / sd for b in (7, 70)}
        if direction is None:
            d['y'] = np.isin(d.y, (0., 1.)).astype(float)
            fit = smf.gee('y ~ zlogp + native', 'family', d, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            kind = 'GEE logistic, family clusters, exchangeable'
        else:
            fit = smf.mixedlm('y ~ zlogp + native', d, groups=d['family']).fit(reml=True)
            kind = 'MixedLM REML, random intercept per merged family'
        ci = fit.conf_int()
        pred = {k: float(fit.predict(pd.DataFrame({'zlogp': [z], 'native': [0]})).iloc[0]) for k, z in at.items()}
        out.append(dict(headline=key, model=kind, n=int(len(d)), n_families=int(d.family.nunique()),
                        slope_per_sd_logp=float(fit.params['zlogp']), slope_lower=float(ci.loc['zlogp', 0]),
                        slope_upper=float(ci.loc['zlogp', 1]), slope_per_10x=float(fit.params['zlogp'] / sd),
                        native=float(fit.params['native']), native_lower=float(ci.loc['native', 0]),
                        native_upper=float(ci.loc['native', 1]), predicted_7B=pred['7B'], predicted_70B=pred['70B'],
                        logp_sd=float(sd), converged=bool(getattr(fit, 'converged', True))))
    return pd.DataFrame(out)


def capability_correlations(e, models, params):
    X, meta = D.load_matrix()
    fams = [meta.family_merged[m] for m in X.index]
    Z = ((X - X.mean()) / X.std(ddof=0)).to_numpy()
    logp = np.log10(params.params_b.reindex(X.index).to_numpy())
    _, codes, counts = family_bootstrap_counts(fams, 5000, SEED_CAP)

    def loo_pc1(j):
        keep = [k for k in range(Z.shape[1]) if k != j]
        Zk = (Z[:, keep] - Z[:, keep].mean(0)) / Z[:, keep].std(0)
        vals, vecs = np.linalg.eigh(Zk.T @ Zk / len(Zk))
        v = vecs[:, -1] * np.sign(vecs[:, -1].sum())
        return Zk @ v

    def corr_ci(x, y):
        point = spearmanr(x, y)[0]
        boot = [spearmanr(np.repeat(x, c[codes]), np.repeat(y, c[codes]))[0] for c in counts]
        return float(point), float(np.nanquantile(boot, .025)), float(np.nanquantile(boot, .975))

    rows = []
    tasks = list(X.columns)
    targets = [(D.ABBR[t], X[t].to_numpy(), tasks.index(t)) for t in tasks]
    for key, (task, endpoint, _) in HEADLINES.items():
        r = e[(e.task == task) & (e.endpoint == endpoint)].set_index('model').value.reindex(X.index).astype(float)
        y = r.to_numpy()
        if key == 'WAG_constant':
            y = np.isin(y, (0., 1.)).astype(float)
        targets.append((key, y, tasks.index(task)))
    for name, y, j in targets:
        a = corr_ci(y, logp)
        b = corr_ci(y, loo_pc1(j))
        rows.append(dict(score=name, rho_log_params=a[0], rho_log_params_lower=a[1], rho_log_params_upper=a[2],
                         rho_loo_pc1=b[0], rho_loo_pc1_lower=b[1], rho_loo_pc1_upper=b[2], n=int(len(y))))
    return pd.DataFrame(rows)


def exploratory(paired, gate_table, params):
    pass_by = gate_table.set_index(['model', 'task']).passed
    d = paired[paired.included == True].copy()
    d['gate'] = [bool(pass_by[(m, t)]) for m, t in zip(d.model, d.task)]
    d['large'] = [params.params_b[m] >= LARGE_B for m in d.model]
    rows = []
    for (task, endpoint), sub in d.groupby(['task', 'endpoint']):
        row = dict(task=D.ABBR[task], endpoint=endpoint, human_mean=float(sub.human_mean.iloc[0]))
        for label, s in (('all', sub), ('gate', sub[sub.gate]), ('large', sub[sub.large])):
            diff = s.value.astype(float) - s.human_mean
            row.update({f'{label}_n': int(len(s)), f'{label}_mean_diff': float(diff.mean()) if len(s) else np.nan,
                        f'{label}_above': int((diff > 0).sum()), f'{label}_below': int((diff < 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reanalysis/human54_posthoc_20261020_gates')
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(f'{a.output} exists; use a new output directory')
    a.output.mkdir(parents=True)
    e, p, nb, paired, models = load()
    params = D.parameter_table(list(models.index))
    gate_table = gates(e, p, nb, models)
    gate_table.to_csv(a.output / 'gates.csv', index=False)
    summary = gate_table.groupby('task').agg(passed=('passed', 'sum'), strict_passed=('strict_passed', 'sum'), n=('model', 'size'))
    summary.to_csv(a.output / 'gate_counts.csv')
    headlines(e, paired, models, gate_table, params).to_csv(a.output / 'headlines_by_stratum.csv', index=False)
    held_out(e, nb, paired, models).to_csv(a.output / 'headlines_heldout.csv', index=False)
    mixed_models(e, models, params).to_csv(a.output / 'mixed_models.csv', index=False)
    capability_correlations(e, models, params).to_csv(a.output / 'capability_correlations.csv', index=False)
    exploratory(paired, gate_table, params).to_csv(a.output / 'exploratory_paired_endpoints.csv', index=False)
    params.to_csv(a.output / 'model_parameters.csv')
    (a.output / 'run_manifest.json').write_text(json.dumps(dict(
        finished_at=datetime.now().astimezone().isoformat(), spec='review-stage/human54_posthoc_20261020/SPEC.md section E',
        alpha=ALPHA, strict=STRICT, large_b=LARGE_B, seed_capability=SEED_CAP, cluster_seed=20261018,
        hostname=os.uname().nodename, post_hoc=True), indent=2) + '\n')
    print(summary.to_string())


if __name__ == '__main__':
    main()
