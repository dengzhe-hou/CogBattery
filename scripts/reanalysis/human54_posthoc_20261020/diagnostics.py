"""Descriptive diagnostics requested by the independent verification (wf_a8348844-a48).

Reads the saved post hoc outputs and recomputes, with the same seeds, the quantities the
first report left out: concentration of the five-group gain in one family, bootstrap
diagnostics of the disattenuated contrast, the scale-removed correlation matrix, a
per-draw-refit PC1 interval, and a digit-span split sensitivity for the human SRO side.
None of these replaces a primary number. CPU only; seconds on the login node.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from scripts.reanalysis.human54_fullbattery_20261019.pca import TASK_GROUPS, family_bootstrap_counts
from . import data as D
from . import lofo as L
from . import reliability as REL
from . import scale as S
from . import sro_reliability as SRO
from .run import SEEDS

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/reanalysis/human54_posthoc_20261020'


def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return dict(lower=float(np.quantile(v, .025)), upper=float(np.quantile(v, .975)), n_valid=int(len(v)))


def prediction_concentration(out):
    cells = pd.read_csv(out / 'prediction/main_cells.csv')
    net = (cells.se_g - cells.se_group).sum()
    ds = cells.family == 'deepseek-r'
    wag = ds & (cells.target == 'post_decision_wagering')
    res = dict(
        deepseek_r_wag_cell=dict(share_of_baseline_squared_error=float(cells.se_g[wag].sum() / cells.se_g.sum()),
                                 share_of_net_improvement=float((cells.se_g - cells.se_group)[wag].sum() / net)),
        deepseek_r_all_cells=dict(share_of_baseline_squared_error=float(cells.se_g[ds].sum() / cells.se_g.sum()),
                                  share_of_net_improvement=float((cells.se_g - cells.se_group)[ds].sum() / net)))
    for target in ('post_decision_wagering', 'confidence_calibration'):
        sub = cells[(cells.target == target) & ~ds]
        res[f'{D.ABBR[target]}_gain_without_deepseek_r_cell'] = L.gain(sub)
    kept = cells[~ds]
    res['aggregate_without_deepseek_r_cells'] = dict(
        rmse_pc1=L.rmse(kept, 'se_g'), rmse_five_group=L.rmse(kept, 'se_group'), gain=L.gain(kept),
        bootstrap=L.paired_bootstrap(kept, {'gain': ('se_group', 'se_g')}, draws=20000, seed=SEEDS['prediction'])['gain'],
        note='fits unchanged; deepseek-r cells removed at aggregation only')
    return res


def reliability_bootstrap(X, halves, families):
    """Same draws as reliability.analyze (seed and family codes), keeping per-draw detail."""
    tasks, models = list(X.columns), list(X.index)
    forms = {t: REL.forms_matrix(halves, t, models).to_numpy() for t in tasks}
    groups = np.array([TASK_GROUPS[D.ABBR[t]] for t in tasks])
    row, col = np.triu_indices(len(tasks), 1)
    within = groups[row] == groups[col]
    Xv = X.to_numpy(float)
    _, codes, counts = family_bootstrap_counts(families, 5000, SEEDS['reliability'])
    wag = tasks.index('post_decision_wagering')
    raw_delta, star_delta, rel_all, above = [], [], [], []
    pair_counts = np.zeros(len(row), int)
    wag_half_r = []
    has_ds = []
    ds_code = sorted(set(families)).index('deepseek-r')
    for c in counts:
        idx = np.repeat(np.arange(len(models)), c[codes])
        has_ds.append(c[ds_code] > 0)
        R = np.corrcoef(Xv[idx], rowvar=False)
        rel = np.array([REL.reliability(forms[t][idx])['rel'] for t in tasks])
        F = forms['post_decision_wagering'][idx]
        F = F[np.isfinite(F).all(axis=1)]
        wag_half_r.append(np.corrcoef(F, rowvar=False)[0, 1] if F.std(axis=0).min() > 0 else np.nan)
        raw_delta.append(R[row, col][within].mean() - R[row, col][~within].mean())
        rel_all.append(rel)
        if np.all(rel > 0):
            Rs = R / np.sqrt(np.outer(rel, rel))
            star_delta.append(Rs[row, col][within].mean() - Rs[row, col][~within].mean())
            hit = np.abs(Rs[row, col]) > 1
            above.append(int(hit.sum()))
            pair_counts += hit
        else:
            star_delta.append(np.nan)
            above.append(0)
    raw_delta, star_delta, rel_all = np.array(raw_delta), np.array(star_delta), np.array(rel_all)
    defined = np.isfinite(star_delta)
    names = [f'{D.ABBR[tasks[i]]}-{D.ABBR[tasks[j]]}' for i, j in zip(row, col)]
    top = sorted(((int(n), name) for n, name in zip(pair_counts, names) if n), reverse=True)[:5]
    # Point-estimate WAG diagnostics.
    F = forms['post_decision_wagering']
    at_half = int(np.sum(np.all(F == .5, axis=1)))
    keep = [m != 'deepseek-r1:14b' for m in models]
    r_loo = np.corrcoef(F[keep], rowvar=False)[0, 1]
    return dict(
        draws=len(counts), seed=SEEDS['reliability'],
        draws_delta_star_defined=int(defined.sum()),
        draws_with_nonpositive_or_undefined_rel={D.ABBR[t]: int(np.sum(~(rel_all[:, j] > 0))) for j, t in enumerate(tasks)
                                                 if np.sum(~(rel_all[:, j] > 0))},
        raw_delta_all_draws=q(raw_delta), raw_delta_defined_draws=q(raw_delta[defined]),
        delta_star_defined_draws=q(star_delta),
        draws_with_abs_r_star_above_one=int(np.sum(np.array(above) > 0)), pairs_above_one_total=int(sum(above)),
        top_pairs_above_one=top,
        deepseek_r_present_share=dict(all_draws=float(np.mean(has_ds)), defined_draws=float(np.mean(np.array(has_ds)[defined]))),
        wag=dict(half_r=float(np.corrcoef(F, rowvar=False)[0, 1]), half_spearman=float(spearmanr(F[:, 0], F[:, 1])[0]),
                 half_r_interval=q(wag_half_r), models_at_half_in_both_forms=at_half,
                 half_r_without_deepseek_r1_14b=float(r_loo),
                 spearman_brown_without_deepseek_r1_14b=float(2 * r_loo / (1 + r_loo))))


def scale_extras(X, out):
    params = D.parameter_table(list(X.index))
    logp = np.log10(params.params_b.to_numpy())
    E, _ = S.residualize(X.to_numpy(float), logp)
    Z = (E - E.mean(0)) / E.std(0)
    R = pd.DataFrame(Z.T @ Z / len(Z), index=[D.ABBR[t] for t in X.columns], columns=[D.ABBR[t] for t in X.columns])
    R.to_csv(out / 'scale/scale_removed_correlations.csv')
    # PC1 refit on every draw (A1 reported intervals with full-sample scores held fixed).
    fams = [D.family_maps(pd.read_csv(D.FULLBATTERY / 'primary_valid_only/metadata.csv', index_col='model'))['merged23'][m]
            for m in X.index]
    _, codes, counts = family_bootstrap_counts(fams, 5000, SEEDS['scale_corr'])
    Xv = X.to_numpy(float)
    rs = []
    for c in counts:
        idx = np.repeat(np.arange(len(Xv)), c[codes])
        Zb = Xv[idx]
        if np.any(Zb.std(0) == 0):
            rs.append(np.nan)
            continue
        Zb = (Zb - Zb.mean(0)) / Zb.std(0)
        vals, vecs = np.linalg.eigh(Zb.T @ Zb / len(Zb))
        v = vecs[:, -1] * np.sign(vecs[:, -1].sum())
        rs.append(pearsonr(Zb @ v, logp[idx])[0])
    off = R.to_numpy()[np.triu_indices(len(R), 1)]
    return dict(scale_removed_mean_pair_correlation=float(off.mean()),
                pc1_log_params_pearson_refit_each_draw=q(rs), seed=SEEDS['scale_corr'])


def sro_ds_split_sensitivity():
    """Human DS halves by odd/even trial order within direction (not first vs later cell visits)."""
    scr = SRO.screened()
    full = SRO.human_scores(scr).dropna()
    ds = scr['digit_span'].copy()
    ds['direction'] = ds.condition.replace({'reverse': 'backward'})
    ds['cell'] = ds.direction + '|' + ds.num_digits.astype(int).astype(str)
    ds = ds[ds.rt != -1]
    halves = []
    for h in (0, 1):
        part = ds[ds.groupby(['worker_id', 'direction']).cumcount() % 2 == h]
        table = part.groupby(['worker_id', 'cell']).correct.mean().unstack('cell')
        cells = [f'{d}|{n}' for d in ('forward', 'backward') for n in range(3, 10)]
        halves.append(SRO.SRO.weighted(table.reindex(columns=cells), SRO.DS_WEIGHTS, False).reindex(full.index))
    F = np.column_stack(halves)
    rel = REL.reliability(F)
    stored = json.loads((ROOT / 'results/reanalysis/human54_posthoc_20261020_sro/sro_reliability.json').read_text())
    rels = {t: stored['human']['reliability'][t]['rel'] for t in SRO.TASKS}
    rels['digit_span'] = rel['rel']
    Rh = np.corrcoef(full.to_numpy(), rowvar=False)
    rv = np.array([rels[t] for t in SRO.TASKS])
    human_star = SRO._contrast(Rh / np.sqrt(np.outer(rv, rv)))
    return dict(rule='alternate trials by order within person x direction; available-cell fixed weights',
                human_ds_rel=rel['rel'], n=rel['n'], human_delta_star=float(human_star),
                difference_delta_star=float(human_star - stored['model']['delta_star']),
                prespecified=dict(human_ds_rel=stored['human']['reliability']['digit_span']['rel'],
                                  human_delta_star=stored['human']['delta_star'],
                                  difference_delta_star=stored['difference']['delta_star']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'diagnostics.json')
    a = parser.parse_args()
    X, meta = D.load_matrix()
    halves = pd.read_csv(ROOT / 'results/reanalysis/human54_posthoc_20261020_halves/halves.csv')
    fams = [D.family_maps(meta)['merged23'][m] for m in X.index]
    result = dict(prediction=prediction_concentration(OUT), reliability=reliability_bootstrap(X, halves, fams),
                  scale=scale_extras(X, OUT), sro_ds_split=sro_ds_split_sensitivity(),
                  source='wf_a8348844-a48 verification requests; descriptive, post hoc')
    a.output.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(json.dumps(result, indent=2, default=float))


if __name__ == '__main__':
    main()
