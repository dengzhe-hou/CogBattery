"""Self-Regulation Ontology (Eisenberg et al. 2019): human condition effects at the LLM endpoints.

Input: data/raw/self_regulation_ontology/Complete_02-16-2019/Individual_Measures/{digit_span,go_nogo,
adaptive_n_back,stroop,attention_network_task}.csv.gz (one row per trial, first session, N=522).

Trial and participant screening follows the original authors' code:
- Per-task participant QC: selfregulation/utils/data_preparation_utils.py::quality_check (thresholds below),
  applied to the same extracted rows that were saved as Individual_Measures (practice rows included).
- DV computation: expfactory-analysis jspsych_processing.py: practice rows dropped; Stroop, ANT and
  adaptive n-back accuracy computed on trials with a response (rt != -1); go/no-go uses all test trials.
LLM endpoint definitions follow review-stage/human_llm_alignment_20261009/scripts/g1work/g1_extract.py.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hr_common as H  # noqa: E402

H.env_check()
T = H.Timer()
SRC = f'{H.RAW}/self_regulation_ontology/Complete_02-16-2019/Individual_Measures'
OUTD = 'sro'
os.makedirs(f'{H.OUT}/{OUTD}', exist_ok=True)


def load(task):
    d = pd.read_csv(f'{SRC}/{task}.csv.gz', low_memory=False)
    d['correct'] = pd.to_numeric(d['correct'].map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0, 'True': 1.0, 'False': 0.0}))
    assert d['correct'].notna().all(), task
    return d.sort_values(['worker_id', 'time_elapsed'], kind='stable').reset_index(drop=True)


# ---------------- authors' per-task QC (quality_check) ----------------
QC_THRESH = {  # rt, acc, missed, response (from quality_check lookups; defaults 200/.6/.25/.95)
    'digit_span': dict(rt=200, acc=0, missed=.25, response=np.nan),
    'go_nogo': dict(rt=200, acc=.6, missed=1, response=.98),
    'adaptive_n_back': dict(rt=200, acc=.6, missed=.25, response=.95),
    'stroop': dict(rt=200, acc=.6, missed=.25, response=.95),
    'attention_network_task': dict(rt=200, acc=.6, missed=.25, response=.95),
}


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
    df = pd.concat([passed_rt.rename('rt'), passed_acc.rename('acc'), passed_miss.rename('missed'),
                    passed_resp.rename('response')], axis=1).reindex(d.worker_id.unique())
    df = df.astype('boolean').fillna(False).astype(bool)
    ok = df.all(axis=1)
    return set(ok[ok].index), {k: int((~df[k]).sum()) for k in df.columns} | {'failed_any': int((~ok).sum()),
                                                                                'n_workers': int(len(ok))}


qc_report = {}


def screened(task):
    d = load(task)
    keep, rep = qc(task, d)
    qc_report[task] = rep
    return d[d.worker_id.isin(keep)].copy()


# ---------------- digit span ----------------
def digit_span():
    d = screened('digit_span')
    assert set(d.exp_stage) == {'test'}
    d['L'] = d['num_digits'].astype(int)
    d['dir'] = d['condition'].map({'forward': 'forward', 'reverse': 'backward'})
    # authors' span DV: mean num_digits after dropping the first 4 trials of each direction
    d['k'] = d.groupby(['worker_id', 'condition']).cumcount()
    span = d[d.k > 3].groupby(['worker_id', 'dir']).L.mean().unstack()
    cell = d.groupby(['worker_id', 'dir', 'L']).correct.mean().unstack(['dir', 'L'])

    def rng_acc(direction, lo, hi):
        x = d[(d.dir == direction) & d.L.between(lo, hi)]
        return x.groupby('worker_id').correct.mean()

    def matched(lo, hi):
        a, b = {}, {}
        for w, g in d.groupby('worker_id'):
            f = g[g.dir == 'forward'].groupby('L').correct.mean()
            r = g[g.dir == 'backward'].groupby('L').correct.mean()
            Ls = [L for L in f.index.intersection(r.index) if lo <= L <= hi]
            if Ls:
                a[w], b[w] = f[Ls].mean(), r[Ls].mean()
        return pd.Series(a), pd.Series(b)

    res = {}
    fa, ba = matched(3, 7)
    res['ds_fwd_gt_bwd_len_matched_3to7'] = H.contrast(fa, ba) | dict(
        endpoint='per subject: mean over lengths 3-7 present in both directions of per-length accuracy')
    fa2, ba2 = matched(1, 99)
    res['ds_fwd_gt_bwd_len_matched_all'] = H.contrast(fa2, ba2) | dict(endpoint='same, all common lengths')
    res['ds_fwd_gt_bwd_unmatched'] = H.contrast(d[d.dir == 'forward'].groupby('worker_id').correct.mean(),
                                                d[d.dir == 'backward'].groupby('worker_id').correct.mean()) | dict(
        endpoint='overall trial accuracy per direction; adaptive staircase sets the lengths, so this is not a length-matched contrast')
    res['ds_span_fwd_gt_bwd_authors_DV'] = H.contrast(span['forward'], span['backward']) | dict(
        endpoint="authors' DV: mean presented length after dropping first 4 trials (digits)")
    for direction in ('backward', 'forward'):
        res[f'ds_{direction}_short_gt_long'] = H.contrast(rng_acc(direction, 3, 4), rng_acc(direction, 5, 7)) | dict(
            endpoint=f'{direction}: pooled trial accuracy at lengths 3-4 vs 5-7')
    bs, bl = rng_acc('backward', 3, 4), rng_acc('backward', 5, 7)
    fs, fl = rng_acc('forward', 3, 4), rng_acc('forward', 5, 7)
    res['ds_len_x_mode'] = H.contrast(bs - bl, fs - fl) | dict(
        endpoint='(backward 3-4 minus 5-7) vs (forward 3-4 minus 5-7); subjects with all four cells')

    tab = []
    for (direction, L), g in d.groupby(['dir', 'L']):
        per = g.groupby('worker_id').correct.mean()
        tab.append(dict(direction=direction, length=int(L), n_subjects=int(per.size), n_trials=int(len(g)),
                        mean_subject_accuracy=float(per.mean()), pooled_accuracy=float(g.correct.mean())))
    tab = pd.DataFrame(tab)
    tab.to_csv(f'{H.OUT}/{OUTD}/digit_span_by_direction_length.csv', index=False)
    # trial position of each length (staircase): mean trial index within direction
    pos = d.groupby(['dir', 'L']).k.mean().round(2)
    span.to_csv(f'{H.OUT}/{OUTD}/digit_span_subject_span.csv')
    return dict(n_after_qc=int(d.worker_id.nunique()), contrasts=res,
                mean_trial_index_by_length={f'{a}_{b}': float(v) for (a, b), v in pos.items()},
                span_authors_dv=dict(forward=float(span['forward'].mean()), backward=float(span['backward'].mean())),
                notes=['All 28 test trials per participant kept (14 per direction). The authors drop the first 4 trials only '
                       'when averaging span length; that is not a validity rule for trials.',
                       'Lengths follow an adaptive staircase, so short lengths occur mostly early in each direction and '
                       'long lengths only after successes. Per-length accuracy is conditional on reaching that length.'])


# ---------------- adaptive n-back ----------------
def n_back():
    d = screened('adaptive_n_back')
    d = d[d.exp_stage == 'adaptive'].copy()
    d['s'] = d['stim'].str.lower()
    rows = []
    n_check = 0
    for (w, b), g in d.groupby(['worker_id', 'block_num'], sort=False):
        g = g.sort_values('time_elapsed', kind='stable')
        n = int(g.load.iloc[0])
        assert (g.load == n).all()
        s = g.s.tolist()
        key = g.key_press.tolist()
        cr = g.correct_response.tolist()
        corr = g.correct.tolist()
        for i in range(len(s)):
            tgt = i >= n and s[i - n] == s[i]
            assert (cr[i] == 37) == tgt, (w, b, i)
            n_check += 1
            lm1 = (not tgt) and n >= 2 and i >= n - 1 and s[i - (n - 1)] == s[i]
            lp1 = (not tgt) and n >= 2 and i >= n + 1 and s[i - (n + 1)] == s[i]
            rows.append((w, n, i, tgt, lm1, lp1, key[i], corr[i]))
    t = pd.DataFrame(rows, columns=['w', 'n', 'pos', 'tgt', 'lm1', 'lp1', 'key', 'correct'])
    t['missed'] = t.key == -1
    t['match'] = t.key == 37
    t['lure'] = t.lm1 | t.lp1
    assert (t.loc[t.missed, 'correct'] == 0).all()  # missed trials are recorded as incorrect
    assert ((t.key == 37) | (t.key == 40) | t.missed).all()

    def per_load(tt, include_missed):
        """include_missed=False: authors' rule, every rate on trials with a response.
        include_missed=True: the LLM rule in g1_extract.py, i.e. accuracy and hit/FA rates over all trials with a
        missed response counted as incorrect / not a match; d-prime still on trials with a response."""
        x = tt if include_missed else tt[~tt.missed]
        resp = tt[~tt.missed]
        dp_by = {}
        for (w, n), g in resp.groupby(['w', 'n']):
            tg, nt = g[g.tgt], g[~g.tgt]
            dp_by[(w, n)] = (H.loglinear_dprime(int(tg.match.sum()), len(tg), int(nt.match.sum()), len(nt))
                             if len(tg) >= 5 and len(nt) >= 5 else np.nan)
        out = {}
        for (w, n), g in x.groupby(['w', 'n']):
            tg, nt = g[g.tgt], g[~g.tgt]
            lu, ct, m1 = nt[nt.lure], nt[~nt.lure], nt[nt.lm1]
            out[(w, n)] = dict(acc=g.correct.mean(),
                               hit=tg.match.mean() if len(tg) else np.nan,
                               fa_all=nt.match.mean() if len(nt) else np.nan,
                               fa_lure=lu.match.mean() if len(lu) else np.nan,
                               fa_ctrl=ct.match.mean() if len(ct) else np.nan,
                               fa_m1=m1.match.mean() if len(m1) else np.nan,
                               dp=dp_by.get((w, n), np.nan), n_tgt=len(tg), n_lure=len(lu), n_ctrl=len(ct),
                               n_trials=len(g))
        return pd.DataFrame(out).T

    res = {}
    for label, incl in (('authors_rule_missed_excluded', False), ('llm_rule_missed_as_incorrect', True)):
        P = per_load(t, incl)
        P.index.names = ['w', 'n']

        def col(n, c):
            try:
                return P.xs(n, level='n')[c].astype(float)
            except KeyError:
                return pd.Series(dtype=float)
        r = {}
        for x, y in ((1, 2), (2, 3), (1, 3)):
            r[f'nb_acc_{x}_gt_{y}'] = H.contrast(col(x, 'acc'), col(y, 'acc'))
        for x, y in ((1, 2), (2, 3)):
            r[f'nb_hit_{x}_gt_{y}'] = H.contrast(col(x, 'hit'), col(y, 'hit'))
            r[f'nb_dp_{x}_gt_{y}'] = H.contrast(col(x, 'dp'), col(y, 'dp'))
        for n in (2, 3):
            r[f'nb_lureFA_gt_ctrl_{n}'] = H.contrast(col(n, 'fa_lure'), col(n, 'fa_ctrl'))
            r[f'nb_lurem1FA_gt_ctrl_{n}'] = H.contrast(col(n, 'fa_m1'), col(n, 'fa_ctrl'))
        r['nb_faAll_2_gt_1'] = H.contrast(col(2, 'fa_all'), col(1, 'fa_all'))
        r['nb_faCtrl_3_gt_2'] = H.contrast(col(3, 'fa_ctrl'), col(2, 'fa_ctrl'))
        loads = {}
        for n in sorted(P.index.get_level_values('n').unique()):
            q = P.xs(n, level='n').astype(float)
            loads[int(n)] = dict(n_subjects=int(len(q)), trials_per_subject_median=float(q.n_trials.median()),
                                 **{k: float(q[k].mean()) for k in ('acc', 'hit', 'fa_all', 'fa_lure', 'fa_ctrl', 'dp')},
                                 n_subjects_with_dp=int(q.dp.notna().sum()),
                                 lure_trials_per_subject_median=float(q.n_lure.median()))
        res[label] = dict(contrasts=r, by_load=loads)
        pd.DataFrame(loads).T.rename_axis('load').to_csv(f'{H.OUT}/{OUTD}/n_back_by_load_{label}.csv')
        if not incl:
            P.to_csv(f'{H.OUT}/{OUTD}/n_back_subject_by_load.csv')
    return dict(n_after_qc=int(d.worker_id.nunique()), target_definition_checks=n_check,
                missed_trial_rate=float(t.missed.mean()), results=res,
                notes=['Stimulus identity is case-insensitive (the task alternates letter case); the target rule '
                       'reproduced correct_response on every adaptive trial.',
                       'Load is adaptive (up after <3 errors, down after >5 errors per block), so the subjects and blocks '
                       'at load 3 are a selected subset; a contrast uses subjects with trials at both loads.',
                       'Control (load 0) and practice blocks are excluded, as in the authors\' DV code.'])


# ---------------- Stroop, ANT, go/no-go ----------------
def cc_task(task, cond_col, C, I):
    d = screened(task)
    d = d[d.exp_stage != 'practice'].copy()
    n_missed = int((d.rt == -1).sum())
    d = d[d.rt != -1]  # authors: accuracy on trials with a response
    d['odd'] = H.odd_even_flags(d, ['worker_id', cond_col], 'time_elapsed')
    g = d.groupby(['worker_id', cond_col]).correct.mean().unstack()
    ci = d[d[cond_col].isin([C, I])]
    tot = ci.groupby('worker_id').correct.mean()
    cost = g[I] - g[C]
    half = {}
    for h, sel in (('odd', ci.odd), ('even', ~ci.odd)):
        x = ci[sel].groupby(['worker_id', cond_col]).correct.mean().unstack()
        half[h] = dict(cost=x[I] - x[C], total=ci[sel].groupby('worker_id').correct.mean())
    rel = {k: H.split_half(half['odd'][k], half['even'][k]) for k in ('cost', 'total')}
    out = dict(n_after_qc=int(d.worker_id.nunique()), missed_trials_dropped=n_missed,
               condition_means={c: float(g[c].mean()) for c in g.columns},
               contrast_C_gt_I=H.contrast(g[C], g[I]),
               error_cost_I_minus_C_pp=float(-cost.mean() * 100), split_half=rel)
    return out, dict(cost=cost, total=tot), d


def go_nogo():
    d = screened('go_nogo')
    d = d[d.exp_stage != 'practice'].copy()
    d['press'] = d.key_press == 32
    d['odd'] = H.odd_even_flags(d, ['worker_id', 'condition'], 'time_elapsed')

    def endpoints(x):
        go, ng = x[x.condition == 'go'], x[x.condition == 'nogo']
        return pd.Series(dict(fa=ng.press.mean(), miss=1 - go.press.mean(),
                              dp=H.loglinear_dprime(int(go.press.sum()), len(go), int(ng.press.sum()), len(ng)),
                              total=x.correct.mean(), n_go=len(go), n_nogo=len(ng)))
    e = d.groupby('worker_id').apply(endpoints, include_groups=False)
    eo = d[d.odd].groupby('worker_id').apply(endpoints, include_groups=False)
    ee = d[~d.odd].groupby('worker_id').apply(endpoints, include_groups=False)
    rel = {k: H.split_half(eo[k], ee[k]) for k in ('dp', 'total', 'fa')}
    const = e[(e.fa == 0) & (e.miss == 1) | (e.fa == 1) & (e.miss == 0)]
    out = dict(n_after_qc=int(len(e)), trials_per_subject=dict(go=float(e.n_go.median()), nogo=float(e.n_nogo.median())),
               go_fraction=float((e.n_go / (e.n_go + e.n_nogo)).median()),
               fa_mean=float(e.fa.mean()), miss_mean=float(e.miss.mean()), dprime_loglinear_mean=float(e.dp.mean()),
               contrast_FA_gt_miss=H.contrast(e.fa, e.miss), constant_responders=int(len(const)),
               split_half=rel)
    return out, dict(dp=e.dp, total=e.total, fa=e.fa), e


def main():
    summary = dict(source=SRC, authorisation='User 2026-10-09 item 2', qc_thresholds=QC_THRESH)
    summary['digit_span'] = digit_span()
    summary['n_back'] = n_back()
    st, st_e, _ = cc_task('stroop', 'condition', 'congruent', 'incongruent')
    an, an_e, _ = cc_task('attention_network_task', 'flanker_type', 'congruent', 'incongruent')
    gn, gn_e, gtab = go_nogo()
    summary['stroop'], summary['ant_flanker'], summary['go_nogo'] = st, an, gn
    gtab.to_csv(f'{H.OUT}/{OUTD}/go_nogo_subject.csv')
    # cognitive-control within-group correlations
    native = dict(stroop_cost=st_e['cost'], ant_cost=an_e['cost'], gng_dprime=gn_e['dp'])
    strict = dict(stroop_total=st_e['total'], ant_total=an_e['total'], gng_total=gn_e['total'])
    rel = dict(stroop_cost=st['split_half']['cost']['spearman_brown'], ant_cost=an['split_half']['cost']['spearman_brown'],
               gng_dprime=gn['split_half']['dp']['spearman_brown'], stroop_total=st['split_half']['total']['spearman_brown'],
               ant_total=an['split_half']['total']['spearman_brown'], gng_total=gn['split_half']['total']['spearman_brown'])
    cors = {}
    for name, block in (('native', native), ('strict', strict)):
        ks = list(block)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                c = H.corr(block[ks[i]], block[ks[j]])
                c['attenuation_ceiling_sqrt_rel_product'] = H.attenuation_ceiling(rel[ks[i]], rel[ks[j]])
                cors[f'{name}:{ks[i]}~{ks[j]}'] = c
    summary['cc_correlations'] = dict(pairs=cors, split_half_spearman_brown=rel,
                                      note='native: Stroop I-C accuracy, ANT flanker I-C accuracy, go/no-go log-linear d-prime '
                                           '(higher = better for all three). strict: overall accuracy (Stroop and ANT on '
                                           'congruent+incongruent trials; go/no-go on all test trials).')
    pd.DataFrame({**native, **strict}).to_csv(f'{H.OUT}/{OUTD}/cc_subject_endpoints.csv')
    summary['qc'] = qc_report
    summary['resource'] = T.report()
    print(H.dump(summary, OUTD, 'summary.json'))


if __name__ == '__main__':
    main()
