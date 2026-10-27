"""Section E report: reads saved gate outputs, adds the diagnostics the verification
(wf_f60bc7c0-35b) asked for, and writes GATES_REPORT.md. CPU, about two minutes."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

from scripts.reanalysis.human54_layers_20261019.analysis_statistics import cluster_mean
from . import gates as G

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/reanalysis/human54_posthoc_20261020_gates'
REPORT = ROOT / 'review-stage/human54_posthoc_20261020/GATES_REPORT.md'
SEED_SLOPE = int(np.random.SeedSequence(20261021).generate_state(1)[0])


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] +
                     ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows])


def f3(x):
    return '' if pd.isna(x) else f'{x:.3f}'


def slope_checks(e, models, params, draws=1000):
    """LRT under ML and whole-family bootstrap of the REML slope (continuous headlines)."""
    logp = np.log10(params.params_b)
    mu, sd = logp.mean(), logp.std(ddof=0)
    out = {}
    rng = np.random.default_rng(SEED_SLOPE)
    for key, (task, endpoint, direction) in G.HEADLINES.items():
        if direction is None:
            continue
        rows = e[(e.task == task) & (e.endpoint == endpoint) & (e.included == True)].set_index('model')
        d = pd.DataFrame({'y': rows.value.astype(float), 'zlogp': (logp[rows.index] - mu) / sd,
                          'native': models.native_thinking[rows.index].astype(int),
                          'family': models.family_merged[rows.index]})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            full = smf.mixedlm('y ~ zlogp + native', d, groups=d['family']).fit(reml=False)
            null = smf.mixedlm('y ~ native', d, groups=d['family']).fit(reml=False)
            lrt = 2 * (full.llf - null.llf)
            fams = sorted(d.family.unique())
            slopes = []
            for _ in range(draws):
                pick = rng.choice(fams, size=len(fams), replace=True)
                parts = [d[d.family == f].assign(family=f'{f}#{i}') for i, f in enumerate(pick)]
                b = pd.concat(parts)
                try:
                    slopes.append(smf.mixedlm('y ~ zlogp + native', b, groups=b['family']).fit(reml=True).params['zlogp'])
                except Exception:
                    slopes.append(np.nan)
        s = np.asarray(slopes, float)
        out[key] = dict(lrt=float(lrt), lrt_p=float(chi2.sf(lrt, 1)),
                        boot_lower=float(np.nanquantile(s, .025) / sd), boot_upper=float(np.nanquantile(s, .975) / sd),
                        boot_valid=int(np.isfinite(s).sum()))
    return out


def nb_heldout_both(e, nb, paired, models):
    human = float(paired[(paired.task == 'n_back') & (paired.endpoint == 'load_2_control')].human_mean.iloc[0])
    rows = []
    for gate_half, est_half in ((0, 1), (1, 0)):
        gated, ungated, fg, fu = [], [], [], []
        for m in models.index:
            g, h = G.nb_counts(nb, m, half=gate_half), G.nb_counts(nb, m, half=est_half)
            fa = h.match_n['control'] / h.denominator['control']
            ungated.append(fa)
            fu.append(models.family_merged[m])
            gate = G.sdt_gate(g.match_n['target'], g.denominator['target'],
                              g.match_n[['lure', 'control']].sum(), g.denominator[['lure', 'control']].sum())
            if gate['p'] < G.ALPHA:
                gated.append(fa)
                fg.append(models.family_merged[m])
        a = cluster_mean(np.array(gated) - human, fg)[0]
        b = cluster_mean(np.array(ungated) - human, fu)[0]
        rows.append([f'half{gate_half} 定门槛，half{est_half} 算终点', len(gated), f'{a["mean"]:+.3f} [{a["lower"]:+.3f}, {a["upper"]:+.3f}]',
                     f'{b["mean"]:+.3f} [{b["lower"]:+.3f}, {b["upper"]:+.3f}]'])
    return rows


def main():
    e, p, nb, paired, models = G.load()
    params = pd.read_csv(OUT / 'model_parameters.csv', index_col='model')
    gates = pd.read_csv(OUT / 'gates.csv')
    counts = pd.read_csv(OUT / 'gate_counts.csv', index_col='task')
    heads = pd.read_csv(OUT / 'headlines_by_stratum.csv')
    held = pd.read_csv(OUT / 'headlines_heldout.csv')
    mm = pd.read_csv(OUT / 'mixed_models.csv')
    cap = pd.read_csv(OUT / 'capability_correlations.csv')
    expl = pd.read_csv(OUT / 'exploratory_paired_endpoints.csv')
    checks = slope_checks(e, models, params)
    (OUT / 'slope_checks.json').write_text(json.dumps(dict(seed=SEED_SLOPE, draws=1000, results=checks), indent=2) + '\n')

    fb = gates[gates.task == 'false_belief']
    st_pass = gates[(gates.task == 'stroop') & gates.passed].model
    word = e[(e.task == 'stroop') & (e.endpoint == 'word_response_on_incongruent')].set_index('model').value.astype(float)
    inc = e[(e.task == 'stroop') & (e.endpoint == 'incongruent_accuracy')].set_index('model').value.astype(float)
    wagB = []
    for m in models.index:
        parts = [(G.value(e, m, 'post_decision_wagering', f'B_{d}|high_wager_rate'),
                  G.value(e, m, 'post_decision_wagering', f'B_{d}|high_wager_rate', 'n_scored')) for d in ('forward', 'reverse')]
        wagB.append(sum(v * n for v, n in parts) / sum(n for _, n in parts))
    wagB_const = int(np.isin(np.array(wagB), (0., 1.)).sum())

    names = dict(NB_control_fa='NB 2-back 控制项虚报', ST_interference='ST 误差干扰', CAL_auroc='CAL type-2 AUROC',
                 WAG_constant='WAG 恒定下注')
    strata = dict(all='全部', gate='过门槛', large='≥27B', gate_and_large='过门槛且 ≥27B')
    L = ['# 理解门槛与规模（SPEC 第 E 节）', '',
         '2026-10-21。事后分析，只用已存派生表，没有调用模型。代码 scripts/reanalysis/human54_posthoc_20261020/gates.py、gates_report.py；'
         '输出 results/reanalysis/human54_posthoc_20261020_gates/。独立核查 wf_f60bc7c0-35b：3 个代理用自己的代码复算全部门槛、分层、留出、混合模型和能力相关，没有发现错数；'
         '下面的说明来自核查意见。', '', '## E1 门槛通过数（54 款）', '']
    rule = gates.groupby('task').rule.first()
    L += [table(['任务', '规则', '通过', '严格（≥.8）'], [[G.D.ABBR[t], rule[t], int(r.passed), int(r.strict_passed)] for t, r in counts.iterrows()]), '']
    L += [f'- SM 的 52 款里含 R1 7B，它只有 91/144 个有效回答（覆盖 .63，低于 .80）；门槛按有效回答算，不看覆盖率。按覆盖率规则应为 51。',
          f'- FB 是两个位置二选一，.5 正好是猜的水平。7/7 全对的有 {int((fb.measure >= 1).sum())} 款，≥6/7 的 {int((fb.measure >= 6/7 - 1e-9).sum())} 款。',
          f'- ST 门槛（一致条件）分不清"看懂了题"和"在读字"：通过的 51 款里，{int((word[st_pass] > .5).sum())} 款在不一致题上过半数回答了字义，{int((inc[st_pass] < 1/3).sum())} 款不一致题低于猜的水平。'
          '所以 ST 门槛只说明"颜色和字一致时能按格式给出颜色名"。',
          '- 显著性门槛受题量影响：NB、GN 题多，平衡准确率刚过 .5 也能通过（NB 通过的 32 款里 19 款低于 .60）；DRM 只有 8 道新词，.65 也可能不过。严格门槛一并报告。',
          '- OS 门槛是 9 次回忆、27 个字母位置（规格原写"9 个位置"，已在变更记录更正）。', '']
    L += ['## E2 头条终点分层', '']
    rows = []
    for _, r in heads.iterrows():
        if r.headline == 'WAG_constant':
            val = f'{int(r.constant)}/{int(r.n_models)} 恒定（{r.share:.0%}）' if r.n_models else '—'
        else:
            val = f'{r.model_mean:.3f}；差 {r.difference:+.3f} [{r.lower:+.3f}, {r.upper:+.3f}]；{int(r.models_in_direction)} 款、{int(r.families_in_direction)} 族同向' if r.n_models else '—'
        rows.append([names[r.headline], '主' if r.gate_rule.startswith('p') else '严格', strata[r.stratum], int(r.n_models), int(r.n_families), val])
    L += [table(['终点', '门槛', '分层', '模型', '家族', '模型均值；模型−人类 [95%]；同向计数'], rows), '',
          '人类均值：NB .071，ST .051，CAL .714；WAG 没有人类比例。≥27B 共 9 款、6 个家族，没有原生思考模型；改用激活参数后为 8 款，结论相同（核查记录）。'
          'NB 严格门槛只有 3 款、都是原生思考模型、来自 2 个家族，门槛和终点共用大量非目标试次，这一行的区间没有信息量。', '']
    L += ['## E3 留出版本', '']
    rows = [[names[r.headline], r.version, int(r.n_models),
             (f'{r.difference:+.3f} [{r.lower:+.3f}, {r.upper:+.3f}]' if r.headline != 'WAG_constant' else f'{int(r.constant)}/{int(r.n_models)} 恒定（{r.share:.0%}）')]
            for _, r in held.iterrows()]
    L += [table(['终点', '做法', '过门槛模型', '结果'], rows), '',
          'NB 两个方向都算，并和同一半数据不加门槛的结果对照（门槛本身的作用要和同一半比）：', '',
          table(['方向', '过门槛', '过门槛：模型−人类 [95%]', '不加门槛（54 款，同一半）'], nb_heldout_both(e, nb, paired, models)), '',
          f'- WAG 留出版本的对照是"只看语法 B 是否恒定"：54 款中 {wagB_const} 款恒定（{wagB_const / 54:.0%}），不是全部 240 题时的 37/54。',
          '- CAL 留出只分开了试次，没有分开题目：位置 0 和位置 1 是同样 40 道题左右互换，两位置准确率跨模型相关 .89。',
          '- ST 有 51/54 款过门槛，按规格不另做留出版本。',
          '- 留出版本的门槛题量减半，门槛更难过，所以"门槛"和"留出"两行的差别同时来自选择偏差和门槛松紧。', '']
    L += ['## E4 混合模型（仿 CogBench）', '']
    hm = dict(NB_control_fa=.0714, ST_interference=.0509, CAL_auroc=.7140)
    rows = []
    for _, r in mm.iterrows():
        lo, hi = r.slope_lower / r.logp_sd, r.slope_upper / r.logp_sd
        c = checks.get(r.headline)
        extra = f'LRT p={c["lrt_p"]:.3f}；家族 bootstrap [{c["boot_lower"]:+.3f}, {c["boot_upper"]:+.3f}]' if c else '对数几率；家族聚类 GEE'
        rows.append([names[r.headline], f'{r.slope_per_10x:+.3f} [{lo:+.3f}, {hi:+.3f}]', extra,
                     f'{r.predicted_7B:.3f} / {r.predicted_70B:.3f}', f'{hm[r.headline]:.3f}' if r.headline in hm else '—'])
    L += [table(['终点', '每 10 倍参数的斜率 [Wald 95%]', '稳健性', '预测 7B / 70B（非思考模型）', '人类'], rows), '',
          '- NB 的斜率为正，但 Wald 区间下限贴着 0，LRT 和家族 bootstrap 都不排除 0，不能说"随规模可靠上升"。CAL 在三种方法下都随规模上升。ST 三种方法都包含 0。',
          '- 是否原生思考这一项只有 3 款模型、2 个家族（deepseek-r 全是思考模型），只作调整，不解释。',
          '- statsmodels 对 CAL 给出"可能在边界上"的警告：它在随机截距方差绝对值小于 .01 时就会报，AUROC 的尺度小，核查确认估计不在边界上。', '']
    L += ['## E5 能力相关（仿 Ren et al. 2024）', '']
    rows = [[r.score, f'{r.rho_log_params:.2f} [{r.rho_log_params_lower:.2f}, {r.rho_log_params_upper:.2f}]',
             f'{r.rho_loo_pc1:.2f} [{r.rho_loo_pc1_lower:.2f}, {r.rho_loo_pc1_upper:.2f}]'] for _, r in cap.iterrows()]
    L += [table(['分数', 'ρ 与 log10 参数量', 'ρ 与去掉本任务的 PC1'], rows), '',
          '53 款（矩阵面板）。区间以全样本 PC1 为条件，与 A1 相同；核查每次重估 PC1，区间边界最多移动约 .05，没有方向变化。Ren 等把低于 .40 视为低相关，这里只作参照。', '']
    L += ['## E7 全部 134 个配对终点（探索）', '',
          f'规格原写 130，实际是论文所说的 134 个源配对终点（已在变更记录更正）。逐终点结果见 exploratory_paired_endpoints.csv。'
          f'模型均值高于人类的终点：全部 {int((expl.all_mean_diff > 0).sum())}/134，过门槛 {int((expl.gate_mean_diff > 0).sum())}/134，≥27B {int((expl.large_mean_diff > 0).sum())}/134。', '']
    L += ['## E6 图', '', 'results/reanalysis/human54_posthoc_20261020_gates/figures/fig_h54_scale_gates.pdf。没有放进论文。', '']
    REPORT.write_text('\n'.join(L) + '\n')
    print(REPORT)


if __name__ == '__main__':
    main()
