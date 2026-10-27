"""Render the saved post hoc outputs as Markdown tables. Reads files only; no fitting."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ABBR

ROOT = Path(__file__).resolve().parents[3]


def pct(x):
    return f'{100 * x:+.2f}%'


def ci_pct(ci):
    return f'[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]'


def ci(d, digits=3):
    return f'[{d["lower"]:.{digits}f}, {d["upper"]:.{digits}f}]'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]
    return '\n'.join(out)


def scale_section(out):
    s = json.loads((out / 'scale/scale.json').read_text())
    diag = json.loads((out / 'diagnostics.json').read_text())
    lines = ['## A. 规模', '']
    rows = []
    for basis, label in (('params_b', '总参数'), ('active_params_b', 'MoE 用激活参数')):
        for name in ('PC1', 'F1', 'F2'):
            d = s[basis][name]
            w = d['within_family']
            rows.append([label, name, f'{d["pearson"]:.3f} {ci(d["pearson_ci"])}', f'{d["spearman"]:.3f} {ci(d["spearman_ci"])}',
                         f'{w["pearson"]:.3f}（{w["n_models"]} 款 / {w["n_families"]} 族）'])
    lines += [table(['参数口径', '分数', 'Pearson r [95%]', 'Spearman ρ [95%]', '家族内 r'], rows), '']
    dg = diag['scale']['pc1_log_params_pearson_refit_each_draw']
    lines += [f'区间以全样本分数为条件（每次抽样不重估 PC1、F1、F2）。PC1 若每次抽样重估，Pearson r 区间为 [{dg["lower"]:.3f}, {dg["upper"]:.3f}]。'
              '参数量取实际加载的权重；qwen3:0.6b、1.7b 把共享的输出矩阵算了两次，gemma3:12b、27b 含约 0.42B 视觉编码器。'
              '核查改用纯文本、去重计数后，r 的变化小于 .003（核查记录 wf_a8348844-a48）。', '']
    st = s['structure_params_b']
    rows = []
    for key, label in (('original', '原矩阵'), ('scale_removed', '去掉 log10 参数量后')):
        d = st[key]
        shares = d['variance_shares']
        rows.append([label, f'{100 * shares[0]:.2f}%', f'{100 * shares[1]:.2f}%', f'{100 * shares[2]:.2f}%',
                     d['pa_retained'], ci({k: 100 * v for k, v in st['pc1_share_ci'][key].items() if k in ('lower', 'upper')}, 2)])
    lines += [table(['矩阵', 'PC1', 'PC2', 'PC3', 'PCA 平行分析保留维数', 'PC1 占比 95% 家族区间（%）'], rows), '',
              '这一表的两行区间来自同一组抽样（种子流 7），每次重新回归和分解，用于比较原矩阵和去规模后的占比。原矩阵 PC1 占比的主区间仍是全电池分析的 [35.35, 46.84]（第 D 节），两者只差种子。'
              f'去规模后任务对的平均相关为 {diag["scale"]["scale_removed_mean_pair_correlation"]:.3f}（原矩阵 .300），相关矩阵存于 scale/scale_removed_correlations.csv。', '']
    slopes = ', '.join(f'{t} {b:+.2f}' for t, b in zip(st['tasks'], st['task_slope_on_log10_params_z']))
    lines += [f'各任务 z 分数对 log10 参数量的斜率（每 10 倍参数）：{slopes}。', '']
    labels = dict(nb_control_fa='n-back 控制项虚报率（主要）', wag_constant_bet='下注恒定（主要）', wag_high_wager_rate='高注比例（主要）',
                  nb_lure_fa='n-back 诱饵项虚报率（次要）', gn_false_alarm='go/no-go 虚报率（次要）',
                  drm_new_fa='DRM 新无关词虚报率（次要）', cvlt_recognition_fa='CVLT 再认虚报率（次要）')
    rows = [[labels[name], f'{d["spearman"]:.3f} {ci(d["spearman_ci"])}', d['spearman_ci']['n_undefined']]
            for name, d in s['f2_bias'].items()]
    lines += [table(['偏向指标', 'F2 的 Spearman ρ [95%]', '无定义抽样'], rows), '',
              '机械相关（规格 A4）：F2 主要由 n-back d′ 构成（回归权重 .79），而 d′ 含非目标虚报；去掉 NB 项后 F2 与控制项虚报的 ρ 由 −.233 变为 −.025。'
              '下注恒定者的 WAG 端点按定义为 .5。另有两处规格未列的重叠：GN 端点 d′ 含虚报、DRM 端点含新词虚报，但二者在 F2 上权重很小（.049、−.057），去掉后 GN 的 ρ 为 −.382（核查记录）。'
              '区间同样以全样本 F2 分数为条件。', '']
    c = s['f2_by_constant_bet']
    lines += [f'下注恒定 {c["constant_n"]} 款的 F2 均值 {c["constant_mean_f2"]:.3f}；不恒定 {c["varied_n"]} 款 {c["varied_mean_f2"]:.3f}。', '']
    return lines


def prediction_section(out):
    p = out / 'prediction'
    main = json.loads((p / 'main.json').read_text())
    sens = json.loads((p / 'sensitivities.json').read_text())
    part = json.loads((p / 'partition_reference.json').read_text())
    per = pd.read_csv(p / 'main_per_target.csv')
    diag = pd.read_csv(p / 'main_fa2_fold_diagnostics.csv')
    b = main['bootstrap']
    lines = ['## B. 五组信息的预测增益', '',
             f'复现：一般轴 RMSE {main["rmse_pc1"]:.4f}，加五组 {main["rmse_five_group"]:.4f}，增益 {pct(main["gain_five_group"])} '
             f'{ci_pct(b["five_group_vs_pc1"]["ci95"])}，与已有结果一致。', '']
    rows = [[r.target, f'{r.rmse_pc1:.4f}', f'{r.rmse_five_group:.4f}', pct(r.gain), f'{100 * r.share_of_net_improvement:+.1f}%']
            for r in per.itertuples()]
    dg = json.loads((out / 'diagnostics.json').read_text())['prediction']
    w, a = dg['deepseek_r_wag_cell'], dg['deepseek_r_all_cells']
    lines += [table(['目标', '一般轴 RMSE', '加五组 RMSE', '增益', '占净改善'], rows), '',
              f'13 个任务增益的中位数 {pct(per.gain.median())}；为正的 {int((per.gain > 0).sum())}/13。', '',
              f'集中度：主分析里 deepseek-r 家族只有 R1 14B 一款。它的 WAG 一格占一般轴平方误差的 {100 * w["share_of_baseline_squared_error"]:.1f}%、'
              f'净改善的 {100 * w["share_of_net_improvement"]:.1f}%；该家族 13 格合计占净改善的 {100 * a["share_of_net_improvement"]:.1f}%。'
              f'去掉这一格，WAG 的逐任务增益是 {pct(dg["WAG_gain_without_deepseek_r_cell"])}，CAL 是 {pct(dg["CAL_gain_without_deepseek_r_cell"])}。', '']
    labels = {'drop_WAG': '去掉 WAG（重拟合）', 'drop_CAL_WAG': '去掉 CAL 和 WAG（重拟合）',
              'aggregate_without_WAG': '只在汇总时去掉 WAG', 'aggregate_without_CAL_WAG': '只在汇总时去掉 CAL、WAG',
              'no_deepseek_r': '去掉 deepseek-r 家族', 'direct51': '只用 51 个直接作答模型',
              'families_raw28': '原始 28 族', 'families_lineage18': '谱系 18 族', 'families_lineage17_aya': '谱系 17 族（含 aya）'}
    rows = [['主分析（53 款，合并 23 族）', f'{main["rmse_pc1"]:.4f}', f'{main["rmse_five_group"]:.4f}',
             pct(main['gain_five_group']), ci_pct(b['five_group_vs_pc1']['ci95']), main['n_families']]]
    for key, label in labels.items():
        d = sens[key]
        extra = f'（无同组任务：{", ".join(d["peerless_targets"])}）' if d.get('peerless_targets') else ''
        rows.append([label + extra, f'{d["rmse_pc1"]:.4f}', f'{d["rmse_five_group"]:.4f}', pct(d['gain_five_group']),
                     ci_pct(d['bootstrap']['five_group_vs_pc1']['ci95']), d['n_families']])
    lines += [table(['设置', '一般轴 RMSE', '加五组 RMSE', '增益', '95% 区间', '家族数'], rows), '']
    d = dg['aggregate_without_deepseek_r_cells']
    rows.insert(5, ['只在汇总时去掉 deepseek-r 的 13 格（拟合不变）', f'{d["rmse_pc1"]:.4f}', f'{d["rmse_five_group"]:.4f}',
                    pct(d['gain']), ci_pct(d['bootstrap']['ci95']), 22])
    lines[-2] = table(['设置', '一般轴 RMSE', '加五组 RMSE', '增益', '95% 区间', '家族数'], rows)
    lines += ['各行 RMSE 的水平不能互相比较：去掉或改划家族会改变被评估的格和目标的标准化尺度，而主分析的 RMSE 主要由上面那一格决定。'
              '“只在汇总时去掉”保持拟合不变；“重拟合”同时改变训练和评估。没有同组任务的目标，扩展模型就等于一般轴模型，增益为 0。', '']
    names = {'main': '主分析', **labels}
    rows = []
    for key, d in part.items():
        q = d['quantiles']
        rows.append([names.get(key, key), pct(d['five_group_gain']), f'{d["rank_from_top"]:,} / {d["n_partitions"]:,}',
                     f'{d["share_at_least_five_group"]:.4f}', pct(q['0.05']), pct(q['0.5']), pct(q['0.95']),
                     pct(d['max']), f'{d["share_positive"]:.3f}'])
    lines += ['随机同规模分组（13 个任务分成 3/3/3/2/2，全部 600,600 种）：', '',
              table(['设置', '五组增益', '名次（从高到低）', '增益不低于五组的比例', '5% 分位', '中位数', '95% 分位', '最大', '增益为正的比例'], rows), '']
    rows = []
    for name, label in (('pc12', 'PC1 + PC2'), ('fa2', '两因子分数')):
        rows.append([label, f'{main["rmse_" + name]:.4f}', pct(main[f'gain_{name}_vs_pc1']),
                     ci_pct(b[f'{name}_vs_pc1']['ci95']), pct(main[f'gain_five_group_vs_{name}']),
                     ci_pct(b[f'five_group_vs_{name}']['ci95'])])
    lines += ['二维基线（同一留一家族管线）：', '',
              table(['基线', 'RMSE', '相对一般轴的增益', '95% 区间', '五组相对它的增益', '95% 区间'], rows), '',
              f'两因子每折诊断：{len(diag)} 折中收敛 {int(diag.converged.sum())}，proper {int(diag.proper.sum())}，'
              f'边界解 {int(diag.boundary.sum())}；不适当解未替换。', '']
    return lines


def reliability_section(out):
    r = json.loads((out / 'reliability/summary.json').read_text())
    per = pd.read_csv(out / 'reliability/per_task.csv')
    lines = ['## C. 信度（模型一侧，53 款）', '']
    rows = []
    for row in per.itertuples():
        rel_ci = ast.literal_eval(row.rel_ci)
        rows.append([row.task, ', '.join(ast.literal_eval(row.forms)), row.k, row.n, f'{row.mean_r:.3f}', f'{row.rel:.3f}', ci(rel_ci)])
    dg = json.loads((out / 'diagnostics.json').read_text())['reliability']
    wag = dg['wag']
    lines += [table(['任务', '复本 / 分半', 'k', '模型数', '平均复本间 r', '信度', '95% 家族区间'], rows), '',
              '信度：两份用 Spearman-Brown，三份以上用 Cronbach α。几处说明：', '',
              f'- WAG：{wag["models_at_half_in_both_forms"]}/53 款两半都是 .5。两半的 Pearson r 为 {wag["half_r"]:.3f}，Spearman 为 {wag["half_spearman"]:.3f}；'
              f'去掉 R1 14B 一款后 r 为 {wag["half_r_without_deepseek_r1_14b"]:.3f}。区间下限 −25 是 Spearman-Brown 在 r 接近 −1 时发散所致；两半 r 本身的区间为 '
              f'[{wag["half_r_interval"]["lower"]:.3f}, {wag["half_r_interval"]["upper"]:.3f}]，{dg["draws_with_nonpositive_or_undefined_rel"]["WAG"]}/5000 次抽样的 WAG 信度不为正或无定义。',
              '- CVLT：五个学习试次在同一会话里，后面的提示含模型自己前面的回答；44/53 款五次得分相同。这是会话内一致性，不是复本信度，可能偏高。',
              '- DRM：两个警告条件不完全平行，结果偏保守。', '']
    iv = r['intervals']
    rows = [['原始相关', f'{r["within"]:.3f} {ci(iv["within"])}', f'{r["between"]:.3f} {ci(iv["between"])}', f'{r["delta"]:+.3f} {ci(iv["delta"])}']]
    if 'delta_star' in r:
        rows.append(['去衰减后', f'{r["within_star"]:.3f} {ci(iv["within_star"])}', f'{r["between_star"]:.3f} {ci(iv["between_star"])}',
                     f'{r["delta_star"]:+.3f} {ci(iv["delta_star"])}'])
    rows.append(['原始相关，只用 δ* 有定义的抽样', '', '', f'{r["delta"]:+.3f} [{dg["raw_delta_defined_draws"]["lower"]:.3f}, {dg["raw_delta_defined_draws"]["upper"]:.3f}]'])
    lines += [table(['', '组内均值', '组间均值', 'δ'], rows), '',
              f'δ* 的区间只来自全部信度为正的 {dg["draws_delta_star_defined"]}/5000 次抽样，这些抽样多数含 R1 14B'
              f'（{100 * dg["deepseek_r_present_share"]["defined_draws"]:.0f}%，全部抽样中为 {100 * dg["deepseek_r_present_share"]["all_draws"]:.0f}%），'
              '所以和第一行不可直接比较；第三行给出同一批抽样上的原始 δ。'
              f'点估计没有 |r*| > 1 的任务对；有定义的抽样中 {dg["draws_with_abs_r_star_above_one"]} 次出现 |r*| > 1，共 {dg["pairs_above_one_total"]} 对，'
              '主要是 CAL–WAG 和 FL–WAG（不截断）。原始 δ 的主区间仍是全电池分析的 [−0.086, 0.080]（第 D 节），本节区间只差种子。', '']
    return lines


def sro_section(path):
    if not path.exists():
        return ['## C4. SRO 五任务两侧信度', '', '（尚未运行）', '']
    s = json.loads(path.read_text())
    names = dict(digit_span='DS', n_back='NB', stroop='ST', flanker='FL', go_nogo='GN')
    rows = []
    for t, name in names.items():
        h, m = s['human']['reliability'][t], s['model']['reliability'][t]
        rows.append([name, f'{h["rel"]:.3f} {ci(h["ci"])}（{h["n"]} 人）', f'{m["rel"]:.3f} {ci(m["ci"])}（{m["n"]} 款）',
                     ', '.join(s['model_forms'][t])])
    lines = ['## C4. SRO 五任务两侧信度（端点 structure_sro）', '',
             table(['任务', '人类分半信度 [95%]', '模型复本信度 [95%]', '模型复本'], rows), '']
    rows = []
    for side, label in (('human', f'人类（{s["n_humans"]} 人）'), ('model', f'模型（{s["n_models"]} 款 / {s["n_families"]} 族）')):
        d = s[side]
        rows.append([label, f'{d["delta"]:+.3f} {ci(d["delta_ci"])}', f'{d["delta_star"]:+.3f} {ci(d["delta_star_ci"])}'])
    d = s['difference']
    rows.append(['人类 − 模型', f'{d["delta"]:+.3f} {ci(d["delta_ci"])}', f'{d["delta_star"]:+.3f} {ci(d["delta_star_ci"])}'])
    lines += [table(['', 'δ（原始）[95%]', 'δ*（去衰减）[95%]'], rows), '',
              f'人类整人、模型整家族各 {s["draws"]} 次独立重抽。原始 δ 的区间与层二报告的区间种子不同，正文仍用层二的区间。'
              f'人类 δ* 在 {s["draws"] - s["human"]["delta_star_ci"]["n_valid"]} 次抽样中无定义（DS 信度不为正），核查复算有 132 次抽样出现 |r*| > 1（模型 0 次）。', '']
    dg = json.loads((path.parent.parent / 'human54_posthoc_20261020/diagnostics.json').read_text())['sro_ds_split']
    lines += [f'人类 DS 分半的注意：DS 是自适应阶梯，多数格每人只有一个试次。预先写定的规则（格内交替）实际把每格第一次出现分到一半、重复出现分到另一半，两半不平行，信度偏低。'
              f'按方向内试次顺序奇偶分半，人类 DS 信度为 {dg["human_ds_rel"]:.3f}，人类 δ* {dg["human_delta_star"]:+.3f}，人类−模型 δ* {dg["difference_delta_star"]:+.3f}'
              f'（预设规则下为 {dg["prespecified"]["human_ds_rel"]:.3f}、{dg["prespecified"]["human_delta_star"]:+.3f}、{dg["prespecified"]["difference_delta_star"]:+.3f}）。'
              '核查的随机分半平均 .431–.511。结论不随分法改变。', '']
    return lines


def family_section(out):
    s = json.loads((out / 'families/structure.json').read_text())
    labels = {'merged23': '合并 23 族（主）', 'raw28': '原始 28 族', 'lineage18': '谱系 18 族',
              'lineage17_aya': '谱系 17 族（含 aya）', 'direct51_merged': '51 款直接作答，合并族'}
    rows = []
    for key, label in labels.items():
        d = s[key]
        iv = d['intervals']
        rows.append([label, d['n_families'], f'{100 * d["pc1_share"]:.2f}% [{100 * iv["pc1_share"]["lower"]:.2f}, {100 * iv["pc1_share"]["upper"]:.2f}]',
                     f'{d["structure"]["group_delta"]:+.3f} {ci(iv["group_delta"])}'])
    return ['## D. 家族定义与面板', '', table(['设置', '家族数', 'PC1 占比 [95%]', 'δ [95%]'], rows), '']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reanalysis/human54_posthoc_20261020')
    parser.add_argument('--report', type=Path, default=ROOT / 'review-stage/human54_posthoc_20261020/TABLES.md')
    a = parser.parse_args()
    lines = ['# human54 事后补充分析：结果表', '', '由 scripts/reanalysis/human54_posthoc_20261020/report.py 从保存的输出生成。全部为事后分析。', '']
    for section in (scale_section, prediction_section, reliability_section):
        lines += section(a.output)
    lines += sro_section(a.output.parent / 'human54_posthoc_20261020_sro/sro_reliability.json')
    lines += family_section(a.output)
    a.report.write_text('\n'.join(lines) + '\n')
    print(a.report)


if __name__ == '__main__':
    main()
