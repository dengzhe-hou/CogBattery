"""Render saved statistics and diagnostics; no PCA or prediction refitting."""
from pathlib import Path
import json

import pandas as pd

from .matrix import TASK_ABBREVIATIONS

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'results/reanalysis/human54_fullbattery_20261019'
REVIEW = ROOT/'review-stage/human54_fullbattery_20261019'
NAMES = {'primary_valid_only': '主分析：有效回答、完整案例',
         'invalid_wrong': '敏感性：无效按错',
         'wag_highlow_sensitivity': '敏感性：WAG原高低注差'}


def interval(lo, hi, *, percent=False):
    scale = 100 if percent else 1
    return f'[{lo*scale:.2f}, {hi*scale:.2f}]' if percent else f'[{lo:.3f}, {hi:.3f}]'


def main():
    summary = pd.read_csv(DATA/'summary.csv').set_index('name')
    pca = json.loads((DATA/'primary_valid_only/pca.json').read_text())
    manifest = json.loads((DATA/'run_manifest.json').read_text())
    targets = pd.read_csv(DATA/'primary_valid_only/prediction_per_target.csv').set_index('target')
    cells = pd.read_csv(DATA/'primary_valid_only/prediction_cells.csv')
    largest = cells.loc[cells.se_g.idxmax()]
    error_share = largest.se_g/cells.se_g.sum()
    improvement_share = (largest.se_g-largest.se_group)/(cells.se_g-cells.se_group).sum()
    diagnostic = dict(post_hoc_after_results=True, family=largest.family, target=largest.target,
        largest_cell_baseline_mse_share=error_share,
        largest_cell_share_of_net_mse_reduction=improvement_share,
        action='description only; no observations removed and no refitting')
    (DATA/'error_concentration_diagnostic.json').write_text(json.dumps(diagnostic, indent=2)+'\n')
    lines = ['# human54全13任务 PCA 与跨家族预测', '',
        '2026-10-19。三项规则经用户批准后，先完成29项合成测试和同一入口的合成接口运行，再在CPU上执行一次正式派生表分析。以下全部为事后追加分析。没有新模型调用、重读正式回答或重新计分；未修改论文正文、未推送Overleaf。', '',
        '## 结果', '',
        '| 配置 | 模型/家族 | PC1方差占比，% [95% CI] | 一般轴RMSE | 加同组信息RMSE | 相对RMSE收益，% [95% CI] |',
        '|---|---:|---:|---:|---:|---:|']
    for key, r in summary.iterrows():
        lines.append(f'| {NAMES[key]} | {int(r.models)}/{int(r.families)} | '
            f'{r.pc1_share*100:.2f} {interval(r.pc1_lower,r.pc1_upper,percent=True)} | '
            f'{r.rmse_g:.4f} | {r.rmse_group:.4f} | '
            f'{r.group_gain*100:+.2f} {interval(r.group_gain_lower,r.group_gain_upper,percent=True)} |')
    main_row = summary.loc['primary_valid_only']
    delta_ci = pca['bootstrap']['intervals']['group_delta']
    lines += ['',
        f'主分析的PC1/PC2/PC3分别解释{main_row.pc1_share*100:.2f}%、'
        f'{main_row.pc2_share*100:.2f}%、{main_row.pc3_share*100:.2f}%方差。PC1的12个任务载荷为正，'
        f'WAG相关载荷为{pca["correlation_loadings"][-1][0]:.3f}，接近零。成分数量未作假设检验，不能把解释方差直接当作认知机制或“只有一个因子”的证明。', '',
        f'原五组的组内平均相关为{main_row.within_group_mean:.3f}，组间为'
        f'{main_row.between_group_mean:.3f}；δ={main_row.group_delta:.3f}，95%家族区间'
        f'{interval(delta_ci["lower"],delta_ci["upper"])}。区间跨零，不是五组相同或不存在的等价检验。', '',
        f'同组信息的主预测点增益为{main_row.group_gain*100:.2f}%，区间含零；'
        f'{int(main_row.n_positive_targets)}/13任务和{int(main_row.n_positive_families)}/23家族的点增益为正。'
        '两项敏感性也没有给出整个区间均为正的收益。此处不据此自动修改论文主张。', '',
        '## 端点和样本', '',
        '每任务一列：DS/OS位置正确率；NB两负荷d′等权；ST/FL条件均衡准确率；GN d′；CVLT五轮总分；'
        'DRM两警告条件类别均衡准确率等权；SM三引语总体准确率；FB会话14道主项；EP mental七阶等权；'
        'CAL/WAG固定方向type-2 AUROC。PCA先按列z标准化。', '',
        '主分析不插补。R1 7B的CVLT覆盖2/5及SM覆盖91/144未达到80%门槛，保留原值但不进入主矩阵。'
        '无效按错配置不设门槛，54款完整。', '',
        '**样本口径更正**：先前“原WAG高低差17款”指这个端点本身可定义的模型数；其中包含R1 7B，'
        '与其余12任务完整案例取交集后，PCA/预测敏感性实际为16款、9家族。已更正规格说明并保留变更记录；'
        '没有改变用户批准的门槛或重跑统计。', '',
        '新WAG AUROC用保存的下注比例、各下注正确率和总数重建四格频数。37款恒下注（29款低、8款高）'
        '均按并列定义得到0.5；并非把原高低差的NA填0。联合频数与原端点一致。', '',
        '## 全部13个目标的主预测结果', '',
        '| 目标 | 一般轴RMSE | 加同组信息RMSE | 相对收益，% |',
        '|---|---:|---:|---:|']
    for task, code in TASK_ABBREVIATIONS.items():
        r = targets.loc[task]
        lines.append(f'| {code} | {r.rmse_g:.4f} | {r.rmse_g_group:.4f} | {r.relative_rmse_gain*100:+.2f} |')
    lines += ['', '表中的逐任务增益仅为点估计。主分析共有689个逐模型目标预测，汇总为299个家族×目标格。'
        '训练均值截距参考RMSE为'+f'{main_row.rmse_intercept:.4f}。', '',
        '## PC1相关载荷', '', '| 任务 | PC1相关载荷 |', '|---|---:|']
    for task, loadings in zip(pca['tasks'], pca['correlation_loadings']):
        lines.append(f'| {task} | {loadings[0]:+.3f} |')
    lines += ['', '以上为变量与PC1得分的相关载荷，区别于单位长度的特征向量。完整13成分、特征向量和模型得分均已保存。', '',
        '## 预测误差的事后描述', '',
        f'最大一般轴误差格是{largest.family}×WAG（主分析该家族仅R1 14B），占全部家族×目标平方误差和的'
        f'{100*error_share:.1f}%，占同组项净平方误差改善的{100*improvement_share:.1f}%。'
        '这是读取已有折外误差后的描述性诊断，说明总体收益的分布不均；没有因此删除家族、目标或更换算法。'
        '完整逐家族表保留在prediction_per_family.csv。', '',
        '## 方法与验证', '',
        '跨家族预测沿用原稿核心G0算法：每次留出整个合并家族，训练家族各以一个原单位质心参与。'
        '训练质心标准化后，用其余12任务的PC1拟合一般轴基线；同组其余任务的均值先对PC1残差化，再作为扩展项。'
        '两个模型均为含截距Ridge，固定alpha=1e-6，无调参。测试家族不参与任何标准化、PCA或拟合。'
        '误差先在家族×目标格内均值，再格等权。', '',
        'PCA区间用5000次整家族重抽，每次重新标准化及分解；预测收益区间用20000次配对家族误差块重抽，'
        '不重新拟合，条件于当前折外预测。主53款和无效按错54款的PCA抽样均5000次有效。'
        '原高低差16款敏感性有4851次有效、149次因常数列无效，区间只用可定义抽样并公开其数量。', '',
        '29项合成测试覆盖门槛/完整案例、WAG不等类及常下注、已知PCA谱、整家族抽样、'
        'heldout目标/输入泄漏、重复家族成员等权及向量化bootstrap标量参照。完整CLI合成运行耗时2.45秒；'
        f'正式三配置总耗时{manifest["wall_seconds"]:.2f}秒，单线程CPU，峰值RSS约'
        f'{manifest["max_rss_kib"]/1024:.1f}MiB。完整seed、线程和输入位置见run_manifest.json。', '',
        '## 文件', '',
        f'- [四面板结果图，PDF]({DATA}/figures/fig_h54_fullbattery.pdf)',
        f'- [四面板结果图，PNG]({DATA}/figures/fig_h54_fullbattery.png)',
        f'- [完整结果总表]({DATA}/summary.csv)',
        f'- [主矩阵53×13]({DATA}/primary_valid_only/matrix.csv)',
        f'- [逐模型折外预测]({DATA}/primary_valid_only/prediction_predictions.csv)',
        f'- [全部逐家族结果]({DATA}/primary_valid_only/prediction_per_family.csv)',
        f'- [事后追加规格]({REVIEW}/SPEC.md)',
        f'- [样本与WAG核对]({REVIEW}/MATRIX_VALIDATION.md)',
        f'- [合成测试]({REVIEW}/synthetic_tests.log)', '',
        '每个配置目录另有pca.json、correlations.csv、correlation_loadings.csv、pca_scores.csv、'
        'prediction.json、全部bootstrap抽样NPZ、逐格误差与训练折trace。原冻结层一/层二结果均未覆盖。']
    report = '\n'.join(lines)+'\n'
    (REVIEW/'RESULTS_REPORT.md').write_text(report)
    (DATA/'RESULTS_REPORT.md').write_text(report)
    print(REVIEW/'RESULTS_REPORT.md')


if __name__ == '__main__':
    main()
