"""Render complete saved PA/EFA outputs without recomputing statistical fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run import ROOT, TASKS
from .stability import align_signed_permutation


NAMES = {
    'primary_valid_only': '主分析：53款',
    'primary_valid_only_centroids': '主分析家族均值：23族',
    'invalid_wrong': '无效按错：54款',
    'invalid_wrong_centroids': '无效按错家族均值：23族',
}


def read(path):
    return json.loads(path.read_text())


def f(value, digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---']*len(header)) + ' |']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    return '\n'.join(lines)


def report(output):
    manifest = read(output / 'run_manifest.json')
    if manifest['status'] != 'complete':
        raise ValueError('Cannot summarize an incomplete run as complete')
    cases = list(manifest['cases'])
    pa = {name: read(output / name / 'parallel.json') for name in cases}
    stability = {name: read(output / name / 'stability.json') for name in cases}
    ks = manifest['efa_candidate_dimensions']
    lines = ['# human54平行分析与旋转因子分析', '',
             f"运行类型：{'合成' if manifest['synthetic'] else '正式派生表'}。只用CPU，无模型调用、无正式回答重计分，未修改论文或推送Overleaf。", '',
             '## 建议保留维数', '',
             table(['口径', 'PCA平行分析', '共同因子平行分析', '共同因子零谱有效次数'],
                   [(NAMES[name], pa[name]['pc']['retained'], pa[name]['fa']['retained'],
                     f"{pa[name]['n_valid_fa_null']}/{manifest['pa_draws']}") for name in cases]), '',
             '每列独立置换，不放回；逐阶95%参考线。保留自第一阶起连续越线且为正的维数；完整13阶谱和越线状态均保存。模型层PA以行可交换为参考，家族均值层改变分析单位，不是对模型层零分布的校正。', '',
             table(['口径','零谱收敛','零谱边界解','零谱不适当解'],
                   [(NAMES[name],pa[name]['null_common_factor_fit_counts']['converged'],
                     pa[name]['null_common_factor_fit_counts']['boundary'],
                     manifest['pa_draws']-pa[name]['null_common_factor_fit_counts']['proper']) for name in cases]), '',
             'PA零谱按写定规则保留收敛且共同度有限的边界/不适当解；“有效谱”不等于“proper因子解”。上述次数公开，不补抽。', '',
             f'共同因子建议维数及其相邻、可识别解的并集为{ks}，以下全部保留。', '',
             '## 所有点估计解', '']
    rows = []
    all_fits = {}
    for name in cases:
        for k in ks:
            fit = read(output / name / f'efa_{k}' / 'fit.json')
            all_fits[(name,k)] = fit
            d = fit['diagnostics']
            lower = ','.join(TASKS[j] for j in d['lower_boundary_variables']) or '无'
            upper = ','.join(TASKS[j] for j in d['upper_boundary_variables']) or '无'
            heywood = ','.join(TASKS[j] for j in d['heywood_variables']) or '无'
            rows.append((NAMES[name], k, f(fit['rmsr']), fit['converged'], fit['proper'], lower, upper, heywood))
    lines += [table(['口径','k','RMSR','收敛','proper','ψ下界','ψ上界','Heywood任务'], rows), '',
              'ψ为提取时优化的独特性参数，界限[.005,1]；实际共同度与独特性由ΛΦΛᵀ计算。上界可对应几乎不载荷的任务，下界与Heywood需分别查看，均不自动视为同一种失败。', '',
              '## 全部bootstrap状态', '']
    rows = []
    for name in cases:
        for k in ks:
            counts = stability[name]['dimensions'][str(k)]['counts']
            rows.append((NAMES[name], k, counts['attempted'], counts['converged_finite'],
                         counts['failed'], counts['finite_improper'], counts['finite_boundary'],
                         counts['proper_nonboundary']))
    lines += [table(['口径','k','请求','收敛有限','失败','有限不适当','有限边界','收敛proper非边界'], rows), '',
              '不适当、边界、失败可重叠。全部抽样保留在分母；未补抽、未删任务。下面的主区间包括全部收敛有限解；另一个仅proper非边界的条件性汇总完整保存在stability.json。', '',
              '## 主53款的载荷与稳定性', '']
    ledger = []
    for k in ks:
        fit = all_fits[('primary_valid_only', k)]
        b = stability['primary_valid_only']['dimensions'][str(k)]
        intervals = b['summaries']['converged_finite']['intervals']
        lines += [f'### {k}因子解', '',
                  f"收敛={fit['converged']}；proper={fit['proper']}；边界={fit['boundary']}；RMSR={f(fit['rmsr'])}。", '']
        rows = []
        for j, task in enumerate(TASKS):
            cells = []
            for factor in range(k):
                point = fit['pattern'][j][factor]
                lo = None if intervals['pattern']['lower'] is None else intervals['pattern']['lower'][j][factor]
                hi = None if intervals['pattern']['upper'] is None else intervals['pattern']['upper'][j][factor]
                cells.append(f'{f(point)} [{f(lo)}, {f(hi)}]')
                ledger.append(dict(case='primary_valid_only', factors=k, factor=factor+1,
                                   task=task, pattern=point, ci_lower=lo, ci_upper=hi))
            rows.append([task] + cells)
        lines += [table(['任务'] + [f'F{i+1} pattern [95%区间]' for i in range(k)], rows), '']
        rows = []
        for factor in range(k):
            v = intervals['congruence']
            rows.append((f'F{factor+1}',
                         f(None if v['median'] is None else v['median'][factor]),
                         f(None if v['lower'] is None else v['lower'][factor]),
                         f(None if v['upper'] is None else v['upper'][factor])))
        lines += ['因子与点解的Tucker载荷一致性（仅排列与符号匹配）：', '',
                  table(['因子','中位数','2.5%分位','97.5%分位'],rows), '',
                  '点解因子相关Φ：', '',
                  table(['因子']+[f'F{i+1}' for i in range(k)],
                        [[f'F{j+1}']+[f(v) for v in row] for j,row in enumerate(fit['phi'])]), '']
    lines += ['## 不同口径的点解载荷对应', '',
              '同一k下，以主53款为参考仅排列/改符号；一致性是已存载荷的描述性比较，不是额外显著性检验。', '']
    rows = []
    for k in ks:
        reference = np.asarray(all_fits[('primary_valid_only', k)]['pattern'])
        for name in cases:
            if name == 'primary_valid_only':
                continue
            fit = all_fits[(name,k)]
            aligned = align_signed_permutation(fit['pattern'],fit['phi'],reference)
            rows.append((NAMES[name],k, ', '.join(f(v) for v in aligned['congruence'])))
    lines += [table(['比较口径','k','F1…Fk Tucker一致性'],rows), '',
              '## 解释范围与文件', '',
              '主成分和共同因子不是同一分解。斜交旋转允许因子相关；pattern与structure分别导出，相关因子的载荷平方不作互不重叠的方差份额。因子编号只是展示编号，名称须根据任务组合提出候选。', '',
              'bootstrap区间条件于固定k和所列拟合状态，不能当作重复选择维数的频率或维数置信区间。质心bootstrap可能产生奇异矩阵，未做平滑；所有实际失败/边界结果均在原输出中。', '',
              f"本次入口耗时{manifest['elapsed_seconds']:.2f}秒，峰值RSS {manifest['max_rss_kib']/1024:.1f} MiB。", '',
              f'- [运行清单]({output / "run_manifest.json"})',
              f'- [完整平行分析表]({output / "parallel_summary.csv"})',
              f'- [全部点解拟合摘要]({output / "fit_summary.csv"})',
              f'- [主53款载荷区间表]({output / "primary_loadings_with_intervals.csv"})',
              f'- [执行规格]({ROOT / "review-stage/human54_fullbattery_20261019/FA_SPEC.md"})', '',
              '每口径另存matrix/metadata、parallel.json/parallel_draws.npz、各efa_k的pattern/structure/Φ/残差、stability.json/stability_draws.npz。']
    pd.DataFrame(ledger).to_csv(output / 'primary_loadings_with_intervals.csv', index=False)
    text = '\n'.join(lines)+'\n'
    (output / 'RESULTS_REPORT.md').write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/reanalysis/human54_factors_20261019')
    parser.add_argument('--review-copy', type=Path)
    args = parser.parse_args()
    text = report(args.output.resolve())
    if args.review_copy:
        args.review_copy.write_text(text)


if __name__ == '__main__':
    main()
