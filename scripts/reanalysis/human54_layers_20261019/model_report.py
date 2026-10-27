"""Format existing derived tables; no formal records or statistical reruns."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
RESULT=ROOT/'results/reanalysis/human54_layers_20261019'
DEST=ROOT/'review-stage/human54_results_20261019/TASK_RESULTS.md'
TASKS=[('digit_span','DS 数字广度'),('n_back','NB n-back'),('operation_span','OS 操作广度'),
       ('stroop','ST Stroop'),('flanker','FL Flanker'),('go_nogo','GN Go/No-Go'),
       ('cvlt_word_list','CVLT-II'),('drm_false_memory','DRM 错误记忆'),('source_monitoring','SM 来源记忆'),
       ('false_belief','FB 错误信念'),('epitome_tom','EPITOME 递归读心'),
       ('confidence_calibration','CAL 信心校准'),('post_decision_wagering','WAG 作答后下注')]


def load(name):
    with (RESULT/name).open() as f:return list(csv.DictReader(f))


def num(value):
    return 'NA' if value in ('',None) else f'{float(value):.4f}'


def main():
    panel=load('layer1/panel_differences.csv')
    panel=[r for r in panel if r['family_mapping']=='merged']
    index={(r['task'],r['rule'],r['endpoint'],r['human_cohort'],r['panel']):r for r in panel}
    primary=[r for r in panel if r['panel']=='all54' and r['rule']=='valid_only' and r['human_cohort']=='primary']
    lines=['# human54 第一层逐范式结果事实表','',
        '来源为已完成的派生表。本文不读取正式 calls，不重新计分、bootstrap 或拟合；不修改论文，也不提出主张或入稿位置。', '',
        '主规格为 `all54 / merged / valid_only`：模型等权均值，端点主覆盖率 ≥ 0.80；区间为模型减固定人类均值的 5,000 次合并家族整簇 bootstrap 95% percentile 区间，seed=20261018。区间是描述性区间，不作显著性判定。`n` 是纳入的模型数，不是有效 bootstrap 次数。', '',
        '比例保持 0–1 原单位；CAL `confidence_mean_percent` 为 0–100；CVLT 回忆与侵入/重复是词数，学习斜率为词/轮；d′ 为原尺度。表中“按错”指 `invalid_wrong` 的模型均值及纳入数，人类第一层来源端点保持不变。所有数字四舍五入至四位小数。', '',
        '所有 13 范式和主来源的全部已配对端点均列出，FB 四种呈现条件也全部保留。逐模型原值、覆盖率和人类百分位见完整派生表。', '',
        '- [面板差及区间](../../results/reanalysis/human54_layers_20261019/layer1/panel_differences.csv)',
        '- [逐模型端点](../../results/reanalysis/human54_layers_20261019/layer1/endpoints.csv)',
        '- [逐模型人类对照](../../results/reanalysis/human54_layers_20261019/layer1/model_human_endpoints.csv)',
        '- [逐格覆盖率](../../results/reanalysis/human54_layers_20261019/coverage/cells.csv)','']
    count=0
    for task,title in TASKS:
        rows=[r for r in primary if r['task']==task]
        if not rows:raise ValueError('Missing task '+task)
        lines+=['## '+title,'','| 端点 | 人类均值（n） | 模型均值 | 模型−人类 | 差的 95% CI | 纳入模型 | 按错均值（n） |',
                '|---|---:|---:|---:|---|---:|---:|']
        for r in rows:
            s=index[(task,'invalid_wrong',r['endpoint'],'primary','all54')]
            lines.append(f"| `{r['endpoint']}` | {num(r['human_mean'])} ({r['human_n']}) | {num(r['model_mean'])} | {num(r['mean'])} | [{num(r['lower'])}, {num(r['upper'])}] | {r['models']} | {num(s['model_mean'])} ({s['models']}) |")
            count+=1
        lines.append('')
        if task=='digit_span':lines+=['倒序长度 9 的主分析纳入 53 款；`yi:6b` 的该端点 0/1 有效。按错规则纳入 54 款。其余列的模型数如表。','']
        if task=='cvlt_word_list':lines+=['`deepseek-r1:7b` 学习总分只有 2/5 回忆试次有效，端点覆盖率 0.40；依冻结门槛保留逐模型值但不入主面板。其余受影响 CVLT 端点同样在上表逐项标出 53/54 的差别。','']
        if task=='source_monitoring':lines+=['`deepseek-r1:7b` 三条件来源作答覆盖率分别为 indirect 35/48、direct_agent_last 26/48、direct_agent_first 30/48，均低于 0.80。主面板三条件各纳入 53 款，按错敏感性纳入 54 款。','']
        if task=='false_belief':
            allcorrect=[r for r in load('layer1/fb_all_correct_counts.csv') if r['rule']=='valid_only' and r['panel']=='all54']
            lines+=['14 道主分析题全对的模型数：'+ '；'.join(f"{r['endpoint'].split('|')[0]} {r['all_correct_models']}/{r['included_models']}" for r in allcorrect)+'。','']
        if task=='post_decision_wagering':lines+=['高注减低注差只在 17/54 款有定义。其余 37 款全部 240 对均有效，但 29 款只下低注、8 款只下高注，因此未定义；这些不是技术失败，也未填零。','']
    lines+=['## 附加冻结敏感性与补充参照','',
            '| 范式及口径 | 端点 | 人类均值 | 模型均值 | 模型−人类 [95% CI] | 纳入模型 |',
            '|---|---|---:|---:|---|---:|']
    extra=[('n_back','nb_r2','primary',['load_2_target','load_2_lure','load_2_control','load_3_target','load_3_lure','load_3_control']),
           ('false_belief','no_location_excluded','primary',['session|FB_accuracy','session|TB_accuracy','session|main_accuracy']),
           ('cvlt_word_list','valid_only','all_diagnoses',['learning_total','learning_slope','recognition_hit_rate','recognition_false_alarm_rate']),
           ('source_monitoring','valid_only','exp2',['indirect_accuracy','direct_agent_first_accuracy','direct_agent_first_minus_indirect'])]
    for task,rule,cohort,keys in extra:
        for endpoint in keys:
            r=index[(task,rule,endpoint,cohort,'all54')]
            lines.append(f"| {task} / {rule} / {cohort} | `{endpoint}` | {num(r['human_mean'])} | {num(r['model_mean'])} | {num(r['mean'])} [{num(r['lower'])}, {num(r['upper'])}] | {r['models']} |")
    lines+=['','NB R2 把无效作为 NO MATCH；本批模型 NB 全部技术有效，模型率与主规则相同，配对人类率改为 P9 R2。FB 排除 no-location 是额外敏感性；主表仍按答错。CVLT 全诊断组改变的是人类参照，完整逐端点结果保留在源表。','',
            '## 19 款与 51 款面板快照','',
            '以下每任务列一项用于速查；所有端点的面板比较完整保留在源表中。数字为各面板模型等权均值（纳入模型数），均为 valid_only。', '',
            '| 范式 | 端点 | all54 | primary19 | direct51 |','|---|---|---:|---:|---:|']
    anchors={'digit_span':'forward_minus_backward','n_back':'load_2_dprime_episode_mean','operation_span':'position_accuracy',
             'stroop':'error_interference','flanker':'error_interference','go_nogo':'dprime','cvlt_word_list':'learning_total',
             'drm_false_memory':'no_warning|delta_dprime','source_monitoring':'indirect_accuracy','false_belief':'session|main_accuracy',
             'epitome_tom':'mental_level_slope','confidence_calibration':'official_bias','post_decision_wagering':'high_minus_low_accuracy'}
    for task,title in TASKS:
        endpoint=anchors[task];values=[]
        for pname in ('all54','primary19','direct51'):
            r=index[(task,'valid_only',endpoint,'primary',pname)]
            values.append(f"{num(r['model_mean'])} ({r['models']})")
        lines.append(f"| {title} | `{endpoint}` | "+' | '.join(values)+' |')
    lines+=['', '## 核对记录','',
            f'- 主表包含 {len(TASKS)} 个任务、{count} 个端点行。显示值均直接取已生成的 `panel_differences.csv`，未从正式记录重算。',
            '- 事后客观纠错：SM Exp2 初算错误共用了 Exp1 的数字答案码。按 Exp2 作者计分公式更正后，indirect 人类准确率由 10.79% 更正为 57.24%（129 人），direct_agent_first 由 13.81% 更正为 50.37%（124 人）；两条件差由 +3.02 个百分点更正为 −6.87 个百分点。上表与下游比较已使用更正值。原始回答、模型分数、Exp1 主分析和结构结果不变；初算保留，正式模型计分未重复执行。证据见 [SM_EXP2_CODING_EVIDENCE.md](SM_EXP2_CODING_EVIDENCE.md)。',
            '- 本文没有修改冻结分析规则、实验范围或论文主张；所有逐模型低覆盖和未定义值仍保留在源表。','']
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text('\n'.join(lines))
    print(DEST)
    print(f'{len(TASKS)} tasks; {count} primary endpoint rows')

if __name__=='__main__':main()
