"""Render existing human54 derived statistics; no scoring or statistical fitting."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'results/reanalysis/human54_layers_20261019'
OUT = ROOT / 'paper-overleaf/h54_supplement.tex'
AUDIT = Path(__file__).parent


def read(relative):
    return list(csv.DictReader((DATA / relative).open()))


panel = read('layer1/panel_differences.csv')
structure = read('layer2/summary.csv')
main = {(r['task'], r['endpoint']): r for r in panel
        if r['rule'] == 'valid_only' and r['human_cohort'] == 'primary'
        and r['panel'] == 'all54' and r['family_mapping'] == 'merged'}
selections = [
 ('DS', 'digit_span', [('Forward, length 3','forward_length_3'),('Forward, length 9','forward_length_9'),('Backward, length 3','backward_length_3'),('Backward, length 9','backward_length_9'),('Forward $-$ backward','forward_minus_backward'),('Short $-$ long','short_minus_long')]),
 ('NB', 'n_back', [(f'{load}-back {label}',f'load_{load}_{endpoint}') for load in (2,3) for label,endpoint in [('target MATCH','target'),('lure MATCH','lure'),('control MATCH','control'),("$d'$",'dprime_episode_mean')]]),
 ('OS', 'operation_span', [('Position accuracy','position_accuracy')]),
 ('ST', 'stroop', [('Congruent accuracy','congruent_accuracy'),('Incongruent accuracy','incongruent_accuracy'),('Error interference','error_interference')]),
 ('FL', 'flanker', [('Congruent accuracy','congruent_accuracy'),('Incongruent accuracy','incongruent_accuracy'),('Neutral accuracy','neutral_accuracy'),('Error interference','error_interference')]),
 ('GN', 'go_nogo', [('Hit rate','hit_rate'),('False-alarm rate','false_alarm_rate'),("$d'$",'dprime')]),
 ('CV', 'cvlt_word_list', [('Learning trial 1 (words)','learning_1'),('Learning trial 5 (words)','learning_5'),('Learning total (words)','learning_total'),('Learning slope (words/trial)','learning_slope'),('Long-delay free (words)','long_free'),('Long-delay cued (words)','long_cued'),('Recognition hit rate','recognition_hit_rate'),('Recognition false-alarm rate','recognition_false_alarm_rate'),('Cued intrusions (words)','cued_intrusions')]),
 ('DRM', 'drm_false_memory', [('No warning: hit rate','no_warning|hit_rate'),('No warning: lure FA','no_warning|lure_false_alarm_rate'),('No warning: new-word FA','no_warning|new_false_alarm_rate'),('No warning: native','no_warning|native'),('Warning: native','warning|native'),('Warning: lure FA','warning|lure_false_alarm_rate'),('Warning $-$ no warning: lure FA','warning_minus_no_warning_lure')]),
 ('SM', 'source_monitoring', [('Indirect accuracy','indirect_accuracy'),('Direct, agent first accuracy','direct_agent_first_accuracy'),('Direct, agent last accuracy','direct_agent_last_accuracy'),('Direct $-$ indirect','direct_minus_indirect')]),
 ('FB', 'false_belief', [('Session: FB accuracy','session|FB_accuracy'),('Session: TB accuracy','session|TB_accuracy'),('Session: catch accuracy','session|catch_accuracy'),('Session: main accuracy','session|main_accuracy'),('Isolated: main accuracy','isolated|main_accuracy'),('Session, no system: main','session_no_system|main_accuracy'),('Isolated, no system: main','isolated_no_system|main_accuracy')]),
 ('EP', 'epitome_tom', [('Mental-state level 1 accuracy','mental|1_accuracy'),('Mental-state level 7 accuracy','mental|7_accuracy'),('Mental-state level slope','mental_level_slope'),('Control level slope','control_level_slope')]),
 ('CAL', 'confidence_calibration', [('Answer accuracy','accuracy'),('Confidence (0--100)','confidence_mean_percent'),('Official bias','official_bias'),('Type-2 AUROC','type2_auc_fixed_direction')]),
 ('WAG', 'post_decision_wagering', [('Answer accuracy','accuracy'),('High $-$ low accuracy','high_minus_low_accuracy')]),
]


def num(x):
    value = float(x)
    return f'{value:.4f}' if 0 < abs(value) < .0005 else f'{value:.3f}'


def interval(r, prefix=''):
    return f"[{num(r[prefix+'lower'])}, {num(r[prefix+'upper'])}]"


used = []
lines = [r'\onecolumn', r'\section{Complementary Results}', r'\label{app:h54_results}',
 r'\begin{table}[!ht]', r'\centering', r'\fontsize{8.0}{8.7}\selectfont',
 r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{0.95}',
 r'\begin{tabular}{llrrrl}', r'\toprule',
 r'Task & Source-native endpoint & Human mean ($n$) & Model mean ($n$) & Model $-$ human & 95\% CI \\',
 r'\midrule']
for task_label, task, entries in selections:
    for i, (label, endpoint) in enumerate(entries):
        r = main[(task,endpoint)]
        used.append(r)
        lines.append(f"{task_label if i == 0 else ''} & {label} & {num(r['human_mean'])} ({r['human_n']}) & {num(r['model_mean'])} ({r['models']}) & {num(r['mean'])} & {interval(r)} " + r'\\')
lines += [r'\bottomrule',r'\end{tabular}',
 r'\caption{Native behavioral endpoints, all54 primary analysis. Human means use the source-specific samples; model means weight included models equally. Intervals resample whole model families 5,000 times with the human mean fixed. Proportions retain their original units; word counts and confidence are labeled. FA denotes false alarms; DRM native is new-word FA minus lure FA. The 0.80 endpoint-coverage rule produces the displayed $n$; WAG discrimination additionally requires both wagers. This compact table covers all 13 paradigms; all 134 primary source-paired endpoint rows, every sensitivity, and individual low-coverage values remain in the accompanying CSV tables.}',
 r'\label{tab:h54_behavior_native}',r'\end{table}',r'\clearpage']

structure_audit=[]
for rule in ['valid_only','invalid_wrong']:
    is_primary=rule=='valid_only'
    title='Primary validity rule' if is_primary else 'Invalid-response sensitivity'
    lines += [rf'\subsection{{{title}}}',
      r'\begin{table}[!ht]',r'\centering',r'\fontsize{8.2}{9.8}\selectfont',r'\setlength{\tabcolsep}{5pt}',r'\renewcommand{\arraystretch}{1.07}',
      r'\begin{tabular}{llrrrrl}',r'\toprule',
      r'Cohort / representation & Panel & $n_H/n_M$ & $\delta_H$ & $\delta_M$ & $\delta_H-\delta_M$ & 95\% CI \\',r'\midrule']
    labels=[('sro','full','SRO'),('sro','no_neutral','SRO, no neutral')]+[(f'ball_{stratum}',rep,f'Ball {human}: {rlab}') for stratum,human in [('pooled','pooled'),('no_warning','no warning'),('warning','warning')] for rep,rlab in [('lure_contrast','native'),('accuracy56','accuracy (56)'),('balanced_accuracy','balanced')]]
    for st,rep,label in labels:
        for pn,panel_name in enumerate(['all54','primary19','direct51']):
            candidates=[r for r in structure if r['rule']==rule and r['structure']==st and r['representation']==rep and r['panel']==panel_name and r['family_mapping']=='merged']
            assert len(candidates)==1,(st,rep,panel_name,candidates)
            r=candidates[0];structure_audit.append(r)
            pl={'all54':'54','primary19':'19','direct51':'51'}[panel_name]
            ci=f"[{num(r['delta_difference_lower'])}, {num(r['delta_difference_upper'])}]"
            lines.append(f"{label if pn==0 else ''} & {pl} & {r['human_n']}/{r['llm_n']} & {num(r['human_delta'])} & {num(r['llm_delta'])} & {num(r['delta_difference'])} & {ci} "+r'\\')
        lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule',r'\end{tabular}',
       rf'\caption{{Structure comparisons, {title.lower()}. $\delta$ is mean within-group correlation minus mean between-group correlation; positive $\delta_H-\delta_M$ indicates a larger grouping contrast in humans. Panel labels denote the full 54, preselected 19, and 51 direct-answer models; $n_M$ is the included count. Intervals use 5,000 independent whole-person/whole-family resamples, with merged families. SRO contains DS/NB (WM) and ST/FL/GN (CC); Ball contains OS (WM) and DRM/SM (EM). Ball SM has task-family correspondence rather than identical human/model materials. All raw-family alternatives, correlation matrices, edge intervals, and eigenvalues are retained in the accompanying structure tables.}}',
       rf'\label{{tab:h54_structure_{rule}}}',r'\end{table}']
    if is_primary:
        lines += [r'\noindent The primary analysis requires at least 0.80 endpoint coverage. R1 7B has 91/144 valid SM answers and is excluded from the primary Ball matrix; its observed scores remain in the data. R1 14B has 128/144 and is included. The 19-model Ball panel therefore contains 18 models; the direct-answer panel contains 51. The three native-thinking models are flagged individually in the score files. Comparing 54 with 51 models is a panel sensitivity, not a causal estimate of thinking.']
    else:
        lines += [r'\noindent Invalid-response handling changes scoring, not the recorded history. The invalid-as-wrong sensitivity retains all models without a coverage threshold; unreported confidence, wagers, ratings and other excluded response types remain governed by the frozen paired-response rules. Additional exported sensitivities include NB invalid-as-NO-MATCH, FB no-location exclusion, and CVLT all-diagnosis human references. The source-monitoring Exp2 supplement uses the corrected author-specified answer coding; the post hoc correction and unchanged primary results are documented with the analysis release.']
    lines += [r'\clearpage']

OUT.write_text('\n'.join(lines)+'\n')
for name,rows in [('behavior_source_rows.csv',used),('structure_source_rows.csv',structure_audit),('body_source_rows.csv',list(main.values()))]:
    with (AUDIT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(AUDIT/'render_manifest.json').write_text(json.dumps({'inputs':['layer1/panel_differences.csv','layer2/summary.csv'],'behavior_rows':len(used),'structure_rows':len(structure_audit),'formal_records_read':0,'statistics_recomputed':False,'output':str(OUT)},indent=2)+'\n')
print(f'Rendered {len(used)} behavioral rows and {len(structure_audit)} structural rows.')

# Counts summarize already-scored records; no formal calls or bootstrap are rerun.
from collections import Counter
profile=read('layer1/profile_comparisons.csv')
fb=read('layer1/fb_all_correct_counts.csv')
endpoints=read('layer1/endpoints.csv')
wager=[r for r in endpoints if r['task']=='post_decision_wagering' and r['rule']=='valid_only' and r['endpoint']=='high_wager_rate']
claims={
 'body_numeric_source':'body_source_rows.csv, keyed by task/endpoint; all54/valid_only/primary/merged',
 'FB_all_correct':[r for r in fb if r['rule']=='valid_only' and r['panel']=='all54'],
 'FB_profile_status':{name:dict(Counter(r['r_status'] for r in profile if r['task']=='false_belief' and r['profile']==name and r['rule']=='valid_only' and r['human_cohort']=='primary')) for name in sorted({r['profile'] for r in profile if r['task']=='false_belief'})},
 'WAG_high_wager_rate':{'always_low':sum(float(r['value'])==0 for r in wager),'always_high':sum(float(r['value'])==1 for r in wager),'both':sum(0<float(r['value'])<1 for r in wager),'n_valid_values':sorted({int(r['n_valid']) for r in wager})},
 'formal_records_read':0, 'statistics_recomputed':False,
}
(AUDIT/'body_count_sources.json').write_text(json.dumps(claims,indent=2)+'\n')
