"""Frozen human54 task endpoints. This module performs no file access or model calls.

Counts called n_valid always describe the primary denominator (including the
prespecified FB no-location exception); n_scored is the rule-specific denominator.
The 0.80 gate uses primary coverage and applies only to valid_only.
"""
from collections import Counter, defaultdict
from statistics import NormalDist
import json
import math

from scripts.experiments.human13_20261017.registry import module_for
from scripts.experiments.human13_20261017.scoring import score_record
from scripts.experiments.human13_20261017 import memory

RULES = ('valid_only', 'invalid_wrong')
NO_CORRECTNESS = {'ready', 'instruction', 'confidence', 'wager', 'drm_rating', 'drm_report'}


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def difference(a, b):
    return None if a is None or b is None else a - b


def slope(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pairs) < 2:
        return None
    xbar = avg(x for x, _ in pairs)
    denom = sum((x - xbar)**2 for x, _ in pairs)
    return sum((x - xbar) * y for x, y in pairs) / denom if denom else None


def dprime(hits, signals, false_alarms, noises):
    if min(signals, noises) < 5:
        return None
    z = NormalDist().inv_cdf
    return z(min(.999, max(.001, (hits + .5) / (signals + 1)))) - z(min(.999, max(.001, (false_alarms + .5) / (noises + 1))))


def classify(record):
    if record is None:
        return 'D', 'not_run'
    if not record['technical_valid']:
        return 'B', '|'.join(record.get('protocol_errors') or ['unspecified_technical_invalid'])
    if record.get('prompt_plus_completion_exceeds_context'):
        return 'C', 'context_exceeded'
    score = record['score']
    if score['behavior_eligible']:
        return 'A', ''
    if record['step']['kind'] == 'fb_answer' and not score.get('parsed_valid'):
        return 'C', 'fb_no_location'
    if score.get('unclassifiable_response'):
        return 'C', 'drm_unclassifiable'
    raise ValueError('Unspecified technical-valid scoring failure: ' + str((record['task'], record['episode_id'], record['step']['id'])))


def prepare_records(bundle, records):
    """Include every planned slot, using None for absent records. Tests may use fixtures."""
    lookup = {}
    for raw in records:
        key = (raw['episode_id'], raw['step']['id'])
        if key in lookup:
            raise ValueError('Duplicate slot ' + str(key))
        lookup[key] = raw
    rows = []
    for ep in bundle['episodes']:
        for step in ep['steps']:
            raw = lookup.pop((ep['id'], step['id']), None)
            if raw is None:
                row = dict(task=ep['task'], episode_id=ep['id'], condition=ep['condition'], step=step,
                           score={}, category='D', reason='not_run')
            else:
                row = {k: raw.get(k) for k in ('model','task','episode_id','condition','step','technical_valid',
                    'protocol_errors','prompt_plus_completion_exceeds_context','response')}
                row['score'] = score_record(module_for(raw['task']), raw)
                row['category'], row['reason'] = classify(row)
            rows.append(row)
    if lookup:
        raise ValueError('Slots outside material bundle')
    return rows


def primary(row):
    return row['category'] == 'A' or row['reason'] == 'fb_no_location'


def eligible(row, rule):
    if row['category'] == 'A':
        return True
    if row['reason'] == 'fb_no_location':
        return rule != 'no_location_excluded'
    return (rule in ('invalid_wrong','nb_r2') and
            (row['category'] == 'B' or row['reason'] == 'context_exceeded') and
            row['step']['kind'] not in NO_CORRECTNESS)


def answer(row, rule):
    if not eligible(row, rule):
        return None
    return float(bool(row['score'].get('is_correct'))) if row['category'] == 'A' else 0.


def positive(row, rule, label):
    if not eligible(row, rule):
        return None
    if row['category'] == 'A':
        return float(row['score']['parsed'] == label)
    if rule == 'nb_r2':
        return 0.
    # Invalid signal trial -> miss; invalid noise trial -> false alarm.
    return float(row['step']['gold'] != label)


def kappa(pairs):
    if not pairs:
        return None
    observed = avg(a == b for a, b in pairs)
    pa, pb = avg(a for a, b in pairs), avg(b for a, b in pairs)
    chance = pa * pb + (1-pa)*(1-pb)
    return (observed-chance)/(1-chance) if chance != 1 else None


class Builder:
    def __init__(self, model, task, rule):
        self.model, self.task, self.rule = model, task, rule
        self.endpoints, self.profiles, self.blocks, self.diagnostics = [], [], [], []

    def add(self, name, value, rows, n_valid=None, n_scored=None, profile=None, cell=None):
        n = len(rows)
        v = sum(primary(r) for r in rows) if n_valid is None else n_valid
        s = sum(eligible(r, self.rule) for r in rows) if n_scored is None else n_scored
        coverage = v / n if n else None
        out = dict(task=self.task, model=self.model, rule=self.rule, value=value,
                   n_valid=v, n_planned=n, n_scored=s, coverage=coverage,
                   included=value is not None and (self.rule != 'valid_only' or (coverage is not None and coverage >= .8)))
        if profile is None:
            out['endpoint'] = name
            out['is_direction'] = (name in ('forward_minus_backward','short_minus_long','error_interference','learning_5_minus_1','warning_minus_no_warning_lure','direct_minus_indirect','direct_agent_first_minus_indirect','high_minus_low_accuracy') or any(name.endswith(x) for x in ('_lure_minus_control','_hit_minus_lure','|lure_minus_new','_mental_minus_control','|FB_minus_TB')))
            self.endpoints.append(out)
        else:
            out.update(profile=profile, cell=str(cell))
            self.profiles.append(out)
        return value

    def rate(self, name, rows, fn=answer, **kwargs):
        vals = [fn(r,self.rule) for r in rows]
        return self.add(name, avg(x for x in vals if x is not None), rows,
                        n_scored=sum(x is not None for x in vals), **kwargs)

    def signal(self, name, rows, label, signal):
        sig, noise = [r for r in rows if signal(r)], [r for r in rows if not signal(r)]
        ss = [positive(r, self.rule, label) for r in sig]
        ns = [positive(r, self.rule, label) for r in noise]
        ss, ns = [x for x in ss if x is not None], [x for x in ns if x is not None]
        return self.add(name, dprime(sum(ss),len(ss),sum(ns),len(ns)), rows, n_scored=len(ss)+len(ns))


def select(rows, **meta):
    return [r for r in rows if all(r['step']['meta'].get(k) == v for k,v in meta.items())]


def grouped(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return dict(out)


def paired_kappas(b, rows):
    for ep, rs in grouped(rows, lambda r:r['episode_id']).items():
        pairs = []
        for r in rs:
            a = answer(r,b.rule)
            source = r['step']['meta'].get('source_row',{})
            h = source.get('correct')
            if a is not None and h not in (None,'','nan'):
                h = str(h).lower()
                pairs.append((int(a), int(h == 'true' or (h != 'false' and float(h) == 1))))
        b.add('kappa_correct|' + ep,kappa(pairs),rs,n_scored=len(pairs))


def weighted_accuracy(b, rows, key, name):
    """Fixed design cell weights, with no reweighting around missing cells."""
    groups = grouped(rows,key)
    vals = [(avg(answer(r,b.rule) for r in rs if eligible(r,b.rule)),len(rs)) for rs in groups.values()]
    value = sum(v*n for v,n in vals)/sum(n for _,n in vals) if vals and all(v is not None for v,_ in vals) else None
    return b.add(name,value,rows)


def wm_cc(b, rows):
    rows = [r for r in rows if r['step']['meta'].get('stage') == 'test' and r['step']['kind'] != 'ready']
    task = b.task
    if task == 'digit_span':
        for (direction,length),rs in sorted(grouped(rows,lambda r:(r['step']['meta']['direction'],r['step']['meta']['length'])).items()):
            value=b.rate(f'{direction}_length_{length}',rs)
            b.add('',value,rs,profile='direction_length',cell=f'{direction}|{length}')
        b.rate('position_accuracy',rows,lambda r,rule: r['score']['partial_credit'] if r['category']=='A' else (0. if eligible(r,rule) else None))
        f,rv = select(rows,direction='forward'),select(rows,direction='backward')
        b.add('forward_minus_backward',difference(b.rate('forward_accuracy',f),b.rate('backward_accuracy',rv)),rows)
        common=[r for r in rows if 3 <= r['step']['meta']['length'] <= 7]
        short=[r for r in common if r['step']['meta']['length'] <= 4]
        long=[r for r in common if r['step']['meta']['length'] >= 5]
        b.add('short_minus_long',difference(b.rate('short_accuracy',short),b.rate('long_accuracy',long)),common)
        weighted_accuracy(b,common,lambda r:(r['step']['meta']['direction'],r['step']['meta']['length']),'structure_sro')
        paired_kappas(b,rows)
    elif task == 'operation_span':
        recall=[r for r in rows if r['step']['kind']=='ospan_recall']
        def pos(name,rs):
            used=[r for r in rs if eligible(r,b.rule)]
            n=sum(len(r['step']['gold']) for r in used)
            value=sum(r['score'].get('recall_correct_positions',0) if r['category']=='A' else 0 for r in used)/n if n else None
            return b.add(name,value,rs)
        pos('position_accuracy',recall)
        for v,rs in sorted(grouped(recall,lambda r:r['step']['meta']['version']).items()): pos(f'version_{v}_position_accuracy',rs)
        for length,rs in sorted(grouped(recall,lambda r:len(r['step']['gold'])).items()): pos(f'length_{length}_position_accuracy',rs)
        b.rate('arithmetic_accuracy',[r for r in rows if r['step']['kind']=='ospan_math'])
    elif task == 'n_back':
        loadacc=[]
        for load,rs in sorted(grouped(rows,lambda r:r['step']['meta']['load']).items()):
            rates={}
            for kind in ('target','lure','control','nontarget'):
                sub=[r for r in rs if (r['step']['meta']['condition']!='target' if kind=='nontarget' else r['step']['meta']['condition']==kind)]
                rates[kind]=b.rate(f'load_{load}_{kind}',sub,lambda r,rule:positive(r,rule,'MATCH'))
                if kind!='nontarget': b.add('',rates[kind],sub,profile='load_kind',cell=f'{load}|{kind}')
            dps=[]; dp_rows=[]; primary_dp_rows=[]; block_accuracies=[]
            for ep,block in grouped(rs,lambda r:r['episode_id']).items():
                main_targets=sum(primary(r) and r['step']['meta']['condition']=='target' for r in block)
                main_noise=sum(primary(r) and r['step']['meta']['condition']!='target' for r in block)
                if min(main_targets,main_noise)>=5: primary_dp_rows.extend(block)
                block_accuracies.append(avg(answer(r,b.rule) for r in block if eligible(r,b.rule)))
                tg=[positive(r,b.rule,'MATCH') for r in block if r['step']['meta']['condition']=='target']
                nt=[positive(r,b.rule,'MATCH') for r in block if r['step']['meta']['condition']!='target']
                tg,nt=[x for x in tg if x is not None],[x for x in nt if x is not None]
                dp=dprime(sum(tg),len(tg),sum(nt),len(nt))
                if dp is not None: dps.append(dp); dp_rows.extend(block)
                for kind in ('target','lure','control'):
                    sub=select(block,condition=kind); codes=[positive(r,b.rule,'MATCH') for r in sub]
                    b.blocks.append(dict(model=b.model,rule=b.rule,episode_id=ep.removeprefix('nb_compact_'),load=load,kind=kind,
                        match_n=sum(x for x in codes if x is not None),denominator=sum(x is not None for x in codes),planned=len(sub)))
            b.add(f'load_{load}_dprime_episode_mean',avg(dps),rs,n_valid=sum(primary(r) for r in primary_dp_rows),n_scored=sum(eligible(r,b.rule) for r in dp_rows))
            b.add(f'load_{load}_lure_minus_control',difference(rates['lure'],rates['control']),[r for r in rs if r['step']['meta']['condition']!='target'])
            b.add(f'load_{load}_hit_minus_lure',difference(rates['target'],rates['lure']),[r for r in rs if r['step']['meta']['condition']!='control'])
            b.rate(f'load_{load}_accuracy',rs)
            loadacc.append(avg(block_accuracies) if all(v is not None for v in block_accuracies) else None)
            pairs=[]
            for r in rs:
                human=r['step']['meta']['human_response']; model=positive(r,b.rule,'MATCH')
                if model is not None and human is not None: pairs.append((model,int(human)))
            b.add(f'load_{load}_kappa_response',kappa(pairs),rs,n_scored=len(pairs))
        b.add('structure_sro',avg(loadacc) if len(loadacc)==2 and all(v is not None for v in loadacc) else None,rows)
    elif task in ('stroop','flanker'):
        vals={}
        for cond,rs in sorted(grouped(rows,lambda r:r['step']['meta']['condition']).items()): vals[cond]=b.rate(cond+'_accuracy',rs)
        b.add('error_interference',difference(vals.get('congruent'),vals.get('incongruent')),select(rows,condition='congruent')+select(rows,condition='incongruent'))
        b.add('structure_sro',avg(vals.values()) if vals and all(v is not None for v in vals.values()) else None,rows)
        if task=='stroop':
            sub=select(rows,condition='incongruent')
            b.rate('word_response_on_incongruent',sub,lambda r,rule: float(r['score'].get('parsed')==r['step']['meta']['stimulus_word'].upper()) if r['category']=='A' else None)
        else:
            for (cue,cond),rs in sorted(grouped(rows,lambda r:(r['step']['meta']['cue'],r['step']['meta']['condition'])).items()):
                value=b.rate(f'{cue}|{cond}_accuracy',rs)
                b.add('',value,rs,profile='cue_condition',cell=f'{cue}|{cond}')
            sub=[r for r in rows if r['step']['meta']['condition']!='neutral']
            b.add('structure_sro_no_neutral',avg([vals['congruent'],vals['incongruent']]) if vals.get('congruent') is not None and vals.get('incongruent') is not None else None,sub)
        paired_kappas(b,rows)
    elif task=='go_nogo':
        go,no=select(rows,condition='go'),select(rows,condition='nogo')
        hit=b.rate('hit_rate',go,lambda r,rule:positive(r,rule,'PRESS'))
        fa=b.rate('false_alarm_rate',no,lambda r,rule:positive(r,rule,'PRESS'))
        b.add('omission_rate',None if hit is None else 1-hit,go)
        b.add('balanced_accuracy',None if hit is None or fa is None else (hit+1-fa)/2,rows)
        b.signal('dprime',rows,'PRESS',lambda r:r['step']['meta']['condition']=='go')
        weighted_accuracy(b,rows,lambda r:r['step']['meta']['condition'],'structure_sro')
        paired_kappas(b,rows)


def cvlt(b, rows):
    lookup={(r['episode_id'],r['step']['id']):r for r in rows}
    trials=[]
    for r in rows:
        if r['step']['kind']!='cvlt_recall': continue
        follow=lookup[(r['episode_id'],r['step']['meta']['followup_id'])]
        primary_ok=primary(r) and primary(follow)
        used=eligible(r,b.rule) and eligible(follow,b.rule)
        score=None
        if used:
            words=(r['score']['parsed'] if r['category']=='A' else [])+(follow['score']['parsed'] if follow['category']=='A' else [])
            score=memory.score_step(r['step'],json.dumps(words))
        trials.append(dict(row=r,follow=follow,main=primary_ok,used=used,score=score))
    def trial_endpoint(name,ts,metric='proportion',scale=16):
        values=[t['score'][metric] for t in ts if t['used']]
        v=avg(values)
        return b.add(name,None if v is None else v*scale,[t['row'] for t in ts],
                     n_valid=sum(t['main'] for t in ts),n_scored=sum(t['used'] for t in ts))
    learning=[t for t in trials if t['row']['step']['meta']['phase']=='learning_recall']
    y=[]
    for i in range(1,6): y.append(trial_endpoint(f'learning_{i}',[t for t in learning if t['row']['step']['meta']['trial']==i]))
    trial_endpoint('learning_total',learning,scale=80)
    b.add('learning_slope',slope(range(1,6),y),[t['row'] for t in learning],n_valid=sum(t['main'] for t in learning),n_scored=sum(t['used'] for t in learning))
    b.add('learning_5_minus_1',difference(y[-1],y[0]),[learning[0]['row'],learning[-1]['row']],n_valid=sum(t['main'] for t in (learning[0],learning[-1])),n_scored=sum(t['used'] for t in (learning[0],learning[-1])))
    for phase in ('interference_recall','short_free','long_free','short_cued','long_cued'):
        trial_endpoint(phase,[t for t in trials if t['row']['step']['meta']['phase']==phase])
    for name,ts in [('free_intrusions',[t for t in trials if t['row']['step']['meta']['phase'] not in ('short_cued','long_cued')]),('cued_intrusions',[t for t in trials if t['row']['step']['meta']['phase'] in ('short_cued','long_cued')]),('repetitions',trials)]:
        vals=[len(t['score']['intrusions']) if 'intrusions' in name else t['score']['repetitions'] for t in ts if t['used']]
        # Count endpoints have no bounded full score; sum only observed scored trials.
        b.add(name,sum(vals) if vals else None,[t['row'] for t in ts],n_valid=sum(t['main'] for t in ts),n_scored=len(vals))
    rec=[r for r in rows if r['step']['kind']=='cvlt_recognition']
    for old,name in ((True,'recognition_hit_rate'),(False,'recognition_false_alarm_rate')):
        b.rate(name,[r for r in rec if r['step']['gold']==old],lambda r,rule:positive(r,rule,True))
    b.signal('recognition_dprime',rec,True,lambda r:r['step']['gold'])
    for kind,rs in grouped(rec,lambda r:r['step']['meta']['source_item_type']).items(): b.rate('recognition_type_'+kind,rs,lambda r,rule:positive(r,rule,True))


def drm(b,rows):
    for warned,group in sorted(grouped([r for r in rows if r['step']['kind']=='drm_recognition'],lambda r:r['step']['meta']['warned']).items()):
        w='warning' if warned else 'no_warning'
        primary_rows=[r for r in group if r['step']['meta']['primary56']]
        rates={}; counts={}
        for kind,rs in grouped(group,lambda r:r['step']['meta']['probe_class']).items():
            v=b.rate(w+'|'+{'old':'hit_rate','CL':'lure_false_alarm_rate','newunrelated':'new_false_alarm_rate'}.get(kind,kind+'_false_alarm_rate'),rs,lambda r,rule:positive(r,rule,True))
            rates[kind]=v; codes=[positive(r,b.rule,True) for r in rs]; codes=[x for x in codes if x is not None]
            counts[kind]=(sum(codes),len(codes))
            if kind in ('old','CL','newunrelated'): b.add('',v,rs,profile='warning_probe',cell=w+'|'+kind)
        h,l,n=[rates.get(k) for k in ('old','CL','newunrelated')]
        sub_lure=[r for r in primary_rows if r['step']['meta']['probe_class']=='CL']
        sub_new=[r for r in primary_rows if r['step']['meta']['probe_class']=='newunrelated']
        # Author Δd′ = z(corrected lure FA) − z(corrected unrelated FA).
        lk,ln=counts.get('CL',(0,0)); nk,nn=counts.get('newunrelated',(0,0))
        b.add(w+'|delta_dprime',dprime(lk,ln,nk,nn) if counts.get('old',(0,0))[1]>=5 else None,primary_rows)
        b.add(w+'|native',difference(n,l),sub_lure+sub_new)
        b.rate(w+'|accuracy_analyzed56',primary_rows)
        b.add(w+'|balanced_accuracy',(h+(1-l)+(1-n))/3 if all(x is not None for x in (h,l,n)) else None,primary_rows)
        b.add(w+'|lure_minus_new',difference(l,n),sub_lure+sub_new)
        for noise in ('CL','newunrelated'):
            signal_noise=[r for r in primary_rows if r['step']['meta']['probe_class'] in ('old',noise)]
            b.signal(w+'|dprime_'+noise,signal_noise,True,lambda r:r['step']['meta']['probe_class']=='old')
        lookup={(r['episode_id'],r['step']['id']):r for r in group}
        reports=[r for r in rows if r['step']['kind']=='drm_report' and r['condition']==group[0]['condition']]
        labels=Counter(); inconsistent=0
        for r in reports:
            rec=lookup[(r['episode_id'],r['step']['meta']['recognition_id'])]
            if not eligible(rec,b.rule) or r['category']!='A': continue
            # Invalid recognition receives no behavioral report (UNCLASSIFIABLE).
            if rec['category']!='A': continue
            valid_labels={'FAMILIAR','REMEMBER','GUESS'} if rec['score']['parsed'] else {'UNFAMILIAR','RECALL_TO_REJECT','WOULD_REMEMBER'}
            label=r['score']['parsed']
            if label in valid_labels: labels[label]+=1
            else: inconsistent+=1
        for label in ('FAMILIAR','REMEMBER','GUESS','UNFAMILIAR','RECALL_TO_REJECT','WOULD_REMEMBER'):
            b.diagnostics.append(dict(task=b.task,model=b.model,rule=b.rule,metric=w+'|report_'+label,count=labels[label]))
        b.diagnostics.append(dict(task=b.task,model=b.model,rule=b.rule,metric=w+'|report_inconsistent',count=inconsistent))
        ratings=[r for r in rows if r['step']['kind']=='drm_rating' and r['condition']==group[0]['condition']]
        b.rate(w+'|meaning_rating',ratings,lambda r,rule:r['score']['parsed'] if r['category']=='A' else None)
    e={r['endpoint']:r['value'] for r in b.endpoints}
    b.add('warning_minus_no_warning_lure',difference(e.get('warning|lure_false_alarm_rate'),e.get('no_warning|lure_false_alarm_rate')),[r for r in rows if r['step']['kind']=='drm_recognition' and r['step']['meta']['probe_class']=='CL'])


def source_memory(b,rows):
    rows=[r for r in rows if r['step']['kind']=='source_choice']
    values={}
    for variant,rs in grouped(rows,lambda r:r['step']['meta']['story_variant']).items():
        values[variant]=b.rate(variant+'_accuracy',rs)
        for item,sub in grouped(rs,lambda r:r['step']['meta']['source_row']).items(): b.rate('',sub,profile='item_condition',cell=f'{variant}|{item}')
        for (gold,chosen),sub in grouped([r for r in rs if r['category']=='A'],lambda r:(r['step']['gold'],r['score']['parsed'])).items():
            b.diagnostics.append(dict(task=b.task,model=b.model,rule=b.rule,metric=f'confusion|{variant}|{gold}|{chosen}',count=len(sub)))
    direct=[v for k,v in values.items() if k!='indirect']
    b.add('direct_minus_indirect',difference(avg(direct) if len(direct)==2 and all(v is not None for v in direct) else None,values.get('indirect')),rows)
    b.add('direct_agent_first_minus_indirect',difference(values.get('direct_agent_first'),values.get('indirect')),[r for r in rows if r['step']['meta']['story_variant'] in ('direct_agent_first','indirect')])
    b.rate('structure_ball',rows)


def fb(b,rows):
    for cond,rs in grouped(rows,lambda r:r['condition']).items():
        main=[r for r in rs if r['step']['meta']['main_analysis']]
        state_rates={}
        for state in ('F','T'):
            state_rates[state]=b.rate(cond+'|'+state+'B_accuracy',[r for r in main if r['step']['meta']['state']==state])
        b.add(cond+'|FB_minus_TB',difference(state_rates['F'],state_rates['T']),main)
        b.add(cond+'|all_correct',float(all(answer(r,b.rule)==1 for r in main)) if main else None,main)
        b.rate(cond+'|catch_accuracy',[r for r in rs if not r['step']['meta']['main_analysis']])
        b.rate(cond+'|main_accuracy',main)
        for item,sub in grouped(main,lambda r:r['step']['meta']['item_id']).items(): b.rate('',sub,profile=cond+'_items',cell=item)
        for error in ('no_location','other_location','story7_other_answer'):
            n=sum(eligible(r,b.rule) and answer(r,b.rule)==0 and (r['reason']=='fb_no_location' if error=='no_location' else r['category']=='A' and (r['step']['meta']['story']==7 if error=='story7_other_answer' else r['step']['meta']['story']!=7)) for r in rs)
            b.diagnostics.append(dict(task=b.task,model=b.model,rule=b.rule,metric=cond+'|'+error,count=n))


def epitome(b,rows):
    rows=[r for r in rows if r['step']['kind']=='tom_answer']
    levels={}
    for (kind,level),rs in sorted(grouped(rows,lambda r:(r['step']['meta']['question_type'],r['step']['meta']['question_level'])).items()):
        v=b.rate(f'{kind}|{level}_accuracy',rs); levels[kind,level]=v
        b.add('',v,rs,profile='type_level',cell=f'{kind}|{level}')
    for level in range(1,8): b.add(f'level_{level}_mental_minus_control',difference(levels.get(('mental',level)),levels.get(('control',level))),select(rows,question_level=level))
    for kind in ('mental','control'): b.add(kind+'_level_slope',slope(range(1,8),[levels.get((kind,l)) for l in range(1,8)]),select(rows,question_type=kind))
    for fmt in ('story_format','question_format'):
        for v,rs in grouped(rows,lambda r:r['step']['meta'][fmt]).items(): b.rate(fmt+'|'+v+'_accuracy',rs)
    for key,rs in grouped(rows,lambda r:tuple(r['step']['meta'][k] for k in ('story_format','question_format','question_type','question_level'))).items(): b.rate('format_type_level|'+'|'.join(map(str,key)),rs)


def metacognition(b,rows):
    cal=b.task=='confidence_calibration'
    kind='cal_answer' if cal else 'wag_answer'; followkind='confidence' if cal else 'wager'
    answers=[r for r in rows if r['step']['kind']==kind]
    lookup={(r['episode_id'],r['step']['id']):r for r in answers}
    pairs=[]
    for r in rows:
        if r['step']['kind']!=followkind: continue
        a=lookup[(r['episode_id'],r['step']['meta']['answer_step_id'])]
        pairs.append(dict(a=a,f=r,main=primary(a) and r['category']=='A',used=eligible(a,b.rule) and r['category']=='A'))
    def summarize(ps,prefix=''):
        rs=[p['a'] for p in ps]; used=[p for p in ps if p['used']]
        ys=[answer(p['a'],b.rule) for p in used]
        def add(name,value): return b.add(prefix+name,value,rs,n_valid=sum(p['main'] for p in ps),n_scored=len(used))
        add('accuracy',avg(ys))
        if cal:
            cs=[p['f']['score']['parsed'] for p in used]
            add('confidence_mean_percent',avg(cs))
            add('official_bias',avg(.5+c/200-y for c,y in zip(cs,ys)))
            right=[c for c,y in zip(cs,ys) if y]; wrong=[c for c,y in zip(cs,ys) if not y]
            auc=avg(float(a>z)+.5*float(a==z) for a in right for z in wrong)
            add('type2_auc_fixed_direction',auc)
        else:
            low=[y for p,y in zip(used,ys) if p['f']['score']['parsed']==1]
            high=[y for p,y in zip(used,ys) if p['f']['score']['parsed']==2]
            add('high_wager_rate',len(high)/len(used) if used else None)
            add('high_minus_low_accuracy',difference(avg(high),avg(low)))
            add('wager_1_accuracy',avg(low)); add('wager_2_accuracy',avg(high))
    summarize(pairs)
    for cond,ps in grouped(pairs,lambda p:p['a']['condition']).items(): summarize(ps,cond+'|')
    if cal:
        for item,ps in grouped(pairs,lambda p:p['a']['step']['meta']['item_id']).items():
            used=[p for p in ps if p['used']]; rs=[p['a'] for p in ps]
            b.add('',avg(answer(p['a'],b.rule) for p in used),rs,n_valid=sum(p['main'] for p in ps),n_scored=len(used),profile='item_accuracy',cell=item)
            b.add('',avg(p['f']['score']['parsed'] for p in used),rs,n_valid=sum(p['main'] for p in ps),n_scored=len(used),profile='item_confidence',cell=item)
        consistency=[]; main_n=0
        for item,rs in grouped(answers,lambda r:r['step']['meta']['item_id']).items():
            if len(rs)!=2: raise ValueError('CAL requires two planned positions')
            if all(primary(r) for r in rs):
                main_n+=1
                choices=[r['step']['meta']['left_option' if r['score']['parsed']=='s' else 'right_option'] for r in rs]
                consistency.append(choices[0]==choices[1])
        b.add('position_answer_consistency',avg(consistency),[answers[i] for i in range(0,len(answers),2)],n_valid=main_n,n_scored=len(consistency))
    else:
        b.rate('training_reproduction_accuracy',[r for r in rows if r['step']['kind']=='wag_training'])


def analyze_cell(model,task,bundle,records):
    rows=prepare_records(bundle,records)
    output=dict(endpoints=[],profiles=[],blocks=[],diagnostics=[])
    rules=RULES+(('nb_r2',) if task=='n_back' else ('no_location_excluded',) if task=='false_belief' else ())
    for rule in rules:
        b=Builder(model,task,rule)
        if task in ('digit_span','n_back','operation_span','stroop','flanker','go_nogo'): wm_cc(b,rows)
        elif task=='cvlt_word_list': cvlt(b,rows)
        elif task=='drm_false_memory': drm(b,rows)
        elif task=='source_monitoring': source_memory(b,rows)
        elif task=='false_belief': fb(b,rows)
        elif task=='epitome_tom': epitome(b,rows)
        elif task in ('confidence_calibration','post_decision_wagering'): metacognition(b,rows)
        else: raise ValueError(task)
        for key in output: output[key].extend(getattr(b,key))
    output['rows']=rows
    return output
