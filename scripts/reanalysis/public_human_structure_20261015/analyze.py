"""Prespecified cohort-specific score structures; no pooled fictitious subjects."""
import argparse
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
GROUPS = dict(digit_span='WM', n_back='WM', operation_span='WM', stroop='CC',
              flanker='CC', go_nogo='CC', cvlt='EM', drm='EM', source_monitoring='EM',
              false_belief='ToM', epitome_tom='ToM', confidence_calibration='MC',
              post_decision_wagering='MC')
METRICS = ['within','between','delta','pc1','mean_correlation']


def jsonable(obj):
    if isinstance(obj, dict): return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray): return jsonable(obj.tolist())
    if isinstance(obj, (float,np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer): return int(obj)
    return obj


def save_json(path,obj):
    path.write_text(json.dumps(jsonable(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def mask_for(tasks, groups):
    i,j = np.triu_indices(len(tasks),1)
    labels = np.array([groups[t] for t in tasks])
    return i,j,labels[i] == labels[j]


def metrics(corr, within):
    p = corr.shape[-1]
    i,j = np.triu_indices(p,1)
    edges = corr[...,i,j]
    shape = edges.shape[:-1]
    w = edges[...,within].mean(axis=-1) if within.any() else np.full(shape,np.nan)
    b = edges[...,~within].mean(axis=-1) if (~within).any() else np.full(shape,np.nan)
    return np.stack([w,b,w-b,np.linalg.eigvalsh(corr)[...,-1]/p,edges.mean(axis=-1)],axis=-1)


def bootstrap(x, families, draws, seed):
    n,p = x.shape
    if families is None:
        units = np.eye(n)
    else:
        codes, names = pd.factorize(families,sort=True)
        units = (np.arange(len(names))[:,None] == codes[None,:]).astype(float)
    k = len(units)
    rng = np.random.default_rng(seed)
    outer = (x[:,:,None]*x[:,None,:]).reshape(n,p*p)
    result = np.full((draws,p,p),np.nan)
    for start in range(0,draws,250):
        b = min(250,draws-start)
        weights = rng.multinomial(k,np.full(k,1/k),size=b) @ units
        total = weights.sum(axis=1)
        means = weights @ x / total[:,None]
        cov = (weights @ outer / total[:,None]).reshape(b,p,p)-means[:,:,None]*means[:,None,:]
        var = np.diagonal(cov,axis1=1,axis2=2)
        valid = (var > 1e-15).all(axis=1)
        sd = np.sqrt(var[valid])
        result[start:start+b][valid] = np.clip(cov[valid]/(sd[:,:,None]*sd[:,None,:]),-1,1)
    return result


def partitions(corr,tasks,groups):
    labels = [groups[t] for t in tasks]
    if len(set(labels)) < 2: return []
    seen, rows = set(), []
    i,j,base = mask_for(tasks,groups)
    if not base.any() or base.all(): return []
    for perm in set(itertools.permutations(labels)):
        key = tuple(sorted(tuple(k for k,v in enumerate(perm) if v == label) for label in set(perm)))
        if key in seen: continue
        seen.add(key)
        w = np.array(perm)[i] == np.array(perm)[j]
        rows.append({'blocks':' / '.join(','.join(tasks[k] for k in block) for block in key),
                     'delta':float(corr[i[w],j[w]].mean()-corr[i[~w],j[~w]].mean()),
                     'proposed':bool(np.array_equal(w,base))})
    return sorted(rows,key=lambda r:r['blocks'])


def population(frame,tasks,groups,population_name,directory,draws,seed):
    directory.mkdir(parents=True,exist_ok=True)
    assert frame.index.is_unique
    selected = frame.dropna(subset=tasks)
    n = len(selected)
    selected.to_csv(directory/f'{population_name}_scores.csv')
    result = {'n_available':len(frame),'n_complete':n,'tasks':tasks,
              'n_by_task':frame[tasks].notna().sum().to_dict(),'ids':selected.index.tolist()}
    if n <= len(tasks) or (selected[tasks].std() <= 1e-12).any():
        result.update(status='not_estimable',reason='too few complete cases or constant task',
                      sd=selected[tasks].std().to_dict())
        return result,None
    x = selected[tasks].to_numpy(float)
    assert np.isfinite(x).all()
    compute_start = time.perf_counter()
    family = selected.family.to_numpy() if population_name == 'llm' else None
    i,j,within = mask_for(tasks,groups)
    c = np.corrcoef(x,rowvar=False)
    spearman = np.corrcoef(rankdata(np.round(x,10),axis=0),rowvar=False)
    boots = bootstrap(x,family,draws,seed)
    good = np.isfinite(boots).all(axis=(1,2))
    bs = np.full((draws,len(METRICS)),np.nan)
    bs[good] = metrics(boots[good],within)
    point = metrics(c,within)
    ci = []
    for z in range(len(METRICS)):
        vals = bs[:,z][np.isfinite(bs[:,z])]
        ci.append(np.quantile(vals,[.025,.975]) if len(vals) else [np.nan,np.nan])
    result.update(status='ok',families=None if family is None else len(set(family)),
                  n_boot=draws,n_boot_valid=int(good.sum()),seed=seed,
                  groups=groups,within_edges=int(within.sum()),between_edges=int((~within).sum()),
                  metrics={name:{'value':point[z],'ci95':ci[z]} for z,name in enumerate(METRICS)},
                  spearman=dict(zip(METRICS,metrics(spearman,within))))
    result['compute_seconds'] = time.perf_counter()-compute_start
    pd.DataFrame(c,index=tasks,columns=tasks).to_csv(directory/f'{population_name}_pearson.csv')
    pd.DataFrame(spearman,index=tasks,columns=tasks).to_csv(directory/f'{population_name}_spearman.csv')
    desc = selected[tasks].describe().T
    desc['fraction_zero'] = selected[tasks].eq(0).mean()
    desc['fraction_one'] = selected[tasks].eq(1).mean()
    desc.to_csv(directory/f'{population_name}_descriptives.csv')
    edge_ci = np.quantile(boots[good][:,i,j],[.025,.975],axis=0) if good.any() else np.full((2,len(i)),np.nan)
    pd.DataFrame([{'task_a':tasks[a],'task_b':tasks[b],'within_group':bool(w),
                   'pearson':c[a,b],'lower':edge_ci[0,k],'upper':edge_ci[1,k],
                   'spearman':spearman[a,b]} for k,(a,b,w) in enumerate(zip(i,j,within))]
                ).to_csv(directory/f'{population_name}_edges.csv',index=False)
    part = partitions(c,tasks,groups)
    if part:
        pd.DataFrame(part).to_csv(directory/f'{population_name}_partitions.csv',index=False)
        delta = result['metrics']['delta']['value']
        result['partition_tail'] = sum(r['delta'] >= delta-1e-12 for r in part)/len(part)
        result['partition_count'] = len(part)
    np.savez_compressed(directory/f'{population_name}_bootstrap.npz',metrics=bs,edges=boots[:,i,j])
    return result,bs


def configs(inputs):
    """All variants are explicit before reading any new structural effects."""
    cnp = pd.read_csv(inputs/'cnp_scores.csv').set_index('participant_id')
    ball = pd.read_csv(inputs/'ball_scores.csv').set_index('participant_id')
    strict = pd.read_csv(inputs/'llm_strict.csv').set_index('model')
    native = pd.read_csv(inputs/'llm_native.csv').set_index('model')
    variants = []
    core = ['digit_span','stroop','flanker','go_nogo','cvlt']
    strata = [('healthy',cnp[cnp.clinical_group.eq('CONTROL')]),('all_diagnoses',cnp)]
    strata += [(d.lower(),cnp[cnp.clinical_group.eq(d)]) for d in sorted(cnp.clinical_group.unique()) if d != 'CONTROL']
    cnp_representations = {
        'performance':({},strict),
        'condition_weighted':({'stroop':'stroop_model_weighted_accuracy','flanker':'flanker_model_weighted_accuracy',
                               'go_nogo':'go_nogo_model_weighted_accuracy'},strict),
        'native_rt':({'digit_span':'digit_span_mean_length','stroop':'stroop_native_negative_conflict_rt',
                       'flanker':'flanker_native_negative_conflict_rt','go_nogo':'go_nogo_dprime'},native),
    }
    for stratum,h in strata:
        for name,(remap,l) in cnp_representations.items():
            hh = h.copy()
            for task,col in remap.items(): hh[task] = h[col]
            for tasks,suffix in [(core,'five'),(core+['source_monitoring'],'six')]:
                for panel in [55,20]:
                    ll = l if panel == 55 else l[l.primary20.eq(1)]
                    variants.append((f'cnp_{stratum}_{name}_{suffix}_m{panel}',hh,ll,tasks,
                                     {t:GROUPS[t] for t in tasks},'related paradigms; protocols and endpoints differ'))
    ball_tasks = ['operation_span','drm','source_monitoring']
    ball_strata = [('pooled',ball)]+list(ball.groupby('warning',sort=True))
    for stratum,h in ball_strata:
        for name,col,l in [('lure_contrast','drm',native),('accuracy56','drm_accuracy_analyzed56',strict),
                           ('balanced_accuracy','drm_balanced_accuracy',strict)]:
            hh = h.copy()
            hh['drm'] = h[col]
            # All non-DRM endpoints use ordinary performance accuracy in both representations.
            ll = strict.copy()
            ll['drm'] = l.drm
            for panel in [55,20]:
                lp = ll if panel == 55 else ll[ll.primary20.eq(1)]
                variants.append((f'ball_{stratum}_{name}_m{panel}',hh,lp,ball_tasks,
                                 {t:GROUPS[t] for t in ball_tasks},'one WM / two EM indicators; source protocols differ'))
    for file in sorted((inputs/'metacognition').glob('*_wide.csv')):
        f = pd.read_csv(file,index_col=0)
        # Every provided endpoint is retained; single-task-only fields cannot define cross-task structure.
        f = f.dropna(axis=1,how='all')
        if f.shape[1] < 2: continue
        tasks = f.columns.tolist()
        variants.append((file.stem,f,None,tasks,{t:'MC_reference' for t in tasks},
                         'human-only original task battery; no corresponding LLM tasks'))
    return variants


def coverage(inputs,out):
    source_tasks = {
        'SRO_existing':['digit_span','n_back','stroop','flanker','go_nogo'],
        'CNP':['digit_span','stroop','flanker','go_nogo','cvlt','source_monitoring'],
        'Ball':['operation_span','drm','source_monitoring']}
    rows=[]
    for a,b in itertools.combinations(GROUPS,2):
        sources=[s for s,ts in source_tasks.items() if a in ts and b in ts]
        rows.append(dict(task_a=a,task_b=b,human_cohorts='|'.join(sources),
                         same_person_source_available=bool(sources),
                         caveat='paradigm-family coverage; not identical protocols/endpoints'))
    pd.DataFrame(rows).to_csv(out/'human_task_pair_coverage.csv',index=False)
    save_json(out/'coverage.json',{'paradigm_union':sorted(set(sum(source_tasks.values(),[]))),
             'observed_source_edges':sum(r['same_person_source_available'] for r in rows),
             'full13_edges':len(rows),'sources':source_tasks,
             'not_a_complete_matrix':True,
             'metacognition':'original batteries summarized separately, not relabeled CogArena tasks',
             'tom':'no participant linkage to these cohorts; task-wise references only',
             'SRO_reused':str(ROOT/'results/reanalysis/sro_structure_20261015/analysis/summary.json')})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--draws',type=int,default=5000)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    t0,c0=time.perf_counter(),time.process_time()
    jobs=configs(args.inputs)
    save_json(args.out/'analysis_inventory.json',[{'name':n,'tasks':ts,'groups':g,'scope':scope,
               'model_comparison':l is not None} for n,h,l,ts,g,scope in jobs])
    summary,rows={},[]
    for k,(name,h,l,tasks,groups,scope) in enumerate(jobs):
        folder=args.out/name
        hs,hb=population(h,tasks,groups,'human',folder,args.draws,20261015+2*k)
        item={'scope':scope,'human':hs}
        if l is not None:
            ls,lb=population(l,tasks,groups,'llm',folder,args.draws,20261015+2*k+1)
            item['llm']=ls
            if hb is not None and lb is not None:
                diff=hb-lb
                dm={}
                for z,metric in enumerate(METRICS):
                    good=np.isfinite(diff[:,z])
                    dm[metric]={'value':hs['metrics'][metric]['value']-ls['metrics'][metric]['value'],
                                'ci95':np.quantile(diff[good,z],[.025,.975]) if good.any() else [np.nan,np.nan],
                                'n_valid':int(good.sum())}
                item['human_minus_llm']=dm
        summary[name]=item
        save_json(folder/'summary.json',item)
        row={'analysis':name,'scope':scope,'human_n':hs['n_complete'],'human_status':hs['status']}
        for group in ['human','llm']:
            if group not in item:continue
            pop=item[group]
            row[group+'_n']=pop['n_complete']
            if 'metrics' in pop:
                for m,v in pop['metrics'].items():
                    row[group+'_'+m]=v['value']
                    row[group+'_'+m+'_lower'],row[group+'_'+m+'_upper']=v['ci95']
        if 'human_minus_llm' in item:
            d=item['human_minus_llm']['delta']
            row['difference_delta']=d['value']
            row['difference_lower'],row['difference_upper']=d['ci95']
        rows.append(row)
    save_json(args.out/'summary.json',summary)
    pd.DataFrame(rows).to_csv(args.out/'summary.csv',index=False)
    coverage(args.inputs,args.out)
    proc=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    save_json(args.out/'runtime.json',{'wall_seconds':time.perf_counter()-t0,'cpu_seconds':time.process_time()-c0,
              'max_rss_kib':int(proc['VmHWM'].split()[0]),'analyses':len(jobs),'draws':args.draws,
              'resampling_and_statistics_seconds':sum(v.get(g,{}).get('compute_seconds',0) for v in summary.values() for g in ['human','llm']),
              'node':os.uname().nodename,'gpu':0,'model_calls':0,'scope':'after imports; includes output serialization'})
    print(json.dumps({'analyses':len(jobs),'draws':args.draws,'output':str(args.out)}))


if __name__=='__main__': main()
