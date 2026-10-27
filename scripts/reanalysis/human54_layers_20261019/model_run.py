"""Run the frozen CPU-only model analysis once, after human-side archival and tests.

Only FORMAL/<model>/<task>/calls/*.json enter scoring. Each cell is released
before the next is loaded; superseded records contribute file counts only.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import resource
import time

from .model_scoring import analyze_cell
from .model_metadata import inference_metadata
from scripts.experiments.human13_20261017.score import validate_records

ROOT=Path(__file__).resolve().parents[3]
B_REASONS=('length_finish','empty_final','invalid_format_json','answer_outside_legal_options',
           'answer_outside_string_pattern','invalid_word_array','invalid_ospan_character_array',
           'no_final_phase','skipped_thinking_unconstrained_answer')
C_REASONS=('fb_no_location','drm_unclassifiable','context_exceeded')
D_REASONS=('context_overflow','no_reply_transport_failure','hard_response_error','prior_attempt_preserved','depends_on_missing_reply')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def verify_summary(summary,records,planned):
    if summary['planned_steps']!=planned or summary['committed_or_reusable']+summary['not_run_total']!=planned:
        raise ValueError('Summary committed + not-run != planned')
    valid=sum(bool(r['technical_valid']) for r in records)
    if (len(records)!=summary['committed_or_reusable'] or valid!=summary['technical_valid'] or
            len(records)-valid!=summary['technical_invalid'] or sum(summary['not_run'].values())!=summary['not_run_total'] or
            abs(summary['valid_coverage']-valid/planned)>1e-12):
        raise ValueError('Summary counts do not match calls')


def coverage(model,task,rows,records,summary,superseded):
    counts=Counter(r['category'] for r in rows)
    reasons=Counter(reason for r in records for reason in r.get('protocol_errors',[]))
    c=Counter(r['reason'] for r in rows if r['category']=='C')
    reused=Counter(r['reuse']['source_scope'] for r in records if r.get('execution_origin')=='reused')
    boundaries=Counter((r.get('budget_verdict') or {}).get('boundary_origin','unreported') for r in records)
    skips=[r for r in records if 'skipped_thinking_unconstrained_answer' in r.get('protocol_errors',[])]
    legal_skips=0
    for r in skips:
        schema=(r.get('wire_request') or {}).get('format') or {}
        legal_skips+=r['response'].strip() in schema.get('enum',[])
    whitespace=0
    for r in records:
        if r['step']['kind'] not in ('cvlt_recall','cvlt_recall_followup'): continue
        try: v=json.loads(r['response'])
        except json.JSONDecodeError: continue
        whitespace+=isinstance(v,list) and bool(v) and all(isinstance(x,str) and not x.strip() for x in v)
    out=dict(model=model,task=task,planned=len(rows),committed=len(records),
        **{k:counts[k] for k in 'ABCD'},technical_valid_coverage=(counts['A']+counts['C'])/len(rows),
        scoring_coverage=counts['A']/len(rows),reused=sum(reused.values()),reused_by_scope=json.dumps(dict(reused),sort_keys=True),
        superseded=superseded,skipped_thinking=len(skips),skipped_history_legal_option=legal_skips,
        cvlt_whitespace_array=whitespace,natural_boundary=boundaries['natural'],forced_boundary=boundaries['forced'],
        no_boundary=boundaries['none'],boundary_unreported=boundaries['unreported'])
    for key in set(B_REASONS)|set(reasons): out['B_'+key]=reasons[key]
    for key in C_REASONS: out['C_'+key]=c[key]
    for key in set(D_REASONS)|set(summary['not_run']): out['D_'+key]=summary['not_run'].get(key,0)
    if abs(out['technical_valid_coverage']-summary['valid_coverage'])>1e-12 or counts['D']!=summary['not_run_total']:
        raise ValueError('A/B/C/D coverage does not match summary')
    return out


def metadata(manifest):
    fam=json.loads((ROOT/'results/reanalysis/aplus_20260718/family_map.json').read_text())
    primary={r['model'] for r in json.loads((ROOT/'scripts/experiments/same_stimulus_20261011/SPEC.json').read_text())['model_panel']}
    native=set(manifest['spec']['inference']['native_think_models'])
    return {model:dict(family_raw=fam['raw'][model],family_merged=fam['merged'][model],
        native_thinking=model in native,panel54=True,panel19=model in primary,panel51=model not in native)
        for model in manifest['spec']['models']}


def structure_rows(endpoints):
    out=[]
    groups={}
    for e in endpoints:
        if e['rule'] not in ('valid_only','invalid_wrong'): continue
        groups.setdefault((e['model'],e['rule']),{})[(e['task'],e['endpoint'])]=e
    for (model,rule),table in groups.items():
        def emit(structure,representation,task,e):
            out.append(dict(model=model,rule=rule,structure=structure,representation=representation,
                            task=task,value=e['value'],coverage=e['coverage'],included=e['included'],
                            **inference_metadata(model,[task])))
        for task in ('digit_span','n_back','stroop','flanker','go_nogo'):
            e=table[(task,'structure_sro')]
            emit('sro','full',task,e)
            emit('sro','no_neutral',task,table[(task,'structure_sro_no_neutral')] if task=='flanker' else e)
        for cohort in ('pooled','no_warning','warning'):
            for representation,name in (('lure_contrast','native'),('accuracy56','accuracy_analyzed56'),('balanced_accuracy','balanced_accuracy')):
                emit('ball_'+cohort,representation,'operation_span',table[('operation_span','position_accuracy')])
                emit('ball_'+cohort,representation,'source_monitoring',table[('source_monitoring','structure_ball')])
                if cohort=='pooled':
                    a,c=(table[('drm_false_memory',w+'|'+name)] for w in ('no_warning','warning'))
                    e=dict(value=(a['value']+c['value'])/2 if a['value'] is not None and c['value'] is not None else None,
                           coverage=(a['n_valid']+c['n_valid'])/(a['n_planned']+c['n_planned']))
                    e['included']=e['value'] is not None and (rule!='valid_only' or e['coverage']>=.8)
                else: e=table[('drm_false_memory',cohort+'|'+name)]
                emit('ball_'+cohort,representation,'drm_false_memory',e)
    return out


def run(manifest_path,formal,output,human_references):
    if not human_references.is_file(): raise ValueError('Human-side references must be archived before opening model records')
    marker=output/'model_run_manifest.json'
    if marker.exists(): raise FileExistsError('This output already has a model-run manifest; preserve the once-run archive')
    start=time.monotonic(); manifest=json.loads(manifest_path.read_text()); meta=metadata(manifest)
    status=dict(started_at=datetime.now().astimezone().isoformat(),status='running',manifest=str(manifest_path),formal=str(formal),
        human_references=str(human_references),rules=['valid_only','invalid_wrong','nb_r2','no_location_excluded'],
        threshold=.8,seed=20261018,threads=1,models=len(meta),tasks=len(manifest['tasks']),
        rules_source='review-stage/human54_methods_20261018/HUMAN54_ANALYSIS_SPEC.md',
        synthetic_tests='scripts/reanalysis/human54_layers_20261019/tests/test_model_scoring.py')
    write_json(marker,status)
    outputs={k:[] for k in ('endpoints','profiles','blocks','diagnostics')}; cov=[]; drop=[]; completed=0
    try:
        bundles={e['task']:json.loads(Path(e['material_file']).read_text()) for e in manifest['tasks']}
        for model in manifest['spec']['models']:
            for entry in manifest['tasks']:
                task=entry['task']; folder=formal/model.replace(':','__')/task
                summary=json.loads((folder/'summary.json').read_text())
                records=[json.loads(p.read_text()) for p in sorted((folder/'calls').glob('*.json'))]
                if any(r.get('synthetic') for r in records): raise ValueError('Synthetic record in formal input')
                if any(r['model']!=model or r['task']!=task for r in records): raise ValueError('Wrong model/task folder')
                # Use the frozen entry's exact step/condition/presentation equality check.
                validate_records(records,{**manifest,'tasks':[entry]})
                verify_summary(summary,records,entry['steps_per_model'])
                cell=analyze_cell(model,task,bundles[task],records)
                superseded=sum(1 for p in (folder/'superseded').glob('*.calls.json'))
                cov.append({**coverage(model,task,cell.pop('rows'),records,summary,superseded),**meta[model]})
                for key,values in cell.items(): outputs[key].extend({**r,**meta[model]} for r in values)
                if task=='false_belief':
                    for r in records:
                        source=r.get('reuse',{}).get('source_path')
                        if source:
                            source_record=json.loads(Path(source).read_text())
                            n=source_record.get('earlier_items_dropped_for_context',source_record.get('context',{}).get('history_pairs_dropped',0))
                        else: n=r.get('context',{}).get('history_pairs_dropped',0)
                        drop.append(dict(model=model,condition=r['condition'],episode_id=r['episode_id'],step_id=r['step']['id'],
                            dropped_pairs=n,execution_origin=r.get('execution_origin'),source_path=source))
                completed+=1
            print(json.dumps(dict(model=model,completed_cells=completed,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
        write_csv(output/'coverage'/'cells.csv',cov)
        write_csv(output/'coverage'/'fb_history_pairs_dropped.csv',drop)
        write_csv(output/'layer1'/'endpoints.csv',outputs['endpoints'])
        write_csv(output/'layer1'/'profiles.csv',outputs['profiles'])
        write_csv(output/'layer1'/'nb_block_counts.csv',outputs['blocks'])
        write_csv(output/'layer1'/'diagnostics.csv',outputs['diagnostics'])
        write_csv(output/'layer2'/'structure_scores.csv',structure_rows(outputs['endpoints']))
        write_csv(output/'coverage'/'models.csv',[dict(model=m,**v) for m,v in meta.items()])
        status.update(status='complete',completed_cells=completed,elapsed_seconds=time.monotonic()-start,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            total_planned=sum(r['planned'] for r in cov),**{'total_'+k:sum(r[k] for r in cov) for k in 'ABCD'},
            output_rows={key:len(value) for key,value in outputs.items()})
        write_json(marker,status)
    except Exception as exc:
        status.update(status='failed',completed_cells=completed,error=repr(exc),elapsed_seconds=time.monotonic()-start)
        write_json(marker,status)
        raise
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'results/human54_unified_20261018/prepared_r7/manifest.json')
    parser.add_argument('--formal',type=Path,default=ROOT/'results/human54_unified_20261018/formal')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--human-references',type=Path,required=True)
    a=parser.parse_args(); print(json.dumps(run(a.manifest,a.formal,a.output,a.human_references),indent=2))


if __name__=='__main__': main()
