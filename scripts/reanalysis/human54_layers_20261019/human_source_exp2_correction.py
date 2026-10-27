"""Correct the documented Exp2 numeric answer-code error in human outputs only.

The first pass is retained in firstpass_before_exp2_correction. This script reads
no formal/pilot/model records and never computes model endpoints or structures.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import pandas as pd
from .human import Reference, build_source, clean, OUT


def affected(row):
    return row.get('task') == 'source_monitoring' and row.get('cohort') == 'exp2'


def correct(out=OUT):
    start=time.monotonic()
    archive=out.parent/'firstpass_before_exp2_correction/human_side/references.json'
    if not archive.is_file():
        raise FileNotFoundError('Preserved firstpass required for this documented correction')
    old=json.loads((out/'references.json').read_text())
    original=json.loads(archive.read_text())
    ref=Reference()
    trials=build_source(ref,out,experiments=(2,),write_trials=False)
    endpoints={x['endpoint']:x for x in ref.endpoints}
    direct=endpoints['direct_agent_first_accuracy']['value']
    indirect=endpoints['indirect_accuracy']['value']
    ref.endpoint('source_monitoring','direct_agent_first_minus_indirect',value=direct-indirect,n=253,
                 cohort='exp2',contrast_definition='supplement: direct-agent-first minus indirect, between participants; model direct contrast averages its two direct conditions')
    for field,key in [('endpoints','endpoint'),('profiles','profile')]:
        replacements={row[key]:clean(row) for row in getattr(ref,field)}
        assert {row[key] for row in old[field] if affected(row)} == set(replacements)
        old[field]=[replacements[row[key]] if affected(row) else row for row in old[field]]
        assert [r for r in old[field] if not affected(r)] == [r for r in original[field] if not affected(r)]
    old['sources']=[clean(ref.sources[0]) if x.get('task')=='source_monitoring' and x.get('experiment')==2 else x for x in old['sources']]
    assert old['matrices']==original['matrices']
    correction=dict(corrected_at_utc=datetime.now(timezone.utc).isoformat(),after_formal_analysis=True,
        kind='objective_source_answer_code_correction_not_analysis_rule_change',
        source='data/raw/eerland2018_source_memory/Data_AnalysisExp2.xlsx',
        official_source='https://osf.io/download/bd6ey/',
        source_cells={'Q41':'Data file!CG3 =IF(M3=3,1,0)','Q42':'Data file!CH3 =IF(N3=1,1,0)',
                      'Q43':'Data file!CI3 =IF(O3=2,1,0)','Q44':'Data file!CJ3 =IF(P3=4,1,0)'},
        speaker_codes={'Jake':3,'Tammy':4,'Olivia':1,'Mike':2},
        affected='SM Exp2 only: 3 human endpoints, 1 96-cell profile, 6072 anonymous scored responses',
        unchanged='All non-Exp2 endpoints/profiles and every SRO/Ball matrix directly equal preserved firstpass; same 253 complete24 records and condition assignment',
        values_before={x['endpoint']:x['value'] for x in original['endpoints'] if affected(x)},
        values_after={x['endpoint']:x['value'] for x in old['endpoints'] if affected(x)},
        resource={'cpu_only':True,'model_calls':0,'formal_record_reads':0,'wall_seconds':time.monotonic()-start},
        firstpass_archive=str(archive.parent.parent))
    old['status']='human_references_complete_with_documented_posthoc_exp2_source_correction'
    old['corrections']=[correction]
    (out/'references.json').write_text(json.dumps(clean(old),indent=2,allow_nan=False)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if k!='values'} for r in old['endpoints']]).to_csv(out/'endpoints.csv',index=False)
    original_trials=pd.read_csv(out/'source_monitoring_anonymous_trials.csv')
    archived_trials=pd.read_csv(archive.parent/'source_monitoring_anonymous_trials.csv')
    unaffected=original_trials[original_trials.experiment==1]
    pd.testing.assert_frame_equal(unaffected.reset_index(drop=True),archived_trials[archived_trials.experiment==1].reset_index(drop=True))
    corrected_trials=pd.concat([unaffected,trials],ignore_index=True)
    # Responses, conditions, row identities and question identities never change.
    pd.testing.assert_frame_equal(corrected_trials.drop(columns='correct'),archived_trials.drop(columns='correct'))
    corrected_trials.to_csv(out/'source_monitoring_anonymous_trials.csv',index=False)
    manifest=json.loads((out/'manifest.json').read_text())
    manifest.update(status=old['status'],sources=old['sources'],corrections=[correction])
    (out/'manifest.json').write_text(json.dumps(clean(manifest),indent=2,allow_nan=False)+'\n')
    (out/'EXP2_CORRECTION.json').write_text(json.dumps(correction,indent=2,allow_nan=False)+'\n')
    print(json.dumps(correction,indent=2))


if __name__=='__main__':correct()
