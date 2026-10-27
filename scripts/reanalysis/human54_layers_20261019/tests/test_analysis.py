"""Deterministic checks of comparison conventions, using no formal responses."""
import json
import numpy as np
import pandas as pd
import pytest

from scripts.reanalysis.human54_layers_20261019.analyze import (
    refs_for_rule, endpoint_comparisons, structural_population)


def test_layer1_keeps_source_endpoint_and_nb_r2_reference():
    refs = [dict(task=t, endpoint='a', rule=r, value=v)
            for t in ('stroop', 'n_back')
            for r, v in [('valid_only', 1), ('invalid_wrong', 2), ('nb_r2', 3)]]
    assert [(r['task'],r['value']) for r in refs_for_rule(refs,'invalid_wrong')] == [('stroop',2),('n_back',1)]
    assert [(r['task'],r['value']) for r in refs_for_rule(refs,'nb_r2')] == [('stroop',3),('n_back',3)]


def test_endpoint_gate_does_not_erase_low_coverage_models(tmp_path):
    rows = pd.DataFrame([dict(task='confidence_calibration',rule='valid_only',endpoint='accuracy',
                             model=m,value=v,included=keep,coverage=c)
                         for m,v,keep,c in [('a',.2,True,1),('b',.9,False,.7)]])
    refs = [dict(task='confidence_calibration',endpoint='accuracy',rule='both',value=.4,n=2,values=[.2,.6])]
    endpoint_comparisons(rows,refs,{'merged':{'a':'f','b':'g'},'raw':{'a':'f','b':'g'}},
                         {'fixture':{'a','b'}},tmp_path)
    individual = pd.read_csv(tmp_path/'model_human_endpoints.csv')
    panel = pd.read_csv(tmp_path/'panel_differences.csv')
    assert len(individual)==2
    assert panel.models.eq(1).all()
    assert panel['mean'].eq(-.2).all()
    assert individual.loc[individual.model.eq('a'),'human_percentile_mid_distribution'].iloc[0]==25


def test_structure_known_correlations_and_degenerate_output(tmp_path):
    tasks=['operation_span','drm_false_memory','source_monitoring']
    # WM opposes two identical EM scores: within=1, between=-1, delta=2.
    x=np.arange(8,dtype=float)
    f=pd.DataFrame({tasks[0]:x,tasks[1]:-x,tasks[2]:-x})
    result,boot=structural_population(f,tasks,'human',False,tmp_path/'defined',20261018)
    assert result['metrics']['delta']['value']==pytest.approx(2)
    assert result['metrics']['pc1']['value']==pytest.approx(1)
    assert result['partition_tail']==pytest.approx(1/3)
    assert boot.shape==(5000,5)
    f[tasks[0]]=1
    result,boot=structural_population(f,tasks,'human',False,tmp_path/'constant',20261018)
    assert result['status']=='not_estimable' and boot is None
    assert (tmp_path/'constant/human_scores.csv').is_file()
