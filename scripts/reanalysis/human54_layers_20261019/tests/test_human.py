"""Deterministic checks of the newly frozen human transformations."""
import unittest
import numpy as np
import pandas as pd
from scripts.reanalysis.human54_layers_20261019 import human as H


class HumanReferenceTests(unittest.TestCase):
    def test_source_experiments_use_their_own_author_answer_codes(self):
        # Literal first four author formulas, with anonymous synthetic question labels.
        header=['Q41','Q42','Q43','Q44','CQ41','CQ42','CQ43','CQ44']
        labels=['umbrella','classroom','home','pub',None,None,None,None]
        questions={q:{'question':q} for q in labels[:4]}
        exp1=[header,labels,[None]*4+['=IF(Q3=1,1,0)','=IF(R3=3,1,0)','=IF(S3=4,1,0)','=IF(T3=2,1,0)']]
        exp2=[header,labels,[None]*4+['=IF(M3=3,1,0)','=IF(N3=1,1,0)','=IF(O3=2,1,0)','=IF(P3=4,1,0)']]
        old=H.parse_source_answer_keys(exp1,questions)
        corrected=H.parse_source_answer_keys(exp2,questions)
        self.assertEqual(list(old.values()),[1,3,4,2])
        self.assertEqual(list(corrected.values()),[3,1,2,4])
        # These are four correct Exp2 answers; a transferred Exp1 key marks all wrong.
        answers=[3,1,2,4]
        self.assertEqual(sum(a==corrected[q] for a,q in zip(answers,questions)),4)
        self.assertEqual(sum(a==old[q] for a,q in zip(answers,questions)),0)

    def test_sro_digit_span_actual_trajectory_weights(self):
        # One correctly recalled heavily sampled backward length-6 cell contributes 12/64.
        x=pd.DataFrame([[int(c=='backward|6') for c in H.DS_WEIGHTS]],columns=H.DS_WEIGHTS)
        self.assertEqual(sum(H.DS_WEIGHTS.values()),64)
        self.assertAlmostEqual(H.fixed_weight(x,H.DS_WEIGHTS).iloc[0],12/64)
        x.loc[0,'forward|3']=np.nan
        self.assertTrue(np.isnan(H.fixed_weight(x,H.DS_WEIGHTS).iloc[0]))

    def test_neutral_and_go_weights(self):
        x=pd.DataFrame([{'congruent':1.,'incongruent':0.,'neutral':1.}])
        self.assertAlmostEqual(H.fixed_weight(x,{'congruent':1,'incongruent':1,'neutral':1}).iloc[0],2/3)
        self.assertEqual(H.fixed_weight(x,{'congruent':1,'incongruent':1}).iloc[0],.5)
        self.assertEqual(H.fixed_weight(pd.DataFrame([{'go':1.,'nogo':0.}]),{'go':945,'nogo':105}).iloc[0],.9)

    def test_profile_pooled_denominators_and_missing_items(self):
        r=H.Reference()
        r.profile('n_back','load_kind',['2|target','3|target'],[[1,0],[0,2]],[[2,0],[0,3]],strata=['2','3'])
        np.testing.assert_allclose(r.profiles[0]['values'],[.5,2/3])
        self.assertEqual(r.profiles[0]['unit_strata'],['2','3'])

    def test_profile_person_mean_is_not_missing_zero(self):
        r=H.Reference()
        r.mean_profile('digit_span','direction_length',pd.DataFrame({'f':[1.,np.nan],'b':[0.,1.]}))
        np.testing.assert_allclose(r.profiles[0]['values'],[1.,.5])

    def test_type2_auc_ties_orientation_and_undefined(self):
        self.assertEqual(H.auc([1,0],[9,1]),1.)
        self.assertEqual(H.auc([1,0],[1,9]),0.)
        self.assertEqual(H.auc([1,0],[5,5]),.5)
        self.assertTrue(np.isnan(H.auc([1,1],[9,1])))

    def test_dprime_minimum_classes(self):
        self.assertTrue(np.isnan(H.SRO.dprime(4,4,0,5)))
        self.assertTrue(np.isnan(H.SRO.dprime(5,5,0,4)))
        self.assertTrue(np.isfinite(H.SRO.dprime(5,5,0,5)))

    def test_cvlt_learning_slope_units(self):
        learning=np.array([2,4,6,8,10])
        self.assertEqual(float(learning @ np.arange(-2,3)/10),2.)

    def test_complete_case_structure_preserves_task_order(self):
        r=H.Reference()
        r.matrix('sro',pd.DataFrame({'digit_span':[1.,np.nan],'n_back':[.5,.7]}),cohort='sro',rule='valid_only')
        self.assertEqual(r.matrices[0]['n'],1)
        self.assertEqual(r.matrices[0]['n_source'],2)
        self.assertEqual(r.matrices[0]['columns'],['digit_span','n_back'])

if __name__=='__main__':unittest.main()
