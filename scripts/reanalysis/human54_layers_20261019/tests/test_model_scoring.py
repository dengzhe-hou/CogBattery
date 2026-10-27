"""Deterministic fixtures only. Prepared stimuli are allowed; no model records read."""
import copy
import json
import unittest
from pathlib import Path

from scripts.reanalysis.human54_layers_20261019.model_scoring import (
    Builder, analyze_cell, dprime, prepare_records, classify, kappa)
from scripts.reanalysis.human54_layers_20261019.model_run import coverage,verify_summary,structure_rows

ROOT=Path(__file__).resolve().parents[4]
MANIFEST=json.loads((ROOT/'results/human54_unified_20261018/prepared_r7/manifest.json').read_text())
BUNDLES={entry['task']:json.loads(Path(entry['material_file']).read_text()) for entry in MANIFEST['tasks']}


def fixtures(task):
    bundle=BUNDLES[task]; rows=[]
    for ep in bundle['episodes']:
        gold_by_id={s['id']:s['gold'] for s in ep['steps']}
        for step in ep['steps']:
            kind=step['kind']; gold=step['gold']
            if kind=='digit_recall': response=' '.join(map(str,gold))
            elif kind=='ospan_recall': response=' '.join(gold)
            elif kind=='cvlt_recall': response=json.dumps(gold)
            elif kind=='cvlt_recall_followup': response='[]'
            elif kind=='cvlt_recognition': response='YES' if gold else 'NO'
            elif kind=='drm_recognition': response='OLD' if gold else 'NEW'
            elif kind=='drm_report': response='REMEMBER' if gold_by_id[step['meta']['recognition_id']] else 'UNFAMILIAR'
            elif kind=='drm_rating': response='4'
            elif kind=='confidence': response='80%'
            elif kind=='wager': response=str(1+step['meta']['answer_step_id'].endswith('q01__answer'))
            else: response=str(gold)
            rows.append(dict(model='fixture:1',task=task,episode_id=ep['id'],condition=ep['condition'],step=step,
                             response=response,technical_valid=True,protocol_valid=True,protocol_errors=[],finish_reason='stop'))
    return rows


def invalid(row,reason='length_finish',response=''):
    row.update(technical_valid=False,protocol_valid=False,protocol_errors=[reason],response=response)


def endpoints(out,rule='valid_only'):
    return {r['endpoint']:r for r in out['endpoints'] if r['rule']==rule}


class DeterministicTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs={task:analyze_cell('fixture:1',task,b,fixtures(task)) for task,b in BUNDLES.items()}

    def test_all_thirteen_perfect_fixtures_native_endpoints(self):
        expected={
            'digit_span':{'forward_length_3':1,'backward_length_9':1,'position_accuracy':1,'forward_minus_backward':0,'short_minus_long':0,'structure_sro':1},
            'n_back':{'load_2_target':1,'load_3_target':1,'load_2_lure':0,'load_2_control':0,'load_3_nontarget':0,'structure_sro':1},
            'operation_span':{'position_accuracy':1,'arithmetic_accuracy':1,'version_1_position_accuracy':1,'length_7_position_accuracy':1},
            'stroop':{'congruent_accuracy':1,'incongruent_accuracy':1,'error_interference':0,'word_response_on_incongruent':0},
            'flanker':{'congruent_accuracy':1,'incongruent_accuracy':1,'neutral_accuracy':1,'error_interference':0,'structure_sro_no_neutral':1},
            'go_nogo':{'hit_rate':1,'false_alarm_rate':0,'omission_rate':0,'balanced_accuracy':1,'structure_sro':1},
            'cvlt_word_list':{'learning_1':16,'learning_5':16,'learning_total':80,'learning_slope':0,'interference_recall':16,'short_free':16,'long_free':16,'short_cued':16,'long_cued':16,'recognition_hit_rate':1,'recognition_false_alarm_rate':0,'free_intrusions':0,'cued_intrusions':0,'repetitions':0},
            'drm_false_memory':{'no_warning|hit_rate':1,'no_warning|lure_false_alarm_rate':0,'warning|new_false_alarm_rate':0,'warning|delta_dprime':0,'warning|native':0,'warning|accuracy_analyzed56':1,'warning|balanced_accuracy':1},
            'source_monitoring':{'indirect_accuracy':1,'direct_agent_first_accuracy':1,'direct_agent_last_accuracy':1,'structure_ball':1,'direct_minus_indirect':0,'direct_agent_first_minus_indirect':0},
            'false_belief':{'session|FB_accuracy':1,'session|TB_accuracy':1,'session|FB_minus_TB':0,'isolated|main_accuracy':1,'session|all_correct':1},
            'epitome_tom':{'mental|1_accuracy':1,'control|7_accuracy':1,'level_3_mental_minus_control':0,'mental_level_slope':0},
            'confidence_calibration':{'accuracy':1,'confidence_mean_percent':80,'official_bias':-.1,'position_answer_consistency':1},
            'post_decision_wagering':{'accuracy':1,'high_minus_low_accuracy':0,'training_reproduction_accuracy':1},
        }
        for task,numbers in expected.items():
            for rule in ('valid_only','invalid_wrong'):
                e=endpoints(self.outputs[task],rule)
                for name,value in numbers.items():
                    with self.subTest(task=task,rule=rule,endpoint=name):
                        self.assertAlmostEqual(e[name]['value'],value)
                        self.assertEqual(e[name]['coverage'],1)
                        self.assertTrue(e[name]['included'])
        self.assertIsNone(endpoints(self.outputs['confidence_calibration'])['type2_auc_fixed_direction']['value'])

    def test_every_emitted_endpoint_and_profile_has_a_defined_denominator(self):
        for task,out in self.outputs.items():
            for r in out['endpoints']+out['profiles']:
                with self.subTest(task=task,name=r.get('endpoint',r.get('cell'))):
                    self.assertGreater(r['n_planned'],0)
                    self.assertGreaterEqual(r['n_scored'],0)
                    self.assertLessEqual(r['n_scored'],r['n_planned'])
                    self.assertLessEqual(r['n_valid'],r['n_planned'])
        self.assertEqual(len([r for r in self.outputs['n_back']['profiles'] if r['rule']=='valid_only']),6)
        self.assertEqual(len([r for r in self.outputs['flanker']['profiles'] if r['rule']=='valid_only']),12)
        self.assertEqual(len([r for r in self.outputs['drm_false_memory']['profiles'] if r['rule']=='valid_only']),6)
        self.assertEqual(len([r for r in self.outputs['epitome_tom']['profiles'] if r['rule']=='valid_only']),14)

    def test_signal_invalid_wrong_opposes_r2_on_noise_and_missing_excluded(self):
        rows=fixtures('n_back')
        target=next(r for r in rows if r['step']['meta']['load']==2 and r['step']['meta']['condition']=='target')
        noise=next(r for r in rows if r['step']['meta']['load']==2 and r['step']['meta']['condition']=='control')
        invalid(target); invalid(noise)
        out=analyze_cell('fixture:1','n_back',BUNDLES['n_back'],rows)
        e,s,r2=(endpoints(out,r) for r in ('valid_only','invalid_wrong','nb_r2'))
        self.assertEqual(e['load_2_target']['value'],1)
        self.assertLess(s['load_2_target']['value'],1)
        self.assertGreater(s['load_2_control']['value'],0)
        self.assertEqual(r2['load_2_control']['value'],0)
        # Remove a real planned slot: never count D as wrong.
        rows.remove(noise); out=analyze_cell('fixture:1','n_back',BUNDLES['n_back'],rows)
        self.assertEqual(endpoints(out,'invalid_wrong')['load_2_control']['value'],0)

    def test_dprime_minimum_each_and_known_counts(self):
        self.assertIsNone(dprime(4,4,0,10)); self.assertIsNone(dprime(10,10,0,4))
        self.assertAlmostEqual(dprime(5,5,0,5),2.765988254201277)
        self.assertAlmostEqual(dprime(3,6,3,6),0)
        rows=fixtures('go_nogo')
        no=[r for r in rows if r['step']['meta'].get('stage')=='test' and r['step']['meta']['condition']=='nogo']
        for r in no[4:]: invalid(r)
        out=analyze_cell('fixture:1','go_nogo',BUNDLES['go_nogo'],rows)
        self.assertIsNone(endpoints(out)['dprime']['value'])
        self.assertIsNotNone(endpoints(out,'invalid_wrong')['dprime']['value'])

    def test_primary_coverage_threshold_boundary(self):
        rows=[dict(category='A',reason='',step=dict(kind='cal_answer'),score=dict(is_correct=True)) for _ in range(4)]
        rows.append(dict(category='B',reason='length_finish',step=dict(kind='cal_answer'),score={}))
        b=Builder('fixture:1','confidence_calibration','valid_only'); b.rate('a',rows)
        self.assertTrue(b.endpoints[0]['included']); self.assertEqual(b.endpoints[0]['coverage'],.8)
        rows.append(rows[-1]); b=Builder('fixture:1','confidence_calibration','valid_only'); b.rate('a',rows)
        self.assertFalse(b.endpoints[0]['included'])
        b=Builder('fixture:1','confidence_calibration','invalid_wrong'); b.rate('a',rows)
        self.assertTrue(b.endpoints[0]['included']); self.assertEqual(b.endpoints[0]['value'],4/6)

    def test_cvlt_two_steps_mean_times_fullscale_and_repeats(self):
        rows=fixtures('cvlt_word_list')
        for r in rows:
            if r['step']['id']=='A1_recall': invalid(r,'invalid_word_array','[" "]')
            if r['step']['id']=='A2_recall_followup': r['response']='["truck", "alien"]'
        out=analyze_cell('fixture:1','cvlt_word_list',BUNDLES['cvlt_word_list'],rows)
        e,s=endpoints(out),endpoints(out,'invalid_wrong')
        self.assertEqual(e['learning_total']['value'],80); self.assertEqual(e['learning_total']['coverage'],.8)
        self.assertIsNone(e['learning_1']['value']); self.assertEqual(s['learning_1']['value'],0)
        self.assertEqual(s['learning_total']['value'],64)
        self.assertEqual(e['free_intrusions']['value'],1); self.assertEqual(e['repetitions']['value'],1)
        for r in rows:
            if r['step']['id']=='short_cued_1_followup': invalid(r)
        out=analyze_cell('fixture:1','cvlt_word_list',BUNDLES['cvlt_word_list'],rows)
        self.assertEqual(endpoints(out)['short_cued']['value'],16)
        self.assertFalse(endpoints(out)['short_cued']['included'])
        self.assertEqual(endpoints(out,'invalid_wrong')['short_cued']['value'],16)

    def test_cal_pair_rules_auc_direction_and_missing_confidence(self):
        rows=fixtures('confidence_calibration')
        answers=[r for r in rows if r['step']['kind']=='cal_answer']
        answers[0]['response']='l' if answers[0]['step']['gold']=='s' else 's'
        confidences=[r for r in rows if r['step']['kind']=='confidence']; confidences[0]['response']='20%'
        invalid(answers[1]); invalid(confidences[2])
        out=analyze_cell('fixture:1','confidence_calibration',BUNDLES['confidence_calibration'],rows)
        e,s=endpoints(out),endpoints(out,'invalid_wrong')
        self.assertEqual(e['accuracy']['n_scored'],78); self.assertEqual(s['accuracy']['n_scored'],79)
        self.assertEqual(e['type2_auc_fixed_direction']['value'],1)
        self.assertAlmostEqual(s['type2_auc_fixed_direction']['value'],.75)

    def test_wager_requires_both_bets_and_invalid_bet_excluded(self):
        rows=fixtures('post_decision_wagering')
        for r in rows:
            if r['step']['kind']=='wager': r['response']='1'
        a=next(r for r in rows if r['step']['kind']=='wag_answer'); invalid(a)
        w=next(r for r in rows if r['step']['kind']=='wager'); invalid(w)
        out=analyze_cell('fixture:1','post_decision_wagering',BUNDLES['post_decision_wagering'],rows)
        self.assertIsNone(endpoints(out)['high_minus_low_accuracy']['value'])
        self.assertEqual(endpoints(out,'invalid_wrong')['accuracy']['n_scored'],239)

    def test_fb_no_location_main_wrong_sensitivity_excluded_and_catch_error(self):
        rows=fixtures('false_belief')
        first=next(r for r in rows if r['condition']=='session' and r['step']['meta']['main_analysis']); first['response']='I do not know.'
        ice=next(r for r in rows if r['condition']=='session' and r['step']['meta']['story']==7); ice['response']='Nothing happened.'
        out=analyze_cell('fixture:1','false_belief',BUNDLES['false_belief'],rows)
        self.assertEqual(endpoints(out)['session|main_accuracy']['value'],13/14)
        self.assertEqual(endpoints(out,'no_location_excluded')['session|main_accuracy']['value'],1)
        d={x['metric']:x['count'] for x in out['diagnostics'] if x['rule']=='valid_only'}
        self.assertEqual(d['session|no_location'],1); self.assertEqual(d['session|story7_other_answer'],1)
        self.assertEqual(d['session|other_location'],0)

    def test_a_b_c_d_categories_and_context(self):
        rows=fixtures('drm_false_memory')
        rec=next(r for r in rows if r['step']['kind']=='drm_recognition'); invalid(rec)
        report=next(r for r in rows if r['step']['kind']=='drm_report'); report['response']='UNCLASSIFIABLE'
        rows.pop(); out=analyze_cell('fixture:1','drm_false_memory',BUNDLES['drm_false_memory'],rows)
        counts=Counter(r['category'] for r in out['rows'])
        self.assertEqual(counts['B'],1); self.assertEqual(counts['C'],1); self.assertEqual(counts['D'],1)
        rows=fixtures('go_nogo'); rows[0]['prompt_plus_completion_exceeds_context']=True
        out=analyze_cell('fixture:1','go_nogo',BUNDLES['go_nogo'],rows)
        self.assertEqual(out['rows'][0]['reason'],'context_exceeded')
        self.assertEqual(out['rows'][0]['category'],'C')

    def test_summary_balance_and_coverage_failure_is_not_silent(self):
        records=fixtures('n_back'); n=len(records)
        s=dict(planned_steps=n,committed_or_reusable=n,not_run_total=0,not_run={},technical_valid=n,technical_invalid=0,valid_coverage=1)
        verify_summary(s,records,n)
        s['committed_or_reusable']-=1
        with self.assertRaises(ValueError): verify_summary(s,records,n)
        with self.assertRaises(ValueError): prepare_records(BUNDLES['n_back'],records+[records[0]])

    def test_drm_author_delta_requires_all_three_primary_classes(self):
        rows=fixtures('drm_false_memory')
        old=[r for r in rows if r['step']['kind']=='drm_recognition' and not r['step']['meta']['warned'] and r['step']['meta']['probe_class']=='old']
        for r in old[4:]: invalid(r)
        out=analyze_cell('fixture:1','drm_false_memory',BUNDLES['drm_false_memory'],rows)
        e=endpoints(out)['no_warning|delta_dprime']
        self.assertIsNone(e['value']);self.assertEqual(e['n_planned'],56)
        self.assertEqual(e['n_valid'],20)
        self.assertEqual(endpoints(out,'invalid_wrong')['no_warning|delta_dprime']['value'],0)

    def test_each_protocol_reason_native_boundaries_and_legal_skip_count(self):
        from scripts.reanalysis.human54_layers_20261019.model_run import B_REASONS
        records=fixtures('source_monitoring')
        for i,reason in enumerate(B_REASONS):
            invalid(records[i],reason,'Tammy')
        records[0]['budget_verdict']={'boundary_origin':'forced'}
        records[1]['budget_verdict']={'boundary_origin':'natural'}
        records[8]['budget_verdict']={'boundary_origin':'none'}
        records[8]['wire_request']={'format':{'enum':['Tammy','Jake']}}
        output=analyze_cell('fixture:1','source_monitoring',BUNDLES['source_monitoring'],records)
        n=len(records); nv=n-len(B_REASONS)
        summary=dict(planned_steps=n,committed_or_reusable=n,not_run_total=0,not_run={},technical_valid=nv,technical_invalid=n-nv,valid_coverage=nv/n)
        report=coverage('fixture:1','source_monitoring',output['rows'],records,summary,3)
        self.assertEqual(report['B'],len(B_REASONS))
        for reason in B_REASONS: self.assertEqual(report['B_'+reason],1)
        self.assertEqual(report['natural_boundary'],1); self.assertEqual(report['forced_boundary'],1)
        self.assertEqual(report['no_boundary'],1); self.assertEqual(report['skipped_history_legal_option'],1)
        self.assertEqual(report['superseded'],3)

    def test_dprime_primary_coverage_fixed_across_sensitivities(self):
        records=fixtures('n_back')
        ep=records[0]['episode_id']
        for r in records:
            if r['episode_id']==ep and r['step']['meta']['condition']=='target': invalid(r)
        output=analyze_cell('fixture:1','n_back',BUNDLES['n_back'],records)
        vals=[endpoints(output,rule)['load_2_dprime_episode_mean']['n_valid'] for rule in ('valid_only','invalid_wrong','nb_r2')]
        self.assertEqual(len(set(vals)),1)
        self.assertLess(vals[0],880)
        self.assertTrue(all(not r['episode_id'].startswith('nb_compact_') for r in output['blocks']))

    def test_structure_all_three_ball_representations_and_sro_variants(self):
        rows=[r for out in self.outputs.values() for r in out['endpoints']]
        scores=structure_rows(rows)
        self.assertEqual(len(scores),2*(10+27))
        self.assertTrue(all(r['value']==1 for r in scores if r['task']!='drm_false_memory' or r['representation']!='lure_contrast'))
        self.assertEqual(kappa([(0,0),(1,1)]),1)
        self.assertIsNone(kappa([(1,1),(1,1)]))


from collections import Counter
if __name__=='__main__': unittest.main()
