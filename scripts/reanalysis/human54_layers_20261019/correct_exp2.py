"""Refresh only Exp2 human comparisons after the documented answer-code fix.

Uses saved model endpoint/profile tables; never opens formal response records.
The complete first pass remains in firstpass_before_exp2_correction/.
"""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time

import pandas as pd

from .analyze import (OUT, ROOT, endpoint_comparisons, panel_names,
                      profile_comparisons, save_json)


def replace_rows(old, new, keys):
    key = lambda row: tuple(str(row[k]) for k in keys)
    replacements = {key(row): row for row in new}
    expected = {key(row) for row in old if row['human_cohort'] == 'exp2'}
    assert expected == set(replacements) and len(replacements) == len(new)
    result = [replacements[key(row)] if row['human_cohort'] == 'exp2' else row
              for row in old]
    assert [r for r in result if r['human_cohort'] != 'exp2'] == [
        r for r in old if r['human_cohort'] != 'exp2']
    return result


def replace_csv(path, new_path, keys):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, old = reader.fieldnames, list(reader)
    with new_path.open(newline='') as stream:
        new = list(csv.DictReader(stream))
    # The full table has optional columns for other human cohorts/endpoints.
    new = [{column: row.get(column, '') for column in columns} for row in new]
    result = replace_rows(old, new, keys)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(result)
    return len(new)


def main():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert os.environ.get(name) == '1', name
    start = time.monotonic()
    archive = OUT/'firstpass_before_exp2_correction'
    assert (archive/'ARCHIVE_NOTE.json').exists()
    human = json.loads((OUT/'human_side/references.json').read_text())
    initial = json.loads((archive/'human_side/references.json').read_text())
    # This correction is confined to the supplemental Exp2 human data.
    for table in ('endpoints', 'profiles'):
        assert [r for r in human[table] if r.get('cohort') != 'exp2'] == [
            r for r in initial[table] if r.get('cohort') != 'exp2']
    assert human['matrices'] == initial['matrices']
    refs = {name: [r for r in human[name] if r.get('cohort') == 'exp2']
            for name in ('endpoints', 'profiles')}
    means = {r['endpoint']: r['value'] for r in refs['endpoints']}
    assert abs(means['indirect_accuracy']-1772/3096) < 1e-12
    assert abs(means['direct_agent_first_accuracy']-1499/2976) < 1e-12
    endpoints = pd.read_csv(OUT/'layer1/endpoints.csv')
    profiles = pd.read_csv(OUT/'layer1/profiles.csv', dtype={'cell': str})
    families = json.loads((ROOT/'results/reanalysis/aplus_20260718/family_map.json').read_text())
    panels = panel_names(sorted(endpoints.model.unique()))
    temp = OUT/'exp2_corrected_comparisons'
    temp.mkdir(exist_ok=True)
    endpoint_comparisons(endpoints[endpoints.task.eq('source_monitoring')],
                         refs['endpoints'], families, panels, temp)
    profile_comparisons(profiles[profiles.task.eq('source_monitoring')],
                        refs['profiles'], pd.DataFrame(), temp)
    common = ['task', 'rule', 'human_cohort']
    keys = {
        'model_human_endpoints': common+['model', 'endpoint'],
        'panel_differences': common+['endpoint', 'panel', 'family_mapping'],
        'profile_comparisons': common+['profile', 'model'],
        'model_human_direction_counts': common+['endpoint', 'panel', 'family_mapping'],
    }
    counts = {}
    for name in ('model_human_endpoints', 'panel_differences', 'profile_comparisons'):
        counts[name] = replace_csv(OUT/'layer1'/f'{name}.csv', temp/f'{name}.csv', keys[name])
    for name in ('profile_comparisons', 'model_human_direction_counts'):
        path = OUT/'layer1'/f'{name}.json'
        old, new = json.loads(path.read_text()), json.loads((temp/path.name).read_text())
        save_json(path, replace_rows(old, new, keys[name]))
    manifest = dict(corrected_at=datetime.now().astimezone().isoformat(),
                    reason='Objective human Exp2 answer-code error; author Exp2 formulas replace Exp1 codes',
                    post_hoc=True, analysis_rules_changed=False, formal_records_read=0,
                    formal_model_scoring_runs_total=1, model_calls=0, cpu_threads=1,
                    affected_rows=counts, unaffected_rows_equal=True,
                    primary_Exp1_and_all_structures_unchanged=True,
                    human_means=means, archive=str(archive),
                    wall_seconds=time.monotonic()-start)
    save_json(OUT/'exp2_correction_manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
