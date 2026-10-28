"""Leaderboard data for the CogBattery release site (2026-10-27). CPU only, reads saved results.

One key endpoint per paradigm (paper Table 3, gen_core_table ROWS) for each of the 54 models, with the
source human mean and the panel mean gap and its family interval (frozen cluster_mean, seed 20261018,
5,000 draws, human mean fixed). Primary scoring rule: invalid task answers count as wrong.
The panel intervals must reproduce paper Table 3, or the run stops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_layers_20261019.analysis_statistics import cluster_mean
from scripts.reanalysis.human54_posthoc_20261020.gen_core_table import ROWS

ROOT = Path(__file__).resolve().parents[2]
LAYER1 = ROOT / 'results/reanalysis/human54_layers_20261019/layer1'
GATES = ROOT / 'results/reanalysis/human54_posthoc_20261020_gates'
# Across-person human SD per key endpoint, as in paper Figure 3 (SM, EP, WAG approximations documented there).
CLOSENESS = ROOT / 'results/reanalysis/human54_model_closeness_20261029/thirteen/summary.json'
RULE = 'invalid_wrong'

TASKS = {
    'DS': ('Digit span', 'Working memory', 'Forward minus backward accuracy', 'SRO'),
    'NB': ('N-back', 'Working memory', '2-back ordinary non-target false alarms', 'SRO'),
    'OS': ('Operation span', 'Working memory', 'Letter-position accuracy', 'Ball et al.'),
    'ST': ('Stroop', 'Cognitive control', 'Error increase with conflict', 'SRO'),
    'FL': ('Flanker', 'Cognitive control', 'Error increase with conflict', 'SRO'),
    'GN': ('Go/no-go', 'Cognitive control', 'No-go false alarms', 'SRO'),
    'CVLT': ('CVLT-II list learning', 'Episodic memory', 'Recall slope (words per trial)', 'CNP'),
    'DRM': ('DRM false memory', 'Episodic memory', 'New minus lure false alarms', 'Ball et al.'),
    'SM': ('Source memory', 'Episodic memory', 'Direct minus indirect source accuracy', 'Eerland et al.'),
    'FB': ('False belief', 'Social reasoning', 'False-belief accuracy', 'Strachan et al.'),
    'EP': ('Recursive mindreading', 'Social reasoning', 'Mental-state accuracy slope', 'EPITOME'),
    'CAL': ('Confidence calibration', 'Metacognition', 'Type-2 AUROC', 'Mazancieux et al.'),
    'WAG': ('Post-decision wagering', 'Metacognition', 'Always places the same wager', 'Dienes and Seth'),
}
# Paper Table 3 (all-model gap and interval), used only as a reproduction check.
PAPER = {'DS': (.372, .280, .501), 'NB': (.257, .159, .341), 'OS': (-.182, -.296, -.080),
         'ST': (.320, .247, .448), 'FL': (.065, -.006, .137), 'GN': (.040, -.065, .191),
         'CVLT': (-1.836, -2.138, -1.476), 'DRM': (.293, .217, .367), 'SM': (.052, .038, .062),
         'FB': (-.210, -.297, -.122), 'EP': (.013, .008, .019), 'CAL': (-.141, -.196, -.105)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    paired = pd.read_csv(LAYER1 / 'model_human_endpoints.csv', low_memory=False)
    paired = paired[(paired.rule == RULE) & (paired.human_cohort == 'primary') & (paired.included == True)]
    e = pd.read_csv(LAYER1 / 'endpoints.csv', low_memory=False)
    e = e[(e.rule == RULE) & (e.included == True)]
    models = pd.read_csv(ROOT / 'results/reanalysis/human54_layers_20261019/coverage/models.csv', index_col='model')
    params = pd.read_csv(GATES / 'model_parameters.csv', index_col='model').params_b
    human_sd = {k: v['sd'] for k, v in json.loads(CLOSENESS.read_text())['human'].items()}
    tasks, values = [], {m: {} for m in models.index}
    for code, task, endpoint, _ in ROWS:
        name, group, measure, source = TASKS[code]
        row = dict(code=code, name=name, group=group, measure=measure, source=source)
        if code == 'WAG':
            v = e[(e.task == task) & (e.endpoint == endpoint)].set_index('model').value.astype(float)
            const = np.isin(v.to_numpy(), (0., 1.))
            row.update(kind='constant', human=None, constant_models=int(const.sum()), models=len(v))
            for m, c in zip(v.index, const):
                values[m][code] = bool(c)
        else:
            s = paired[(paired.task == task) & (paired.endpoint == endpoint)].set_index('model')
            human = float(s.human_mean.iloc[0])
            x = s.value.astype(float)
            stats, _ = cluster_mean(x.to_numpy() - human, [models.family_merged[m] for m in x.index])
            got = tuple(round(stats[k], 3) for k in ('mean', 'lower', 'upper'))
            if got != PAPER[code]:
                raise ValueError(f'{code}: {got} does not reproduce paper Table 3 {PAPER[code]}')
            row.update(kind='value', human=human, human_n=int(s.human_n.iloc[0]), human_sd=human_sd[code], model_mean=float(x.mean()),
                       gap=stats['mean'], lower=stats['lower'], upper=stats['upper'], models=len(x))
            for m, val in x.items():
                values[m][code] = float(val)
        tasks.append(row)
    out = dict(
        version='human54 formal_r7_unified_20261018', scoring='invalid task answers count as wrong',
        interval='95% percentile interval, 5,000 whole-family resamples, human mean fixed',
        tasks=tasks,
        models=[dict(model=m, family=models.family_merged[m], params_b=round(float(params[m]), 2),
                     native_thinking=bool(models.native_thinking[m]), serving='paper',
                     values=values[m]) for m in models.index])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + '\n')
    print(f'{len(out["models"])} models, {len(tasks)} tasks -> {a.output}')


if __name__ == '__main__':
    main()
