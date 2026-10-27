"""Generate endpoint-schema fixtures without reading any research results."""
from __future__ import annotations

import numpy as np

from .matrix import DIRECT, MEANS, METADATA_FIELDS, RULES, TASKS


def make_synthetic_rows():
    rng = np.random.default_rng(7031)
    family_g = rng.normal(size=23)
    family_group = rng.normal(size=(23, 5))
    groups = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    rows = []
    for i in range(54):
        model, family = f'synthetic_model_{i:02d}', f'synthetic_family_{i % 23:02d}'
        scores = .5 + .17 * np.tanh(family_g[i % 23] +
            .8 * family_group[i % 23, groups] + rng.normal(0, .2, 13))
        for rule in RULES:
            for j, task in enumerate(TASKS):
                value = scores[j]
                endpoints = DIRECT[task][1] if task in DIRECT else MEANS.get(task)
                values = {}
                if endpoints is not None:
                    if task in ['n_back', 'go_nogo']:
                        value *= 4
                    if task == 'cvlt_word_list':
                        value *= 80
                    for k, endpoint in enumerate(endpoints):
                        values[endpoint] = value + (.005 * k if task in MEANS else 0)
                else:
                    high_rate = .5 if 1 <= i <= 17 else float(i % 2)
                    effect = .15 * np.tanh(family_group[i % 23, 4] + rng.normal(0, .2))
                    low_accuracy = .55 - effect / 2 if high_rate < 1 else None
                    high_accuracy = .55 + effect / 2 if high_rate > 0 else None
                    accuracy = ((1 - high_rate) * (low_accuracy or 0) +
                                high_rate * (high_accuracy or 0))
                    values.update(accuracy=accuracy, high_wager_rate=high_rate,
                        wager_1_accuracy=low_accuracy, wager_2_accuracy=high_accuracy,
                        high_minus_low_accuracy=effect if 0 < high_rate < 1 else None)
                for endpoint, endpoint_value in values.items():
                    coverage = .4 if i == 0 and task == 'cvlt_word_list' else (
                        91/144 if i == 0 and task == 'source_monitoring' else 1.)
                    n = 240 if task == 'post_decision_wagering' else 100
                    rows.append(dict(model=model, task=task, rule=rule, endpoint=endpoint,
                        value=endpoint_value, coverage=coverage, n_planned=n,
                        n_valid=n * coverage, n_scored=n,
                        family_raw=family, family_merged=family, native_thinking=False,
                        panel54=True, panel19=False, panel51=True))
    return rows
