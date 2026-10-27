"""Token-AUROC strata for models with both option tokens on at least 40 of 80 items (paper Table 10).

This reproduces results/reanalysis/human54_confwager_20261022/cal_strata_min40.csv, first computed
inline on 2026-10-22; the 40-item minimum was set after the specification (SPEC.md change log).
CPU only, reads saved outputs of analyze.py.
"""
import argparse
from pathlib import Path

import pandas as pd

from scripts.experiments.human54_confwager_20261021.analyze import stratum

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/reanalysis/human54_confwager_20261022'
HUMAN = 0.7140042319515715
MIN_ITEMS = 40


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=OUT / 'cal_strata_min40.csv')
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    p = pd.read_csv(OUT / 'cal_token_by_model.csv', index_col='model')
    q = p[p.n_both_options >= MIN_ITEMS]
    rows = []
    for label, keep in (('all', q.index == q.index), ('gate', q.cal_gate.fillna(False).astype(bool)), ('large', q.params_b >= 27)):
        s = q[keep]
        rows.append(dict(measure='token_auroc_min40', **stratum(s.token_auroc, HUMAN, s.family, label)))
        rows.append(dict(measure='verbal_auroc_rerun_same_models', **stratum(s.verbal_auroc_rerun, HUMAN, s.family, label)))
    pd.DataFrame(rows).to_csv(a.output, index=False)


if __name__ == '__main__':
    main()
