"""Token versus verbal type-2 AUROC on the same items (fourth review round, 2026-10-24). CPU only.

analyze.py computes token AUROC on the items where both option tokens are in the top 20 and verbal AUROC on
all valid items. Here both are computed per model on the token-usable items only, using the same answers
(cal_logprobs; identical to cal_original answers in all 4,080 rows) and the cal_original verbal confidence,
as analyze.py does. Family intervals use cluster_mean (5,000 whole-family draws). Human .714 stays the
full 40-item value; no human item restriction is applied.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.human54_confwager_20261021.analyze import auroc, stratum
from scripts.reanalysis.human54_layers_20261019.analysis_statistics import cluster_mean

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/reanalysis/human54_confwager_20261022'
HUMAN = 0.7140042319515715


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=OUT / 'same_items.csv')
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    items = pd.read_csv(OUT / 'token_confidence_items.csv')
    per = pd.read_csv(OUT / 'cal_token_by_model.csv', index_col='model')
    rows = []
    for model, g in items.groupby('model', sort=False):
        u = g.dropna(subset=['token_conf', 'verbal'])
        y = u.correct.astype(bool)
        rows.append(dict(model=model, n_items=len(u), n_wrong=int((~y).sum()),
                         token_same=auroc(u.token_conf, y), verbal_same=auroc(u.verbal, y)))
    m = pd.DataFrame(rows).set_index('model').join(per[['n_both_options', 'family', 'params_b', 'cal_gate', 'verbal_auroc_rerun']])
    m.to_csv(a.output.with_name('same_items_by_model.csv'))
    out = []
    for panel, q in (('min40', m[m.n_both_options >= 40]), ('all_computable', m.dropna(subset=['token_same', 'verbal_same']))):
        q = q.dropna(subset=['token_same', 'verbal_same'])
        for label, keep in (('all', q.index == q.index), ('gate', q.cal_gate.fillna(False).astype(bool)), ('large', q.params_b >= 27)):
            s = q[keep]
            if not len(s):
                continue
            diff, _ = cluster_mean(s.token_same - s.verbal_same, s.family)
            out.append(dict(panel=panel, stratum=label, n_models=len(s),
                            token_same=s.token_same.mean(), verbal_same=s.verbal_same.mean(),
                            verbal_all_items=s.verbal_auroc_rerun.mean(),
                            token_minus_verbal=diff['mean'], tmv_lower=diff['lower'], tmv_upper=diff['upper'],
                            token_gt_verbal=int((s.token_same > s.verbal_same).sum()),
                            **{f'verbal_vs_human_{k}': v for k, v in stratum(s.verbal_same, HUMAN, s.family, label).items()
                               if k in ('difference', 'lower', 'upper')},
                            **{f'token_vs_human_{k}': v for k, v in stratum(s.token_same, HUMAN, s.family, label).items()
                               if k in ('difference', 'lower', 'upper')}))
    pd.DataFrame(out).to_csv(a.output, index=False)


if __name__ == '__main__':
    main()
