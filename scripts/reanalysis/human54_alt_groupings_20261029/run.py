"""Alternative task groupings for the leave-one-family-out prediction test (2026-10-29).

Exploratory, added after the formal results in answer to a reviewer request (compare the
predefined five groups with competing and data-driven groupings). Every grouping keeps the
predefined block sizes 3, 3, 3, 2, 2, so it is comparable with the 600,600-partition reference.
  response_format  span or recall output (DS OS CVLT), yes/no detection with false alarms
                   (NB GN DRM), labelled choice (ST FL SM), story questions about minds (FB EP),
                   confidence reports (CAL WAG). Fixed before computing.
  data_driven      chosen inside each fold: the partition with the highest mean within-block
                   Pearson r among training-family centroids, so the held-out family never
                   informs its own grouping.
Uses the paper's own fold predictions (human54_posthoc_20261020.lofo) and bootstrap. CPU only.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_scoring_revision_20261026 import data as D
from scripts.reanalysis.human54_scoring_revision_20261026.run import SEEDS
from scripts.reanalysis.human54_posthoc_20261020 import lofo as L

FORMAT = {'span_recall': ['DS', 'OS', 'CVLT'], 'detection': ['NB', 'GN', 'DRM'], 'labelled_choice': ['ST', 'FL', 'SM'],
          'mind_stories': ['FB', 'EP'], 'confidence': ['CAL', 'WAG']}
DRAWS = 20000


def summary(cells, gains):
    g = L.gain(cells)
    ci = L.paired_bootstrap(cells, {'g': ('se_group', 'se_g')}, draws=DRAWS, seed=SEEDS['prediction'])['g']['ci95']
    return dict(gain=g, ci95=ci, rmse_pc1=L.rmse(cells, 'se_g'), rmse_group=L.rmse(cells, 'se_group'),
                share_random_at_least=float(np.mean(gains >= g - 1e-12)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    X, meta = D.load_matrix()
    fam = D.family_maps(meta)['merged23']
    tasks = list(X.columns)
    code = {t: D.ABBR[t] for t in tasks}
    task_of = {c: t for t, c in code.items()}
    partitions = L.enumerate_partitions(13, (3, 3, 3, 2, 2))
    rows, _ = L.fold_predictions(X, fam, {t: L.peer_catalog(tasks, t) for t in tasks}, two_dim=False)
    gains, rmse_g, _ = L.partition_reference(rows, tasks, partitions)

    def peers_from(blocks):
        out = {}
        for b in blocks:
            for t in b:
                out[t] = frozenset(x for x in b if x != t)
        return out

    res = {}
    theory = {t: L.theory_peers(tasks)[t][0] for t in tasks}
    res['predefined'] = summary(L.cell_table(rows, theory, extra=()), gains)
    fmt_blocks = [[task_of[c] for c in cs] for cs in FORMAT.values()]
    assert sorted(t for b in fmt_blocks for t in b) == sorted(tasks)
    res['response_format'] = summary(L.cell_table(rows, peers_from(fmt_blocks), extra=()), gains)

    # Data-driven: per held-out family, the max mean within-block r partition on training centroids.
    s = pd.Series(fam)
    blocks_idx = [[list(b) for b in part] for part in partitions]
    pair_index = {(i, j): k for k, (i, j) in enumerate((i, j) for i in range(13) for j in range(i + 1, 13))}
    member = np.zeros((len(partitions), len(pair_index)), dtype=np.float32)
    for n, part in enumerate(blocks_idx):
        for b in part:
            for x in range(len(b)):
                for y in range(x + 1, len(b)):
                    i, j = sorted((b[x], b[y]))
                    member[n, pair_index[(i, j)]] = 1
    member /= member.sum(axis=1, keepdims=True)
    chosen, fold_peers = {}, {}
    for f in sorted(set(fam.values())):
        train = X.loc[s.reindex(X.index) != f]
        cent = train.groupby(s.reindex(train.index)).mean()
        r = np.corrcoef(cent.to_numpy(float).T)
        vec = np.array([r[i, j] for (i, j) in pair_index])
        best = int(np.argmax(member @ vec))
        chosen[f] = tuple(sorted(tuple(sorted(code[tasks[i]] for i in b)) for b in blocks_idx[best]))
        fold_peers[f] = peers_from([[tasks[i] for i in b] for b in blocks_idx[best]])
    cells = []
    for r in rows:
        peers = fold_peers[r['family']][r['target']]
        cells.append(dict(family=r['family'], target=r['target'], se_g=float(np.mean((r['truth'] - r['pred_g'])**2)),
                          se_group=float(np.mean((r['truth'] - r['peer'][peers])**2))))
    res['data_driven'] = summary(pd.DataFrame(cells), gains)
    common = Counter(chosen.values()).most_common(3)
    res['data_driven']['most_common_partitions'] = [dict(blocks=[list(b) for b in k], folds=v) for k, v in common]
    res['reference'] = dict(n_partitions=len(gains), median=float(np.median(gains)), max=float(gains.max()),
                            rmse_pc1=rmse_g)
    (a.output / 'summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
