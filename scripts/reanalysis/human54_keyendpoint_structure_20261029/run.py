"""Structure sensitivity on the key endpoints of Table 4 instead of the structural scores (2026-10-29).

Exploratory, added after the formal results. WAG is dropped because its key endpoint (constant
wagering) is binary and its contrast is undefined for 37 models. Each remaining key endpoint is
oriented to correlate positively with the same task's structural score, so higher means better
task performance. Reports parallel analysis (5,000 column permutations, 95th percentile), PC1
share, PC1 correlation with log size, the within-minus-between correlation contrast, and the
paper's leave-one-family-out five-group prediction with the full partition reference for block
sizes 3, 3, 3, 2, 1 (CAL is the only metacognition task left). CPU only.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reanalysis.human54_scoring_revision_20261026 import data as D
from scripts.reanalysis.human54_posthoc_20261020 import lofo as L

ROOT = Path(__file__).resolve().parents[3]
SEEDS = {'prediction': 3939289603}  # paper prediction seed (human54_scoring_revision_20261026.run.SEEDS)


# Copied from human54_scoring_revision_20261026.run, which is not in this repository.
def prediction_summary(cells, *, draws, extra=()):
    pairs = {'five_group_vs_pc1': ('se_group', 'se_g')}
    for name in extra:
        pairs[f'{name}_vs_pc1'] = ('se_' + name, 'se_g')
        pairs[f'five_group_vs_{name}'] = ('se_group', 'se_' + name)
    out = dict(rmse_pc1=L.rmse(cells, 'se_g'), rmse_five_group=L.rmse(cells, 'se_group'),
               rmse_intercept=L.rmse(cells, 'se_intercept'), gain_five_group=L.gain(cells),
               n_cells=int(len(cells)), n_families=int(cells.family.nunique()))
    for name in extra:
        out['rmse_' + name] = L.rmse(cells, 'se_' + name)
        out[f'gain_{name}_vs_pc1'] = L.gain(cells, 'se_' + name, 'se_g')
        out[f'gain_five_group_vs_{name}'] = L.gain(cells, 'se_group', 'se_' + name)
    out['bootstrap'] = L.paired_bootstrap(cells, pairs, draws=draws, seed=SEEDS['prediction'])
    return out
LEADERBOARD = ROOT / 'docs/data/leaderboard.json'
SEED, PERMS = 20261029, 5000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    S, meta = D.load_matrix()
    fam = D.family_maps(meta)['merged23']
    code_of = {t: D.ABBR[t] for t in S.columns}
    lb = json.loads(LEADERBOARD.read_text())
    key = pd.DataFrame({m['model']: m['values'] for m in lb['models']}).T
    tasks = [t for t in S.columns if code_of[t] != 'WAG']
    K = pd.DataFrame({t: pd.to_numeric(key.loc[S.index, code_of[t]]) for t in tasks}, index=S.index)
    assert K.notna().all().all(), K.isna().sum()
    orient = {}
    for t in tasks:
        r = np.corrcoef(K[t], S[t])[0, 1]
        orient[code_of[t]] = float(r)
        if r < 0:
            K[t] = -K[t]
    Z = (K - K.mean()) / K.std(ddof=0)
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.to_numpy().T)))[::-1]
    rng = np.random.default_rng(SEED)
    perm = np.empty((PERMS, len(tasks)))
    Zn = Z.to_numpy()
    for i in range(PERMS):
        Zp = np.column_stack([rng.permutation(Zn[:, j]) for j in range(Zn.shape[1])])
        perm[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Zp.T)))[::-1]
    ref95 = np.quantile(perm, .95, axis=0)
    retained = int(np.argmax(ev <= ref95)) if np.any(ev <= ref95) else len(ev)
    w, V = np.linalg.eigh(np.corrcoef(Zn.T)); pc1 = Zn @ V[:, -1]
    params = pd.read_csv(ROOT / 'results/reanalysis/human54_posthoc_20261020_gates/model_parameters.csv', index_col='model').params_b
    r_size = float(abs(np.corrcoef(pc1, np.log(params.loc[S.index]))[0, 1]))
    R = np.corrcoef(Zn.T); grp = [L.FIVE_GROUPS[t] for t in tasks]
    wi = [R[i, j] for i in range(len(tasks)) for j in range(i + 1, len(tasks)) if grp[i] == grp[j]]
    bt = [R[i, j] for i in range(len(tasks)) for j in range(i + 1, len(tasks)) if grp[i] != grp[j]]
    partitions = L.enumerate_partitions(len(tasks), (3, 3, 3, 2, 1))
    # CAL has no same-group task once WAG is dropped, so its catalog includes the empty peer set.
    rows, _ = L.fold_predictions(K, fam, {t: L.peer_catalog(tasks, t) + [frozenset()] for t in tasks}, two_dim=False)
    theory = {t: L.theory_peers(tasks)[t][0] for t in tasks}
    summ = prediction_summary(L.cell_table(rows, theory, extra=()), draws=20000)
    gains, _, _ = L.partition_reference(rows, tasks, partitions)
    k = L.theory_partition_index(partitions, tasks)
    assert np.isclose(gains[k], summ['gain_five_group'], atol=1e-12)
    summ['partition_reference'] = dict(share_at_least_five_group=float(np.mean(gains >= gains[k])), n_partitions=int(len(gains)))
    out = dict(note=__doc__, orientation_r_with_structural_score=orient,
               eigenvalues=ev[:4].tolist(), permutation_95=ref95[:4].tolist(), retained_dimensions=retained,
               pc1_share=float(ev[0] / ev.sum()), pc1_abs_r_log_size=r_size,
               delta_within_minus_between=float(np.mean(wi) - np.mean(bt)),
               prediction=dict(gain=summ['gain_five_group'], ci95=summ['bootstrap']['five_group_vs_pc1']['ci95'],
                               rmse_pc1=summ['rmse_pc1'], rmse_group=summ['rmse_five_group'],
                               share_random_at_least=summ['partition_reference']['share_at_least_five_group'],
                               n_partitions=summ['partition_reference']['n_partitions']))
    (a.output / 'summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1))


if __name__ == '__main__':
    main()
