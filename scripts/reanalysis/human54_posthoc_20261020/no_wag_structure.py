"""12-task structure sensitivity without WAG (2026-10-27 review: constant wagerers enter as AUROC .5).

Reuses the frozen PCA (fullbattery pca.analyze_pca, same seed) and parallel analysis
(factors parallel.parallel_analysis) on the saved primary 54-model invalid-as-wrong matrix
(the paper's primary scoring since ea44ce9) with the WAG column dropped. The full 13-column
matrix is run first and must reproduce PC1 39.23%.
CPU only; no model calls.
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '1')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.reanalysis.human54_fullbattery_20261019.pca import analyze_pca
from scripts.reanalysis.human54_fullbattery_20261019.matrix import TASK_ABBREVIATIONS
from scripts.reanalysis.human54_factors_20261019.parallel import parallel_analysis

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'results/reanalysis/human54_fullbattery_20261019/invalid_wrong'
OUT = ROOT / 'results/reanalysis/human54_posthoc_20261027_nowag'
PCA_SEED = int(np.random.SeedSequence(20261018).spawn(6)[2].generate_state(1)[0])


def run(frame, fams, label):
    codes = [TASK_ABBREVIATIONS[t] for t in frame.columns]
    pca, _ = analyze_pca(frame.to_numpy(), fams, codes, seed=PCA_SEED, draws=5000)
    par = parallel_analysis(frame.to_numpy(), draws=5000, seed=20261019)
    ci = pca['bootstrap']['intervals']['pc1_share']
    s = par[0] if isinstance(par, tuple) else par
    return dict(label=label, tasks=len(codes), models=len(frame), pc1=pca['variance_shares'][0],
                pc1_lower=ci['lower'], pc1_upper=ci['upper'], pc2=pca['variance_shares'][1],
                parallel=s if isinstance(s, dict) else str(s)[:2000])


m = pd.read_csv(SRC / 'matrix.csv', index_col=0)
meta = pd.read_csv(SRC / 'metadata.csv', index_col=0).reindex(m.index)
fams = meta.family_merged.tolist()
full = run(m, fams, 'all13')
assert abs(full['pc1'] - 0.3923) < 5e-4, full['pc1']
nowag = run(m.drop(columns=['post_decision_wagering']), fams, 'no_wag12')
(OUT / 'summary.json').write_text(json.dumps(dict(full=full, nowag=nowag), indent=1, default=float) + '\n')
print(json.dumps({k: {kk: v[kk] for kk in ('tasks', 'models', 'pc1', 'pc1_lower', 'pc1_upper', 'pc2')} for k, v in (('full', full), ('nowag', nowag))}, default=float))
