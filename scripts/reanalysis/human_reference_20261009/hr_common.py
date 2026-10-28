"""Shared helpers for the human-reference reanalysis (W2, user authorisation 2026-10-09 item 2).

Descriptive statistics only: subject-level condition contrasts, sign tests, within-group
correlations and odd/even split-half reliability. No human-LLM test lives here.
"""
import json
import math
import os
import resource
import time

import numpy as np
import pandas as pd
from scipy import stats

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
RAW = os.environ.get('COGBATTERY_RAW', os.path.join(_REPO, 'data', 'raw'))
OUT = os.environ.get('COGBATTERY_HUMAN_REFERENCE_OUT', os.path.join(_REPO, 'results', 'reanalysis', 'human_reference_20261009'))
TIE = 1e-12


def _f(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def contrast(a, b, expected='a>b'):
    """a, b: pandas Series indexed by subject (expected a > b). Subjects with both values only."""
    df = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    d = df['a'] - df['b']
    pos = int((d > TIE).sum())
    neg = int((d < -TIE).sum())
    tie = int(len(d) - pos - neg)
    n = len(d)
    p_excl = stats.binomtest(pos, pos + neg, 0.5, alternative='greater').pvalue if pos + neg else None
    p_paper = stats.binomtest(pos, n, 0.5, alternative='greater').pvalue if n else None
    return dict(
        expected=expected, n_subjects=n,
        mean_a=_f(df['a'].mean()), sd_a=_f(df['a'].std(ddof=1)),
        mean_b=_f(df['b'].mean()), sd_b=_f(df['b'].std(ddof=1)),
        mean_diff=_f(d.mean()), sd_diff=_f(d.std(ddof=1)), median_diff=_f(d.median()),
        pos_neg_tie=f'{pos}/{neg}/{tie}',
        p_sign_one_sided_excl_ties=_f(p_excl),
        p_sign_one_sided_ties_as_not_in_direction=_f(p_paper),
    )


def corr(x, y):
    df = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
    n = len(df)
    if n < 4:
        return dict(n=n)
    pr = stats.pearsonr(df['x'], df['y'])
    # round before ranking so that equal rationals computed along different float paths stay tied
    sr = stats.spearmanr(df['x'].round(10), df['y'].round(10))
    z = np.arctanh(pr.statistic)
    se = 1 / math.sqrt(n - 3)
    return dict(n=n, pearson_r=_f(pr.statistic), pearson_p_two_sided=_f(pr.pvalue),
                pearson_95ci_fisher=[_f(np.tanh(z - 1.96 * se)), _f(np.tanh(z + 1.96 * se))],
                spearman_rho=_f(sr.statistic), spearman_p_two_sided=_f(sr.pvalue))


def split_half(odd, even):
    """Pearson r between odd- and even-trial scores across subjects, Spearman-Brown corrected."""
    df = pd.concat([odd.rename('o'), even.rename('e')], axis=1).dropna()
    n = len(df)
    if n < 4 or df['o'].std() == 0 or df['e'].std() == 0:
        return dict(n=n, r_odd_even=None, spearman_brown=None)
    r = stats.pearsonr(df['o'], df['e']).statistic
    return dict(n=n, r_odd_even=_f(r), spearman_brown=_f(2 * r / (1 + r)))


def odd_even_flags(df, group_cols, order_col):
    """Return a boolean Series: True for odd-numbered trials (1st, 3rd, ...) within group_cols, ordered by order_col."""
    rank = df.sort_values(order_col).groupby(group_cols, sort=False).cumcount()
    return (rank.reindex(df.index) % 2 == 0)


def loglinear_dprime(hits, n_signal, fas, n_noise):
    """Hautus (1995) log-linear correction, as in cogarena GoNoGo.aggregate (clamped .001-.999)."""
    h = min(.999, max(.001, (hits + .5) / (n_signal + 1)))
    f = min(.999, max(.001, (fas + .5) / (n_noise + 1)))
    return float(stats.norm.ppf(h) - stats.norm.ppf(f))


def attenuation_ceiling(rel_x, rel_y):
    if rel_x is None or rel_y is None or rel_x <= 0 or rel_y <= 0:
        return None
    return math.sqrt(rel_x * rel_y)


class Timer:
    """Wall clock, process CPU time and peak RSS (ru_maxrss, KiB on Linux) for one script run."""

    def __init__(self):
        self.w0 = time.time()
        self.c0 = time.process_time()

    def report(self):
        return dict(wall_seconds=round(time.time() - self.w0, 3),
                    cpu_seconds=round(time.process_time() - self.c0, 3),
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def clean(o):
    """Recursively turn NaN/inf and numpy scalars into JSON-safe values (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(obj, *parts):
    obj = clean(obj)
    path = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, allow_nan=False)
    return path


def env_check():
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ.get(k) == '1', f'{k} must be 1 on the login node'
