"""Descriptive statistics fixed in HUMAN54_ANALYSIS_SPEC.md; CPU, no inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261018
PANEL_DRAWS = 5000
PROFILE_DRAWS = 2000
TIE = 1e-12


def divide(numerator, denominator):
    n, d = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d != 0)


def interval(draws):
    x = np.asarray(draws, float)
    x = x[np.isfinite(x)]
    return {'lower': float(np.quantile(x, .025)) if len(x) else None,
            'upper': float(np.quantile(x, .975)) if len(x) else None,
            'n_valid': len(x)}


def cluster_mean(values, families, draws=PANEL_DRAWS, seed=SEED):
    """Model-weighted point; resample whole families, retaining all members."""
    x = np.asarray(values, float)
    labels = np.asarray(families)
    if len(x) != len(labels) or not np.isfinite(x).all() or not len(x):
        raise ValueError('Cluster means need aligned, finite nonempty observations')
    codes, names = pd.factorize(labels, sort=True)
    sums = np.bincount(codes, weights=x)
    counts = np.bincount(codes)
    weights = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1/len(names)), size=draws)
    boot = (weights @ sums) / (weights @ counts)
    return {'mean': float(x.mean()), 'families': len(names), 'models': len(x),
            'family_equal_mean': float(np.mean(sums/counts)), **interval(boot)}, boot


def direction_counts(values, families):
    x = np.asarray(values, float)
    f = np.asarray(families)
    good = np.isfinite(x)
    x, f = x[good], f[good]
    grouped = pd.DataFrame({'v': x, 'f': f}).groupby('f', sort=True).v.mean().to_numpy()
    def count(y):
        return {'positive': int((y > TIE).sum()), 'negative': int((y < -TIE).sum()),
                'tie': int((np.abs(y) <= TIE).sum()), 'total': len(y)}
    return {'models': count(x), 'families': count(grouped)}


def profile_metrics(human, model):
    """Keep every prespecified cell; r needs >=6 cells and nonconstant vectors."""
    h, m = np.broadcast_arrays(np.atleast_2d(np.asarray(human, float)),
                               np.atleast_2d(np.asarray(model, float)))
    complete = np.isfinite(h).all(axis=1) & np.isfinite(m).all(axis=1)
    mae = np.where(complete, np.mean(np.abs(m-h), axis=1), np.nan)
    hc, mc = h-h.mean(axis=1, keepdims=True), m-m.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(hc**2, axis=1)*np.sum(mc**2, axis=1))
    r = divide(np.sum(hc*mc, axis=1), denominator)
    defined = complete & (h.shape[1] >= 6) & (np.ptp(h, axis=1) > 0) & (np.ptp(m, axis=1) > 0)
    return mae, np.where(defined, np.clip(r, -1, 1), np.nan)


def stratified_weights(strata, draws=PROFILE_DRAWS, seed=SEED):
    """One multinomial multiplicity per whole participant/block within stratum."""
    strata = np.asarray(strata)
    weights = np.zeros((draws, len(strata)), dtype=int)
    rng = np.random.default_rng(seed)
    for group in sorted(set(strata)):
        ix = np.flatnonzero(strata == group)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=draws)
    return weights


def profile_bootstrap(human_numerator, human_denominator, strata, model_values,
                      model_numerator=None, model_denominator=None,
                      draws=PROFILE_DRAWS, seed=SEED):
    """Human whole-person bootstrap; optional paired model blocks for n-back."""
    hn, hd = np.asarray(human_numerator, float), np.asarray(human_denominator, float)
    if hn.shape != hd.shape or hn.ndim != 2 or len(hn) != len(strata):
        raise ValueError('Profile numerator/denominator/strata must align')
    if not np.isfinite(hn).all() or not np.isfinite(hd).all() or (hd < 0).any():
        raise ValueError('Missing cells use zero denominator, not NaN counts')
    w = stratified_weights(strata, draws, seed)
    h = divide(hn.sum(axis=0), hd.sum(axis=0))
    hb = divide(w @ hn, w @ hd)
    if model_numerator is None:
        m = np.asarray(model_values, float)
        mb = np.broadcast_to(m, hb.shape)
    else:
        mn, md = np.asarray(model_numerator, float), np.asarray(model_denominator, float)
        if mn.shape != hn.shape or md.shape != hd.shape:
            raise ValueError('Paired n-back model/human block counts must align')
        m = divide(mn.sum(axis=0), md.sum(axis=0))
        mb = divide(w @ mn, w @ md)
    mae, r = profile_metrics(h, m)
    maeb, rb = profile_metrics(hb, mb)
    def number(x):
        return float(x) if np.isfinite(x) else None
    return {'n_cells': len(h), 'mae': number(mae[0]), 'r': number(r[0]),
            'mae_ci': interval(maeb), 'r_ci': interval(rb),
            'human_values': h.tolist(), 'model_values': m.tolist(),
            'draws': draws, 'seed': seed,
            'bootstrap_unit': 'paired human/model block' if model_numerator is not None else 'human participant',
            'r_status': 'defined' if np.isfinite(r[0]) else
                        ('fewer_than_six_cells' if len(h) < 6 else
                         'incomplete_profile' if not np.isfinite(h).all() or not np.isfinite(m).all() else 'constant_profile')}
