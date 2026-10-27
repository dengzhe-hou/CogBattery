"""Whole-family bootstrap uncertainty for fixed-dimensional rotated EFA.

This module reads no experiment data. Every requested dimensionality is fitted
to the same resampled families. Factor matching is a signed permutation only:
continuous Procrustes rotation would change the oblimin solution being tested.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.reanalysis.human54_fullbattery_20261019.pca import family_bootstrap_counts
from scripts.reanalysis.human54_factors_20261019.core import (
    ConstantColumnError, correlation_matrix, fit_minres,
)


def align_signed_permutation(pattern: np.ndarray, phi: np.ndarray,
                             reference: np.ndarray) -> dict:
    """Match factors by absolute Tucker congruence, then fix their signs.

    The returned permutation indexes original candidate columns in reference
    order. Phi is transformed with that same signed permutation, preserving
    the candidate's reproduced common covariance exactly.
    """
    pattern = np.asarray(pattern, dtype=float)
    reference = np.asarray(reference, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if pattern.shape != reference.shape or pattern.ndim != 2:
        raise ValueError("Candidate and reference patterns must have the same 2-D shape")
    k = pattern.shape[1]
    if phi.shape != (k, k):
        raise ValueError("Phi must align with candidate factor columns")
    if not all(np.isfinite(value).all() for value in (pattern, reference, phi)):
        raise ValueError("Factor alignment requires finite patterns and Phi")
    reference_norm = np.linalg.norm(reference, axis=0)
    candidate_norm = np.linalg.norm(pattern, axis=0)
    if np.any(reference_norm == 0) or np.any(candidate_norm == 0):
        raise ValueError("A zero loading column has undefined Tucker congruence")
    congruence = reference.T @ pattern / np.outer(reference_norm, candidate_norm)
    rows, columns = linear_sum_assignment(-np.abs(congruence))
    permutation = np.empty(k, dtype=int)
    permutation[rows] = columns
    matched = congruence[np.arange(k), permutation]
    signs = np.where(matched < 0, -1., 1.)
    aligned_pattern = pattern[:, permutation] * signs
    aligned_phi = phi[np.ix_(permutation, permutation)] * np.outer(signs, signs)
    return {
        "pattern": aligned_pattern,
        "phi": aligned_phi,
        "structure": aligned_pattern @ aligned_phi,
        "congruence": matched * signs,
        "permutation": permutation,
        "signs": signs,
    }


def _intervals(values: np.ndarray, mask: np.ndarray) -> dict:
    """Percentile intervals conditional on the explicitly supplied fit mask."""
    selected = values[mask]
    if not len(selected):
        return {"n": 0, "lower": None, "median": None, "upper": None}
    quantiles = np.quantile(selected, [.025, .5, .975], axis=0)
    return {
        "n": len(selected),
        "lower": quantiles[0].tolist(),
        "median": quantiles[1].tolist(),
        "upper": quantiles[2].tolist(),
    }


def _reference_status(fit: dict) -> dict:
    return {
        "converged": bool(fit["converged"]),
        "proper": bool(fit["proper"]),
        "boundary": bool(fit["boundary"]),
        "rmsr": float(fit["rmsr"]),
        "diagnostics": _json_value(fit["diagnostics"]),
    }


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def bootstrap_factors(X: np.ndarray, families: Sequence[str],
                      references: dict[int, dict], draws: int = 2000,
                      seed: int = 20261019,
                      progress: Callable[[dict], None] | None = None):
    """Return JSON-ready summaries and raw arrays suitable for ``np.savez``.

    A draw samples the original number of families with replacement and keeps
    every model of each sampled family, with the sampled multiplicity. Columns
    are standardized again through the weighted correlation calculation. The
    intervals condition on each fixed k; no parallel-analysis selection occurs
    within this bootstrap and no dimension-selection frequency is estimated.

    Finite converged fits are summarized even when improper or on a boundary.
    A second explicitly restricted summary uses proper, non-boundary fits.
    Failed draws retain their rows, reason and denominator membership.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("Factor bootstrap requires at least two rows and two columns")
    if len(families) != len(X) or not np.isfinite(X).all():
        raise ValueError("Families must align with a complete finite matrix")
    if not references or any(not isinstance(k, (int, np.integer)) or k < 1
                             for k in references):
        raise ValueError("Provide at least one reference indexed by positive integer k")
    ks = sorted(references)
    for k in ks:
        if np.asarray(references[k]["pattern"]).shape != (X.shape[1], k):
            raise ValueError(f"Reference pattern shape does not match k={k}")
        # Establish that the actual reference has matchable, nonzero columns;
        # being improper/boundary does not itself prevent bootstrapping it.
        align_signed_permutation(references[k]["pattern"], references[k]["phi"],
                                 references[k]["pattern"])
    names, family_codes, counts = family_bootstrap_counts(families, draws, seed)
    arrays = {
        "family_counts": counts,
        "family_codes": family_codes,
        "sample_sizes": counts[:, family_codes].sum(axis=1),
    }
    p = X.shape[1]
    for k in ks:
        for key, shape in (
            ("pattern", (p, k)), ("structure", (p, k)), ("phi", (k, k)),
            ("congruence", (k,)), ("rmsr", ()),
            ("communalities", (p,)), ("uniqueness", (p,)),
        ):
            arrays[f"k{k}_{key}"] = np.full((draws, *shape), np.nan)
        for key in ("finite", "converged", "proper", "boundary"):
            arrays[f"k{k}_{key}"] = np.zeros(draws, dtype=bool)
        for key in ("optimizer_success", "rotation_converged"):
            arrays[f"k{k}_{key}"] = np.zeros(draws, dtype=bool)
        for key in ("lower_boundary_variables", "upper_boundary_variables", "heywood_variables"):
            arrays[f"k{k}_{key}"] = np.zeros((draws, p), dtype=bool)
        arrays[f"k{k}_failure_reason"] = np.full(draws, "", dtype="U256")
        arrays[f"k{k}_permutation"] = np.full((draws, k), -1, dtype=int)
        arrays[f"k{k}_signs"] = np.full((draws, k), np.nan)

    for draw, family_counts in enumerate(counts):
        weights = family_counts[family_codes]
        if np.count_nonzero(weights) < 2:
            for k in ks:
                arrays[f"k{k}_failure_reason"][draw] = "fewer_than_two_distinct_rows"
            corr = None
        else:
            try:
                corr = correlation_matrix(X, weights=weights)
            except ConstantColumnError as exc:
                for k in ks:
                    arrays[f"k{k}_failure_reason"][draw] = f"constant_column: {exc}"
                corr = None
        if corr is not None:
            for k in ks:
                prefix = f"k{k}_"
                try:
                    fit = fit_minres(corr, k, rotate=True)
                except np.linalg.LinAlgError as exc:
                    arrays[prefix + "failure_reason"][draw] = f"linear_algebra: {exc}"
                    continue
                for key in ("converged", "proper", "boundary"):
                    arrays[prefix + key][draw] = bool(fit[key])
                for key in ("optimizer_success", "rotation_converged"):
                    arrays[prefix + key][draw] = bool(fit["diagnostics"][key])
                for key in ("lower_boundary_variables", "upper_boundary_variables", "heywood_variables"):
                    indices = np.asarray(fit["diagnostics"][key], dtype=int)
                    arrays[prefix + key][draw, indices] = True
                finite = all(np.isfinite(np.asarray(fit[key], dtype=float)).all()
                             for key in ("pattern", "phi", "structure", "rmsr",
                                         "communalities", "uniqueness"))
                if not finite:
                    arrays[prefix + "failure_reason"][draw] = "nonfinite_fit"
                    continue
                # Zero columns can occur in overfactored samples and must be
                # reported rather than assigned arbitrary matching signs.
                if np.any(np.linalg.norm(fit["pattern"], axis=0) == 0):
                    arrays[prefix + "failure_reason"][draw] = "zero_loading_column"
                    continue
                aligned = align_signed_permutation(fit["pattern"], fit["phi"],
                                                    references[k]["pattern"])
                arrays[prefix + "finite"][draw] = True
                for key in ("pattern", "structure", "phi", "congruence",
                            "permutation", "signs"):
                    arrays[prefix + key][draw] = aligned[key]
                for key in ("rmsr", "communalities", "uniqueness"):
                    arrays[prefix + key][draw] = fit[key]
                if not fit["converged"]:
                    arrays[prefix + "failure_reason"][draw] = "not_converged"
        if progress is not None and ((draw + 1) % 100 == 0 or draw + 1 == draws):
            progress({"phase": "factor_bootstrap", "draw": draw + 1,
                      "draws": draws, "dimensions": ks})

    result = {
        "models": len(X), "tasks": p, "families": names,
        "n_families": len(names), "draws": int(draws), "seed": int(seed),
        "unit": "whole family, all members retained with sampled multiplicity",
        "restandardize_each_draw": True,
        "alignment": "Hungarian maximum absolute Tucker congruence, then signs only",
        "interval": "2.5th and 97.5th percentiles conditional on fixed k and stated fit mask",
        "dimension_selection_repeated": False,
        "same_family_draws_for_all_dimensions": True,
        "dimensions": {},
    }
    statistics = ("pattern", "structure", "phi", "congruence", "rmsr",
                  "communalities", "uniqueness")
    for k in ks:
        prefix = f"k{k}_"
        finite = arrays[prefix + "finite"]
        converged = arrays[prefix + "converged"]
        proper = arrays[prefix + "proper"]
        boundary = arrays[prefix + "boundary"]
        masks = {"converged_finite": finite & converged,
                 "proper_nonboundary": finite & converged & proper & ~boundary}
        reasons = arrays[prefix + "failure_reason"]
        entry = {
            "reference": _reference_status(references[k]),
            "counts": {
                "attempted": int(draws), "finite": int(finite.sum()),
                "converged_finite": int(masks["converged_finite"].sum()),
                "failed": int((~masks["converged_finite"]).sum()),
                "finite_not_converged": int((finite & ~converged).sum()),
                "finite_improper": int((finite & ~proper).sum()),
                "finite_boundary": int((finite & boundary).sum()),
                "optimizer_not_converged": int(np.sum(finite & ~arrays[prefix + "optimizer_success"])),
                "rotation_not_converged": int(np.sum(finite & ~arrays[prefix + "rotation_converged"])),
                "proper_nonboundary": int(masks["proper_nonboundary"].sum()),
            },
            "diagnostic_counts_by_variable": {
                key: arrays[prefix + key].sum(axis=0).tolist()
                for key in ("lower_boundary_variables", "upper_boundary_variables", "heywood_variables")
            },
            "failure_reasons": dict(Counter(str(reason) for reason in reasons if reason)),
            "summaries": {},
        }
        for label, mask in masks.items():
            entry["summaries"][label] = {
                "n": int(mask.sum()),
                "fraction_of_all_draws": float(mask.mean()) if draws else None,
                "intervals": {key: _intervals(arrays[prefix + key], mask)
                              for key in statistics},
            }
        result["dimensions"][str(k)] = entry
    return result, arrays
