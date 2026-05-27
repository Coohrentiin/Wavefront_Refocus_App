"""Spline interpolation of chosen planes across frame indices.

Fits a smoothing spline to the (frame index, chosen sample-space displacement)
pairs of the user-chosen frames and evaluates it at every other frame index.
Beyond the chosen range we use explicit linear extrapolation from the two
nearest chosen points rather than trusting spline extrapolation.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
from scipy.interpolate import UnivariateSpline


def _linear_extrap(x: np.ndarray, y: np.ndarray, q: float) -> float:
    """Linearly extrapolate at ``q`` using the two nearest endpoint samples."""
    if q <= x[0]:
        if len(x) == 1:
            return float(y[0])
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return float(y[0] + slope * (q - x[0]))
    # q >= x[-1]
    if len(x) == 1:
        return float(y[-1])
    slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return float(y[-1] + slope * (q - x[-1]))


def build_evaluator(
    chosen_idx: Sequence[float],
    chosen_value: Sequence[float],
    smoothing: float = 0.0,
) -> Callable[[float], float]:
    """Return ``f(x) -> value`` interpolating the chosen points.

    Uses a cubic smoothing spline for ≥4 points, linear interpolation for
    2–3, and a constant for 1. Beyond the chosen range it switches to explicit
    linear extrapolation from the two nearest endpoints.
    """
    if len(chosen_idx) == 0:
        raise ValueError("need at least one chosen frame to interpolate")

    x = np.asarray(chosen_idx, dtype=float)
    y = np.asarray(chosen_value, dtype=float)

    # Sort and de-duplicate x (UnivariateSpline requires strictly increasing x).
    order = np.argsort(x)
    x, y = x[order], y[order]
    uniq = np.unique(x)
    if len(uniq) != len(x):
        y = np.array([y[x == ux].mean() for ux in uniq])
        x = uniq

    lo, hi = x[0], x[-1]

    if len(x) >= 4:
        spl = UnivariateSpline(x, y, s=float(smoothing), k=3, ext=0)
        inner = lambda q: float(spl(q))
    elif len(x) >= 2:
        inner = lambda q: float(np.interp(q, x, y))
    else:
        inner = lambda q: float(y[0])

    def evaluate(q: float) -> float:
        q = float(q)
        if q < lo or q > hi:
            return _linear_extrap(x, y, q)
        return inner(q)

    return evaluate


def fit_and_eval(
    chosen_idx: Sequence[int],
    chosen_dz_sample: Sequence[float],
    all_idx: Sequence[int],
    smoothing: float = 0.0,
) -> Dict[int, float]:
    """Interpolate/extrapolate the sample-space displacement for every index.

    Returns a dict mapping (int) frame index -> value.
    """
    evaluate = build_evaluator(chosen_idx, chosen_dz_sample, smoothing)
    return {int(i): evaluate(float(i)) for i in all_idx}
