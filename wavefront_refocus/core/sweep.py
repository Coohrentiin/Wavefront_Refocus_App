"""Reconcile UI (half-range, N planes) with the engine's sweep parameters.

``compute_opd_stack`` builds its sweep as::

    n_half = floor(dz_sample_max / dz_step_sample)
    dz_sample_array = arange(-n_half, n_half + 1) * dz_step_sample   # Z = 2*n_half+1

so to obtain a sweep of exactly ``n_planes`` planes we pick the step that makes
``floor(half_range / step) == n_half`` with ``n_half = (n - 1) // 2``.

``n_planes`` is rounded **up** to the next odd value so the centre (dz == 0)
plane is always present.
"""
from __future__ import annotations


def resolve_sweep(half_range_sample: float, n_planes: int) -> dict:
    """Return ``{dz_sample_max, dz_step_sample}`` for ``compute_opd_stack``.

    Guarantees the resulting stack has exactly ``odd(n_planes)`` planes,
    symmetric about 0.
    """
    if half_range_sample <= 0:
        raise ValueError("half_range_sample must be > 0")

    n = max(3, int(n_planes))
    if n % 2 == 0:
        n += 1  # force odd so dz == 0 is a plane
    n_half = (n - 1) // 2

    step = half_range_sample / n_half
    # Nudge the max up by a hair so floating-point floor() lands on n_half
    # rather than n_half - 1 (which would drop the outermost pair of planes).
    dz_sample_max = half_range_sample + step * 1e-9

    return {"dz_sample_max": dz_sample_max, "dz_step_sample": step}


def resolved_n_planes(n_planes: int) -> int:
    """The odd plane count actually used for a requested ``n_planes``."""
    n = max(3, int(n_planes))
    if n % 2 == 0:
        n += 1
    return n
