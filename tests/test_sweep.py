import numpy as np
import pytest

from wavefront_refocus.core.sweep import resolve_sweep, resolved_n_planes


def _z_from_sweep(sw):
    n_half = int(np.floor(sw["dz_sample_max"] / sw["dz_step_sample"]))
    return 2 * n_half + 1


@pytest.mark.parametrize("n", [3, 5, 41, 101])
def test_odd_n_planes_exact(n):
    sw = resolve_sweep(10e-6, n)
    assert _z_from_sweep(sw) == n


@pytest.mark.parametrize("n", [4, 40, 100])
def test_even_rounds_up(n):
    sw = resolve_sweep(10e-6, n)
    assert _z_from_sweep(sw) == n + 1
    assert resolved_n_planes(n) == n + 1


def test_symmetric_about_zero():
    sw = resolve_sweep(7e-6, 21)
    n_half = int(np.floor(sw["dz_sample_max"] / sw["dz_step_sample"]))
    arr = np.arange(-n_half, n_half + 1) * sw["dz_step_sample"]
    assert arr.size == 21
    assert np.allclose(arr, -arr[::-1])
    assert arr.max() == pytest.approx(7e-6)


def test_invalid_range():
    with pytest.raises(ValueError):
        resolve_sweep(0.0, 41)
