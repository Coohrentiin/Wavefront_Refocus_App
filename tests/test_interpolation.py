import pytest

import numpy as np

from wavefront_refocus.core.interpolation import build_evaluator, fit_and_eval


def test_linear_interp_two_points():
    out = fit_and_eval([0, 10], [0.0, 10.0], list(range(11)), 0.0)
    assert out[5] == pytest.approx(5.0)


def test_spline_passes_through_nodes():
    x = [0, 5, 10, 15]
    y = [1.0, 2.0, 1.5, 3.0]
    out = fit_and_eval(x, y, x, 0.0)  # s=0 interpolating
    for xi, yi in zip(x, y):
        assert out[xi] == pytest.approx(yi, abs=1e-6)


def test_linear_extrapolation_beyond_range():
    out = fit_and_eval([2, 4, 6, 8], [2.0, 4.0, 6.0, 8.0], [0, 10], 0.0)
    # data is y = x, slope 1 each side
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[10] == pytest.approx(10.0, abs=1e-6)


def test_single_point_constant():
    out = fit_and_eval([5], [3.0], [0, 5, 9], 0.0)
    assert out[0] == pytest.approx(3.0)
    assert out[9] == pytest.approx(3.0)


def test_duplicate_indices_averaged():
    out = fit_and_eval([5, 5, 8], [2.0, 4.0, 9.0], [5], 0.0)
    assert out[5] == pytest.approx(3.0)


def test_empty_raises():
    with pytest.raises(ValueError):
        fit_and_eval([], [], [0, 1], 0.0)


def test_build_evaluator_continuous_query():
    # Evaluator must accept arbitrary float queries (used for the dense curve)
    # without integer-key collisions.
    ev = build_evaluator([0, 10], [0.0, 10.0], 0.0)
    xs = np.linspace(0, 10, 21)
    ys = np.array([ev(x) for x in xs])
    assert np.allclose(ys, xs)  # y = x


def test_build_evaluator_extrapolates():
    ev = build_evaluator([2, 4, 6, 8], [2.0, 4.0, 6.0, 8.0], 0.0)
    assert ev(0.0) == pytest.approx(0.0, abs=1e-6)
    assert ev(12.0) == pytest.approx(12.0, abs=1e-6)
