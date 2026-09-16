import numpy as np
import pytest

from wavefront_refocus.core.engine import build_compute_kwargs, refocus_one
from wavefront_refocus.core.model import OpticalParams, PropagationResult, SweepParams
from wavefront_refocus.vendor.utils_propag import _opd_from_field, compute_opd_stack


OPTICS = OpticalParams(pixel_pitch=5.86e-6, magnification=20.0,
                       wavelength=660e-9, NA=0.8)


def test_refocus_one_matches_stack_slice(synthetic_wavefront):
    phase, amp = synthetic_wavefront
    optics = OpticalParams(pixel_pitch=5.86e-6, magnification=20.0,
                           wavelength=660e-9, NA=0.8)
    sweep = SweepParams(half_range_sample=4e-6, n_planes=9)

    kwargs = build_compute_kwargs(phase, amp, optics, sweep, use_torch=False)
    result = compute_opd_stack(**kwargs)

    z = result["z0_index"] + 2  # pick a non-centre plane
    dz_sample = float(result["dz_sample_array"][z])

    field = refocus_one(phase, amp, dz_sample, optics)
    opd_one = _opd_from_field(field, optics.wavelength)

    # Should match the precomputed stack slice to FFT precision.
    assert np.allclose(opd_one, result["opd_stack"][z], atol=1e-3)


def _result(phase, amp, sweep, center):
    kwargs = build_compute_kwargs(phase, amp, OPTICS, sweep, use_torch=False,
                                  center_dz_sample=center)
    return PropagationResult.from_dict(
        compute_opd_stack(**kwargs), center, OPTICS.magnification
    )


def test_recentred_sweep_reports_absolute_dz(synthetic_wavefront):
    """A sweep centred on +4 µm must span 2…6 µm, not -2…+2 µm."""
    phase, amp = synthetic_wavefront
    sweep = SweepParams(half_range_sample=2e-6, n_planes=9)
    res = _result(phase, amp, sweep, 4e-6)

    assert res.center_dz_sample == pytest.approx(4e-6)
    assert res.dz_sample_array[0] == pytest.approx(2e-6)
    assert res.dz_sample_array[-1] == pytest.approx(6e-6)
    assert res.dz_sample_array[res.z0_index] == pytest.approx(4e-6)
    # image-space array stays the exact counterpart
    assert np.allclose(res.dz_image_array,
                       res.dz_sample_array * OPTICS.magnification ** 2)


def test_recentred_plane_matches_direct_propagation(synthetic_wavefront):
    """Plane dz of a recentred sweep == a single propagation to that dz."""
    phase, amp = synthetic_wavefront
    sweep = SweepParams(half_range_sample=2e-6, n_planes=9)
    res = _result(phase, amp, sweep, 4e-6)

    z = res.z0_index + 2
    dz = float(res.dz_sample_array[z])
    direct = _opd_from_field(refocus_one(phase, amp, dz, OPTICS), OPTICS.wavelength)
    direct = direct - direct.mean()
    slice_ = res.opd_stack[z] - res.opd_stack[z].mean()
    # Propagating in two hops (to the centre, then to the plane) vs one hop
    # agrees closely; band-limiting is applied per hop so allow a small margin.
    corr = np.corrcoef(direct.ravel(), slice_.ravel())[0, 1]
    assert corr > 0.99


def test_zero_center_is_unchanged(synthetic_wavefront):
    """center=0 must reproduce the original behaviour exactly."""
    phase, amp = synthetic_wavefront
    sweep = SweepParams(half_range_sample=2e-6, n_planes=9)
    base = compute_opd_stack(
        **build_compute_kwargs(phase, amp, OPTICS, sweep, use_torch=False)
    )
    res = _result(phase, amp, sweep, 0.0)
    assert np.allclose(res.dz_sample_array, base["dz_sample_array"])
    assert np.array_equal(res.opd_stack, base["opd_stack"])


def test_zero_plane_is_near_input(synthetic_wavefront):
    phase, amp = synthetic_wavefront
    optics = OpticalParams(pixel_pitch=5.86e-6, magnification=20.0,
                           wavelength=660e-9, NA=0.8)
    field = refocus_one(phase, amp, 0.0, optics)
    # dz=0 is a band-limited identity; correlation with input phase is high.
    a = _opd_from_field(field, optics.wavelength).ravel()
    b = _opd_from_field(amp * np.exp(1j * phase), optics.wavelength).ravel()
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.99
