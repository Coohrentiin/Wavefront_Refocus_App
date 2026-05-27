import numpy as np

from wavefront_refocus.core.engine import build_compute_kwargs, refocus_one
from wavefront_refocus.core.model import OpticalParams, SweepParams
from wavefront_refocus.vendor.utils_propag import _opd_from_field, compute_opd_stack


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
