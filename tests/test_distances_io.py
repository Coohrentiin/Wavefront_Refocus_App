import json

from wavefront_refocus.core.model import Frame, OpticalParams
from wavefront_refocus.io import distances as dist_io


def _optics():
    return OpticalParams(pixel_pitch=5.86e-6, magnification=20.0, wavelength=660e-9, NA=0.8)


def test_apply_and_write_roundtrip(tmp_path):
    raw = {"0": {"660.000": -0.0155}, "1": {"660.000": -0.0150}}
    frames = [Frame(index=0, source=None), Frame(index=1, source=None)]
    optics = _optics()
    dist_io.apply_base_distances(frames, raw, optics.wavelength_nm)
    assert frames[0].base_dz_image == -0.0155

    # Choose a plane on frame 0: +2 um sample -> +2e-6 * 400 image.
    frames[0].chosen_dz_sample = 2e-6
    frames[0].is_chosen = True

    out = tmp_path / "out.json"
    dist_io.write_distances(str(out), frames, optics)
    data = json.load(open(out))
    assert data["0"]["660.000"] == -0.0155 + 2e-6 * 400
    # frame 1 had no position -> base distance preserved
    assert data["1"]["660.000"] == -0.0150


def test_missing_frame_defaults_zero(recwarn):
    raw = {"0": {"660.000": -0.01}}
    val = dist_io.base_dz_for_frame(raw, 5, 660.0)
    assert val == 0.0
    assert len(recwarn) >= 1


def test_nearest_wavelength_rebind():
    raw = {"0": {"650.000": -0.01, "660.000": -0.02}}
    frames = [Frame(index=0, source=None)]
    dist_io.apply_base_distances(frames, raw, 659.0)
    assert frames[0].base_dz_image == -0.02
    dist_io.apply_base_distances(frames, raw, 651.0)
    assert frames[0].base_dz_image == -0.01
