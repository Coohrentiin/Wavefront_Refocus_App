import pytest

from wavefront_refocus.core.optics import (
    base_dz_image_for_frame,
    image_to_sample,
    nearest_wavelength_key,
    sample_to_image,
)


def test_sample_image_roundtrip():
    M = 20.0
    dz = 3.2e-6
    assert sample_to_image(dz, M) == pytest.approx(dz * 400)
    assert image_to_sample(sample_to_image(dz, M), M) == pytest.approx(dz)


def test_nearest_wavelength_key():
    wl = {"650.000": 1.0, "660.000": 2.0, "670.000": 3.0}
    assert nearest_wavelength_key(wl, 659) == "660.000"
    assert nearest_wavelength_key(wl, 651) == "650.000"
    assert nearest_wavelength_key(wl, 700) == "670.000"


def test_base_dz_image_for_frame():
    entry = {"650.000": -0.015, "660.000": -0.016}
    assert base_dz_image_for_frame(entry, 661) == -0.016


def test_empty_map_raises():
    with pytest.raises(ValueError):
        nearest_wavelength_key({}, 660)
