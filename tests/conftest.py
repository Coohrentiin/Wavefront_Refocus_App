import numpy as np
import pytest
import tifffile

from wavefront_refocus.vendor.utils_images import save_stack


@pytest.fixture
def synthetic_wavefront():
    """A small square phase/amp pair with a smooth (unwrappable) phase."""
    N = 48
    yy, xx = np.mgrid[0:N, 0:N]
    phase = (0.01 * ((xx - N / 2) ** 2 + (yy - N / 2) ** 2)).astype(np.float32)
    amp = np.ones((N, N), dtype=np.float32)
    return phase, amp


@pytest.fixture
def stack_file(tmp_path, synthetic_wavefront):
    """A 3-frame wavefront stack TIFF (T, H, W, 2) = (phase, intensity)."""
    phase, amp = synthetic_wavefront
    T = 3
    frames = np.stack(
        [np.stack([phase * (i + 1), amp], axis=-1) for i in range(T)], axis=0
    )
    path = tmp_path / "wf.tif"
    save_stack(str(path), frames, wavelengths=(660.0,))
    return str(path), T


@pytest.fixture
def intensity_phase_dirs(tmp_path, synthetic_wavefront):
    """Two folders of per-frame intensity and phase tifs."""
    phase, amp = synthetic_wavefront
    idir = tmp_path / "Intensity"
    pdir = tmp_path / "Phase"
    idir.mkdir()
    pdir.mkdir()
    for i in range(3):
        tifffile.imwrite(str(idir / f"frame_{i}.tif"), amp)
        tifffile.imwrite(str(pdir / f"frame_{i}.tif"), phase * (i + 1))
    return str(idir), str(pdir)
