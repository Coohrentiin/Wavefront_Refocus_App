import numpy as np
import pytest

# Skip cleanly if a Qt platform plugin is unavailable.
pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def test_display_range_updates_clim(qapp):
    from wavefront_refocus.gui.phase_view import PhaseView

    view = PhaseView()
    data = np.tile(np.linspace(0, 100, 64, dtype=np.float32), (64, 1))
    view.show_image(data, "t")

    # Tight percentiles -> narrower clim than full-range percentiles.
    view.set_display_range(40.0, 60.0)
    lo1, hi1 = view._im.get_clim()
    view.set_display_range(0.0, 100.0)
    lo2, hi2 = view._im.get_clim()
    assert (hi1 - lo1) < (hi2 - lo2)
    assert view.display_range()[:2] == (0.0, 100.0)


def test_margin_change_reshows(qapp):
    from wavefront_refocus.gui.phase_view import PhaseView

    view = PhaseView()
    data = np.random.rand(64, 64).astype(np.float32)
    view.show_image(data, "t")
    view.set_display_range(1.0, 99.0, margin=8)
    # cropped data is 64 - 2*8 = 48 on a side
    assert view._last_data.shape == (48, 48)
    assert view.display_range()[2] == 8
