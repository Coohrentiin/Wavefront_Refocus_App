"""QThread wiring around the vendored ``PropagationWorker``.

``PropagationController`` owns the thread/worker lifetime and relays progress,
results (wrapped as :class:`PropagationResult`) and failures as Qt signals.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..vendor.utils_propag import PropagationWorker
from .model import PropagationResult


class PropagationController(QObject):
    """Run a propagation sweep on a background thread."""

    progress = Signal(int, int)        # (done, total)
    finished = Signal(object)          # PropagationResult
    failed = Signal(str)               # traceback / message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: PropagationWorker | None = None
        self._center_dz_sample: float = 0.0
        self._magnification: float = 1.0

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, kwargs: dict, center_dz_sample: float = 0.0,
              magnification: float = 1.0) -> bool:
        """Start a sweep. Returns False if one is already running.

        ``center_dz_sample`` is the sample-space plane the sweep is centred on;
        it is folded back into the result's dz arrays so they stay relative to
        the frame's base plane.
        """
        if self.is_running():
            return False

        self._center_dz_sample = float(center_dz_sample)
        self._magnification = float(magnification)

        self._thread = QThread()
        self._worker = PropagationWorker(kwargs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._thread.start()
        return True

    @Slot()
    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    @Slot(object)
    def _on_finished(self, result: object) -> None:
        wrapped = (
            PropagationResult.from_dict(
                result, self._center_dz_sample, self._magnification
            )
            if result is not None
            else None
        )
        self._teardown()
        if wrapped is not None:
            self.finished.emit(wrapped)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._teardown()
        self.failed.emit(message)

    def _teardown(self) -> None:
        thread, worker = self._thread, self._worker
        self._thread, self._worker = None, None
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()
