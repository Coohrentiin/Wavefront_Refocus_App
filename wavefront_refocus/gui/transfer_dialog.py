"""Dialog to transfer refocusing from one pair of distances JSONs onto the
currently loaded frames.

The user picks a **target** table (where frames should end up) and optionally
an **input** table (where they were when the target was produced). What gets
applied is the per-frame difference, so refocusing carries across even when the
current sequence sits on a different base. See
:func:`~wavefront_refocus.io.distances.transfer_distances`.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..io import distances as dist_io


class _JsonRow(QWidget):
    """Read-only path field with a Browse button for a .json file."""

    def __init__(self, on_change, parent=None):
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.edit.textChanged.connect(lambda _: on_change())
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.edit.clear())
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit)
        lay.addWidget(btn)
        lay.addWidget(clear)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select distances JSON", filter="JSON (*.json)"
        )
        if path:
            self.edit.setText(path)

    def path(self) -> str:
        return self.edit.text()


class TransferDialog(QDialog):
    """Preview and confirm a distance transfer onto the loaded frames.

    ``report`` holds the :class:`~wavefront_refocus.io.distances.TransferReport`
    for the accepted transfer; the caller applies it.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transfer distances")
        self.resize(620, 460)
        self._session = session
        self.report = None
        self.target_raw: dict = {}
        self.input_raw: dict = {}

        self.input_row = _JsonRow(self._refresh)
        self.input_row.edit.setPlaceholderText(
            "optional — leave empty if the current distances are the baseline"
        )
        self.input_row.setToolTip(
            "Where the frames sat when the target was produced.\n"
            "Leave empty to apply the target outright (input treated as 0)."
        )

        self.target_row = _JsonRow(self._refresh)
        self.target_row.edit.setPlaceholderText("required — distances to transfer")
        self.target_row.setToolTip("Where the frames should end up.")

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Select a target JSON to preview.")

        form = QFormLayout()
        form.addRow("Input distances", self.input_row)
        form.addRow("Target distances", self.target_row)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.button(QDialogButtonBox.Ok).setText("Apply transfer")
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("Preview"))
        lay.addWidget(self.summary, 1)
        lay.addWidget(self.buttons)

        self._refresh()

    # ── preview ──
    def _refresh(self):
        """Re-run the transfer in preview mode and describe what it would do."""
        self.report = None
        self.target_raw = {}
        self.input_raw = {}
        ok = self.buttons.button(QDialogButtonBox.Ok)

        target_path = self.target_row.path()
        if not target_path:
            self.summary.setPlainText("")
            ok.setEnabled(False)
            return

        try:
            self.target_raw = dist_io.load_distances(target_path)
            if self.input_row.path():
                self.input_raw = dist_io.load_distances(self.input_row.path())
            report = dist_io.transfer_distances(
                self._session.frames,
                self.target_raw,
                self._session.optics,
                self.input_raw,
                apply_to_frames=False,      # preview only
            )
        except Exception as exc:  # noqa: BLE001 — show the problem, don't crash
            self.summary.setPlainText(f"Cannot transfer:\n{exc}")
            ok.setEnabled(False)
            return

        self.report = report
        self.summary.setPlainText(self._describe(report))
        ok.setEnabled(report.n_applied > 0)

    def _describe(self, report) -> str:
        lines = []
        n = len(self._session.frames)
        lines.append(
            f"Applies to {report.n_applied} of {n} loaded frame(s)."
        )
        if not self.input_raw:
            lines.append(
                "No input JSON: the target is applied outright "
                "(input treated as 0)."
            )
        lo, hi = report.shift_range_m()
        if report.n_applied:
            lines.append(
                f"Displacement range: {lo * 1e6:+.3f} µm to {hi * 1e6:+.3f} µm "
                "(sample space)."
            )
        if report.overwritten:
            lines.append(
                f"Overwrites the position of {len(report.overwritten)} frame(s) "
                f"you already set: {_short(report.overwritten)}."
            )
        if report.uncovered:
            lines.append(
                f"Not in the target, left unchanged "
                f"({len(report.uncovered)}): {_short(report.uncovered)}."
            )
        if report.assumed_zero_input:
            lines.append(
                f"Missing from the input JSON, so the input was taken as 0 "
                f"({len(report.assumed_zero_input)}): "
                f"{_short(report.assumed_zero_input)}."
            )
        if report.unused:
            lines.append(
                f"In the target but not loaded here, ignored "
                f"({len(report.unused)}): {_short(report.unused)}."
            )

        lines.append("")
        lines.append("Per-frame displacement (sample space):")
        items = sorted(report.applied.items())
        for idx, dz in items[:40]:
            lines.append(f"  frame {idx}: {dz * 1e6:+.3f} µm")
        if len(items) > 40:
            lines.append(f"  … and {len(items) - 40} more")
        return "\n".join(lines)

    def _on_accept(self):
        if self.report is None or self.report.n_applied == 0:
            return
        self.accept()


def _short(items: list, limit: int = 10) -> str:
    head = ", ".join(str(i) for i in items[:limit])
    return head if len(items) <= limit else f"{head}, … (+{len(items) - limit})"
