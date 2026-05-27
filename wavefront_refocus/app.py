"""Application entry point."""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .gui.theme import apply_dark

    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark(app)

    # Import after rcParams are set so figures pick up the dark palette.
    from .gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
