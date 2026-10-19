import sys

from PySide6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)  # noqa: F841 - must stay alive while the GUI runs
    # Before the GUI module is imported, since that loads OpenGL.
    from lithophane.glsetup import match_qt_opengl

    match_qt_opengl()
    from lithophane.gui import run

    run()


if __name__ == "__main__":
    main()
