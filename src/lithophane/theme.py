"""Dark theme: palette, stylesheet and the small icons the stylesheet needs."""

import atexit
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap, QRadialGradient,
)

BACKGROUND = "#131518"
SIDEBAR = "#1b1e23"
CARD = "#22262c"
INPUT = "#2b3037"
HOVER = "#343a43"
BORDER = "#363c45"
TEXT = "#e7e9ec"
DIM = "#8d949e"
ACCENT = "#f0a53a"
ACCENT_HOVER = "#f6b95e"
ON_ACCENT = "#1d1407"
VIEWPORT = "#181a1f"
VIEWPORT_BACKLIT = "#07080a"


def _icon_dir():
    """Stylesheets can only take images from files, so draw them once into a temp folder."""
    folder = Path(tempfile.mkdtemp(prefix="lithophane-theme-"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)

    def draw(name, size, paint):
        # Drawn at 2x so they stay crisp on high-DPI screens.
        pixmap = QPixmap(size[0] * 2, size[1] * 2)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(2, 2)
        paint(painter)
        painter.end()
        pixmap.save(str(folder / name))

    def chevron(points, colour):
        def paint(painter):
            painter.setPen(QPen(QColor(colour), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            path = QPainterPath(QPointF(*points[0]))
            for p in points[1:]:
                path.lineTo(QPointF(*p))
            painter.drawPath(path)
        return paint

    draw("check.png", (12, 12), chevron([(2.5, 6.3), (4.9, 8.6), (9.5, 3.6)], ON_ACCENT))
    draw("up.png", (9, 6), chevron([(1.5, 4.5), (4.5, 1.5), (7.5, 4.5)], DIM))
    draw("down.png", (9, 6), chevron([(1.5, 1.5), (4.5, 4.5), (7.5, 1.5)], DIM))
    return folder.as_posix()


def app_icon():
    """A small glowing panel, like a lit lithophane."""
    icon = QIcon()
    for size in (16, 32, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(size * 0.12, size * 0.06, size * 0.76, size * 0.88)
        glow = QRadialGradient(rect.center(), size * 0.55)
        glow.setColorAt(0.0, QColor("#ffe2a8"))
        glow.setColorAt(0.55, QColor(ACCENT))
        glow.setColorAt(1.0, QColor("#7a4a12"))
        painter.setBrush(glow)
        painter.setPen(QPen(QColor("#3a2508"), max(1.0, size / 32)))
        painter.drawRoundedRect(rect, size * 0.1, size * 0.1)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def apply(app):
    app.setStyle("Fusion")
    palette = QPalette()
    for role, colour in (
        (QPalette.Window, SIDEBAR),
        (QPalette.WindowText, TEXT),
        (QPalette.Base, INPUT),
        (QPalette.AlternateBase, CARD),
        (QPalette.Text, TEXT),
        (QPalette.Button, INPUT),
        (QPalette.ButtonText, TEXT),
        (QPalette.Highlight, ACCENT),
        (QPalette.HighlightedText, ON_ACCENT),
        (QPalette.ToolTipBase, INPUT),
        (QPalette.ToolTipText, TEXT),
        (QPalette.PlaceholderText, DIM),
        (QPalette.Link, ACCENT),
    ):
        palette.setColor(role, QColor(colour))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor("#5d636c"))
    app.setPalette(palette)
    app.setWindowIcon(app_icon())

    icons = _icon_dir()
    app.setStyleSheet(f"""
        QMainWindow, QWidget#central {{ background: {BACKGROUND}; }}
        QToolTip {{
            background: {INPUT}; color: {TEXT}; border: 1px solid {BORDER};
            border-radius: 4px; padding: 6px;
        }}

        QFrame#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
        QLabel#appTitle {{ font-size: 15pt; font-weight: 600; }}
        QLabel#appSubtitle {{ color: {DIM}; }}
        QScrollArea, QWidget#sidebarContent {{ background: transparent; border: none; }}

        QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}
        QFrame#card QLabel {{ color: {DIM}; }}
        QFrame#card QLabel#sectionTitle {{ color: {TEXT}; font-weight: 600; font-size: 11pt; }}
        QFrame#card QLabel#note {{ color: {DIM}; font-style: italic; }}

        QLabel#thumb {{
            background: {INPUT}; border: 1px dashed #4a515b; border-radius: 8px; color: {DIM};
        }}
        QLabel#thumb[loaded="true"] {{ background: #0f1012; border: 1px solid {BORDER}; }}

        QFrame#info {{ background: #26221b; border: 1px solid #4a3a20; border-radius: 10px; }}
        QFrame#info QLabel {{ color: {TEXT}; }}

        QAbstractSpinBox {{
            background: {INPUT}; border: 1px solid {BORDER}; border-radius: 6px;
            padding: 4px 6px; min-height: 20px; selection-background-color: {ACCENT};
            selection-color: {ON_ACCENT};
        }}
        QAbstractSpinBox:hover {{ border-color: #4a515b; }}
        QAbstractSpinBox:focus {{ border-color: {ACCENT}; }}
        QAbstractSpinBox:disabled {{ color: #5d636c; }}
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
            subcontrol-origin: border; width: 20px; border-left: 1px solid {BORDER};
            background: transparent;
        }}
        QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 6px; }}
        QAbstractSpinBox::down-button {{
            subcontrol-position: bottom right; border-bottom-right-radius: 6px;
        }}
        QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
            background: {HOVER};
        }}
        QAbstractSpinBox::up-arrow {{ image: url({icons}/up.png); width: 9px; height: 6px; }}
        QAbstractSpinBox::down-arrow {{ image: url({icons}/down.png); width: 9px; height: 6px; }}

        QCheckBox {{ spacing: 8px; color: {TEXT}; }}
        QCheckBox::indicator {{
            width: 16px; height: 16px; border-radius: 4px; border: 1px solid #4a515b;
            background: {INPUT};
        }}
        QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
        QCheckBox::indicator:checked {{
            background: {ACCENT}; border-color: {ACCENT}; image: url({icons}/check.png);
        }}

        QPushButton {{
            background: {INPUT}; border: 1px solid {BORDER}; border-radius: 7px;
            padding: 6px 14px; color: {TEXT};
        }}
        QPushButton:hover {{ background: {HOVER}; }}
        QPushButton:pressed {{ background: {BORDER}; }}
        QPushButton:disabled {{ color: #5d636c; }}
        QPushButton#primary {{
            background: {ACCENT}; color: {ON_ACCENT}; border: none; font-weight: 600;
            font-size: 11pt; padding: 11px;
        }}
        QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
        QPushButton#primary:disabled {{ background: {INPUT}; color: #5d636c; }}

        QPushButton[segment] {{ border-radius: 0; padding: 6px 12px; margin: 0; }}
        QPushButton[segment="first"] {{
            border-top-left-radius: 7px; border-bottom-left-radius: 7px;
        }}
        QPushButton[segment="last"] {{
            border-top-right-radius: 7px; border-bottom-right-radius: 7px;
        }}
        QPushButton[segment="middle"], QPushButton[segment="last"] {{ border-left: none; }}
        QPushButton[segment]:checked {{
            background: {ACCENT}; color: {ON_ACCENT}; border-color: {ACCENT};
        }}

        QFrame#viewbar {{ background: {SIDEBAR}; border-bottom: 1px solid {BORDER}; }}
        QLabel#viewTitle {{ font-weight: 600; }}
        QLabel#hint {{ color: {DIM}; }}

        QStatusBar {{ background: {SIDEBAR}; color: {DIM}; border-top: 1px solid {BORDER}; }}
        QStatusBar::item {{ border: none; }}

        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: #3a4048; border-radius: 3px; min-height: 30px; }}
        QScrollBar::handle:vertical:hover {{ background: #4a515b; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    """)
