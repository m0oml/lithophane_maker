"""Small custom widgets for the sidebar: switch, segmented buttons and section cards."""

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractButton, QButtonGroup, QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from lithophane import theme


class ToggleSwitch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self):
        return QSize(38, 22)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.isEnabled():
            painter.setOpacity(0.4)
        track = QRectF(1, 2, 36, 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.ACCENT if self.isChecked() else "#3d434c"))
        painter.drawRoundedRect(track, 9, 9)
        knob_x = track.right() - 16 if self.isChecked() else track.left() + 2
        painter.setBrush(QColor("#fbfbfb"))
        painter.drawEllipse(QRectF(knob_x, track.top() + 2, 14, 14))


class SegmentedControl(QWidget):
    """A row of joined buttons, exactly one of which is selected."""

    changed = Signal(str)

    def __init__(self, options, parent=None):
        """options is a list of (label, key); changed and current() use the key."""
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.group = QButtonGroup(self)
        self.keys = {}
        for i, (label, key) in enumerate(options):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            position = "first" if i == 0 else "last" if i == len(options) - 1 else "middle"
            button.setProperty("segment", position)
            self.group.addButton(button)
            self.keys[button] = key
            layout.addWidget(button)
        self.group.buttons()[0].setChecked(True)
        self.group.buttonToggled.connect(
            lambda button, on: on and self.changed.emit(self.keys[button])
        )

    def current(self):
        return self.keys[self.group.checkedButton()]

    def set_current(self, key):
        for button, k in self.keys.items():
            if k == key:
                button.setChecked(True)


class Section(QFrame):
    """Card with a title and a form; checkable ones fold their form away when switched off."""

    toggled = Signal(bool)

    def __init__(self, title, checkable=False, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        self.header.addWidget(title_label)
        self.header.addStretch()
        outer.addLayout(self.header)

        self.note = QLabel()
        self.note.setObjectName("note")
        self.note.setWordWrap(True)
        self.note.hide()
        outer.addWidget(self.note)

        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setHorizontalSpacing(12)
        self.form.setVerticalSpacing(8)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        outer.addWidget(self.body)

        self.switch = None
        if checkable:
            self.switch = ToggleSwitch()
            self.switch.toggled.connect(self._on_toggled)
            self.header.addWidget(self.switch)
            self.body.setVisible(False)

    def _on_toggled(self, on):
        self.body.setVisible(on and self.isEnabled())
        self.toggled.emit(on)

    def isChecked(self):
        return self.switch is None or self.switch.isChecked()

    def setChecked(self, on):
        if self.switch is not None:
            self.switch.setChecked(on)

    def align_labels(self, width):
        for row in range(self.form.rowCount()):
            label = self.form.itemAt(row, QFormLayout.LabelRole)
            field = self.form.itemAt(row, QFormLayout.FieldRole)
            if label is not None and label.widget() is not None:
                label.widget().setFixedWidth(width)
                # As tall as its box, text centred, so it lines up with the value.
                label.widget().setMinimumHeight(field.widget().sizeHint().height())
                label.widget().setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def set_available(self, available, reason=""):
        """Grey the section out and show why, e.g. the stand needs the frame."""
        self.setEnabled(available)
        self.note.setText(reason)
        self.note.setVisible(not available and bool(reason))
        self.body.setVisible(available and self.isChecked())
