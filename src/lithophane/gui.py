from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import shaders
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPixmap, QVector3D
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from lithophane import theme

from lithophane.mesh import (
    arc_radius_mm,
    circle_picture_diameter_mm,
    frame_extent_mm,
    generate_lithophane_arc,
    generate_lithophane_circle,
    generate_lithophane_flat,
    generate_stand,
    panel_dimensions,
    panel_dimensions_from_height,
    standing_lithophane,
)
from lithophane.widgets import Section, SegmentedControl


# pyqtgraph's own "shaded" program lights faces pointing away from the camera, which leaves
# the picture side dark; this one shines from above-left of the viewer, so relief shows up.
shaders.ShaderProgram("litho", [
    shaders.VertexShader("""
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    shaders.FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            float key = max(dot(v_normal, normalize(vec3(-0.4, 0.6, 1.0))), 0.0);
            float fill = max(dot(v_normal, normalize(vec3(0.6, -0.3, 0.4))), 0.0);
            gl_FragColor = vec4(v_color.rgb * (0.15 + 0.75 * key + 0.2 * fill), v_color.a);
        }
    """),
])

# Backlit: each vertex carries the colour of light getting through the material there, so
# the picture glows like a printed lithophane on a lamp. A touch of key light keeps the
# shape readable.
shaders.ShaderProgram("backlit", [
    shaders.VertexShader("""
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """),
    shaders.FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            float key = max(dot(v_normal, normalize(vec3(-0.4, 0.6, 1.0))), 0.0);
            gl_FragColor = vec4(v_color.rgb + 0.05 * key, v_color.a);
        }
    """),
])

PLASTIC_COLOUR = (0.95, 0.93, 0.88, 1.0)
STAND_COLOUR = (0.45, 0.47, 0.52, 1.0)
STAND_BACKLIT_COLOUR = (0.12, 0.12, 0.13, 1.0)
LAMP_COLOUR = np.array([1.0, 0.86, 0.62])
# Light through white PLA falls off roughly exponentially with thickness; the thinnest
# part shows the full lamp and the thickest picture part 1/20 of it, about what prints show.
BACKLIT_CONTRAST = 20.0


class ModelPreviewWidget(gl.GLViewWidget):
    """Rotatable 3D view of the model: drag to orbit, scroll to zoom."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 420)
        self.setBackgroundColor(QColor(theme.VIEWPORT))
        self.mesh_item = None
        self.stand_item = None
        self.backlit = False
        self._mesh = None

    def set_backlit(self, backlit: bool):
        self.backlit = backlit
        self.setBackgroundColor(QColor(theme.VIEWPORT_BACKLIT if backlit else theme.VIEWPORT))
        if self._mesh is not None:
            self._show(*self._mesh)

    def set_mesh(self, vertices, faces, thickness, thickness_range, picture_faces_y: bool,
                 reset_view: bool, stand=None):
        """thickness is each vertex's distance from the flat side, for the backlit view.

        stand is an optional (vertices, faces) drawn alongside in its own colour.
        """
        self._mesh = (vertices, faces, thickness, thickness_range, stand)
        self._show(*self._mesh)

        everything = vertices if stand is None else np.concatenate([vertices, stand[0]])
        self._bounds = (everything.min(axis=0), everything.max(axis=0))
        self._picture_faces_y = picture_faces_y
        self.opts["center"] = QVector3D(*((self._bounds[0] + self._bounds[1]) / 2))
        if reset_view:
            self.reset_camera()

    def reset_camera(self):
        if self._mesh is None:
            return
        lo, hi = self._bounds
        self.opts["center"] = QVector3D(*((lo + hi) / 2))
        # Face the picture: arcs face +y with z up; flat pieces face +z with y up.
        if self._picture_faces_y:
            self.setCameraPosition(elevation=15, azimuth=90)
        else:
            self.setCameraPosition(elevation=70, azimuth=-90)
        self.setCameraPosition(distance=1.5 * float(np.max(hi - lo)))

    def _show(self, vertices, faces, thickness, thickness_range, stand):
        self._remove_items()
        if self.backlit:
            thinnest, thickest = thickness_range
            falloff = np.log(BACKLIT_CONTRAST) / max(thickest - thinnest, 1e-6)
            light = np.exp(-falloff * np.clip(thickness - thinnest, 0.0, None))
            # Points on the flat side aren't thin; without this they glow along the outer edges.
            light[thickness < 1e-3] = 0.0
            colours = np.column_stack([light[:, None] * LAMP_COLOUR, np.ones(len(light))])
            self.mesh_item = gl.GLMeshItem(
                vertexes=vertices, faces=faces, vertexColors=colours, smooth=False,
                shader="backlit",
            )
        else:
            self.mesh_item = gl.GLMeshItem(
                vertexes=vertices, faces=faces, smooth=False, shader="litho",
                color=PLASTIC_COLOUR,
            )
        self.addItem(self.mesh_item)
        if stand is not None:
            self.stand_item = gl.GLMeshItem(
                vertexes=stand[0], faces=stand[1], smooth=False, shader="litho",
                color=STAND_BACKLIT_COLOUR if self.backlit else STAND_COLOUR,
            )
            self.addItem(self.stand_item)

    def _remove_items(self):
        for item in (self.mesh_item, self.stand_item):
            if item is not None:
                self.removeItem(item)
        self.mesh_item = self.stand_item = None

    def clear_mesh(self):
        self._mesh = None
        self._remove_items()


def _spin(low, high, value, suffix, step=None, decimals=2, tip=None):
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSuffix(suffix)
    if step is not None:
        spin.setSingleStep(step)
    if tip:
        spin.setToolTip(tip)
    return spin


class ThumbnailLabel(QLabel):
    """Photo preview that also opens the file picker when clicked."""

    def __init__(self, on_click):
        super().__init__("Click or drop a photo here")
        self.setObjectName("thumb")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(190)
        self.setCursor(Qt.PointingHandCursor)
        self.on_click = on_click

    def mousePressEvent(self, event):
        self.on_click()


LABEL_WIDTH = 112  # same label column in every card, so the boxes line up
SHAPES = [("Flat", "Flat panel"), ("Arc", "Curved arc"), ("Circle", "Circle")]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lithophane Maker")
        self.setAcceptDrops(True)
        self.image_path: str | None = None
        self.image_size: tuple[int, int] | None = None

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self._build_viewport(), stretch=1)

        self.setStatusBar(QStatusBar())
        # Start on the most used set-up: a 100 mm tall, 60° arc with the frame on.
        self.set_shape("Curved arc")
        self.size_mm.setValue(100.0)
        self.frame_group.setChecked(True)
        self._connect_updates()
        self._on_shape_changed(self._shape())

    # ----- layout -----

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(360)
        column = QVBoxLayout(sidebar)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        brand = QHBoxLayout()
        brand.setContentsMargins(18, 16, 18, 10)
        brand.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(theme.app_icon().pixmap(30, 30))
        brand.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("Lithophane Maker")
        title.setObjectName("appTitle")
        subtitle = QLabel("Photo to printable lithophane")
        subtitle.setObjectName("appSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        brand.addLayout(titles)
        brand.addStretch()
        column.addLayout(brand)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("sidebarContent")
        cards = QVBoxLayout(content)
        cards.setContentsMargins(14, 4, 10, 14)
        cards.setSpacing(10)
        for section in (
            self._photo_section(), self._shape_section(), self._thickness_section(),
            self._frame_section(), self._hook_section(), self._stand_section(),
        ):
            section.align_labels(LABEL_WIDTH)
            cards.addWidget(section)
        cards.addStretch()
        scroll.setWidget(content)
        column.addWidget(scroll, stretch=1)

        footer = QVBoxLayout()
        footer.setContentsMargins(14, 10, 14, 14)
        footer.setSpacing(10)
        info = QFrame()
        info.setObjectName("info")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(14, 10, 14, 10)
        self.size_label = QLabel()
        self.size_label.setWordWrap(True)
        info_layout.addWidget(self.size_label)
        footer.addWidget(info)
        self.generate_button = QPushButton("Export STL…")
        self.generate_button.setObjectName("primary")
        self.generate_button.setCursor(Qt.PointingHandCursor)
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_stl)
        footer.addWidget(self.generate_button)
        column.addLayout(footer)
        return sidebar

    def _photo_section(self):
        section = Section("Photo")
        open_button = QPushButton("Choose…")
        open_button.setCursor(Qt.PointingHandCursor)
        open_button.clicked.connect(self.open_image)
        section.header.addWidget(open_button)
        self.preview_label = ThumbnailLabel(self.open_image)
        section.form.addRow(self.preview_label)
        self.photo_name = QLabel()
        self.photo_name.setObjectName("note")
        self.photo_name.hide()
        section.form.addRow(self.photo_name)
        return section

    def _shape_section(self):
        section = Section("Shape")
        self.shape_picker = SegmentedControl(SHAPES)
        self.shape_picker.changed.connect(self._on_shape_changed)
        section.form.addRow(self.shape_picker)
        self.size_mm = _spin(10.0, 1000.0, 80.0, " mm", decimals=1)
        self.size_row_label = QLabel("Width")
        section.form.addRow(self.size_row_label, self.size_mm)
        self.arc_angle_deg = _spin(1.0, 350.0, 60.0, " °", decimals=0)
        self.arc_angle_row_label = QLabel("Angle")
        section.form.addRow(self.arc_angle_row_label, self.arc_angle_deg)
        self.inside_checkbox = QCheckBox("Picture on the inside of the curve")
        section.form.addRow(self.inside_checkbox)
        return section

    def _thickness_section(self):
        section = Section("Thickness")
        self.min_thickness_mm = _spin(0.1, 10.0, 0.6, " mm", step=0.1)
        section.form.addRow("Thinnest", self.min_thickness_mm)
        self.max_thickness_mm = _spin(0.2, 20.0, 3.0, " mm", step=0.1)
        section.form.addRow("Thickest", self.max_thickness_mm)
        self.mm_per_pixel = _spin(
            0.05, 2.0, 0.1, " mm", step=0.05,
            tip="Size of one pixel on the print. Smaller is sharper but makes bigger files.",
        )
        section.form.addRow("Detail", self.mm_per_pixel)
        self.invert_checkbox = QCheckBox("Invert (dark areas thin)")
        section.form.addRow(self.invert_checkbox)
        return section

    def _frame_section(self):
        self.frame_group = section = Section("Frame", checkable=True)
        self.frame_thickness_mm = _spin(0.5, 50.0, 3.0, " mm", step=0.5)
        section.form.addRow("Thickness", self.frame_thickness_mm)
        self.frame_depth_mm = _spin(0.0, 50.0, 4.0, " mm", step=0.5)
        section.form.addRow("Depth", self.frame_depth_mm)
        self.frame_angle_deg = _spin(0.0, 80.0, 45.0, " °", decimals=0)
        section.form.addRow("Bevel angle", self.frame_angle_deg)
        return section

    def _hook_section(self):
        self.hook_group = section = Section("Hook", checkable=True)
        self.hook_position_pct = _spin(
            -50.0, 50.0, 0.0, " %", decimals=0, tip="0% is the top; positive goes clockwise."
        )
        section.form.addRow("Position", self.hook_position_pct)
        self.hook_hole_mm = _spin(1.0, 20.0, 3.5, " mm", step=0.5)
        section.form.addRow("Hole diameter", self.hook_hole_mm)
        self.hook_offset_mm = _spin(0.0, 30.0, 0.0, " mm", step=0.5)
        section.form.addRow("Offset", self.hook_offset_mm)
        return section

    def _stand_section(self):
        self.stand_group = section = Section("Stand", checkable=True)
        section.setToolTip(
            "Extends the frame's bottom edge into a tab and saves a second STL:\n"
            "a stand with a matching slot."
        )
        self.stand_slot_mm = _spin(
            2.0, 50.0, 8.0, " mm", step=0.5, tip="How far the litho's tab goes into the stand."
        )
        section.form.addRow("Slot depth", self.stand_slot_mm)
        self.stand_clearance_mm = _spin(
            0.0, 2.0, 0.3, " mm", step=0.05, tip="Gap on each side of the tab, so it slips in."
        )
        section.form.addRow("Clearance", self.stand_clearance_mm)
        self.stand_wall_mm = _spin(
            1.0, 20.0, 3.0, " mm", step=0.5, tip="Thickness of the stand's floor and end walls."
        )
        section.form.addRow("Wall", self.stand_wall_mm)
        self.stand_depth_mm = _spin(
            5.0, 200.0, 30.0, " mm", decimals=1, tip="Front-to-back size of the stand's footprint."
        )
        section.form.addRow("Footprint depth", self.stand_depth_mm)
        self.stand_chamfer_mm = _spin(
            0.0, 20.0, 2.0, " mm", step=0.5,
            tip="45° chamfer on the stand's outer top edge (0 for a square edge).\n"
                "Must be less than the wall.",
        )
        section.form.addRow("Chamfer", self.stand_chamfer_mm)
        return section

    def _build_viewport(self):
        viewport = QWidget()
        column = QVBoxLayout(viewport)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        bar = QFrame()
        bar.setObjectName("viewbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(16, 8, 12, 8)
        row.setSpacing(10)
        title = QLabel("Preview")
        title.setObjectName("viewTitle")
        row.addWidget(title)
        hint = QLabel("Drag to rotate · scroll to zoom · shown at reduced detail")
        hint.setObjectName("hint")
        row.addWidget(hint)
        row.addStretch()
        self.view_mode = SegmentedControl([("Model", "model"), ("Backlit", "backlit")])
        self.view_mode.setToolTip("Backlit shows roughly how it looks with a light behind it.")
        row.addWidget(self.view_mode)
        reset = QPushButton("Reset view")
        reset.setCursor(Qt.PointingHandCursor)
        row.addWidget(reset)
        column.addWidget(bar)

        self.model_preview = ModelPreviewWidget()
        placeholder = QLabel(
            f"<div style='font-size:15pt; color:{theme.TEXT}'>No photo yet</div>"
            f"<div style='color:{theme.DIM}; margin-top:6px'>"
            "Choose a photo, or drop one anywhere on the window,<br>"
            "to see your lithophane here.</div>"
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(f"background: {theme.VIEWPORT};")
        self.viewport_stack = QStackedWidget()
        self.viewport_stack.addWidget(placeholder)
        self.viewport_stack.addWidget(self.model_preview)
        column.addWidget(self.viewport_stack, stretch=1)
        self.view_mode.changed.connect(
            lambda mode: self.model_preview.set_backlit(mode == "backlit")
        )
        reset.clicked.connect(self.model_preview.reset_camera)
        return viewport

    def _connect_updates(self):
        for spin in (
            self.size_mm, self.arc_angle_deg, self.mm_per_pixel,
            self.max_thickness_mm, self.frame_thickness_mm, self.frame_depth_mm,
            self.frame_angle_deg,
        ):
            spin.valueChanged.connect(self._update_estimate)
        self.frame_group.toggled.connect(self._update_estimate)
        self.frame_group.toggled.connect(self._update_stand_availability)
        self._update_stand_availability()

        # Rebuild the 3D preview shortly after the last change, not on every keystroke.
        self._reset_view = True
        self._picture_faced_y = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._update_model_preview)
        for spin in (
            self.size_mm, self.arc_angle_deg, self.min_thickness_mm, self.max_thickness_mm,
            self.frame_thickness_mm, self.frame_depth_mm, self.frame_angle_deg,
            self.hook_position_pct, self.hook_hole_mm, self.hook_offset_mm,
            self.stand_slot_mm, self.stand_clearance_mm, self.stand_wall_mm, self.stand_depth_mm,
            self.stand_chamfer_mm,
        ):
            spin.valueChanged.connect(self._schedule_model_preview)
        for toggle in (
            self.invert_checkbox, self.inside_checkbox, self.frame_group, self.hook_group,
            self.stand_group,
        ):
            toggle.toggled.connect(self._schedule_model_preview)

    # ----- behaviour -----

    def _shape(self):
        return self.shape_picker.current()

    def set_shape(self, shape):
        self.shape_picker.set_current(shape)

    def _update_stand_availability(self):
        self.stand_group.set_available(
            self.frame_group.isChecked(), "Turn on the frame to add a stand."
        )

    def _on_shape_changed(self, shape: str):
        is_arc = shape == "Curved arc"
        is_circle = shape == "Circle"
        for w in (self.arc_angle_row_label, self.arc_angle_deg):
            w.setVisible(is_arc)
        self.inside_checkbox.setVisible(is_arc)
        self.hook_group.setVisible(is_circle)
        self.stand_group.setVisible(not is_circle)
        self.size_row_label.setText("Diameter" if is_circle else "Height" if is_arc else "Width")
        self._update_estimate()
        self._reset_view = True
        self._schedule_model_preview()

    def _set_info(self, rows):
        """rows is a list of (label, value) pairs, or a single message string."""
        if isinstance(rows, str):
            self.size_label.setText(rows)
            return
        cells = "".join(
            f"<tr><td style='color:{theme.DIM}; padding-right:14px'>{label}</td>"
            f"<td>{value}</td></tr>"
            for label, value in rows
        )
        self.size_label.setText(f"<table cellspacing='0' cellpadding='2'>{cells}</table>")

    def _update_estimate(self):
        if not self.image_size:
            self._set_info("Choose a photo to get started.")
            return
        mm = self.mm_per_pixel.value()
        extent = frame_extent_mm(
            self.frame_group.isChecked(), self.max_thickness_mm.value(),
            self.frame_thickness_mm.value(), self.frame_depth_mm.value(),
            self.frame_angle_deg.value(),
        )
        if self._shape() == "Circle":
            picture = circle_picture_diameter_mm(self.size_mm.value(), extent)
            if picture < 4 * mm:
                self._set_info("Diameter is too small to fit the frame.")
                return
            triangles = 1.6 * (picture / mm) ** 2
            self._set_info([
                ("Picture", f"{picture:.1f} mm across (centre-cropped)"),
                ("File size", f"about {triangles * 50 / 1e6:.0f} MB"),
            ])
            return
        try:
            if self._shape() == "Curved arc":
                rows, cols, width = panel_dimensions_from_height(
                    self.image_size, self.size_mm.value(), mm, extent
                )
                radius = arc_radius_mm(width, self.arc_angle_deg.value())
                dims = [
                    ("Width", f"{width:.1f} mm around the curve"),
                    ("Radius", f"{radius:.1f} mm (flat side)"),
                ]
            else:
                rows, cols, height = panel_dimensions(
                    self.image_size, self.size_mm.value(), mm, extent
                )
                dims = [("Height", f"{height:.1f} mm")]
        except ValueError as exc:
            self._set_info(str(exc))
            return
        triangles = 2 * rows * cols + 12 * (rows + cols)
        self._set_info(dims + [("File size", f"about {triangles * 50 / 1e6:.0f} MB")])

    def _stand_wanted(self):
        return (
            self._shape() != "Circle"
            and self.frame_group.isChecked()
            and self.stand_group.isChecked()
        )

    def _build_stand(self, vertices, output_path):
        """Stand the litho up and make its stand; returns both in the assembled position."""
        curved = self._shape() == "Curved arc"
        standing = standing_lithophane(vertices, curved)
        stand = generate_stand(
            standing, curved, output_path,
            slot_depth_mm=self.stand_slot_mm.value(),
            clearance_mm=self.stand_clearance_mm.value(),
            wall_mm=self.stand_wall_mm.value(),
            stand_depth_mm=self.stand_depth_mm.value(),
            chamfer_mm=self.stand_chamfer_mm.value(),
        )
        return standing, stand

    def _build_model(self, output_path, mm_per_pixel):
        common = dict(
            max_thickness_mm=self.max_thickness_mm.value(),
            min_thickness_mm=self.min_thickness_mm.value(),
            mm_per_pixel=mm_per_pixel,
            invert=self.invert_checkbox.isChecked(),
            frame_enabled=self.frame_group.isChecked(),
            frame_thickness_mm=self.frame_thickness_mm.value(),
            frame_depth_mm=self.frame_depth_mm.value(),
            frame_angle_deg=self.frame_angle_deg.value(),
        )
        shape = self._shape()
        if shape != "Circle":
            common["stand_tab_mm"] = self.stand_slot_mm.value() if self._stand_wanted() else 0.0
        if shape == "Curved arc":
            return generate_lithophane_arc(
                self.image_path,
                output_path,
                arc_angle_deg=self.arc_angle_deg.value(),
                height_mm=self.size_mm.value(),
                relief_inward=self.inside_checkbox.isChecked(),
                **common,
            )
        if shape == "Circle":
            return generate_lithophane_circle(
                self.image_path,
                output_path,
                diameter_mm=self.size_mm.value(),
                hook_enabled=self.hook_group.isChecked(),
                hook_position_pct=self.hook_position_pct.value(),
                hook_hole_diameter_mm=self.hook_hole_mm.value(),
                hook_offset_mm=self.hook_offset_mm.value(),
                **common,
            )
        return generate_lithophane_flat(
            self.image_path, output_path, width_mm=self.size_mm.value(), **common
        )

    def _schedule_model_preview(self):
        if self.image_path:
            self._preview_timer.start()

    def _update_model_preview(self):
        if not self.image_path:
            return
        # About 250 points across keeps the preview quick; the STL uses full detail.
        mm = max(self.mm_per_pixel.value(), self.size_mm.value() / 250)
        try:
            vertices, faces = self._build_model(None, mm)
            stand = None
            if self._stand_wanted():
                standing, stand = self._build_stand(vertices, None)
        except ValueError as exc:
            self.model_preview.clear_mesh()
            self.statusBar().showMessage(str(exc))
            return
        shape = self._shape()
        if shape == "Curved arc":
            # The flat side is the innermost radius, or the outermost when the picture is inside.
            r = np.hypot(vertices[:, 0], vertices[:, 1])
            radius = r.max() if self.inside_checkbox.isChecked() else r.min()
            thickness = np.abs(r - radius)
        else:
            thickness = vertices[:, 2]
        if stand is not None:
            vertices = standing
        picture_faces_y = shape == "Curved arc" or stand is not None
        # Re-aim the camera when the model turns to face a different way.
        if picture_faces_y != self._picture_faced_y:
            self._reset_view = True
            self._picture_faced_y = picture_faces_y
        self.model_preview.set_mesh(
            vertices, faces, thickness,
            (self.min_thickness_mm.value(), self.max_thickness_mm.value()),
            picture_faces_y=picture_faces_y,
            reset_view=self._reset_view,
            stand=stand,
        )
        self._reset_view = False

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self.load_image(path)

    def load_image(self, path):
        original = QPixmap(path)
        if original.isNull():
            QMessageBox.warning(self, "Can't open photo", f"Couldn't read {path} as an image.")
            return
        self.image_path = path
        self.image_size = (original.width(), original.height())
        self.viewport_stack.setCurrentWidget(self.model_preview)
        self._update_estimate()
        self._reset_view = True
        self._update_model_preview()
        self.preview_label.setPixmap(original.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.preview_label.setProperty("loaded", True)
        self.preview_label.style().polish(self.preview_label)
        self.photo_name.setText(f"{Path(path).name}  ·  {original.width()} × {original.height()} px")
        self.photo_name.show()
        self.generate_button.setEnabled(True)
        self.statusBar().showMessage(f"Loaded {path}")

    def _dropped_image(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):
        if self._dropped_image(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._dropped_image(event)
        if path:
            self.load_image(path)

    def generate_stl(self):
        if not self.image_path:
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save STL", str(Path(self.image_path).with_suffix(".stl")), "STL files (*.stl)"
        )
        if not output_path:
            return

        saved = [output_path]
        self.setCursor(Qt.WaitCursor)
        try:
            vertices, _ = self._build_model(output_path, self.mm_per_pixel.value())
            if self._stand_wanted():
                stand_path = str(Path(output_path).with_name(Path(output_path).stem + "_stand.stl"))
                self._build_stand(vertices, stand_path)
                saved.append(stand_path)
        except Exception as exc:
            QMessageBox.critical(self, "Error generating STL", str(exc))
        else:
            self.statusBar().showMessage("Saved " + " and ".join(saved))
        finally:
            self.unsetCursor()


def run():
    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    window = MainWindow()
    window.resize(1320, 840)
    window.show()
    app.exec()
