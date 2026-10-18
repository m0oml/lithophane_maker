from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import shaders
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPixmap, QVector3D
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

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
        self.setBackgroundColor(QColor("#30343a"))
        self.mesh_item = None
        self.stand_item = None
        self.backlit = False
        self._mesh = None

    def set_backlit(self, backlit: bool):
        self.backlit = backlit
        self.setBackgroundColor(QColor("#0c0d0f" if backlit else "#30343a"))
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
        lo, hi = everything.min(axis=0), everything.max(axis=0)
        self.opts["center"] = QVector3D(*((lo + hi) / 2))
        if reset_view:
            # Face the picture: arcs face +y with z up; flat pieces face +z with y up.
            if picture_faces_y:
                self.setCameraPosition(elevation=15, azimuth=90)
            else:
                self.setCameraPosition(elevation=70, azimuth=-90)
            self.setCameraPosition(distance=1.6 * float(np.max(hi - lo)))

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lithophane Maker")
        self.image_path: str | None = None
        self.image_size: tuple[int, int] | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("No image loaded")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(240, 240)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        open_button = QPushButton("Open Image…")
        open_button.clicked.connect(self.open_image)
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(open_button)
        preview_layout.addStretch()
        layout.addLayout(preview_layout)

        model_layout = QVBoxLayout()
        self.model_preview = ModelPreviewWidget()
        model_layout.addWidget(self.model_preview, stretch=1)
        model_footer = QHBoxLayout()
        self.backlit_checkbox = QCheckBox("Backlit")
        self.backlit_checkbox.setToolTip("Show roughly how it will look with a light behind it.")
        self.backlit_checkbox.toggled.connect(self.model_preview.set_backlit)
        model_footer.addWidget(self.backlit_checkbox)
        model_footer.addStretch()
        model_footer.addWidget(QLabel("Drag to rotate, scroll to zoom. Shown at reduced detail."))
        model_layout.addLayout(model_footer)
        layout.addLayout(model_layout, stretch=1)

        form_layout = QVBoxLayout()
        form = QFormLayout()

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Flat panel", "Curved arc", "Circle"])
        self.shape_combo.currentTextChanged.connect(self._on_shape_changed)
        form.addRow("Shape", self.shape_combo)

        self.size_mm = QDoubleSpinBox()
        self.size_mm.setRange(10.0, 1000.0)
        self.size_mm.setValue(80.0)
        self.size_mm.setSuffix(" mm")
        self.size_row_label = QLabel("Width")
        form.addRow(self.size_row_label, self.size_mm)

        self.arc_angle_deg = QDoubleSpinBox()
        self.arc_angle_deg.setRange(1.0, 350.0)
        self.arc_angle_deg.setValue(60.0)
        self.arc_angle_deg.setSuffix(" °")
        self.arc_angle_row_label = QLabel("Angle")
        form.addRow(self.arc_angle_row_label, self.arc_angle_deg)

        self.min_thickness_mm = QDoubleSpinBox()
        self.min_thickness_mm.setRange(0.1, 10.0)
        self.min_thickness_mm.setSingleStep(0.1)
        self.min_thickness_mm.setValue(0.6)
        self.min_thickness_mm.setSuffix(" mm")
        form.addRow("Min thickness", self.min_thickness_mm)

        self.max_thickness_mm = QDoubleSpinBox()
        self.max_thickness_mm.setRange(0.2, 20.0)
        self.max_thickness_mm.setSingleStep(0.1)
        self.max_thickness_mm.setValue(3.0)
        self.max_thickness_mm.setSuffix(" mm")
        form.addRow("Max thickness", self.max_thickness_mm)

        self.mm_per_pixel = QDoubleSpinBox()
        self.mm_per_pixel.setRange(0.05, 2.0)
        self.mm_per_pixel.setDecimals(2)
        self.mm_per_pixel.setSingleStep(0.05)
        self.mm_per_pixel.setValue(0.1)
        self.mm_per_pixel.setSuffix(" mm")
        form.addRow("mm per pixel", self.mm_per_pixel)

        self.invert_checkbox = QCheckBox("Invert (dark = thin)")
        form.addRow(self.invert_checkbox)

        self.inside_checkbox = QCheckBox("Inside (picture on the inside of the curve)")
        form.addRow(self.inside_checkbox)

        form_layout.addLayout(form)

        frame_group = QGroupBox("Frame")
        frame_group.setCheckable(True)
        frame_group.setChecked(False)
        frame_form = QFormLayout(frame_group)

        self.frame_thickness_mm = QDoubleSpinBox()
        self.frame_thickness_mm.setRange(0.5, 50.0)
        self.frame_thickness_mm.setSingleStep(0.5)
        self.frame_thickness_mm.setValue(3.0)
        self.frame_thickness_mm.setSuffix(" mm")
        frame_form.addRow("Thickness", self.frame_thickness_mm)

        self.frame_depth_mm = QDoubleSpinBox()
        self.frame_depth_mm.setRange(0.0, 50.0)
        self.frame_depth_mm.setSingleStep(0.5)
        self.frame_depth_mm.setValue(4.0)
        self.frame_depth_mm.setSuffix(" mm")
        frame_form.addRow("Depth", self.frame_depth_mm)

        self.frame_angle_deg = QDoubleSpinBox()
        self.frame_angle_deg.setRange(0.0, 80.0)
        self.frame_angle_deg.setValue(45.0)
        self.frame_angle_deg.setSuffix(" °")
        frame_form.addRow("Bevel angle", self.frame_angle_deg)

        self.frame_group = frame_group
        form_layout.addWidget(frame_group)

        self.hook_group = QGroupBox("Hook")
        self.hook_group.setCheckable(True)
        self.hook_group.setChecked(False)
        hook_form = QFormLayout(self.hook_group)

        self.hook_position_pct = QDoubleSpinBox()
        self.hook_position_pct.setRange(-50.0, 50.0)
        self.hook_position_pct.setValue(0.0)
        self.hook_position_pct.setSuffix(" %")
        self.hook_position_pct.setToolTip("0% is the top; positive goes clockwise.")
        hook_form.addRow("Position", self.hook_position_pct)

        self.hook_hole_mm = QDoubleSpinBox()
        self.hook_hole_mm.setRange(1.0, 20.0)
        self.hook_hole_mm.setSingleStep(0.5)
        self.hook_hole_mm.setValue(3.5)
        self.hook_hole_mm.setSuffix(" mm")
        hook_form.addRow("Hole diameter", self.hook_hole_mm)

        self.hook_offset_mm = QDoubleSpinBox()
        self.hook_offset_mm.setRange(0.0, 30.0)
        self.hook_offset_mm.setSingleStep(0.5)
        self.hook_offset_mm.setValue(0.0)
        self.hook_offset_mm.setSuffix(" mm")
        hook_form.addRow("Offset", self.hook_offset_mm)

        form_layout.addWidget(self.hook_group)

        self.stand_group = QGroupBox("Stand (extra STL to slot the litho into)")
        self.stand_group.setCheckable(True)
        self.stand_group.setChecked(False)
        self.stand_group.setToolTip(
            "Extends the frame's bottom edge into a tab and makes a stand with a matching slot.\n"
            "Needs the frame turned on."
        )
        stand_form = QFormLayout(self.stand_group)

        self.stand_slot_mm = QDoubleSpinBox()
        self.stand_slot_mm.setRange(2.0, 50.0)
        self.stand_slot_mm.setSingleStep(0.5)
        self.stand_slot_mm.setValue(8.0)
        self.stand_slot_mm.setSuffix(" mm")
        self.stand_slot_mm.setToolTip("How far the litho's tab goes down into the stand.")
        stand_form.addRow("Slot depth", self.stand_slot_mm)

        self.stand_clearance_mm = QDoubleSpinBox()
        self.stand_clearance_mm.setRange(0.0, 2.0)
        self.stand_clearance_mm.setSingleStep(0.05)
        self.stand_clearance_mm.setValue(0.3)
        self.stand_clearance_mm.setSuffix(" mm")
        self.stand_clearance_mm.setToolTip("Gap on each side of the tab, so it slips in.")
        stand_form.addRow("Clearance", self.stand_clearance_mm)

        self.stand_wall_mm = QDoubleSpinBox()
        self.stand_wall_mm.setRange(1.0, 20.0)
        self.stand_wall_mm.setSingleStep(0.5)
        self.stand_wall_mm.setValue(3.0)
        self.stand_wall_mm.setSuffix(" mm")
        self.stand_wall_mm.setToolTip("Thickness of the stand's floor and end walls.")
        stand_form.addRow("Wall", self.stand_wall_mm)

        self.stand_depth_mm = QDoubleSpinBox()
        self.stand_depth_mm.setRange(5.0, 200.0)
        self.stand_depth_mm.setValue(30.0)
        self.stand_depth_mm.setSuffix(" mm")
        self.stand_depth_mm.setToolTip("Front-to-back size of the stand's footprint.")
        stand_form.addRow("Footprint depth", self.stand_depth_mm)

        self.stand_chamfer_mm = QDoubleSpinBox()
        self.stand_chamfer_mm.setRange(0.0, 20.0)
        self.stand_chamfer_mm.setSingleStep(0.5)
        self.stand_chamfer_mm.setValue(2.0)
        self.stand_chamfer_mm.setSuffix(" mm")
        self.stand_chamfer_mm.setToolTip(
            "45° chamfer on the stand's outer top edge (0 for a square edge).\n"
            "Must be less than the wall."
        )
        stand_form.addRow("Chamfer", self.stand_chamfer_mm)

        form_layout.addWidget(self.stand_group)
        self.frame_group.toggled.connect(self.stand_group.setEnabled)
        self.stand_group.setEnabled(self.frame_group.isChecked())

        self.size_label = QLabel()
        form_layout.addWidget(self.size_label)

        for spin in (
            self.size_mm, self.arc_angle_deg, self.mm_per_pixel,
            self.max_thickness_mm, self.frame_thickness_mm, self.frame_depth_mm,
            self.frame_angle_deg,
        ):
            spin.valueChanged.connect(self._update_estimate)
        self.frame_group.toggled.connect(self._update_estimate)

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

        self.generate_button = QPushButton("Generate STL…")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_stl)
        form_layout.addWidget(self.generate_button)
        form_layout.addStretch()

        layout.addLayout(form_layout)

        self.setStatusBar(QStatusBar())
        self._on_shape_changed(self.shape_combo.currentText())

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

    def _update_estimate(self):
        if not self.image_size:
            self.size_label.setText("Open an image to see the size.")
            return
        mm = self.mm_per_pixel.value()
        extent = frame_extent_mm(
            self.frame_group.isChecked(), self.max_thickness_mm.value(),
            self.frame_thickness_mm.value(), self.frame_depth_mm.value(),
            self.frame_angle_deg.value(),
        )
        if self.shape_combo.currentText() == "Circle":
            picture = circle_picture_diameter_mm(self.size_mm.value(), extent)
            if picture < 4 * mm:
                self.size_label.setText("Diameter is too small to fit the frame.")
                return
            triangles = 1.6 * (picture / mm) ** 2
            self.size_label.setText(
                f"Picture diameter: {picture:.1f} mm (photo is centre-cropped)\n"
                f"Estimated file size: {triangles * 50 / 1e6:.0f} MB"
            )
            return
        try:
            if self.shape_combo.currentText() == "Curved arc":
                rows, cols, width = panel_dimensions_from_height(
                    self.image_size, self.size_mm.value(), mm, extent
                )
                radius = arc_radius_mm(width, self.arc_angle_deg.value())
                dims = (
                    f"Width (around the curve): {width:.1f} mm\n"
                    f"Radius (flat side): {radius:.1f} mm"
                )
            else:
                rows, cols, height = panel_dimensions(
                    self.image_size, self.size_mm.value(), mm, extent
                )
                dims = f"Height: {height:.1f} mm"
        except ValueError as exc:
            self.size_label.setText(str(exc))
            return
        triangles = 2 * rows * cols + 12 * (rows + cols)
        megabytes = triangles * 50 / 1e6
        self.size_label.setText(f"{dims}\nEstimated file size: {megabytes:.0f} MB")

    def _stand_wanted(self):
        return (
            self.shape_combo.currentText() != "Circle"
            and self.frame_group.isChecked()
            and self.stand_group.isChecked()
        )

    def _build_stand(self, vertices, output_path):
        """Stand the litho up and make its stand; returns both in the assembled position."""
        curved = self.shape_combo.currentText() == "Curved arc"
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
        shape = self.shape_combo.currentText()
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
        shape = self.shape_combo.currentText()
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
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",
        )
        if not path:
            return
        self.image_path = path
        original = QPixmap(path)
        self.image_size = (original.width(), original.height())
        self._update_estimate()
        self._reset_view = True
        self._update_model_preview()
        pixmap = original.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)
        self.generate_button.setEnabled(True)
        self.statusBar().showMessage(f"Loaded {path}")

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
    window = MainWindow()
    window.resize(1240, 680)
    window.show()
    app.exec()
