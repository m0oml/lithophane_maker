"""Convert a photo into a lithophane STL (bright pixels -> thin, dark pixels -> thick).

Supports a flat panel or a curved arc (a segment of a cylindrical shell, for
lamp-shade style prints), each with an optional beveled frame around the edge.
"""

import numpy as np
from PIL import Image
from stl import mesh as stl_mesh


def frame_extent_mm(frame_enabled, max_thickness_mm, thickness_mm, depth_mm, angle_deg):
    """How far the frame reaches beyond the picture edge, including the bevel."""
    if not frame_enabled:
        return 0.0
    return thickness_mm + max(depth_mm, max_thickness_mm) * np.tan(np.radians(angle_deg))


def panel_dimensions(image_size, width_mm, mm_per_pixel, frame_extent):
    """Picture grid size and overall height for a given overall width.

    width_mm is the whole piece, frame included (for an arc, measured around
    the curve); the height follows from the photo's aspect ratio.
    """
    img_w, img_h = image_size
    picture_w = width_mm - 2 * frame_extent
    if picture_w < 4 * mm_per_pixel:
        raise ValueError("Width is too small to fit the frame.")
    cols = max(4, round(picture_w / mm_per_pixel) + 1)
    rows = max(4, round((cols - 1) * img_h / img_w) + 1)
    height_mm = (rows - 1) * mm_per_pixel + 2 * frame_extent
    return rows, cols, height_mm


def panel_dimensions_from_height(image_size, height_mm, mm_per_pixel, frame_extent):
    """Picture grid size and overall width for a given overall height (frame included)."""
    img_w, img_h = image_size
    picture_h = height_mm - 2 * frame_extent
    if picture_h < 4 * mm_per_pixel:
        raise ValueError("Height is too small to fit the frame.")
    rows = max(4, round(picture_h / mm_per_pixel) + 1)
    cols = max(4, round((rows - 1) * img_w / img_h) + 1)
    width_mm = (cols - 1) * mm_per_pixel + 2 * frame_extent
    return rows, cols, width_mm


def arc_radius_mm(width_mm, arc_angle_deg):
    """Radius of the arc's flat side, for an overall width wrapped through the angle."""
    return width_mm / np.radians(arc_angle_deg)


def _load_thickness_map(img, cols, rows, max_thickness_mm, min_thickness_mm, invert):
    img = img.convert("L").resize((cols, rows), Image.LANCZOS)
    gray = np.asarray(img, dtype=np.float64) / 255.0
    if invert:
        gray = 1.0 - gray

    return min_thickness_mm + (1.0 - gray) * (max_thickness_mm - min_thickness_mm)


def _coarse_back_faces(rows, cols):
    """Triangulate the smooth side using only its edge vertices.

    It is flat (or a cylinder, which is straight along its height), so
    full-height strips per column are exact. The two end strips are fans so
    they still meet every edge vertex, keeping the mesh watertight.
    """

    def idx(r, c):
        return r * cols + c

    tris = []
    for c in range(1, cols - 2):
        tris.append((idx(0, c), idx(0, c + 1), idx(rows - 1, c)))
        tris.append((idx(0, c + 1), idx(rows - 1, c + 1), idx(rows - 1, c)))
    mid = rows // 2
    for edge_c, inner_c in ((0, 1), (cols - 1, cols - 2)):
        top, bottom = idx(0, inner_c), idx(rows - 1, inner_c)
        for r in range(rows - 1):
            tris.append((idx(r, edge_c), idx(r + 1, edge_c), top if r < mid else bottom))
        tris.append((idx(mid, edge_c), bottom, top))
    tris = np.array(tris)

    # Match the winding of the other faces: positive area in (column, row) layout.
    r, c = np.divmod(tris, cols)
    area = (c[:, 1] - c[:, 0]) * (r[:, 2] - r[:, 0]) - (r[:, 1] - r[:, 0]) * (c[:, 2] - c[:, 0])
    tris[area < 0] = tris[area < 0][:, ::-1]
    return tris


def _build_shell(varying_surface, baseline_surface, walls=True):
    """Stitch two (rows, cols, 3) vertex grids into a closed, watertight shell.

    With walls=False the edge is left open for a frame to close it.
    """
    rows, cols, _ = varying_surface.shape
    vertices = np.concatenate([varying_surface.reshape(-1, 3), baseline_surface.reshape(-1, 3)])
    offset = rows * cols
    grid = np.arange(rows * cols).reshape(rows, cols)
    a = grid[:-1, :-1].ravel()
    b = grid[:-1, 1:].ravel()
    c = grid[1:, :-1].ravel()
    d = grid[1:, 1:].ravel()

    varying_faces = np.concatenate([np.stack([a, c, b], axis=1), np.stack([b, c, d], axis=1)])
    baseline_faces = _coarse_back_faces(rows, cols) + offset

    if not walls:
        return vertices, np.concatenate([varying_faces, baseline_faces])

    def wall(ia, ib):
        ja, jb = ia + offset, ib + offset
        t1 = np.stack([ia, ib, ja], axis=1)
        t2 = np.stack([ib, jb, ja], axis=1)
        return np.concatenate([t1, t2])

    back, front = grid[0, :], grid[-1, :]
    left, right = grid[:, 0], grid[:, -1]
    wall_faces = np.concatenate(
        [
            wall(back[:-1], back[1:]),
            wall(front[1:], front[:-1]),
            wall(left[1:], left[:-1]),
            wall(right[:-1], right[1:]),
        ]
    )
    return vertices, np.concatenate([varying_faces, baseline_faces, wall_faces])


def _grid_boundary_loop(grid):
    """Trace the perimeter of a (rows, cols, ...) grid as an ordered, closed loop."""
    top = grid[0, :-1]
    right = grid[:-1, -1]
    bottom = grid[-1, :0:-1]
    left = grid[::-1, 0][:-1]
    return np.concatenate([top, right, bottom, left])


def _miter_corners(U, rows, cols):
    """Corner vertices get the sum of both edge directions, giving square frame corners."""
    n = U.shape[0]
    corners = [0, cols - 1, cols - 1 + rows - 1, 2 * (cols - 1) + rows - 1]
    U = U.copy()
    for i in corners:
        U[i] = U[i - 1] + U[(i + 1) % n]
    return U


def _outward_directions(rows, cols):
    """In-plane outward direction for each vertex of a grid's boundary loop.

    Works in the panel's flat (or unrolled) layout: x grows with column, y
    shrinks with row.
    """
    n_top, n_right, n_bottom = cols - 1, rows - 1, cols - 1
    U = np.empty((2 * (rows - 1) + 2 * (cols - 1), 3))
    U[:n_top] = (0, 1, 0)
    U[n_top : n_top + n_right] = (1, 0, 0)
    U[n_top + n_right : n_top + n_right + n_bottom] = (0, -1, 0)
    U[n_top + n_right + n_bottom :] = (-1, 0, 0)
    return _miter_corners(U, rows, cols)


def _build_frame(
    relief_loop, baseline_loop, layout_loop, U, wrap, max_thickness_mm, thickness_mm, depth_mm, angle_deg,
    tab_mm=0.0,
):
    """Beveled frame on the picture side that closes the panel's open edge.

    The frame is laid out flat (layout_loop is the boundary in the panel's
    flat/unrolled layout, picture side = +z) and wrap maps it onto the real
    surface, so on an arc the frame follows the curve. The frame stands
    depth_mm proud of the flat back (never less than the max thickness) and
    its back stays flush with it. Cross-section: inner bevel sloping down to
    the picture, front rim, then square outer wall and flat back so the outer
    edges (including the one it stands on) are flat. tab_mm extends the bottom
    band downwards so it can slot into a stand.
    """
    height = max(depth_mm, max_thickness_mm)
    flare = height * np.tan(np.radians(angle_deg))
    up = np.array([0.0, 0.0, height])
    # Bottom-edge points (including the two bottom corners) have a downward outward direction.
    outer = U * (flare + thickness_mm) + np.minimum(U * [0, 1, 0], 0) * tab_mm
    rings = [
        relief_loop,
        wrap(layout_loop + up + U * flare),
        wrap(layout_loop + up + outer),
        wrap(layout_loop + outer),
        baseline_loop,
    ]
    picture_side = wrap(layout_loop + up + np.array([0.0, 0.0, 1.0])) - rings[1]

    n = relief_loop.shape[0]
    ia = np.arange(n)
    ib = np.roll(ia, -1)
    vertices = np.concatenate(rings)
    faces = np.concatenate(
        [
            np.concatenate(
                [np.stack([ia, ib, ia + n], axis=1), np.stack([ib, ib + n, ia + n], axis=1)]
            )
            + k * n
            for k in range(len(rings) - 1)
        ]
    )

    # All strips share one sweep direction, so orient them together using the
    # front rim, which always has non-zero area.
    rim = vertices[faces[2 * n : 4 * n]]
    normals = np.cross(rim[:, 1] - rim[:, 0], rim[:, 2] - rim[:, 0])
    if np.sum(normals * np.concatenate([picture_side, picture_side])) < 0:
        faces = faces[:, ::-1]
    return vertices, faces


def _save(vertices, faces, output_path):
    out = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    out.vectors = vertices[faces]
    out.save(output_path)


def generate_lithophane_flat(
    image_path: str,
    output_path: str | None,
    width_mm: float = 100.0,
    max_thickness_mm: float = 3.0,
    min_thickness_mm: float = 0.6,
    mm_per_pixel: float = 0.1,
    invert: bool = False,
    frame_enabled: bool = False,
    frame_thickness_mm: float = 3.0,
    frame_depth_mm: float = 4.0,
    frame_angle_deg: float = 45.0,
    stand_tab_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat panel. width_mm is the whole piece, frame included."""
    img = Image.open(image_path)
    extent = frame_extent_mm(
        frame_enabled, max_thickness_mm, frame_thickness_mm, frame_depth_mm, frame_angle_deg
    )
    rows, cols, _ = panel_dimensions(img.size, width_mm, mm_per_pixel, extent)
    thickness = _load_thickness_map(img, cols, rows, max_thickness_mm, min_thickness_mm, invert)
    xs = np.arange(cols) * mm_per_pixel
    ys = (rows - 1 - np.arange(rows)) * mm_per_pixel
    xx, yy = np.meshgrid(xs, ys)

    top = np.stack([xx, yy, thickness], axis=-1)
    bottom = np.stack([xx, yy, np.zeros_like(thickness)], axis=-1)
    vertices, faces = _build_shell(top, bottom, walls=not frame_enabled)

    if frame_enabled:
        bottom_loop = _grid_boundary_loop(bottom)
        frame_vertices, frame_faces = _build_frame(
            _grid_boundary_loop(top), bottom_loop, bottom_loop, _outward_directions(rows, cols),
            lambda p: p, max_thickness_mm, frame_thickness_mm, frame_depth_mm, frame_angle_deg,
            stand_tab_mm,
        )
        offset = vertices.shape[0]
        vertices = np.concatenate([vertices, frame_vertices])
        faces = np.concatenate([faces, frame_faces + offset])

    if output_path:
        _save(vertices, faces, output_path)
    return vertices, faces


def generate_lithophane_arc(
    image_path: str,
    output_path: str | None,
    arc_angle_deg: float = 60.0,
    height_mm: float = 100.0,
    max_thickness_mm: float = 3.0,
    min_thickness_mm: float = 0.6,
    mm_per_pixel: float = 0.1,
    invert: bool = False,
    relief_inward: bool = False,
    frame_enabled: bool = False,
    frame_thickness_mm: float = 3.0,
    frame_depth_mm: float = 4.0,
    frame_angle_deg: float = 45.0,
    stand_tab_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lithophane as a segment of a cylindrical shell (e.g. a lamp-shade panel).

    height_mm and arc_angle_deg are for the whole piece, frame included; the
    width (measured around the curve) follows from the photo's aspect ratio and
    the radius from width and angle. By default the picture and frame sit on the
    outside, viewed from outside; relief_inward flips them to the inside. The
    other side is smooth.
    """
    img = Image.open(image_path)
    extent = frame_extent_mm(
        frame_enabled, max_thickness_mm, frame_thickness_mm, frame_depth_mm, frame_angle_deg
    )
    rows, cols, width_mm = panel_dimensions_from_height(img.size, height_mm, mm_per_pixel, extent)
    radius_mm = arc_radius_mm(width_mm, arc_angle_deg)
    thickness = _load_thickness_map(img, cols, rows, max_thickness_mm, min_thickness_mm, invert)
    if not relief_inward:
        # Columns run anticlockwise, which reads correctly from inside but mirrored from outside.
        thickness = thickness[:, ::-1]
    theta = (np.arange(cols) - (cols - 1) / 2.0) * (mm_per_pixel / radius_mm)
    z = (rows - 1 - np.arange(rows)) * mm_per_pixel
    theta_grid, z_grid = np.meshgrid(theta, z)

    relief_sign = 1.0 if relief_inward else -1.0
    varying_radius = radius_mm - relief_sign * thickness
    baseline = np.stack(
        [radius_mm * np.sin(theta_grid), radius_mm * np.cos(theta_grid), z_grid], axis=-1
    )
    varying = np.stack(
        [varying_radius * np.sin(theta_grid), varying_radius * np.cos(theta_grid), z_grid], axis=-1
    )
    vertices, faces = _build_shell(varying, baseline, walls=not frame_enabled)
    if not relief_inward:
        # Relief on the outside mirrors the shell, which would leave it inside-out.
        faces = faces[:, ::-1]

    if frame_enabled:
        # Unrolled layout: x = arc length, y = height, z = distance towards the picture side.
        unrolled = np.stack(
            [theta_grid * radius_mm, z_grid, np.zeros_like(z_grid)], axis=-1
        )

        def wrap(p):
            r = radius_mm - relief_sign * p[:, 2]
            th = p[:, 0] / radius_mm
            return np.stack([r * np.sin(th), r * np.cos(th), p[:, 1]], axis=1)

        frame_vertices, frame_faces = _build_frame(
            _grid_boundary_loop(varying), _grid_boundary_loop(baseline),
            _grid_boundary_loop(unrolled), _outward_directions(rows, cols), wrap,
            max_thickness_mm, frame_thickness_mm, frame_depth_mm, frame_angle_deg,
            stand_tab_mm,
        )
        offset = vertices.shape[0]
        vertices = np.concatenate([vertices, frame_vertices])
        faces = np.concatenate([faces, frame_faces + offset])

    if output_path:
        _save(vertices, faces, output_path)
    return vertices, faces


HOOK_WALL_MM = 2.5


def _zip_loops(a_idx, a_ang, b_idx, b_ang):
    """Triangulate the band between two closed loops, each ordered anticlockwise by angle.

    The loops may have different numbers of points; with equal counts at the
    same angles this is an ordinary quad strip.
    """

    def prep(idx, ang):
        ang = np.mod(ang, 2 * np.pi)
        start = int(np.argmin(ang))
        idx, ang = np.roll(idx, -start), np.unwrap(np.roll(ang, -start))
        return np.append(idx, idx[0]), np.append(ang, ang[0] + 2 * np.pi)

    ai, aa = prep(np.asarray(a_idx), np.asarray(a_ang))
    bi, ba = prep(np.asarray(b_idx), np.asarray(b_ang))
    m, n = len(ai) - 1, len(bi) - 1
    i = j = 0
    tris = []
    while i < m or j < n:
        if j == n or (i < m and aa[i + 1] <= ba[j + 1]):
            tris.append((ai[i], ai[i + 1], bi[j]))
            i += 1
        else:
            tris.append((ai[i], bi[j + 1], bi[j]))
            j += 1
    return np.array(tris)


class _MeshBuilder:
    def __init__(self):
        self.vertices, self.faces, self.count = [], [], 0

    def add_vertices(self, pts):
        base = self.count
        self.vertices.append(np.asarray(pts, dtype=np.float64))
        self.count += len(pts)
        return base + np.arange(len(pts))

    def add_faces(self, faces, outward):
        """Add a connected, consistently wound piece, flipped as a whole to face `outward`.

        outward(centroids) -> (n, 3) directions the piece should face.
        """
        verts = np.concatenate(self.vertices)
        tri = verts[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if np.sum(normals * outward(tri.mean(axis=1))) < 0:
            faces = faces[:, ::-1]
        self.faces.append(faces)

    def result(self):
        return np.concatenate(self.vertices), np.concatenate(self.faces)


def _radial(centre):
    def outward(p):
        d = p[:, :2] - centre
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return np.column_stack([d, np.zeros(len(p))])

    return outward


def _up(p):
    return np.tile((0.0, 0.0, 1.0), (len(p), 1))


def _down(p):
    return -_up(p)


def _circle_ring(builder, centre, radius, z, angles):
    pts = np.column_stack(
        [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles), np.full(len(angles), z)]
    )
    return builder.add_vertices(pts)


def _picture_disc(builder, thickness, inside, xs, ys):
    """Height-field top surface over the grid points inside the circle.

    Returns the anticlockwise boundary loop (vertex indices) of the picture.
    """
    rows, cols = inside.shape
    grid = np.full((rows, cols), -1)
    xx, yy = np.meshgrid(xs, ys)
    grid[inside] = builder.add_vertices(
        np.column_stack([xx[inside], yy[inside], thickness[inside]])
    )

    valid = np.flatnonzero(inside.any(axis=1))
    first = inside.argmax(axis=1)
    last = cols - 1 - inside[:, ::-1].argmax(axis=1)

    tris = []
    for r in valid[:-1]:
        a1, b1, a2, b2 = first[r], last[r], first[r + 1], last[r + 1]
        lo, hi = max(a1, a2), min(b1, b2)
        c = np.arange(lo, hi)
        p, q, s, t = grid[r, c], grid[r, c + 1], grid[r + 1, c], grid[r + 1, c + 1]
        tris += [np.column_stack([p, s, q]), np.column_stack([q, s, t])]
        for row, other, start, end, apex_col in (
            (r, r + 1, a1, lo, lo),
            (r + 1, r, a2, lo, lo),
            (r, r + 1, hi, b1, hi),
            (r + 1, r, hi, b2, hi),
        ):
            c = np.arange(start, end)
            if len(c):
                tris.append(
                    np.column_stack([grid[row, c], grid[row, c + 1], np.full(len(c), grid[other, apex_col])])
                )
    faces = np.concatenate(tris)

    # A height field projects one-to-one onto the plane, so each triangle can
    # simply be made anticlockwise seen from above (normal +z).
    verts = np.concatenate(builder.vertices)
    tri = verts[faces]
    area = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (
        tri[:, 1, 1] - tri[:, 0, 1]
    ) * (tri[:, 2, 0] - tri[:, 0, 0])
    faces[area < 0] = faces[area < 0][:, ::-1]
    builder.faces.append(faces)

    top, bottom = valid[0], valid[-1]
    loop = np.concatenate(
        [
            grid[bottom, first[bottom] : last[bottom] + 1],
            grid[valid[-2:0:-1], last[valid[-2:0:-1]]],
            grid[top, first[top] : last[top] + 1][::-1],
            grid[valid[1:-1], first[valid[1:-1]]],
        ]
    )
    return loop


def _add_hook(builder, outer_radius_mm, band_mm, height_mm, position_pct, hole_diameter_mm, offset_mm):
    """A flat eyelet tab with a hole, overlapping the edge of the disc.

    It is its own closed solid; slicers merge overlapping solids in one file.
    """
    hole_r = hole_diameter_mm / 2
    tab_r = hole_r + HOOK_WALL_MM
    direction = np.radians(90.0 - 3.6 * position_pct)
    u = np.array([np.cos(direction), np.sin(direction)])
    centre = (outer_radius_mm + hole_r + 1.0 + offset_mm) * u

    # Outline: convex hull of the tab circle and a small circle inside the frame band,
    # so the tab always reaches into the disc. Sampled by angle around the hole.
    inner_r = min(tab_r, band_mm / 2)
    inner_offset = np.linalg.norm(centre) - (outer_radius_mm - band_mm / 2)
    k = max(96, int(2 * np.pi * tab_r / 0.25))
    psi = np.linspace(0, 2 * np.pi, k, endpoint=False)
    normals = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    nx, ny = np.cos(normals), np.sin(normals)
    # local frame: x points from the hole towards the disc centre
    support = np.maximum(tab_r, inner_offset * nx + inner_r)
    dots = np.cos(psi)[:, None] * nx[None, :] + np.sin(psi)[:, None] * ny[None, :]
    with np.errstate(divide="ignore"):
        dist = np.where(dots > 1e-6, support[None, :] / dots, np.inf)
    outline_r = dist.min(axis=1)

    towards_centre = -u
    side = np.array([-towards_centre[1], towards_centre[0]])
    world_angles = np.arctan2(
        np.cos(psi) * towards_centre[1] + np.sin(psi) * side[1],
        np.cos(psi) * towards_centre[0] + np.sin(psi) * side[0],
    )
    order = np.argsort(np.mod(world_angles, 2 * np.pi))
    world_angles, outline_r = world_angles[order], outline_r[order]

    def ring(radii, z):
        pts = np.column_stack(
            [
                centre[0] + radii * np.cos(world_angles),
                centre[1] + radii * np.sin(world_angles),
                np.full(k, z),
            ]
        )
        return builder.add_vertices(pts)

    hole = np.full(k, hole_r)
    outer_top, hole_top = ring(outline_r, height_mm), ring(hole, height_mm)
    hole_bottom, outer_bottom = ring(hole, 0.0), ring(outline_r, 0.0)
    radial = _radial(centre)
    builder.add_faces(_zip_loops(outer_top, world_angles, hole_top, world_angles), _up)
    builder.add_faces(
        _zip_loops(hole_top, world_angles, hole_bottom, world_angles), lambda p: -radial(p)
    )
    builder.add_faces(_zip_loops(hole_bottom, world_angles, outer_bottom, world_angles), _down)
    builder.add_faces(_zip_loops(outer_bottom, world_angles, outer_top, world_angles), radial)


def circle_picture_diameter_mm(diameter_mm, frame_extent):
    return diameter_mm - 2 * frame_extent


def generate_lithophane_circle(
    image_path: str,
    output_path: str | None,
    diameter_mm: float = 100.0,
    max_thickness_mm: float = 3.0,
    min_thickness_mm: float = 0.6,
    mm_per_pixel: float = 0.1,
    invert: bool = False,
    frame_enabled: bool = False,
    frame_thickness_mm: float = 3.0,
    frame_depth_mm: float = 4.0,
    frame_angle_deg: float = 45.0,
    hook_enabled: bool = False,
    hook_position_pct: float = 0.0,
    hook_hole_diameter_mm: float = 3.5,
    hook_offset_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Round flat lithophane; the photo is centre-cropped to a circle.

    diameter_mm is the whole disc, frame included (the hook sticks out beyond it).
    """
    extent = frame_extent_mm(
        frame_enabled, max_thickness_mm, frame_thickness_mm, frame_depth_mm, frame_angle_deg
    )
    picture_r = circle_picture_diameter_mm(diameter_mm, extent) / 2
    if picture_r < 4 * mm_per_pixel:
        raise ValueError("Diameter is too small to fit the frame.")
    outer_r = diameter_mm / 2

    img = Image.open(image_path)
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    n = max(5, 2 * round(picture_r / mm_per_pixel) + 1)
    thickness = _load_thickness_map(img, n, n, max_thickness_mm, min_thickness_mm, invert)
    xs = (np.arange(n) - (n - 1) / 2) * mm_per_pixel
    ys = xs[::-1]
    xx, yy = np.meshgrid(xs, ys)
    inside = xx**2 + yy**2 <= picture_r**2 + 1e-9

    b = _MeshBuilder()
    loop = _picture_disc(b, thickness, inside, xs, ys)
    verts = np.concatenate(b.vertices)
    loop_angles = np.arctan2(verts[loop, 1], verts[loop, 0])

    origin = np.zeros(2)
    radial = _radial(origin)
    ring_count = max(96, round(2 * np.pi * outer_r / (2 * mm_per_pixel)))
    angles = np.linspace(0, 2 * np.pi, ring_count, endpoint=False)

    if frame_enabled:
        height = max(frame_depth_mm, max_thickness_mm)
        flare = height * np.tan(np.radians(frame_angle_deg))
        inner_top = _circle_ring(b, origin, picture_r + flare, height, angles)
        outer_top = _circle_ring(b, origin, outer_r, height, angles)
        outer_bottom = _circle_ring(b, origin, outer_r, 0.0, angles)
        b.add_faces(_zip_loops(loop, loop_angles, inner_top, angles), lambda p: _up(p) - radial(p))
        b.add_faces(_zip_loops(inner_top, angles, outer_top, angles), _up)
        b.add_faces(_zip_loops(outer_top, angles, outer_bottom, angles), radial)
        rim, rim_angles = outer_bottom, angles
    else:
        height = max_thickness_mm
        base = verts[loop].copy()
        base[:, 2] = 0.0
        rim = b.add_vertices(base)
        rim_angles = loop_angles
        b.add_faces(_zip_loops(loop, loop_angles, rim, rim_angles), radial)

    centre_idx = b.add_vertices([[0.0, 0.0, 0.0]])[0]
    order = np.argsort(np.mod(rim_angles, 2 * np.pi))
    ring = np.asarray(rim)[order]
    b.add_faces(np.column_stack([np.full(len(ring), centre_idx), np.roll(ring, -1), ring]), _down)

    if hook_enabled:
        band = frame_thickness_mm if frame_enabled else 2.0
        _add_hook(
            b, outer_r, band, height, hook_position_pct, hook_hole_diameter_mm, hook_offset_mm
        )

    vertices, faces = b.result()
    if output_path:
        _save(vertices, faces, output_path)
    return vertices, faces


def standing_lithophane(vertices, curved):
    """Litho vertices as it stands in its stand: upright, picture facing +y, tab bottom at z=0.

    Arcs are built upright already; flat panels lie picture-up, so they are
    stood on their bottom edge (a rotation, so face winding stays valid).
    """
    if curved:
        v = vertices.copy()
    else:
        v = np.column_stack([-vertices[:, 0], vertices[:, 2], vertices[:, 1]])
    v[:, 2] -= v[:, 2].min()
    return v


def _rect_loop(a_side, b_side, between):
    """Anticlockwise (u, v) loop round a four-sided region.

    a_side = (u_start, u_end, v) is the low-v side, b_side the high-v side;
    both sides also pass through the u values in between. Loops sharing the
    same between have matching points, so any two can be joined by a strip of
    quads that don't twist.
    """
    (a0, a1, av), (b0, b1, bv) = a_side, b_side
    ua = np.concatenate([[a0], between, [a1]])
    ub = np.concatenate([[b0], between, [b1]])
    side_a = np.column_stack([ua, np.full(len(ua), av)])
    side_b = np.column_stack([ub, np.full(len(ub), bv)])[::-1]
    return np.concatenate([side_a, side_b])


def _slotted_tray(loops, n, mapping):
    """Solid swept through (u, v) loops at given heights, capped at both ends.

    loops run outer bottom -> up the outside -> across the top -> down the
    slot to its floor; each is ((u, v) points, z). Built in (u, v, z) space and
    mapped, so the mapping must keep handedness.
    """
    m = 2 * n + 2
    param = np.concatenate([np.column_stack([uv, np.full(len(uv), z)]) for uv, z in loops])
    ia = np.arange(m)
    ib = np.roll(ia, -1)
    sweep = np.concatenate([
        np.concatenate([np.stack([ia, ib, ib + m], 1), np.stack([ia, ib + m, ia + m], 1)]) + k * m
        for k in range(len(loops) - 1)
    ])

    def cap(offset):
        # Quads straight across from the low-v side (0..n) to the high-v side (2n+1..n+1).
        a = np.arange(n) + offset
        b = 2 * n + 1 - np.arange(n) + offset
        return np.concatenate([np.stack([a, a + 1, b - 1], 1), np.stack([a, b - 1, b], 1)])

    def orient(faces, outward):
        tri = param[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return faces if np.sum(normals * outward(tri.mean(axis=1))) >= 0 else faces[:, ::-1]

    centre = param[:m, :2].mean(axis=0)
    first = sweep[: 2 * m]  # the outer wall, facing away from the middle
    flip = orient(first, lambda p: np.column_stack([p[:, :2] - centre, np.zeros(len(p))]))
    if flip is not first:
        sweep = sweep[:, ::-1]
    faces = np.concatenate([
        sweep,
        orient(cap(0), lambda p: np.tile((0, 0, -1), (len(p), 1))),
        orient(cap((len(loops) - 1) * m), lambda p: np.tile((0, 0, 1), (len(p), 1))),
    ])
    return mapping(param), faces


def generate_stand(
    standing_vertices,
    curved: bool,
    output_path: str | None,
    slot_depth_mm: float = 8.0,
    clearance_mm: float = 0.3,
    wall_mm: float = 3.0,
    stand_depth_mm: float = 30.0,
    chamfer_mm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Block with a slot the litho's bottom tab slips into, sized from the litho itself.

    standing_vertices comes from standing_lithophane. The returned mesh is in
    the same frame (slot floor at z=0, so it shows assembled); the saved STL
    is moved up to sit on the bed. For an arc the stand curves to match. The
    outer top edge gets a 45 degree chamfer of chamfer_mm.
    """
    c = clearance_mm
    if curved:
        th = np.arctan2(standing_vertices[:, 0], standing_vertices[:, 1])
        r = np.hypot(standing_vertices[:, 0], standing_vertices[:, 1])
        r0, r1 = r.min(), r.max()
        half_depth = max(stand_depth_mm / 2, (r1 - r0) / 2 + c + wall_mm)
        mid = (r0 + r1) / 2
        if mid - half_depth <= 1.0:
            raise ValueError("The stand is too deep for this arc's radius.")
        # Measure clearance and end walls at the innermost radius, so they're never thinner.
        slot_u = (th.min() - c / r0, th.max() + c / r0)
        outer_u = (
            slot_u[0] - wall_mm / (mid - half_depth), slot_u[1] + wall_mm / (mid - half_depth)
        )
        if outer_u[1] - outer_u[0] >= 2 * np.pi:
            raise ValueError("The arc is too wide to fit a stand; reduce the angle.")
        # Points every 1 mm or so round the curve, inside the slot so every loop can use them.
        steps = max(1, int(np.ceil((slot_u[1] - slot_u[0]) * (mid + half_depth) / 1.0)))
        between = np.linspace(*slot_u, steps + 1)[1:-1]

        def inset(r):
            # Angle that moves a point at radius r chamfer_mm square-on away from an end face.
            return np.arcsin(min(chamfer_mm / r, 1.0))

        def mapping(p):
            return np.column_stack(
                [p[:, 1] * np.sin(p[:, 0]), p[:, 1] * np.cos(p[:, 0]), p[:, 2]]
            )
    else:
        x0, x1 = standing_vertices[:, 0].min(), standing_vertices[:, 0].max()
        y0, y1 = standing_vertices[:, 1].min(), standing_vertices[:, 1].max()
        r0, r1 = y0, y1
        half_depth = max(stand_depth_mm / 2, (y1 - y0) / 2 + c + wall_mm)
        mid = (y0 + y1) / 2
        slot_u = (x0 - c, x1 + c)
        outer_u = (slot_u[0] - wall_mm, slot_u[1] + wall_mm)
        between = np.array([])

        def inset(r):
            return chamfer_mm

        def mapping(p):
            return p

    v_in, v_out = mid - half_depth, mid + half_depth
    slot_v = (r0 - c, r1 + c)
    thinnest_wall = min(wall_mm, slot_v[0] - v_in, v_out - slot_v[1])
    if chamfer_mm >= min(thinnest_wall, slot_depth_mm + wall_mm):
        raise ValueError("The chamfer is too big for the stand's walls.")

    top, bottom = slot_depth_mm, -wall_mm
    n = len(between) + 1
    outer = _rect_loop((*outer_u, v_in), (*outer_u, v_out), between)
    loops = [(outer, bottom)]
    if chamfer_mm > 0:
        lo, hi = v_in + chamfer_mm, v_out - chamfer_mm
        loops += [
            (outer, top - chamfer_mm),
            (_rect_loop(
                (outer_u[0] + inset(lo), outer_u[1] - inset(lo), lo),
                (outer_u[0] + inset(hi), outer_u[1] - inset(hi), hi), between,
            ), top),
        ]
    else:
        loops.append((outer, top))
    slot = _rect_loop((*slot_u, slot_v[0]), (*slot_u, slot_v[1]), between)
    loops += [(slot, top), (slot, 0.0)]

    vertices, faces = _slotted_tray(loops, n, mapping)
    if output_path:
        _save(vertices + [0.0, 0.0, wall_mm], faces, output_path)
    return vertices, faces
