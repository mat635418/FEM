"""Mesh generators for common FEM geometries.

All functions return:
    verts : numpy array, shape (N, 3)
    faces : numpy array of triangle indices, shape (M, 3)
"""

import struct
import numpy as np


# ---------------------------------------------------------------------------
# Rectangular Beam (3-D surface mesh)
# ---------------------------------------------------------------------------

def make_beam_mesh(L: float, w: float, h: float, n: int):
    """Surface mesh of a rectangular beam along the X-axis.

    Parameters
    ----------
    L : float  — length along X
    w : float  — width along Y
    h : float  — height along Z
    n : int    — number of longitudinal divisions
    """
    n = max(n, 1)
    x_vals = np.linspace(0, L, n + 1)

    verts = []
    # Build the 4-corner cross-sections
    for x in x_vals:
        verts.append([x,  0,  0])
        verts.append([x,  w,  0])
        verts.append([x,  w,  h])
        verts.append([x,  0,  h])
    verts = np.array(verts, dtype=float)  # shape: (4*(n+1), 3)

    faces = []
    for i in range(n):
        base = 4 * i
        # 4 lateral faces (each quad → 2 triangles)
        quad_pairs = [
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        for a, b, c, d in quad_pairs:
            faces.append([base + a, base + b, base + c])
            faces.append([base + a, base + c, base + d])

    # End caps
    # Front cap (x=0)
    faces.append([0, 1, 2])
    faces.append([0, 2, 3])
    # Back cap (x=L)
    base = 4 * n
    faces.append([base + 0, base + 2, base + 1])
    faces.append([base + 0, base + 3, base + 2])

    faces = np.array(faces, dtype=int)
    return verts, faces


# ---------------------------------------------------------------------------
# Plate (2-D mesh in XY-plane, flat Z=0)
# ---------------------------------------------------------------------------

def make_plate_mesh(w: float, h: float, nx: int, ny: int):
    """2-D mesh of a rectangular plate (Z = 0).

    Also suitable for use with the 2-D FEM solver (nodes_2d = verts[:, :2]).
    """
    nx = max(nx, 1)
    ny = max(ny, 1)
    xs = np.linspace(0, w, nx + 1)
    ys = np.linspace(0, h, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = np.zeros_like(x_flat)
    verts = np.column_stack([x_flat, y_flat, z_flat])

    faces = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            faces.append([n0, n1, n2])
            faces.append([n0, n2, n3])

    faces = np.array(faces, dtype=int)
    return verts, faces


# ---------------------------------------------------------------------------
# Cylinder (surface mesh)
# ---------------------------------------------------------------------------

def make_cylinder_mesh(r: float, L: float, nc: int, nl: int):
    """Surface mesh of a cylinder aligned with the X-axis.

    Parameters
    ----------
    r  : radius
    L  : length
    nc : circumferential divisions
    nl : longitudinal divisions
    """
    nc = max(nc, 3)
    nl = max(nl, 1)
    thetas = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    xs = np.linspace(0, L, nl + 1)

    verts = []
    for x in xs:
        for t in thetas:
            verts.append([x, r * np.cos(t), r * np.sin(t)])
    # Cap centres
    front_centre = len(verts)
    verts.append([0, 0, 0])
    back_centre = len(verts)
    verts.append([L, 0, 0])
    verts = np.array(verts, dtype=float)

    faces = []
    # Lateral surface
    for il in range(nl):
        for ic in range(nc):
            n0 = il * nc + ic
            n1 = il * nc + (ic + 1) % nc
            n2 = (il + 1) * nc + (ic + 1) % nc
            n3 = (il + 1) * nc + ic
            faces.append([n0, n1, n2])
            faces.append([n0, n2, n3])

    # End caps
    for ic in range(nc):
        n0 = ic
        n1 = (ic + 1) % nc
        faces.append([front_centre, n1, n0])  # reversed normal → outward

        n0b = nl * nc + ic
        n1b = nl * nc + (ic + 1) % nc
        faces.append([back_centre, n0b, n1b])

    faces = np.array(faces, dtype=int)
    return verts, faces


# ---------------------------------------------------------------------------
# Box / Cuboid (surface mesh)
# ---------------------------------------------------------------------------

def make_box_mesh(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int):
    """Surface mesh of a box (cuboid) with per-axis divisions."""
    nx = max(nx, 1)
    ny = max(ny, 1)
    nz = max(nz, 1)

    all_verts = []
    all_faces = []

    def _add_quad_face(p00, p10, p11, p01, nv, nu):
        """Add a quad patch subdivided into nv×nu quads."""
        base = len(all_verts)
        for j in range(nv + 1):
            for i in range(nu + 1):
                u = i / nu
                v = j / nv
                pt = (
                    (1 - u) * (1 - v) * p00
                    + u * (1 - v) * p10
                    + u * v * p11
                    + (1 - u) * v * p01
                )
                all_verts.append(pt)
        for j in range(nv):
            for i in range(nu):
                n0 = base + j * (nu + 1) + i
                n1 = n0 + 1
                n2 = n0 + (nu + 1) + 1
                n3 = n0 + (nu + 1)
                all_faces.append([n0, n1, n2])
                all_faces.append([n0, n2, n3])

    corners = [
        np.array([0,  0,  0]),
        np.array([lx, 0,  0]),
        np.array([lx, ly, 0]),
        np.array([0,  ly, 0]),
        np.array([0,  0,  lz]),
        np.array([lx, 0,  lz]),
        np.array([lx, ly, lz]),
        np.array([0,  ly, lz]),
    ]

    _add_quad_face(corners[0], corners[1], corners[2], corners[3], ny, nx)  # bottom
    _add_quad_face(corners[4], corners[5], corners[6], corners[7], ny, nx)  # top
    _add_quad_face(corners[0], corners[1], corners[5], corners[4], nz, nx)  # front
    _add_quad_face(corners[3], corners[2], corners[6], corners[7], nz, nx)  # back
    _add_quad_face(corners[0], corners[3], corners[7], corners[4], nz, ny)  # left
    _add_quad_face(corners[1], corners[2], corners[6], corners[5], nz, ny)  # right

    verts = np.array(all_verts, dtype=float)
    faces = np.array(all_faces, dtype=int)
    return verts, faces


# ---------------------------------------------------------------------------
# L-Bracket (surface mesh)
# ---------------------------------------------------------------------------

def make_lbracket_mesh(l1: float, l2: float, t: float, n: int):
    """Surface mesh of an L-bracket.

    The bracket lies in the XY-plane with thickness *t* in Z.
    l1 : length of the horizontal leg
    l2 : length of the vertical leg
    t  : thickness (both legs share the same thickness)
    n  : divisions per unit length (approx)
    """
    n = max(n, 2)
    total = l1 + l2
    n_horizontal = max(1, int(n * l1 / total))
    n_vertical   = max(1, int(n * l2 / total))
    # Horizontal leg: x in [0, l1], y in [0, t]
    # Vertical leg  : x in [0, t],  y in [t, t+l2]
    h_verts, h_faces = make_plate_mesh(l1, t, n_horizontal, n)
    # Shift h_verts' z to [0, t] (extrude in Z)
    h_verts3d_top = h_verts.copy()
    h_verts3d_top[:, 2] = t

    v_verts, v_faces = make_plate_mesh(t, l2, n, n_vertical)
    v_verts[:, 1] += t  # shift Y by t to sit above horizontal leg

    # Combine both legs (top face only for simplicity, closed by box caps)
    all_verts = np.vstack([h_verts3d_top, v_verts])
    offset = len(h_verts3d_top)
    combined_faces = np.vstack([h_faces, v_faces + offset])
    return all_verts, combined_faces


# ---------------------------------------------------------------------------
# STL loader
# ---------------------------------------------------------------------------

def load_stl_mesh(file_bytes: bytes):
    """Parse an STL file (binary or ASCII) from raw bytes.

    Returns (verts, faces) as numpy arrays.
    """
    try:
        return _parse_binary_stl(file_bytes)
    except Exception:
        pass
    try:
        return _parse_ascii_stl(file_bytes)
    except Exception:
        pass
    # Return empty mesh on failure
    return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)


def _parse_binary_stl(data: bytes):
    if len(data) < 84:
        raise ValueError("Too short for binary STL")
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + n_tri * 50
    if len(data) < expected_size:
        raise ValueError("Binary STL size mismatch")

    verts = []
    faces = []
    offset = 84
    for i in range(n_tri):
        offset += 12  # skip normal
        v0 = struct.unpack_from("<3f", data, offset); offset += 12
        v1 = struct.unpack_from("<3f", data, offset); offset += 12
        v2 = struct.unpack_from("<3f", data, offset); offset += 12
        offset += 2   # attribute byte count
        base = len(verts)
        verts.extend([list(v0), list(v1), list(v2)])
        faces.append([base, base + 1, base + 2])

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def _parse_ascii_stl(data: bytes):
    text = data.decode("utf-8", errors="replace")
    verts = []
    faces = []
    tri_verts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            tri_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("endloop"):
            if len(tri_verts) == 3:
                base = len(verts)
                verts.extend(tri_verts)
                faces.append([base, base + 1, base + 2])
            tri_verts = []
    return np.array(verts, dtype=float), np.array(faces, dtype=int)
