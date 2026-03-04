"""2-D triangular mesh generation for FEM Lab geometries."""

from __future__ import annotations

import numpy as np
import skfem


def make_cantilever_mesh(
    L: float,
    H: float,
    nx: int = 20,
    ny: int = 6,
) -> skfem.MeshTri:
    """Axis-aligned rectangular mesh for a cantilever beam.

    The beam occupies [0, L] × [0, H].  Boundary tags produced:
    ``left``, ``right``, ``top``, ``bottom``.

    Parameters
    ----------
    L:
        Beam length (m).
    H:
        Beam height (m).
    nx:
        Number of elements along the x-axis.
    ny:
        Number of elements along the y-axis.
    """
    mesh = skfem.MeshTri.init_tensor(
        np.linspace(0.0, L, nx + 1),
        np.linspace(0.0, H, ny + 1),
    )
    mesh = mesh.with_defaults()
    return mesh


def make_l_bracket_mesh(refine: int = 2) -> skfem.MeshTri:
    """True L-shaped cross-section mesh with re-entrant corner.

    The L-shape occupies the union of:
    * lower rectangle: [0, 2] × [0, 1]
    * left rectangle:  [0, 1] × [1, 2]

    Boundary tags:
    ``bottom`` (y=0), ``right`` (x=2, y∈[0,1]), ``inner_top`` (y=1, x∈[1,2]),
    ``inner_right`` (x=1, y∈[1,2]), ``top`` (y=2), ``left`` (x=0).

    Parameters
    ----------
    refine:
        Number of uniform refinement steps.
    """
    # Build from a tensor mesh and remove the upper-right quadrant
    nx = ny = 4  # initial resolution (4×4 quads per side unit)
    mesh = skfem.MeshTri.init_tensor(
        np.linspace(0.0, 2.0, 2 * nx + 1),
        np.linspace(0.0, 2.0, 2 * ny + 1),
    )
    # Remove elements whose centroid falls inside the upper-right cutout (x>1, y>1)
    cx = mesh.p[0, mesh.t].mean(axis=0)
    cy = mesh.p[1, mesh.t].mean(axis=0)
    bad_idx = np.where((cx > 1.0) & (cy > 1.0))[0]
    mesh = mesh.remove_elements(bad_idx).remove_unused_nodes()

    for _ in range(refine):
        mesh = mesh.refined()

    # Tag boundaries
    tol = 1e-10
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: x[0] < tol,
            "bottom": lambda x: x[1] < tol,
            "right": lambda x: (x[0] > 2.0 - tol) & (x[1] < 1.0 + tol),
            "inner_top": lambda x: (x[1] > 1.0 - tol) & (x[0] > 1.0 - tol) & (x[1] < 1.0 + tol),
            "inner_right": lambda x: (x[0] > 1.0 - tol) & (x[0] < 1.0 + tol) & (x[1] > 1.0 - tol),
            "top": lambda x: x[1] > 2.0 - tol,
        }
    )
    return mesh


def make_hollow_section_mesh(
    R_out: float,
    R_in: float,
    refine: int = 2,
) -> skfem.MeshTri:
    """Annular (hollow circular) cross-section mesh.

    Boundary tags: ``outer`` and ``inner``.

    Parameters
    ----------
    R_out:
        Outer radius (m).
    R_in:
        Inner radius (m).
    refine:
        Number of uniform refinement steps.
    """
    if R_in >= R_out * 0.9:
        raise ValueError(
            f"Inner radius ({R_in}) must be less than 90 % of outer radius ({R_out})."
        )

    n_theta = 24
    n_r = 8
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radii = np.linspace(R_in, R_out, n_r)

    pts: list[list[float]] = []
    for r in radii:
        for t in theta:
            pts.append([r * np.cos(t), r * np.sin(t)])
    pts_arr = np.array(pts, dtype=float)

    from scipy.spatial import Delaunay  # noqa: PLC0415

    tri = Delaunay(pts_arr)

    # Remove triangles whose centroid falls inside the inner hole
    centroids = pts_arr[tri.simplices].mean(axis=1)
    r_centroids = np.linalg.norm(centroids, axis=1)
    valid = r_centroids > R_in + 1e-6
    mesh = skfem.MeshTri(pts_arr.T, tri.simplices[valid].T)

    for _ in range(refine):
        mesh = mesh.refined()

    tol = 1e-6
    mesh = mesh.with_boundaries(
        {
            "outer": lambda x: np.abs(np.linalg.norm(x, axis=0) - R_out) < tol * 10 + R_out * 0.05,
            "inner": lambda x: np.abs(np.linalg.norm(x, axis=0) - R_in) < tol * 10 + R_in * 0.05,
            "bottom": lambda x: x[1] < -R_out * 0.9,
            "top": lambda x: x[1] > R_out * 0.9,
            "left": lambda x: x[0] < -R_out * 0.9,
            "right": lambda x: x[0] > R_out * 0.9,
        }
    )
    return mesh
