import numpy as np


def von_mises_2d(sxx: float, syy: float, txy: float) -> float:
    """2D plane-stress Von Mises stress."""
    return float(np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * txy**2))


def von_mises_3d(
    sxx: float,
    syy: float,
    szz: float,
    txy: float,
    tyz: float,
    txz: float,
) -> float:
    """3D Von Mises stress."""
    return float(
        np.sqrt(
            0.5
            * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6 * (txy**2 + tyz**2 + txz**2)
            )
        )
    )


def principal_stresses_2d(
    sxx: float, syy: float, txy: float
) -> tuple:
    """Return (sigma1, sigma2, angle_deg) for 2D plane-stress state."""
    avg = (sxx + syy) / 2.0
    r = np.sqrt(((sxx - syy) / 2.0) ** 2 + txy**2)
    s1 = avg + r
    s2 = avg - r
    angle = 0.5 * np.degrees(np.arctan2(2 * txy, sxx - syy))
    return float(s1), float(s2), float(angle)


def displacement_magnitude(U: np.ndarray) -> np.ndarray:
    """Compute per-node displacement magnitude from interleaved DOF vector [u0,v0,u1,v1,...].

    Works for both 2-DOF/node and 3-DOF/node vectors.
    Assumes 2 DOFs per node (planar) — reshape accordingly.
    """
    U = np.asarray(U, dtype=float)
    n_dof = 2
    n_nodes = len(U) // n_dof
    reshaped = U[: n_nodes * n_dof].reshape(n_nodes, n_dof)
    return np.linalg.norm(reshaped, axis=1)


def apply_deformation(
    verts: np.ndarray,
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    disp_z: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Return deformed vertices = verts + scale * disp."""
    verts = np.asarray(verts, dtype=float)
    dx = np.asarray(disp_x, dtype=float)
    dy = np.asarray(disp_y, dtype=float)
    dz = np.asarray(disp_z, dtype=float)
    deformed = verts.copy()
    deformed[:, 0] += scale * dx
    deformed[:, 1] += scale * dy
    deformed[:, 2] += scale * dz
    return deformed


def safety_factor(von_mises: float, yield_strength: float) -> float:
    """Factor of safety = yield_strength / von_mises (clipped to avoid div/0)."""
    if von_mises < 1e-12:
        return float("inf")
    return float(yield_strength / von_mises)
