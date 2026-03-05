"""Post-processing utilities: Von Mises stress, safety factor, strain energy."""

from __future__ import annotations

import numpy as np
import scipy.sparse
from skfem import Basis


def compute_von_mises(
    basis: Basis,
    u: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """Compute Von Mises stress at every mesh node.

    Uses the plane-stress constitutive relation to recover stresses from nodal
    displacements, evaluated at element barycentres and scatter-averaged to nodes.

    Parameters
    ----------
    basis:
        The assembled ``skfem.Basis`` for the displacement field.
    u:
        Displacement vector returned by :func:`~fem.solver.solve_plane_stress`.
    E:
        Young's modulus (Pa).
    nu:
        Poisson's ratio (–).

    Returns
    -------
    vm:
        Von Mises stress at each mesh node (Pa), shape ``(n_nodes,)``.
    """
    return _compute_von_mises_p1_gradient(basis, u, E, nu)


def _compute_von_mises_p1_gradient(
    basis: Basis,
    u: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """Compute Von Mises stress by evaluating P1 displacement gradients at element
    barycentres, then scatter-averaging the result to mesh nodes.
    """
    mesh = basis.mesh

    # Plane-stress stiffness coefficients
    c11 = E / (1.0 - nu**2)
    c12 = nu * c11
    c33 = E / (2.0 * (1.0 + nu))

    n_nodes = mesh.p.shape[1]
    n_elems = mesh.t.shape[1]
    vm_elem = np.zeros(n_elems)

    pts = mesh.p   # (2, n_nodes)
    tris = mesh.t  # (3, n_elems)

    for e in range(n_elems):
        # Vertex indices
        v = tris[:, e]  # (3,)
        xy = pts[:, v]  # (2, 3)

        # Jacobian of reference-to-physical mapping
        x1, y1 = xy[0, 0], xy[1, 0]
        x2, y2 = xy[0, 1], xy[1, 1]
        x3, y3 = xy[0, 2], xy[1, 2]

        J = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        Jinv = np.linalg.inv(J)

        # Gradients of linear (P1) basis functions in physical space
        dNdr = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        dNdx = Jinv @ dNdr  # (2, 3); rows: d/dx, d/dy; cols: nodes

        # Nodal DOF indices for corner nodes only (P2 has corner + edge DOFs)
        ix = basis.dofs.nodal_dofs[0, v]  # x-component DOF indices
        iy = basis.dofs.nodal_dofs[1, v]  # y-component DOF indices

        ux = u[ix]  # (3,)
        uy = u[iy]  # (3,)

        eps_xx = float(dNdx[0] @ ux)
        eps_yy = float(dNdx[1] @ uy)
        eps_xy = 0.5 * float(dNdx[1] @ ux + dNdx[0] @ uy)

        # Plane-stress constitutive law
        sig_xx = c11 * eps_xx + c12 * eps_yy
        sig_yy = c12 * eps_xx + c11 * eps_yy
        sig_xy = 2.0 * c33 * eps_xy

        vm_elem[e] = np.sqrt(
            max(sig_xx**2 - sig_xx * sig_yy + sig_yy**2 + 3.0 * sig_xy**2, 0.0)
        )

    # Scatter-average element stresses to corner nodes
    vm_nodes = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    for e in range(n_elems):
        v = tris[:, e]
        vm_nodes[v] += vm_elem[e]
        count[v] += 1.0
    count = np.maximum(count, 1)
    return vm_nodes / count


def compute_safety_factor(vm_stress: np.ndarray, yield_strength: float) -> float:
    """Return the safety factor as ``yield_strength / max(Von Mises stress)``."""
    peak = float(vm_stress.max())
    if peak <= 0.0:
        return float("inf")
    return yield_strength / peak


def compute_strain_energy(K: scipy.sparse.csr_matrix, u: np.ndarray) -> float:
    """Return the elastic strain energy ``0.5 * uᵀ K u`` (J)."""
    return 0.5 * float(u @ K @ u)
