"""Real 2-D plane-stress FEM solver using scikit-fem.

The solver assembles a global stiffness matrix from triangular P2 elements,
applies Dirichlet (fixed) boundary conditions, applies a Neumann (traction)
boundary condition as a distributed transverse load, and solves the resulting
sparse system with SciPy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import skfem
from skfem import Basis, ElementVector, FacetBasis, LinearForm, condense
from skfem import solve as sksolve
from skfem.models.elasticity import lame_parameters, linear_elasticity


def _boundary_length(mesh: skfem.MeshTri, boundary_name: str) -> float:
    """Return the total arc-length of a named boundary."""
    facet_ids = mesh.boundaries[boundary_name]
    facet_nodes = mesh.facets[:, facet_ids]          # shape (2, n_facets)
    coords = mesh.p[:, facet_nodes]                   # shape (2, 2, n_facets)
    edge_lengths = np.linalg.norm(
        coords[:, 1, :] - coords[:, 0, :], axis=0
    )
    return float(edge_lengths.sum())


def solve_plane_stress(
    mesh: skfem.MeshTri,
    E: float,
    nu: float,
    force: float,
    fixed_boundary: str = "left",
    load_boundary: str = "right",
) -> tuple[np.ndarray, Basis]:
    """Assemble and solve a 2-D plane-stress problem.

    The load is interpreted as total applied force (N per unit thickness).
    It is distributed uniformly as a traction over ``load_boundary`` in the
    −y direction (downward).

    Parameters
    ----------
    mesh:
        A ``skfem.MeshTri`` with named boundary tags.
    E:
        Young's modulus (Pa).
    nu:
        Poisson's ratio (–).
    force:
        Total applied force (N).  Negative values point in the −y direction.
    fixed_boundary:
        Name of the boundary where all DOFs are set to zero.
    load_boundary:
        Name of the boundary where the distributed transverse traction is applied.

    Returns
    -------
    u:
        Displacement vector (n_dofs,).  DOFs are interleaved as
        ``[u_x_0, u_y_0, u_x_1, u_y_1, …]``.
    basis:
        The ``skfem.Basis`` object (useful for post-processing).
    """
    element = ElementVector(skfem.ElementTriP2())
    basis = Basis(mesh, element)

    lam, mu = lame_parameters(E, nu)
    K = linear_elasticity(lam, mu).assemble(basis)
    f = basis.zeros()

    # Neumann BC: distributed traction on load_boundary
    fb = FacetBasis(mesh, element, facets=mesh.boundaries[load_boundary])
    bl = _boundary_length(mesh, load_boundary)
    traction = force / bl  # N/m

    @LinearForm
    def load_form(v, _):  # type: ignore[override]
        return traction * v.value[1]

    f += load_form.assemble(fb)

    # Dirichlet BC: fix all DOFs on fixed_boundary
    fixed_dofs = basis.get_dofs(mesh.boundaries[fixed_boundary])
    u = sksolve(*condense(K, f, D=fixed_dofs.all()))
    return u, basis


def assemble_stiffness(
    basis: Basis,
    E: float,
    nu: float,
) -> scipy.sparse.csr_matrix:
    """Assemble and return the global stiffness matrix (CSR format)."""
    lam, mu = lame_parameters(E, nu)
    return linear_elasticity(lam, mu).assemble(basis)
