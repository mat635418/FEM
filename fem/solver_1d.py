"""1-D Euler-Bernoulli beam FEM solver.

DOFs per node: [v (transverse displacement), θ (rotation)]
Node numbering: 0, 1, ..., n  (total n+1 nodes, n elements)
"""

import numpy as np
from scipy.linalg import solve


# ---------------------------------------------------------------------------
# Element stiffness matrix (2-node Hermite beam element)
# ---------------------------------------------------------------------------

def _beam_element_stiffness(E: float, I: float, le: float) -> np.ndarray:
    """4×4 Hermite beam element stiffness matrix."""
    k = E * I / le**3
    return np.array([
        [ 12,   6*le,  -12,   6*le],
        [  6*le, 4*le**2, -6*le, 2*le**2],
        [-12,  -6*le,   12,  -6*le],
        [  6*le, 2*le**2, -6*le, 4*le**2],
    ]) * k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_beam_fem(
    L: float,
    n: int,
    E: float,
    I: float,
    load_type: str,
    load_value: float,
    load_pos: float,
    bc_type: str,
):
    """Assemble and solve a 1-D Euler-Bernoulli beam FEM problem.

    Parameters
    ----------
    L          : beam length [m]
    n          : number of elements
    E          : Young's modulus [Pa]
    I          : second moment of area [m^4]
    load_type  : one of 'Point Force', 'Distributed Load', 'Couple/Moment'
    load_value : magnitude (N, N/m, or N·m)
    load_pos   : load position along beam [0..1] (fraction of L)
    bc_type    : 'Fixed-Free', 'Simply Supported', 'Fixed-Fixed', 'Free'

    Returns
    -------
    U  : global displacement vector (length 2*(n+1))
    le : element length [m]
    """
    n = max(n, 2)
    le = L / n
    n_nodes = n + 1
    n_dof = 2 * n_nodes  # v_0, θ_0, v_1, θ_1, ...

    # --- Assemble global stiffness matrix ---
    K = np.zeros((n_dof, n_dof))
    ke = _beam_element_stiffness(E, I, le)
    for e in range(n):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += ke[i, j]

    # --- Assemble load vector ---
    F = np.zeros(n_dof)
    x_load = load_pos * L  # physical position

    if load_type == "Point Force":
        # Find element and local coordinate
        e_idx = min(int(x_load / le), n - 1)
        xi = (x_load - e_idx * le) / le  # in [0, 1]
        # Hermite shape functions for transverse displacement
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = le * xi * (1 - xi)**2
        N3 = 3*xi**2 - 2*xi**3
        N4 = le * xi**2 * (xi - 1)
        dofs = [2*e_idx, 2*e_idx+1, 2*e_idx+2, 2*e_idx+3]
        for coef, d in zip([N1, N2, N3, N4], dofs):
            F[d] += load_value * coef

    elif load_type == "Distributed Load":
        # Uniform distributed load over entire beam
        for e in range(n):
            # Consistent load vector for UDL q
            q = load_value
            dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
            fe = q * le / 12.0 * np.array([6, le, 6, -le])
            for i, d in enumerate(dofs):
                F[d] += fe[i]

    elif load_type in ("Couple/Moment", "Moment"):
        # Apply moment at the closest node
        node = round(x_load / le)
        node = min(max(node, 0), n_nodes - 1)
        F[2*node + 1] += load_value  # θ DOF

    # --- Apply boundary conditions ---
    constrained = _get_constrained_dofs(bc_type, n_nodes)
    free = [d for d in range(n_dof) if d not in constrained]

    K_red = K[np.ix_(free, free)]
    F_red = F[free]

    u_red = solve(K_red, F_red)

    U = np.zeros(n_dof)
    for i, d in enumerate(free):
        U[d] = u_red[i]

    return U, le


def _get_constrained_dofs(bc_type: str, n_nodes: int) -> list:
    """Return list of constrained DOF indices for the given BC type."""
    last = n_nodes - 1
    if bc_type == "Fixed-Free":
        return [0, 1]              # clamp left end
    elif bc_type == "Simply Supported":
        return [0, 2 * last]       # v=0 at both ends
    elif bc_type == "Fixed-Fixed":
        return [0, 1, 2*last, 2*last+1]
    elif bc_type == "Free":
        # Rigid-body modes — add minimal support to make system non-singular
        return [0, 1]
    else:
        return [0, 1]


def compute_beam_stress(
    U: np.ndarray,
    n: int,
    le: float,
    E: float,
    height: float,
) -> np.ndarray:
    """Compute bending stress per element midpoint.

    σ = E * (h/2) * |κ|,  κ = d²v/dx² from Hermite shape functions.

    Parameters
    ----------
    U      : global displacement vector (length 2*(n+1))
    n      : number of elements
    le     : element length [m]
    E      : Young's modulus [Pa]
    height : cross-section height [m]

    Returns
    -------
    sigma : array of shape (n,) — bending stress per element [Pa]
    """
    sigma = np.zeros(n)
    xi = 0.5  # midpoint of element (ξ ∈ [0,1])
    for e in range(n):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        ue = U[dofs]
        # Second derivative of Hermite shape functions at ξ
        d2N1 = (-6 + 12*xi) / le**2
        d2N2 = (-4 + 6*xi) / le
        d2N3 = ( 6 - 12*xi) / le**2
        d2N4 = (-2 + 6*xi) / le
        kappa = d2N1*ue[0] + d2N2*ue[1] + d2N3*ue[2] + d2N4*ue[3]
        sigma[e] = E * (height / 2.0) * abs(kappa)
    return sigma
