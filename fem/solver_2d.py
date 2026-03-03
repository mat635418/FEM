"""2-D plane-stress FEM solver using CST (Constant Strain Triangle) elements."""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# CST element stiffness matrix
# ---------------------------------------------------------------------------

def _cst_stiffness(coords: np.ndarray, E: float, nu: float, t: float) -> np.ndarray:
    """6×6 CST element stiffness matrix.

    Parameters
    ----------
    coords : shape (3, 2) — node (x, y) coordinates
    E      : Young's modulus
    nu     : Poisson's ratio
    t      : thickness

    Returns
    -------
    ke : (6, 6) array
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # Area
    A = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    if A < 1e-15:
        return np.zeros((6, 6))

    # Shape function derivatives
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

    # Strain-displacement matrix B (3×6)
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i]   = b[i]
        B[1, 2*i+1] = c[i]
        B[2, 2*i]   = c[i]
        B[2, 2*i+1] = b[i]
    B /= (2.0 * A)

    # Plane-stress constitutive matrix D (3×3)
    D = E / (1.0 - nu**2) * np.array([
        [1,  nu, 0],
        [nu, 1,  0],
        [0,  0,  (1 - nu) / 2.0],
    ])

    ke = t * A * (B.T @ D @ B)
    return ke


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_plate_fem(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
    forces_dict: dict,
    fixed_dofs: list,
):
    """Assemble and solve a 2-D plane-stress FEM problem.

    Parameters
    ----------
    nodes       : (N, 2) node coordinates
    elements    : (M, 3) node indices (triangles)
    E           : Young's modulus [Pa]
    nu          : Poisson's ratio
    thickness   : plate thickness [m]
    forces_dict : {node_index: [fx, fy]}
    fixed_dofs  : list of constrained global DOF indices

    Returns
    -------
    U : (2*N,) displacement vector [u0, v0, u1, v1, ...]
    K : sparse global stiffness matrix (lil → csr)
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes

    K = lil_matrix((n_dof, n_dof))
    F = np.zeros(n_dof)

    # Assemble global K
    for elem in elements:
        coords = nodes[elem]
        ke = _cst_stiffness(coords, E, nu, thickness)
        for i_loc, i_glob in enumerate(elem):
            for j_loc, j_glob in enumerate(elem):
                K[2*i_glob,   2*j_glob  ] += ke[2*i_loc,   2*j_loc  ]
                K[2*i_glob,   2*j_glob+1] += ke[2*i_loc,   2*j_loc+1]
                K[2*i_glob+1, 2*j_glob  ] += ke[2*i_loc+1, 2*j_loc  ]
                K[2*i_glob+1, 2*j_glob+1] += ke[2*i_loc+1, 2*j_loc+1]

    # Apply forces
    for node_idx, (fx, fy) in forces_dict.items():
        F[2*node_idx]   += fx
        F[2*node_idx+1] += fy

    # Apply boundary conditions (penalty or elimination)
    fixed_set = set(fixed_dofs)
    free = [d for d in range(n_dof) if d not in fixed_set]

    K_csr = K.tocsr()
    K_red = K_csr[np.ix_(free, free)]
    F_red = F[free]

    u_red = spsolve(K_red, F_red)

    U = np.zeros(n_dof)
    for i, d in enumerate(free):
        U[d] = u_red[i]

    return U, K_csr


def compute_stress_2d(
    U: np.ndarray,
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    nu: float,
):
    """Compute per-element stress components for plane-stress CST elements.

    Returns
    -------
    sxx : (M,) array
    syy : (M,) array
    txy : (M,) array
    vm  : (M,) Von Mises stress array
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_elem = len(elements)

    sxx = np.zeros(n_elem)
    syy = np.zeros(n_elem)
    txy = np.zeros(n_elem)
    vm  = np.zeros(n_elem)

    D = E / (1.0 - nu**2) * np.array([
        [1,  nu, 0],
        [nu, 1,  0],
        [0,  0,  (1 - nu) / 2.0],
    ])

    for idx, elem in enumerate(elements):
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        A = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if A < 1e-15:
            continue

        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2*i]   = b[i]
            B[1, 2*i+1] = c[i]
            B[2, 2*i]   = c[i]
            B[2, 2*i+1] = b[i]
        B /= (2.0 * A)

        ue = np.array([U[2*n] for n in elem] + [U[2*n+1] for n in elem])
        # Reorder: [u0, v0, u1, v1, u2, v2]
        ue_ord = np.zeros(6)
        for i, n in enumerate(elem):
            ue_ord[2*i]   = U[2*n]
            ue_ord[2*i+1] = U[2*n+1]

        stress = D @ B @ ue_ord
        sxx[idx] = stress[0]
        syy[idx] = stress[1]
        txy[idx] = stress[2]
        vm[idx] = np.sqrt(
            stress[0]**2 - stress[0]*stress[1] + stress[1]**2 + 3*stress[2]**2
        )

    return sxx, syy, txy, vm
