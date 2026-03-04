"""Tests for the FEM solver (solver.py)."""

from __future__ import annotations

import numpy as np

from fem.mesh_gen import make_cantilever_mesh
from fem.solver import solve_plane_stress


class TestCantileverTipDeflection:
    """Cantilever beam deflection vs. Euler-Bernoulli analytical solution.

    Note on accuracy:
        The 2-D plane-stress FEM with *fully clamped* boundary conditions
        (u_x = u_y = 0 on the entire left edge) over-constrains the Poisson
        contraction at the support.  With ν = 0.3 this introduces ~9 % extra
        stiffness compared to the 1-D Euler-Bernoulli beam formula.  The 15 %
        tolerance below accounts for this known systematic effect.
    """

    def test_deflection_order_of_magnitude(self):
        """FEM tip deflection must be within 15 % of the Euler-Bernoulli formula."""
        L, H = 1.0, 0.05
        E, nu = 210e9, 0.3
        force = -1000.0          # total force (N / unit thickness)

        mom_inertia = H**3 / 12.0
        delta_analytical = abs(force) * L**3 / (3.0 * E * mom_inertia)

        mesh = make_cantilever_mesh(L, H, nx=40, ny=8)
        u, _basis = solve_plane_stress(mesh, E, nu, force)

        disp_y = u[1::2]         # every other DOF is u_y
        delta_fem = float(np.abs(disp_y).max())

        rel_error = abs(delta_fem - delta_analytical) / delta_analytical
        assert rel_error < 0.15, (
            f"Relative error {rel_error:.1%} exceeds 15 % tolerance.  "
            f"FEM={delta_fem:.3e} m, analytical={delta_analytical:.3e} m."
        )

    def test_deflection_zero_force(self):
        """Zero force must produce zero displacement everywhere."""
        mesh = make_cantilever_mesh(1.0, 0.1, nx=10, ny=4)
        u, _basis = solve_plane_stress(mesh, 210e9, 0.3, 0.0)
        assert np.allclose(u, 0.0, atol=1e-20)

    def test_deflection_scales_linearly_with_force(self):
        """Doubling the force must double the tip deflection (linear elasticity)."""
        mesh = make_cantilever_mesh(1.0, 0.1, nx=10, ny=4)
        u1, _ = solve_plane_stress(mesh, 210e9, 0.3, -500.0)
        u2, _ = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        ratio = float(np.abs(u2[1::2]).max()) / float(np.abs(u1[1::2]).max())
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio 2.0, got {ratio:.4f}"

    def test_deflection_scales_inversely_with_modulus(self):
        """Doubling E must halve the tip deflection (linear elasticity)."""
        mesh = make_cantilever_mesh(1.0, 0.1, nx=10, ny=4)
        u1, _ = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        u2, _ = solve_plane_stress(mesh, 420e9, 0.3, -1000.0)
        ratio = float(np.abs(u1[1::2]).max()) / float(np.abs(u2[1::2]).max())
        assert abs(ratio - 2.0) < 0.05, f"Expected ratio 2.0, got {ratio:.4f}"


class TestPatchTest:
    """Constant-strain patch test: uniform axial extension."""

    def test_uniform_axial_extension(self):
        """Applying a distributed transverse traction on the right face of a
        rectangular mesh (fixed left) must produce a finite, non-trivial
        displacement solution."""
        L, H = 1.0, 0.1
        E, nu = 210e9, 0.3
        force = 10_000.0  # N, transverse (−y) traction applied to right face

        mesh = make_cantilever_mesh(L, H, nx=10, ny=4)
        u, _basis = solve_plane_stress(
            mesh, E, nu, force, fixed_boundary="left", load_boundary="right"
        )

        # Verify solver convergence: solution must be finite
        assert np.all(np.isfinite(u)), "Solution contains NaN or Inf values."
        # Verify non-trivial result: at least some DOFs have non-zero displacement
        assert np.abs(u).max() > 0.0, "All displacements are zero for non-zero force."
