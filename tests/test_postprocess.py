"""Unit tests for post-processing functions (postprocess.py)."""

from __future__ import annotations

import numpy as np
import pytest
import skfem
from skfem import Basis, ElementVector

from fem.mesh_gen import make_cantilever_mesh
from fem.postprocess import (
    compute_safety_factor,
    compute_strain_energy,
    compute_von_mises,
)
from fem.solver import assemble_stiffness, solve_plane_stress


class TestComputeVonMises:
    def test_returns_node_array(self):
        mesh = make_cantilever_mesh(1.0, 0.1, nx=8, ny=4)
        u, basis = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        vm = compute_von_mises(basis, u, 210e9, 0.3)
        assert vm.shape == (mesh.p.shape[1],)

    def test_non_negative(self):
        mesh = make_cantilever_mesh(1.0, 0.1, nx=8, ny=4)
        u, basis = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        vm = compute_von_mises(basis, u, 210e9, 0.3)
        assert np.all(vm >= 0.0), "Von Mises stress must be non-negative"

    def test_zero_displacement_gives_zero_stress(self):
        mesh = make_cantilever_mesh(1.0, 0.1, nx=6, ny=3)
        element = ElementVector(skfem.ElementTriP2())
        basis = Basis(mesh, element)
        u_zero = np.zeros(basis.N)
        vm = compute_von_mises(basis, u_zero, 210e9, 0.3)
        assert np.allclose(vm, 0.0, atol=1e-10)

    def test_stress_increases_with_force(self):
        mesh = make_cantilever_mesh(1.0, 0.1, nx=8, ny=4)
        u1, basis = solve_plane_stress(mesh, 210e9, 0.3, -500.0)
        u2, _ = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        vm1 = compute_von_mises(basis, u1, 210e9, 0.3)
        vm2 = compute_von_mises(basis, u2, 210e9, 0.3)
        assert vm2.max() > vm1.max()


class TestComputeSafetyFactor:
    def test_basic(self):
        vm = np.array([100e6, 50e6, 200e6])
        sf = compute_safety_factor(vm, 235e6)
        assert sf == pytest.approx(235e6 / 200e6)

    def test_zero_stress_returns_inf(self):
        vm = np.zeros(10)
        sf = compute_safety_factor(vm, 235e6)
        assert sf == float("inf")

    def test_high_stress_below_one(self):
        vm = np.array([300e6])
        sf = compute_safety_factor(vm, 235e6)
        assert sf < 1.0


class TestComputeStrainEnergy:
    def test_positive(self):
        mesh = make_cantilever_mesh(1.0, 0.1, nx=6, ny=3)
        u, basis = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        K = assemble_stiffness(basis, 210e9, 0.3)
        se = compute_strain_energy(K, u)
        assert se > 0.0

    def test_zero_displacement_gives_zero(self):
        mesh = make_cantilever_mesh(1.0, 0.1, nx=6, ny=3)
        _u, basis = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        K = assemble_stiffness(basis, 210e9, 0.3)
        u_zero = np.zeros(K.shape[0])
        se = compute_strain_energy(K, u_zero)
        assert se == pytest.approx(0.0, abs=1e-20)

    def test_scales_quadratically_with_force(self):
        """Strain energy ∝ F²  (linear elasticity)."""
        mesh = make_cantilever_mesh(1.0, 0.1, nx=6, ny=3)
        u1, basis = solve_plane_stress(mesh, 210e9, 0.3, -500.0)
        u2, _ = solve_plane_stress(mesh, 210e9, 0.3, -1000.0)
        K = assemble_stiffness(basis, 210e9, 0.3)
        se1 = compute_strain_energy(K, u1)
        se2 = compute_strain_energy(K, u2)
        ratio = se2 / se1
        assert abs(ratio - 4.0) < 0.05, f"Expected ratio 4.0, got {ratio:.4f}"
