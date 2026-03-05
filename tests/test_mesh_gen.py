"""Smoke tests for mesh generation (mesh_gen.py)."""

from __future__ import annotations

import numpy as np
import pytest
import skfem

from fem.mesh_gen import (
    make_cantilever_mesh,
    make_hollow_section_mesh,
    make_l_bracket_mesh,
)


class TestCantileverMesh:
    def test_returns_meshTri(self):
        mesh = make_cantilever_mesh(1.0, 0.1)
        assert isinstance(mesh, skfem.MeshTri)

    def test_dimensions(self):
        L, H = 3.0, 0.5
        mesh = make_cantilever_mesh(L, H, nx=15, ny=5)
        # All nodes in [0, L] × [0, H]
        assert mesh.p[0].min() == pytest.approx(0.0)
        assert mesh.p[0].max() == pytest.approx(L)
        assert mesh.p[1].min() == pytest.approx(0.0)
        assert mesh.p[1].max() == pytest.approx(H)

    def test_boundary_tags(self):
        mesh = make_cantilever_mesh(1.0, 0.1)
        for tag in ("left", "right", "top", "bottom"):
            assert tag in mesh.boundaries, f"Missing boundary tag '{tag}'"

    def test_element_count(self):
        mesh = make_cantilever_mesh(1.0, 0.1, nx=10, ny=4)
        assert mesh.nelements >= 10, "Too few elements"

    def test_valid_mesh(self):
        mesh = make_cantilever_mesh(2.0, 0.2, nx=8, ny=4)
        assert mesh.is_valid()


class TestLBracketMesh:
    def test_returns_meshTri(self):
        mesh = make_l_bracket_mesh(refine=1)
        assert isinstance(mesh, skfem.MeshTri)

    def test_l_shape_extent(self):
        mesh = make_l_bracket_mesh(refine=1)
        # Nodes must span [0,2] × [0,2]
        assert mesh.p[0].max() == pytest.approx(2.0, abs=0.05)
        assert mesh.p[1].max() == pytest.approx(2.0, abs=0.05)

    def test_no_nodes_in_cutout(self):
        mesh = make_l_bracket_mesh(refine=1)
        # No node should be in the upper-right quadrant (x>1, y>1)
        mask = (mesh.p[0] > 1.0 + 0.05) & (mesh.p[1] > 1.0 + 0.05)
        assert not mask.any(), "Nodes found inside the L-bracket cutout"

    def test_boundary_tags(self):
        mesh = make_l_bracket_mesh(refine=1)
        for tag in ("left", "bottom"):
            assert tag in mesh.boundaries, f"Missing boundary tag '{tag}'"

    def test_valid_mesh(self):
        mesh = make_l_bracket_mesh(refine=1)
        assert mesh.is_valid()


class TestHollowSectionMesh:
    def test_returns_meshTri(self):
        mesh = make_hollow_section_mesh(1.0, 0.3)
        assert isinstance(mesh, skfem.MeshTri)

    def test_no_nodes_inside_hole(self):
        R_in = 0.3
        mesh = make_hollow_section_mesh(1.0, R_in, refine=0)
        r_nodes = np.linalg.norm(mesh.p, axis=0)
        assert r_nodes.min() >= R_in * 0.85, (
            "Nodes found inside the inner hole"
        )

    def test_raises_on_invalid_radii(self):
        with pytest.raises(ValueError, match="Inner radius"):
            make_hollow_section_mesh(1.0, 0.95)  # 0.95 ≥ 0.9*1.0 → invalid

    def test_valid_mesh(self):
        mesh = make_hollow_section_mesh(1.0, 0.3, refine=1)
        assert mesh.is_valid()
