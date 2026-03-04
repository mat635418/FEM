"""FEM Lab — Streamlit UI (thin orchestrator).

This file contains only Streamlit presentation logic.
All physics is delegated to the ``fem`` package.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pyvista as pv
import skfem as _skfem
import streamlit as st
from PIL import Image
from skfem import Basis, ElementVector

from fem.materials import MATERIALS
from fem.mesh_gen import (
    make_cantilever_mesh,
    make_hollow_section_mesh,
    make_l_bracket_mesh,
)
from fem.postprocess import (
    compute_safety_factor,
    compute_strain_energy,
    compute_von_mises,
)
from fem.solver import assemble_stiffness, solve_plane_stress

# ── Critical: headless rendering must be set before any PyVista calls ──────
pv.OFF_SCREEN = True

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Python FEM Lab", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏗️ FEM Lab Settings")

    # ── 1. Geometry ──────────────────────────────────────────────────────────
    st.header("1. Geometry Selection")
    shape_type = st.selectbox(
        "Select Cross-Section Template",
        ["Cantilever Beam", "Hollow Section", "L-Bracket"],
    )

    st.header("2. Geometry Parameters")
    geo_params: dict

    if shape_type == "Cantilever Beam":
        L = st.slider("Length (m)", 0.5, 10.0, 2.0, step=0.5)
        H = st.slider("Height (m)", 0.02, 1.0, 0.1, step=0.01)
        nx = st.slider("Elements along length", 10, 60, 20, step=5)
        ny = st.slider("Elements along height", 2, 20, 6, step=1)
        geo_params = {"L": L, "H": H, "nx": nx, "ny": ny}

    elif shape_type == "Hollow Section":
        outer_rad = st.slider("Outer Radius (m)", 0.2, 2.0, 1.0, step=0.1)
        inner_rad = st.slider("Inner Radius (m)", 0.05, 1.8, 0.4, step=0.05)
        refine = st.slider("Mesh refinement steps", 1, 3, 2)
        geo_params = {"R_out": outer_rad, "R_in": inner_rad, "refine": refine}

        if inner_rad >= outer_rad * 0.9:
            st.error(
                f"Inner radius ({inner_rad:.2f} m) must be less than 90 % of "
                f"outer radius ({outer_rad:.2f} m).  Please adjust the sliders."
            )
            st.stop()

    else:  # L-Bracket
        refine = st.slider("Mesh refinement steps", 1, 3, 2)
        geo_params = {"refine": refine}

    # ── 3. Material & Load ───────────────────────────────────────────────────
    st.divider()
    st.header("3. Material & Load")

    material_name = st.selectbox("Material", list(MATERIALS.keys()))
    mat = MATERIALS[material_name]

    if mat is not None:
        E_default = float(mat["E"])
        nu_default = float(mat["nu"])
        yield_strength = float(mat["yield_strength"])
        st.caption(
            f"E = {E_default/1e9:.0f} GPa  |  ν = {nu_default}  |  "
            f"σ_y = {yield_strength/1e6:.0f} MPa"
        )
        E = E_default
    else:
        E = st.number_input("Young's Modulus (GPa)", value=210.0, min_value=1.0) * 1e9
        nu_default = 0.30
        yield_strength = (
            st.number_input("Yield strength (MPa)", value=235.0, min_value=1.0) * 1e6
        )

    nu = st.slider(
        "Poisson's ratio (ν)",
        0.01,
        0.49,
        float(nu_default),
        step=0.01,
    )

    force_kn = st.slider("Vertical Force (kN)", -200.0, 200.0, -50.0, step=5.0)
    force_val = force_kn * 1_000.0  # N

    if force_val == 0.0:
        st.warning("Force is zero — displacement will be zero everywhere.")

    # ── Visualisation options ─────────────────────────────────────────────────
    st.divider()
    st.header("4. Visualisation")
    show_deformed = st.checkbox("Show deformed shape", value=True)
    warp_factor = st.slider("Deformation scale factor", 1, 5_000, 100)
    cmap = st.selectbox(
        "Colormap", ["viridis", "plasma", "inferno", "coolwarm", "jet"], index=0
    )


# ── Cached solver call ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️ Running FEM solver…", ttl=300)
def cached_solve(
    shape: str,
    geo: tuple,
    _E: float,
    _nu: float,
    _force: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run FEM solver; return displacement u, node coordinates, element connectivity."""
    if shape == "Cantilever Beam":
        mesh = make_cantilever_mesh(geo[0], geo[1], nx=int(geo[2]), ny=int(geo[3]))
        fixed, load = "left", "right"
    elif shape == "Hollow Section":
        mesh = make_hollow_section_mesh(geo[0], geo[1], refine=int(geo[2]))
        fixed, load = "left", "right"
    else:
        mesh = make_l_bracket_mesh(refine=int(geo[0]))
        fixed, load = "left", "right"

    u, _basis = solve_plane_stress(mesh, _E, _nu, _force, fixed, load)
    return u, mesh.p.copy(), mesh.t.copy()


# ── Build geo tuple (cache key) ──────────────────────────────────────────────
if shape_type == "Cantilever Beam":
    geo_tuple: tuple = (geo_params["L"], geo_params["H"], geo_params["nx"], geo_params["ny"])
elif shape_type == "Hollow Section":
    geo_tuple = (geo_params["R_out"], geo_params["R_in"], geo_params["refine"])
else:
    geo_tuple = (geo_params["refine"],)

# ── Solve ────────────────────────────────────────────────────────────────────
st.title(f"🔬 FEM Lab — {shape_type}")

try:
    u, pts, tris = cached_solve(shape_type, geo_tuple, E, nu, force_val)
except Exception as exc:
    st.error(
        f"**Solver failed:** {exc}\n\n"
        "Try reducing the force magnitude, using a coarser mesh, or choosing a "
        "different geometry."
    )
    st.stop()

# ── Post-processing (not cached — runs fast) ─────────────────────────────────
if shape_type == "Cantilever Beam":
    _mesh = make_cantilever_mesh(
        geo_params["L"],
        geo_params["H"],
        nx=geo_params["nx"],
        ny=geo_params["ny"],
    )
elif shape_type == "Hollow Section":
    _mesh = make_hollow_section_mesh(
        geo_params["R_out"], geo_params["R_in"], refine=geo_params["refine"]
    )
else:
    _mesh = make_l_bracket_mesh(refine=geo_params["refine"])

_element = ElementVector(_skfem.ElementTriP2())
_basis = Basis(_mesh, _element)
_K = assemble_stiffness(_basis, E, nu)

vm_stress = compute_von_mises(_basis, u, E, nu)
sf = compute_safety_factor(vm_stress, yield_strength)
se = compute_strain_energy(_K, u)

max_disp = float(np.sqrt(u[0::2] ** 2 + u[1::2] ** 2).max())
peak_vm = float(vm_stress.max())
n_dofs = int(u.shape[0])

# ── Layout ───────────────────────────────────────────────────────────────────
col_plot, col_metrics = st.columns([3, 1])

with col_metrics:
    st.subheader("📊 Results")
    st.metric("Max Displacement", f"{max_disp:.4e} m")
    st.metric("Peak Von Mises", f"{peak_vm:.3e} Pa")
    st.metric("DOF count", f"{n_dofs:,}")
    st.metric("Strain Energy", f"{se:.3e} J")

    if sf == float("inf"):
        sf_label = "∞"
        sf_icon = "🟢"
    else:
        sf_label = f"{sf:.2f}"
        sf_icon = "🟢" if sf > 2.0 else ("🟠" if sf > 1.0 else "🔴")
    st.metric("Safety Factor", f"{sf_icon} {sf_label}")

    st.write("### Simulation Notes")
    st.caption(
        "Plane-stress 2-D FEM using triangular P2 elements (scikit-fem).  "
        "Load is applied as a uniform distributed traction on the free edge."
    )

with col_plot:
    # ── Build PyVista mesh for visualisation ──────────────────────────────────
    n_nodes_2d = pts.shape[1]
    pts_3d = np.zeros((n_nodes_2d, 3))
    pts_3d[:, 0] = pts[0]
    pts_3d[:, 1] = pts[1]

    n_elems = tris.shape[1]
    faces = np.hstack(
        [np.full((n_elems, 1), 3, dtype=np.int32), tris.T.astype(np.int32)]
    ).ravel()

    pv_mesh = pv.PolyData(pts_3d, faces)
    pv_mesh.point_data["Von Mises (Pa)"] = vm_stress

    disp_3d = np.zeros((n_nodes_2d, 3))
    disp_3d[:, 0] = u[0::2]
    disp_3d[:, 1] = u[1::2]
    pv_mesh.point_data["Displacement"] = disp_3d

    plot_mesh = (
        pv_mesh.warp_by_vector("Displacement", factor=warp_factor)
        if show_deformed
        else pv_mesh
    )

    plotter = pv.Plotter(window_size=[900, 600], off_screen=True)
    plotter.add_mesh(
        plot_mesh,
        scalars="Von Mises (Pa)",
        cmap=cmap,
        show_edges=True,
        scalar_bar_args={"title": "Von Mises Stress (Pa)"},
    )
    plotter.background_color = "white"
    plotter.view_xy()
    screenshot = plotter.screenshot(return_img=True)
    st.image(screenshot, use_container_width=True)

    # ── Export buttons ────────────────────────────────────────────────────────
    st.subheader("📥 Export")
    ecols = st.columns(3)

    df = pd.DataFrame(
        {
            "x": pts[0],
            "y": pts[1],
            "disp_x": u[0::2],
            "disp_y": u[1::2],
            "von_mises_pa": vm_stress,
        }
    )
    ecols[0].download_button(
        "📄 Results (CSV)", df.to_csv(index=False), "fem_results.csv", "text/csv"
    )

    problem_def = {
        "shape": shape_type,
        "geometry": geo_params,
        "material": {
            "name": material_name,
            "E_Pa": E,
            "nu": nu,
            "yield_strength_Pa": yield_strength,
        },
        "load_N": force_val,
    }
    ecols[1].download_button(
        "📋 Problem (JSON)",
        json.dumps(problem_def, indent=2),
        "problem.json",
        "application/json",
    )

    img_buf = io.BytesIO()
    Image.fromarray(screenshot).save(img_buf, format="PNG")
    ecols[2].download_button(
        "🖼️ Plot (PNG)", img_buf.getvalue(), "fem_result.png", "image/png"
    )

st.success("✅ Simulation complete. Adjust sidebar controls to update.")
