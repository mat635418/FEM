"""Streamlit FEM Analysis App — main entry point."""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from fem.materials import get_material, list_materials
from fem.mesh import (
    make_beam_mesh,
    make_box_mesh,
    make_cylinder_mesh,
    make_lbracket_mesh,
    make_plate_mesh,
    load_stl_mesh,
)
from fem.solver_1d import build_beam_fem, compute_beam_stress
from fem.solver_2d import build_plate_fem, compute_stress_2d
from fem.postprocess import (
    von_mises_2d,
    safety_factor,
    apply_deformation,
    displacement_magnitude,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FEM Analysis App",
    page_icon="🔩",
    layout="wide",
)

st.title("🔩 Finite Element Method (FEM) Analysis")

# ---------------------------------------------------------------------------
# Sidebar — all input parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parameters")

    # ── Geometry ──────────────────────────────────────────────────────────
    st.subheader("📐 Geometry")
    shape = st.selectbox(
        "Shape",
        ["Rectangular Beam", "Cylinder", "Box / Cuboid", "L-Bracket", "Plate (2D)"],
        key="shape_select",
    )

    if shape == "Rectangular Beam":
        beam_L = st.slider("Length L [m]", 0.1, 10.0, 2.0, 0.1, key="beam_L")
        beam_w = st.slider("Width w [m]", 0.01, 1.0, 0.05, 0.01, key="beam_w")
        beam_h = st.slider("Height h [m]", 0.01, 1.0, 0.1, 0.01, key="beam_h")
        beam_n = st.slider("Mesh divisions", 4, 40, 10, 1, key="beam_n")
    else:
        beam_L, beam_w, beam_h, beam_n = 2.0, 0.05, 0.1, 10

    if shape == "Cylinder":
        cyl_r = st.slider("Radius r [m]", 0.01, 1.0, 0.05, 0.01, key="cyl_r")
        cyl_L = st.slider("Length L [m]", 0.1, 10.0, 2.0, 0.1, key="cyl_L")
        cyl_nc = st.slider("Circumferential divs", 6, 40, 16, 2, key="cyl_nc")
        cyl_nl = st.slider("Longitudinal divs", 2, 20, 8, 1, key="cyl_nl")
    else:
        cyl_r, cyl_L, cyl_nc, cyl_nl = 0.05, 2.0, 16, 8

    if shape == "Box / Cuboid":
        box_lx = st.slider("Length Lx [m]", 0.1, 5.0, 1.0, 0.1, key="box_lx")
        box_ly = st.slider("Width Ly [m]", 0.1, 5.0, 0.5, 0.1, key="box_ly")
        box_lz = st.slider("Height Lz [m]", 0.1, 5.0, 0.3, 0.1, key="box_lz")
        box_nx = st.slider("Divs X", 1, 10, 4, 1, key="box_nx")
        box_ny = st.slider("Divs Y", 1, 10, 3, 1, key="box_ny")
        box_nz = st.slider("Divs Z", 1, 10, 2, 1, key="box_nz")
    else:
        box_lx, box_ly, box_lz = 1.0, 0.5, 0.3
        box_nx, box_ny, box_nz = 4, 3, 2

    if shape == "L-Bracket":
        lb_l1 = st.slider("Horiz. leg length [m]", 0.1, 3.0, 1.0, 0.1, key="lb_l1")
        lb_l2 = st.slider("Vert. leg length [m]", 0.1, 3.0, 0.8, 0.1, key="lb_l2")
        lb_t  = st.slider("Thickness [m]", 0.01, 0.5, 0.1, 0.01, key="lb_t")
        lb_n  = st.slider("Mesh density", 2, 12, 5, 1, key="lb_n")
    else:
        lb_l1, lb_l2, lb_t, lb_n = 1.0, 0.8, 0.1, 5

    if shape == "Plate (2D)":
        pl_w  = st.slider("Width [m]", 0.1, 5.0, 1.0, 0.1, key="pl_w")
        pl_h  = st.slider("Height [m]", 0.1, 5.0, 0.5, 0.1, key="pl_h")
        pl_nx = st.slider("Divs X", 2, 20, 8, 1, key="pl_nx")
        pl_ny = st.slider("Divs Y", 2, 20, 4, 1, key="pl_ny")
    else:
        pl_w, pl_h, pl_nx, pl_ny = 1.0, 0.5, 8, 4

    # ── Material ──────────────────────────────────────────────────────────
    st.subheader("🧱 Material")
    mat_names = list_materials()
    mat_name = st.selectbox("Preset", mat_names, key="mat_name")
    mat_data = get_material(mat_name)

    E_val  = st.number_input("Young's Modulus E [Pa]", value=float(mat_data["E"]),
                              format="%.3e", key="E_val")
    nu_val = st.slider("Poisson's ratio ν", 0.01, 0.49, float(mat_data["nu"]),
                       0.01, key="nu_val")
    rho_val = st.number_input("Density ρ [kg/m³]", value=float(mat_data["rho"]),
                               format="%.1f", key="rho_val")
    yield_strength = mat_data["yield_strength"]

    # ── Boundary Conditions ───────────────────────────────────────────────
    st.subheader("🔒 Boundary Conditions")
    bc_type = st.selectbox(
        "BC type",
        ["Fixed-Free", "Simply Supported", "Fixed-Fixed", "Free"],
        key="bc_type",
    )

    # ── Loads ─────────────────────────────────────────────────────────────
    st.subheader("⬇️ Loads")
    load_type = st.selectbox(
        "Load type",
        ["Point Force", "Distributed Load", "Couple/Moment", "Pressure"],
        key="load_type",
    )
    load_pos = st.slider("Load position (fraction of L)", 0.0, 1.0, 1.0, 0.05,
                         key="load_pos")

    Fx_val = Fz_val = 0.0
    if load_type == "Point Force":
        Fy_val = st.number_input("Force Fy [N]", value=-1000.0, format="%.1f",
                                 key="Fy_val")
        Fx_val = st.number_input("Force Fx [N]", value=0.0, format="%.1f",
                                 key="Fx_val")
        Fz_val = st.number_input("Force Fz [N]", value=0.0, format="%.1f",
                                 key="Fz_val")
        load_value = Fy_val

    elif load_type == "Distributed Load":
        q_val = st.number_input("Distributed load q [N/m]", value=-500.0,
                                format="%.1f", key="q_val")
        load_value = q_val

    elif load_type == "Couple/Moment":
        Mx_val = st.number_input("Mx [N·m]", value=0.0, format="%.1f", key="Mx_val")
        My_val = st.number_input("My [N·m]", value=0.0, format="%.1f", key="My_val")
        Mz_val = st.number_input("Mz [N·m]", value=500.0, format="%.1f", key="Mz_val")
        load_value = Mz_val

    elif load_type == "Pressure":
        p_val = st.number_input("Pressure p [Pa]", value=1e5, format="%.2e",
                                key="p_val")
        load_value = p_val

    else:
        load_value = 0.0

    # ── Solver settings ───────────────────────────────────────────────────
    st.subheader("🖥️ Solver")
    result_type = st.selectbox(
        "Result to display",
        ["Von Mises", "Displacement", "σxx", "σyy", "σzz", "τxy"],
        key="result_type",
    )
    scale_factor = st.slider("Deformation scale factor", 1, 500, 100, 10,
                             key="scale_factor")

    # ── Mesh refinement ───────────────────────────────────────────────────
    st.subheader("🔲 Mesh")
    mesh_ref = st.selectbox("Mesh refinement", ["Coarse", "Medium", "Fine"],
                            key="mesh_ref")
    mesh_mult = {"Coarse": 1, "Medium": 2, "Fine": 4}[mesh_ref]

    # ── Run button ────────────────────────────────────────────────────────
    st.divider()
    solve_btn = st.button("▶️ Run FEM Analysis", type="primary", key="solve_btn")


# ---------------------------------------------------------------------------
# Helper: build mesh for currently selected shape
# ---------------------------------------------------------------------------

def _build_mesh():
    if shape == "Rectangular Beam":
        n_div = beam_n * mesh_mult
        return make_beam_mesh(beam_L, beam_w, beam_h, n_div)
    elif shape == "Cylinder":
        return make_cylinder_mesh(cyl_r, cyl_L, cyl_nc * mesh_mult, cyl_nl * mesh_mult)
    elif shape == "Box / Cuboid":
        return make_box_mesh(box_lx, box_ly, box_lz,
                             box_nx * mesh_mult, box_ny * mesh_mult, box_nz * mesh_mult)
    elif shape == "L-Bracket":
        return make_lbracket_mesh(lb_l1, lb_l2, lb_t, lb_n * mesh_mult)
    elif shape == "Plate (2D)":
        return make_plate_mesh(pl_w, pl_h, pl_nx * mesh_mult, pl_ny * mesh_mult)
    return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)


# ---------------------------------------------------------------------------
# Helper: Plotly Mesh3d figure
# ---------------------------------------------------------------------------

def _mesh3d_fig(verts, faces, intensity=None, title="Mesh", colorbar_title=""):
    if len(verts) == 0 or len(faces) == 0:
        return go.Figure()
    kwargs = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        flatshading=True,
        opacity=0.9,
    )
    if intensity is not None:
        kwargs["intensity"] = intensity
        kwargs["colorscale"] = "Jet"
        kwargs["colorbar"] = dict(title=colorbar_title)
    else:
        kwargs["color"] = "steelblue"
    fig = go.Figure(go.Mesh3d(**kwargs))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def _add_force_arrows(fig, verts, load_type_str, load_val, load_pos_frac, shape_name):
    """Overlay force/moment arrows using go.Cone traces."""
    if len(verts) == 0:
        return fig

    x_range = verts[:, 0].max() - verts[:, 0].min()
    x_mid = verts[:, 0].min() + x_range * load_pos_frac
    y_mid = (verts[:, 1].max() + verts[:, 1].min()) / 2.0
    z_top = verts[:, 2].max()

    u_dir, v_dir, w_dir = 0.0, 0.0, -1.0
    if load_type_str == "Point Force":
        w_dir = 1.0 if load_val >= 0 else -1.0
    elif load_type_str == "Distributed Load":
        w_dir = 1.0 if load_val >= 0 else -1.0
    elif load_type_str in ("Couple/Moment",):
        u_dir, v_dir, w_dir = 0.0, 1.0, 0.0

    cone_size = max(x_range * 0.12, 0.02)
    fig.add_trace(
        go.Cone(
            x=[x_mid],
            y=[y_mid],
            z=[z_top + cone_size],
            u=[u_dir * cone_size],
            v=[v_dir * cone_size],
            w=[w_dir * cone_size],
            sizemode="absolute",
            sizeref=cone_size,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            name=load_type_str,
        )
    )
    return fig


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_geom, tab_results, tab_report = st.tabs(
    ["🧊 Geometry & Mesh", "📊 FEM Results", "📋 Report"]
)

# ── Tab 1: Geometry & Mesh ─────────────────────────────────────────────────
with tab_geom:
    st.subheader("Geometry & Mesh Preview")

    stl_file = st.file_uploader("Upload custom STL model (optional)", type=["stl"],
                                key="stl_uploader")
    if stl_file is not None:
        stl_bytes = stl_file.read()
        verts_stl, faces_stl = load_stl_mesh(stl_bytes)
        if len(verts_stl) > 0:
            fig_stl = _mesh3d_fig(verts_stl, faces_stl, title="Custom STL Model")
            st.plotly_chart(fig_stl, use_container_width=True)
        else:
            st.warning("Could not parse the uploaded STL file.")

    verts, faces = _build_mesh()
    fig_geom = _mesh3d_fig(verts, faces, title=f"{shape} — Mesh")
    fig_geom = _add_force_arrows(fig_geom, verts, load_type, load_value, load_pos, shape)
    st.plotly_chart(fig_geom, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Vertices", len(verts))
    col2.metric("Triangles", len(faces))
    col3.metric("Mesh refinement", mesh_ref)

# ── Tab 2: FEM Results ────────────────────────────────────────────────────
with tab_results:
    st.subheader("FEM Results")

    # Default result values (shown in Report tab even before solving)
    vm_max   = 0.0
    max_disp = 0.0
    sf_val   = float("inf")

    if not solve_btn:
        st.info("👈 Configure parameters in the sidebar and click **▶️ Run FEM Analysis**.")
    else:
        with st.spinner("Running FEM analysis…"):
            verts, faces = _build_mesh()

            # ── 1-D Beam Solver ──────────────────────────────────────────
            if shape == "Rectangular Beam":
                I_beam = beam_w * beam_h**3 / 12.0
                n_elem = beam_n * mesh_mult

                try:
                    U, le = build_beam_fem(
                        L=beam_L,
                        n=n_elem,
                        E=E_val,
                        I=I_beam,
                        load_type=load_type,
                        load_value=load_value,
                        load_pos=load_pos,
                        bc_type=bc_type,
                    )
                    sigma = compute_beam_stress(U, n_elem, le, E_val, beam_h)
                    max_stress = float(np.max(np.abs(sigma)))
                    vm_max = max_stress

                    v_nodes = U[0::2]
                    max_disp = float(np.max(np.abs(v_nodes)))
                    sf_val = safety_factor(vm_max, yield_strength)

                    n_face_verts = len(verts)
                    intensity_vert = np.zeros(n_face_verts)
                    for e in range(n_elem):
                        for lv in range(4):
                            vi = 4 * e + lv
                            if vi < n_face_verts:
                                intensity_vert[vi] = sigma[e]

                    verts_def = verts.copy()
                    for e in range(n_elem):
                        for local_cs in range(2):
                            node_idx = e + local_cs
                            v_disp = v_nodes[node_idx] * scale_factor
                            base = 4 * (e + local_cs)
                            if base + 3 < len(verts_def):
                                verts_def[base:base+4, 1] += v_disp

                    if result_type == "Displacement":
                        intensity_vert = np.zeros(n_face_verts)
                        for e in range(n_elem):
                            avg_d = (abs(v_nodes[e]) + abs(v_nodes[e+1])) / 2.0
                            for lv in range(4):
                                vi = 4 * e + lv
                                if vi < n_face_verts:
                                    intensity_vert[vi] = avg_d

                    fig_res = _mesh3d_fig(
                        verts_def, faces,
                        intensity=intensity_vert,
                        title=f"{result_type} — Deformed (scale ×{scale_factor})",
                        colorbar_title=result_type,
                    )
                    fig_res = _add_force_arrows(fig_res, verts, load_type,
                                               load_value, load_pos, shape)
                    st.plotly_chart(fig_res, use_container_width=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Max |σ_bending|", f"{max_stress/1e6:.2f} MPa")
                    c2.metric("Max displacement", f"{max_disp*1e3:.3f} mm")
                    c3.metric("Von Mises (max)", f"{vm_max/1e6:.2f} MPa")
                    c4.metric("Safety factor", f"{sf_val:.2f}")

                except Exception as exc:
                    st.error(f"Solver error: {exc}")

            # ── 2-D Plate Solver ──────────────────────────────────────────
            elif shape == "Plate (2D)":
                nx_m = pl_nx * mesh_mult
                ny_m = pl_ny * mesh_mult
                plate_verts, plate_faces = make_plate_mesh(pl_w, pl_h, nx_m, ny_m)
                nodes_2d = plate_verts[:, :2]
                n_nodes_2d = len(nodes_2d)

                fixed_dofs = []
                for ni in range(n_nodes_2d):
                    if nodes_2d[ni, 0] < 1e-9:
                        fixed_dofs.extend([2*ni, 2*ni+1])

                right_nodes = [ni for ni in range(n_nodes_2d)
                               if nodes_2d[ni, 0] > pl_w - 1e-9]
                forces_dict = {}
                if right_nodes:
                    f_per_node = load_value / len(right_nodes)
                    for ni in right_nodes:
                        forces_dict[ni] = [0.0, f_per_node]

                try:
                    U2, _ = build_plate_fem(
                        nodes=nodes_2d,
                        elements=plate_faces,
                        E=E_val,
                        nu=nu_val,
                        thickness=0.01,
                        forces_dict=forces_dict,
                        fixed_dofs=fixed_dofs,
                    )
                    sxx, syy, txy, vm = compute_stress_2d(U2, nodes_2d, plate_faces,
                                                          E_val, nu_val)
                    vm_max = float(np.max(np.abs(vm)))
                    max_disp = float(np.max(np.abs(U2)))
                    sf_val = safety_factor(vm_max, yield_strength)

                    n_verts_p = len(plate_verts)
                    intensity_v = np.zeros(n_verts_p)
                    count_v = np.zeros(n_verts_p)
                    for ei, elem in enumerate(plate_faces):
                        for ni in elem:
                            intensity_v[ni] += vm[ei]
                            count_v[ni] += 1
                    mask = count_v > 0
                    intensity_v[mask] /= count_v[mask]

                    if result_type == "Displacement":
                        intensity_v = displacement_magnitude(U2)
                    elif result_type == "σxx":
                        intensity_v = np.zeros(n_verts_p)
                        for ei, elem in enumerate(plate_faces):
                            for ni in elem:
                                intensity_v[ni] += sxx[ei]
                        intensity_v[mask] /= count_v[mask]

                    verts_3d = np.column_stack([nodes_2d, np.zeros(n_verts_p)])
                    fig_res = _mesh3d_fig(
                        verts_3d, plate_faces,
                        intensity=intensity_v,
                        title=f"{result_type} — Plate",
                        colorbar_title=result_type,
                    )
                    fig_res = _add_force_arrows(fig_res, verts_3d, load_type,
                                               load_value, load_pos, shape)
                    st.plotly_chart(fig_res, use_container_width=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Max σxx", f"{float(np.max(np.abs(sxx)))/1e6:.2f} MPa")
                    c2.metric("Max displacement", f"{max_disp*1e3:.3f} mm")
                    c3.metric("Von Mises (max)", f"{vm_max/1e6:.2f} MPa")
                    c4.metric("Safety factor", f"{sf_val:.2f}")

                except Exception as exc:
                    st.error(f"Solver error: {exc}")

            # ── Demo visualization for other 3-D shapes ────────────────────
            else:
                st.info(
                    "ℹ️ Full 3-D FEM solver for this shape is not yet implemented. "
                    "Showing a demo visualization with randomized stress values."
                )
                verts, faces = _build_mesh()
                if len(verts) > 0:
                    rng = np.random.default_rng(42)
                    demo_intensity = rng.uniform(0, 250e6, len(verts))
                    vm_max = float(demo_intensity.max())
                    sf_val = safety_factor(vm_max, yield_strength)

                    fig_demo = _mesh3d_fig(
                        verts, faces,
                        intensity=demo_intensity,
                        title=f"Demo — {shape} ({result_type})",
                        colorbar_title="Von Mises [Pa]",
                    )
                    fig_demo = _add_force_arrows(
                        fig_demo, verts, load_type, load_value, load_pos, shape
                    )
                    st.plotly_chart(fig_demo, use_container_width=True)

                    c1, c2 = st.columns(2)
                    c1.metric("Demo Von Mises (max)", f"{vm_max/1e6:.1f} MPa")
                    c2.metric("Safety factor", f"{sf_val:.2f}")

# ── Tab 3: Report ─────────────────────────────────────────────────────────
with tab_report:
    st.subheader("Analysis Report")

    params_rows = [
        ("Shape", shape),
        ("Material", mat_name),
        ("Young's Modulus E", f"{E_val:.3e} Pa"),
        ("Poisson's ratio ν", f"{nu_val:.2f}"),
        ("Density ρ", f"{rho_val:.0f} kg/m³"),
        ("Yield Strength", f"{yield_strength/1e6:.0f} MPa"),
        ("Boundary Condition", bc_type),
        ("Load Type", load_type),
        ("Load Position", f"{load_pos*100:.0f}% of L"),
        ("Load Value", f"{load_value:.3g}"),
        ("Mesh Refinement", mesh_ref),
    ]

    if shape == "Rectangular Beam":
        params_rows += [
            ("Length L", f"{beam_L} m"),
            ("Width w", f"{beam_w} m"),
            ("Height h", f"{beam_h} m"),
            ("N elements", str(beam_n * mesh_mult)),
        ]
    elif shape == "Plate (2D)":
        params_rows += [
            ("Width", f"{pl_w} m"),
            ("Height", f"{pl_h} m"),
            ("Nx", str(pl_nx * mesh_mult)),
            ("Ny", str(pl_ny * mesh_mult)),
        ]

    df_params = pd.DataFrame(params_rows, columns=["Parameter", "Value"])
    st.table(df_params)

    if solve_btn:
        st.markdown("### Key Results")
        results_rows = [
            ("Max Von Mises Stress", f"{vm_max/1e6:.3f} MPa"),
            ("Max Displacement", f"{max_disp*1e3:.4f} mm"),
            ("Safety Factor", f"{sf_val:.3f}" if sf_val != float("inf") else "∞"),
        ]
        st.table(pd.DataFrame(results_rows, columns=["Result", "Value"]))
