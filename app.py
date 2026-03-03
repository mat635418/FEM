import streamlit as st
import numpy as np
import pyvista as pv

# --- CRITICAL: Must be set before any PyVista/VTK calls ---
pv.OFF_SCREEN = True

# --- Page Config ---
st.set_page_config(page_title="Python FEM Lab", layout="wide")

# --- Sidebar UI ---
with st.sidebar:
    st.title("🏗️ FEM Lab Settings")

    st.header("1. Geometry Selection")
    shape_type = st.selectbox(
        "Select Solid Template",
        ["Cantilever Beam", "Hollow Cylinder", "L-Bracket"]
    )

    st.header("2. Physical Parameters")
    if shape_type == "Cantilever Beam":
        L = st.slider("Length (m)", 1.0, 10.0, 5.0)
        W = st.slider("Width (m)", 0.1, 1.0, 0.3)
        H = st.slider("Height (m)", 0.1, 1.0, 0.5)
        mesh = pv.Box(bounds=(0, L, -W/2, W/2, -H/2, H/2), level=1)

    elif shape_type == "Hollow Cylinder":
        rad = st.slider("Outer Radius", 0.5, 2.0, 1.0)
        inner_rad = st.slider("Inner Radius", 0.1, 0.4, 0.2)
        height = st.slider("Height", 1.0, 5.0, 3.0)
        # CylinderStructured returns a StructuredGrid; cast to Unstructured for warp support
        mesh = pv.CylinderStructured(
            radius=[inner_rad, rad],
            height=height,
            theta_resolution=20
        ).cast_to_unstructured_grid()

    elif shape_type == "L-Bracket":
        mesh = pv.Box(bounds=(0, 2, 0, 0.5, 0, 2))
        st.info("L-Bracket template uses a standard structural mesh.")

    st.divider()

    st.header("3. Material & Load")
    E = st.number_input("Young's Modulus (GPa)", value=210) * 1e9  # Steel default
    force_val = st.slider("Vertical Force (kN)", -100, 100, -50) * 1000
    warp_factor = st.slider("Exaggerate Deformation", 1, 1000, 100)


# --- FEM Logic (Simplified Solver) ---
def run_simulation(mesh, force_magnitude, youngs_modulus):
    """
    Simulates stress by calculating a displacement vector
    based on the vertical Z-load applied to the 'tip' of the mesh.
    Returns displacement (Nx3 array) and stress (N,) scalar array.
    """
    x_coords = mesh.points[:, 0]

    # Simplified displacement field: delta_z ~ (F * x^2) / (2 * E * I)
    displacement = np.zeros((mesh.n_points, 3), dtype=float)
    displacement[:, 2] = (force_magnitude * (x_coords ** 2)) / (youngs_modulus * 1e-4)

    # Von Mises Stress proxy (proportional to curvature / derivative of displacement)
    stress = np.abs(displacement[:, 2]) * (youngs_modulus / 1e11)

    return displacement, stress


# --- Main App ---
st.title(f"Visualizing Stress: {shape_type}")

col1, col2 = st.columns([3, 1])

with col1:
    # Run Solver
    disp, stress = run_simulation(mesh, force_val, E)

    # --- FIX: Store displacement vectors in mesh point_data BEFORE warping ---
    mesh.point_data["Displacement"] = disp        # Nx3 vector array
    mesh.point_data["Stress (Pa)"] = stress       # N scalar array

    # --- FIX: warp_by_vector references the named vector field ---
    warped_mesh = mesh.warp_by_vector("Displacement", factor=warp_factor)

    # Setup PyVista Plotter (off_screen required for Streamlit / headless)
    plotter = pv.Plotter(window_size=[800, 600], off_screen=True)
    plotter.add_mesh(
        warped_mesh,
        scalars="Stress (Pa)",
        cmap="jet",
        show_edges=True
    )
    plotter.add_scalar_bar(title="Von Mises Stress Proxy")
    plotter.background_color = "white"
    plotter.view_isometric()

    # Render to image and display in Streamlit
    screenshot = plotter.screenshot(return_img=True)
    st.image(screenshot, use_container_width=True)

with col2:
    st.metric("Max Displacement", f"{np.max(np.abs(disp[:, 2])):.4f} m")
    st.metric("Peak Stress", f"{np.max(stress):.2e} Pa")

    st.write("### Simulation Notes")
    st.caption("""
    The visualization shows the mesh in a **deformed state**.
    The red areas indicate high stress concentration where
    structural failure is most likely to occur.
    """)

st.success("Simulation Complete. Adjust sliders to see real-time updates!")
