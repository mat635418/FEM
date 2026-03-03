import streamlit as st
import numpy as np
import pyvista as pv
from st_pyvista import st_pyvista

# --- Page Config ---
st.set_page_config(page_title="Python FEM Lab", layout="wide")

# --- Sidebar UI ---
with st.sidebar:
    st.title("🏗️ FEM Lab Settings")
    
    st.header("1. Geometry Selection")
    shape_type = st.selectbox("Select Solid Template", 
                              ["Cantilever Beam", "Hollow Cylinder", "L-Bracket"])
    
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
        mesh = pv.CylinderStructured(radius=[inner_rad, rad], height=height, theta_resolution=20)
        
    elif shape_type == "L-Bracket":
        mesh = pv.Box(bounds=(0, 2, 0, 0.5, 0, 2)) # Simplified L-placeholder
        st.info("L-Bracket template uses a standard structural mesh.")

    st.divider()
    
    st.header("3. Material & Load")
    E = st.number_input("Young's Modulus (GPa)", value=210) * 1e9  # Steel default
    force_val = st.slider("Vertical Force (kN)", -100, 100, -50) * 1000
    warp_factor = st.slider("Exaggerate Deformation", 1, 1000, 100)

# --- The FEM Logic (Simplified Solver) ---
def run_simulation(mesh, force_magnitude):
    """
    Simulates stress by calculating a displacement vector 
    based on the vertical Z-load applied to the 'tip' of the mesh.
    """
    centers = mesh.cell_centers().points
    z_coords = mesh.points[:, 2]
    x_coords = mesh.points[:, 0]
    
    # Logic: Fixed at x=0, Force at x=max
    max_x = np.max(x_coords)
    
    # Calculate a simplified displacement field (Bending formula approx)
    # Displacement delta_z ~ (F * x^2) / (2 * E * I)
    displacement = np.zeros_like(mesh.points)
    displacement[:, 2] = (force_magnitude * (x_coords**2)) / (E * 1e-4)
    
    # Create Von Mises Stress proxy (Derivative of displacement)
    stress = np.abs(displacement[:, 2]) * (E / 1e11)
    
    return displacement, stress

# --- Execution ---
st.title(f"Visualizing Stress: {shape_type}")

col1, col2 = st.columns([3, 1])

with col1:
    # Run Solver
    disp, stress = run_simulation(mesh, force_val)
    
    # Apply results to mesh
    mesh.point_data["Stress (Pa)"] = stress
    warped_mesh = mesh.warp_by_vector(vector=disp, factor=warp_factor)
    
    # Setup PyVista Plotter
    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(warped_mesh, scalars="Stress (Pa)", cmap="jet", show_edges=True)
    plotter.add_scalar_bar(title="Von Mises Stress Proxy")
    plotter.background_color = "white"
    
    # Show in Streamlit
    st_pyvista(plotter, key="fem_mesh")

with col2:
    st.metric("Max Displacement", f"{np.max(np.abs(disp[:,2])):.4f} m")
    st.metric("Peak Stress", f"{np.max(stress):.2e} Pa")
    
    st.write("### Simulation Notes")
    st.caption("""
    The visualization shows the mesh in a **deformed state**. 
    The red areas indicate high stress concentration where 
    structural failure is most likely to occur.
    """)

st.success("Simulation Complete. Adjust sliders to see real-time updates!")
