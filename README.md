# FEM Analysis App

A complete, self-contained **Finite Element Method (FEM) analysis application** built with Python and Streamlit. No external FEM library is required — all solvers are implemented from scratch using NumPy and SciPy.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## Features

### Shapes
- Rectangular Beam (1-D Euler-Bernoulli FEM solver)
- Plate / 2-D Panel (2-D plane-stress CST FEM solver)
- Cylinder (demo visualization)
- Box / Cuboid (demo visualization)
- L-Bracket (demo visualization)
- Custom STL upload

### Load Types
- Point Force (Fx, Fy, Fz)
- Distributed Load (q)
- Couple / Moment (Mx, My, Mz)
- Pressure

### Materials
- Steel, Aluminum, Titanium, Copper, Custom
- User-editable E, ν, ρ

### Boundary Conditions
- Fixed-Free (Cantilever)
- Simply Supported
- Fixed-Fixed
- Free

### Result Types
- Von Mises stress
- Displacement magnitude
- σxx, σyy, σzz, τxy

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Streamlit Cloud Deploy

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to `app.py`.
3. Streamlit Cloud will automatically install `requirements.txt` and `packages.txt`.

---

## Project Structure

```
FEM/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages (Streamlit Cloud)
├── README.md
└── fem/
    ├── __init__.py         # Package init
    ├── materials.py        # Material library (Steel, Aluminum, etc.)
    ├── mesh.py             # Mesh generators (beam, plate, cylinder, box, L-bracket, STL)
    ├── postprocess.py      # Post-processing utilities (Von Mises, safety factor, etc.)
    ├── solver_1d.py        # 1-D Euler-Bernoulli beam FEM solver
    └── solver_2d.py        # 2-D plane-stress CST FEM solver
```

---

## Screenshots

_Add screenshots here after deployment._
