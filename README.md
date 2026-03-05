# 🔬 FEM Lab

[![CI](https://github.com/mat635418/FEM/actions/workflows/ci.yml/badge.svg)](https://github.com/mat635418/FEM/actions/workflows/ci.yml)

A production-quality **Finite Element Method** web application built with Streamlit and [scikit-fem](https://scikit-fem.readthedocs.io).

---

## What it does

FEM Lab solves 2-D plane-stress structural problems on three cross-section types:

| Template | Description |
|---|---|
| **Cantilever Beam** | Rectangular beam, fixed at the left face, loaded at the right |
| **Hollow Section** | Annular cross-section (tube) |
| **L-Bracket** | L-shaped bracket with characteristic re-entrant corner |

Results displayed:

- Von Mises stress field (colour map)
- Deformed shape overlay (with adjustable scale factor)
- Peak stress, max displacement, safety factor, strain energy
- Export as CSV, JSON problem definition, or PNG screenshot

---

## Physics background

### Plane stress

For thin-walled structures under in-plane loads, the through-thickness stress
components are negligible. The plane-stress constitutive relation is:

```
σ_xx = E/(1–ν²) · (ε_xx + ν·ε_yy)
σ_yy = E/(1–ν²) · (ε_yy + ν·ε_xx)
σ_xy = E/(2(1+ν)) · γ_xy
```

### Von Mises criterion

Material yielding is predicted when the Von Mises equivalent stress exceeds the
uniaxial yield strength:

```
σ_VM = √(σ_xx² – σ_xx·σ_yy + σ_yy² + 3·σ_xy²)
```

A **safety factor** SF = σ_y / max(σ_VM) > 1 means no yield is predicted.

---

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Project structure

```
FEM/
├── app.py                  ← Streamlit UI (thin orchestrator)
├── fem/
│   ├── solver.py           ← Real FEM: assemble K, apply BCs, solve
│   ├── mesh_gen.py         ← 2-D triangular mesh per geometry
│   ├── postprocess.py      ← Von Mises, safety factor, strain energy
│   └── materials.py        ← Material library
├── tests/
│   ├── test_solver.py      ← Cantilever + patch tests
│   ├── test_mesh_gen.py    ← Mesh smoke tests
│   └── test_postprocess.py ← Stress computation unit tests
├── examples/               ← Example problem JSON files
├── requirements.txt        ← Pinned production dependencies
├── requirements-dev.txt    ← Dev tools (pytest, ruff, mypy)
└── pyproject.toml
```

---

## Dependency versions

| Package | Version |
|---|---|
| streamlit | 1.43.0 |
| pyvista | 0.44.2 |
| numpy | 1.26.4 |
| scikit-fem | 10.0.2 |
| scipy | 1.13.1 |
| pandas | 2.2.3 |
| meshio | 5.3.5 |
