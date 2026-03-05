"""Material library for FEM Lab."""

from __future__ import annotations

from typing import Optional, TypedDict


class MaterialProps(TypedDict):
    E: float               # Young's modulus (Pa)
    nu: float              # Poisson's ratio (–)
    yield_strength: float  # Yield strength (Pa)
    density: float         # Density (kg/m³)


MATERIALS: dict[str, Optional[MaterialProps]] = {
    "Steel (S235)": {
        "E": 210e9,
        "nu": 0.30,
        "yield_strength": 235e6,
        "density": 7850,
    },
    "Aluminium 6061-T6": {
        "E": 69e9,
        "nu": 0.33,
        "yield_strength": 276e6,
        "density": 2700,
    },
    "Titanium Ti-6Al-4V": {
        "E": 114e9,
        "nu": 0.34,
        "yield_strength": 880e6,
        "density": 4430,
    },
    "CFRP (UD 0°)": {
        "E": 135e9,
        "nu": 0.27,
        "yield_strength": 600e6,
        "density": 1600,
    },
    "Concrete C30/37": {
        "E": 33e9,
        "nu": 0.20,
        "yield_strength": 30e6,
        "density": 2400,
    },
    "Custom": None,
}
