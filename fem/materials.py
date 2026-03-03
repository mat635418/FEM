MATERIALS = {
    "Steel":    {"E": 210e9, "nu": 0.30, "rho": 7850, "yield_strength": 250e6},
    "Aluminum": {"E":  70e9, "nu": 0.33, "rho": 2700, "yield_strength": 270e6},
    "Titanium": {"E": 116e9, "nu": 0.32, "rho": 4500, "yield_strength": 880e6},
    "Copper":   {"E": 110e9, "nu": 0.34, "rho": 8960, "yield_strength": 210e6},
    "Custom":   {"E": 200e9, "nu": 0.30, "rho": 7800, "yield_strength": 250e6},
}


def get_material(name: str) -> dict:
    """Return material property dict for *name*."""
    if name not in MATERIALS:
        raise KeyError(f"Unknown material '{name}'. Available: {list(MATERIALS.keys())}")
    return MATERIALS[name].copy()


def list_materials() -> list:
    """Return list of available material names."""
    return list(MATERIALS.keys())
