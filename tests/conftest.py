"""Shared test fixtures and constants for all tests.

Provides module-level constants for use in parametrize decorators and
pytest fixtures for commonly used mineral lattice and crystal objects.

Crystal system coverage:
- Calcite (CaCO3): trigonal, hexagonal setting
- NaCl: cubic
- Corundum (Al2O3): trigonal, hexagonal setting
- Forsterite (Mg2SiO4): orthorhombic
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from diffraction import Crystal, DirectLattice

# ---------------------------------------------------------------------------
# Mineral lattice parameter constants
# Use these directly in @pytest.mark.parametrize — they are not fixtures.
# ---------------------------------------------------------------------------

CALCITE_LATTICE_PARAMS = (4.99, 4.99, 17.002, 90.0, 90.0, 120.0)
NACL_LATTICE_PARAMS = (5.6402, 5.6402, 5.6402, 90.0, 90.0, 90.0)
CORUNDUM_LATTICE_PARAMS = (4.758, 4.758, 12.991, 90.0, 90.0, 120.0)
FORSTERITE_LATTICE_PARAMS = (4.758, 10.225, 5.994, 90.0, 90.0, 90.0)

# CIF data dict for calcite with correct gamma=120 (the ICSD CIF value).
# Keys are CIF data names as used by cif_helpers.get_cif_data.
CALCITE_CIF_DATA = OrderedDict(
    [
        ("cell_length_a", "4.9900(2)"),
        ("cell_length_b", "4.9900(2)"),
        ("cell_length_c", "17.002(1)"),
        ("cell_angle_alpha", "90."),
        ("cell_angle_beta", "90."),
        ("cell_angle_gamma", "120."),
    ]
)

# Precomputed 3x3 direct metric tensor for calcite.
# Used in vector tests that need a concrete metric without a real lattice.
CALCITE_DIRECT_METRIC = np.array(
    [[24.9001, -12.45005, 0], [-12.45005, 24.9001, 0], [0, 0, 289.068004]]
)

# ---------------------------------------------------------------------------
# CIF file paths
# ---------------------------------------------------------------------------

VALID_CIFS_DIR = Path(__file__).parent / "functional" / "static" / "valid_cifs"
CALCITE_CIF_PATH = VALID_CIFS_DIR / "calcite_icsd.cif"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calcite_cif_path() -> Path:
    """Return the path to the calcite ICSD CIF file."""
    return CALCITE_CIF_PATH


@pytest.fixture
def calcite_lattice() -> DirectLattice:
    """Return a real DirectLattice for calcite (trigonal, hexagonal setting)."""
    return DirectLattice(CALCITE_LATTICE_PARAMS)


@pytest.fixture
def nacl_lattice() -> DirectLattice:
    """Return a real DirectLattice for NaCl (cubic)."""
    return DirectLattice(NACL_LATTICE_PARAMS)


@pytest.fixture
def calcite_crystal() -> Crystal:
    """Return a real Crystal for calcite with space group R -3 c H."""
    return Crystal(list(CALCITE_LATTICE_PARAMS), "R -3 c H")
