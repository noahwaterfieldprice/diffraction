"""Unit tests for Crystal and Site classes.

All tests use real Crystal, DirectLattice, and Site instances.
No mocks are used anywhere in this file.
"""

from pathlib import Path

import pytest
from numpy import array, ndarray
from numpy.testing import assert_array_almost_equal, assert_array_equal

from diffraction import Crystal, Site

# Path to the real calcite CIF file used for from_cif tests.
CALCITE_CIF_PATH = (
    Path(__file__).parent.parent
    / "functional"
    / "static"
    / "valid_cifs"
    / "calcite_icsd.cif"
)

CALCITE_PARAMS = [4.99, 4.99, 17.002, 90.0, 90.0, 120.0]
CALCITE_SPACE_GROUP = "R -3 c H"

CALCITE_SITES = {
    "Ca1": ("Ca2+", [0, 0, 0]),
    "C1": ("C4+", [0, 0, 0.25]),
    "O1": ("O2-", [0.25706, 0, 0.25]),
}


# ---------------------------------------------------------------------------
# TestCreatingSites
# ---------------------------------------------------------------------------


class TestCreatingSites:
    def test_creating_atom(self):
        atom1 = Site("Ca2+", [0, 0, 0])

        assert atom1.ion == "Ca2+"
        assert_array_almost_equal(atom1.position, array([0, 0, 0]))
        assert isinstance(atom1.position, ndarray)

    def test_atom_representation(self):
        atom1 = Site("Ca2+", [0, 0, 0])

        assert repr(atom1) == f"Site({atom1.ion!r}, {atom1.position!r})"
        assert str(atom1) == f"Site({atom1.ion!r}, {atom1.position!r})"

    def test_atom_equivalence(self):
        atom_1 = Site("Ca2+", [0, 0, 0])
        atom_2 = Site("Ca2+", [0, 0, 0])
        atom_3 = Site("Ca3+", [0, 0, 0])
        atom_4 = Site("Ca2+", [0, 0, 1])

        assert atom_1 == atom_2
        assert atom_1 != atom_3
        assert atom_1 != atom_4

    def test_atom_positions_are_equivalent_up_to_set_precision(self):
        atom_1 = Site("Ca2+", [0, 0, 0])
        atom_2 = Site("Ca2+", [0, 0, 1e-6])
        atom_3 = Site("Ca2+", [0, 0, 1e-5])

        # default precision should make atom_1 and atom_2 equal
        assert atom_1 == atom_2
        # atom_1 and atom_3 should only be equal after lowering the precision
        assert atom_1 != atom_3
        atom_1.precision = 1e-5
        assert atom_1 == atom_3


# ---------------------------------------------------------------------------
# TestCreatingCrystalFromSequence
# ---------------------------------------------------------------------------


class TestCreatingCrystalFromSequence:
    def test_crystal_stores_lattice_parameters(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)

        assert c.a == pytest.approx(4.99)
        assert c.b == pytest.approx(4.99)
        assert c.c == pytest.approx(17.002)
        assert c.alpha == pytest.approx(90.0)
        assert c.beta == pytest.approx(90.0)
        assert c.gamma == pytest.approx(120.0)

    def test_crystal_stores_space_group(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)

        assert c.space_group == "R -3 c H"

    def test_crystal_repr_includes_space_group(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)

        assert "R -3 c H" in repr(c)
        assert "Crystal(" in repr(c)


# ---------------------------------------------------------------------------
# TestCreatingCrystalFromDict
# ---------------------------------------------------------------------------


class TestCreatingCrystalFromDict:
    def test_crystal_from_dict_assigns_parameters_and_space_group(self):
        params = {
            "a": 4.99,
            "b": 4.99,
            "c": 17.002,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 120.0,
            "space_group": "R -3 c H",
        }
        c = Crystal.from_dict(params)

        assert c.a == pytest.approx(4.99)
        assert c.b == pytest.approx(4.99)
        assert c.c == pytest.approx(17.002)
        assert c.alpha == pytest.approx(90.0)
        assert c.beta == pytest.approx(90.0)
        assert c.gamma == pytest.approx(120.0)
        assert c.space_group == "R -3 c H"

    def test_crystal_from_dict_raises_if_space_group_missing(self):
        params = {
            "a": 4.99,
            "b": 4.99,
            "c": 17.002,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 120.0,
        }
        with pytest.raises(ValueError, match="space_group"):
            Crystal.from_dict(params)

    def test_crystal_from_dict_loads_sites_when_provided(self):
        params = {
            "a": 4.99,
            "b": 4.99,
            "c": 17.002,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 120.0,
            "space_group": "R -3 c H",
            "sites": CALCITE_SITES,
        }
        c = Crystal.from_dict(params)

        expected_sites = {
            name: Site(ion, position) for name, (ion, position) in CALCITE_SITES.items()
        }
        assert c.sites == expected_sites


# ---------------------------------------------------------------------------
# TestCreatingCrystalFromCIF
# ---------------------------------------------------------------------------


class TestCreatingCrystalFromCIF:
    def test_crystal_from_cif_loads_calcite(self):
        c = Crystal.from_cif(str(CALCITE_CIF_PATH), load_sites=False)

        assert c.a == pytest.approx(4.99, abs=0.01)
        assert c.b == pytest.approx(4.99, abs=0.01)
        assert c.c == pytest.approx(17.002, abs=0.01)
        assert c.alpha == pytest.approx(90.0)
        assert c.beta == pytest.approx(90.0)
        assert c.gamma == pytest.approx(120.0)
        assert c.space_group == "R -3 c H"

    def test_crystal_from_cif_loads_atomic_sites(self):
        c = Crystal.from_cif(str(CALCITE_CIF_PATH))

        assert len(c.sites) > 0
        # Calcite has Ca1, C1, O1 sites
        assert "Ca1" in c.sites
        assert "C1" in c.sites
        assert "O1" in c.sites


# ---------------------------------------------------------------------------
# TestAddingAndModifyingAtomicSites
# ---------------------------------------------------------------------------


class TestAddingAndModifyingAtomicSites:
    def test_adding_single_site_to_crystal(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)
        c.add_sites({"Ca1": ("Ca2+", [0, 0, 0])})

        assert "Ca1" in c.sites
        assert c.sites["Ca1"] == Site("Ca2+", [0, 0, 0])

    def test_adding_multiple_sites_to_crystal(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)
        c.add_sites(CALCITE_SITES)

        expected_sites = {
            name: Site(ion, position) for name, (ion, position) in CALCITE_SITES.items()
        }
        assert c.sites == expected_sites

    def test_modifying_site_position(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)
        c.add_sites({"Ca1": ("Ca2+", [0, 0, 0])})

        c.sites["Ca1"].position = [0, 0, 0.5]
        assert_array_equal(c.sites["Ca1"].position, array([0, 0, 0.5]))
        assert isinstance(c.sites["Ca1"].position, ndarray)


# ---------------------------------------------------------------------------
# TestCrystalEdgeCases
# ---------------------------------------------------------------------------


class TestCrystalEdgeCases:
    def test_crystal_getattr_delegates_lattice_params(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)

        # All six lattice parameters should be accessible directly on Crystal
        assert isinstance(c.a, float)
        assert isinstance(c.b, float)
        assert isinstance(c.c, float)
        assert isinstance(c.alpha, float)
        assert isinstance(c.beta, float)
        assert isinstance(c.gamma, float)

    def test_crystal_without_sites_has_empty_sites_dict(self):
        c = Crystal(CALCITE_PARAMS, CALCITE_SPACE_GROUP)

        assert c.sites == {}
