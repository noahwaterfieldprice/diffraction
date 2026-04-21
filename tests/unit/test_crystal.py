"""Unit tests for Crystal and Site classes.

All tests use real Crystal, DirectLattice, and Site instances.
No mocks are used anywhere in this file.
"""

from collections.abc import Sequence

import numpy as np
import pytest
from conftest import CALCITE_LATTICE_PARAMS  # type: ignore[import-not-found]
from numpy import array, ndarray
from numpy.testing import assert_array_almost_equal, assert_array_equal

from diffraction import Crystal, Site, SpaceGroup

CALCITE_SPACE_GROUP = "R -3 c H"

CALCITE_SITES: dict[str, tuple[str, Sequence[float]]] = {
    "Ca1": ("Ca2+", [0, 0, 0]),
    "C1": ("C4+", [0, 0, 0.25]),
    "O1": ("O2-", [0.25706, 0, 0.25]),
}


# ---------------------------------------------------------------------------
# TestCreatingSites
# ---------------------------------------------------------------------------


class TestCreatingSites:
    def test_creating_atom(self) -> None:
        atom1 = Site("Ca2+", [0, 0, 0])

        assert atom1.ion == "Ca2+"
        assert_array_almost_equal(atom1.position, array([0, 0, 0]))
        assert isinstance(atom1.position, ndarray)  # type: ignore[unreachable]

    def test_atom_representation(self) -> None:
        atom1 = Site("Ca2+", [0, 0, 0])

        assert repr(atom1) == f"Site({atom1.ion!r}, {atom1.position!r})"
        assert str(atom1) == f"Site({atom1.ion!r}, {atom1.position!r})"

    def test_atom_equivalence(self) -> None:
        atom_1 = Site("Ca2+", [0, 0, 0])
        atom_2 = Site("Ca2+", [0, 0, 0])
        atom_3 = Site("Ca3+", [0, 0, 0])
        atom_4 = Site("Ca2+", [0, 0, 1])

        assert atom_1 == atom_2
        assert atom_1 != atom_3
        assert atom_1 != atom_4

    def test_atom_positions_are_equivalent_up_to_set_precision(self) -> None:
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
    def test_crystal_stores_lattice_parameters(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)

        assert c.a == pytest.approx(4.99)
        assert c.b == pytest.approx(4.99)
        assert c.c == pytest.approx(17.002)
        assert c.alpha == pytest.approx(90.0)
        assert c.beta == pytest.approx(90.0)
        assert c.gamma == pytest.approx(120.0)

    def test_crystal_stores_space_group(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)

        assert c.space_group == "R -3 c H"

    def test_crystal_repr_includes_space_group(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)

        assert "R -3 c H" in repr(c)
        assert "Crystal(" in repr(c)


# ---------------------------------------------------------------------------
# TestCreatingCrystalFromDict
# ---------------------------------------------------------------------------


class TestCreatingCrystalFromDict:
    def test_crystal_from_dict_assigns_parameters_and_space_group(self) -> None:
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

    def test_crystal_from_dict_raises_if_space_group_missing(self) -> None:
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

    def test_crystal_from_dict_loads_sites_when_provided(self) -> None:
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
    def test_crystal_from_cif_loads_calcite(self, calcite_cif_path: object) -> None:
        c = Crystal.from_cif(str(calcite_cif_path), load_sites=False)

        assert c.a == pytest.approx(4.99, abs=0.01)
        assert c.b == pytest.approx(4.99, abs=0.01)
        assert c.c == pytest.approx(17.002, abs=0.01)
        assert c.alpha == pytest.approx(90.0)
        assert c.beta == pytest.approx(90.0)
        assert c.gamma == pytest.approx(120.0)
        assert c.space_group == "R -3 c H"
        assert not hasattr(c, "sites")

    def test_crystal_from_cif_loads_atomic_sites(
        self, calcite_cif_path: object
    ) -> None:
        c = Crystal.from_cif(str(calcite_cif_path))

        assert len(c.sites) > 0
        # Calcite has Ca1, C1, O1 sites
        assert "Ca1" in c.sites
        assert "C1" in c.sites
        assert "O1" in c.sites


# ---------------------------------------------------------------------------
# TestAddingAndModifyingAtomicSites
# ---------------------------------------------------------------------------


class TestAddingAndModifyingAtomicSites:
    def test_adding_single_site_to_crystal(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)
        c.add_sites({"Ca1": ("Ca2+", [0, 0, 0])})

        assert "Ca1" in c.sites
        assert c.sites["Ca1"] == Site("Ca2+", [0, 0, 0])

    def test_adding_multiple_sites_to_crystal(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)
        c.add_sites(CALCITE_SITES)

        expected_sites = {
            name: Site(ion, position) for name, (ion, position) in CALCITE_SITES.items()
        }
        assert c.sites == expected_sites

    def test_modifying_site_position(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)
        c.add_sites({"Ca1": ("Ca2+", [0, 0, 0])})

        c.sites["Ca1"].position = [0, 0, 0.5]
        assert_array_equal(c.sites["Ca1"].position, array([0, 0, 0.5]))
        assert isinstance(c.sites["Ca1"].position, ndarray)  # type: ignore[unreachable]


# ---------------------------------------------------------------------------
# TestCrystalEdgeCases
# ---------------------------------------------------------------------------


class TestCrystalEdgeCases:
    def test_crystal_getattr_delegates_lattice_params(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)

        # All six lattice parameters should be accessible directly on Crystal
        assert isinstance(c.a, float)
        assert isinstance(c.b, float)
        assert isinstance(c.c, float)
        assert isinstance(c.alpha, float)
        assert isinstance(c.beta, float)
        assert isinstance(c.gamma, float)

    def test_crystal_getattr_raises_for_non_lattice_param(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)

        with pytest.raises(AttributeError, match="no attribute"):
            _ = c.metric  # no longer delegated via __getattr__

    def test_crystal_without_sites_has_empty_sites_dict(self) -> None:
        c = Crystal(CALCITE_LATTICE_PARAMS, CALCITE_SPACE_GROUP)

        assert c.sites == {}


# ---------------------------------------------------------------------------
# TestExpandSites
# ---------------------------------------------------------------------------

NACL_LATTICE_PARAMS = [5.64, 5.64, 5.64, 90, 90, 90]
NACL_SPACE_GROUP = "Fm-3m"


class TestExpandSites:
    def _nacl_crystal(self) -> Crystal:
        """NaCl asymmetric unit: Na at (0,0,0), Cl at (0.5,0.5,0.5)."""
        crystal = Crystal(NACL_LATTICE_PARAMS, NACL_SPACE_GROUP)
        crystal.add_sites(
            {
                "Na1": ("Na", [0, 0, 0]),
                "Cl1": ("Cl", [0.5, 0.5, 0.5]),
            }
        )
        return crystal

    def test_expand_nacl_gives_8_sites(self) -> None:
        crystal = self._nacl_crystal()
        sg = SpaceGroup("Fm-3m")

        expanded = crystal.expand_sites(sg)

        assert len(expanded) == 8
        na_sites = [s for s in expanded if s.ion == "Na"]
        cl_sites = [s for s in expanded if s.ion == "Cl"]
        assert len(na_sites) == 4
        assert len(cl_sites) == 4

    def test_expand_nacl_positions_wrapped(self) -> None:
        crystal = self._nacl_crystal()
        sg = SpaceGroup("Fm-3m")

        expanded = crystal.expand_sites(sg)

        for site in expanded:
            for coord in site.position:
                assert 0.0 <= coord < 1.0

    def test_expand_nacl_na_positions(self) -> None:
        crystal = self._nacl_crystal()
        sg = SpaceGroup("Fm-3m")

        expanded = crystal.expand_sites(sg)
        na_positions = [s.position for s in expanded if s.ion == "Na"]

        expected = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]
        # Order-independent comparison
        for exp_pos in expected:
            assert any(
                np.allclose(exp_pos, pos, atol=1e-6) for pos in na_positions
            ), f"Expected Na position {exp_pos} not found in {na_positions}"

    def test_expand_p1_identity(self) -> None:
        crystal = Crystal([4.0, 5.0, 6.0, 90, 90, 90], "P1")
        crystal.add_sites({"A1": ("Fe", [0.1, 0.2, 0.3])})
        sg = SpaceGroup("P1")

        expanded = crystal.expand_sites(sg)

        assert len(expanded) == 1
        assert expanded[0].ion == "Fe"
        assert np.allclose(expanded[0].position, [0.1, 0.2, 0.3], atol=1e-6)

    def test_expand_deduplicates_overlapping_sites(self) -> None:
        # NaCl Fm-3m has 48 operators * 4 centering vectors = 192 raw
        # positions per site, but deduplication yields 4 unique per atom.
        crystal = self._nacl_crystal()
        sg = SpaceGroup("Fm-3m")

        expanded = crystal.expand_sites(sg)

        # Verify no duplicate positions for the same ion
        na_positions = [s.position for s in expanded if s.ion == "Na"]
        for i, pos_i in enumerate(na_positions):
            for j, pos_j in enumerate(na_positions):
                if i != j:
                    assert not np.allclose(pos_i, pos_j, atol=1e-6), (
                        f"Duplicate Na positions found at indices {i} and {j}"
                    )

    def test_expand_returns_list_of_sites(self) -> None:
        crystal = self._nacl_crystal()
        sg = SpaceGroup("Fm-3m")

        expanded = crystal.expand_sites(sg)

        assert isinstance(expanded, list)
        for site in expanded:
            assert isinstance(site, Site)
            assert isinstance(site.ion, str)
            assert isinstance(site.position, np.ndarray)  # type: ignore[unreachable]

    def test_expand_inversion_center(self) -> None:
        # P-1 (SG 2) has identity + inversion
        crystal = Crystal([4.0, 5.0, 6.0, 90, 90, 90], "P-1")
        crystal.add_sites({"A1": ("O", [0.1, 0.2, 0.3])})
        sg = SpaceGroup(number=2)

        expanded = crystal.expand_sites(sg)

        assert len(expanded) == 2
        positions = [s.position for s in expanded]
        expected_pos1 = np.array([0.1, 0.2, 0.3])
        expected_pos2 = np.array([0.9, 0.8, 0.7])  # -0.1 % 1, etc.
        assert any(np.allclose(expected_pos1, p, atol=1e-6) for p in positions)
        assert any(np.allclose(expected_pos2, p, atol=1e-6) for p in positions)

    def test_expand_sites_centering_vectors_include_identity(self) -> None:
        # Both P and F centering must include ["0","0","0"] as the trivial vector
        sg_p = SpaceGroup("P1")
        sg_f = SpaceGroup("Fm-3m")

        trivial = ["0", "0", "0"]
        assert trivial in sg_p.centering_vectors
        assert trivial in sg_f.centering_vectors

    def test_expand_sites_raises_valueerror_for_missing_trivial_centering(
        self,
    ) -> None:
        crystal = Crystal([4.0, 5.0, 6.0, 90, 90, 90], "P1")
        crystal.add_sites({"A1": ("Fe", [0.1, 0.2, 0.3])})
        sg = SpaceGroup("P1")
        # SpaceGroup is a frozen dataclass; bypass freeze to simulate corrupt data.
        object.__setattr__(sg, "centering_vectors", [])

        with pytest.raises(ValueError, match="centering_vectors"):
            crystal.expand_sites(sg)
