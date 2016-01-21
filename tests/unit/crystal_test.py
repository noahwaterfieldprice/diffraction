from collections import OrderedDict

from numpy import array, ndarray
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from diffraction.cif.helpers import (CIF_NAMES, NUMERICAL_DATA_VALUE,
                                     NUMERICAL_DATA_NAMES, TEXTUAL_DATA_NAMES)
from diffraction.crystal import Site, Crystal
from diffraction.lattice import DirectLattice, LATTICE_PARAMETER_KEYS

# Data names and parameters for testing
CRYSTAL_PARAMETERS = LATTICE_PARAMETER_KEYS + ["space_group"]
CRYSTAL_DATA_NAMES = [CIF_NAMES[parameter] for parameter in CRYSTAL_PARAMETERS]

# Example data for testing
CALCITE_DATA = OrderedDict([
    ("a", 4.99), ("b", 4.99), ("c", 17.002),
    ("alpha", 90), ("beta", 90), ("gamma", 120),
    ("space_group", "R -3 c H")
])
CALCITE_CIF = OrderedDict([
    ("cell_length_a", "4.9900(2)"),
    ("cell_length_b", "4.9900(2)"),
    ("cell_length_c", "17.002(1)"),
    ("cell_angle_alpha", "90."),
    ("cell_angle_beta", "90."),
    ("cell_angle_gamma", "90."),
    ("symmetry_space_group_name_H-M", "R -3 c H"),
    ("atom_site_label", ["Ca1", "C1", "O1"]),
    ("atom_site_type_symbol", ["Ca2+", "C4+", "O2-"]),
    ("atom_site_symmetry_multiplicity", ["6", "6", "18"]),
    ("atom_site_Wyckoff_symbol", ["b", "a", "e"]),
    ("atom_site_fract_x", ["0", "0", "0.25706(33)"]),
    ("atom_site_fract_y", ["0", "0", "0"]),
    ("atom_site_fract_z", ["0", "0.25", "0.25"]),
    ("atom_site_B_iso_or_equiv", [".", ".", "."]),
    ("atom_site_occupancy", ["1.", "1.", "1."]),
    ("atom_site_attached_hydrogens", ["0", "0", "0"]),
])
CALCITE_ATOMIC_SITES = OrderedDict([
    ("Ca1", ["Ca2+", [0, 0, 0]]),
    ("C1", ["C4+", [0, 0, 0.25]]),
    ("O1", ["O2-", [0.25706, 0, 0.25]])
])


class TestCreatingCrystalFromSequence:  # TODO: add test to check space group validity
    def test_lattice_parameters_stored_lattice_object(self, mocker):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        mock_lattice = mocker.Mock(spec=DirectLattice)
        m = mocker.patch("diffraction.crystal.DirectLattice", return_value=mock_lattice)
        c = Crystal(lattice_parameters, space_group)

        # test lattice parameters are passed DirectLattice object
        m.assert_called_once_with(lattice_parameters)
        assert c.lattice == mock_lattice

    def test_lattice_parameters_are_directly_available_from_crystal(self, mocker):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        # mock DirectLattice to return mock with lattice parameter attributes
        mock_lattice = mocker.Mock(spec=DirectLattice)
        for key, value in zip(LATTICE_PARAMETER_KEYS, lattice_parameters):
            setattr(mock_lattice, key, float(value))
        mocker.patch("diffraction.crystal.DirectLattice", return_value=mock_lattice)

        c = Crystal(lattice_parameters, space_group)
        for key in LATTICE_PARAMETER_KEYS:
            assert getattr(c, key) == CALCITE_DATA[key]
            assert isinstance(getattr(c, key), float)

    def test_string_representation_of_crystal(self):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        c = Crystal(lattice_parameters, space_group)

        assert repr(c) == "Crystal({0}, '{1}')".format(
            [float(parameter) for parameter in lattice_parameters], space_group)
        assert str(c) == "Crystal({0}, '{1}')".format(
            [float(parameter) for parameter in lattice_parameters], space_group)


class TestCreatingCrystalFromMapping:
    def test_lattice_parameters_and_space_group_are_assigned(self, mocker):
        mock_lattice = mocker.Mock(spec=DirectLattice)
        mocker.patch("diffraction.crystal.DirectLattice.from_dict",
                     return_value=mock_lattice)

        c = Crystal.from_dict(CALCITE_DATA)
        assert c.space_group == CALCITE_DATA["space_group"]
        assert c.lattice == mock_lattice

    def test_error_if_space_group_not_given(self, mocker):
        dict_with_space_group_missing = CALCITE_CIF.copy()
        dict_with_space_group_missing.pop("symmetry_space_group_name_H-M")
        mocker.patch("diffraction.crystal.DirectLattice")

        with pytest.raises(ValueError) as exception_info:
            Crystal.from_dict(dict_with_space_group_missing)
        assert str(exception_info.value) == ("Parameter: 'space_group' "
                                             "missing from input dictionary")

    def test_loading_atomic_sites_from_dict(self, mocker):
        mocker.patch("diffraction.crystal.DirectLattice.from_dict")
        calcite_data_with_sites = CALCITE_DATA.copy()
        calcite_data_with_sites["sites"] = CALCITE_ATOMIC_SITES

        c = Crystal.from_dict(calcite_data_with_sites)
        expected_sites = {name: Site(element, position)
                          for name, (element, position) in CALCITE_ATOMIC_SITES.items()}
        assert c.sites == expected_sites


class TestCreatingCrystalFromCIF:
    def test_lattice_parameters_and_space_group_are_assigned(self, mocker):
        mocker.patch("diffraction.crystal.load_data_block", return_value=CALCITE_CIF)
        mock_lattice = mocker.Mock(spec=DirectLattice)
        m = mocker.patch("diffraction.crystal.DirectLattice.from_cif",
                         return_value=mock_lattice)

        c = Crystal.from_cif("some/cif/file.cif", load_sites=False)
        m.assert_called_once_with("some/cif/file.cif", None)
        assert c.lattice == mock_lattice
        assert c.space_group == "R -3 c H"

    def test_loading_atomic_sites_from_cif(self, mocker):
        mocker.patch("diffraction.crystal.load_data_block", return_value=CALCITE_CIF)
        mock = mocker.MagicMock()
        mock.add_sites_from_cif = Crystal.add_sites_from_cif
        mock.sites = {}

        mock.add_sites_from_cif(mock, "some/cif/file.cif")
        expected_sites = {name: Site(element, position)
                          for name, (element, position) in CALCITE_ATOMIC_SITES.items()}
        assert mock.sites == expected_sites


class TestAddingAndModifyingAtomicSites:
    def test_creating_atom(self):
        atom1 = Site("Ca2+", [0, 0, 0])

        assert atom1.element == "Ca2+"
        assert_array_almost_equal(atom1.position, array([0, 0, 0]))
        assert isinstance(atom1.position, ndarray)

    def test_atom_representation(self):
        atom1 = Site("Ca2+", [0, 0, 0])

        assert repr(atom1) == "Site({0.element!r}, {0.position!r})".format(atom1)
        assert str(atom1) == "Site({0.element!r}, {0.position!r})".format(atom1)

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
        atom_2 = Site("Ca2+", [0, 0, 1E-6])
        atom_3 = Site("Ca2+", [0, 0, 1E-5])

        # default precision should make atom_1 and atom_2 equal
        assert atom_1 == atom_2
        # atom_1 and atom_3 should only be equal after lowering the precision
        assert atom_1 != atom_3
        atom_1.precision = 1E-5
        assert atom_1 == atom_3

    def test_adding_single_sites(self, mocker):
        mock = mocker.MagicMock()
        mock.add_sites = Crystal.add_sites
        mock.sites = {}

        mock.add_sites(mock, {"Ca1": ["Ca2+", [0, 0, 0]]})
        assert mock.sites == {"Ca1": Site("Ca2+", [0, 0, 0])}
        mock.add_sites(mock, {"C1": ["C4+", [0, 0, 0.25]]})
        assert mock.sites == {"Ca1": Site("Ca2+", [0, 0, 0]),
                              "C1": Site("C4+", [0, 0, 0.25])}

    def test_adding_multiple_sites(self, mocker):
        mock = mocker.MagicMock()
        mock.add_sites = Crystal.add_sites

        mock.sites = {}
        mock.add_sites(mock, CALCITE_ATOMIC_SITES)
        expected_sites = {name: Site(element, position)
                          for name, (element, position) in CALCITE_ATOMIC_SITES.items()}
        assert mock.sites == expected_sites

    def test_modifying_atom_position(self, mocker):
        mock = mocker.MagicMock()
        mock.add_sites = Crystal.add_sites

        mock.sites = {}
        mock.add_sites(mock, CALCITE_ATOMIC_SITES)
        mock.sites["Ca1"].position = [0, 0, 0.5]
        assert_array_equal(mock.sites["Ca1"].position, array([0, 0, 0.5]))
        assert isinstance(mock.sites["Ca1"].position, ndarray)
