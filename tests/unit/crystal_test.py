from collections import OrderedDict
import random
import re
import string

import pytest

from diffraction.cif import (CIF_NAMES, CIF_NUMERICAL, cif_numerical, CIF_TEXTUAL,
                             cif_textual, NUMERICAL_DATA_NAMES, TEXTUAL_DATA_NAMES)
from diffraction.crystal import (Crystal, load_data_block, cif_numerical, cif_textual,
                                 LATTICE_PARAMETERS)
from diffraction.lattice import DirectLattice
from cif_test import fake_cif_data

# Data names and parameters for testing
CRYSTAL_PARAMETERS = LATTICE_PARAMETERS + ["space_group"]
CRYSTAL_DATA_NAMES = [CIF_NAMES[parameter] for parameter in CRYSTAL_PARAMETERS]

# Example data for testing
CALCITE_DATA = OrderedDict([("a", 4.99), ("b", 4.99), ("c", 17.002),
                            ("alpha", 90), ("beta", 90), ("gamma", 120),
                            ("space_group", "R -3 c H")])
CALCITE_CIF = OrderedDict([("cell_length_a", "4.9900(2)"),
                           ("cell_length_b", "4.9900(2)"),
                           ("cell_length_c", "17.002(1)"),
                           ("cell_angle_alpha", "90."),
                           ("cell_angle_beta", "90."),
                           ("cell_angle_gamma", "90."),
                           ("symmetry_space_group_name_H-M", "'R -3 c H'")])


class TestCreatingCrystalFromSequence:  # TODO: add test to check space group validity
    def test_error_if_lattice_parameter_missing_from_input_list(self):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        lattice_parameters_missing_one = lattice_parameters[:5]

        with pytest.raises(ValueError) as exception_info:
            c = Crystal(lattice_parameters_missing_one, space_group)
        assert str(exception_info.value) == "Missing lattice parameter from input"

    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    def test_error_if_invalid_lattice_parameter_given(self, invalid_value):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        invalid_lattice_parameters = lattice_parameters[:]
        invalid_lattice_parameters[0] = invalid_value

        with pytest.raises(ValueError) as exception_info:
            c = Crystal(invalid_lattice_parameters, space_group)
        assert str(exception_info.value) == \
            "Invalid lattice parameter a: {}".format(invalid_value)

    def test_lattice_parameters_stored_lattice_object(self, mocker):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        mock_lattice = mocker.Mock(spec=DirectLattice)
        m = mocker.patch("diffraction.crystal.DirectLattice", return_value=mock_lattice)
        c = Crystal(lattice_parameters, space_group)

        # test lattice parameters are passed DirectLattice object
        m.assert_called_once_with(lattice_parameters)
        assert isinstance(c.lattice, DirectLattice)

    def test_lattice_parameters_are_directly_available_from_crystal(self, mocker):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        # mock DirectLattice to return mock with lattice parameter attributes
        mock_lattice = mocker.Mock(spec=DirectLattice)
        for parameter, value in zip(LATTICE_PARAMETERS, lattice_parameters):
            setattr(mock_lattice, parameter, float(value))
        m = mocker.patch("diffraction.crystal.DirectLattice", return_value=mock_lattice)

        c = Crystal(lattice_parameters, space_group)
        for parameter in LATTICE_PARAMETERS:
            assert getattr(c, parameter) == CALCITE_DATA[parameter]
            assert isinstance(getattr(c, parameter), float)

    def test_string_representation_of_crystal(self):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        c = Crystal(lattice_parameters, space_group)

        assert repr(c) == "Crystal({0}, '{1}')".format(
            [float(parameter) for parameter in lattice_parameters], space_group)
        assert str(c) == "Crystal({0}, '{1}')".format(
            [float(parameter) for parameter in lattice_parameters], space_group)


class TestCreatingCrystalFromMapping:
    @pytest.mark.parametrize("missing_parameter", LATTICE_PARAMETERS + ["space_group"])
    def test_error_if_parameter_missing_from_dict(self, missing_parameter):
        dict_with_missing_parameter = CALCITE_DATA.copy()
        del dict_with_missing_parameter[missing_parameter]

        with pytest.raises(ValueError) as exception_info:
            Crystal.from_dict(dict_with_missing_parameter)
        assert str(exception_info.value) == \
            "Parameter: '{}' missing from input dictionary".format(missing_parameter)

    def test_parameters_are_assigned_with_correct_type(self):
        c = Crystal.from_dict(CALCITE_DATA)

        for parameter in LATTICE_PARAMETERS:
            assert getattr(c, parameter) == CALCITE_DATA[parameter]
            assert isinstance(getattr(c, parameter), float)
        assert getattr(c, "space_group") == CALCITE_DATA["space_group"]
        assert isinstance(getattr(c, "space_group"), str)


class TestCreatingCrystalFromCIF: # TODO: add multi data block test
    @pytest.mark.parametrize("missing_data_name", CALCITE_CIF.keys())
    def test_error_if_parameter_is_missing_from_cif(self, missing_data_name, mocker):
        data_items_with_missing_item = CALCITE_CIF.copy()
        data_items_with_missing_item.pop(missing_data_name)
        mocker.patch("diffraction.crystal.load_data_block",
                     return_value=data_items_with_missing_item)

        with pytest.raises(ValueError) as exception_info:
            Crystal.from_cif("some/cif/file.cif")
        assert str(exception_info.value) == \
            "Parameter: '{}' missing from input CIF".format(missing_data_name)

    def test_parameters_assigned_for_single_data_block(self, mocker):
        mocker.patch("diffraction.crystal.load_data_block", return_value=CALCITE_CIF)
        c = Crystal.from_cif("some/single/data/block/cif")

        # test numerical data is assigned with the correct type
        for data_name, parameter in zip(CALCITE_CIF.keys(), CALCITE_DATA.keys()):
            if data_name in NUMERICAL_DATA_NAMES:
                value = float(CIF_NUMERICAL.match(CALCITE_CIF[data_name]).group(1))
                assert getattr(c, parameter) == value
                assert isinstance(getattr(c, parameter), float)
            elif data_name in TEXTUAL_DATA_NAMES:
                value = CIF_TEXTUAL.match(CALCITE_CIF[data_name]).group(1)
                assert getattr(c, parameter) == value
                assert isinstance(getattr(c, parameter), str)
