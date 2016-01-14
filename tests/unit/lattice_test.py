from collections import OrderedDict

import pytest

from diffraction.cif import CIF_NAMES, CIF_NUMERICAL
from diffraction.lattice import DirectLattice, LATTICE_PARAMETERS
from cif_test import fake_cif_data

CALCITE_LATTICE = OrderedDict([("a", 4.99), ("b", 4.99), ("c", 17.002),
                               ("alpha", 90), ("beta", 90), ("gamma", 120)])
CALCITE_CIF = OrderedDict([("cell_length_a", "4.9900(2)"),
                           ("cell_length_b", "4.9900(2)"),
                           ("cell_length_c", "17.002(1)"),
                           ("cell_angle_alpha", "90."),
                           ("cell_angle_beta", "90."),
                           ("cell_angle_gamma", "90.")])


class TestCreatingDirectLatticeFromSequence:

    def test_error_if_lattice_parameter_missing_from_input_list(self):
        lattice_parameters = list(CALCITE_LATTICE.values())
        lattice_parameters_missing_one = lattice_parameters[:5]

        with pytest.raises(ValueError) as exception_info:
            l = DirectLattice(lattice_parameters_missing_one)
        assert str(exception_info.value) == "Missing lattice parameter from input"

    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    def test_error_if_invalid_lattice_parameter_given(self, invalid_value):
        invalid_lattice_parameters = list(CALCITE_LATTICE.values())
        invalid_lattice_parameters[0] = invalid_value

        with pytest.raises(ValueError) as exception_info:
            l = DirectLattice(invalid_lattice_parameters)
        assert str(exception_info.value) == \
            "Invalid lattice parameter a: {}".format(invalid_value)

    def test_parameters_are_assigned_with_correct_type(self):
        lattice_parameters = CALCITE_LATTICE.values()
        l = DirectLattice(lattice_parameters)

        # test lattice parameters are assigned as floats
        for parameter, value in CALCITE_LATTICE.items():
            assert getattr(l, parameter) == value
            assert isinstance(getattr(l, parameter), float)

    def test_string_representation_of_crystal(self):
        lattice_parameters = CALCITE_LATTICE.values()
        l = DirectLattice(lattice_parameters)

        assert repr(l) == "DirectLattice({0})".format(
            [float(parameter) for parameter in lattice_parameters])
        assert str(l) == "DirectLattice({0})".format(
            [float(parameter) for parameter in lattice_parameters])


class TestCreatingDirectLatticeFromMapping:
    @pytest.mark.parametrize("missing_parameter", LATTICE_PARAMETERS)
    def test_error_if_parameter_missing_from_dict(self, missing_parameter):
        dict_with_missing_parameter = CALCITE_LATTICE.copy()
        del dict_with_missing_parameter[missing_parameter]

        with pytest.raises(ValueError) as exception_info:
            DirectLattice.from_dict(dict_with_missing_parameter)
        assert str(exception_info.value) == \
            "Lattice parameter {} missing from input dictionary".format(missing_parameter)

    def test_parameters_are_assigned_with_correct_type(self):
        c = DirectLattice.from_dict(CALCITE_LATTICE)

        for parameter in LATTICE_PARAMETERS:
            assert getattr(c, parameter) == CALCITE_LATTICE[parameter]
            assert isinstance(getattr(c, parameter), float)


class TestCreatingDirectLatticeFromCIF: # add multiple data block test, DRY! with crystal tests
    @pytest.mark.parametrize("missing_data_name", CALCITE_CIF.keys())
    def test_error_if_parameter_is_missing_from_cif(self, missing_data_name, mocker):
        data_items_with_missing_item = CALCITE_CIF.copy()
        data_items_with_missing_item.pop(missing_data_name)
        mocker.patch("diffraction.lattice.load_data_block",
                     return_value=data_items_with_missing_item)

        with pytest.raises(ValueError) as exception_info:
            DirectLattice.from_cif("some/cif/file")
        assert str(exception_info.value) == \
            "Lattice parameter {} missing from input CIF".format(missing_data_name)

    def test_parameters_assigned_for_single_data_block(self, mocker):
        mocker.patch("diffraction.lattice.load_data_block", return_value=CALCITE_CIF)
        l = DirectLattice.from_cif("some/single/data/block/cif")

        # test numerical data is assigned with the correct type
        for data_name, parameter in zip(CALCITE_CIF.keys(), CALCITE_LATTICE.keys()):
            value = float(CIF_NUMERICAL.match(CALCITE_CIF[data_name]).group(1))
            assert getattr(l, parameter) == value
            assert isinstance(getattr(l, parameter), float)
