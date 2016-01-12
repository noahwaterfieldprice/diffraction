import collections
import random
import re
import string

import pytest

from diffraction import Crystal
from diffraction.crystal import (
    LATTICE_PARAMETERS, CIF_NAMES, CIF_NUMERICAL, CIF_TEXTUAL,
    load_data_block, get_input_value, cif_numerical, cif_textual)

# # Numerical and textual data names and attributes for testing
NUMERICAL_ATTRIBUTES = LATTICE_PARAMETERS
TEXTUAL_ATTRIBUTES = ["space_group"]
NUMERICAL_DATA_NAMES = [CIF_NAMES[key] for key in NUMERICAL_ATTRIBUTES]
TEXTUAL_DATA_NAMES = [CIF_NAMES[key] for key in TEXTUAL_ATTRIBUTES]
DATA_NAMES = NUMERICAL_DATA_NAMES + TEXTUAL_DATA_NAMES

# ATTRIBUTES = list(NUMERICAL_ATTRIBUTES) + list(TEXTUAL_ATTRIBUTES)

# Example data for testing
CALCITE_DATA = collections.OrderedDict([("a", 4.99), ("b", 4.99), ("c", 17.002),
                                       ("alpha", 90), ("beta", 90), ("gamma", 120),
                                       ("space_group", "R -3 c H")])


def fake_num_data(data_names, errors=False, no_data_blocks=1):
    """Generates dummy numerical input cif data for testing"""
    if errors:
        data_values = ["{0}.{0}({0})".format(i)
                       for i in range(len(data_names))]
    else:
        data_values = ["{0}.{0}".format(i) for i in range(len(data_names))]
    return data_values


def fake_text_data(data_names, no_data_blocks=1):
    """Generates dummy textual input cif data for testing"""
    data_values = ["'{0}'".format(string.ascii_letters[i] * 5)
                   for i in range(len(data_names))]
    return data_values


def fake_cif_data(data_names, errors=False, no_data_blocks=1):
    num_data_names = [name for name in data_names
                      if name in NUMERICAL_DATA_NAMES]
    text_data_names = [name for name in data_names
                       if name in TEXTUAL_DATA_NAMES]
    data_values = fake_num_data(num_data_names, errors, no_data_blocks) + \
        fake_text_data(text_data_names, no_data_blocks)

    data = {}
    for i in range(no_data_blocks):
        data_items = dict(zip(num_data_names + text_data_names, data_values))
        data["data_block_{}".format(i)] = data_items
    return data


class TestCreatingCrystalFromSequence:
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

    def test_parameters_are_assigned_with_correct_type(self):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        c = Crystal(lattice_parameters, space_group)

        # test lattice parameters are assigned as floats
        for parameter in LATTICE_PARAMETERS:
            assert getattr(c, parameter) == CALCITE_DATA[parameter]
            assert isinstance(getattr(c, parameter), float)
        # test space group assigned as string
        assert getattr(c, "space_group") == CALCITE_DATA["space_group"]
        assert isinstance(getattr(c, "space_group"), str)

    def test_string_representation_of_crystal(self):
        *lattice_parameters, space_group = CALCITE_DATA.values()
        c = Crystal(lattice_parameters, space_group)

        assert repr(c) == "Crystal({0}, '{1}')".format(
            [float(parameter) for parameter in lattice_parameters], space_group)


class TestCreatingCrystalFromMapping:
    @pytest.mark.parametrize("missing_parameter", LATTICE_PARAMETERS + ["space_group"])
    def test_error_if_parameter_missing_from_dict(self, missing_parameter):
        dict_with_missing_parameter = CALCITE_DATA.copy()
        del dict_with_missing_parameter[missing_parameter]

        with pytest.raises(ValueError) as exception_info:
            get_input_value(missing_parameter, dict_with_missing_parameter, "dictionary")
        assert str(exception_info.value) == \
            "{} missing from input dictionary".format(missing_parameter)

    def test_parameters_are_assigned_with_correct_type(self):
        c = Crystal.from_dict(CALCITE_DATA)

        for parameter in LATTICE_PARAMETERS:
            assert getattr(c, parameter) == CALCITE_DATA[parameter]
            assert isinstance(getattr(c, parameter), float)
        assert getattr(c, "space_group") == CALCITE_DATA["space_group"]
        assert isinstance(getattr(c, "space_group"), str)


class TestLoadingDataFromCIF:
    def test_single_datablock_loaded_automatically(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        data_items = load_data_block("single/data/block/cif")
        assert data_items == input_dict["data_block_0"]

    def test_error_if_data_block_not_given_for_multi_data_blocks(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        with pytest.raises(TypeError) as exception_info:
            load_data_block("multi/data/block/cif")
        assert str(exception_info.value) == \
            ("__init__() missing keyword argument: 'data_block'. "
             "Required when input CIF has multiple data blocks.")

    def test_data_block_loads_for_multi_data_blocks(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        assert load_data_block("multi/data/block/cif", "data_block_0") == \
            input_dict["data_block_0"]


class TestCreatingCrystalFromCIF:
    @pytest.mark.parametrize("missing_data_name", DATA_NAMES)
    def test_error_if_parameter_is_missing_from_cif(self, missing_data_name):
        data_names_with_missing_name = DATA_NAMES[:]
        data_names_with_missing_name.remove(missing_data_name)
        data_items = fake_cif_data(data_names_with_missing_name)["data_block_0"]

        with pytest.raises(ValueError) as exception_info:
            get_input_value(missing_data_name, data_items, "CIF file")
        assert str(exception_info.value) == \
            "{} missing from input CIF file".format(missing_data_name)

    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    def test_error_if_invalid_lattice_parameter_data_in_cif(self, invalid_value):

        with pytest.raises(ValueError) as exception_info:
            cif_numerical("cell_length_a", invalid_value)
        assert str(exception_info.value) == \
            "Invalid numerical value in input CIF cell_length_a: {}".format(invalid_value)

    def test_numerical_data_values_stripped_of_errors(self):
        data_items = fake_cif_data(NUMERICAL_DATA_NAMES, errors=True)["data_block_0"]

        for data_name, data_value in data_items.items():
            value = cif_numerical(data_name, data_value)
            assert re.match(r"\d+\.?\d*", value)

    def test_textual_data_values_are_stripped_of_ending_quotes(self):
        data_items = fake_cif_data(TEXTUAL_DATA_NAMES)["data_block_0"]

        for data_value in data_items.values():
            value = cif_textual(data_value)
            assert re.match(r"[^'].*?[^']", value)

    def test_parameters_assigned_for_single_data_block(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)
        c = Crystal.from_cif('some/single/data/block/cif')

        data_items = list(input_dict.values())[0]
        # test numerical data is assigned with the correct type
        for num_data_name, num_attribute in zip(NUMERICAL_DATA_NAMES, NUMERICAL_ATTRIBUTES):
            value = float(CIF_NUMERICAL.match(data_items[num_data_name]).group(1))
            assert getattr(c, num_attribute) == value
            assert isinstance(getattr(c, num_attribute), float)
        # test textual data is assigned with the correct type
        for text_data_name, text_attribute in zip(TEXTUAL_DATA_NAMES, TEXTUAL_ATTRIBUTES):
            value = CIF_TEXTUAL.match(data_items[text_data_name]).group(1)
            assert getattr(c, text_attribute) == value
            assert isinstance(getattr(c, text_attribute), str)

    def test_parameters_assigned_for_multi_data_blocks(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        for data_block_header, data_items in input_dict.items():
            c = Crystal.from_cif('some_file', data_block=data_block_header)
            # test numerical data is assigned with the correct type
            for num_data_name, num_attribute in zip(NUMERICAL_DATA_NAMES, NUMERICAL_ATTRIBUTES):
                value = float(CIF_NUMERICAL.match(data_items[num_data_name]).group(1))
                assert getattr(c, num_attribute) == value
                assert isinstance(getattr(c, num_attribute), float)
            # test textual data is assigned with the correct type
            for text_data_name, text_attribute in zip(TEXTUAL_DATA_NAMES, TEXTUAL_ATTRIBUTES):
                value = CIF_TEXTUAL.match(data_items[text_data_name]).group(1)
                assert getattr(c, text_attribute) == value
                assert isinstance(getattr(c, text_attribute), str)
