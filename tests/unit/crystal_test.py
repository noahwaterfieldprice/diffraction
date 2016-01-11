import collections
import random
import re
import string

import pytest

from diffraction import Crystal
from diffraction.crystal import (
    NUMERICAL_DATA_NAMES, NUMERICAL_ATTRIBUTES,
    TEXTUAL_DATA_NAMES, TEXTUAL_ATTRIBUTES,
    CIF_NUMERICAL, CIF_TEXTUAL, load_data_block,
    numerical_parameter, cif_numerical_parameter,
    textual_parameter, cif_textual_parameter)

# Numerical and textual data names and attributes for testing
DATA_NAMES = list(NUMERICAL_DATA_NAMES) + list(TEXTUAL_DATA_NAMES)
ATTRIBUTES = list(NUMERICAL_ATTRIBUTES) + list(TEXTUAL_ATTRIBUTES)


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
            if missing_data_name in NUMERICAL_DATA_NAMES:
                cif_numerical_parameter(missing_data_name, data_items)
            elif missing_data_name in TEXTUAL_DATA_NAMES:
                cif_textual_parameter(missing_data_name, data_items)
        assert str(exception_info.value) == \
            "{} missing from input CIF file".format(missing_data_name)

    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    def test_error_if_invalid_numerical_value_data_in_cif(self, invalid_value):
        data_items = fake_cif_data(NUMERICAL_DATA_NAMES)["data_block_0"]
        data_items["cell_length_a"] = invalid_value

        with pytest.raises(ValueError) as exception_info:
            cif_numerical_parameter("cell_length_a", data_items)
        assert str(exception_info.value) == \
            "Invalid numerical parameter cell_length_a: {}".format(invalid_value)

    @pytest.mark.parametrize("errors", [True, False])
    def test_numerical_data_values_are_converted_to_float(self, errors):
        data_items = fake_cif_data(NUMERICAL_DATA_NAMES, errors=errors)["data_block_0"]

        for data_name in data_items.keys():
            value = cif_numerical_parameter(data_name, data_items)
            assert isinstance(value, float)

    def test_textual_data_values_are_stripped_of_ending_quotes(self):
        data_items = fake_cif_data(TEXTUAL_DATA_NAMES)["data_block_0"]

        for data_name in data_items.keys():
            value = cif_textual_parameter(data_name, data_items)
            assert re.match(r"[^'].*?[^']", value)

    def test_parameters_assigned_for_single_data_block(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)
        c = Crystal('some/single/data/block/cif')

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
            c = Crystal('some_file', data_block=data_block_header)
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


class TestCreatingCrystalFromDict():
    crystal_info = {"a": 4.99, "b": 4.99, "c": 17.003,
                    "alpha": 90, "beta": 90, "gamma": 120,
                    "space_group": "R -3 c H"}

    @pytest.mark.parametrize("missing_parameter", ATTRIBUTES)
    def test_error_if_parameter_missing_from_dict(self, missing_parameter):
        dict_with_missing_parameter = self.crystal_info.copy()
        del dict_with_missing_parameter[missing_parameter]

        with pytest.raises(ValueError) as exception_info:
            if missing_parameter in NUMERICAL_ATTRIBUTES:
                numerical_parameter(missing_parameter, dict_with_missing_parameter)
            elif missing_parameter in TEXTUAL_ATTRIBUTES:
                textual_parameter(missing_parameter, dict_with_missing_parameter)
        assert str(exception_info.value) == \
            "{} missing from input dictionary".format(missing_parameter)

    @pytest.mark.parametrize("invalid_parameter", ["abc", "123@%£", "1232.433.21"])
    def test_error_if_invalid_numerical_parameter_in_dict(self, invalid_parameter):
        dict_with_invalid_parameter = self.crystal_info.copy()
        dict_with_invalid_parameter["a"] = invalid_parameter

        with pytest.raises(ValueError) as exception_info:
            numerical_parameter("a", dict_with_invalid_parameter)
        assert str(exception_info.value) == \
            "Invalid numerical parameter a: {}".format(invalid_parameter)

    def test_parameters_are_assigned_with_correct_type(self):
        c = Crystal(self.crystal_info)

        # test numerical data is assigned with the correct type
        for num_attribute in NUMERICAL_ATTRIBUTES:
            assert getattr(c, num_attribute) == self.crystal_info[num_attribute]
            assert isinstance(getattr(c, num_attribute), float)
        # test textual data is assigned with the correct type
        for text_attribute in TEXTUAL_ATTRIBUTES:
            assert getattr(c, text_attribute) == self.crystal_info[text_attribute]
            assert isinstance(getattr(c, text_attribute), str)
