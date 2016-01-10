import collections
import random
import re
import string

import pytest

from diffraction import Crystal
from diffraction.crystal import (NUMERICAL_PARAMETERS, TEXTUAL_PARAMETERS,
                                 CIF_NUMERICAL, CIF_TEXTUAL, load_data_block,
                                 numerical_data_value, textual_data_value)

# Numerical and textual data names and attributes for testing
NUM_DATA_NAMES, NUM_ATTRIBUTES = zip(*NUMERICAL_PARAMETERS.items())
TEXT_DATA_NAMES, TEXT_ATTRIBUTES = zip(*TEXTUAL_PARAMETERS.items())

DATA_NAMES = list(NUM_DATA_NAMES) + list(TEXT_DATA_NAMES)
ATTRIBUTES = list(NUM_ATTRIBUTES) + list(TEXT_ATTRIBUTES)


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
                      if name in NUM_DATA_NAMES]
    text_data_names = [name for name in data_names
                       if name in TEXT_DATA_NAMES]
    data_values = fake_num_data(num_data_names, errors, no_data_blocks) + \
        fake_text_data(text_data_names, no_data_blocks)

    data = {}
    for i in range(no_data_blocks):
        data_items = dict(zip(num_data_names + text_data_names, data_values))
        data["data_block_{}".format(i)] = data_items
    return data


class TestLoadingDataFromSingleDatablockCIF:
    def test_single_datablock_loaded_automatically(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        data_items = load_data_block("single/data/block/cif")
        assert data_items == input_dict["data_block_0"]


class TestLoadingDataFromMultiDatablockCIF:
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


class TestConvertingCIFData:
    @pytest.mark.parametrize("missing_data_name", DATA_NAMES)
    def test_error_if_parameter_is_missing_from_cif(self, missing_data_name):
        data_names_with_missing_name = DATA_NAMES[:]
        data_names_with_missing_name.remove(missing_data_name)
        data_items = fake_cif_data(data_names_with_missing_name)["data_block_0"]

        with pytest.raises(ValueError) as exception_info:
            if missing_data_name in NUM_DATA_NAMES:
                numerical_data_value(missing_data_name, data_items)
            elif missing_data_name in TEXT_DATA_NAMES:
                textual_data_value(missing_data_name, data_items)
        assert str(exception_info.value) == \
            "{} missing from input CIF file".format(missing_data_name)

    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    def test_error_if_invalid_numerical_value_data_in_cif(self, invalid_value):
        data_items = fake_cif_data(NUM_DATA_NAMES)["data_block_0"]
        data_items["cell_length_a"] = invalid_value

        with pytest.raises(ValueError) as exception_info:
            numerical_data_value("cell_length_a", data_items)
        assert str(exception_info.value) == \
            "Invalid lattice parameter cell_length_a: {}".format(invalid_value)

    @pytest.mark.parametrize("errors", [True, False])
    def test_numerical_data_values_are_converted_to_float(self, errors):
        data_items = fake_cif_data(NUM_DATA_NAMES, errors=errors)["data_block_0"]

        for data_name in data_items.keys():
            value = numerical_data_value(data_name, data_items)
            assert isinstance(value, float)

    def test_textual_data_values_are_stripped_of_ending_quotes(self):
        data_items = fake_cif_data(TEXT_DATA_NAMES)["data_block_0"]

        for data_name in data_items.keys():
            value = textual_data_value(data_name, data_items)
            assert re.match(r"[^'].*?[^']", value)


class TestCreatingCrystalFromCIF:
    def test_parameters_assigned_for_single_data_block(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)
        c = Crystal('some/single/data/block/cif')

        data_items = list(input_dict.values())[0]
        # test numerical data is assigned with the correct type
        for num_data_name, num_attribute in zip(NUM_DATA_NAMES, NUM_ATTRIBUTES):
            value = float(CIF_NUMERICAL.match(data_items[num_data_name]).group(1))
            assert getattr(c, num_attribute) == value
            assert isinstance(getattr(c, num_attribute), float)
        # test textual data is assigned with the correct type
        for text_data_name, text_attribute in zip(TEXT_DATA_NAMES, TEXT_ATTRIBUTES):
            value = CIF_TEXTUAL.match(data_items[text_data_name]).group(1)
            assert getattr(c, text_attribute) == value
            assert isinstance(getattr(c, text_attribute), str)

    def test_parameters_assigned_for_multi_data_blocks(self, mocker):
        input_dict = fake_cif_data(DATA_NAMES, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        for data_block_header, data_items in input_dict.items():
            c = Crystal('some_file', data_block=data_block_header)
            # test numerical data is assigned with the correct type
            for num_data_name, num_attribute in zip(NUM_DATA_NAMES, NUM_ATTRIBUTES):
                value = float(CIF_NUMERICAL.match(data_items[num_data_name]).group(1))
                assert getattr(c, num_attribute) == value
                assert isinstance(getattr(c, num_attribute), float)
            # test textual data is assigned with the correct type
            for text_data_name, text_attribute in zip(TEXT_DATA_NAMES, TEXT_ATTRIBUTES):
                value = CIF_TEXTUAL.match(data_items[text_data_name]).group(1)
                assert getattr(c, text_attribute) == value
                assert isinstance(getattr(c, text_attribute), str)
