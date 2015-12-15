import collections
import random

import pytest

from diffraction import Crystal
from diffraction.crystal import (LATTICE_PARAMETERS, CIF_NUMERICAL,
                                 load_data_block, numerical_data_value)


def random_numerical_cif_data(data_names, errors=False, no_data_blocks=1):
    data = {}
    if errors:
        data_values = ["{0}({1})".format(random.random(), random.randint(1, 9))
                       for _ in range(len(data_names))]
    else:
        data_values = [str(random.random()) for _ in range(len(data_names))]
    for i in range(no_data_blocks):
        data_items = dict(zip(data_names, data_values))
        data["data_block_{}".format(i)] = data_items
    return data


class TestConvertingCIFData:
    def test_single_datablock_loaded_automatically(self, mocker):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        assert load_data_block("single/data/block/cif") == \
               input_dict["data_block_0"]

    def test_error_if_data_block_not_given_for_multi_data_blocks(self, mocker):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        with pytest.raises(TypeError) as exception_info:
            load_data_block("multi/data/block/cif")
        assert str(exception_info.value) == \
               ("__init__() missing keyword argument: 'data_block'. "
                "Required when input CIF has multiple data blocks.")

    @pytest.mark.parametrize("data_block", ["data_block_{}".format(i)
                                            for i in range(5)])
    def test_data_block_loaded_for_multi_data_blocks(self, mocker, data_block):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        assert load_data_block("multi/data/block/cif", data_block) == \
               input_dict[data_block]

    def test_error_if_numerical_parameter_is_missing_from_cif(self):
        data_names = collections.deque(LATTICE_PARAMETERS.keys())
        # test exception is raised for each possible missing parameter
        for _ in range(len(data_names)):
            missing_data_name, *remaining_data_names = data_names
            data_items = random_numerical_cif_data(
                remaining_data_names)["data_block_0"]

            with pytest.raises(ValueError) as exception_info:
                numerical_data_value(missing_data_name, data_items)
            assert str(exception_info.value) == \
                   "{} missing from input CIF file".format(missing_data_name)

            data_names.rotate(-1)

    def test_error_if_invalid_numerical_data_values_in_cif(self):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        data_items = random_numerical_cif_data(data_names)["data_block_0"]
        for invalid_value in ["abc", "123@%£", "1232.43543.21"]:
            data_items["cell_length_a"] = invalid_value

            with pytest.raises(ValueError) as exception_info:
                numerical_data_value("cell_length_a", data_items)
            assert str(exception_info.value) == \
                   "Invalid lattice parameter cell_length_a: {}".format(
                       invalid_value)

    @pytest.mark.parametrize("errors", [True, False])
    def test_numerical_data_values_are_converted_to_float(self, errors):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        data_items = random_numerical_cif_data(
            data_names, errors=errors)["data_block_0"]

        for data_name in data_items.keys():
            value = numerical_data_value(data_name, data_items)
            assert isinstance(value, float)


class TestCreatingCrystalFromCIF:
    def test_lattice_parameters_assigned_for_single_data_block(self, mocker):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names)
        mocker.patch('diffraction.crystal.load_cif',
                     return_value=input_dict)
        c = Crystal('some/single/data/block/cif')

        data_items = list(input_dict.values())[0]
        for data_name, attribute in zip(data_names, attributes):
            value = float(CIF_NUMERICAL.match(data_items[data_name]).group(1))
            assert getattr(c, attribute) == value
            assert isinstance(getattr(c, attribute), float)

    def test_lattice_parameters_assigned_for_multi_data_blocks(self, mocker):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        for data_block_header, data_items in input_dict.items():
            c = Crystal('some_file', data_block=data_block_header)
            for data_name, attribute in zip(data_names, attributes):
                value = float(
                    CIF_NUMERICAL.match(data_items[data_name]).group(1))
                assert getattr(c, attribute) == value
                assert isinstance(getattr(c, attribute), float)
