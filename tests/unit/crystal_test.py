import collections
import random

import pytest

from diffraction import Crystal
from diffraction.crystal import LATTICE_PARAMETERS, CIF_NUMERICAL


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


class TestCreatingCrystalFromCIF:
    def test_error_if_lattice_parameter_is_missing_from_cif(self, mocker):
        data_names = collections.deque(LATTICE_PARAMETERS.keys())
        # test exception is raised for each possible missing parameter
        for _ in range(len(data_names)):
            missing_data_name, *remaining_data_names = data_names
            input_dict = random_numerical_cif_data(remaining_data_names)
            mocker.patch('diffraction.crystal.load_cif',
                         return_value=input_dict)

            with pytest.raises(ValueError) as exception_info:
                Crystal('some/cif/with/missing/parameter')
            assert str(exception_info.value) == \
                   "{} missing from input CIF file".format(missing_data_name)

            data_names.rotate(-1)

    def test_error_if_invalid_lattice_parameter_values_in_cif(self, mocker):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names)
        for invalid_value in ["abc", "123@%£", "1232.43543.21"]:
            input_dict["data_block_0"]["cell_length_a"] = invalid_value
            mocker.patch('diffraction.crystal.load_cif',
                         return_value=input_dict)

            with pytest.raises(ValueError) as exception_info:
                Crystal('some/single/data/block/cif')
            assert str(exception_info.value) == \
                "Invalid lattice parameter cell_length_a: {}".format(
                    invalid_value)

    @pytest.mark.parametrize("errors", [True, False])
    def test_lattice_parameters_assigned_for_single_data_block(self,
                                                               mocker,
                                                               errors):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names, errors=errors)
        mocker.patch('diffraction.crystal.load_cif',
                     return_value=input_dict)
        c = Crystal('some/single/data/block/cif')

        data_items = list(input_dict.values())[0]
        for data_name, attribute in zip(data_names, attributes):
            value = float(CIF_NUMERICAL.match(data_items[data_name]).group(1))
            assert getattr(c, attribute) == value
            assert isinstance(getattr(c, attribute), float)

    def test_error_if_datablock_not_given_for_multi_data_blocks(self, mocker):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names, no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        with pytest.raises(TypeError) as exception_info:
            Crystal("some/multi/data/block/cif")
        assert str(exception_info.value) == \
               ("__init__() missing keyword argument: 'data_block'. "
                "Required when input CIF has multiple data blocks.")

    @pytest.mark.parametrize("errors", [True, False])
    def test_lattice_parameters_assigned_for_multi_data_blocks(self, mocker,
                                                               errors):
        data_names, attributes = zip(*LATTICE_PARAMETERS.items())
        input_dict = random_numerical_cif_data(data_names, errors=errors,
                                               no_data_blocks=5)
        mocker.patch('diffraction.crystal.load_cif', return_value=input_dict)

        for data_block_header, data_items in input_dict.items():
            c = Crystal('some_file', data_block=data_block_header)
            for data_name, attribute in zip(data_names, attributes):
                value = float(
                    CIF_NUMERICAL.match(data_items[data_name]).group(1))
                assert getattr(c, attribute) == value
                assert isinstance(getattr(c, attribute), float)
