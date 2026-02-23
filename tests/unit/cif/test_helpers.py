import re
import string
from collections.abc import Sequence
from typing import Any

import pytest

from diffraction.cif.helpers import (
    NUMERICAL_DATA_NAMES,
    TEXTUAL_DATA_NAMES,
    cif_numerical,
    get_cif_data,
    load_data_block,
)


def fake_num_data(
    data_names: Sequence[str], errors: bool = False, no_data_blocks: int = 1
) -> list[str]:
    """Generates dummy numerical input cif data for testing"""
    if errors:
        data_values = [f"{i}.{i}({i})" for i in range(len(data_names))]
    else:
        data_values = [f"{i}.{i}" for i in range(len(data_names))]
    return data_values


def fake_text_data(
    data_names: Sequence[str], no_data_blocks: int = 1
) -> list[str]:
    """Generates dummy textual input cif data for testing"""
    data_values = [f"'{string.ascii_letters[i] * 5}'" for i in range(len(data_names))]
    return data_values


def fake_cif_data(
    data_names: Sequence[str], errors: bool = False, no_data_blocks: int = 1
) -> dict[str, dict[str, str]]:
    num_data_names = [name for name in data_names if name in NUMERICAL_DATA_NAMES]
    text_data_names = [name for name in data_names if name in TEXTUAL_DATA_NAMES]
    data_values = fake_num_data(
        num_data_names, errors, no_data_blocks
    ) + fake_text_data(text_data_names, no_data_blocks)

    data: dict[str, dict[str, str]] = {}
    for i in range(no_data_blocks):
        data_items = dict(
            zip(num_data_names + text_data_names, data_values, strict=True)
        )
        data[f"data_block_{i}"] = data_items
    return data


class TestLoadingDataItemsFromDataBlocks:
    def test_single_datablock_loaded_automatically(self, mocker: Any) -> None:
        input_dict = fake_cif_data(NUMERICAL_DATA_NAMES + TEXTUAL_DATA_NAMES)
        mocker.patch("diffraction.cif.helpers.load_cif", return_value=input_dict)

        data_items = load_data_block("single/data/block/cif")
        assert data_items == input_dict["data_block_0"]

    def test_error_if_data_block_not_given_for_multi_data_blocks(
        self, mocker: Any
    ) -> None:
        input_dict = fake_cif_data(
            NUMERICAL_DATA_NAMES + TEXTUAL_DATA_NAMES, no_data_blocks=5
        )
        mocker.patch("diffraction.cif.helpers.load_cif", return_value=input_dict)

        with pytest.raises(TypeError) as exception_info:
            load_data_block("multi/data/block/cif")
        assert str(exception_info.value) == (
            "__init__() missing keyword argument: 'data_block'. "
            "Required when input CIF has multiple data blocks."
        )

    def test_data_block_loads_for_multi_data_blocks(self, mocker: Any) -> None:
        input_dict = fake_cif_data(
            NUMERICAL_DATA_NAMES + TEXTUAL_DATA_NAMES, no_data_blocks=5
        )
        mocker.patch("diffraction.cif.helpers.load_cif", return_value=input_dict)

        assert (
            load_data_block("multi/data/block/cif", "data_block_0")
            == input_dict["data_block_0"]
        )


class TestLoadingSpecificDataItems:
    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    def test_error_if_invalid_numerical_parameter_data_in_cif(
        self, invalid_value: str
    ) -> None:
        with pytest.raises(ValueError) as exception_info:
            cif_numerical("cell_length_a", invalid_value)
        assert (
            str(exception_info.value)
            == f"Invalid numerical value in input CIF cell_length_a: {invalid_value}"
        )

    @pytest.mark.parametrize("missing_data_item", "abcdef")
    def test_error_if_parameter_missing_from_cif(self, missing_data_item: str) -> None:
        data_items_mutable: dict[str, str | list[str]] = dict(
            zip("abcdef", [str(i) for i in range(6)], strict=True)
        )
        data_items_mutable.pop(missing_data_item)

        with pytest.raises(ValueError) as exception_info:
            get_cif_data(data_items_mutable, missing_data_item)
        assert (
            str(exception_info.value)
            == f"Parameter: '{missing_data_item}' missing from input CIF"
        )

    def test_get_numerical_cif_data_by_name(self, mocker: Any) -> None:
        test_data_items: dict[str, str | list[str]] = {
            "cell_length_a": "4.9900(2)",
            "cell_length_b": "4.9900(2)",
            "cell_angle_gamma": "120.",
            "atom_site_fract_x": "-2.34",
        }
        cif_num_mock = mocker.patch(
            "diffraction.cif.helpers.cif_numerical",
            side_effect=lambda data_name, data_value: data_value,
        )

        data_names, data_values = zip(*test_data_items.items(), strict=True)

        data = get_cif_data(test_data_items, *data_names)
        expected_calls = [
            ((data_name, data_value),)
            for data_name, data_value in zip(data_names, data_values, strict=True)
        ]
        assert cif_num_mock.call_args_list == expected_calls
        assert data == list(data_values)

    def test_get_textual_cif_data_by_name(self) -> None:
        test_data_items: dict[str, str | list[str]] = {
            "symmetry_space_group_name_H-M": "R -3 c H",
            "chemical_name_mineral ": "Calcite",
            "chemical_formula_sum": "C1 Ca1 O3",
        }

        data_names, data_values = zip(*test_data_items.items(), strict=True)
        data = get_cif_data(test_data_items, *data_names)
        assert data == list(data_values)

    def test_get_textual_loop_cif_data_by_name(self) -> None:
        test_data_items: dict[str, str | list[str]] = {
            "cell_length_a": "4.9900(2)",
            "atom_site_label": ["Ca1", "C1", "O1"],
            "atom_type_symbol": ["Ca2+", "C4+", "O2-"],
        }

        labels, symbols = get_cif_data(
            test_data_items, "atom_site_label", "atom_type_symbol"
        )
        assert labels == ["Ca1", "C1", "O1"]
        assert symbols == ["Ca2+", "C4+", "O2-"]

    def test_get_numerical_loop_cif_data_by_name(self, mocker: Any) -> None:
        test_data_items: dict[str, str | list[str]] = {
            "cell_length_a": "4.9900(2)",
            "atom_site_fract_x": ["0", "0", "0.25706(33)"],
            "atom_site_fract_y": ["0", "0.25", "0.25"],
        }
        cif_num_mock = mocker.patch(
            "diffraction.cif.helpers.cif_numerical",
            side_effect=lambda data_name, data_value: data_value,
        )

        expected_calls = [
            (("atom_site_fract_x", test_data_items["atom_site_fract_x"]),),
            (("atom_site_fract_y", test_data_items["atom_site_fract_y"]),),
        ]
        x, y = get_cif_data(test_data_items, "atom_site_fract_x", "atom_site_fract_y")
        assert cif_num_mock.call_args_list == expected_calls
        assert x == test_data_items["atom_site_fract_x"]
        assert y == test_data_items["atom_site_fract_y"]

    def test_getting_single_data_value(self) -> None:
        test_data_items: dict[str, str | list[str]] = {
            "symmetry_space_group_name_H-M": "R -3 c H",
            "chemical_name_mineral ": "Calcite",
            "chemical_formula_sum": "C1 Ca1 O3",
        }

        data_name = "chemical_formula_sum"
        [data] = get_cif_data(test_data_items, data_name)
        assert data == "C1 Ca1 O3"

    def test_numerical_data_values_stripped_of_errors(self, mocker: Any) -> None:
        data_items = fake_cif_data(NUMERICAL_DATA_NAMES, errors=True)["data_block_0"]
        mocker.patch("diffraction.cif.helpers.float", side_effect=lambda x: x)

        for data_name, data_value in data_items.items():
            value = cif_numerical(data_name, data_value)
            assert re.match(r"\d+\.?\d*", str(value))
