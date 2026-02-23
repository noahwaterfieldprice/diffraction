from collections import OrderedDict
from typing import Any, ClassVar
from unittest import mock

import pytest

from diffraction import load_cif, validate_cif
from diffraction.cif.cif import (
    INLINE_DATA_ITEM,
    SEMICOLON_DATA_ITEM,
    CIFParseError,
    CIFParser,
    CIFValidator,
    DataBlock,
    strip_quotes,
)


class TestParsingFile:
    def test_datablock_class_abbreviates_raw_data_when_printed(self) -> None:
        # test when raw_data is shorter than 18 characters
        data_block = DataBlock("header", "a" * 10)
        assert repr(data_block) == "DataBlock('header', '%s', {})" % ("a" * 10)
        # test when raw_data is longer than 18 characters
        data_block = DataBlock("header", "a" * 100)
        assert repr(data_block) == "DataBlock('header', '%s...', {})" % ("a" * 15)

    def test_file_contents_are_stored_as_raw_string_attribute(
        self, mocker: Any
    ) -> None:
        contents = [
            "_data_name_1 data_value_1",
            "_data_name_2 data_value_2",
            "_etc etc",
        ]
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data="\n".join(contents)))

        filepath = "/some_directory/some_file.cif"
        p = CIFParser(filepath)
        assert p.raw_data == "\n".join(contents)

    def test_comments_and_blank_lines_are_stripped_out(self, mocker: Any) -> None:
        contents = [
            "# Here is a comment on the first line",
            "# Here is another comment. The next line is just whitespace",
            "\t\t\t\t\t",
            "",
            "_some_normal_line previous_line_was_blank",
            "  _another_normal_line starting_with_whitespace",
            "# Final comment ## with # extra hashes ### in ##",
        ]
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data="\n".join(contents)))
        expected_remaining_lines = contents[4:6]

        p = CIFParser("/some_directory/some_file.cif")
        p._strip_comments_and_blank_lines()
        assert p.raw_data == "\n".join(expected_remaining_lines)

    def test_file_split_by_data_blocks(self, mocker: Any) -> None:
        block_1 = [
            "data_block_header",
            "_data_name_A data_value_A",
            "_data_name_B data_value_B",
        ]
        block_2 = [
            "DATA_block_2",
            "loop_",
            "_loop_data_name_A",
            "loop_data_value_A1",
            "loop_data_value_A2",
        ]
        block_3 = ["dATa_block_the_third", "_data_name_C data_value_C"]
        contents = block_1 + block_2 + block_3
        mocker.patch(
            "pathlib.Path.open", mock.mock_open(read_data=str("\n".join(contents)))
        )
        # generate expected output - each data block stored in DataBlock object
        expected = []
        for block in [block_1, block_2, block_3]:
            header, *raw_data = block
            expected.append(DataBlock(header, "\n".join(raw_data)))

        p = CIFParser("/some_directory/some_file.cif")
        p._extract_data_blocks()
        assert p.data_blocks == expected

    def test_textual_data_values_are_stripped_of_ending_quotes(self) -> None:
        test_data_values = [
            "'data value with single quotes'",
            '"data value with double quotes"',
            "'data value with \n newline and single quotes'",
            "data value with no quotes",
            "12345.6789",
            "''",
        ]
        expected_data_values = [
            "data value with single quotes",
            "data value with double quotes",
            "data value with \n newline and single quotes",
            "data value with no quotes",
            "12345.6789",
            "",
        ]

        for test, expected in zip(test_data_values, expected_data_values, strict=True):
            assert strip_quotes(test) == expected

    def test_semicolon_data_items_are_assigned(self, mocker: Any) -> None:
        contents = [
            "_data_name_1",
            ";",
            "very long semicolon text field with many words",
            ";",
            "_data_name_2 data_value_2",
            "_data_name_3",
            ";",
            "semicolon text field with",
            "two lines of text",
            ";_data_name_4 data_value_4",
            "_data_name_5",
            ";",
            "semicolon text ; field containing ;;; semicolons",
            ";",
        ]
        data_block = DataBlock("data_block_header", "\n".join(contents))
        semicolon_data_items = OrderedDict(
            [
                ("data_name_1", "very long semicolon text field with many words"),
                ("data_name_3", "semicolon text field with\ntwo lines of text"),
                ("data_name_5", "semicolon text ; field containing ;;; semicolons"),
            ]
        )
        strip_quotes_mock = mocker.patch(
            "diffraction.cif.cif.strip_quotes",
            side_effect=lambda data_value: data_value,
        )

        data_block.extract_data_items(SEMICOLON_DATA_ITEM)
        expected_calls = [
            mock.call(data_value) for data_value in semicolon_data_items.values()
        ]
        assert strip_quotes_mock.call_args_list == expected_calls
        assert data_block.data_items == semicolon_data_items

    def test_semicolon_data_items_are_stripped_out(self) -> None:
        contents = [
            "_data_name_1 data_value_1",
            "_data_name_2",
            ";",
            "semicolon text field ",
            ";",
            "_data_name_3 data_value_3",
            "_data_name_4",
            ";",
            "semicolon text field with",
            "two lines of text",
            ";",
        ]
        data_block = DataBlock("data_block_header", "\n".join(contents))
        expected_remaining_data = "\n".join([contents[0], contents[5]])

        data_block.extract_data_items(SEMICOLON_DATA_ITEM)
        assert data_block.raw_data == expected_remaining_data

    def test_inline_declared_variables_are_assigned(self, mocker: Any) -> None:
        data_items = OrderedDict(
            [
                ("data_name", "value"),
                ("four_word_data_name", "four_word_data_value"),
                ("data_name-with_hyphens-in-it", "some_data_value"),
                ("data_name_4", "'data value inside single quotes'"),
                ("data_name_5", '"data value inside double quotes"'),
            ]
        )
        contents = [
            f"_{data_name} {data_value}" for data_name, data_value in data_items.items()
        ]
        data_block = DataBlock("data_block_header", "\n".join(contents))
        strip_quotes_mock = mocker.patch(
            "diffraction.cif.cif.strip_quotes",
            side_effect=lambda data_value: data_value,
        )

        data_block.extract_data_items(INLINE_DATA_ITEM)
        expected_calls = [mock.call(data_value) for data_value in data_items.values()]
        assert strip_quotes_mock.call_args_list == expected_calls
        assert data_block.data_items == data_items

    def test_inline_declared_variables_are_stripped_out(self) -> None:
        contents = [
            "_data_name_1 value",
            "_DatA_name-two another_value",
            "loop_",
            "_loop_data_name_A",
            "_loop_data_name_B",
            "value_A1 'value A2'",
            "value-B1 value_B2",
            "_one_more_data_item_ one_more_data_value",
        ]
        data_block = DataBlock("data_block_header", "\n".join(contents))
        expected_remaining_data = "\n" + "\n".join(contents[2:7])

        data_block.extract_data_items(INLINE_DATA_ITEM)
        assert data_block.raw_data == expected_remaining_data

    def test_variables_declared_in_loop_are_assigned(self, mocker: Any) -> None:
        data_items = {
            "number": ["1", "2222", "3456789"],
            "symbol": [".", "-", "?"],
            "number_and_symbol": ["-1.0", "2.0(3)", "3.0e10"],
            "letter": ["a", "bbb", "cdefghi"],
            "letter_and_symbol": ["x?*", "abc_(rt)", "sin(3*10^3)"],
            "single_quotes": ["'x y z'", "'s = 3.2(3)'", "'x -y+2/3 z-0.876'"],
            "double_quotes": ['"a b c"', '"s = 4.6(1)"', '"x-1/3 y+0.34 -z"'],
        }
        # convert the data items into corresponding CIF input
        contents = ["loop_"]
        data_names = data_items.keys()
        contents.extend("_" + data_name for data_name in data_names)
        contents.extend(
            "{} {} {} {} {} {} {}".format(
                *[data_items[data_name][i] for data_name in data_names]
            )
            for i in range(3)
        )
        data_block = DataBlock("data_block_header", "\n".join(contents))
        strip_quotes_mock = mocker.patch(
            "diffraction.cif.cif.strip_quotes",
            side_effect=lambda data_value: data_value,
        )

        data_block.extract_loop_data_items()
        assert strip_quotes_mock.call_count == 21
        assert data_block.data_items == data_items

    def test_parse_method_calls_in_correct_order(self) -> None:
        p: Any = mock.Mock(spec=CIFParser)
        data_block: Any = mock.Mock(spec=DataBlock)
        p.data_blocks = [data_block]
        CIFParser.parse(p)
        expected_calls = [
            mock.call._strip_comments_and_blank_lines(),
            mock.call._extract_data_blocks(),
            mock.call.extract_data_items(SEMICOLON_DATA_ITEM),
            mock.call.extract_data_items(INLINE_DATA_ITEM),
            mock.call.extract_loop_data_items(),
        ]
        assert p.method_calls + data_block.method_calls == expected_calls


class TestCIFSyntaxExceptions:
    valid_comments: ClassVar[list[str]] = [
        "# some comment",
        "# another comment - next line blank                      ",
        "# final comment ## with hashes ## in",
    ]
    valid_inline_items: ClassVar[list[str]] = [
        "data_block_header_1",
        "_data_name_1 value",
        "_DatA_name-two another_value",
    ]
    valid_loop: ClassVar[list[str]] = [
        "loop_",
        "# comment inside a loop before data names- next line blank",
        "                      ",
        "_loop_data_name_A",
        "_loop_data_name_B",
        "# comment inside loop after data names",
        "value_A1 'value A2'",
    ]
    valid_semicolon_field: ClassVar[list[str]] = [
        "_data_name_4",
        ";",
        "semicolon text field with",
        "two lines of text",
        ";",
    ]

    def test_error_throws_correct_exception_with_message(self) -> None:
        message = "Oh no! An exception has been raised...."

        with pytest.raises(CIFParseError) as exception_info:
            v: Any = mock.Mock(spec=CIFValidator)
            v.error = CIFValidator.error
            v.error(v, message, 1, "Erroneous line")
        assert str(exception_info.value) == f'{message} on line 1: "Erroneous line"'

    @pytest.mark.parametrize(
        "valid_contents",
        [valid_comments, valid_inline_items, valid_loop, valid_semicolon_field],
    )
    def test_valid_syntax_raises_no_exception(self, valid_contents: list[str]) -> None:
        v = CIFValidator("\n".join(valid_contents))
        assert v.validate() is True

    def test_warning_if_file_is_empty(self) -> None:
        # test when file is empty
        with pytest.warns(UserWarning):
            CIFValidator("")

        # test when is only whitespace
        with pytest.warns(UserWarning):
            CIFValidator("     \n    \n  \t \t \n   ")

    @pytest.mark.parametrize(
        "invalid_line",
        [
            "value_with_missing_data_name",
            "  starting_with_whitespace",
            "'in single quotes'",
            '"in double quotes"',
        ],
    )
    def test_error_if_missing_inline_data_name(self, invalid_line: str) -> None:
        contents = [
            "_data_name_1 value_1",
            "_data_name_2 value_2",
        ]
        contents.insert(1, invalid_line)
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert (
            str(exception_info.value)
            == f'Missing inline data name on line 2: "{invalid_line}"'
        )

    def test_error_if_invalid_inline_data_value(self) -> None:
        contents = ["_data_name_1 value_1", "_data_name_2 ", "_data_name_3 value_3"]

        # test when final line of file
        v = CIFValidator("\n".join(contents[:2]))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert (
            str(exception_info.value)
            == 'Invalid inline data value on line 2: "_data_name_2 "'
        )

        # test when followed by another line
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert (
            str(exception_info.value)
            == 'Invalid inline data value on line 2: "_data_name_2 "'
        )

    @pytest.mark.parametrize("invalid_line", ["value_A1", "value_A1 value_B1 value_C1"])
    def test_error_if_unmatched_data_items_in_loop(self, invalid_line: str) -> None:
        contents = ["loop_", "_data_name_A", "_data_name_B ", "value_A1 value_B1"]
        contents.insert(4, invalid_line)
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert str(exception_info.value) == (
            f'Unmatched data values to data names in loop on line 5: "{invalid_line}"'
        )

    def test_error_if_semicolon_data_item_not_closed(self) -> None:
        contents = [
            "_data_name_1",
            ";",
            "# some comment inside the text field",
            "Unclosed text field",
            "_data_item_2",
        ]
        # test when field is terminated by end of file
        v = CIFValidator("\n".join(contents[:4]))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert (
            str(exception_info.value)
            == 'Unclosed semicolon text field on line 4: "Unclosed text field"'
        )

        # test when field is terminated by another data item
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert (
            str(exception_info.value)
            == 'Unclosed semicolon text field on line 4: "Unclosed text field"'
        )


class TestLoadCif:
    def test_load_cif_returns_dict_keyed_by_data_block_header(
        self, mocker: Any
    ) -> None:
        content = "data_test\n_cell_length_a 5.00\n"
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data=content))

        result = load_cif("/fake/path.cif")
        assert "data_test" in result

    def test_load_cif_parses_multiple_data_blocks(self, mocker: Any) -> None:
        content = "data_block_one\n_key_a valueA\ndata_block_two\n_key_b valueB\n"
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data=content))

        result = load_cif("/fake/path.cif")
        assert "data_block_one" in result
        assert "data_block_two" in result

    def test_load_cif_extracts_data_values(self, mocker: Any) -> None:
        content = "data_test\n_cell_length_a 4.99\n_cell_length_b 4.99\n"
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data=content))

        result = load_cif("/fake/path.cif")
        data = result["data_test"]
        assert "cell_length_a" in data
        assert data["cell_length_a"] == "4.99"

    def test_load_cif_reads_real_calcite_file(self, calcite_cif_path: Any) -> None:
        result = load_cif(str(calcite_cif_path))

        # calcite_icsd.cif has a single data block keyed by its ICSD code
        assert len(result) == 1
        block = next(iter(result.values()))
        assert "cell_length_a" in block
        assert "symmetry_space_group_name_H-M" in block

    def test_load_cif_real_calcite_file_has_expected_data_block(
        self, calcite_cif_path: Any
    ) -> None:
        result = load_cif(str(calcite_cif_path))

        # The data block header in calcite_icsd.cif is data_18166-ICSD
        assert "data_18166-ICSD" in result


class TestValidateCif:
    def test_validate_cif_returns_true_for_valid_content(self, mocker: Any) -> None:
        content = "data_test\n_cell_length_a 5.00\n"
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data=content))

        result = validate_cif("/fake/path.cif")
        assert result is True

    def test_validate_cif_raises_for_value_without_data_name(self, mocker: Any) -> None:
        # A lone data value (not preceded by a _data_name) is a CIF syntax error
        content = "data_test\n4.99\n"
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data=content))

        with pytest.raises(CIFParseError, match="Missing inline data name"):
            validate_cif("/fake/path.cif")

    def test_validate_cif_raises_for_unclosed_semicolon_field(
        self, mocker: Any
    ) -> None:
        content = "data_test\n_data_name_1\n;\nunclosed text field\n"
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data=content))

        with pytest.raises(CIFParseError, match="Unclosed semicolon text field"):
            validate_cif("/fake/path.cif")

    def test_validate_cif_validates_multi_block_content(self, mocker: Any) -> None:
        content = (
            "data_block_one\n_cell_length_a 5.00\n"
            "data_block_two\n_cell_length_b 6.00\n"
        )
        mocker.patch("pathlib.Path.open", mock.mock_open(read_data=content))

        result = validate_cif("/fake/path.cif")

        assert result is True

    def test_validate_cif_accepts_real_calcite_file(self, calcite_cif_path: Any) -> None:
        result = validate_cif(str(calcite_cif_path))

        assert result is True
