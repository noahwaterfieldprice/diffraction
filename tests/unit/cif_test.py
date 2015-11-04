import os
import json
from collections import OrderedDict
from unittest import mock

import pytest

from diffraction import load_cif, CIFParser, CIFValidator, CIFParseError

from diffraction.cif import DataBlock, INLINE_DATA_ITEM, SEMICOLON_DATA_ITEM

OPEN = "builtins.open"


class TestLoadingFile:
    def test_load_cif_opens_correct_file(self, mocker):
        mocker.patch(OPEN)
        mocker.patch("diffraction.cif.CIFParser")
        filepath = "/some_directory/some_file.cif"
        load_cif(filepath)
        open.assert_called_with(filepath, "r")

    def test_raises_warning_if_file_extension_is_not_cif(self, mocker):
        mocker.patch(OPEN)
        mocker.patch("diffraction.cif.CIFParser")
        non_cif_filepath = "/some_directory/some_file.not_cif"
        with pytest.warns(UserWarning):
            load_cif(non_cif_filepath)


class TestParsingFile:
    def test_datablock_class_abbreviates_raw_data_when_printed(self):
        # test when raw_data is shorter than 18 characters
        data_block = DataBlock("header", "a" * 10, {})
        assert repr(data_block) == \
               "DataBlock('header', '%s', {})" % ("a" * 10)
        # test when raw_data is longer than 18 characters
        data_block = DataBlock("header", "a" * 100, {})
        assert repr(data_block) == \
               "DataBlock('header', '%s...', {})" % ("a" * 15)

    def test_file_contents_are_stored_as_raw_string_attribute(self, mocker):
        contents = [
            "_data_name_1 data_value_1",
            "_data_name_2 data_value_2",
            "_etc etc",
        ]
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        filepath = "/some_directory/some_file.cif"
        p = CIFParser(filepath)
        # make sure correct file was loaded
        open.assert_called_with(filepath, "r")
        assert p.raw_data == '\n'.join(contents)

    def test_comments_and_blank_lines_are_stripped_out(self, mocker):
        contents = [
            "# Here is a comment on the first line",
            "# Here is another comment. The next line is just whitespace",
            "\t\t\t\t\t",
            "",
            "_some_normal_line previous_line_was_blank",
            "  _another_normal_line starting_with_whitespace",
            '# Final comment ## with # extra hashes ### in ##'
        ]
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))
        expected_remaining_lines = contents[4:6]

        p = CIFParser("/some_directory/some_file.cif")
        p._strip_comments_and_blank_lines()
        assert p.raw_data == "\n".join(expected_remaining_lines)

    def test_file_split_by_data_blocks(self, mocker):
        block_1 = [
            "data_block_header",
            "_data_name_A data_value_A",
            "_data_name_B data_value_B"
        ]
        block_2 = [
            "DATA_block_2",
            "loop_",
            "_loop_data_name_A",
            "loop_data_value_A1",
            "loop_data_value_A2"
        ]
        block_3 = [
            "dATa_block_the_third",
            "_data_name_C data_value_C"
        ]
        contents = block_1 + block_2 + block_3
        mocker.patch(OPEN, mock.mock_open(read_data=str("\n".join(contents))))
        # generate expected output - each data block stored in DataBlock object
        expected = []
        for block in [block_1, block_2, block_3]:
            header, *raw_data = block
            expected.append(DataBlock(header, "\n".join(raw_data), {}))

        p = CIFParser("/some_directory/some_file.cif")
        p._extract_data_blocks()
        assert p.data_blocks == expected

    def test_semicolon_data_items_are_assigned(self):
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
            ";"
            "_data_name_4 data_value_4",
            "_data_name_5",
            ";",
            "semicolon text ; field containing ;;; semicolons",
            ";"
        ]
        data_block = DataBlock('data_block_header', "\n".join(contents), {})
        semicolon_data_items = {
            "data_name_1": "'very long semicolon text field with many words'",
            "data_name_3": "'semicolon text field with\ntwo lines of text'",
            "data_name_5": "'semicolon text ; field containing ;;; semicolons'"
        }

        data_block.extract_data_items(SEMICOLON_DATA_ITEM)
        assert data_block.data_items == semicolon_data_items

    def test_semicolon_data_items_are_stripped_out(self):
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
            ";"
        ]
        data_block = DataBlock('data_block_header', "\n".join(contents), {})
        expected_remaining_data = "\n".join([contents[0], contents[5]])

        data_block.extract_data_items(SEMICOLON_DATA_ITEM)
        assert data_block.raw_data == expected_remaining_data

    def test_inline_declared_variables_are_assigned(self):
        data_items = {
            "data_name": "value",
            "four_word_data_name": "four_word_data_value",
            "data_name-with_hyphens-in-it": "some_data_value",
            "data_name_4": "'data value inside single quotes'",
            "data_name_5": '"data value inside double quotes"'
        }
        contents = ['_{} {}'.format(data_name, data_value)
                    for data_name, data_value in data_items.items()]
        data_block = DataBlock('data_block_header', "\n".join(contents), {})

        data_block.extract_data_items(INLINE_DATA_ITEM)
        assert data_block.data_items == data_items

    def test_inline_declared_variables_are_stripped_out(self):
        contents = [
            "_data_name_1 value",
            "_DatA_name-two another_value",
            "loop_",
            "_loop_data_name_A",
            "_loop_data_name_B",
            "value_A1 'value A2'",
            "value-B1 value_B2",
            "_one_more_data_item_ one_more_data_value"
        ]
        data_block = DataBlock('data_block_header', "\n".join(contents), {})
        expected_remaining_data = "\n" + "\n".join(contents[2:7])

        data_block.extract_data_items(INLINE_DATA_ITEM)
        assert data_block.raw_data == expected_remaining_data

    def test_variables_declared_in_loop_are_assigned(self):
        data_items = {
            "number": ["1", "2222", "3456789"],
            "symbol": [".", "-", "?"],
            "number_and_symbol": ["-1.0", "2.0(3)", "3.0e10"],
            "letter": ["a", "bbb", "cdefghi"],
            "letter_and_symbol": ["x?*", "abc_(rt)", "sin(3*10^3)"],
            "single_quotes": ["'x y z'", "'s = 3.2(3)'", "'x -y+2/3 z-0.876'"],
            "double_quotes": ['"a b c"', '"s = 4.6(1)"', '"x-1/3 y+0.34 -z"']
        }
        # convert the data items into corresponding CIF input
        contents = ["loop_"]
        data_names = data_items.keys()
        contents.extend('_' + data_name for data_name in data_names)
        contents.extend('{} {} {} {} {} {} {}'.format(
            *[data_items[data_name][i] for data_name in data_names])
            for i in range(3))
        data_block = DataBlock('data_block_header', "\n".join(contents), {})

        data_block.extract_loop_data_items()
        assert data_block.data_items['loop_1'] == data_items

    def test_parse_method_calls_in_correct_order(self):
        p = mock.Mock(spec=CIFParser)
        data_block = mock.Mock(spec=DataBlock)
        p.data_blocks = [data_block]
        CIFParser.parse(p)
        expected_calls = [
            mock.call._strip_comments_and_blank_lines(),
            mock.call._extract_data_blocks(),
            mock.call.extract_data_items(SEMICOLON_DATA_ITEM),
            mock.call.extract_data_items(INLINE_DATA_ITEM),
            mock.call.extract_loop_data_items()
        ]
        assert p.method_calls + data_block.method_calls == expected_calls


class TestSavingFile:
    data_items_1 = {"data_name_1": "data_value_1",
                    "data_name_2": "data_value_2",
                    "loop_1": {"loop_data_name_A": ["A1", "A2", "A3"],
                               "loop_data_name_B": ["B1", "B2", "B3"]
                               },
                    "loop_2": {"loop_data_name_A": ["A1", "A2", "A3"],
                               "loop_data_name_C": ["C1", "C2", "C3"]
                               },
                    }
    data_items_2 = {"data_name_1": "data_value_1",
                    "data_name_3": "data_value_3",
                    }

    example_data_blocks = [DataBlock("data_block_1", "", data_items_1),
                           DataBlock("data_block_2", "", data_items_2)]

    def test_data_written_as_valid_json(self, mocker):
        m = mocker.patch(OPEN, mock.mock_open())
        filepath = "some_directory/some_file.json"
        p = mock.Mock(spec=CIFParser)
        p.data_blocks = self.example_data_blocks
        CIFParser.save(p, filepath)

        # test correct file is written to exactly once
        open.assert_called_with(filepath, "w")
        assert m().write.call_count == 1
        # test content is valid json
        write_call_json = m().write.call_args[0][0]
        assert json.loads(write_call_json)

    def test_contents_are_sorted_and_stored_correctly(self, tmpdir):
        filepath = str(tmpdir.join("temp.json"))
        p = mock.Mock(spec=CIFParser)
        p.data_blocks = self.example_data_blocks
        CIFParser.save(p, filepath)

        with open(filepath, 'r') as json_file:
            data = json.load(json_file, object_pairs_hook=OrderedDict)
        assert list(data.keys()) == ["data_block_1", "data_block_2"]
        assert data["data_block_1"] == OrderedDict(
            sorted(self.data_items_1.items()))
        assert data["data_block_2"] == OrderedDict(
            sorted(self.data_items_2.items()))


class TestCIFSyntaxExceptions:
    valid_comments = [
        "# some comment",
        "# another comment - next line blank"
        "                      ",
        "# final comment ## with hashes ## in"
    ]
    valid_inline_items = [
        "data_block_header_1",
        "_data_name_1 value",
        "_DatA_name-two another_value"
    ]
    valid_loop = [
        "loop_",
        "# comment inside a loop before data names- next line blank",
        "                      ",
        "_loop_data_name_A",
        "_loop_data_name_B",
        "# comment inside loop after data names",
        "value_A1 'value A2'",
    ]
    valid_semicolon_field = [
        "_data_name_4",
        ";",
        "semicolon text field with",
        "two lines of text",
        ";"
    ]

    def test_error_throws_correct_exception_with_message(self):
        message = "Oh no! An exception has been raised...."

        with pytest.raises(CIFParseError) as exception_info:
            v = mock.Mock(spec=CIFValidator)
            v.error = CIFValidator.error
            v.error(v, message, 1, "Erroneous line")
        assert str(exception_info.value) == \
               '{} on line 1: "Erroneous line"'.format(message)

    @pytest.mark.parametrize("valid_contents", [valid_comments,
                                                valid_inline_items,
                                                valid_loop,
                                                valid_semicolon_field])
    def test_valid_syntax_raises_no_exception(self, valid_contents):
        v = CIFValidator("\n".join(valid_contents))
        assert v.validate() == True

    def test_warning_if_file_is_empty(self):
        # test when file is empty
        with pytest.warns(UserWarning):
            CIFValidator("")

        # test when is only whitespace
        with pytest.warns(UserWarning):
            CIFValidator("     \n    \n  \t \t \n   ")

    @pytest.mark.parametrize("invalid_line", ["value_with_missing_data_name",
                                              "  starting_with_whitespace",
                                              "'in single quotes'",
                                              '"in double quotes"'])
    def test_error_if_missing_inline_data_name(self, invalid_line):
        contents = [
            "_data_name_1 value_1",
            "_data_name_2 value_2",
        ]
        contents.insert(1, invalid_line)
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert str(exception_info.value) == \
               'Missing inline data name on line 2: "{}"'.format(invalid_line)

    def test_error_if_invalid_inline_data_value(self):
        contents = [
            "_data_name_1 value_1",
            "_data_name_2 ",
            "_data_name_3 value_3"
        ]

        # test when final line of file
        v = CIFValidator("\n".join(contents[:2]))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert str(exception_info.value) == \
               'Invalid inline data value on line 2: "_data_name_2 "'

        # test when followed by another line
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert str(exception_info.value) == \
               'Invalid inline data value on line 2: "_data_name_2 "'

    @pytest.mark.parametrize("invalid_line", ["value_A1",
                                              "value_A1 value_B1 value_C1"])
    def test_error_if_unmatched_data_items_in_loop(self, invalid_line):
        contents = [
            "loop_",
            "_data_name_A",
            "_data_name_B ",
            "value_A1 value_B1"
        ]
        contents.insert(4, invalid_line)
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert str(exception_info.value) == \
               ('Unmatched data values to data names in loop '
                'on line 5: "{}"'.format(invalid_line))

    def test_error_if_semicolon_data_item_not_closed(self):
        contents = [
            "_data_name_1",
            ";",
            "# some comment inside the text field",
            "Unclosed text field",
            "_data_item_2"
        ]
        # test when field is terminated by end of file
        v = CIFValidator("\n".join(contents[:4]))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert str(exception_info.value) == \
               'Unclosed semicolon text field on line 4: "Unclosed text field"'

        # test when field is terminated by another data item
        v = CIFValidator("\n".join(contents))

        with pytest.raises(CIFParseError) as exception_info:
            v.validate()
        assert str(exception_info.value) == \
               'Unclosed semicolon text field on line 4: "Unclosed text field"'
