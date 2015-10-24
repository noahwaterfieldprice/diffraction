import os
import json
from collections import OrderedDict
from unittest import mock

import pytest

from diffraction.cif import (load_cif, CIFParseError, CIFParser, CIFValidator,
                             DataBlock, SEMICOLON_DATA_ITEM, INLINE_DATA_ITEM,
                             LOOP_NAMES)

OPEN = "builtins.open"


class TestLoadingFile:
    def test_load_cif_opens_correct_file(self, mocker):
        mocker.patch(OPEN)
        filepath = "/some_directory/some_file.cif"
        load_cif(filepath)
        open.assert_called_with(filepath, "r")

    def test_raises_warning_if_file_extension_is_not_cif(self, mocker):
        mocker.patch(OPEN)
        non_cif_filepath = "/some_directory/some_file.not_cif"
        with pytest.warns(UserWarning):
            load_cif(non_cif_filepath)


class TestParsingFile:
    def test_file_contents_are_stored_as_raw_string_attribute(self, mocker):
        contents = [
            "Here is the first line",
            "Here is the second",
            "And here is the third line",
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
            "Here is one normal line. Previous line was blank",
            "  Here's another normal line starting with whitespace",
            '# Final comment ## with # extra hashes ### in ##'
        ]
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))
        expected_remaining_lines = contents[4:6]

        p = CIFParser("/some_directory/some_file.cif")
        p.strip_comments_and_blank_lines()
        assert p.raw_data == "\n".join(expected_remaining_lines)

    def test_file_split_by_data_blocks(self, mocker):
        block_1 = [
            "data_block_heading",
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
            heading, *raw_data = block
            expected.append(DataBlock(heading, "\n".join(raw_data), {}))

        p = CIFParser("/some_directory/some_file.cif")
        p.extract_data_blocks()
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
            "_data_name_4 data_value_4"
        ]
        data_block = DataBlock('data_block_heading', "\n".join(contents), {})
        semicolon_data_items = {
            "data_name_1": "'very long semicolon text field with many words'",
            "data_name_3": "'semicolon text field with\ntwo lines of text'"
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
        data_block = DataBlock('data_block_heading', "\n".join(contents), {})
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
        data_block = DataBlock('data_block_heading', "\n".join(contents), {})

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
        data_block = DataBlock('data_block_heading', "\n".join(contents), {})
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
        data_block = DataBlock('data_block_heading', "\n".join(contents), {})

        data_block.extract_loop_data_items()
        assert data_block.data_items['loop_1'] == data_items

    def test_parse_method_calls_in_correct_order(self):
        p = mock.Mock(spec=CIFParser)
        data_block = mock.Mock(spec=DataBlock)
        p.data_blocks = [data_block]
        CIFParser.parse(p)
        expected_calls = [
            mock.call.strip_comments_and_blank_lines(),
            mock.call.extract_data_blocks(),
            mock.call.extract_data_items(SEMICOLON_DATA_ITEM),
            mock.call.extract_data_items(INLINE_DATA_ITEM),
            mock.call.extract_loop_data_items()
        ]
        assert p.method_calls + data_block.method_calls == expected_calls


class TestBasicDataInterpretation:
    @pytest.mark.parametrize("key_data_name, loop_name", LOOP_NAMES)
    def test_key_loops_are_renamed(self, key_data_name, loop_name):
        contents = [
            "loop_",
            "data_name_1"
        ]
        contents.insert(1, "_{}".format(key_data_name))
        data_block = DataBlock("data_block_heading", "\n".join(contents), {})

        data_block.extract_loop_data_items()
        assert list(data_block.data_items.keys())[0] == loop_name


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


class TestReadingExceptions:
    def test_error_throws_correct_exception_with_message(self):
        message = "Oh no! An exception has been raised...."

        with pytest.raises(CIFParseError) as exception_info:
            v = mock.Mock(spec=CIFValidator)
            v.error = CIFValidator.error
            v.error(v, message, 1, "Erroneous line")
        assert str(exception_info.value) == \
               '{} on line 1: "Erroneous line"'.format(message)

    def test_validation_returns_true_when_no_exception_raised(self, mocker):
        contents = (
            "# some comment",
            "# another comment - next line blank"
            "                      ",
            "data_block_heading_1",
            "_data_name_1 value",
            "_DatA_name-two another_value",
            "loop_",
            "_loop_data_name_A",
            "_loop_data_name_B",
            "value_A1 'value A2'",
            "loop_",
            "_loop_data_name_C",
            "value_C1",
            "_one_more_data_item_ one_more_data_value",
            "_data_name_4",
            ";",
            "semicolon text field with",
            "two lines of text",
            ";"
        )
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        p = CIFParser("some_directory/valid_cif_file.cif")
        p.validate()

    @pytest.mark.parametrize("invalid_line", ["value_with_missing_data_name",
                                              "  starting_with_whitespace",
                                              "'in single quotes'",
                                              '"in double quotes"'])
    def test_error_if_missing_inline_data_name(self, mocker, invalid_line):
        contents = [
            "_data_name_1 value_1",
            "_data_name_2 value_2",
        ]
        contents.insert(1, invalid_line)
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        p = CIFParser("some_directory/missing_inline_data_name.cif")
        with pytest.raises(CIFParseError) as exception_info:
            p.validate()
        assert str(exception_info.value) == \
               'Missing inline data name on line 2: "{}"'.format(invalid_line)

    def test_error_if_invalid_inline_data_value(self, mocker):
        contents = [
            "_data_name_1 value_1",
            "_data_name_2 "
        ]
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        p = CIFParser("some_directory/invalid_inline_data_value.cif")
        with pytest.raises(CIFParseError) as exception_info:
            p.validate()
        assert str(exception_info.value) == \
               'Invalid inline data value on line 2: "_data_name_2 "'

    @pytest.mark.parametrize("invalid_line", ["value_A1",
                                              "value_A1 value_B1 value_C1"])
    def test_error_if_unmatched_data_items_in_loop(self, mocker, invalid_line):
        contents = [
            "loop_",
            "_data_name_A",
            "_data_name_B ",
            "value_A1 value_B1"
        ]
        contents.insert(4, invalid_line)
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        p = CIFParser("some_directory/unmatched_loop_data_items.cif")
        with pytest.raises(CIFParseError) as exception_info:
            p.validate()
        assert str(exception_info.value) == \
               ('Unmatched data values to data names in loop '
                'on line 5: "{}"'.format(invalid_line))

    def test_error_if_semicolon_data_item_not_closed(self, mocker):
        contents = [
            "_data_name_1",
            ";",
            "Unclosed text field",
            "_data_item_2"
        ]
        # test when field is terminated by end of file
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents[:3])))
        p = CIFParser("some_directory/unclosed_semicolon_field.cif")

        with pytest.raises(CIFParseError) as exception_info:
            p.validate()
        assert str(exception_info.value) == \
               'Unclosed semicolon text field on line 3: "Unclosed text field"'

        # test when field is terminated by another data item
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))
        p = CIFParser("some_directory/unclosed_semicolon_field.cif")

        with pytest.raises(CIFParseError) as exception_info:
            p.validate()
        assert str(exception_info.value) == \
               'Unclosed semicolon text field on line 3: "Unclosed text field"'
