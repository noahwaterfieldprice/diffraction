from unittest import mock

import pytest

from diffraction.cif import load_cif, CIFParser, DataBlock

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
            "# Here is another comment. The next line is blank",
            "\t\t\t\t\t",
            "Here is one normal line",
            "  Here's another normal line starting with whitespace",
            '# Final comment ## with # extra hashes ### in ##'
        ]
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))
        expected_remaining_lines = contents[3:5]
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

    def test_semicolon_data_fields_are_assigned(self, mocker):
        contents = [
            "_data_name_1",
            ";",
            "very long semicolon text field with lots of info",
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
            "data_name_1": "'very long semicolon text field with lots of info'",
            "data_name_3": "'semicolon text field with two lines of text'"
        }

        CIFParser.extract_semicolon_data_items(data_block)
        assert data_block.data_items == semicolon_data_items

    def test_semicolon_data_fields_are_stripped_out(self):
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

        CIFParser.extract_semicolon_data_items(data_block)
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

        CIFParser.extract_inline_data_items(data_block)
        assert data_block.data_items == data_items

    def test_inline_declared_variables_are_stripped_out(self):
        contents = [
            "_data_name_1 value",
            "_DatA_name-two another_value",
            "loop_",
            "_loop_data_name_A",
            "_loop_data_name_B",
            "value_A1 'value A2'",
            "value-B1 value_B2"
            "_one_more_data_item_ one_more_data_value"
        ]
        data_block = DataBlock('data_block_heading', "\n".join(contents), {})

        expected_remaining_data = "\n".join(contents[2:7])
        CIFParser.extract_inline_data_items(data_block)
        assert data_block.raw_data == expected_remaining_data

    def test_variables_declared_in_loop_are_assigned(self):
        data_items = {
            "number": ["1", "2222", "3456789"],
            "symbol": [".", "-", "_"],
            "number_and_symbol": ["-1.0", "2.0(3)", "3.0e10"],
            "letter": ["a", "bbb", "cdefghi"],
            "single_quotes": ["'x y z'", "'s = 3.2(3)'", "'x -y+2/3 z-0.8796'"],
            "double_quotes": ['"a b c"', '"s = 4.6(1)"', '"x-1/3 y+0.34 -z"']
        }
        # convert the data items into corresponding CIF input
        contents = ["loop_"]
        data_names = data_items.keys()
        contents.extend('_' + data_name for data_name in data_names)
        contents.extend('{} {} {} {} {} {}'.format(
            *[data_items[data_name][i] for data_name in data_names])
                        for i in range(3))
        data_block = DataBlock('data_block_heading', "\n".join(contents), {})

        CIFParser.extract_loop_data_items(data_block)
        assert data_block.data_items == data_items

    def test_parse_method_calls_in_correct_order(self):
        p = mock.Mock(spec=CIFParser)
        p.parse = CIFParser.parse
        p.data_blocks = ["data_block_1", "data_block_2"]
        p.parse(p)
        expected_calls = [
            mock.call.strip_comments_and_blank_lines(),
            mock.call.extract_data_blocks(),
            mock.call.extract_semicolon_data_items("data_block_1"),
            mock.call.extract_inline_data_items("data_block_1"),
            mock.call.extract_loop_data_items("data_block_1"),
            mock.call.extract_semicolon_data_items("data_block_2"),
            mock.call.extract_inline_data_items("data_block_2"),
            mock.call.extract_loop_data_items("data_block_2")
        ]
        assert p.method_calls == expected_calls
