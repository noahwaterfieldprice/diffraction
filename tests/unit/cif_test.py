from unittest import mock

import pytest

from diffraction.cif import load_cif, CIFParser

OPEN = "builtins.open"


class TestLoadingFile:
    def test_load_cif_opens_correct_file(self, mocker):
        mocker.patch(OPEN)
        filepath = "/some_directory/some_file.cif"
        load_cif(filepath)
        open.assert_called_with(filepath, "r")

    def test_raises_warning_if_file_extension_is_not_cif(self, mocker):
        mocker.patch(OPEN)
        non_cif_filepath = "/some_directory/some_file.notcif"
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
            "Here is the only normal line",
            '# Final comment ## with # extra hashes ### in ##'
        ]
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        p = CIFParser("/some_directory/some_file.cif")
        p.strip_comments_and_blank_lines()
        assert p.raw_data == "Here is the only normal line"

    def test_inline_declared_variables_are_assigned(self, mocker):
        data_items = {
            "data_name": "value",
            "four_word_data_name": "four_word_data_value",
            "data_name-with_hyphens-in-it": "some_data_value",
            "data_name_4": "'data value inside single quotes'",
            "data_name_5": '"data value inside double quotes"'
        }
        contents = ['_{} {}'.format(data_name, data_value)
                    for data_name, data_value in data_items.items()]
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        p = CIFParser("/some_directory/some_file.cif")
        p.extract_inline_data_items()
        assert p.data_items == data_items

    def test_variables_declared_in_loop_are_assigned(self, mocker):
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
        mocker.patch(OPEN, mock.mock_open(read_data='\n'.join(contents)))

        p = CIFParser("/some_directory/some_file.cif")
        p.extract_loop_data_items()
        assert p.data_items == data_items
