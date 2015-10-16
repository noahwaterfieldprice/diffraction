from unittest import mock

import pytest

from diffraction.cif import load_cif, CIFParser


class TestLoadingFile:
    def test_load_cif_opens_correct_file(self, mocker):
        mocker.patch("builtins.open")
        filepath = "/tmp/some_file.cif"
        load_cif(filepath)
        open.assert_called_with(filepath, "r")

    def test_raises_warning_if_file_extension_is_not_cif(self, mocker):
        mocker.patch("builtins.open")
        non_cif_filepath = "/tmp/some_file.notcif"
        with pytest.warns(UserWarning):
            load_cif(non_cif_filepath)


class TestParsingFile:
    def test_file_contents_are_stored_as_raw_string_attribute(self, mocker):
        contents = [
            "Here is the first line",
            "Here is the second",
            "And here is the third line",
        ]

        mocker.patch("builtins.open",
                     mock.mock_open(read_data='\n'.join(contents)))

        filepath = "/tmp/some_file.cif"
        p = CIFParser(filepath)
        # make sure correct file was loaded
        open.assert_called_with(filepath, "r")
        assert p.raw_data == '\n'.join(contents)

    def test_comments_and_blank_lines_are_stripped_out(self, mocker):
        contents = [
            "# Here is a comment the first line",
            "# Here is another comment. The next line is blank",
            "\t\t\t\t\t",
            "Here is the only normal line",
            '# Final comment ## with # extra hashes ### in ##'
        ]

        mocker.patch("builtins.open",
                     mock.mock_open(read_data='\n'.join(contents)))

        filepath = "/tmp/some_file.cif"
        p = CIFParser(filepath)
        assert p.raw_data == "Here is the only normal line"
