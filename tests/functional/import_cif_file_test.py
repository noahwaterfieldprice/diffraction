import pytest

from diffraction.cif import load_cif, CIFParseError


class TestCIFLoading:

    def test_can_load_cif_file(self):
        EXAMPLE_CIF_FILEPATH = 'static/BiFeO3.cif'
        crystal = load_cif(EXAMPLE_CIF_FILEPATH)
        assert crystal

    def test_loading_cif_from_invalid_filepath_raises_exception(self):
            INVALID_FILEPATH = '/no/cif/file/here'
            with pytest.raises(FileNotFoundError):
                load_cif(INVALID_FILEPATH)

    def test_loading_invalid_cif_file_raises_exception(self):
            INVALID_FILEPATH = '/no/cif/file/here'
            with pytest.raises(CIFParseError):
                load_cif(INVALID_FILEPATH)



