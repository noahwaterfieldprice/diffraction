import glob

import pytest

from diffraction.cif import CIFParseError, Crystal, load_cif


class TestCIFLoading:
    def test_loading_cif_from_invalid_filepath_raises_exception(self):
        with pytest.raises(FileNotFoundError):
            load_cif('/no/cif/file/here')

    @pytest.mark.parametrize("filepath", glob.glob(
        'tests/functional/static/valid_cifs/*'))
    def test_can_load_crystal_structure_from_cif_file(self):
        crystal = load_cif('tests/functional/static/calcite.cif')
        assert isinstance(crystal, Crystal)

    @pytest.mark.parametrize("filepath", glob.glob(
        'tests/functional/static/invalid_cifs/*'))
    def test_loading_invalid_cif_file_raises_exception(self, filepath):
        with pytest.raises(CIFParseError):
            load_cif(filepath)
