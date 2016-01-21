import glob

import pytest

from diffraction import load_cif, validate_cif, CIFParseError
from diffraction.cif.cif import CIFParser


class TestFileLoading:
    def test_raises_warning_if_file_extension_is_not_cif(self):
        cif_filepath = "tests/functional/static/valid_cifs/non_cif_extension.txt"

        # test for warning on validating cif
        with pytest.warns(UserWarning):
            validate_cif(cif_filepath)
        # test for warning on loading cif
        with pytest.warns(UserWarning):
            load_cif(cif_filepath)

    def test_loading_cif_from_invalid_filepath_raises_exception(self):
        cif_filepath = "/no/cif/file/here"

        # test for exception on validating cif
        with pytest.raises(FileNotFoundError):
            validate_cif(cif_filepath)
        # test for exception on loading cif
        with pytest.raises(FileNotFoundError):
            load_cif(cif_filepath)


class TestCIFValidating:
    invalid_files = [
        "calcite_icsd_missing_data_name_in_loop.cif",
        "calcite_icsd_missing_data_value_in_loop.cif",
        "calcite_icsd_missing_data_name_inline.cif",
        "calcite_icsd_missing_data_value_inline.cif",
    ]
    error_messages = [
        'Unmatched data values to data names in loop on line 81: "Ca2+ 2"',
        'Unmatched data values to data names in loop on line 77: "36"',
        'Missing inline data name on line 29: " 4.9900(2)"',
        'Invalid inline data value on line 28: "_cell_length_a"',
    ]

    @pytest.mark.parametrize("filename, error_message", zip(invalid_files, error_messages))
    def test_exception_with_invalid_cif(self, filename, error_message):
        filepath = "tests/functional/static/invalid_cifs/" + filename
        with pytest.raises(CIFParseError) as exception_info:
            validate_cif(filepath)
        assert str(exception_info.value) == error_message

    @pytest.mark.parametrize("filepath", glob.glob('tests/functional/static/valid_cifs/*'))
    def test_no_exception_and_return_true_with_valid_cif(self, filepath):
        assert validate_cif(filepath) == True


class TestCIFReading:
    def test_can_load_crystal_data_from_vesta_cif(self):
        p = CIFParser("tests/functional/static/valid_cifs/calcite_vesta.cif")
        p.parse()

        # basic checks that correct number of data items were caught
        assert p.data_blocks[0].header == 'data_VESTA_phase_1'
        assert len(p.data_blocks) == 1
        data_items = p.data_blocks[0].data_items
        assert len(data_items) == 25

        # check the loops operated correctly
        pos = data_items["symmetry_equiv_pos_as_xyz"]
        assert len(pos) == 36
        assert data_items["atom_site_occupancy"] == ["1.0", "1.0", "1.0"]
        assert data_items["atom_site_aniso_label"] == ["Ca1", "C1", "O1"]

        # check a few inline data items
        assert data_items["cell_length_a"] == "4.9900(2)"
        assert data_items["symmetry_space_group_name_H-M"] == "R -3 c"
        assert data_items["symmetry_Int_Tables_number"] == "167"

    def test_can_load_crystal_data_from_icsd_cif(self):
        p = CIFParser("tests/functional/static/valid_cifs/calcite_icsd.cif")
        p.parse()

        # basic checks that correct number of data items were caught
        assert p.data_blocks[0].header == "data_18166-ICSD"
        assert len(p.data_blocks) == 1
        data_items = p.data_blocks[0].data_items
        assert len(data_items) == 51

        # check the loops operated correctly
        ids = data_items["symmetry_equiv_pos_site_id"]
        assert ids == [str(i) for i in range(1, 37)]
        assert data_items["atom_site_label"] == ["Ca1", "C1", "O1"]
        assert data_items["atom_site_aniso_U_22"] == ["0.01775(90)"]
        assert data_items["publ_author_name"] == \
            ["Chessin, H.", "Hamilton, W.C.", "Post, B."]

        # check a few inline data items
        assert data_items["cell_length_a"] == "4.9900(2)"
        assert data_items["chemical_name_mineral"] == "Calcite"
        assert data_items["cell_formula_units_Z"] == "6"

    def test_can_load_crystal_data_from_multi_data_block_cif(self):
        p = CIFParser("tests/functional/static/valid_cifs/multi_data_block.cif")
        p.parse()

        # basic checks that correct number of data items were caught
        assert len(p.data_blocks) == 20
        assert p.data_blocks[0].header == "data_CSD_CIF_ACAGUG"
        assert p.data_blocks[11].header == "data_CSD_CIF_AHUKOD"
        data_items_1 = p.data_blocks[0].data_items
        data_items_2 = p.data_blocks[11].data_items
        assert len(data_items_1) == 39
        assert len(data_items_2) == 41

        # check loops operated correctly
        assert len(data_items_1["atom_site_label"]) == 119
        assert len(data_items_2["atom_site_label"]) == 69
        assert data_items_1["atom_type_radius_bond"] == \
            ["0.68", "0.23", "1.35", "0.68", "1.02"]
        assert data_items_2["atom_type_radius_bond"] == \
            ["0.68", "0.23", "1.21", "0.64", "1.40", "1.02"]

        # check semicolon text fields assigned correctly
        assert data_items_1["refine_special_details"] == \
            "One of the water molecules is disordered over two sites."
        assert data_items_2["chemical_name_systematic"] == \
            ("tris(bis(Ethylenedithio)tetrathiafulvalene) \n"
             "2,5-difluoro-1,4-bis(iodoethynyl)benzene bromide")

        # check a few inline data items
        assert data_items_1["journal_year"] == "2001"
        assert data_items_1["exptl_crystal_colour"] == "dark brown"
        assert data_items_2["journal_name_full"] == "J.Mater.Chem. "
        assert data_items_2["cell_angle_gamma"] == "76.35(2)"
