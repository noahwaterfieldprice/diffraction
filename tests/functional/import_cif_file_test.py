import pytest

from diffraction import cif


class TestCIFReading:

    invalid_files = [
        "calcite_icsd_missing_data_name_in_loop.cif",
        "calcite_icsd_missing_data_value_in_loop.cif",
        "calcite_icsd_missing_data_name_inline.cif",
        "calcite_icsd_missing_data_value_inline.cif",
        "calcite_icsd_duplicate_site_id.cif",
        "calcite_icsd_missing_site.cif",
        "calcite_icsd_duplicate_site_position.cif"
    ]

    error_messages = [
        'Unmatched data values to data names in loop on line 81: "Ca2+ 2"',
        'Unmatched data values to data names in loop on line 77: "36"',
        'Missing inline data name on line 29: " 4.9900(2)"',
        'Invalid inline data value on line 28: "_cell_length_a"',
        '',
        '',
        ''
    ]

    def test_loading_cif_from_invalid_filepath_raises_exception(self):
        with pytest.raises(FileNotFoundError):
            cif.load_cif('/no/cif/file/here')

    def test_can_load_crystal_data_from_vesta_cif_file(self):
        p = cif.CIFParser(
            "tests/functional/static/valid_cifs/calcite_vesta.cif")
        p.parse()
        # basic checks that correct number of data items were caught
        assert p.data_blocks[0].heading == 'data_VESTA_phase_1'
        assert len(p.data_blocks) == 1
        data_items = p.data_blocks[0].data_items
        assert len(data_items) == 12
        # check the loops operated correctly
        assert len(data_items["loop_1"]["symmetry_equiv_pos_as_xyz"]) == 36
        assert data_items["loop_2"]["atom_site_occupancy"] == ["1.0", "1.0",
                                                               "1.0"]
        assert data_items["loop_3"]["atom_site_aniso_label"] == ["Ca1", "C1",
                                                                 "O1"]
        # check a few inline data items
        assert data_items["cell_length_a"] == "4.9900(2)"
        assert data_items["symmetry_space_group_name_H-M"] == "'R -3 c'"
        assert data_items["symmetry_Int_Tables_number"] == "167"

    def test_can_load_crystal_data_from_icsd_cif_file(self):
        p = cif.CIFParser(
            "tests/functional/static/valid_cifs/calcite_icsd.cif")
        p.parse()
        # basic checks that correct number of data items were caught
        assert p.data_blocks[0].heading == "data_18166-ICSD"
        assert len(p.data_blocks) == 1
        data_items = p.data_blocks[0].data_items
        assert len(data_items) == 27
        # check the loops operated correctly
        assert data_items["loop_3"]["symmetry_equiv_pos_site_id"] == \
               [str(i) for i in range(1, 37)]
        assert data_items["loop_5"]["atom_site_label"] == ["Ca1", "C1", "O1"]
        assert data_items["loop_6"]["atom_site_aniso_U_22"] == ["0.01775(90)"]
        assert data_items["loop_2"]["publ_author_name"] == \
               ["'Chessin, H.'", "'Hamilton, W.C.'", "'Post, B.'"]
        # check a few inline data items
        assert data_items["cell_length_a"] == "4.9900(2)"
        assert data_items["chemical_name_mineral"] == "Calcite"
        assert data_items["cell_formula_units_Z"] == "6"

    def test_can_load_crystal_data_from_multi_data_block_cif_file(self):
        p = cif.CIFParser(
            "tests/functional/static/valid_cifs/multi_data_block.cif")
        p.parse()
        # basic checks that correct number of data items were caught
        assert len(p.data_blocks) == 20
        assert p.data_blocks[0].heading == "data_CSD_CIF_ACAGUG"
        assert p.data_blocks[11].heading == "data_CSD_CIF_AHUKOD"
        data_items_1 = p.data_blocks[0].data_items
        data_items_2 = p.data_blocks[11].data_items
        assert len(data_items_1) == 33
        assert len(data_items_2) == 35
        # check loops operated correctly
        assert len(data_items_1["loop_4"]["atom_site_label"]) == 119
        assert len(data_items_2["loop_4"]["atom_site_label"]) == 69
        assert data_items_1["loop_3"]["atom_type_radius_bond"] == \
               ["0.68", "0.23", "1.35", "0.68", "1.02"]
        assert data_items_2["loop_3"]["atom_type_radius_bond"] == \
               ["0.68", "0.23", "1.21", "0.64", "1.40", "1.02"]
        # check semicolon text fields assigned correctly
        assert data_items_1["refine_special_details"] == \
               "'One of the water molecules is disordered over two sites.'"
        assert data_items_2["chemical_name_systematic"] == \
               ("'tris(bis(Ethylenedithio)tetrathiafulvalene) \n"
                "2,5-difluoro-1,4-bis(iodoethynyl)benzene bromide'")
        # check a few inline data items
        assert data_items_1["journal_year"] == "2001"
        assert data_items_1["exptl_crystal_colour"] == "'dark brown'"
        assert data_items_2["journal_name_full"] == "'J.Mater.Chem. '"
        assert data_items_2["cell_angle_gamma"] == "76.35(2)"

    @pytest.mark.parametrize("filename, error_message",
                             zip(invalid_files, error_messages))
    def test_exception_with_invalid_cif_file(self, filename, error_message):
        filepath = "tests/functional/static/invalid_cifs/" + filename
        with pytest.raises(cif.CIFParseError) as exception_info:
            p = cif.CIFParser(filepath)
            p.validate()
        assert str(exception_info.value) == error_message
