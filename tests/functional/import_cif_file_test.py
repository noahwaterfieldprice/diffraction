import glob

import pytest

from diffraction import cif


class TestCIFReading:
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
        assert len(data_items) == 25
        # check the loops operated correctly
        assert len(data_items["symmetry_equiv_pos_as_xyz"]) == 36
        assert data_items["atom_site_occupancy"] == ["1.0", "1.0", "1.0"]
        assert data_items["atom_site_aniso_label"] == ["Ca1", "C1", "O1"]
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
        assert len(data_items) == 51
        # check the loops operated correctly
        assert data_items["symmetry_equiv_pos_site_id"] == \
               [str(i) for i in range(1, 37)]
        assert data_items["atom_site_type_symbol"] == ["Ca2+", "C4+", "O2-"]
        assert data_items["atom_site_aniso_U_22"] == ["0.01775(90)"]
        assert data_items["publ_author_name"] == \
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
        assert len(data_items_1) == 39
        assert len(data_items_2) == 41
        # check loops operated correctly
        assert len(data_items_1["atom_site_label"]) == 119
        assert len(data_items_2["atom_site_label"]) == 69
        assert data_items_1["symmetry_equiv_pos_site_id"] == ["1", "2"]
        assert data_items_2["symmetry_equiv_pos_site_id"] == ["1", "2"]
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

    @pytest.mark.parametrize("filepath", glob.glob(
        'tests/functional/static/invalid_cifs/*'))
    def test_loading_invalid_cif_file_raises_exception(self, filepath):
        with pytest.raises(CIFParseError):
            load_cif(filepath)


class TestCreatingCrystal:
    @pytest.mark.parametrize("filepath", glob.glob(
        'tests/functional/static/valid_cifs/*'))
    def test_can_load_crystal_structure_from_cif_file(self, filepath):
        crystal = cif.load_cif(filepath)
        assert isinstance(crystal, Crystal)
