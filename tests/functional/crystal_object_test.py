import pytest

from diffraction import Crystal


class TestCreatingFromSequence:
    def test_can_create_from_sequence(self):
        calcite = Crystal([4.99, 4.99, 17.002, 90, 90, 120], "R -3 c H")

        assert calcite.a == 4.99
        assert calcite.b == 4.99
        assert calcite.c == 17.002
        assert calcite.alpha == 90
        assert calcite.beta == 90
        assert calcite.gamma == 120
        assert calcite.space_group == "R -3 c H"


class TestCreatingFromCIF:
    def test_can_create_crystal_from_single_datablock_cif(self):
        calcite = Crystal.from_cif("tests/functional/static/valid_cifs/calcite_icsd.cif")

        assert calcite.a == 4.99
        assert calcite.b == 4.99
        assert calcite.c == 17.002
        assert calcite.alpha == 90
        assert calcite.beta == 90
        assert calcite.gamma == 120
        assert calcite.space_group == "R -3 c H"

    def test_error_if_lattice_parameter_is_missing_from_cif(selfs):
        with pytest.raises(ValueError) as exception_info:
            Crystal.from_cif(
                "tests/functional/static/invalid_cifs/calcite_icsd_missing_lattice_parameter.cif")
        assert str(exception_info.value) == "cell_length_b missing from input CIF file"

    def test_error_datablock_not_given_for_multi_data_block_cif(self):
        with pytest.raises(TypeError) as exception_info:
            Crystal.from_cif("tests/functional/static/valid_cifs/multi_data_block.cif")
        assert str(exception_info.value) == \
            ("__init__() missing keyword argument: 'data_block'. "
             "Required when input CIF has multiple data blocks.")

    def test_can_create_crystal_from_multi_data_block_cif(self):
        CHFeNOS = Crystal.from_cif(
            "tests/functional/static/valid_cifs/multi_data_block.cif",
            data_block="data_CSD_CIF_ACAKOF")

        assert CHFeNOS.a == 6.1250
        assert CHFeNOS.b == 9.2460
        assert CHFeNOS.c == 10.147
        assert CHFeNOS.alpha == 77.16
        assert CHFeNOS.beta == 83.44
        assert CHFeNOS.gamma == 80.28
        assert CHFeNOS.space_group == "P -1"


class TestCreatingFromMapping:
    def test_can_create_crystal_from_dictionary(self):
        crystal_info = {"a": 4.99, "b": 4.99, "c": 17.002,
                        "alpha": 90, "beta": 90, "gamma": 120,
                        "space_group": "R -3 c H"}
        calcite = Crystal.from_dict(crystal_info)

        assert calcite.a == 4.99
        assert calcite.b == 4.99
        assert calcite.c == 17.002
        assert calcite.alpha == 90
        assert calcite.beta == 90
        assert calcite.gamma == 120
        assert calcite.space_group == "R -3 c H"

    def test_error_if_lattice_parameter_missing_from_dict(self):
        crystal_info = {"a": 4.99, "c": 17.002,
                        "alpha": 90, "beta": 90, "gamma": 120,
                        "space_group": "R -3 c H"}
        with pytest.raises(ValueError) as exception_info:
            calcite = Crystal.from_dict(crystal_info)
        assert str(exception_info.value) == "b missing from input dictionary"

    def test_error_if_space_group_missing_from_dict(self):
        crystal_info = {"a": 4.99, "b": 4.99, "c": 17.002,
                        "alpha": 90, "beta": 90, "gamma": 120}
        with pytest.raises(ValueError) as exception_info:
            calcite = Crystal.from_dict(crystal_info)
        assert str(exception_info.value) == "space_group missing from input dictionary"
