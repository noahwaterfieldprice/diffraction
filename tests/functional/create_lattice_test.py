import pytest

from diffraction import DirectLattice


class TestCreatingDirectLatticeFromSequence:
    def test_can_create_from_sequence(self):
        lattice = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])

        assert lattice.a == 4.99
        assert lattice.b == 4.99
        assert lattice.c == 17.002
        assert lattice.alpha == 90
        assert lattice.beta == 90
        assert lattice.gamma == 120

    def test_error_if_lattice_parameter_missing_from_sequence(self):

        with pytest.raises(ValueError) as exception_info:
            lattice = DirectLattice([4.99, 17.002, 90, 90, 120])
        assert str(exception_info.value) == "Missing lattice parameter from input"


class TestCreatingDirectLatticeFromMapping:
    def test_can_create_crystal_from_dictionary(self):
        lattice_parameters = {"a": 4.99, "b": 4.99, "c": 17.002,
                              "alpha": 90, "beta": 90, "gamma": 120}
        lattice = DirectLattice.from_dict(lattice_parameters)

        assert lattice.a == 4.99
        assert lattice.b == 4.99
        assert lattice.c == 17.002
        assert lattice.alpha == 90
        assert lattice.beta == 90
        assert lattice.gamma == 120

    def test_error_if_lattice_parameter_missing_from_dict(self):
        lattice_parameters = {"a": 4.99, "c": 17.002,
                              "alpha": 90, "beta": 90, "gamma": 120}
        with pytest.raises(ValueError) as exception_info:
            lattice = DirectLattice.from_dict(lattice_parameters)
        assert str(exception_info.value) == "Parameter: 'b' missing from input dictionary"


class TestCreatingFromCIF:
    def test_can_create_crystal_from_single_datablock_cif(self):
        lattice = DirectLattice.from_cif("tests/functional/static/valid_cifs/calcite_icsd.cif")

        assert lattice.a == 4.99
        assert lattice.b == 4.99
        assert lattice.c == 17.002
        assert lattice.alpha == 90
        assert lattice.beta == 90
        assert lattice.gamma == 120

    def test_error_if_lattice_parameter_is_missing_from_cif(selfs):
        with pytest.raises(ValueError) as exception_info:
            DirectLattice.from_cif(
                "tests/functional/static/invalid_cifs/calcite_icsd_missing_lattice_parameter.cif")
        assert str(exception_info.value) == \
            "Parameter: 'cell_length_b' missing from input CIF"

    def test_error_datablock_not_given_for_multi_data_block_cif(self):
        with pytest.raises(TypeError) as exception_info:
            DirectLattice.from_cif("tests/functional/static/valid_cifs/multi_data_block.cif")
        assert str(exception_info.value) == \
            ("__init__() missing keyword argument: 'data_block'. "
             "Required when input CIF has multiple data blocks.")

    def test_can_create_direct_lattice_from_multi_data_block_cif(self):
        CHFeNOS = DirectLattice.from_cif(
            "tests/functional/static/valid_cifs/multi_data_block.cif",
            data_block="data_CSD_CIF_ACAKOF")

        assert CHFeNOS.a == 6.1250
        assert CHFeNOS.b == 9.2460
        assert CHFeNOS.c == 10.147
        assert CHFeNOS.alpha == 77.16
        assert CHFeNOS.beta == 83.44
        assert CHFeNOS.gamma == 80.28
