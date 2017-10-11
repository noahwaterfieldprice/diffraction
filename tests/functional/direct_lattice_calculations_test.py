import pytest
from numpy import array, sqrt
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from diffraction import DirectLattice, DirectLatticeVector, ReciprocalLattice

CALCITE_LATTICE_PARAMETERS = (4.99, 4.99, 17.002, 90.0, 90.0, 120.0)
CALCITE_DIRECT_METRIC = array([[24.9001, -12.45005, 0],
                               [-12.45005, 24.9001, 0],
                               [0, 0, 289.068004]])
CALCITE_RECIPROCAL_LATTICE_PARAMETERS = (0.2314, 0.2314, 0.0588, 90, 90, 60)


class TestCreatingDirectLatticeFromSequence:
    def test_can_create_from_sequence(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)

        assert lattice.lattice_parameters == CALCITE_LATTICE_PARAMETERS

    def test_error_if_lattice_parameter_missing_from_sequence(self):
        lattice_parameters_missing_one = CALCITE_LATTICE_PARAMETERS[:5]

        with pytest.raises(ValueError) as exception_info:
            DirectLattice(lattice_parameters_missing_one)
        assert str(exception_info.value) == "Missing lattice parameter from input"

    def test_error_if_invalid_lattice_parameter_given(self):
        invalid_lattice_parameters = CALCITE_LATTICE_PARAMETERS[:5] + ("abcdef",)

        with pytest.raises(ValueError) as exception_info:
            DirectLattice(invalid_lattice_parameters)
        assert str(exception_info.value) == "Invalid lattice parameter gamma: abcdef"


class TestCreatingDirectLatticeFromMapping:
    def test_can_create_crystal_from_dictionary(self):
        lattice_parameters = {"a": 4.99, "b": 4.99, "c": 17.002,
                              "alpha": 90, "beta": 90, "gamma": 120}
        lattice = DirectLattice.from_dict(lattice_parameters)

        assert lattice.lattice_parameters == CALCITE_LATTICE_PARAMETERS

    def test_error_if_lattice_parameter_missing_from_dict(self):
        lattice_parameters = {"a": 4.99, "c": 17.002,
                              "alpha": 90, "beta": 90, "gamma": 120}
        with pytest.raises(ValueError) as exception_info:
            DirectLattice.from_dict(lattice_parameters)
        assert str(exception_info.value) == "Parameter: 'b' missing from input dictionary"


class TestCreatingDirectLatticeFromCIF:
    def test_can_create_direct_lattice_from_single_datablock_cif(self):
        lattice = DirectLattice.from_cif("tests/functional/static/valid_cifs/calcite_icsd.cif")

        assert lattice.lattice_parameters == CALCITE_LATTICE_PARAMETERS

    def test_error_if_lattice_parameter_is_missing_from_cif(selfs):
        with pytest.raises(ValueError) as exception_info:
            DirectLattice.from_cif("tests/functional/static/invalid_cifs/"
                                   "calcite_icsd_missing_lattice_parameter.cif")
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

        assert CHFeNOS.lattice_parameters == \
            (6.1250, 9.2460, 10.147, 77.16, 83.44, 80.28)


class TestCreatingDirectLatticeFromReciprocalLattice:
    def test_can_create_direct_lattice_from_reciprocal_lattice(self):
        reciprocal_lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)
        direct_lattice = reciprocal_lattice.direct()

        assert isinstance(direct_lattice, DirectLattice)
        assert_almost_equal(direct_lattice.lattice_parameters,
                            CALCITE_LATTICE_PARAMETERS, decimal=2)


class TestDirectSpaceCalculations:
    def test_lattice_parameters_available_as_attribute(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)

        assert lattice.lattice_parameters == CALCITE_LATTICE_PARAMETERS

    def test_calculating_metric_tensor(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)

        assert_array_almost_equal(lattice.metric, CALCITE_DIRECT_METRIC)

    def test_calculating_unit_cell_volume(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        a, b, c, *_ = CALCITE_LATTICE_PARAMETERS
        expected_volume = sqrt(3) / 2 * a * a * c

        assert_almost_equal(lattice.unit_cell_volume, expected_volume)

    def test_creating_direct_lattice_vectors(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        v1 = DirectLatticeVector([1, 2, 3], lattice)
        v2 = lattice.vector([1, 2, 3])
        assert v1 == v2

    def test_calculating_length_of_direct_lattice_vector(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        v1 = DirectLatticeVector([1, 1, 0], lattice)
        v2 = DirectLatticeVector([1, 2, 3], lattice)

        assert_almost_equal(v1.norm(), 4.99)
        assert_almost_equal(v2.norm(), 51.7330874)

    def test_calculating_inner_product(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        v1 = DirectLatticeVector([1, 0, 0], lattice)
        v2 = DirectLatticeVector([0, 1, 0], lattice)
        v3 = DirectLatticeVector([0, 0, 1], lattice)
        v4 = DirectLatticeVector([1, 4, 2], lattice)

        assert_almost_equal(v1.inner(v2), -12.45005)
        assert_almost_equal(v2.inner(v1), -12.45005)
        assert_almost_equal(v1.inner(v3), 0)
        assert_almost_equal(v1.inner(v4), -24.9001)

    def test_calculating_angle_between_two_vectors(self):
        lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        v1 = DirectLatticeVector([1, 0, 0], lattice)
        v2 = DirectLatticeVector([0, 1, 0], lattice)
        v3 = DirectLatticeVector([0, 0, 1], lattice)
        v4 = DirectLatticeVector([1, 4, 2], lattice)

        assert_almost_equal(v1.angle(v2), 120)
        assert_almost_equal(v2.angle(v1), 120)
        assert_almost_equal(v1.angle(v3), 90)
        assert_almost_equal(v1.angle(v4), 97.4528371)
