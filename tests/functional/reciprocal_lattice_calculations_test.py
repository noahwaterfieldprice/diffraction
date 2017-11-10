from numpy import array, sqrt
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

from diffraction import (DirectLattice, DirectLatticeVector,
                         ReciprocalLatticeVector, ReciprocalLattice)

CALCITE_RECIPROCAL_LATTICE_PARAMETERS = (0.2314, 0.2314, 0.0588, 90, 90, 60)
CALCITE_RECIPROCAL_METRIC_TENSOR = array([[0.053545, 0.026773, 0.],
                                          [0.026773, 0.053545, 0.],
                                          [0., 0.,  0.003457]])
CALCITE_LATTICE_PARAMETERS = (4.99, 4.99, 17.002, 90.0, 90.0, 120.0)


class TestCreatingReciprocalLatticeFromSequence:
    def test_can_create_reciprocal_lattice_from_sequence(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)

        assert lattice.lattice_parameters == CALCITE_RECIPROCAL_LATTICE_PARAMETERS

    def test_error_if_lattice_parameter_missing_from_sequence(self):
        lattice_parameters_missing_one = CALCITE_RECIPROCAL_LATTICE_PARAMETERS[:5]

        with pytest.raises(ValueError) as exception_info:
            ReciprocalLattice(lattice_parameters_missing_one)
        assert str(exception_info.value) == "Missing lattice parameter from input"

    def test_error_if_invalid_lattice_parameter_given(self):
        invalid_lattice_parameters = CALCITE_RECIPROCAL_LATTICE_PARAMETERS[:5] + ("abcdef",)

        with pytest.raises(ValueError) as exception_info:
            ReciprocalLattice(invalid_lattice_parameters)
        assert str(exception_info.value) == "Invalid lattice parameter gamma_star: abcdef"


class TestCreatingReciprocalLatticeFromMapping:
    def test_can_create_reciprocal_lattice_from_dictionary(self):
        lattice_parameters = {"a_star": 0.2314, "b_star": 0.2314, "c_star": 0.0588,
                              "alpha_star": 90, "beta_star": 90, "gamma_star": 60}
        lattice = ReciprocalLattice.from_dict(lattice_parameters)

        assert lattice.lattice_parameters == CALCITE_RECIPROCAL_LATTICE_PARAMETERS

    def test_error_if_lattice_parameter_missing_from_dict(self):
        lattice_parameters = {"a_star": 0.2314, "c_star": 0.0588,
                              "alpha_star": 90, "beta_star": 90, "gamma_star": 60}
        with pytest.raises(ValueError) as exception_info:
            ReciprocalLattice.from_dict(lattice_parameters)
        assert str(exception_info.value) == "Parameter: 'b_star' missing from input dictionary"


class TestCreatingReciprocalLatticeFromCIF:
    def test_can_create_reciprocal_lattice_from_single_datablock_cif(self):
        lattice = ReciprocalLattice.from_cif(
            "tests/functional/static/valid_cifs/calcite_icsd.cif")

        assert_almost_equal(lattice.lattice_parameters,
                            CALCITE_RECIPROCAL_LATTICE_PARAMETERS, decimal=4)

    def test_error_if_lattice_parameter_is_missing_from_cif(selfs):
        with pytest.raises(ValueError) as exception_info:
            ReciprocalLattice.from_cif("tests/functional/static/invalid_cifs/"
                                       "calcite_icsd_missing_lattice_parameter.cif")

        assert str(exception_info.value) == \
            "Parameter: 'cell_length_b' missing from input CIF"

    def test_error_datablock_not_given_for_multi_data_block_cif(self):
        with pytest.raises(TypeError) as exception_info:
            ReciprocalLattice.from_cif(
                "tests/functional/static/valid_cifs/multi_data_block.cif")
        assert str(exception_info.value) == \
            ("__init__() missing keyword argument: 'data_block'. "
             "Required when input CIF has multiple data blocks.")

    def test_can_create_direct_lattice_from_multi_data_block_cif(self):
        CHFeNOS = ReciprocalLattice.from_cif(
            "tests/functional/static/valid_cifs/multi_data_block.cif",
            data_block="data_CSD_CIF_ACAKOF")

        assert_almost_equal(CHFeNOS.lattice_parameters,
                            [0.1662, 0.1122, 0.1014, 101.9615, 94.5792, 98.5165],
                            decimal=4)


class TestCreatingReciprocalLatticeFromDirectLattice:
    def test_can_create_reciprocal_lattice_from_direct_lattice(self):
        direct_lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        reciprocal_lattice = direct_lattice.reciprocal()

        assert isinstance(reciprocal_lattice, ReciprocalLattice)
        assert_almost_equal(reciprocal_lattice.lattice_parameters,
                            CALCITE_RECIPROCAL_LATTICE_PARAMETERS, decimal=4)


class TestReciprocalSpaceCalculations:
    def test_lattice_parameters_available_as_attribute(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)

        assert lattice.lattice_parameters == CALCITE_RECIPROCAL_LATTICE_PARAMETERS

    def test_calculating_reciprocal_metric_tensor(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)

        assert_array_almost_equal(lattice.metric, CALCITE_RECIPROCAL_METRIC_TENSOR)

    def test_calculating_unit_cell_volume(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)
        a_star, b_star, c_star, *_ = CALCITE_RECIPROCAL_LATTICE_PARAMETERS
        expected_volume = sqrt(3) / 2 * a_star * a_star * c_star

        assert_almost_equal(lattice.unit_cell_volume, expected_volume)

    def test_creating_reciprocal_lattice_vectors(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)
        v1 = ReciprocalLatticeVector([1, 2, 3], lattice)
        v2 = lattice.vector([1, 2, 3])
        assert v1 == v2

    def test_calculating_length_of_reciprocal_lattice_vector(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)
        v1 = ReciprocalLatticeVector([1, 1, 0], lattice)
        v2 = ReciprocalLatticeVector([1, 2, 3], lattice)

        assert_almost_equal(v1.norm(), 0.4008, decimal=4)
        assert_almost_equal(v2.norm(), 0.6372, decimal=4)

    def test_calculating_inner_product(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)
        v1 = ReciprocalLatticeVector([1, 0, 0], lattice)
        v2 = ReciprocalLatticeVector([0, 1, 0], lattice)
        v3 = ReciprocalLatticeVector([0, 0, 1], lattice)
        v4 = ReciprocalLatticeVector([1, 4, 2], lattice)

        assert_almost_equal(v1.inner(v2), 0.0268, decimal=4)
        assert_almost_equal(v2.inner(v1), 0.0268, decimal=4)
        assert_almost_equal(v1.inner(v3), 0)
        assert_almost_equal(v1.inner(v4), 0.1606, decimal=4)

    def test_calculating_angle_between_two_vectors(self):
        lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)
        v1 = ReciprocalLatticeVector([1, 0, 0], lattice)
        v2 = ReciprocalLatticeVector([0, 1, 0], lattice)
        v3 = ReciprocalLatticeVector([0, 0, 1], lattice)
        v4 = ReciprocalLatticeVector([1, 4, 2], lattice)

        assert_almost_equal(v1.angle(v2), 60)
        assert_almost_equal(v2.angle(v1), 60)
        assert_almost_equal(v1.angle(v3), 90)
        assert_almost_equal(v1.angle(v4), 49.4084, decimal=4)

    def test_calculating_inner_product_with_direct_lattice_vector(self):
        direct_lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        reciprocal_lattice = direct_lattice.reciprocal()
        v1_direct = DirectLatticeVector([1, 0, 0], direct_lattice)
        v2_direct = DirectLatticeVector([1, 4, 2], direct_lattice)
        v1_reciprocal = ReciprocalLatticeVector([1, 0, 0], reciprocal_lattice)
        v2_reciprocal = ReciprocalLatticeVector([1, 4, 2], reciprocal_lattice)

        assert_almost_equal(v1_reciprocal.inner(v1_direct), 1)
        assert_almost_equal(v2_reciprocal.inner(v2_direct), 21)
        assert_almost_equal(v1_reciprocal.inner(v2_direct), 1)
        assert_almost_equal(v2_reciprocal.inner(v1_direct), 1)

    def test_calculating_angle_with_direct_lattice_vector(self):
        direct_lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        reciprocal_lattice = direct_lattice.reciprocal()
        v1_direct = DirectLatticeVector([1, 0, 0], direct_lattice)
        v2_direct = DirectLatticeVector([0, 4, 2], direct_lattice)
        v1_reciprocal = ReciprocalLatticeVector([1, 0, 0], reciprocal_lattice)
        v2_reciprocal = ReciprocalLatticeVector([0, 4, 2], reciprocal_lattice)

        assert_almost_equal(v1_reciprocal.angle(v1_direct), 30)
        assert_almost_equal(v2_reciprocal.angle(v2_direct), 57.0690, decimal=4)
        assert_almost_equal(v1_reciprocal.angle(v2_direct), 90)
        assert_almost_equal(v2_reciprocal.angle(v1_direct), 90)

    def test_error_if_calculating_inner_product_with_different_lattices(self):
        reciprocal_lattice1 = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)
        reciprocal_lattice2 = ReciprocalLattice([0.1, 0.2, 0.3, 90, 90, 120])
        v1_reciprocal = ReciprocalLatticeVector([1, 0, 0], reciprocal_lattice1)
        v2_reciprocal = ReciprocalLatticeVector([1, 0, 0], reciprocal_lattice2)

        with pytest.raises(TypeError) as exception_info:
            v1_reciprocal.inner(v2_reciprocal)

        assert str(exception_info.value) == "lattice must be the same " \
                                            "for both ReciprocalLatticeVectors"

        direct_lattice2 = reciprocal_lattice2.direct()
        v2_direct = DirectLatticeVector([1, 0, 0], direct_lattice2)

        with pytest.raises(TypeError) as exception_info:
            v1_reciprocal.inner(v2_direct)

        assert str(exception_info.value) == "ReciprocalLatticeVector and DirectLatticeVector" \
                                            " lattices must be reciprocally related."
