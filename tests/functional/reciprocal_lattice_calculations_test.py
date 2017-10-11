from numpy import array, sqrt
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

from diffraction import DirectLattice, ReciprocalLatticeVector, ReciprocalLattice

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


class TestCreatingReciprocalLatticeFromDirectLattice:
    def test_can_create_reciprocal_lattice_from_direct_lattice(self):
        direct_lattice = DirectLattice(CALCITE_LATTICE_PARAMETERS)
        reciprocal_lattice = direct_lattice.reciprocal()

        assert isinstance(reciprocal_lattice, ReciprocalLattice)
        assert_almost_equal(reciprocal_lattice.lattice_parameters,
                            CALCITE_RECIPROCAL_LATTICE_PARAMETERS, decimal=4)


class TestReciprocalLatticeCalculations:
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
        lattice = ReciprocalLattice(CALCITE_LATTICE_PARAMETERS)
        v1 = ReciprocalLatticeVector([1, 2, 3], lattice)
        v2 = lattice.vector([1, 2, 3])
        assert v1 == v2
