import pytest

from diffraction import ReciprocalLattice

CALCITE_RECIPROCAL_LATTICE_PARAMETERS = (0.2314, 0.2314, 0.0588, 90, 90, 60)


class TestCreatingReciprocalLatticeFromSequence:
    def test_can_create_from_sequence(self):
        r_lattice = ReciprocalLattice(CALCITE_RECIPROCAL_LATTICE_PARAMETERS)

        assert r_lattice.lattice_parameters == CALCITE_RECIPROCAL_LATTICE_PARAMETERS

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
    def test_can_create_crystal_from_dictionary(self):
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
