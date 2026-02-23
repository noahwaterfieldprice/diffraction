"""Unit tests for diffraction.lattice.

Tests are grouped by concern:
  - Utility functions (_to_radians, _to_degrees, metric_tensor, reciprocalise)
  - DirectLattice creation (sequence, dict, CIF, reciprocal round-trip)
  - Lattice properties with known crystallographic values
  - Lattice validation edge cases
  - DirectLatticeVector creation and magic methods (approved MagicMock(metric=…))
  - ReciprocalLatticeVector creation and magic methods
  - DirectLatticeVector calculations
  - ReciprocalLatticeVector calculations
  - Cross-space vector calculations

Mock usage is restricted to vector tests where MagicMock(metric=…) is the
approved boundary-mock pattern. Domain objects (DirectLattice, ReciprocalLattice)
are always created as real instances.
"""

from math import pi
from pathlib import Path
from typing import Any

import pytest
from numpy import array
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest_mock import MockerFixture

from diffraction import DirectLattice, ReciprocalLattice
from diffraction.lattice import (
    DirectLatticeVector,
    Lattice,
    ReciprocalLatticeVector,
    _to_degrees,
    _to_radians,
    metric_tensor,
    reciprocalise,
)

# Path to the functional test CIF directory, used for real CIF loading tests.
CIF_DIR = Path(__file__).parent.parent / "functional" / "static" / "valid_cifs"

# Lattice parameters used across tests — canonical values match
# CALCITE_LATTICE_PARAMS in tests/conftest.py.
CALCITE_LATTICE_PARAMS = (4.99, 4.99, 17.002, 90.0, 90.0, 120.0)
NACL_PARAMS = (5.6402, 5.6402, 5.6402, 90.0, 90.0, 90.0)
CORUNDUM_PARAMS = (4.758, 4.758, 12.991, 90.0, 90.0, 120.0)
FORSTERITE_PARAMS = (4.758, 10.225, 5.994, 90.0, 90.0, 90.0)

# Precomputed 3x3 metric tensors for use in boundary-mock vector tests.
CALCITE_DIRECT_METRIC = array(
    [[24.9001, -12.45005, 0], [-12.45005, 24.9001, 0], [0, 0, 289.068004]]
)
CALCITE_RECIPROCAL_METRIC = array(
    [[2.1138, 1.0569, 0], [1.0569, 2.1138, 0], [0, 0, 0.1366]]
)

# Legacy dict used by utility function tests.
CALCITE_LATTICE_DICT = {
    "a": 4.99,
    "b": 4.99,
    "c": 17.002,
    "alpha": 90,
    "beta": 90,
    "gamma": 120,
}
CALCITE_RECIPROCAL_PARAMS = (1.4539, 1.4539, 0.3696, 90, 90, 60)


# ---------------------------------------------------------------------------
# Fake concrete subclass used only for ABC testing
# ---------------------------------------------------------------------------


class FakeAbstractLattice(Lattice):
    """Minimal concrete subclass for testing ABC behaviour."""

    lattice_parameter_keys = ("k1", "k2", "k3", "k4", "k5", "k6")

    @classmethod
    def from_cif(
        cls, filepath: str, data_block: str | None = None
    ) -> "FakeAbstractLattice":
        super().from_cif(filepath, data_block)
        return cls([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # unreachable; super() raises


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    def test_converting_lattice_parameters_to_radians(self) -> None:
        lattice_parameters_deg = [1, 2, 3, 90, 120, 45]
        expected = (1, 2, 3, pi / 2, 2 * pi / 3, pi / 4)

        lattice_parameters_rad = _to_radians(lattice_parameters_deg)
        assert_array_almost_equal(lattice_parameters_rad, expected)

    def test_converting_lattice_parameters_to_degrees(self) -> None:
        lattice_parameters_rad = (1.0, 2.0, 3.0, pi / 2, 2 * pi / 3, pi / 4)
        expected = (1, 2, 3, 90, 120, 45)
        lattice_parameters_deg = _to_degrees(lattice_parameters_rad)
        assert_array_almost_equal(lattice_parameters_deg, expected)

    def test_calculating_metric_tensor(self) -> None:
        lattice_parameters = list(CALCITE_LATTICE_DICT.values())
        assert_array_almost_equal(
            metric_tensor(lattice_parameters), CALCITE_DIRECT_METRIC
        )

    def test_transforming_to_reciprocal_basis(self) -> None:
        lattice_parameters = list(CALCITE_LATTICE_DICT.values())

        reciprocal_lattice_parameters = reciprocalise(lattice_parameters)
        assert_array_almost_equal(
            reciprocal_lattice_parameters,
            CALCITE_RECIPROCAL_PARAMS,
            decimal=4,
        )


# ---------------------------------------------------------------------------
# DirectLattice and ReciprocalLattice creation
# ---------------------------------------------------------------------------


class TestDirectLatticeCreation:
    def test_direct_lattice_from_sequence_assigns_all_parameters(self) -> None:
        lattice = DirectLattice(CALCITE_LATTICE_PARAMS)

        assert lattice.a == pytest.approx(4.99)
        assert lattice.b == pytest.approx(4.99)
        assert lattice.c == pytest.approx(17.002)
        assert lattice.alpha == pytest.approx(90.0)
        assert lattice.beta == pytest.approx(90.0)
        assert lattice.gamma == pytest.approx(120.0)
        assert lattice.lattice_parameters == CALCITE_LATTICE_PARAMS

    def test_direct_lattice_from_dict_assigns_all_parameters(self) -> None:
        lattice = DirectLattice.from_dict(CALCITE_LATTICE_DICT)

        assert lattice.lattice_parameters == CALCITE_LATTICE_PARAMS

    def test_direct_lattice_from_cif_loads_calcite_parameters(self) -> None:
        cif_path = str(CIF_DIR / "calcite_icsd.cif")
        lattice = DirectLattice.from_cif(cif_path)

        assert lattice.lattice_parameters == CALCITE_LATTICE_PARAMS

    def test_reciprocal_lattice_from_cif_loads_calcite_parameters(self) -> None:
        cif_path = str(CIF_DIR / "calcite_icsd.cif")
        lattice = ReciprocalLattice.from_cif(cif_path)

        # Reciprocal lattice parameters should be physically reasonable
        assert isinstance(lattice, ReciprocalLattice)
        assert lattice.a_star == pytest.approx(1.4539, rel=1e-3)
        assert lattice.gamma_star == pytest.approx(60.0, rel=1e-4)

    def test_direct_lattice_reciprocal_returns_reciprocal_lattice(self) -> None:
        lattice = DirectLattice(CALCITE_LATTICE_PARAMS)
        rl = lattice.reciprocal()

        assert isinstance(rl, ReciprocalLattice)
        assert rl.a_star == pytest.approx(1.4539, rel=1e-3)
        assert rl.gamma_star == pytest.approx(60.0, rel=1e-4)

    def test_reciprocal_lattice_direct_returns_direct_lattice(self) -> None:
        rl = ReciprocalLattice(CALCITE_RECIPROCAL_PARAMS)
        direct = rl.direct()

        assert isinstance(direct, DirectLattice)
        assert_almost_equal(direct.lattice_parameters, CALCITE_LATTICE_PARAMS, decimal=2)

    def test_abstract_lattice_from_cif_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            FakeAbstractLattice.from_cif("some/file/path.cif")


# ---------------------------------------------------------------------------
# Lattice properties with known values
# ---------------------------------------------------------------------------


class TestLatticePropertiesWithKnownValues:
    def test_calcite_metric_tensor_matches_known_value(self) -> None:
        lattice = DirectLattice(CALCITE_LATTICE_PARAMS)

        assert_array_almost_equal(lattice.metric, CALCITE_DIRECT_METRIC)

    def test_calcite_unit_cell_volume_matches_known_value(self) -> None:
        lattice = DirectLattice(CALCITE_LATTICE_PARAMS)

        assert_almost_equal(lattice.unit_cell_volume, 366.6332, decimal=4)

    @pytest.mark.parametrize(
        "params, expected_volume",
        [
            (CALCITE_LATTICE_PARAMS, 366.6332),   # trigonal, hexagonal setting
            (NACL_PARAMS, 179.4252),       # cubic
            (CORUNDUM_PARAMS, 254.6960),   # trigonal, hexagonal setting
            (FORSTERITE_PARAMS, 291.6114), # orthorhombic
        ],
    )
    def test_unit_cell_volume_for_multiple_crystal_systems(
        self, params: tuple[float, ...], expected_volume: float
    ) -> None:
        lattice = DirectLattice(params)

        assert_almost_equal(lattice.unit_cell_volume, expected_volume, decimal=4)

    def test_lattice_parameters_tuple_updates_when_attribute_set(self) -> None:
        lattice = DirectLattice(CALCITE_LATTICE_PARAMS)
        lattice.a = 5.5

        assert lattice.lattice_parameters[0] == pytest.approx(5.5)
        assert lattice.lattice_parameters[1:] == CALCITE_LATTICE_PARAMS[1:]


# ---------------------------------------------------------------------------
# Lattice validation edge cases
# ---------------------------------------------------------------------------


class TestLatticeValidationEdgeCases:
    def test_from_dict_raises_for_missing_parameter(self) -> None:
        incomplete = {"a": 4.99, "b": 4.99, "c": 17.002, "alpha": 90, "beta": 90}

        with pytest.raises(ValueError) as exc_info:
            DirectLattice.from_dict(incomplete)
        assert "gamma" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    @pytest.mark.parametrize("position", range(6))
    def test_check_lattice_parameters_rejects_non_numeric(
        self, position: int, invalid_value: str
    ) -> None:
        params = list(CALCITE_LATTICE_PARAMS)
        params[position] = invalid_value  # type: ignore[call-overload]
        expected_key = DirectLattice.lattice_parameter_keys[position]

        with pytest.raises(ValueError, match=expected_key):
            DirectLattice(params)

    def test_repr_shows_lattice_type_and_parameters(self) -> None:
        lattice = DirectLattice(CALCITE_LATTICE_PARAMS)

        result = repr(lattice)
        assert result.startswith("DirectLattice(")
        assert "4.99" in result
        assert "17.002" in result
        assert "120.0" in result


# ---------------------------------------------------------------------------
# DirectLatticeVector creation and magic methods
# (Uses approved MagicMock(metric=...) boundary-mock pattern)
# ---------------------------------------------------------------------------


class TestDirectLatticeVectorCreationAndMagicMethods:
    lattice_cls: Any = DirectLattice
    cls: Any = DirectLatticeVector

    def test_creating_lattice_vector_directly(self, mocker: MockerFixture) -> None:
        lattice: Any = mocker.MagicMock()

        vector = self.cls([1, 0, 0], lattice)
        assert vector.lattice == lattice

    def test_creating_lattice_vector_from_lattice(self, mocker: MockerFixture) -> None:
        lattice: Any = mocker.MagicMock()

        v1 = self.cls([1, 2, 3], lattice)
        v2 = self.lattice_cls.vector(lattice, [1, 2, 3])
        assert v1 == v2

    def test_scalar_multiplication_preserves_type_and_lattice(
        self, mocker: MockerFixture
    ) -> None:
        lattice: Any = mocker.MagicMock()

        v1 = self.cls([1, 0, 0], lattice)
        v2 = 2.0 * v1
        assert isinstance(v2, self.cls)
        assert v2.lattice == lattice

    def test_scalar_division_preserves_type_and_lattice(
        self, mocker: MockerFixture
    ) -> None:
        lattice: Any = mocker.MagicMock()

        v1 = self.cls([2, 4, 6], lattice)
        v2 = v1 / 2.0
        assert isinstance(v2, self.cls)
        assert v2.lattice == lattice

    def test_negation_preserves_type_and_lattice(
        self, mocker: MockerFixture
    ) -> None:
        lattice: Any = mocker.MagicMock()

        v1 = self.cls([1, 0, 0], lattice)
        v2 = -v1
        assert isinstance(v2, self.cls)
        assert v2.lattice == lattice

    def test_direct_lattice_vector_equivalence(self, mocker: MockerFixture) -> None:
        lattice_1: Any = mocker.MagicMock()
        lattice_2: Any = mocker.MagicMock()
        v1 = self.cls([1, 0, 0], lattice_1)
        v2 = self.cls([1, 0, 0], lattice_1)
        v3 = self.cls([1, 0, 0], lattice_2)
        v4 = self.cls([0, 1, 0], lattice_1)

        assert v1 == v2
        assert v1 != v3
        assert v1 != v4

    def test_adding_and_subtracting_direct_lattice_vectors(
        self, mocker: MockerFixture
    ) -> None:
        lattice: Any = mocker.MagicMock()
        v1 = self.cls([1, 0, 0], lattice)
        v2 = self.cls([0, 2, 3], lattice)
        v3 = self.cls([1, 2, 3], lattice)

        assert v1 + v2 == v3
        assert v3 - v2 == v1

    def test_error_if_adding_or_subtracting_with_different_lattices(
        self, mocker: MockerFixture
    ) -> None:
        lattice_1: Any = mocker.MagicMock()
        lattice_2: Any = mocker.MagicMock()
        v1 = self.cls([1, 0, 0], lattice_1)
        v2 = self.cls([0, 2, 3], lattice_2)

        with pytest.raises(TypeError) as exception_info:
            v1 + v2
        assert str(exception_info.value) == (
            f"lattice must be the same for both {self.cls.__name__:s}s"
        )
        with pytest.raises(TypeError) as exception_info:
            v1 - v2
        assert str(exception_info.value) == (
            f"lattice must be the same for both {self.cls.__name__:s}s"
        )

    def test_string_representation_of_lattice_vectors(
        self, mocker: MockerFixture
    ) -> None:
        """Verify repr and str include class name and lattice."""
        lattice: Any = mocker.MagicMock()
        v1 = self.cls([1, 2, 3], lattice)

        # repr should contain the class name
        assert self.cls.__name__ in repr(v1)
        # str should contain the class name
        assert self.cls.__name__ in str(v1)


class TestReciprocalLatticeVectorCreationAndMagicMethods(
    TestDirectLatticeVectorCreationAndMagicMethods
):
    lattice_cls = ReciprocalLattice
    cls = ReciprocalLatticeVector


# ---------------------------------------------------------------------------
# DirectLatticeVector calculations
# ---------------------------------------------------------------------------


class TestDirectLatticeVectorCalculations:
    def test_calculating_norm_of_direct_lattice_vector(
        self, mocker: MockerFixture
    ) -> None:
        lattice: Any = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        v1 = DirectLatticeVector([1, 1, 0], lattice)
        v2 = DirectLatticeVector([1, 2, 3], lattice)

        assert_almost_equal(v1.norm(), 4.99)
        assert_almost_equal(v2.norm(), 51.7330874)

    def test_error_if_calculating_inner_product_or_angle_with_different_lattices(
        self, mocker: MockerFixture
    ) -> None:
        lattice_1: Any = mocker.MagicMock()
        lattice_2: Any = mocker.MagicMock()
        v1 = ReciprocalLatticeVector([1, 0, 0], lattice_1)
        v2 = ReciprocalLatticeVector([0, 2, 3], lattice_2)

        with pytest.raises(TypeError) as exception_info:
            v1.inner(v2)
        assert (
            str(exception_info.value) == "lattice must be the same "
            "for both ReciprocalLatticeVectors"
        )
        with pytest.raises(TypeError) as exception_info:
            v1.angle(v2)
        assert (
            str(exception_info.value) == "lattice must be the same "
            "for both ReciprocalLatticeVectors"
        )

    @pytest.mark.parametrize(
        "uvw,result",
        [
            ([0, 1, 0], 12.45005),
            ([0, 0, 1], 289.068004),
            ([1, -1, 0], 0),
            ([1, 2, 3], 904.554162),
        ],
    )
    def test_calculating_inner_product_of_vectors(
        self, mocker: MockerFixture, uvw: list[int], result: float
    ) -> None:
        lattice: Any = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        v1 = DirectLatticeVector([1, 1, 1], lattice)
        v2 = DirectLatticeVector(uvw, lattice)

        assert_almost_equal(v1.inner(v2), result)

    @pytest.mark.parametrize(
        "uvw,result",
        [
            ([0, 1, 0], 81.90538705),
            ([0, 0, 1], 16.3566939),
            ([1, -1, 0], 90),
            ([1, 2, 3], 9.324336578),
        ],
    )
    def test_calculating_angle_between_two_vectors(
        self, mocker: MockerFixture, uvw: list[int], result: float
    ) -> None:
        lattice: Any = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        v1 = DirectLatticeVector([1, 1, 1], lattice)
        v2 = DirectLatticeVector(uvw, lattice)

        assert_almost_equal(v1.angle(v2), result)


# ---------------------------------------------------------------------------
# ReciprocalLatticeVector calculations
# ---------------------------------------------------------------------------


class TestReciprocalLatticeVectorCalculations:
    def test_calculating_norm_of_reciprocal_lattice_vector(
        self, mocker: MockerFixture
    ) -> None:
        lattice: Any = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        v1 = ReciprocalLatticeVector([1, 1, 0], lattice)
        v2 = ReciprocalLatticeVector([1, 2, 3], lattice)

        assert_almost_equal(v1.norm(), 2.5182, decimal=4)
        assert_almost_equal(v2.norm(), 4.0032, decimal=4)

    def test_error_if_calculating_inner_product_or_angle_with_different_lattices(
        self, mocker: MockerFixture
    ) -> None:
        lattice_1: Any = mocker.MagicMock()
        lattice_2: Any = mocker.MagicMock()
        v1 = ReciprocalLatticeVector([1, 0, 0], lattice_1)
        v2 = ReciprocalLatticeVector([0, 2, 3], lattice_2)

        with pytest.raises(TypeError) as exception_info:
            v1.inner(v2)
        assert (
            str(exception_info.value) == "lattice must be the same "
            "for both ReciprocalLatticeVectors"
        )
        with pytest.raises(TypeError) as exception_info:
            v1.angle(v2)
        assert (
            str(exception_info.value) == "lattice must be the same "
            "for both ReciprocalLatticeVectors"
        )

    @pytest.mark.parametrize(
        "hkl,result",
        [
            ([0, 1, 0], 3.1707),
            ([0, 0, 1], 0.1366),
            ([1, -1, 0], 0),
            ([1, 2, 3], 9.9219),
        ],
    )
    def test_calculating_inner_product_of_vectors(
        self, mocker: MockerFixture, hkl: list[int], result: float
    ) -> None:
        lattice: Any = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        v1 = ReciprocalLatticeVector([1, 1, 1], lattice)
        v2 = ReciprocalLatticeVector(hkl, lattice)

        assert_almost_equal(v1.inner(v2), result, decimal=4)

    @pytest.mark.parametrize(
        "hkl,result",
        [
            ([0, 1, 0], 31.0357),
            ([0, 0, 1], 81.6504),
            ([1, -1, 0], 90),
            ([1, 2, 3], 13.1489),
        ],
    )
    def test_calculating_angle_between_two_vectors(
        self, mocker: MockerFixture, hkl: list[int], result: float
    ) -> None:
        lattice: Any = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        v1 = ReciprocalLatticeVector([1, 1, 1], lattice)
        v2 = ReciprocalLatticeVector(hkl, lattice)

        assert_almost_equal(v1.angle(v2), result, decimal=4)


# ---------------------------------------------------------------------------
# Cross-space vector calculations
# ---------------------------------------------------------------------------


class TestDirectAndReciprocalLatticeVectorCalculations:
    def test_error_if_calculating_inner_product_or_angle_with_unreciprocal_lattices(
        self, mocker: MockerFixture
    ) -> None:
        direct_lattice: Any = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        reciprocal_lattice: Any = mocker.MagicMock(
            metric=CALCITE_RECIPROCAL_METRIC * 1.02
        )
        direct_vector = DirectLatticeVector([1, 0, 0], direct_lattice)
        reciprocal_vector = ReciprocalLatticeVector([0, 2, 3], reciprocal_lattice)

        with pytest.raises(TypeError) as exception_info:
            direct_vector.inner(reciprocal_vector)
        assert (
            str(exception_info.value)
            == "DirectLatticeVector and ReciprocalLatticeVector"
            " lattices must be reciprocally related."
        )
        with pytest.raises(TypeError) as exception_info:
            direct_vector.angle(reciprocal_vector)
        assert (
            str(exception_info.value)
            == "DirectLatticeVector and ReciprocalLatticeVector"
            " lattices must be reciprocally related."
        )

    @pytest.mark.parametrize(
        "uvw,hkl,result",
        [
            ([1, 0, 0], [0, 0, 1], 0),
            ([1, 0, 0], [1, 0, 0], 2 * pi),
            ([1, -1, 0], [1, 2, 3], -2 * pi),
            ([1, 2, 3], [0, 0, 1], 6 * pi),
        ],
    )
    def test_calculating_inner_product_of_direct_and_reciprocal_lattice_vectors(
        self,
        mocker: MockerFixture,
        uvw: list[int],
        hkl: list[int],
        result: float,
    ) -> None:
        direct_lattice: Any = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        reciprocal_lattice: Any = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        direct_vector = DirectLatticeVector(uvw, direct_lattice)
        reciprocal_vector = ReciprocalLatticeVector(hkl, reciprocal_lattice)

        assert_almost_equal(direct_vector.inner(reciprocal_vector), result)
        assert_almost_equal(reciprocal_vector.inner(direct_vector), result)

    @pytest.mark.parametrize(
        "uvw,hkl,result",
        [
            ([1, 0, 0], [0, 0, 1], 90),
            ([1, 0, 0], [1, 0, 0], 30),
            ([1, -1, 0], [0, 0, 1], 90),
            ([1, 2, 3], [0, 0, 1], 9.6527),
        ],
    )
    def test_calculating_angle_between_direct_and_reciprocal_lattice_vectors(
        self,
        mocker: MockerFixture,
        uvw: list[int],
        hkl: list[int],
        result: float,
    ) -> None:
        direct_lattice: Any = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        reciprocal_lattice: Any = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        direct_vector = DirectLatticeVector(uvw, direct_lattice)
        reciprocal_vector = ReciprocalLatticeVector(hkl, reciprocal_lattice)

        assert_almost_equal(direct_vector.angle(reciprocal_vector), result, decimal=2)
        assert_almost_equal(reciprocal_vector.angle(direct_vector), result, decimal=2)
