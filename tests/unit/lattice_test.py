from collections import OrderedDict

from numpy import add, array, array_equal, ndarray, pi, sqrt
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest

from diffraction.cif.helpers import NUMERICAL_DATA_VALUE
from diffraction.lattice import (AbstractLattice, DirectLattice,
                                 DirectLatticeVector,
                                 _to_radians, _to_degrees, metric_tensor,
                                 ReciprocalLattice, ReciprocalLatticeVector,
                                 reciprocalise)

CALCITE_CIF = OrderedDict([("cell_length_a", "4.9900(2)"),
                           ("cell_length_b", "4.9900(2)"),
                           ("cell_length_c", "17.002(1)"),
                           ("cell_angle_alpha", "90."),
                           ("cell_angle_beta", "90."),
                           ("cell_angle_gamma", "90.")])

CALCITE_LATTICE = OrderedDict(
    [("a", 4.99), ("b", 4.99), ("c", 17.002),
     ("alpha", 90), ("beta", 90), ("gamma", 120)])

CALCITE_DIRECT_METRIC = array([[24.9001, -12.45005, 0],
                               [-12.45005, 24.9001, 0],
                               [0, 0, 289.068004]])

CALCITE_DIRECT_CELL_VOLUME = 366.6331539

CALCITE_RECIPROCAL_LATTICE = OrderedDict(
    [("a_star", 1.4539), ("b_star", 1.4539), ("c_star", 0.3696),
     ("alpha_star", 90), ("beta_star", 90), ("gamma_star", 60)])

CALCITE_RECIPROCAL_METRIC = array([[2.1138, 1.0569, 0],
                                   [1.0569, 2.1138, 0],
                                   [0, 0, 0.1366]])

CALCITE_RECIPROCAL_CELL_VOLUME = 0.6766


class FakeAbstractLattice(AbstractLattice):
    """Fake concrete AbstractLattice class for testing"""
    lattice_parameter_keys = ("k1", "k2", "k3", "k4", "k5", "k6")


class TestUtilityFunctions:
    def test_converting_lattice_parameters_to_radians(self):
        lattice_parameters_deg = [1, 2, 3, 90, 120, 45]
        expected = (1, 2, 3, pi / 2, 2 * pi / 3, pi / 4)

        lattice_parameters_rad = _to_radians(lattice_parameters_deg)
        assert_array_almost_equal(lattice_parameters_rad, expected)

    def test_converting_lattice_parameters_to_degrees(self):
        lattice_parameters_rad = [1, 2, 3, pi / 2, 2 * pi / 3, pi / 4]
        expected = (1, 2, 3, 90, 120, 45)
        lattice_parameters_deg = _to_degrees(lattice_parameters_rad)
        assert_array_almost_equal(lattice_parameters_deg, expected)

    def test_calculating_metric_tensor(self):
        lattice_parameters = CALCITE_LATTICE.values()
        assert_array_almost_equal(metric_tensor(lattice_parameters),
                                  CALCITE_DIRECT_METRIC)

    def test_transforming_to_reciprocal_basis(self):
        lattice_parameters = CALCITE_LATTICE.values()

        reciprocal_lattice_parameters = reciprocalise(lattice_parameters)
        assert_array_almost_equal(reciprocal_lattice_parameters,
                                  tuple(CALCITE_RECIPROCAL_LATTICE.values()),
                                  decimal=4)


class TestCreatingAbstractLattice:
    cls = FakeAbstractLattice
    test_dict = OrderedDict([("k1", 2), ("k2", 5), ("k3", 10),
                             ("k4", 90), ("k5", 90), ("k6", 120)])

    def test_error_if_lattice_parameter_missing_from_input_list(self, mocker):
        lattice_parameters_missing_one = list(self.test_dict.values())[:5]
        mock = mocker.MagicMock()
        mock.lattice_parameter_keys = self.test_dict.keys()

        with pytest.raises(ValueError) as exception_info:
            self.cls.convert_parameters(mock, lattice_parameters_missing_one)
        assert str(
            exception_info.value) == "Missing lattice parameter from input"

    def test_error_if_parameter_missing_from_input_dict(self):
        for missing_parameter in self.test_dict.keys():
            dict_with_missing_parameter = self.test_dict.copy()
            del dict_with_missing_parameter[missing_parameter]

            with pytest.raises(ValueError) as exception_info:
                self.cls.from_dict(dict_with_missing_parameter)
            assert str(exception_info.value) == \
                "Parameter: '{}' missing from input dictionary".format(
                    missing_parameter)

    def test_parameters_are_assigned_with_values_read_from_dict(self, mocker):
        mock = mocker.patch("diffraction.lattice.AbstractLattice.__init__",
                            return_value=None)
        self.cls.from_dict(self.test_dict)
        mock.assert_called_once_with(list(self.test_dict.values()))

    @pytest.mark.parametrize("invalid_value", ["abc", "123@%£", "1232.433.21"])
    @pytest.mark.parametrize("position", range(6))
    def test_error_if_invalid_lattice_parameter_given(self, invalid_value,
                                                      position, mocker):
        invalid_lattice_parameters = list(self.test_dict.values())
        invalid_lattice_parameters[position] = invalid_value
        mock = mocker.MagicMock()
        mock.lattice_parameter_keys = tuple(self.test_dict.keys())

        with pytest.raises(ValueError) as exception_info:
            self.cls.convert_parameters(mock, invalid_lattice_parameters)
        assert str(exception_info.value) == \
            "Invalid lattice parameter {}: {}".format(
                mock.lattice_parameter_keys[position], invalid_value)

    def test_parameters_are_assigned_with_correct_type(self, mocker):
        lattice_parameters = self.test_dict.values()
        lattice = self.cls(lattice_parameters)
        mocker.patch("diffraction.lattice.AbstractLattice.convert_parameters",
                     return_value=self.test_dict.values())

        # test lattice parameters are assigned as floats
        for parameter, value in self.test_dict.items():
            assert getattr(lattice, parameter) == value
            assert isinstance(getattr(lattice, parameter), float)

    def test_string_representation_of_lattice(self):
        lattice_parameters = self.test_dict.values()
        lattice = self.cls(lattice_parameters)

        assert repr(lattice) == "{0}({1})".format(
            lattice.__class__.__name__,
            [float(parameter) for parameter in lattice_parameters])
        assert str(lattice) == "{0}({1})".format(
            lattice.__class__.__name__,
            [float(parameter) for parameter in lattice_parameters])

    def test_loading_from_cif(self):
        with pytest.raises(NotImplementedError) as exception:
            self.cls.from_cif("some/file/path.cif")


class TestCreatingDirectLattice(TestCreatingAbstractLattice):
    cls = DirectLattice
    test_dict = CALCITE_LATTICE

    def test_loading_from_cif(self, mocker):
        load_data_block_mock = mocker.patch(
            "diffraction.lattice.load_data_block",
            return_value="data_items")
        get_cif_data_mock = mocker.patch("diffraction.lattice.get_cif_data",
                                         return_value=list(
                                             self.test_dict.values()))
        mock = mocker.patch("diffraction.lattice.AbstractLattice.__init__",
                            return_value=None)

        self.cls.from_cif("some/single/data/block/cif")
        load_data_block_mock.assert_called_with("some/single/data/block/cif",
                                                None)
        get_cif_data_mock.assert_called_with("data_items", *CALCITE_CIF.keys())
        assert_almost_equal(mock.call_args[0][0],
                            tuple(self.test_dict.values()))

    def test_creating_from_reciprocal_lattice(self, mocker):
        mock = mocker.MagicMock()
        mock.lattice_parameters = "reciprocal_lattice_parameters"
        m1 = mocker.patch("diffraction.lattice.reciprocalise",
                          return_value="direct_lattice_parameters")
        m2 = mocker.patch("diffraction.lattice.DirectLattice")

        ReciprocalLattice.direct(mock)
        m1.assert_called_once_with("reciprocal_lattice_parameters")
        m2.assert_called_once_with("direct_lattice_parameters")


class TestCreatingReciprocalLattice(TestCreatingAbstractLattice):
    cls = ReciprocalLattice
    test_dict = CALCITE_RECIPROCAL_LATTICE

    def test_loading_from_cif(self, mocker):
        load_data_block_mock = mocker.patch(
            "diffraction.lattice.load_data_block",
            return_value="data_items")
        get_cif_data_mock = mocker.patch("diffraction.lattice.get_cif_data",
                                         return_value=list(
                                             CALCITE_LATTICE.values()))
        mock = mocker.patch("diffraction.lattice.AbstractLattice.__init__",
                            return_value=None)

        self.cls.from_cif("some/single/data/block/cif")
        load_data_block_mock.assert_called_with("some/single/data/block/cif",
                                                None)
        get_cif_data_mock.assert_called_with("data_items", *CALCITE_CIF.keys())
        assert_almost_equal(mock.call_args[0][0],
                            tuple(self.test_dict.values()),
                            decimal=4)

    def test_creating_from_direct_lattice(self, mocker):
        mock = mocker.MagicMock()
        mock.lattice_parameters = "direct_lattice_parameters"
        m1 = mocker.patch("diffraction.lattice.reciprocalise",
                          return_value="reciprocal_lattice_parameters")
        m2 = mocker.patch("diffraction.lattice.ReciprocalLattice")

        DirectLattice.reciprocal(mock)
        m1.assert_called_once_with("direct_lattice_parameters")
        m2.assert_called_once_with("reciprocal_lattice_parameters")


class TestAccessingComputedProperties:
    # tests both DirectLattice and ReciprocalLattice objects

    @pytest.mark.parametrize("lattice, lattice_class", [
        (CALCITE_LATTICE, DirectLattice),
        (CALCITE_RECIPROCAL_LATTICE, ReciprocalLattice)])
    def test_can_get_lattice_parameters_as_a_tuple(self, mocker, lattice,
                                                   lattice_class):
        mock = mocker.MagicMock(**lattice)
        mock.lattice_parameter_keys = lattice_class.lattice_parameter_keys
        mock.lattice_parameters = lattice_class.lattice_parameters

        assert mock.lattice_parameters.fget(mock) == tuple(lattice.values())

    @pytest.mark.parametrize("lattice, lattice_class, parameter", [
        (CALCITE_LATTICE, DirectLattice, 'a'),
        (CALCITE_RECIPROCAL_LATTICE, ReciprocalLattice, 'a_star')])
    def test_lattice_parameters_updated_if_lattice_parameter_changed(
            self, mocker, lattice, lattice_class, parameter):
        mock = mocker.MagicMock(**lattice)
        mock.lattice_parameter_keys = lattice_class.lattice_parameter_keys
        mock.lattice_parameters = lattice_class.lattice_parameters
        expected_lattice_parameters = (10,) + tuple(lattice.values())[1:]

        setattr(mock, parameter, 10)
        assert mock.lattice_parameters.fget(
            mock) == expected_lattice_parameters

    @pytest.mark.parametrize("lattice, lattice_class", [
        (CALCITE_LATTICE, DirectLattice),
        (CALCITE_RECIPROCAL_LATTICE, ReciprocalLattice)])
    def test_lattice_metric_is_calculated_with_correct_input(self, mocker,
                                                             lattice,
                                                             lattice_class):
        lattice_parameters = tuple(lattice.values())
        mock = mocker.MagicMock(lattice_parameters=lattice_parameters)
        m = mocker.patch("diffraction.lattice.metric_tensor")
        mock.metric = lattice_class.metric

        mock.metric.fget(mock)
        m.assert_called_once_with(lattice_parameters)

    @pytest.mark.parametrize("lattice, lattice_class, metric, cell_volume", [
        (CALCITE_LATTICE, DirectLattice,
         CALCITE_DIRECT_METRIC, CALCITE_DIRECT_CELL_VOLUME),
        (CALCITE_RECIPROCAL_LATTICE, ReciprocalLattice,
         CALCITE_RECIPROCAL_METRIC, CALCITE_RECIPROCAL_CELL_VOLUME)])
    def test_unit_cell_volume_is_calculated_correctly(self, mocker, lattice,
                                                      lattice_class,
                                                      metric, cell_volume):
        mock = mocker.MagicMock(**lattice)
        mock.unit_cell_volume = lattice_class.unit_cell_volume
        mock.metric = metric

        assert_almost_equal(mock.unit_cell_volume.fget(mock),
                            cell_volume, decimal=4)


class TestDirectLatticeVectorCreationAndMagicMethods:
    lattice_cls = DirectLattice
    cls = DirectLatticeVector
    cls_name = 'DirectLatticeVector'

    def test_creating_lattice_vector_directly(self, mocker):
        lattice = mocker.MagicMock()

        vector = self.cls([1, 0, 0], lattice)
        assert issubclass(self.cls, ndarray)
        assert vector.lattice == lattice

    def test_creating_lattice_vector_from_lattice(self, mocker):
        lattice = mocker.MagicMock()

        v1 = self.cls([1, 2, 3], lattice)
        v2 = self.lattice_cls.vector(lattice, [1, 2, 3])
        assert v1 == v2

    def test_lattice_attribute_persists_when_new_array_created(self, mocker):
        lattice = mocker.MagicMock()

        v1 = self.cls([1, 0, 0], lattice)
        v2 = 2 * v1
        v3 = v1.copy()
        assert v2.lattice == lattice
        assert v3.lattice == lattice

    def test_direct_lattice_vector_equivalence(self, mocker):
        lattice_1 = mocker.MagicMock()
        lattice_2 = mocker.MagicMock()
        v1 = self.cls([1, 0, 0], lattice_1)
        v2 = self.cls([1, 0, 0], lattice_1)
        v3 = self.cls([1, 0, 0], lattice_2)
        v4 = self.cls([0, 1, 0], lattice_1)

        assert v1 == v2
        assert v1 != v3
        assert v1 != v4

    def test_adding_and_subtracting_direct_lattice_vectors(self, mocker):
        lattice = mocker.MagicMock()
        v1 = self.cls([1, 0, 0], lattice)
        v2 = self.cls([0, 2, 3], lattice)
        v3 = self.cls([1, 2, 3], lattice)

        assert v1 + v2 == v3
        assert v3 - v2 == v1

    def test_error_if_adding_or_subtracting_with_different_lattices(self,
                                                                    mocker):
        lattice_1 = mocker.MagicMock()
        lattice_2 = mocker.MagicMock()
        v1 = self.cls([1, 0, 0], lattice_1)
        v2 = self.cls([0, 2, 3], lattice_2)

        with pytest.raises(TypeError) as exception_info:
            v1 + v2
        assert str(exception_info.value) == "lattice must be the same " \
                                            "for both {:s}s".format(
            self.cls_name)
        with pytest.raises(TypeError) as exception_info:
            v1 - v2
        assert str(exception_info.value) == "lattice must be the same " \
                                            "for both {:s}s".format(
            self.cls_name)


class TestReciprocalLatticeVectorCreationAndMagicMethods(
        TestDirectLatticeVectorCreationAndMagicMethods):
    lattice_cls = ReciprocalLattice
    cls = ReciprocalLatticeVector
    cls_name = "ReciprocalLatticeVector"


class TestDirectLatticeVectorCalculations:
    def test_calculating_norm_of_direct_lattice_vector(self, mocker):
        lattice = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        v1 = DirectLatticeVector([1, 1, 0], lattice)
        v2 = DirectLatticeVector([1, 2, 3], lattice)

        assert_almost_equal(v1.norm(), 4.99)
        assert_almost_equal(v2.norm(), 51.7330874)

    def test_error_if_calculating_inner_product_or_angle_with_different_lattices(self, mocker):
        lattice_1 = mocker.MagicMock()
        lattice_2 = mocker.MagicMock()
        v1 = ReciprocalLatticeVector([1, 0, 0], lattice_1)
        v2 = ReciprocalLatticeVector([0, 2, 3], lattice_2)

        with pytest.raises(TypeError) as exception_info:
            v1.inner(v2)
        assert str(exception_info.value) == "lattice must be the same " \
                                            "for both ReciprocalLatticeVectors"
        with pytest.raises(TypeError) as exception_info:
            v1.angle(v2)
        assert str(exception_info.value) == "lattice must be the same " \
                                            "for both ReciprocalLatticeVectors"

    @pytest.mark.parametrize("uvw,result", [
        ([0, 1, 0], 12.45005),
        ([0, 0, 1], 289.068004),
        ([1, -1, 0], 0,),
        ([1, 2, 3], 904.554162)])
    def test_calculating_inner_product_of_vectors(self, mocker, uvw, result):
        lattice = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        v1 = DirectLatticeVector([1, 1, 1], lattice)
        v2 = DirectLatticeVector(uvw, lattice)

        assert_almost_equal(v1.inner(v2), result)

    @pytest.mark.parametrize("uvw,result", [
        ([0, 1, 0], 81.90538705),
        ([0, 0, 1], 16.3566939),
        ([1, -1, 0], 90),
        ([1, 2, 3], 9.324336578)])
    def test_calculating_angle_between_two_vectors(self, mocker, uvw, result):
        lattice = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        v1 = DirectLatticeVector([1, 1, 1], lattice)
        v2 = DirectLatticeVector(uvw, lattice)

        assert_almost_equal(v1.angle(v2), result)


class TestReciprocalLatticeVectorCalculations:
    def test_calculating_norm_of_reciprocal_lattice_vector(self, mocker):
        lattice = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        v1 = ReciprocalLatticeVector([1, 1, 0], lattice)
        v2 = ReciprocalLatticeVector([1, 2, 3], lattice)

        assert_almost_equal(v1.norm(), 2.5182, decimal=4)
        assert_almost_equal(v2.norm(), 4.0032, decimal=4)

    def test_error_if_calculating_inner_product_or_angle_with_different_lattices(self, mocker):
        lattice_1 = mocker.MagicMock()
        lattice_2 = mocker.MagicMock()
        v1 = ReciprocalLatticeVector([1, 0, 0], lattice_1)
        v2 = ReciprocalLatticeVector([0, 2, 3], lattice_2)

        with pytest.raises(TypeError) as exception_info:
            v1.inner(v2)
        assert str(exception_info.value) == "lattice must be the same " \
                                            "for both ReciprocalLatticeVectors"
        with pytest.raises(TypeError) as exception_info:
            v1.angle(v2)
        assert str(exception_info.value) == "lattice must be the same " \
                                            "for both ReciprocalLatticeVectors"

    @pytest.mark.parametrize("hkl,result", [
        ([0, 1, 0], 3.1707),
        ([0, 0, 1], 0.1366),
        ([1, -1, 0], 0,),
        ([1, 2, 3], 9.9219)])
    def test_calculating_inner_product_of_vectors(self, mocker, hkl, result):
        lattice = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        v1 = ReciprocalLatticeVector([1, 1, 1], lattice)
        v2 = ReciprocalLatticeVector(hkl, lattice)

        assert_almost_equal(v1.inner(v2), result, decimal=4)

    @pytest.mark.parametrize("hkl,result", [
        ([0, 1, 0], 31.0357),
        ([0, 0, 1], 81.6504),
        ([1, -1, 0], 90),
        ([1, 2, 3], 13.1489)])
    def test_calculating_angle_between_two_vectors(self, mocker, hkl, result):
        lattice = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        v1 = ReciprocalLatticeVector([1, 1, 1], lattice)
        v2 = ReciprocalLatticeVector(hkl, lattice)

        assert_almost_equal(v1.angle(v2), result, decimal=4)


class TestDirectAndReciprocalLatticeVectorCalculations:
    def test_error_if_calculating_inner_product_or_angle_with_unreciprocal_lattices(self, mocker):
        direct_lattice = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        reciprocal_lattice = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC * 1.02)
        direct_vector = DirectLatticeVector([1, 0, 0], direct_lattice)
        reciprocal_vector = ReciprocalLatticeVector([0, 2, 3], reciprocal_lattice)

        with pytest.raises(TypeError) as exception_info:
            direct_vector.inner(reciprocal_vector)
        assert str(exception_info.value) == "DirectLatticeVector and ReciprocalLatticeVector" \
                                            " lattices must be reciprocally related."
        with pytest.raises(TypeError) as exception_info:
            direct_vector.angle(reciprocal_vector)
        assert str(exception_info.value) == "DirectLatticeVector and ReciprocalLatticeVector" \
                                            " lattices must be reciprocally related."

    @pytest.mark.parametrize("uvw,hkl,result", [
        ([1, 0, 0], [0, 0, 1], 0),
        ([1, 0, 0], [1, 0, 0], 2 * pi),
        ([1, -1, 0], [1, 2, 3], -2 * pi),
        ([1, 2, 3], [0, 0, 1], 6 * pi)])
    def test_calculating_inner_product_of_direct_and_reciprocal_lattice_vectors(
            self, mocker, uvw, hkl, result):
        direct_lattice = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        reciprocal_lattice = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        direct_vector = DirectLatticeVector(uvw, direct_lattice)
        reciprocal_vector = ReciprocalLatticeVector(hkl, reciprocal_lattice)

        assert_almost_equal(direct_vector.inner(reciprocal_vector), result)
        assert_almost_equal(reciprocal_vector.inner(direct_vector), result)

    @pytest.mark.parametrize("uvw,hkl,result", [
        ([1, 0, 0], [0, 0, 1], 90),
        ([1, 0, 0], [1, 0, 0], 30),
        ([1, -1, 0], [0, 0, 1], 90),
        ([1, 2, 3], [0, 0, 1], 9.6527)])
    def test_calculating_angle_between_direct_and_reciprocal_lattice_vectors(
            self, mocker, uvw, hkl, result):
        direct_lattice = mocker.MagicMock(metric=CALCITE_DIRECT_METRIC)
        reciprocal_lattice = mocker.MagicMock(metric=CALCITE_RECIPROCAL_METRIC)
        direct_vector = DirectLatticeVector(uvw, direct_lattice)
        reciprocal_vector = ReciprocalLatticeVector(hkl, reciprocal_lattice)

        assert_almost_equal(direct_vector.angle(reciprocal_vector), result, decimal=2)
        assert_almost_equal(reciprocal_vector.angle(direct_vector), result, decimal=2)
