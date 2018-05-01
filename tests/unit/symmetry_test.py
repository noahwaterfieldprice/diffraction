import pytest

from diffraction import PointGroup

TEST_POINT_GROUP = {
    "number": 2, "symbol": "-1",
    "operators": {"xyz": ["x,y,z", "-x,-y,-z"],
                  "matrix": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                             [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]],
                  "ita": ["1", "-1 0,0,0"]}
}


class TestCreatingPointGroups:
    def test_error_if_neither_symbol_nor_number_is_given(self):
        with pytest.raises(ValueError) as exception:
            point_group = PointGroup()
        assert str(exception.value) == ("Either the point group symbol or "
                                        "point group number must be given.")

    def test_operations_loaded_from_correct_file_for_given_symbol(self,
                                                                  mocker):
        point_group_mock = mocker.MagicMock()
        open_mock = mocker.patch(
            "diffraction.symmetry.pkg_resources.resource_string",
            return_value="json_string")
        json_mock = mocker.patch("diffraction.symmetry.json.loads",
                                 return_value={"some_name": "some_value"})

        PointGroup._load_point_group_operators(point_group_mock, "-6m2", None)
        open_mock.assert_called_once_with("diffraction.symmetry",
                                          "static/point_groups/26.json")
        json_mock.assert_called_once_with("json_string")
        assert point_group_mock.some_name == "some_value"

    def test_operations_loaded_from_correct_file_for_given_number(self,
                                                                  mocker):
        point_group_mock = mocker.MagicMock()
        open_mock = mocker.patch(
            "diffraction.symmetry.pkg_resources.resource_string",
            return_value="json_string")
        json_mock = mocker.patch("diffraction.symmetry.json.loads")

        PointGroup._load_point_group_operators(point_group_mock, None, 26)
        open_mock.assert_called_once_with("diffraction.symmetry",
                                          "static/point_groups/26.json")
        json_mock.assert_called_once_with("json_string")

    def test_string_representation_of_point_group(self, mocker):
        point_group_mock = mocker.MagicMock(symbol="6/mmm")
        point_group_mock.__repr__ = PointGroup.__repr__
        point_group_mock.__str__ = PointGroup.__str__
        point_group_mock.__class__.__name__ = "PointGroup"

        assert repr(point_group_mock) == "PointGroup(\"6/mmm\")"
        assert str(point_group_mock) == "PointGroup(\"6/mmm\")"

    def test_point_attributes_are_loaded_correctly(self, mocker):
        point_group_mock = mocker.MagicMock()
        open_mock = mocker.patch(
            "diffraction.symmetry.pkg_resources.resource_string")
        json_mock = mocker.patch("diffraction.symmetry.json.loads",
                                 return_value=TEST_POINT_GROUP)

        PointGroup._load_point_group_operators(point_group_mock, "-1", None)

        assert point_group_mock.number == TEST_POINT_GROUP["number"]
        assert point_group_mock.symbol == TEST_POINT_GROUP["symbol"]
        assert point_group_mock.operators == TEST_POINT_GROUP["operators"]
