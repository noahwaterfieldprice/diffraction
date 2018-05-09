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

    def test_operations_loaded_from_correct_file_for_given_symbol(self, mocker):
        open_mock = mocker.patch(
            "diffraction.symmetry.pkg_resources.resource_string",
            return_value="json_string")
        json_mock = mocker.patch("diffraction.symmetry.json.loads",
                                 return_value=TEST_POINT_GROUP)

        PointGroup._load_point_group_data("-6m2", None)
        open_mock.assert_called_once_with("diffraction.symmetry",
                                          "static/point_groups/26.json")
        json_mock.assert_called_once_with("json_string")

    def test_operations_loaded_from_correct_file_for_given_number(self, mocker):
        open_mock = mocker.patch(
            "diffraction.symmetry.pkg_resources.resource_string",
            return_value="json_string")
        json_mock = mocker.patch("diffraction.symmetry.json.loads",
                                 return_value=TEST_POINT_GROUP)

        PointGroup._load_point_group_data(None, 26)
        open_mock.assert_called_once_with("diffraction.symmetry",
                                          "static/point_groups/26.json")
        json_mock.assert_called_once_with("json_string")

    def test_point_attributes_are_loaded_correctly(self, mocker):
        mocker.patch("diffraction.symmetry.pkg_resources.resource_string")
        mocker.patch("diffraction.symmetry.json.loads",
                     return_value=TEST_POINT_GROUP)

        symbol, number, operators = PointGroup._load_point_group_data("-1", None)
        assert number == TEST_POINT_GROUP["number"]
        assert symbol == TEST_POINT_GROUP["symbol"]
        assert operators == TEST_POINT_GROUP["operators"]

    def test_string_representation_of_point_group(self, mocker):
        point_group_mock = mocker.MagicMock(symbol="6/mmm")
        point_group_mock.__repr__ = PointGroup.__repr__
        point_group_mock.__str__ = PointGroup.__str__
        point_group_mock.__class__.__name__ = "PointGroup"

        assert repr(point_group_mock) == "PointGroup(\"6/mmm\")"
        assert str(point_group_mock) == "PointGroup(\"6/mmm\")"
