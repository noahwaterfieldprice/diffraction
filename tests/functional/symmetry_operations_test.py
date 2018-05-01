from diffraction.symmetry import PointGroup


def test_loading_point_group_from_symbol():
    point_group = PointGroup("432")

    assert point_group.symbol == "432"
    assert point_group.number == 30


def test_loading_point_group_from_number():
    point_group = PointGroup(number=11)

    assert point_group.number == 11
    assert point_group.symbol == "4/m"


def test_retrieving_point_group_operations_xyz_form():
    point_group = PointGroup("-6m2")

    assert len(point_group.operators["xyz"]) == 12
    assert point_group.operators["xyz"][0] == "x,y,z"
    assert point_group.operators["xyz"][5] == "-x+y,-x,-z"
    assert point_group.operators["xyz"][8] == "x,x-y,z"


def test_retrieving_point_group_operations_matrix_form():
    point_group = PointGroup("4/m")

    assert len(point_group.operators["matrices"]) == 8
    assert point_group.operators["matrices"][0] == [
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert point_group.operators["matrices"][3] == [
        [0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    assert point_group.operators["matrices"][5] == [
        [1, 0, 0], [0, 1, 0], [0, 0, -1]]


def test_retrieving_point_group_operations_ita_form():
    point_group = PointGroup("m-3")

    assert len(point_group.operators["ita"]) == 24
    assert point_group.operators["ita"][0] == "1"
    assert point_group.operators["ita"][5] == "3+ -x,x,-x"
    assert point_group.operators["ita"][19] == "-3+ -x,-x,x; 0,0,0"
