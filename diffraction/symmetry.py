import json
import pkg_resources
__all__ = ["PointGroup"]

POINT_GROUP_NUMBERS = {
    '1': 1, '-1': 2, '2': 3, 'm': 4, '2/m': 5, '222': 6, 'mm2': 7, 'mmm': 8,
    '4': 9, '-4': 10, '4/m': 11, '422': 12, '4mm': 13, '-42': 14, '4/mmm': 15,
    '3': 16, '-3': 17, '32': 18, '3m': 19, '-3m': 20, '6': 21, '-6': 22,
    '6/m': 23, '622': 24, '6mm': 25, '-6m2': 26, '6/mmm': 27, '23': 28,
    'm-3': 29, '432': 30, '-43': 31, 'm-3m': 32
}


class PointGroup:  # TODO: write a better docstring
    """Class to represent a 3D crystallographic point group.

    Parameters
    ----------
    symbol: str
        The international (or Hermann–Mauguin) symbol denoting the
        point group.
    number: int
        An integer from 1 to 32 denoting the point group.

    Attributes
    ----------
    symbol: str
        The Hermann–Mauguin (or international) symbol denoting the
        point group.
    number: int
        An integer from 1 to 32 denoting the point group.
    operations: dict
        A dictionary containing all the symmetry elements of the point
        group with keys "xyz", "matrices", and "ita" corresponding to
        the x,y,z, matrix and international representations of the
        symmetry operations in the point group.


    Examples
    --------
    >>> from diffraction import PointGroup
    >>> point_group = PointGroup("4/m")
    >>> point_group.operatons["xyz"][]

    >>> calcite.gamma
    120.0
    >>> calcite.space_group
    'R -3 c H'
    """
    def __init__(self, symbol=None, number=None):
        if symbol is None and number is None:
            raise ValueError("Either the point group symbol or point group "
                             "number must be given.")
        self._load_point_group_operations(symbol, number)

    def _load_point_group_operations(self, symbol, number):
        if symbol is not None:
            number = POINT_GROUP_NUMBERS[symbol]

        json_string = pkg_resources.resource_string(
            __name__, "static/point_groups/{}.json".format(number))
        point_group_data = json.loads(json_string)

        self.__dict__.update(point_group_data)

    def __repr__(self):
        return "{0}(\"{1}\")".format(self.__class__.__name__, self.symbol)
