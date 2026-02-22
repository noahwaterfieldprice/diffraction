import json
from importlib import resources

__all__ = ["PointGroup"]

Matrix = list[list[int]]

POINT_GROUP_NUMBERS = {
    "1": 1,
    "-1": 2,
    "2": 3,
    "m": 4,
    "2/m": 5,
    "222": 6,
    "mm2": 7,
    "mmm": 8,
    "4": 9,
    "-4": 10,
    "4/m": 11,
    "422": 12,
    "4mm": 13,
    "-42": 14,
    "4/mmm": 15,
    "3": 16,
    "-3": 17,
    "32": 18,
    "3m": 19,
    "-3m": 20,
    "6": 21,
    "-6": 22,
    "6/m": 23,
    "622": 24,
    "6mm": 25,
    "-6m2": 26,
    "6/mmm": 27,
    "23": 28,
    "m-3": 29,
    "432": 30,
    "-43": 31,
    "m-3m": 32,
}


class PointGroup:
    """Class to represent a 3D crystallographic point group.

    Parameters
    ----------
    symbol: str
        The international (or Hermann-Mauguin) symbol denoting the
        point group.
    number: int
        An integer from 1 to 32 denoting the point group.

    Attributes
    ----------
    symbol: str
        The Hermann-Mauguin (or international) symbol denoting the
        point group.
    number: int
        An integer from 1 to 32 denoting the point group.
    operators: dict
        A dictionary containing all the symmetry elements of the point
        group with keys "xyz", "matrix", and "ita" corresponding to
        the x,y,z, matrix and international representations of the
        symmetry operators in the point group. The x,y,z and
        international representations are stored as strings and the
        matrix representation is stored as a 3x3 list.

    Examples
    --------
    >>> from diffraction import PointGroup
    >>> point_group = PointGroup("4/m")
    >>> point_group.operators["xyz"][:4]
    ['x,y,z', '-x,-y,z', '-y,x,z', 'y,-x,z']
    >>> point_group.operators["matrix"][2]
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    >>> point_group.operators["ita"][4]
    '4- 0,0,z'
    """

    def __init__(self, symbol: str | None = None, number: int | None = None):
        if symbol is None and number is None:
            raise ValueError(
                "Either the point group symbol or point group number must be given."
            )
        self.symbol, self.number, self.operators = self._load_point_group_data(
            symbol, number
        )

    @staticmethod
    def _load_point_group_data(
        symbol: str | None = None, number: int | None = None
    ) -> tuple[str, int, dict[str, list[str] | list[Matrix]]]:
        """Load the point group symbol, name and operators from file."""
        if symbol is not None:
            number = POINT_GROUP_NUMBERS[symbol]

        data_file = (
            resources.files("diffraction")
            / "static"
            / "point_groups"
            / f"{number}.json"
        )
        point_group_data = json.loads(data_file.read_text())

        symbol, number, operators = (
            point_group_data["symbol"],
            point_group_data["number"],
            point_group_data["operators"],
        )
        return symbol, number, operators

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.symbol}")'
