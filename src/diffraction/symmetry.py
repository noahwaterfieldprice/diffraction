"""Crystallographic point group symmetry.

Represent the 32 crystallographic point groups using the PointGroup class.
Each group is identified by its Hermann-Mauguin symbol or an integer from
1 to 32. Symmetry operator data (xyz notation, matrix form, ITA notation)
are loaded from bundled JSON files under ``static/point_groups/``.
"""

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
    """One of the 32 three-dimensional crystallographic point groups.

    Load the symmetry operators for the specified point group from a bundled
    JSON data file. The group may be specified by Hermann-Mauguin symbol or
    by its ITA number (1-32).

    Args:
        symbol: Hermann-Mauguin symbol of the point group, e.g. ``'4/m'``.
            Mutually exclusive with ``number``; at least one must be given.
        number: Integer from 1 to 32 identifying the point group. Mutually
            exclusive with ``symbol``; at least one must be given.

    Attributes:
        symbol: Hermann-Mauguin symbol of the point group.
        number: Integer from 1 to 32 identifying the point group.
        operators: Dictionary of symmetry operators with three keys:
            ``'xyz'`` (list of coordinate-triplet strings),
            ``'matrix'`` (list of 3x3 integer matrices), and
            ``'ita'`` (list of ITA notation strings).

    Raises:
        ValueError: If neither ``symbol`` nor ``number`` is provided.

    Examples:
        Create a point group by Hermann-Mauguin symbol and inspect operators:

        >>> from diffraction import PointGroup
        >>> pg = PointGroup("4/m")
        >>> pg.operators["xyz"][:4]
        ['x,y,z', '-x,-y,z', '-y,x,z', 'y,-x,z']
        >>> pg.operators["matrix"][2]
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

        Create the same group by number:

        >>> pg2 = PointGroup(number=11)
        >>> pg2.symbol
        '4/m'
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
        """Load point group symbol, number, and operators from a JSON data file."""
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
