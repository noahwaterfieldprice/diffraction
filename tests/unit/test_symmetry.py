"""Unit tests for diffraction.symmetry.

All tests use real PointGroup instances — no mocks. PointGroup loads
symmetry operators from bundled JSON files, which is fast and reliable
enough to use in unit tests without mocking.

Tests cover:
  - Creating PointGroup by symbol and by number
  - Operator data loaded correctly
  - String representation
  - Error handling for invalid inputs
  - Parametrized lookup table across representative point groups
"""

import pytest

from diffraction import PointGroup


class TestPointGroupCreation:
    def test_point_group_created_by_symbol(self) -> None:
        pg = PointGroup("-6m2")

        assert pg.symbol == "-6m2"
        assert pg.number == 26

    def test_point_group_created_by_number(self) -> None:
        pg = PointGroup(number=26)

        assert pg.symbol == "-6m2"

    def test_point_group_operators_loaded(self) -> None:
        pg = PointGroup("-1")

        xyz_ops = pg.operators["xyz"]
        assert len(xyz_ops) == 2
        assert "x,y,z" in xyz_ops
        assert "-x,-y,-z" in xyz_ops

    def test_point_group_repr_shows_symbol(self) -> None:
        pg = PointGroup("4/m")

        assert repr(pg) == 'PointGroup("4/m")'

    def test_point_group_raises_when_neither_symbol_nor_number_given(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            PointGroup()
        assert str(exc_info.value) == (
            "Either the point group symbol or point group number must be given."
        )

    def test_point_group_raises_for_invalid_symbol(self) -> None:
        with pytest.raises(KeyError):
            PointGroup("not_a_symbol")

    def test_point_group_raises_for_out_of_range_number(self) -> None:
        # Numbers outside 1-32 have no corresponding JSON data file.
        with pytest.raises(FileNotFoundError):
            PointGroup(number=0)
        with pytest.raises(FileNotFoundError):
            PointGroup(number=33)

    @pytest.mark.parametrize(
        "symbol, number",
        [
            ("1", 1),       # triclinic, no symmetry
            ("-1", 2),      # triclinic, inversion
            ("mmm", 8),     # orthorhombic
            ("4/mmm", 15),  # tetragonal, highest symmetry
            ("m-3m", 32),   # cubic, highest symmetry
        ],
    )
    def test_point_group_symbol_and_number_are_consistent(
        self, symbol: str, number: int
    ) -> None:
        pg_by_symbol = PointGroup(symbol)
        pg_by_number = PointGroup(number=number)

        assert pg_by_symbol.number == number
        assert pg_by_number.symbol == symbol
