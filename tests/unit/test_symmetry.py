"""Unit tests for diffraction.symmetry.

All tests use real PointGroup and SpaceGroup instances — no mocks. Both
classes load symmetry operators from bundled JSON files, which is fast
and reliable enough to use in unit tests without mocking.

Tests cover:
  - Creating PointGroup by symbol and by number
  - Operator data loaded correctly
  - String representation
  - Error handling for invalid inputs
  - PointGroup is a frozen dataclass (immutability)
  - ValueError with close-match suggestions for invalid symbols
  - Parametrized lookup table across representative point groups
  - Creating SpaceGroup by symbol and by number
  - SpaceGroup origin choice and setting selection
  - SpaceGroup properties (point_group, centering_type, crystal_system)
  - SpaceGroup systematic absence filtering
  - SpaceGroupError with did-you-mean suggestions
"""

import dataclasses

import pytest

from diffraction import PointGroup, SpaceGroup, SpaceGroupError


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

    def test_point_group_repr_shows_number_and_symbol(self) -> None:
        pg = PointGroup("4/m")

        assert repr(pg) == "PointGroup(number=11, symbol='4/m')"

    def test_point_group_raises_when_neither_symbol_nor_number_given(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            PointGroup()
        assert str(exc_info.value) == (
            "Either the point group symbol or point group number must be given."
        )

    def test_point_group_raises_for_invalid_symbol(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            PointGroup("not_a_symbol")
        assert "Unknown point group symbol 'not_a_symbol'" in str(exc_info.value)

    def test_point_group_invalid_symbol_suggests_close_match(self) -> None:
        # "mmn" is close to "mmm" (valid symbol for point group 8)
        with pytest.raises(ValueError) as exc_info:
            PointGroup("mmn")
        assert "mmm" in str(exc_info.value)

    def test_point_group_raises_for_out_of_range_number(self) -> None:
        # Numbers outside 1-32 have no corresponding JSON data file.
        with pytest.raises(FileNotFoundError):
            PointGroup(number=0)
        with pytest.raises(FileNotFoundError):
            PointGroup(number=33)

    def test_point_group_is_frozen(self) -> None:
        pg = PointGroup("4/m")

        with pytest.raises(dataclasses.FrozenInstanceError):
            pg.number = 99  # type: ignore[misc]

    @pytest.mark.parametrize(
        "symbol, number",
        [
            ("1", 1),  # triclinic, no symmetry
            ("-1", 2),  # triclinic, inversion
            ("mmm", 8),  # orthorhombic
            ("4/mmm", 15),  # tetragonal, highest symmetry
            ("m-3m", 32),  # cubic, highest symmetry
        ],
    )
    def test_point_group_symbol_and_number_are_consistent(
        self, symbol: str, number: int
    ) -> None:
        pg_by_symbol = PointGroup(symbol)
        pg_by_number = PointGroup(number=number)

        assert pg_by_symbol.number == number
        assert pg_by_number.symbol == symbol


class TestSpaceGroupCreation:
    def test_space_group_created_by_symbol(self) -> None:
        sg = SpaceGroup("Fm-3m")

        assert sg.number == 225
        assert sg.symbol == "Fm-3m"

    def test_space_group_created_by_number(self) -> None:
        sg = SpaceGroup(number=225)

        assert sg.symbol == "Fm-3m"

    def test_space_group_created_by_spaced_symbol(self) -> None:
        sg = SpaceGroup("F m -3 m")

        assert sg.number == 225

    def test_space_group_default_origin_choice_2(self) -> None:
        # SG 227 defaults to setting "2" per ITA convention
        sg = SpaceGroup(number=227)

        assert sg.symbol == "Fd-3m"
        assert ":2" in sg.xhm_symbol

    def test_space_group_explicit_origin_choice_1(self) -> None:
        sg = SpaceGroup("Fd-3m:1")

        assert ":1" in sg.xhm_symbol

    def test_space_group_explicit_origin_choice_via_kwarg(self) -> None:
        sg1 = SpaceGroup("Fd-3m:1")
        sg2 = SpaceGroup("Fd-3m", setting=1)

        assert sg1.xhm_symbol == sg2.xhm_symbol

    def test_space_group_rhombohedral_default_hexagonal(self) -> None:
        # SG 167 defaults to hexagonal axes
        sg = SpaceGroup(number=167)

        assert "H" in sg.xhm_symbol

    def test_space_group_rhombohedral_explicit(self) -> None:
        sg = SpaceGroup("R-3c", setting="R")

        assert "R" in sg.xhm_symbol
        assert "H" not in sg.xhm_symbol


class TestSpaceGroupProperties:
    def test_point_group_property(self) -> None:
        sg = SpaceGroup(number=227)

        pg = sg.point_group
        assert isinstance(pg, PointGroup)
        assert pg.symbol == "m-3m"

    def test_centering_type(self) -> None:
        sg = SpaceGroup("Fm-3m")

        assert sg.centering_type == "F"

    def test_crystal_system(self) -> None:
        sg = SpaceGroup(number=227)

        assert sg.crystal_system == "cubic"

    def test_operators_loaded(self) -> None:
        sg = SpaceGroup(number=225)

        assert len(sg.operators) == 48
        op = sg.operators[0]
        assert "W" in op
        assert "t" in op
        assert len(op["W"]) == 3
        assert len(op["W"][0]) == 3
        assert len(op["t"]) == 3

    def test_centering_vectors_loaded(self) -> None:
        # F-centering has 4 centering vectors
        sg = SpaceGroup("Fm-3m")

        assert len(sg.centering_vectors) == 4

    def test_is_frozen_dataclass(self) -> None:
        sg = SpaceGroup("Fm-3m")

        with pytest.raises(dataclasses.FrozenInstanceError):
            sg.number = 99  # type: ignore[misc]


class TestSpaceGroupErrors:
    def test_invalid_symbol_raises_space_group_error(self) -> None:
        with pytest.raises(SpaceGroupError):
            SpaceGroup("Fd3m")

    def test_space_group_error_is_value_error(self) -> None:
        # SpaceGroupError must be a subclass of ValueError for backward compat
        with pytest.raises(ValueError):
            SpaceGroup("Fd3m")

    def test_error_message_includes_suggestion(self) -> None:
        with pytest.raises(SpaceGroupError) as exc_info:
            SpaceGroup("Fd3m")
        assert "Fd-3m" in str(exc_info.value)

    def test_neither_symbol_nor_number_raises(self) -> None:
        with pytest.raises(ValueError):
            SpaceGroup()

    def test_invalid_number_raises(self) -> None:
        with pytest.raises(SpaceGroupError):
            SpaceGroup(number=0)
        with pytest.raises(SpaceGroupError):
            SpaceGroup(number=231)

    def test_both_symbol_and_number_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both"):
            SpaceGroup(symbol="Fm-3m", number=225)

    def test_both_symbol_and_number_raises_even_if_wrong_number(self) -> None:
        # Reject before any lookup — the conflict itself is the error
        with pytest.raises(ValueError, match="Cannot specify both"):
            SpaceGroup(symbol="Fm-3m", number=100)

    def test_setting_with_number_raises(self) -> None:
        with pytest.raises(ValueError, match="setting"):
            SpaceGroup(number=225, setting=1)


class TestSpaceGroupRepr:
    def test_repr(self) -> None:
        sg = SpaceGroup(number=1)

        assert repr(sg) == "SpaceGroup(number=1, symbol='P1')"


class TestSystematicAbsences:
    def test_f_centering_absent_odd_sum(self) -> None:
        # F-centering: h+k, h+l, k+l must all be even; (1,0,0) has h+k=1 (odd)
        sg = SpaceGroup("Fm-3m")

        assert sg.is_systematically_absent((1, 0, 0)) is True

    def test_f_centering_allowed_even_sum(self) -> None:
        sg = SpaceGroup("Fm-3m")

        assert sg.is_systematically_absent((2, 0, 0)) is False

    def test_diamond_glide_absent(self) -> None:
        # Fd-3m (SG 227 setting 2): (2,0,0) is absent due to d-glide
        sg = SpaceGroup("Fd-3m")

        assert sg.is_systematically_absent((2, 0, 0)) is True

    def test_p1_nothing_absent(self) -> None:
        sg = SpaceGroup("P1")

        assert sg.is_systematically_absent((1, 1, 1)) is False

    def test_p21c11_h00_absent_odd(self) -> None:
        # P21/c11 (SG 14, a-axis unique): 21 screw along a -> h00 absent if h odd
        # Symbol is "P21/c11" in our data (a-axis unique setting)
        sg = SpaceGroup("P21/c11")

        assert sg.is_systematically_absent((1, 0, 0)) is True

    def test_p21c11_h00_allowed_even(self) -> None:
        sg = SpaceGroup("P21/c11")

        assert sg.is_systematically_absent((2, 0, 0)) is False


@pytest.mark.parametrize(
    "number, symbol, crystal_system, pg_symbol",
    [
        (1, "P1", "triclinic", "1"),
        (14, "P21/c11", "monoclinic", "2/m"),
        (62, "Pnam", "orthorhombic", "mmm"),
        (136, "P42/mnm", "tetragonal", "4/mmm"),
        (167, "R-3c", "trigonal", "-3m"),
        (194, "P63/mmc", "hexagonal", "6/mmm"),
        (225, "Fm-3m", "cubic", "m-3m"),
    ],
)
def test_space_group_cross_system(
    number: int, symbol: str, crystal_system: str, pg_symbol: str
) -> None:
    sg = SpaceGroup(number=number)

    assert sg.symbol == symbol
    assert sg.crystal_system == crystal_system
    assert sg.point_group.symbol == pg_symbol

