"""Crystallographic point group and space group symmetry.

Represent the 32 crystallographic point groups using the PointGroup class
and all 230 space groups using the SpaceGroup class.

Each point group is identified by its Hermann-Mauguin symbol or an integer
from 1 to 32. Each space group is identified by its Hermann-Mauguin symbol
(compact or spaced xHM form) or its ITA number (1-230). Symmetry operator
data are loaded from bundled JSON files under ``static/point_groups/`` and
``static/space_groups/`` respectively.
"""

import difflib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from importlib import resources
from typing import Any, TypeAlias

import numpy as np

from .exceptions import SpaceGroupError

__all__ = ["PointGroup", "SpaceGroup"]

Matrix: TypeAlias = list[list[int]]

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


@dataclass(frozen=True, init=False)
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
        ValueError: If ``symbol`` is not a recognised Hermann-Mauguin symbol.

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

    number: int
    symbol: str
    operators: dict[str, list[str] | list[Matrix]] = field(repr=False)

    def __init__(self, symbol: str | None = None, number: int | None = None) -> None:
        if symbol is None and number is None:
            raise ValueError(
                "Either the point group symbol or point group number must be given."
            )
        if symbol is not None:
            try:
                resolved_number = POINT_GROUP_NUMBERS[symbol]
            except KeyError:
                close = difflib.get_close_matches(
                    symbol, POINT_GROUP_NUMBERS.keys(), n=3, cutoff=0.6
                )
                suggestion = (
                    f" Did you mean: {', '.join(repr(m) for m in close)}?"
                    if close
                    else ""
                )
                raise ValueError(
                    f"Unknown point group symbol {symbol!r}.{suggestion}"
                ) from None
        else:
            resolved_number = number  # type: ignore[assignment]
        data = self._load_point_group_data(resolved_number)
        object.__setattr__(self, "number", data["number"])
        object.__setattr__(self, "symbol", data["symbol"])
        object.__setattr__(self, "operators", data["operators"])

    @staticmethod
    def _load_point_group_data(number: int) -> dict[str, object]:
        """Load point group data from a JSON data file."""
        data_file = (
            resources.files("diffraction")
            / "static"
            / "point_groups"
            / f"{number}.json"
        )
        result: dict[str, object] = json.loads(data_file.read_text())
        return result


# ---------------------------------------------------------------------------
# SpaceGroup module-level lookup tables (loaded lazily on first use)
# ---------------------------------------------------------------------------

# Mapping: sg_number -> raw JSON data dict
_SPACE_GROUP_DATA: dict[int, dict[str, Any]] = {}

# Mapping: normalised symbol -> (sg_number, setting_key)
# Both compact (spaces stripped) and spaced xHM forms are registered.
_SPACE_GROUP_SYMBOLS: dict[str, tuple[int, str]] = {}

_sg_data_loaded: bool = False


def _normalize_sg_symbol(symbol: str) -> str:
    """Normalize a space group symbol to compact form (no spaces).

    Strips spaces from the base symbol while preserving any setting suffix
    after ':'.

    Examples::

        'F d -3 m :2' -> 'Fd-3m:2'
        'Fd-3m'       -> 'Fd-3m'
        'F m -3 m'    -> 'Fm-3m'
        'R-3c:H'      -> 'R-3c:H'
    """
    if ":" in symbol:
        base, _, suffix = symbol.partition(":")
        return base.replace(" ", "") + ":" + suffix.strip()
    return symbol.replace(" ", "")


def _load_all_space_group_data() -> None:
    """Populate _SPACE_GROUP_DATA and _SPACE_GROUP_SYMBOLS from JSON files."""
    global _sg_data_loaded

    for sg_number in range(1, 231):
        data_file = (
            resources.files("diffraction")
            / "static"
            / "space_groups"
            / f"{sg_number}.json"
        )
        data: dict[str, Any] = json.loads(data_file.read_text())
        _SPACE_GROUP_DATA[sg_number] = data

        default_setting = data["default_setting"]
        sg_symbol = data["symbol"]  # compact symbol from JSON top-level

        for setting_key, setting_data in data["settings"].items():
            xhm: str = setting_data["xhm_symbol"]
            # Compact (no-spaces) forms: e.g. 'Fd-3m:2', 'Fm-3m'
            compact_with_suffix = _normalize_sg_symbol(xhm)

            # Register the full compact-with-suffix form
            _SPACE_GROUP_SYMBOLS[compact_with_suffix] = (sg_number, setting_key)

            # Also register the bare spaced xHM form (spaces intact, no suffix strip)
            # e.g. 'F d -3 m :2'
            _SPACE_GROUP_SYMBOLS[xhm] = (sg_number, setting_key)

            # For the default setting, also register the bare symbol without suffix
            if setting_key == default_setting:
                # Compact bare (e.g. 'Fd-3m', 'Fm-3m', 'R-3c')
                _SPACE_GROUP_SYMBOLS[sg_symbol] = (sg_number, setting_key)
                # Spaced bare (e.g. 'F m -3 m')
                spaced_bare = xhm.split(":")[0].strip() if ":" in xhm else xhm
                if spaced_bare != sg_symbol:
                    _SPACE_GROUP_SYMBOLS[spaced_bare] = (sg_number, setting_key)

    _sg_data_loaded = True


def _ensure_sg_data_loaded() -> None:
    """Lazily load all space group data on first access."""
    if not _sg_data_loaded:
        _load_all_space_group_data()


# ---------------------------------------------------------------------------
# SpaceGroup frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class SpaceGroup:
    """One of the 230 three-dimensional crystallographic space groups.

    Load the symmetry operators and centering vectors for the specified
    space group from bundled JSON data files. The group may be specified
    by Hermann-Mauguin symbol (compact or spaced xHM form, with optional
    setting suffix) or by its ITA number (1-230).

    For space groups with multiple settings (origin choices 1/2, hexagonal/
    rhombohedral axes), the default setting follows ITA conventions:
    origin choice 2 for the 24 centrosymmetric groups, hexagonal axes for
    the 7 rhombohedral groups.

    Args:
        symbol: Hermann-Mauguin symbol of the space group, e.g. ``'Fd-3m'``
            or ``'F d -3 m'``. An explicit setting suffix may be appended:
            ``'Fd-3m:1'``, ``'R-3c:H'``. Mutually exclusive with
            ``number``; at least one must be given.
        number: Integer from 1 to 230 identifying the space group. Mutually
            exclusive with ``symbol``; at least one must be given.
        setting: Optional setting key (int or str) to select a non-default
            setting when using ``symbol``. E.g. ``setting=1`` selects
            origin choice 1 for SG 227.

    Attributes:
        number: ITA number (1-230).
        symbol: Compact Hermann-Mauguin symbol, e.g. ``'Fd-3m'``.
        xhm_symbol: Full spaced xHM symbol from ITA, e.g. ``'F d -3 m :2'``.
        operators: List of symmetry operators, each a dict with keys
            ``'W'`` (3x3 integer matrix as list-of-lists) and
            ``'t'`` (translation as 3-element list of fraction strings).
        centering_vectors: List of centering translation vectors, each a
            3-element list of fraction strings.

    Raises:
        ValueError: If neither ``symbol`` nor ``number`` is provided.
        SpaceGroupError: If ``symbol`` is not recognised (with did-you-mean
            suggestion) or if ``number`` is outside 1-230.

    Examples:
        Load diamond-structure space group by symbol and check absences:

        >>> from diffraction import SpaceGroup
        >>> sg = SpaceGroup("Fd-3m")
        >>> sg.number
        227
        >>> sg.is_systematically_absent((2, 0, 0))
        True

        Load by number and inspect properties:

        >>> sg2 = SpaceGroup(number=225)
        >>> sg2.crystal_system
        'cubic'
        >>> sg2.centering_type
        'F'
    """

    number: int
    symbol: str
    xhm_symbol: str
    operators: list[dict[str, Any]] = field(repr=False)
    centering_vectors: list[list[str]] = field(repr=False)
    _point_group_symbol: str = field(repr=False)
    _crystal_system: str = field(repr=False)

    def __init__(
        self,
        symbol: str | None = None,
        number: int | None = None,
        setting: int | str | None = None,
    ) -> None:
        _ensure_sg_data_loaded()

        if symbol is None and number is None:
            raise ValueError(
                "Either the space group symbol or space group number must be given."
            )

        if symbol is not None and number is not None:
            raise ValueError(
                "Cannot specify both 'symbol' and 'number'. Use one or the other."
            )

        if number is not None and symbol is None:
            if setting is not None:
                raise ValueError(
                    "The 'setting' parameter is only valid when specifying a space "
                    "group by 'symbol', not by 'number'."
                )
            # Look up by number
            if number not in _SPACE_GROUP_DATA:
                raise SpaceGroupError(
                    f"Space group number {number} is out of range. "
                    "Valid numbers are 1 to 230."
                )
            data = _SPACE_GROUP_DATA[number]
            setting_key = data["default_setting"]
        else:
            # Look up by symbol (with optional setting override)
            assert symbol is not None
            lookup_sym = _normalize_sg_symbol(symbol)
            if setting is not None:
                # Append setting suffix if not already present
                base = lookup_sym.partition(":")[0]
                lookup_sym = f"{base}:{setting}"

            if lookup_sym not in _SPACE_GROUP_SYMBOLS:
                # Build did-you-mean suggestion from all known compact symbols
                all_symbols = list(_SPACE_GROUP_SYMBOLS.keys())
                close = difflib.get_close_matches(
                    lookup_sym, all_symbols, n=3, cutoff=0.6
                )
                suggestion = (
                    f" Did you mean: {', '.join(repr(m) for m in close)}?"
                    if close
                    else ""
                )
                raise SpaceGroupError(
                    f"Unknown space group symbol {symbol!r}.{suggestion}"
                )

            sg_number, setting_key = _SPACE_GROUP_SYMBOLS[lookup_sym]
            data = _SPACE_GROUP_DATA[sg_number]

        setting_data = data["settings"][setting_key]
        object.__setattr__(self, "number", data["number"])
        object.__setattr__(self, "symbol", data["symbol"])
        object.__setattr__(self, "xhm_symbol", setting_data["xhm_symbol"])
        object.__setattr__(self, "operators", setting_data["operators"])
        object.__setattr__(self, "centering_vectors", setting_data["centering_vectors"])
        object.__setattr__(self, "_point_group_symbol", data["point_group"])
        object.__setattr__(self, "_crystal_system", data["crystal_system"])

    @property
    def point_group(self) -> PointGroup:
        """The crystallographic point group of this space group."""
        return PointGroup(self._point_group_symbol)

    @property
    def centering_type(self) -> str:
        """Lattice centering letter (P, A, B, C, I, F, or R)."""
        return self.xhm_symbol.strip()[0]

    @property
    def crystal_system(self) -> str:
        """Crystal system name (triclinic, monoclinic, ...)."""
        return self._crystal_system

    def is_systematically_absent(self, hkl: tuple[int, int, int]) -> bool:
        """Return True if the reflection *hkl* is systematically absent.

        Uses the exact h·W = h criterion with Fraction arithmetic for the
        phase shift h·t. A reflection is absent when an operator leaves h
        invariant (W^T h = h) but introduces a non-integer phase shift.

        Args:
            hkl: Miller indices (h, k, l) of the reflection.

        Returns:
            True if the reflection is systematically absent, False otherwise.
        """
        h = np.array(hkl, dtype=int)
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Build full operator list: crystallographic operators plus centering
        # translations represented as (W=I, t=centering_vector).
        all_ops: list[dict[str, Any]] = list(self.operators)
        all_ops.extend(
            {"W": identity, "t": cv}
            for cv in self.centering_vectors
            if any(v != "0" for v in cv)
        )
        for op in all_ops:
            w_arr = np.array(op["W"], dtype=int)
            # Check h·W = h  (equivalently W^T h = h)
            if np.array_equal(w_arr.T @ h, h):
                # Compute h·t as exact rational arithmetic
                t = op["t"]
                ht = sum(int(h[i]) * Fraction(t[i]) for i in range(3))
                if ht.denominator != 1:
                    return True

        return False

    def __repr__(self) -> str:
        return f"SpaceGroup(number={self.number}, symbol={self.symbol!r})"
