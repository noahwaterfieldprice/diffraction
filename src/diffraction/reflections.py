"""Reflection enumeration: generate symmetry-allowed hkl with d-spacings and 2θ."""

from __future__ import annotations

import functools
import math
import numbers
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .crystal import Crystal
    from .lattice import DirectLattice
    from .symmetry import SpaceGroup

__all__ = ["ReflectionList", "generate_reflections"]


# ---------------------------------------------------------------------------
# ReflectionList frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class ReflectionList:
    """Sorted, symmetry-allowed hkl with derived d-spacings and 2θ on demand.

    Construct via :func:`generate_reflections` — do not instantiate directly
    in normal use.

    The `hkl` array is a read-only C-contiguous int ndarray (D-09). The
    `d_spacings` property is computed on first access via the reciprocal
    metric tensor in the 2π convention and cached on the instance (D-06).

    Attributes:
        hkl: (N, 3) integer ndarray of Miller indices. writeable=False.
        lattice: The DirectLattice used to derive reciprocal-space values.
    """

    hkl: NDArray[np.int64] = field(repr=False)
    lattice: DirectLattice

    def __init__(
        self,
        hkl: NDArray[np.int64],
        lattice: DirectLattice,
        *,
        _d_spacings: NDArray[np.float64] | None = None,
    ) -> None:
        arr = np.ascontiguousarray(hkl, dtype=np.int64).reshape(-1, 3)
        arr.flags.writeable = False
        object.__setattr__(self, "hkl", arr)
        object.__setattr__(self, "lattice", lattice)
        # `_d_spacings` is a private fast-path used by `generate_reflections`
        # to avoid re-running the einsum that already happened during
        # enumeration. It seeds the cached_property cache directly. External
        # callers that construct ReflectionList by hand should leave it None
        # and let d_spacings compute on first access.
        if _d_spacings is not None:
            d_arr = np.ascontiguousarray(_d_spacings, dtype=np.float64).reshape(-1)
            if d_arr.shape[0] != arr.shape[0]:
                raise ValueError(
                    f"_d_spacings length {d_arr.shape[0]} does not match "
                    f"hkl rows {arr.shape[0]}"
                )
            d_arr.flags.writeable = False
            object.__setattr__(self, "d_spacings", d_arr)

    def __repr__(self) -> str:
        n = int(self.hkl.shape[0])
        if n == 0:
            return "ReflectionList(0 reflections)"
        d_min_val = float(self.d_spacings.min())
        return f"ReflectionList({n} reflections, d_min≈{d_min_val:.3f} Å)"

    @functools.cached_property
    def d_spacings(self) -> NDArray[np.float64]:
        """d-spacing in Å for each row of `hkl` — 2π / √(h G* h)."""
        if self.hkl.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        g_star = self.lattice.reciprocal().metric
        q_sq = np.einsum("ni,ij,nj->n", self.hkl, g_star, self.hkl)
        result: NDArray[np.float64] = (2 * math.pi) / np.sqrt(q_sq)
        return result

    def two_thetas(self, wavelength: float) -> NDArray[np.float64]:
        """2θ in degrees for each reflection at the given wavelength.

        Reachability is checked all-or-nothing: if any reflection in
        `self` has `d < wavelength/2` (i.e. lies outside the Ewald
        sphere at this wavelength), the call raises rather than
        returning a shorter array. This preserves the index
        correspondence with `self.hkl` and `self.d_spacings`, which
        callers (e.g. the Phase 7 intensity engine) rely on. To work
        with a partial set, regenerate the reflection list with a
        larger `d_min` so every reflection is reachable.

        Args:
            wavelength: Incident wavelength in Å. Must be a positive
                real number (Python int/float or numpy scalar) and at
                most 2·min(d_spacings).

        Returns:
            (N,) float64 ndarray of 2θ in degrees, aligned with
            `self.hkl` row-for-row.

        Raises:
            ValueError: If `wavelength <= 0`, if `wavelength` is not a
                real number, or if any reflection has `d < wavelength/2`
                (Bragg reach).
        """
        w_in: object = wavelength
        if type(w_in) is bool or not isinstance(w_in, numbers.Real):
            raise ValueError(f"wavelength must be a positive float, got {wavelength!r}")
        w = float(w_in)
        if w <= 0:
            raise ValueError(f"wavelength must be a positive float, got {wavelength!r}")
        d = self.d_spacings
        if d.size == 0:
            return np.empty(0, dtype=np.float64)
        worst_idx = int(np.argmin(d))
        d_worst = float(d[worst_idx])
        if w > 2 * d_worst:
            hkl_worst = tuple(int(x) for x in self.hkl[worst_idx])
            raise ValueError(
                f"wavelength {w} A cannot reach hkl {hkl_worst} "
                f"(d={d_worst:.4f} A); use a shorter wavelength or a "
                f"larger d_min"
            )
        # Clip against floating-point roundoff at the Bragg limit (w ≈ 2·d):
        # the explicit check above rejects out-of-domain inputs, but ratios
        # can drift ~1 ULP above 1.0 and return NaN from np.arcsin otherwise.
        ratio = np.clip(w / (2 * d), -1.0, 1.0)
        result: NDArray[np.float64] = 2 * np.degrees(np.arcsin(ratio))
        return result


# ---------------------------------------------------------------------------
# generate_reflections free function
# ---------------------------------------------------------------------------


def _axis_box_bounds(
    reciprocal_metric: NDArray[np.float64], d_min: float
) -> tuple[int, int, int]:
    """Return (h_max, k_max, l_max) for the reciprocal-metric 2π convention.

    For each axis i with |h_i| ≤ h_max and other indices zero, the bound
    comes from ``|Q|² = h_i² · G*_ii ≤ (2π/d_min)²`` →
    ``h_max = ceil(2π / (d_min · √G*_ii))``. The 2π factor is required
    because this project's reciprocal metric already carries it (see
    `lattice.py::_reciprocalise`); using the textbook no-2π form
    under-enumerates by a factor of ~6.
    """
    q_max = 2 * math.pi / d_min
    diag = np.diag(reciprocal_metric)
    return (
        math.ceil(q_max / math.sqrt(diag[0])),
        math.ceil(q_max / math.sqrt(diag[1])),
        math.ceil(q_max / math.sqrt(diag[2])),
    )


def generate_reflections(
    crystal: Crystal,
    space_group: SpaceGroup,
    *,
    d_min: float,
) -> ReflectionList:
    """Enumerate symmetry-allowed hkl at `d ≥ d_min`.

    The result contains every allowed hkl (both positive and negative
    indices), excludes (0, 0, 0), removes systematic absences via
    :meth:`SpaceGroup.is_systematically_absent`, and is sorted by
    ascending |Q| with lexicographic tiebreak on (h, k, l).

    Args:
        crystal: Target crystal; its lattice supplies the reciprocal
            metric.
        space_group: Space group whose operators define systematic
            absences.
        d_min: Minimum d-spacing in Å. Keyword-only. Must be positive.

    Returns:
        ReflectionList containing the allowed hkl.

    Raises:
        ValueError: If `d_min <= 0`.

    Examples:
        >>> from diffraction import Crystal, SpaceGroup, generate_reflections
        >>> silicon = Crystal.from_cif("examples/silicon.cif")  # doctest: +SKIP
        >>> sg = SpaceGroup("Fd-3m")  # doctest: +SKIP
        >>> rl = generate_reflections(silicon, sg, d_min=0.8)  # doctest: +SKIP
    """
    if d_min <= 0:
        raise ValueError(f"d_min must be positive, got {d_min!r}")

    g_star = crystal.lattice.reciprocal().metric
    h_max, k_max, l_max = _axis_box_bounds(g_star, d_min)

    grid = np.indices((2 * h_max + 1, 2 * k_max + 1, 2 * l_max + 1))
    hkl_all = grid.reshape(3, -1).T - np.array([h_max, k_max, l_max], dtype=np.int64)
    nonzero = np.any(hkl_all != 0, axis=1)
    hkl_all = hkl_all[nonzero]

    if hkl_all.shape[0] == 0:
        return ReflectionList(np.empty((0, 3), dtype=np.int64), crystal.lattice)

    q_sq = np.einsum("ni,ij,nj->n", hkl_all, g_star, hkl_all)
    d_all = (2 * math.pi) / np.sqrt(q_sq)
    d_mask = d_all >= d_min
    hkl_after_d = hkl_all[d_mask]
    d_after_d = d_all[d_mask]

    if hkl_after_d.shape[0] == 0:
        return ReflectionList(np.empty((0, 3), dtype=np.int64), crystal.lattice)

    # D-16: per-hkl Python loop; tuple pre-conversion avoids rebuilding
    # np.array inside `is_systematically_absent`.
    candidate_tuples: list[tuple[int, int, int]] = [
        (int(h), int(k), int(ll)) for h, k, ll in hkl_after_d
    ]
    allowed_mask = np.array(
        [not space_group.is_systematically_absent(t) for t in candidate_tuples],
        dtype=bool,
    )
    hkl_allowed = hkl_after_d[allowed_mask]
    d_allowed = d_after_d[allowed_mask]

    if hkl_allowed.shape[0] == 0:
        return ReflectionList(np.empty((0, 3), dtype=np.int64), crystal.lattice)

    # D-04: primary key = -d (ascending |Q|); tiebreak = lex (h, k, l).
    # np.lexsort uses the LAST key as primary.
    order = np.lexsort(
        (
            hkl_allowed[:, 2],
            hkl_allowed[:, 1],
            hkl_allowed[:, 0],
            -d_allowed,
        )
    )
    sorted_hkl = hkl_allowed[order]
    sorted_d = d_allowed[order]
    return ReflectionList(sorted_hkl, crystal.lattice, _d_spacings=sorted_d)
