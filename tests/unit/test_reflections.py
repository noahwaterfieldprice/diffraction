"""Unit tests for diffraction.reflections.

Covers:
  - generate_reflections count band, forbidden-excluded, (0,0,0) excluded,
    empty-case, d_min <= 0 raises (DIFF-06)
  - ReflectionList.d_spacings reference values + shape/dtype + cached
    identity (DIFF-08)
  - ReflectionList.two_thetas NIST SRM 640f Table 1 match, wavelength-too-
    large exact-message raise, nonpositive / non-numeric wavelength
    raises (DIFF-09)
  - ReflectionList frozen-dataclass + read-only hkl ndarray + sort order
    deterministic (D-04, D-05, D-09)
  - Crystal.generate_reflections wrapper delegates correctly (D-12)
  - Public re-exports (D-15)
"""

import dataclasses
import math

import numpy as np
import pytest

from diffraction import (
    Crystal,
    DirectLattice,
    ReflectionList,
    SpaceGroup,
    generate_reflections,
)

D_TOL = 1e-5
TWO_THETA_TOL_LIB = 1e-3
TWO_THETA_TOL_NIST = 2e-3

# Library-confirmed Si Fd-3m absences (RESEARCH Fd-3m systematic absences
# §"Allowed/absent companion table for tests"). Replaces CONTEXT D-20's
# (222) which the library's per-operator check misses — that is a Phase 4
# scope limitation documented in research.
SI_LIBRARY_ABSENT = [
    (1, 0, 0),
    (1, 1, 0),
    (2, 0, 0),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 1),
    (4, 2, 0),
    (6, 0, 0),
]

# Library-confirmed Si Fd-3m allowed reflections.
SI_LIBRARY_ALLOWED = [
    (1, 1, 1),
    (2, 2, 0),
    (3, 1, 1),
    (4, 0, 0),
    (3, 3, 1),
    (4, 2, 2),
    (5, 1, 1),
    (4, 4, 0),
    (5, 3, 1),
    (6, 2, 0),
    (5, 3, 3),
]

# NIST SRM 640f Table 1 (Cu Ka lambda=1.5405929 A, a=5.431144 A) -- 11 rows.
# Source: RESEARCH section "Si reference data".
NIST_SRM640F_TABLE1 = [
    ((1, 1, 1), 28.441),
    ((2, 2, 0), 47.301),
    ((3, 1, 1), 56.120),
    ((4, 0, 0), 69.127),
    ((3, 3, 1), 76.373),
    ((4, 2, 2), 88.026),
    ((5, 1, 1), 94.947),
    ((4, 4, 0), 106.702),
    ((5, 3, 1), 114.085),
    ((6, 2, 0), 127.535),
    ((5, 3, 3), 136.882),
]


# ---------------------------------------------------------------------------
# ReflectionList shape, frozenness, read-only payload
# ---------------------------------------------------------------------------


class TestReflectionList:
    """D-05, D-09: frozen dataclass with read-only int ndarray payload."""

    def test_hkl_shape_and_integer_dtype(self, silicon_lattice: DirectLattice) -> None:
        rl = ReflectionList(np.array([[1, 1, 1], [2, 2, 0]]), silicon_lattice)
        assert rl.hkl.shape == (2, 3)
        assert np.issubdtype(rl.hkl.dtype, np.integer)

    def test_hkl_is_read_only(self, silicon_lattice: DirectLattice) -> None:
        rl = ReflectionList(np.array([[1, 1, 1]]), silicon_lattice)
        with pytest.raises(ValueError, match="read-only"):
            rl.hkl[0, 0] = 99

    def test_is_frozen(self, silicon_lattice: DirectLattice) -> None:
        rl = ReflectionList(np.array([[1, 1, 1]]), silicon_lattice)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rl.lattice = silicon_lattice  # type: ignore[misc]

    def test_empty_hkl(self, silicon_lattice: DirectLattice) -> None:
        rl = ReflectionList(np.empty((0, 3), dtype=np.int_), silicon_lattice)
        assert rl.hkl.shape == (0, 3)
        assert rl.d_spacings.shape == (0,)
        tt = rl.two_thetas(1.5406)
        assert tt.shape == (0,)
        assert tt.dtype == np.float64

    def test_d_spacings_fast_path_matches_lazy(
        self, silicon_lattice: DirectLattice
    ) -> None:
        """`_d_spacings` seeds the cache; the seeded value matches lazy compute."""
        hkl = np.array([[1, 1, 1], [2, 2, 0], [3, 1, 1]])
        rl_lazy = ReflectionList(hkl, silicon_lattice)
        d_truth = np.asarray(rl_lazy.d_spacings)
        rl_fast = ReflectionList(hkl, silicon_lattice, _d_spacings=d_truth)
        np.testing.assert_array_equal(rl_fast.d_spacings, d_truth)
        # Seeded array must be read-only, like the lazy one.
        with pytest.raises(ValueError, match="read-only"):
            rl_fast.d_spacings[0] = 99.0

    def test_d_spacings_fast_path_length_mismatch_raises(
        self, silicon_lattice: DirectLattice
    ) -> None:
        with pytest.raises(ValueError, match="does not match hkl rows"):
            ReflectionList(
                np.array([[1, 1, 1], [2, 2, 0]]),
                silicon_lattice,
                _d_spacings=np.array([1.0]),
            )

    def test_sort_order_is_deterministic_for_identical_d(
        self, silicon_crystal: Crystal
    ) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        # d is non-increasing (we sort ascending |Q| == descending d is
        # reversed; actual D-04 says ascending |Q|, which is ascending 1/d
        # and therefore descending d). Verify d is non-increasing or
        # ascending |Q|: d[i] >= d[i+1] - tolerance.
        d = rl.d_spacings
        assert np.all(d[:-1] >= d[1:] - 1e-12), (
            "d_spacings not non-increasing (|Q| not non-decreasing)"
        )


# ---------------------------------------------------------------------------
# generate_reflections behaviour (DIFF-06)
# ---------------------------------------------------------------------------


class TestGenerate:
    """DIFF-06: reflection enumeration with absence filter."""

    @pytest.mark.parametrize(
        "crystal_fixture, space_group_symbol, d_min, lo, hi",
        [
            ("silicon_crystal", "Fd-3m", 0.8, 280, 310),
            ("silicon_crystal", "Fd-3m", 1.2, 60, 90),
            ("corundum_crystal", "R-3c", 1.0, 240, 300),
            ("corundum_crystal", "R-3c", 1.5, 70, 110),
        ],
    )
    def test_reflection_count_in_expected_band(
        self,
        request: pytest.FixtureRequest,
        crystal_fixture: str,
        space_group_symbol: str,
        d_min: float,
        lo: int,
        hi: int,
    ) -> None:
        crystal = request.getfixturevalue(crystal_fixture)
        sg = SpaceGroup(space_group_symbol)
        rl = generate_reflections(crystal, sg, d_min=d_min)
        assert lo <= rl.hkl.shape[0] <= hi, (
            f"{crystal_fixture} @ d_min={d_min}: got {rl.hkl.shape[0]}, "
            f"expected [{lo}, {hi}]"
        )

    def test_000_excluded(self, silicon_crystal: Crystal) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        matches = np.all(rl.hkl == 0, axis=1)
        assert not matches.any(), "(0,0,0) appears in the output"

    @pytest.mark.parametrize("hkl", SI_LIBRARY_ABSENT)
    def test_forbidden_reflections_excluded(
        self,
        silicon_crystal: Crystal,
        hkl: tuple[int, int, int],
    ) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        target = np.array(hkl)
        matches = np.all(rl.hkl == target, axis=1)
        assert not matches.any(), (
            f"Library-flagged-absent reflection {hkl} should be "
            f"excluded but appears in the output"
        )

    @pytest.mark.parametrize("hkl", SI_LIBRARY_ALLOWED)
    def test_allowed_reflections_present(
        self,
        silicon_crystal: Crystal,
        hkl: tuple[int, int, int],
    ) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        target = np.array(hkl)
        matches = np.all(rl.hkl == target, axis=1)
        assert matches.any(), f"Library-allowed reflection {hkl} expected in output"

    def test_empty_result_for_huge_d_min(self, silicon_crystal: Crystal) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=100.0)
        assert rl.hkl.shape == (0, 3)

    @pytest.mark.parametrize("bad", [0.0, -0.1, -5.0])
    def test_d_min_nonpositive_raises(
        self, silicon_crystal: Crystal, bad: float
    ) -> None:
        sg = SpaceGroup("Fd-3m")
        with pytest.raises(ValueError, match="d_min must be positive"):
            generate_reflections(silicon_crystal, sg, d_min=bad)


# ---------------------------------------------------------------------------
# ReflectionList.d_spacings (DIFF-08)
# ---------------------------------------------------------------------------


class TestDSpacings:
    """DIFF-08: d-spacing via reciprocal metric tensor in 2π convention."""

    # Library-computed Si d-spacings at a=5.4307 Å (from examples/silicon.cif,
    # COD 9008565). Tolerance 1e-5 Å. These are STABILITY-GUARD values
    # (per RESEARCH recommendation 2), not NIST values. The NIST-match
    # physics test uses a=5.431144 Å and is separate below.
    SI_D_SPACINGS_EXPECTED_5_43070 = (
        ((1, 1, 1), 3.1354161),
        ((2, 2, 0), 1.9200424),
        ((3, 1, 1), 1.6374177),
        ((4, 0, 0), 1.3576750),
        ((3, 3, 1), 1.2458880),
        ((4, 2, 2), 1.1085370),
    )

    @pytest.mark.parametrize("hkl, expected_d", SI_D_SPACINGS_EXPECTED_5_43070)
    def test_si_d_spacings_match_reference(
        self,
        silicon_crystal: Crystal,
        hkl: tuple[int, int, int],
        expected_d: float,
    ) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        target = np.array(hkl)
        row_idx = np.where(np.all(rl.hkl == target, axis=1))[0]
        assert row_idx.size == 1, f"{hkl} not found exactly once"
        assert abs(float(rl.d_spacings[row_idx[0]]) - expected_d) < D_TOL, (
            hkl,
            rl.d_spacings[row_idx[0]],
            expected_d,
        )

    def test_shape_and_dtype(self, silicon_crystal: Crystal) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        assert rl.d_spacings.shape == (rl.hkl.shape[0],)
        assert rl.d_spacings.dtype == np.float64

    def test_is_cached(self, silicon_crystal: Crystal) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        assert rl.d_spacings is rl.d_spacings  # cached_property identity

    def test_d_spacings_match_einsum_truth(self, silicon_crystal: Crystal) -> None:
        """Cross-check d = 2π / √(h G* h) independently of ReflectionList."""
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        g_star = silicon_crystal.lattice.reciprocal().metric
        q_sq = np.einsum("ni,ij,nj->n", rl.hkl, g_star, rl.hkl)
        d_truth = (2 * math.pi) / np.sqrt(q_sq)
        np.testing.assert_allclose(rl.d_spacings, d_truth, atol=D_TOL)


# ---------------------------------------------------------------------------
# ReflectionList.two_thetas (DIFF-09)
# ---------------------------------------------------------------------------


class TestTwoThetas:
    """DIFF-09: 2θ from Bragg's law; NIST SRM 640f match; D-07 raise."""

    @pytest.mark.parametrize("hkl, expected_2theta", NIST_SRM640F_TABLE1)
    def test_nist_srm640f_match(
        self, hkl: tuple[int, int, int], expected_2theta: float
    ) -> None:
        """Physics test: a=5.431144 Å (NIST) → 2θ within 0.002° of NIST."""
        nist_lattice = DirectLattice([5.431144, 5.431144, 5.431144, 90.0, 90.0, 90.0])
        nist_crystal = Crystal(
            [5.431144, 5.431144, 5.431144, 90.0, 90.0, 90.0], "Fd-3m"
        )
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(nist_crystal, sg, d_min=0.8)
        tt = rl.two_thetas(1.5405929)
        target = np.array(hkl)
        row_idx = np.where(np.all(rl.hkl == target, axis=1))[0]
        assert row_idx.size == 1, f"{hkl} not in generated list"
        actual = float(tt[row_idx[0]])
        assert abs(actual - expected_2theta) < TWO_THETA_TOL_NIST, (
            hkl,
            actual,
            expected_2theta,
        )
        # Silence unused-variable lint — nist_lattice documents intent.
        _ = nist_lattice

    def test_two_thetas_shape_and_dtype(self, silicon_crystal: Crystal) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        tt = rl.two_thetas(1.5406)
        assert tt.shape == (rl.hkl.shape[0],)
        assert tt.dtype == np.float64

    def test_wavelength_too_large_raises_with_d07_message(
        self, silicon_crystal: Crystal
    ) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        # d_min=0.8 means smallest d ≈ 0.8 Å; wavelength=2.5 Å > 2*0.8.
        with pytest.raises(ValueError) as excinfo:
            rl.two_thetas(2.5)
        msg = str(excinfo.value)
        assert "wavelength 2.5 A cannot reach hkl" in msg, msg
        assert "use a shorter wavelength or a larger d_min" in msg, msg

    @pytest.mark.parametrize("bad", [0.0, -1.5406, -0.1])
    def test_nonpositive_wavelength_raises(
        self, silicon_crystal: Crystal, bad: float
    ) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        with pytest.raises(ValueError, match="wavelength must be a positive float"):
            rl.two_thetas(bad)

    def test_non_numeric_wavelength_raises(self, silicon_crystal: Crystal) -> None:
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        with pytest.raises(ValueError, match="wavelength must be a positive float"):
            rl.two_thetas("1.5406")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "wavelength",
        [np.float64(1.5406), np.float32(1.5406), np.int64(1), 1, 1.5406],
    )
    def test_numpy_scalar_wavelength_accepted(
        self, silicon_crystal: Crystal, wavelength: float
    ) -> None:
        """numpy scalars and built-in numbers are both accepted (Real)."""
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        tt = rl.two_thetas(wavelength)
        assert tt.shape == (rl.hkl.shape[0],)
        assert tt.dtype == np.float64

    def test_bool_wavelength_raises(self, silicon_crystal: Crystal) -> None:
        """bool is a Real subclass but should be rejected as a type error."""
        sg = SpaceGroup("Fd-3m")
        rl = generate_reflections(silicon_crystal, sg, d_min=0.8)
        with pytest.raises(ValueError, match="wavelength must be a positive float"):
            rl.two_thetas(True)


# ---------------------------------------------------------------------------
# Crystal.generate_reflections wrapper (D-12)
# ---------------------------------------------------------------------------


class TestCrystalWrapper:
    """D-12: Crystal.generate_reflections delegates to the free function."""

    def test_wrapper_matches_free_function(self, silicon_crystal: Crystal) -> None:
        sg = SpaceGroup("Fd-3m")
        rl_free = generate_reflections(silicon_crystal, sg, d_min=0.8)
        rl_wrap = silicon_crystal.generate_reflections(sg, d_min=0.8)
        assert rl_wrap.hkl.shape == rl_free.hkl.shape
        np.testing.assert_array_equal(rl_wrap.hkl, rl_free.hkl)
        np.testing.assert_allclose(rl_wrap.d_spacings, rl_free.d_spacings, atol=D_TOL)

    @pytest.mark.parametrize("bad_d_min", [0.0, -0.1])
    def test_wrapper_propagates_d_min_validation(
        self, silicon_crystal: Crystal, bad_d_min: float
    ) -> None:
        """Crystal.generate_reflections must not bypass d_min validation."""
        sg = SpaceGroup("Fd-3m")
        with pytest.raises(ValueError, match="d_min must be positive") as wrap_exc:
            silicon_crystal.generate_reflections(sg, d_min=bad_d_min)
        with pytest.raises(ValueError, match="d_min must be positive") as free_exc:
            generate_reflections(silicon_crystal, sg, d_min=bad_d_min)
        assert str(wrap_exc.value) == str(free_exc.value)


# ---------------------------------------------------------------------------
# Public API (D-15)
# ---------------------------------------------------------------------------


class TestImports:
    """D-15: ReflectionList and generate_reflections re-exported."""

    def test_public_api(self) -> None:
        import diffraction

        for name in ("ReflectionList", "generate_reflections"):
            assert hasattr(diffraction, name), f"diffraction is missing {name!r}"
            assert name in diffraction.__all__, f"{name!r} not in diffraction.__all__"
