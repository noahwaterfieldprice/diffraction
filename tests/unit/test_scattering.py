"""Unit tests for diffraction.scattering.

Covers:
  - get_element lookup by symbol and by atomic number (SCAT-03)
  - Element frozen dataclass shape, dtype, contiguity (SCAT-04)
  - Ionic label rejection with exact D-18 message (OPT-02 guardrail)
  - Unknown symbol / unknown atomic number with exact D-18 messages
  - get_neutral_symbol against both grammars (Na1+, Fe3+, Cl1-, O2-)
  - Module-level caching: JSON read happens at most once (SCAT-06)
  - Bundled data shape: 95 entries, Z=1-95 covered
  - Neutron-missing raises for Po, At, Rn, Fr, Ac, Pu; Tc returns a float
"""

import dataclasses

import numpy as np
import pytest

from diffraction import (
    Element,  # noqa: F401 — exercised by TestImports
    ScatteringDataError,
    get_element,
    get_neutral_symbol,
    neutron_scattering_length,
    xray_form_factor,
)


class TestBundledData:
    """Tests at the loaded-module level (SCAT-01, SCAT-02 data-shape)."""

    def test_element_count(self) -> None:
        get_element("Si")  # force cache populate
        from diffraction import scattering as _scat

        assert len(_scat._BY_SYMBOL) == 95
        assert len(_scat._BY_NUMBER) == 95

    def test_z_range_is_1_to_95(self) -> None:
        get_element("Si")
        from diffraction import scattering as _scat

        assert sorted(_scat._BY_NUMBER.keys()) == list(range(1, 96))

    def test_tc_is_present_with_neutron_value(self) -> None:
        # Guardrail: Tc (Z=43) is NOT in the neutron-missing set.
        tc = get_element("Tc")
        assert tc.neutron_b_coh is not None
        assert abs(tc.neutron_b_coh - 6.8000) < 1e-3


class TestGetElement:
    """SCAT-03: polymorphic get_element(str | int)."""

    def test_lookup_by_symbol(self) -> None:
        fe = get_element("Fe")
        assert fe.z == 26
        assert fe.symbol == "Fe"
        assert fe.name == "Iron"

    def test_lookup_by_atomic_number(self) -> None:
        fe = get_element(26)
        assert fe.z == 26
        assert fe.symbol == "Fe"

    def test_symbol_and_number_return_same_instance(self) -> None:
        assert get_element("Fe") is get_element(26)

    def test_unknown_symbol_raises(self) -> None:
        with pytest.raises(ScatteringDataError, match=r"Unknown element symbol 'Fm'"):
            get_element("Fm")

    def test_unknown_atomic_number_raises(self) -> None:
        with pytest.raises(ScatteringDataError, match=r"Unknown atomic number 100"):
            get_element(100)

    def test_ionic_label_raises_with_exact_message(self) -> None:
        # D-18 mandates the exact wording; downstream callers match against it.
        with pytest.raises(
            ScatteringDataError,
            match=(
                r"Ionic form factors not available in v1\.0; "
                r"use neutral symbol 'Fe' or call get_neutral_symbol\(\) first"
            ),
        ):
            get_element("Fe3+")

    def test_cif_ionic_label_also_raises(self) -> None:
        # CIF grammar: Na1+, Cl1-.
        with pytest.raises(
            ScatteringDataError, match=r"Ionic form factors not available"
        ):
            get_element("Na1+")

    def test_float_key_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            get_element(26.0)  # type: ignore[arg-type]

    def test_bool_key_raises_type_error(self) -> None:
        # Guard against Python treating bool as int: get_element(True) != get_element(1).
        with pytest.raises(TypeError):
            get_element(True)

    def test_lookup_strips_whitespace(self) -> None:
        # CIF-style labels occasionally arrive with stray whitespace; match the
        # normalization applied by get_neutral_symbol.
        assert get_element(" Si ") is get_element("Si")

    def test_lookup_is_case_insensitive(self) -> None:
        assert get_element("fe") is get_element("Fe")
        assert get_element("CL") is get_element("Cl")


class TestElementContract:
    """SCAT-04: array-based coefficient schema and frozen-dataclass contract."""

    def test_cromer_mann_a_is_contiguous_float64_shape_4(self) -> None:
        si = get_element("Si")
        assert isinstance(si.cromer_mann_a, np.ndarray)
        assert si.cromer_mann_a.shape == (4,)
        assert si.cromer_mann_a.dtype == np.float64
        assert si.cromer_mann_a.flags["C_CONTIGUOUS"]

    def test_cromer_mann_b_is_contiguous_float64_shape_4(self) -> None:
        si = get_element("Si")
        assert isinstance(si.cromer_mann_b, np.ndarray)
        assert si.cromer_mann_b.shape == (4,)
        assert si.cromer_mann_b.dtype == np.float64
        assert si.cromer_mann_b.flags["C_CONTIGUOUS"]

    def test_cromer_mann_c_is_float(self) -> None:
        assert isinstance(get_element("Si").cromer_mann_c, float)

    def test_neutron_b_coh_is_float_for_tabulated(self) -> None:
        assert isinstance(get_element("Si").neutron_b_coh, float)

    def test_neutron_b_coh_is_none_for_missing(self) -> None:
        # Pu (Z=94) is one of the six tabulated-missing elements.
        assert get_element("Pu").neutron_b_coh is None

    def test_element_is_frozen(self) -> None:
        si = get_element("Si")
        with pytest.raises(dataclasses.FrozenInstanceError):
            si.z = 99  # type: ignore[misc]

    def test_cromer_mann_arrays_are_read_only(self) -> None:
        # The cached Element instance is shared across callers; mutating the
        # arrays in place would silently corrupt the module-level cache.
        # Mirror the lattice.py writeable=False guarantee for cached numpy state.
        si = get_element("Si")
        with pytest.raises(ValueError, match="read-only"):
            si.cromer_mann_a[0] = 0.0
        with pytest.raises(ValueError, match="read-only"):
            si.cromer_mann_b[0] = 0.0


class TestGetNeutralSymbol:
    """D-12: charge-stripping helper must accept both ITC and CIF grammars."""

    @pytest.mark.parametrize(
        "label, expected",
        [
            ("Fe", "Fe"),
            ("Si", "Si"),
            ("Fe3+", "Fe"),
            ("O2-", "O"),
            ("Na1+", "Na"),  # CIF grammar
            ("Cl1-", "Cl"),  # CIF grammar
        ],
    )
    def test_strips_charge(self, label: str, expected: str) -> None:
        assert get_neutral_symbol(label) == expected

    def test_malformed_label_raises(self) -> None:
        with pytest.raises(ScatteringDataError):
            get_neutral_symbol("not_an_element!")

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ScatteringDataError):
            get_neutral_symbol("")


class TestNeutronLookupEdgeCases:
    """Tabulated-missing set (D-03 amended 2026-04-21)."""

    @pytest.mark.parametrize(
        "symbol, z",
        [("Po", 84), ("At", 85), ("Rn", 86), ("Fr", 87), ("Ac", 89), ("Pu", 94)],
    )
    def test_neutron_raises_not_tabulated(self, symbol: str, z: int) -> None:
        with pytest.raises(
            ScatteringDataError,
            match=rf"Neutron scattering length not tabulated for {symbol} \(Z={z}\)",
        ):
            neutron_scattering_length(symbol)

    def test_tc_has_neutron_value(self) -> None:
        # Tc (Z=43) has Coh_b=6.8000 in the source file. Earlier CONTEXT drafts
        # listed it as missing; D-03 was amended 2026-04-21 after source inspection.
        assert isinstance(neutron_scattering_length("Tc"), float)

    def test_neutron_raises_unknown_symbol(self) -> None:
        with pytest.raises(ScatteringDataError, match=r"Unknown element symbol 'Fm'"):
            neutron_scattering_length("Fm")

    def test_neutron_raises_on_ionic_label(self) -> None:
        with pytest.raises(
            ScatteringDataError, match=r"Ionic form factors not available"
        ):
            neutron_scattering_length("Fe3+")


class TestCaching:
    """SCAT-06: load once, reuse thereafter."""

    def test_second_call_does_not_reload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from diffraction import scattering as _scat

        # Clear module-level state so this test reloads deterministically.
        monkeypatch.setattr(_scat, "_element_data_loaded", False)
        _scat._BY_SYMBOL.clear()
        _scat._BY_NUMBER.clear()

        # Patch the loader on the module to count how many times it runs.
        # SCAT-06 contract: once _BY_SYMBOL/_BY_NUMBER are populated, the
        # loader must not re-execute for subsequent lookups.
        call_count = {"n": 0}
        real_loader = _scat._load_element_data

        def counting_loader() -> None:
            call_count["n"] += 1
            real_loader()

        monkeypatch.setattr(_scat, "_load_element_data", counting_loader)

        get_element("Si")
        get_element("Fe")
        get_element(29)
        get_neutral_symbol("Na1+")  # stateless helper must not touch the cache

        assert call_count["n"] == 1


class TestImports:
    """D-25: public re-exports from the package root."""

    def test_all_names_importable_from_package_root(self) -> None:
        import diffraction

        expected = {
            "Element",
            "ScatteringDataError",
            "get_element",
            "get_neutral_symbol",
            "neutron_scattering_length",
            "xray_form_factor",
        }
        for name in expected:
            assert hasattr(diffraction, name), f"diffraction is missing {name!r}"
            assert name in diffraction.__all__, f"{name!r} not in diffraction.__all__"

    def test_scattering_data_error_double_inheritance(self) -> None:
        from diffraction import DiffractionError

        assert issubclass(ScatteringDataError, DiffractionError)
        assert issubclass(ScatteringDataError, ValueError)


class TestFormFactorPhysics:
    """SCAT-05: f(stol=0) ≈ Z; reference values match source coefficients."""

    # Empirical residuals from Dans source file:
    #   Si f(0) = 13.9976 (residual 0.0024)
    #   Fe f(0) = 25.9904 (residual 0.0096)
    # Tolerance per CONTEXT D-13: abs(f(0) - Z) < 0.05.
    _F_ZERO_TOL = 0.05

    def test_si_f_at_stol_zero_approx_z(self) -> None:
        f0 = xray_form_factor("Si", stol=0.0)
        assert isinstance(f0, float)
        assert abs(f0 - 14.0) < self._F_ZERO_TOL

    def test_fe_f_at_stol_zero_approx_z(self) -> None:
        f0 = xray_form_factor("Fe", stol=0.0)
        assert isinstance(f0, float)
        assert abs(f0 - 26.0) < self._F_ZERO_TOL

    def test_si_f_zero_self_consistent(self) -> None:
        # Self-consistency: f(0) from source coefficients equals 13.9976.
        f0 = xray_form_factor("Si", 0.0)
        assert isinstance(f0, float)
        assert abs(f0 - 13.9976) < 1e-4

    def test_fe_f_stol_0p5_self_consistent(self) -> None:
        # Reference 11.5057 computed from Dans coefficients in 05-RESEARCH.md.
        f = xray_form_factor("Fe", 0.5)
        assert isinstance(f, float)
        assert abs(f - 11.5057) < 1e-3

    def test_form_factor_monotonically_decreasing_for_fe(self) -> None:
        stols = [0.0, 0.3, 0.6, 1.0]
        fvals: list[float] = []
        for s in stols:
            f = xray_form_factor("Fe", s)
            assert isinstance(f, float)
            fvals.append(f)
        for i in range(len(fvals) - 1):
            assert fvals[i] > fvals[i + 1], (stols[i], fvals[i], fvals[i + 1])

    def test_form_factor_raises_on_unknown_symbol(self) -> None:
        with pytest.raises(ScatteringDataError, match=r"Unknown element symbol"):
            xray_form_factor("Fm", 0.0)

    def test_form_factor_raises_on_ionic_label(self) -> None:
        with pytest.raises(
            ScatteringDataError, match=r"Ionic form factors not available"
        ):
            xray_form_factor("Fe3+", 0.0)


class TestFormFactorContract:
    """D-14: scalar-in/scalar-out and array-in/array-out shape contract."""

    def test_scalar_float_returns_python_float(self) -> None:
        result = xray_form_factor("Si", 0.5)
        # Must be a Python float, not a numpy scalar or 0-d ndarray.
        assert type(result) is float

    def test_scalar_int_returns_python_float(self) -> None:
        result = xray_form_factor("Si", 0)
        assert type(result) is float

    def test_numpy_scalar_returns_python_float(self) -> None:
        # np.asarray(np.float64(...)).ndim == 0, so scalar-in detection must
        # treat numpy scalars identically to Python floats.
        result = xray_form_factor("Si", np.float64(0.5))
        assert type(result) is float

    def test_array_stol_returns_contiguous_float64_ndarray(self) -> None:
        # N=5 deliberately differs from CM coefficient count (4) so a Pitfall 3
        # regression returning shape (4, N) would produce (4, 5) not (5,).
        stol_arr = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        result = xray_form_factor("Si", stol_arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        assert result.dtype == np.float64
        assert result.flags["C_CONTIGUOUS"]

    def test_list_stol_is_accepted(self) -> None:
        result = xray_form_factor("Si", [0.0, 0.3, 0.6])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_length_one_array_returns_length_one_ndarray(self) -> None:
        result = xray_form_factor("Si", np.array([0.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)


class TestNeutronReference:
    """SCAT-02: coherent scattering lengths match NIST reference values."""

    @pytest.mark.parametrize(
        "symbol, nist_fm",
        [
            ("H", -3.7390),
            ("C", 6.6460),
            ("O", 5.8030),
            ("Si", 4.1491),
            ("Fe", 9.4500),
        ],
    )
    def test_neutron_b_coh_matches_nist(self, symbol: str, nist_fm: float) -> None:
        result = neutron_scattering_length(symbol)
        assert abs(result - nist_fm) < 1e-4, (symbol, result, nist_fm)


class TestBundleCoverage:
    """SCAT-01: every bundled Z=1..95 symbol is reachable via the public API."""

    def test_all_95_symbols_lookupable(self) -> None:
        get_element("Si")  # force cache populate
        from diffraction import scattering as _scat

        assert len(_scat._BY_NUMBER) == 95
        for z in range(1, 96):
            elem = get_element(z)
            assert elem.z == z

    def test_every_element_has_complete_cromer_mann(self) -> None:
        for z in range(1, 96):
            elem = get_element(z)
            assert elem.cromer_mann_a.shape == (4,)
            assert elem.cromer_mann_b.shape == (4,)
            assert isinstance(elem.cromer_mann_c, float)

    def test_every_element_f_zero_is_finite(self) -> None:
        # Every bundled element must produce a finite f(stol=0) ≈ Z.
        for z in range(1, 96):
            elem = get_element(z)
            f0 = xray_form_factor(elem.symbol, 0.0)
            assert isinstance(f0, float)
            assert np.isfinite(f0)
            # Generous envelope — Cromer-Mann approximates Z at stol=0 within ~1.0
            # for the full bundled range; tighter per-element tolerances live in
            # TestFormFactorPhysics.
            assert abs(f0 - z) < 1.0, (elem.symbol, z, f0)

    def test_only_six_elements_missing_neutron(self) -> None:
        # D-03 amended 2026-04-21: exactly Po, At, Rn, Fr, Ac, Pu.
        missing: list[str] = []
        for z in range(1, 96):
            elem = get_element(z)
            if elem.neutron_b_coh is None:
                missing.append(elem.symbol)
        assert sorted(missing) == ["Ac", "At", "Fr", "Po", "Pu", "Rn"]
