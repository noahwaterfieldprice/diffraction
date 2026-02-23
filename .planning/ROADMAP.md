# Roadmap

**Project:** diffraction
**Milestone:** v0.2 — Single-crystal diffraction engine with fitting API
**Created:** 2026-02-22

---

## Phase Overview

| # | Phase | Goal | Requirements | Est. Complexity |
|---|-------|------|-------------|-----------------|
| 1 | 3/3 | Complete   | 2026-02-22 | Medium |
| 2 | 2/2 | Complete    | 2026-02-22 | Medium |
| 3 | ~~Code Quality~~ | ~~Absorbed into Phase 02.1.1~~ | — | — |
| 4 | Foundations | Space groups, coordinate transforms | 4.1–4.8 | High |
| 5 | Scattering Data | Bundle x-ray and neutron scattering tables | 5.1–5.8 | Medium |
| 6 | Diffraction Engine | Structure factor calculation, validation | 6.1–6.12 | High |
| 7 | Fitting API | Parameterized models, scipy integration | 7.1–7.6 | Medium |
| 8 | Publication Readiness | Documentation, packaging, CI | 8.1–8.4 | Low |

## Dependency Graph

```
Phase 1 (Style) ──→ Phase 2 (Tests) ──→ Phase 02.1 (Tooling)
                                              │
                                              ▼
                                        Phase 02.1.1 (Review/Quality)
                                              │
                                              ▼
                                        Phase 4 (Foundations)
                                          │         │
                                          ▼         │
                                    Phase 5 (Data)  │
                                          │         │
                                          ▼         ▼
                                    Phase 6 (Engine) ←──┘
                                          │
                                          ▼
                                    Phase 7 (Fitting)
                                          │
                                          ▼
                                    Phase 8 (Publication)
```

**Critical path:** 1 → 2 → 02.1 → 02.1.1 → 4 → 6 → 7 → 8
**Parallel opportunity:** Phase 5 (scattering data) can run in parallel with Phase 4 (foundations)
**Note:** Phase 3 (Code Quality) absorbed into Phase 02.1.1

---

## Phase 1: Style Guide Compliance

**Goal:** Align codebase with `~/Documents/dev/python_style_guide.md` before building new features.

**Rationale:** Changing file layout and import structure after adding new code is harder. Do it once, cleanly, first.

### Deliverables

| # | Deliverable | Files Affected | Acceptance Criteria |
|---|-------------|---------------|-------------------|
| 1 | Migrate to src layout | All source files | Package at `src/diffraction/`, `uv run pytest` passes |
| 2 | Consolidate config in pyproject.toml | `pyproject.toml`, `pytest.ini` | pytest.ini deleted, all config migrated |
| 3 | Configure and apply Ruff | `pyproject.toml`, all `.py` files | Ruff check passes with specified rules |
| 4 | Fix imports | `__init__.py`, all modules | Explicit imports, module imports for third-party |
| 5 | Modernize type hints | All `.py` files | `x \| None`, `list[str]`, no `typing` builtins |
| 6 | Standardize docstrings | All `.py` files | Google style, summary on opening line |
| 7 | Add exception chaining | All except blocks | `raise X from Y` throughout |

### Success Criteria
- `uv run ruff check .` passes
- `uv run ruff format --check .` passes
- `uv run pytest` passes with all existing tests
- Package importable as `from diffraction import Crystal`

### Risks
- src layout migration may break import paths in tests
- Ruff auto-fixes may change semantics (review each rule)

**Plans:** 3/3 plans complete

Plans:
- [x] 01-01-PLAN.md — Migrate to src layout with hatchling, consolidate pytest config, rename test files
- [x] 01-02-PLAN.md — Configure Ruff, apply auto-fixes, resolve manual violations, fix imports and exception chaining
- [x] 01-03-PLAN.md — Rewrite all docstrings to Google style

---

## Phase 2: Test Overhaul

**Goal:** Tests verify behaviour, not implementation details. Remove excessive mocking.

**Rationale:** Current tests are brittle — they test how code is called internally, not what it produces. This blocks confident refactoring in later phases.

### Deliverables

| # | Deliverable | Files Affected | Acceptance Criteria |
|---|-------------|---------------|-------------------|
| 1 | Rewrite Crystal tests | `tests/unit/crystal_test.py` | Real Crystal objects, property assertions |
| 2 | Rewrite Lattice from_cif/from_dict tests | `tests/unit/lattice_test.py` | Test with real data, verify attributes |
| 3 | Rewrite property and add_sites tests | `tests/unit/crystal_test.py`, `lattice_test.py` | Normal property access, real instances |
| 4 | Consolidate fixtures into conftest.py | `tests/conftest.py` | Shared CALCITE_DATA, CALCITE_CIF fixtures |
| 5 | Add missing coverage | Test files | Edge cases: validation, zero vectors, malformed CIF, invalid PointGroup |
| 6 | Add load_cif / validate_cif tests | `tests/unit/cif/cif_test.py` | Syntax errors, edge cases covered |

### Success Criteria
- Coverage ≥ 90% on all existing modules
- No `MagicMock` used as a fake domain object (Crystal, Lattice)
- No `.fget(mock)` pattern anywhere
- `pytest-mock` import count reduced by > 50%

**Plans:** 2/2 plans complete

Plans:
- [x] 02-01-PLAN.md — Create shared fixtures, rewrite lattice and symmetry tests with real instances
- [x] 02-02-PLAN.md — Rewrite crystal tests, add load_cif/validate_cif unit tests

---

### Phase 02.1: Tooling & Dependencies (INSERTED)

**Goal:** Add mypy to dev deps with strict config, update all dependencies to latest stable versions, fix any typing errors and breakage from either change.
**Depends on:** Phase 2
**Plans:** 3/3 plans complete

Plans:
- [x] 02.1-01-PLAN.md — Add mypy dev dep, create mypy.ini with strict config, bump Python to 3.11+, update CI matrix
- [x] 02.1-02-PLAN.md — Fix all mypy strict errors in source code (src/diffraction/)
- [x] 02.1-03-PLAN.md — Fix all mypy strict errors in test files (tests/), finalize mypy.ini

### Phase 02.1.1: Address REVIEW.md (INSERTED — absorbs Phase 3)

**Goal:** Address all REVIEW.md findings (M1–M5, m1–m15, R1–R5) and absorb Phase 3 deliverables. Comprehensive code quality pass covering vector redesign (composition over numpy subclassing), exception hierarchy, Lattice ABC cleanup, dataclass conversions, @overload additions, and project metadata.
**Depends on:** Phase 02.1
**Plans:** 4 plans in 3 waves

Context: `.planning/phases/02.1.1-address-review-md/02.1.1-CONTEXT.md`

Plans:
- [ ] 02.1.1-01-PLAN.md — Vector redesign: composition over numpy subclassing (M1/R1, m7/R5, m10, m11)
- [ ] 02.1.1-02-PLAN.md — Exceptions + PointGroup errors + frozen dataclass + TypeAlias (m1/R3, M3/R4, m9, m12, m15)
- [ ] 02.1.1-03-PLAN.md — Lattice ABC cleanup + Crystal API + Site dataclass + naming + validation (M2/R2, m2, m3, m4, m8, m13)
- [ ] 02.1.1-04-PLAN.md — Project metadata + test constant dedup + m6 closure (M4, m6, m14)

---

## Phase 4: Foundations

**Goal:** Space group implementation and coordinate transforms — prerequisites for diffraction calculation.

**Rationale:** Structure factor calculation requires expanding the asymmetric unit using space group operations and converting between coordinate systems.

### Deliverables

| # | Deliverable | Files Affected | Acceptance Criteria |
|---|-------------|---------------|-------------------|
| 1 | Data preparation script | `scripts/prepare_space_groups.py` | Merges 3 V0 data sources → 230 JSON files |
| 2 | Space group JSON data | `src/diffraction/static/space_groups/*.json` | 230 files with rotation matrices, translations, xyz strings, metadata |
| 3 | SpaceGroup class | `src/diffraction/symmetry.py` | Load by number or HM symbol, expose operations, point group, crystal system |
| 4 | HM symbol lookup table | `src/diffraction/symmetry.py` | All common variants map to correct space group |
| 5 | Equivalent position generation | `src/diffraction/symmetry.py` or `crystal.py` | Apply operations + centering, deduplicate with tolerance |
| 6 | Systematic absence check | `src/diffraction/symmetry.py` | Algorithmic h·W=h method, verified against known conditions |
| 7 | Coordinate transforms | `src/diffraction/lattice.py` | Orthogonalization matrix, fractional ↔ Cartesian conversion |
| 8 | Space group validation | `src/diffraction/symmetry.py` | Invalid HM symbols raise ValueError with suggestions |

### Success Criteria
- All 230 standard space groups loadable
- Equivalent positions match cctbx output for test structures (Si, NaCl, calcite, corundum)
- Systematic absences correct for all centering types (P, A, B, C, I, F, R)
- Coordinate transforms verified against known distances

### Risks
- **Critical:** Float symmetry operations accumulate errors. Mitigate: use integer rotation matrices + rational translations (multiples of 1/12)
- V0 data quality issues in OLDER JSONs (point_group field stores index not HM symbol)
- Origin choice ambiguity for some space groups (use choice 2 / modern ITC standard)

### Research Needed
- Inspect V0 `space_group_generators.txt` format for exact parsing approach
- Verify V0 data against `space_groups.txt` for cross-validation

---

## Phase 5: Scattering Data

**Goal:** Bundle x-ray and neutron scattering factor tables as static package data.

**Rationale:** Structure factor calculation needs scattering amplitudes. Bundle as static data for offline use, no external dependencies.

**Parallel with:** Phase 4 (no dependency between scattering data and space groups)

### Deliverables

| # | Deliverable | Files Affected | Acceptance Criteria |
|---|-------------|---------------|-------------------|
| 1 | Data conversion script | `scripts/prepare_scattering_data.py` | Parses V0 `Dans Element Properties.txt` → JSON |
| 2 | X-ray form factor JSON | `src/diffraction/static/elements/xray_form_factors.json` | Cromer-Mann coefficients, array-based schema, neutral atoms + common ions |
| 3 | Neutron scattering JSON | `src/diffraction/static/elements/neutron_scattering.json` | Coherent scattering lengths for all practical elements |
| 4 | Element properties JSON | `src/diffraction/static/elements/element_properties.json` | Z, symbol, name, weight |
| 5 | ScatteringFactor API | `src/diffraction/elements.py` | Lookup by symbol, evaluate f0(Q), lazy loading with numpy cache |
| 6 | Validation tests | `tests/unit/elements_test.py` | f0(0) = Z for neutrals, known values match, neutron lengths correct |

### Success Criteria
- All elements Z=1–95 have x-ray form factors
- Common ions (Fe2+, Fe3+, O2-, Na+, etc.) included
- Neutron scattering lengths match NIST database
- Lazy loading: data loaded on first access, cached thereafter

---

## Phase 6: Diffraction Engine

**Goal:** Calculate structure factors for x-ray and neutron radiation, validate against known reference structures.

**Depends on:** Phase 4 (space groups, coordinate transforms), Phase 5 (scattering data)

### Deliverables

| # | Deliverable | Files Affected | Acceptance Criteria |
|---|-------------|---------------|-------------------|
| 1 | Structure factor function | `src/diffraction/scattering.py` | Vectorized F(hkl) for x-ray and neutron |
| 2 | Debye-Waller factor | `src/diffraction/scattering.py` | Isotropic B_iso and anisotropic U_ij → β_ij |
| 3 | Reflection list generation | `src/diffraction/scattering.py` | Enumerate hkl within limits, filter absences |
| 4 | Intensity calculation | `src/diffraction/scattering.py` | I = \|F\|², Lorentz-polarization correction |
| 5 | Q-vector / d-spacing / 2θ | `src/diffraction/lattice.py` or `scattering.py` | From reciprocal metric tensor |
| 6 | Silicon validation | `tests/functional/` | F(hkl) matches published values |
| 7 | NaCl validation | `tests/functional/` | F(hkl) matches, systematic absences correct |
| 8 | Corundum validation | `tests/functional/` | Non-cubic, rhombohedral symmetry correct |

### Success Criteria
- Structure factors match V0 `calXint`/`calNint` output for same structures
- Friedel pair test: F(h) = F*(-h) for all non-anomalous calculations
- All systematically absent reflections have F = 0
- Function signatures are FFI-ready: numpy arrays in/out, contiguous, stateless

### Risks
- Form factor normalization: s = |Q|/(4π), not Q directly (critical pitfall)
- Phase convention: use exp(+2πi h·r), matching ITC Vol B and V0 code
- Displacement parameter convention mismatches (B_iso vs U_iso vs β_ij)

---

## Phase 7: Fitting API

**Goal:** Parameterized intensity calculation compatible with scipy.optimize for structural refinement.

**Depends on:** Phase 6 (working structure factor calculation)

### Deliverables

| # | Deliverable | Files Affected | Acceptance Criteria |
|---|-------------|---------------|-------------------|
| 1 | ParameterSet class | `src/diffraction/fitting.py` | Named params, bounds, vary/fixed, pack/unpack arrays |
| 2 | Crystal model update | `src/diffraction/crystal.py` | Accept variable structural parameters from ParameterSet |
| 3 | Residual function | `src/diffraction/fitting.py` | Weighted residuals, returns array for least_squares |
| 4 | scipy integration | `src/diffraction/fitting.py` | Works with `scipy.optimize.least_squares()` |
| 5 | R-factor calculation | `src/diffraction/fitting.py` | R1, wR2 statistics |
| 6 | Optional lmfit adapter | `src/diffraction/fitting.py` | Wraps lmfit.Parameters when lmfit is available |

### Success Criteria
- Refine lattice parameters of Si from noisy synthetic data, converge to known values
- Refine atomic positions and B-factors for NaCl
- R-factor decreases monotonically during refinement
- Works with both `scipy.optimize.least_squares` and `lmfit.minimize`

---

## Phase 8: Publication Readiness

**Goal:** Library is documented, packaged, and ready for PyPI release.

**Depends on:** Phase 7 (complete feature set to document)

### Deliverables

| # | Deliverable | Files Affected | Acceptance Criteria |
|---|-------------|---------------|-------------------|
| 1 | Sphinx documentation | `docs/` | API reference auto-generated, usage examples |
| 2 | Usage examples | `docs/examples/` | Load CIF, calculate F(hkl), run refinement |
| 3 | PyPI packaging | `pyproject.toml` | `uv build` produces installable wheel |
| 4 | README update | `README.md` | Quick-start, features, installation |
| 5 | CI/CD verification | `.github/workflows/` | Lint + test + build on push |

### Success Criteria
- `cd docs && make html` builds without warnings
- Package installs in clean venv: `pip install dist/diffraction-*.whl`
- All CI checks pass on clean clone

---

## Success Criteria (Milestone)

The milestone is complete when:

1. **Correct calculations:** Structure factors for Si, NaCl, and corundum match published reference values for both x-ray and neutron radiation
2. **Fitting works:** Can refine structural parameters from synthetic data using scipy.optimize
3. **Clean codebase:** Ruff passes, tests are behavioural, coverage ≥ 90%
4. **Documented:** Sphinx docs build, usage examples work
5. **Installable:** PyPI-ready wheel builds and installs

---

*Last updated: 2026-02-23 (Phase 3 absorbed into 02.1.1; CONTEXT.md written)*
