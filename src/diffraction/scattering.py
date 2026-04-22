"""X-ray and neutron scattering data for neutral atoms Z=1-95."""

import json
import re
import threading
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt

from .exceptions import ScatteringDataError

__all__ = [
    "Element",
    "get_element",
    "get_neutral_symbol",
    "neutron_scattering_length",
    "xray_form_factor",
]


# ---------------------------------------------------------------------------
# Module-level cache (populated lazily on first use)
# ---------------------------------------------------------------------------

_BY_SYMBOL: dict[str, "Element"] = {}
_BY_NUMBER: dict[int, "Element"] = {}
_element_data_loaded: bool = False
_element_data_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Ion-label grammar (accepts both ITC 'Fe3+' and CIF 'Na1+' forms)
# ---------------------------------------------------------------------------

# Matches any symbol with optional digits and optional charge sign.
_ION_LABEL_RE = re.compile(r"^([A-Za-z]{1,2})\d*[+-]?$")

# Matches labels that CARRY an explicit charge sign — used as the ion-label
# guard in get_element to produce the D-18 error BEFORE cache-miss fires.
_ION_CHARGE_RE = re.compile(r"^([A-Za-z]{1,2})\d*[+-]$")


def _load_element_data() -> None:
    """Populate ``_BY_SYMBOL`` and ``_BY_NUMBER`` from ``elements.json``."""
    global _element_data_loaded
    data_file = resources.files("diffraction") / "static" / "elements.json"
    raw: dict[str, Any] = json.loads(data_file.read_text())
    for sym, entry in raw.items():
        elem = Element.__new__(Element)
        object.__setattr__(elem, "z", int(entry["z"]))
        object.__setattr__(elem, "symbol", str(entry["symbol"]))
        object.__setattr__(elem, "name", str(entry["name"]))
        a = np.ascontiguousarray(entry["cromer_mann"]["a"], dtype=np.float64)
        b = np.ascontiguousarray(entry["cromer_mann"]["b"], dtype=np.float64)
        # Mirror the immutable-cached-array pattern in lattice.py: the Element
        # dataclass is frozen, but the numpy arrays it holds would otherwise
        # allow in-place mutation of the module-level cache (`elem.cromer_mann_a[0]
        # = x` would corrupt every subsequent lookup). Flip writeable=False so
        # the frozen contract extends to the array payload too.
        a.flags.writeable = False
        b.flags.writeable = False
        object.__setattr__(elem, "cromer_mann_a", a)
        object.__setattr__(elem, "cromer_mann_b", b)
        object.__setattr__(elem, "cromer_mann_c", float(entry["cromer_mann"]["c"]))
        # JSON null arrives as Python None; never use math.isnan on this field.
        b_coh = entry["neutron_b_coh"]
        object.__setattr__(
            elem, "neutron_b_coh", None if b_coh is None else float(b_coh)
        )
        _BY_SYMBOL[sym] = elem
        _BY_NUMBER[elem.z] = elem
    _element_data_loaded = True


def _ensure_element_data_loaded() -> None:
    """Lazily load element data on first access (thread-safe)."""
    if _element_data_loaded:
        return
    with _element_data_lock:
        if not _element_data_loaded:
            _load_element_data()


# ---------------------------------------------------------------------------
# Element frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class Element:
    """A neutral atom's x-ray and neutron scattering data.

    Construct via :func:`get_element` — do not instantiate directly.

    Attributes:
        z: Atomic number (1-95).
        symbol: Element symbol, e.g. ``'Fe'``.
        name: Element name, e.g. ``'Iron'``.
        cromer_mann_a: 4-element C-contiguous float64 ndarray of Cromer-Mann
            a coefficients.
        cromer_mann_b: 4-element C-contiguous float64 ndarray of Cromer-Mann
            b coefficients.
        cromer_mann_c: Cromer-Mann c offset.
        neutron_b_coh: Coherent neutron scattering length in femtometres, or
            ``None`` if not tabulated (Po, At, Rn, Fr, Ac, Pu).
    """

    z: int
    symbol: str
    name: str
    cromer_mann_a: np.ndarray = field(repr=False)
    cromer_mann_b: np.ndarray = field(repr=False)
    cromer_mann_c: float
    neutron_b_coh: float | None


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def get_element(key: str | int) -> Element:
    """Look up an :class:`Element` by symbol or atomic number.

    Args:
        key: Element symbol (``'Fe'``) or atomic number (``26``).

    Returns:
        Cached :class:`Element` for the given key.

    Raises:
        ScatteringDataError: Unknown symbol, out-of-range atomic number, or
            ionic label (v1.0 does not ship ion-specific coefficients).
        TypeError: If ``key`` is neither ``str`` nor ``int`` (``bool`` is
            rejected even though Python treats it as ``int``).
    """
    _ensure_element_data_loaded()
    if isinstance(key, bool) or not isinstance(key, (str, int)):
        raise TypeError(f"key must be str or int, not {type(key).__name__}")
    if isinstance(key, str):
        # Normalize like get_neutral_symbol so CIF-style labels with stray
        # whitespace or non-canonical casing resolve to the cached entry.
        key = key.strip().capitalize()
        # Ion-label guard MUST run before cache lookup — see RESEARCH.md
        # anti-pattern: get_element('Fe3+') must yield the ionic-label error,
        # NOT the unknown-symbol error.
        if _ION_CHARGE_RE.match(key):
            m = _ION_LABEL_RE.match(key)
            assert m is not None  # _ION_CHARGE_RE matched, so this matches too
            neutral = m.group(1).capitalize()
            raise ScatteringDataError(
                f"Ionic form factors not available in v1.0; "
                f"use neutral symbol {neutral!r} or call "
                f"get_neutral_symbol() first"
            )
        if key not in _BY_SYMBOL:
            raise ScatteringDataError(f"Unknown element symbol {key!r}")
        return _BY_SYMBOL[key]
    # isinstance(key, int) and not bool
    if key not in _BY_NUMBER:
        raise ScatteringDataError(f"Unknown atomic number {key}")
    return _BY_NUMBER[key]


def get_neutral_symbol(label: str) -> str:
    """Return the neutral element symbol from an ion label.

    Accepts both ITC short form (``'Fe3+'``, ``'O2-'``) and the CIF
    ``_atom_site_type_symbol`` grammar with a digit between the symbol
    and the sign (``'Na1+'``, ``'Cl1-'``, verified in
    ``examples/nacl.cif``). Returns the bare symbol unchanged when no
    charge is present.

    Args:
        label: Ion label or bare element symbol.

    Returns:
        Neutral element symbol with canonical capitalisation
        (``'Fe'``, not ``'fe'``).

    Raises:
        ScatteringDataError: Label cannot be parsed.
    """
    m = _ION_LABEL_RE.match(label.strip())
    if m is None:
        raise ScatteringDataError(f"Cannot parse element symbol from label {label!r}")
    return m.group(1).capitalize()


# ---------------------------------------------------------------------------
# Physics functions
# ---------------------------------------------------------------------------


def xray_form_factor(symbol: str, stol: float | npt.ArrayLike) -> float | np.ndarray:
    """X-ray atomic form factor f(stol) using Cromer-Mann parametrization.

    The formula (ITC Vol C Table 6.1.1.4) is::

        f(stol) = sum(a_i * exp(-b_i * stol**2)) + c

    where ``stol = sin(theta) / lambda`` in inverse Angstroms.

    Args:
        symbol: Element symbol, e.g. ``'Fe'``. Ionic labels raise
            :class:`ScatteringDataError`; call :func:`get_neutral_symbol`
            first if a CIF label carries a charge.
        stol: ``sin(theta) / lambda`` in inverse Angstroms. Scalar or
            array-like.

    Returns:
        ``float`` when ``stol`` is scalar; a C-contiguous float64
        ``np.ndarray`` of the same shape as ``stol`` when array-like.

    Raises:
        ScatteringDataError: Unknown or ionic ``symbol``.
    """
    elem = get_element(symbol)
    s = np.asarray(stol, dtype=np.float64)
    scalar_in = s.ndim == 0
    s = np.atleast_1d(s)
    a = elem.cromer_mann_a  # (4,)
    b = elem.cromer_mann_b  # (4,)
    c = elem.cromer_mann_c
    # (4,1) * exp(-(4,1) * (1,N)^2) -> (4,N); sum axis=0 -> (N,)
    result = np.sum(a[:, None] * np.exp(-b[:, None] * s[None, :] ** 2), axis=0) + c
    result = np.ascontiguousarray(result)
    return float(result[0]) if scalar_in else result


def neutron_scattering_length(symbol: str) -> float:
    """Neutron coherent scattering length b_coh in femtometres.

    Args:
        symbol: Element symbol. Ionic labels raise
            :class:`ScatteringDataError`.

    Returns:
        Coherent scattering length in femtometres.

    Raises:
        ScatteringDataError: Element has no tabulated b_coh (Po, At, Rn,
            Fr, Ac, Pu), symbol is unknown, or symbol carries a charge.
    """
    elem = get_element(symbol)
    if elem.neutron_b_coh is None:
        raise ScatteringDataError(
            f"Neutron scattering length not tabulated for {elem.symbol} (Z={elem.z})"
        )
    return elem.neutron_b_coh
