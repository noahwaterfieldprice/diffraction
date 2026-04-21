"""Direct and reciprocal lattices and lattice vectors.

Provide objects for direct and reciprocal space lattice calculations.
Naming conventions follow the International Tables for Crystallography.

Classes:
    DirectLattice: Direct lattice with parameters, metric tensor, and volume.
    ReciprocalLattice: Reciprocal lattice derived from a direct lattice.
    DirectLatticeVector: Vector in direct space attached to a DirectLattice.
    ReciprocalLatticeVector: Vector in reciprocal space attached to a
        ReciprocalLattice.

Examples:
    Create a direct lattice from lattice parameters:

    >>> from diffraction import DirectLattice
    >>> calcite_lattice = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
    >>> calcite_lattice.c
    17.002

    Construct the corresponding reciprocal lattice:

    >>> calcite_reciprocal_lattice = calcite_lattice.reciprocal()
    >>> calcite_reciprocal_lattice.b_star
    1.4539473861596934

    Perform lattice vector calculations in direct space:

    >>> u = DirectLatticeVector([1, 0, 0], lattice=calcite_lattice)
    >>> u.norm()
    4.99
    >>> v = calcite_lattice.vector([0, 0, 1])
    >>> u.inner(v)
    0.0
"""

from __future__ import annotations

import abc
import functools
import math
from collections.abc import Sequence
from numbers import Real
from typing import ClassVar, TypeAlias, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from .cif import helpers as cif_helpers

LatticeParameters: TypeAlias = Sequence[float]

_LT = TypeVar("_LT", bound="Lattice")

__all__ = [
    "DirectLattice",
    "DirectLatticeVector",
    "ReciprocalLattice",
    "ReciprocalLatticeVector",
]

# Named constants replacing magic literals in metric/reciprocal calculations.
_ROUNDING_PRECISION: int = 10
# Relative tolerance for verifying reciprocal lattice relationship. Generous
# to accommodate floating-point accumulation through matrix inversion.
_RECIPROCAL_LATTICE_RTOL: float = 1e-2


def _to_radians(lattice_parameters: LatticeParameters) -> tuple[float, ...]:
    """Convert angles in lattice parameters from degrees to radians.

    Args:
        lattice_parameters: Six lattice parameters (a, b, c, alpha, beta,
            gamma) with angles in degrees.

    Returns:
        Lattice parameters with the three angle values converted to radians.
    """
    lengths = tuple(lattice_parameters)[:3]
    angles_in_radians = tuple(
        math.radians(angle) for angle in tuple(lattice_parameters)[3:]
    )
    return lengths + angles_in_radians


def _to_degrees(lattice_parameters: tuple[float, ...]) -> tuple[float, ...]:
    """Convert angles in lattice parameters from radians to degrees.

    Args:
        lattice_parameters: Six lattice parameters (a, b, c, alpha, beta,
            gamma) with angles in radians.

    Returns:
        Lattice parameters with the three angle values converted to degrees.
    """
    lengths = tuple(lattice_parameters)[:3]
    angles_in_degrees = tuple(
        math.degrees(angle) for angle in tuple(lattice_parameters)[3:]
    )
    return lengths + angles_in_degrees


def _metric_tensor(lattice_parameters: LatticeParameters) -> NDArray[np.float64]:
    """Calculate the metric tensor for a lattice.

    Args:
        lattice_parameters: Six lattice parameters (a, b, c, alpha, beta,
            gamma) with angles in degrees.

    Returns:
        The 3x3 metric tensor as a numpy array.

    Notes:
        See International Tables for Crystallography, Vol. B, Section 1.1.
    """
    a, b, c, al, be, ga = _to_radians(lattice_parameters)
    tensor: NDArray[np.float64] = np.around(
        [
            [a**2, a * b * np.cos(ga), a * c * math.cos(be)],
            [a * b * math.cos(ga), b**2, b * c * math.cos(al)],
            [a * c * math.cos(be), b * c * np.cos(al), c**2],
        ],
        _ROUNDING_PRECISION,
    )
    return tensor


def _orthogonalization_matrix(
    lattice_parameters: LatticeParameters,
) -> NDArray[np.float64]:
    """Compute the orthogonalization matrix M (ITC Vol B section 1.1.5.2).

    Convention: a parallel to x, b in the xy-plane.
    x_cartesian = M @ x_fractional.

    Args:
        lattice_parameters: Six lattice parameters (a, b, c, alpha, beta,
            gamma) with angles in degrees.

    Returns:
        3x3 orthogonalization matrix.
    """
    a, b, c, al, be, ga = _to_radians(lattice_parameters)
    cos_al, cos_be, cos_ga = math.cos(al), math.cos(be), math.cos(ga)
    sin_ga = math.sin(ga)
    # V/(abc) = sqrt(1 - cos^2(al) - cos^2(be) - cos^2(ga) + 2*cos(al)*cos(be)*cos(ga))
    vol_factor = math.sqrt(
        1 - cos_al**2 - cos_be**2 - cos_ga**2 + 2 * cos_al * cos_be * cos_ga
    )
    # Upper-triangular orthogonalization matrix (ITC Vol B eq. 1.1.5.2)
    matrix: NDArray[np.float64] = np.array(
        [
            [a, b * cos_ga, c * cos_be],
            [0.0, b * sin_ga, c * (cos_al - cos_be * cos_ga) / sin_ga],
            [0.0, 0.0, c * vol_factor / sin_ga],
        ]
    )
    return matrix


def _reciprocalise(lattice_parameters: LatticeParameters) -> tuple[float, ...]:
    """Transform lattice parameters to those of the reciprocally related lattice.

    Convert direct lattice parameters to reciprocal lattice parameters, or
    vice versa. The transformation is its own inverse.

    Args:
        lattice_parameters: Six lattice parameters (a, b, c, alpha, beta,
            gamma) with angles in degrees.

    Returns:
        Six lattice parameters of the reciprocally related lattice, with
        angles in degrees.

    Notes:
        See International Tables for Crystallography, Vol. B, Section 1.1.
        Reciprocal lengths are defined with the 2*pi convention, matching
        the physics convention used for wavevectors.
    """
    a, b, c, al, be, ga = _to_radians(lattice_parameters)
    cell_volume = float(np.sqrt(np.linalg.det(_metric_tensor(lattice_parameters))))
    pi, sin, cos, arccos = math.pi, math.sin, math.cos, math.acos

    a_ = 2 * pi * b * c * sin(al) / cell_volume
    b_ = 2 * pi * a * c * sin(be) / cell_volume
    c_ = 2 * pi * a * b * sin(ga) / cell_volume
    alpha_ = arccos((cos(be) * cos(ga) - cos(al)) / (sin(be) * sin(ga)))
    beta_ = arccos((cos(al) * cos(ga) - cos(be)) / (sin(al) * sin(ga)))
    gamma_ = arccos((cos(al) * cos(be) - cos(ga)) / (sin(al) * sin(be)))

    return _to_degrees((a_, b_, c_, alpha_, beta_, gamma_))


class Lattice(abc.ABC):
    """Abstract base class for direct and reciprocal lattice objects.

    Store lattice parameters as instance attributes and compute derived
    properties (metric tensor, unit cell volume) on demand. Concrete
    subclasses must define ``lattice_parameter_keys``.

    Args:
        lattice_parameters: Six lattice parameters in the order
            (a, b, c, alpha, beta, gamma) with angles in degrees.

    Attributes:
        lattice_parameter_keys: Class-level tuple of parameter name strings
            used as attribute names and dict keys.
    """

    lattice_parameter_keys: ClassVar[tuple[str, ...]]

    def __init__(self, lattice_parameters: LatticeParameters) -> None:
        lattice_parameters = self.check_lattice_parameters(lattice_parameters)
        for key, value in zip(
            self.lattice_parameter_keys, lattice_parameters, strict=True
        ):
            setattr(self, key, value)

    def check_lattice_parameters(
        self, lattice_parameters: LatticeParameters
    ) -> list[float]:
        """Validate and coerce lattice parameters to floats.

        Args:
            lattice_parameters: Six lattice parameters (a, b, c, alpha,
                beta, gamma) with angles in degrees.

        Returns:
            List of float-coerced lattice parameter values.

        Raises:
            ValueError: If fewer than six parameters are provided, if any
                parameter cannot be converted to float, if any length is
                non-positive, or if any angle is outside (0, 180) degrees.
        """
        if len(lattice_parameters) < 6:
            raise ValueError(
                f"Expected at least 6 lattice parameters, got {len(lattice_parameters)}"
            )
        lattice_parameters_: list[float] = []
        for key, value in zip(
            self.lattice_parameter_keys, lattice_parameters, strict=False
        ):
            try:
                lattice_parameters_.append(float(value))
            except ValueError as exc:
                raise ValueError(f"Invalid lattice parameter {key}: {value}") from exc
        for key, value in zip(
            self.lattice_parameter_keys[:3], lattice_parameters_[:3], strict=True
        ):
            if value <= 0:
                raise ValueError(f"Lattice length {key} must be positive, got {value}")
        for key, value in zip(
            self.lattice_parameter_keys[3:], lattice_parameters_[3:], strict=True
        ):
            if not (0 < value < 180):
                raise ValueError(
                    f"Lattice angle {key} must be in (0, 180) degrees, got {value}"
                )
        return lattice_parameters_

    @classmethod
    def from_dict(cls: type[_LT], input_dict: dict[str, float]) -> _LT:
        """Create a lattice from a parameter dictionary.

        Args:
            input_dict: Mapping from parameter name strings to values. Must
                contain all keys in ``lattice_parameter_keys``.

        Returns:
            A new lattice instance populated from the dictionary.

        Raises:
            ValueError: If any required lattice parameter key is missing from
                the dictionary. All missing keys are reported at once.
        """
        missing = [k for k in cls.lattice_parameter_keys if k not in input_dict]
        if missing:
            raise ValueError(
                f"Parameters missing from input dictionary: "
                f"{', '.join(repr(k) for k in missing)}"
            )
        lattice_parameters = [float(input_dict[k]) for k in cls.lattice_parameter_keys]
        return cls(lattice_parameters)

    @property
    def lattice_parameters(self) -> tuple[float, ...]:
        """Return all lattice parameters as a tuple."""
        return tuple(getattr(self, name) for name in self.lattice_parameter_keys)

    @property
    def metric(self) -> NDArray[np.float64]:
        """Return the 3x3 metric tensor for this lattice."""
        return _metric_tensor(self.lattice_parameters)

    @property
    def unit_cell_volume(self) -> float:
        """Return the unit cell volume computed from the metric tensor."""
        return float(np.sqrt(np.linalg.det(self.metric)))

    def __repr__(self) -> str:
        repr_string = "{0}([{1!r}, {2!r}, {3!r}, {4!r}, {5!r}, {6!r}])"
        rounded_lattice_parameters = [
            round(parameter, 4) for parameter in self.lattice_parameters
        ]
        return repr_string.format(self.__class__.__name__, *rounded_lattice_parameters)

    def __str__(self) -> str:
        return repr(self)


class DirectLattice(Lattice):
    """Direct lattice defined by six lattice parameters.

    Store and expose lattice parameters (a, b, c, alpha, beta, gamma) and
    compute the metric tensor and unit cell volume. Can be constructed from
    a parameter list, a dictionary, or a CIF file.

    Args:
        lattice_parameters: Six lattice parameters in the order
            (a, b, c, alpha, beta, gamma) with angles in degrees.

    Attributes:
        a: Length of the a-axis in angstroms.
        b: Length of the b-axis in angstroms.
        c: Length of the c-axis in angstroms.
        alpha: Angle between b and c axes in degrees.
        beta: Angle between a and c axes in degrees.
        gamma: Angle between a and b axes in degrees.
        lattice_parameters: All six parameters as a tuple (a, b, c, alpha,
            beta, gamma).
        metric: The 3x3 metric tensor of the direct basis.
        unit_cell_volume: Volume of the unit cell in angstroms cubed.

    Examples:
        Create a calcite direct lattice and inspect its properties:

        >>> from diffraction import DirectLattice
        >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
        >>> calcite.b
        4.99
        >>> calcite.gamma
        120.0
        >>> calcite.lattice_parameters
        (4.99, 4.99, 17.002, 90.0, 90.0, 120.0)
        >>> calcite.unit_cell_volume
        366.63315390345286
        >>> calcite.metric
        array([[ 24.9001  , -12.45005 ,   0.      ],
               [-12.45005 ,  24.9001  ,   0.      ],
               [  0.      ,   0.      , 289.068004]])
    """

    lattice_parameter_keys = ("a", "b", "c", "alpha", "beta", "gamma")

    # Declared explicitly so mypy can resolve these dynamically-set attributes.
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @classmethod
    def from_cif(cls, filepath: str, data_block: str | None = None) -> DirectLattice:
        """Create a DirectLattice from a CIF file.

        Args:
            filepath: Path to the input CIF file.
            data_block: Data block header to read from, required only when
                the CIF contains multiple data blocks.

        Returns:
            A DirectLattice populated with parameters from the CIF.

        Raises:
            ValueError: If any required lattice parameter is missing or is
                not valid numerical data.
            TypeError: If the CIF has multiple data blocks but data_block
                is not given.
        """
        data_items = cif_helpers.load_data_block(filepath, data_block)
        data_names = [cif_helpers.CIF_NAMES[key] for key in cls.lattice_parameter_keys]
        lattice_parameters = cif_helpers.get_numerical_cif_data(data_items, *data_names)
        return cls(lattice_parameters)

    def vector(self, uvw: Sequence[float]) -> DirectLatticeVector:
        """Return a direct lattice vector defined on this lattice.

        Args:
            uvw: Miller indices (u, v, w) of the direct lattice vector.

        Returns:
            A DirectLatticeVector with the given indices attached to this
            lattice.
        """
        return DirectLatticeVector(uvw, self)

    def reciprocal(self) -> ReciprocalLattice:
        """Return the corresponding reciprocal lattice.

        Returns:
            A ReciprocalLattice with parameters derived from this direct
            lattice using the 2*pi convention.
        """
        reciprocal_lattice_parameters = _reciprocalise(self.lattice_parameters)
        return ReciprocalLattice(reciprocal_lattice_parameters)

    @functools.cached_property
    def _ortho_matrix(self) -> NDArray[np.float64]:
        """Cached orthogonalization matrix (ITC Vol B section 1.1.5.2).

        Computed once on first access. Safe to cache because lattice
        parameters are set at construction and are plain floats with no
        setters that would invalidate the cache.

        Returns:
            3x3 orthogonalization matrix.
        """
        return _orthogonalization_matrix(self.lattice_parameters)

    @functools.cached_property
    def _inv_ortho_matrix(self) -> NDArray[np.float64]:
        """Cached inverse orthogonalization matrix.

        Computed once on first access. Used by cartesian_to_fractional
        to avoid recomputing np.linalg.inv() on every call.

        Returns:
            3x3 inverse orthogonalization matrix.
        """
        return np.asarray(np.linalg.inv(self._ortho_matrix), dtype=np.float64)

    def fractional_to_cartesian(
        self,
        r: Sequence[float] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert fractional coordinates to Cartesian.

        Uses the ITC Vol B section 1.1.5.2 orthogonalization matrix with
        a parallel to x and b in the xy-plane.

        Args:
            r: Fractional coordinates (x, y, z).

        Returns:
            Cartesian coordinates as a 1-D numpy array.

        Raises:
            ValueError: If r does not have shape (3,).
        """
        coords = np.asarray(r, dtype=np.float64)
        if coords.shape != (3,):
            raise ValueError(
                f"Coordinates must be a 1-D array of length 3, got shape {coords.shape}"
            )
        return self._ortho_matrix @ coords

    def cartesian_to_fractional(
        self,
        r: Sequence[float] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert Cartesian coordinates to fractional.

        Args:
            r: Cartesian coordinates (x, y, z).

        Returns:
            Fractional coordinates as a 1-D numpy array.

        Raises:
            ValueError: If r does not have shape (3,).
        """
        coords = np.asarray(r, dtype=np.float64)
        if coords.shape != (3,):
            raise ValueError(
                f"Coordinates must be a 1-D array of length 3, got shape {coords.shape}"
            )
        result = self._inv_ortho_matrix @ coords
        return np.asarray(result, dtype=np.float64)


class ReciprocalLattice(Lattice):
    """Reciprocal lattice defined by six reciprocal lattice parameters.

    Store and expose reciprocal lattice parameters (a_star, b_star, c_star,
    alpha_star, beta_star, gamma_star) using the 2*pi convention. Can be
    constructed from a parameter list, a dictionary, or via
    ``DirectLattice.from_cif(...).reciprocal()``.

    Attributes:
        a_star: Length of the a* axis.
        b_star: Length of the b* axis.
        c_star: Length of the c* axis.
        alpha_star: Angle between b* and c* axes in degrees.
        beta_star: Angle between a* and c* axes in degrees.
        gamma_star: Angle between a* and b* axes in degrees.
        lattice_parameters: All six reciprocal parameters as a tuple.
        metric: The 3x3 metric tensor of the reciprocal basis.
        unit_cell_volume: Volume of the reciprocal unit cell.

    Examples:
        Create a reciprocal lattice from a direct lattice:

        >>> from diffraction import DirectLattice
        >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
        >>> rl = calcite.reciprocal()
        >>> rl.b_star
        1.4539473861596934
        >>> rl.gamma_star
        60.00000000000002
    """

    lattice_parameter_keys = (
        "a_star",
        "b_star",
        "c_star",
        "alpha_star",
        "beta_star",
        "gamma_star",
    )

    # Declared explicitly so mypy can resolve these dynamically-set attributes.
    a_star: float
    b_star: float
    c_star: float
    alpha_star: float
    beta_star: float
    gamma_star: float

    def vector(self, hkl: Sequence[float]) -> ReciprocalLatticeVector:
        """Return a reciprocal lattice vector defined on this lattice.

        Args:
            hkl: Miller indices (h, k, l) of the reciprocal lattice vector.

        Returns:
            A ReciprocalLatticeVector with the given indices attached to
            this lattice.
        """
        return ReciprocalLatticeVector(hkl, self)

    def direct(self) -> DirectLattice:
        """Return the corresponding direct lattice.

        Returns:
            A DirectLattice with parameters derived from this reciprocal
            lattice by applying the inverse reciprocalise transform.
        """
        direct_lattice_parameters = _reciprocalise(self.lattice_parameters)
        return DirectLattice(direct_lattice_parameters)


class LatticeVector:
    """Composition-based base class for lattice vectors.

    Store a 3-component vector alongside a reference lattice. Subclasses
    specialise for direct and reciprocal space. Direct construction should
    use the concrete subclasses ``DirectLatticeVector`` and
    ``ReciprocalLatticeVector`` rather than this base class.

    Attributes:
        components: Read-only view of the underlying ndarray.
        lattice: The lattice this vector is defined on.
    """

    __slots__ = ("_components", "_lattice")

    def __init__(
        self,
        components: Sequence[float] | NDArray[np.float64],
        lattice: Lattice,
    ) -> None:
        self._components: NDArray[np.float64] = np.asarray(components, dtype=np.float64)
        if self._components.shape != (3,):
            raise ValueError(
                f"components must be 3-dimensional, got shape {self._components.shape}"
            )
        self._lattice = lattice

    @property
    def components(self) -> NDArray[np.float64]:
        """Read-only view of the vector components as a numpy array."""
        result: NDArray[np.float64] = self._components.view()
        result.flags.writeable = False
        return result

    @components.setter
    def components(self, value: Sequence[float]) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"components must be 3-dimensional, got shape {arr.shape}")
        self._components = arr

    @property
    def lattice(self) -> Lattice:
        """The lattice this vector is defined on."""
        return self._lattice

    def __array__(
        self, dtype: np.typing.DTypeLike | None = None
    ) -> NDArray[np.float64]:
        """Return components as a read-only numpy array.

        Implements the ``__array__`` protocol so that ``np.asarray(v)``
        returns a read-only view of the underlying data.

        Args:
            dtype: Optional dtype for the returned array.

        Returns:
            A read-only numpy array of the vector components.
        """
        arr = self._components if dtype is None else self._components.astype(dtype)
        result = arr.view()
        result.flags.writeable = False
        return result

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        other_lv = other  # same type confirmed by type() check above
        assert isinstance(other_lv, LatticeVector)
        return bool(
            self._lattice == other_lv._lattice
            and np.allclose(self._components, other_lv._components)
        )

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._components.tolist()})"

    def __str__(self) -> str:
        return repr(self)

    def __add__(self, other: object) -> LatticeVector:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, LatticeVector)
        if self._lattice != other._lattice:
            raise TypeError(
                f"lattice must be the same for both {self.__class__.__name__}s"
            )
        return type(self)(self._components + other._components, self._lattice)

    def __sub__(self, other: object) -> LatticeVector:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, LatticeVector)
        if self._lattice != other._lattice:
            raise TypeError(
                f"lattice must be the same for both {self.__class__.__name__}s"
            )
        return type(self)(self._components - other._components, self._lattice)

    def __neg__(self) -> LatticeVector:
        return type(self)(-self._components, self._lattice)

    def __mul__(self, scalar: object) -> LatticeVector:
        if not isinstance(scalar, Real):
            return NotImplemented
        return type(self)(self._components * float(scalar), self._lattice)

    def __rmul__(self, scalar: object) -> LatticeVector:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: object) -> LatticeVector:
        if not isinstance(scalar, Real):
            return NotImplemented
        return type(self)(self._components / float(scalar), self._lattice)

    def norm(self) -> float:
        """Calculate the norm (magnitude) of the vector.

        Returns:
            The length of the vector computed using the lattice metric tensor.

        Examples:
            >>> from diffraction import DirectLattice, DirectLatticeVector
            >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
            >>> v = DirectLatticeVector([1, 0, 0], calcite)
            >>> v.norm()
            4.99
        """
        c = self._components
        return float(np.sqrt(c @ self._lattice.metric @ c))

    def angle(self, other: LatticeVector) -> float:
        """Calculate the angle between this vector and another.

        Args:
            other: Another lattice vector (direct or reciprocal).

        Returns:
            The angle in degrees.
        """
        inner_product = self.inner(other)
        norm_self = self.norm()
        norm_other = other.norm()
        if norm_self == 0.0 or norm_other == 0.0:
            raise ValueError("Cannot compute angle with a zero-length vector")
        return math.degrees(math.acos(inner_product / (norm_self * norm_other)))

    def inner(self, other: LatticeVector) -> float:
        """Calculate the inner product with another lattice vector.

        Subclasses override this to provide metric-aware and cross-space
        variants.

        Args:
            other: Another lattice vector.

        Returns:
            The inner product value.
        """
        raise NotImplementedError


class DirectLatticeVector(LatticeVector):
    """Vector in direct space attached to a DirectLattice.

    Use a composition-based design wrapping a numpy array. The attached
    lattice provides the metric tensor required for computing norms, inner
    products, and angles in physical units.

    Args:
        uvw: Components (u, v, w) of the direct lattice vector.
        lattice: The DirectLattice this vector is defined on.

    Attributes:
        components: Read-only view of the underlying ndarray.
        lattice: The DirectLattice this vector is attached to.

    Examples:
        Create a direct lattice vector and compute its norm:

        >>> from diffraction import DirectLattice, DirectLatticeVector
        >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
        >>> u = DirectLatticeVector([1, 0, 0], lattice=calcite)
        >>> u.norm()
        4.99
        >>> v = calcite.vector([0, 0, 1])
        >>> u.inner(v)
        0.0
    """

    def __init__(self, uvw: Sequence[float], lattice: DirectLattice) -> None:
        super().__init__(uvw, lattice)

    @property
    def lattice(self) -> DirectLattice:
        """The DirectLattice this vector is defined on."""
        return self._lattice  # type: ignore[return-value]

    @overload  # type: ignore[override]
    def inner(self, other: DirectLatticeVector) -> float: ...

    @overload
    def inner(self, other: ReciprocalLatticeVector) -> float: ...

    def inner(self, other: LatticeVector) -> float:
        """Calculate the inner product with another lattice vector.

        For two direct lattice vectors on the same lattice, compute the
        standard metric inner product. If ``other`` is a
        ReciprocalLatticeVector whose lattice is reciprocally related to
        this vector's lattice, compute the cross-space inner product
        (scaled by 2*pi).

        Args:
            other: A DirectLatticeVector on the same lattice, or a
                ReciprocalLatticeVector on the reciprocally related lattice.

        Returns:
            The inner product in angstroms squared (same-space) or
            dimensionless radians (cross-space, scaled by 2*pi).

        Raises:
            TypeError: If other is a ReciprocalLatticeVector whose lattice
                is not reciprocally related to this vector's lattice.
            TypeError: If other is a DirectLatticeVector on a different
                lattice.

        Examples:
            Same-space inner product:

            >>> from diffraction import DirectLattice, DirectLatticeVector
            >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
            >>> u = DirectLatticeVector([1, 0, 0], calcite)
            >>> v = DirectLatticeVector([0, 0, 1], calcite)
            >>> u.inner(v)
            0.0
        """
        if isinstance(other, ReciprocalLatticeVector):
            if not np.allclose(
                self._lattice.metric,
                np.linalg.inv(other.lattice.metric / (2 * np.pi) ** 2),
                rtol=_RECIPROCAL_LATTICE_RTOL,
            ):
                self_name = self.__class__.__name__
                other_name = other.__class__.__name__
                raise TypeError(
                    f"{self_name} and {other_name} lattices must be reciprocally"
                    " related."
                )
            return float(2 * np.pi * self._components @ other._components)

        if not isinstance(other, DirectLatticeVector):
            raise TypeError(
                f"Cannot compute inner product between {self.__class__.__name__}"
                f" and {type(other).__name__}"
            )
        if self._lattice != other._lattice:
            raise TypeError(
                f"lattice must be the same for both {self.__class__.__name__}s"
            )
        return float(self._components @ self._lattice.metric @ other._components)


class ReciprocalLatticeVector(LatticeVector):
    """Vector in reciprocal space attached to a ReciprocalLattice.

    Use a composition-based design wrapping a numpy array. The attached
    reciprocal lattice provides the metric tensor for computing norms,
    inner products, and angles in reciprocal space.

    Args:
        hkl: Miller indices (h, k, l) of the reciprocal lattice vector.
        lattice: The ReciprocalLattice this vector is defined on.

    Attributes:
        components: Read-only view of the underlying ndarray.
        lattice: The ReciprocalLattice this vector is defined on.

    Examples:
        Create a reciprocal lattice vector and compute its norm:

        >>> from diffraction import DirectLattice, ReciprocalLatticeVector
        >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
        >>> rl = calcite.reciprocal()
        >>> u_ = ReciprocalLatticeVector([1, 0, 0], lattice=rl)
        >>> u_.norm()
        4.361842158457823
    """

    def __init__(self, hkl: Sequence[float], lattice: ReciprocalLattice) -> None:
        super().__init__(hkl, lattice)

    @property
    def lattice(self) -> ReciprocalLattice:
        """The ReciprocalLattice this vector is defined on."""
        return self._lattice  # type: ignore[return-value]

    @overload  # type: ignore[override]
    def inner(self, other: ReciprocalLatticeVector) -> float: ...

    @overload
    def inner(self, other: DirectLatticeVector) -> float: ...

    def inner(self, other: LatticeVector) -> float:
        """Calculate the inner product with another lattice vector.

        For two reciprocal lattice vectors on the same lattice, compute the
        standard metric inner product. If ``other`` is a
        DirectLatticeVector whose lattice is reciprocally related to this
        vector's lattice, compute the cross-space inner product scaled by
        2*pi.

        Args:
            other: A ReciprocalLatticeVector on the same lattice, or a
                DirectLatticeVector on the reciprocally related lattice.

        Returns:
            The inner product in reciprocal angstroms squared (same-space)
            or dimensionless radians (cross-space, scaled by 2*pi).

        Raises:
            TypeError: If other is a DirectLatticeVector whose lattice is
                not reciprocally related to this vector's lattice.
            TypeError: If other is a ReciprocalLatticeVector on a different
                lattice.

        Examples:
            Same-space inner product:

            >>> from diffraction import DirectLattice, ReciprocalLatticeVector
            >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
            >>> rl = calcite.reciprocal()
            >>> u_ = ReciprocalLatticeVector([1, 0, 0], rl)
            >>> v_ = ReciprocalLatticeVector([0, 0, 1], rl)
            >>> u_.inner(v_)
            0.0
        """
        if isinstance(other, DirectLatticeVector):
            if not np.allclose(
                self._lattice.metric,
                np.linalg.inv(other.lattice.metric) * (2 * np.pi) ** 2,
                rtol=_RECIPROCAL_LATTICE_RTOL,
            ):
                self_name = self.__class__.__name__
                other_name = other.__class__.__name__
                raise TypeError(
                    f"{self_name} and {other_name} lattices must be reciprocally"
                    " related."
                )
            return float(2 * np.pi * self._components @ other._components)

        if not isinstance(other, ReciprocalLatticeVector):
            raise TypeError(
                f"Cannot compute inner product between {self.__class__.__name__}"
                f" and {type(other).__name__}"
            )
        if self._lattice != other._lattice:
            raise TypeError(
                f"lattice must be the same for both {self.__class__.__name__}s"
            )
        return float(self._components @ self._lattice.metric @ other._components)
