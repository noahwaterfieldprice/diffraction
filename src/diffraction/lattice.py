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
    >>> calcite_lattice.metric
    array([[ 24.9001  , -12.45005 ,   0.      ],
           [-12.45005 ,  24.9001  ,   0.      ],
           [  0.      ,   0.      , 289.068004]])

    Construct the corresponding reciprocal lattice:

    >>> calcite_reciprocal_lattice = calcite_lattice.reciprocal()
    >>> calcite_reciprocal_lattice.b_star
    1.4539473861596934
    >>> calcite_reciprocal_lattice.gamma_star
    60.00000000000002

    Perform lattice vector calculations in direct space:

    >>> u = DirectLatticeVector([1, 0, 0], lattice=calcite_lattice)
    >>> u.norm()
    4.99
    >>> v = calcite_lattice.vector([0, 0, 1])
    >>> u.inner(v)
    0.0

    And in reciprocal space, including the cross-space inner product:

    >>> u_ = ReciprocalLatticeVector([1, 0, 0], lattice=calcite_reciprocal_lattice)
    >>> u_.norm()
    4.361842158457823
    >>> u.inner(u_)  # equals 2 * pi
    6.283185307179586
"""

import abc
import math
from collections.abc import Callable, Sequence
from functools import wraps
from typing import ClassVar, TypeAlias, TypeVar, cast

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

_VT = TypeVar("_VT", bound="DirectLatticeVector")


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


def metric_tensor(lattice_parameters: LatticeParameters) -> NDArray[np.float64]:
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
        10,
    )
    return tensor


def reciprocalise(lattice_parameters: LatticeParameters) -> tuple[float, ...]:
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
    cell_volume = float(np.sqrt(np.linalg.det(metric_tensor(lattice_parameters))))
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
    subclasses must define ``lattice_parameter_keys`` and implement
    ``from_cif``.

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
            ValueError: If fewer than six parameters are provided, or if any
                parameter cannot be converted to float.
        """
        if len(lattice_parameters) < 6:
            raise (ValueError("Missing lattice parameter from input"))
        lattice_parameters_: list[float] = []
        for key, value in zip(
            self.lattice_parameter_keys, lattice_parameters, strict=False
        ):
            try:
                lattice_parameters_.append(float(value))
            except ValueError as exc:
                raise ValueError(f"Invalid lattice parameter {key}: {value}") from exc
        return lattice_parameters_

    @classmethod
    @abc.abstractmethod
    def from_cif(cls, filepath: str, data_block: str | None = None) -> "Lattice":
        raise NotImplementedError

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
                the dictionary.
        """
        lattice_parameters = []
        for parameter in cls.lattice_parameter_keys:
            try:
                lattice_parameters.append(input_dict[parameter])
            except KeyError as exc:  # TODO: Is OK that reports just 1st missing para?
                raise ValueError(
                    f"Parameter: '{parameter}' missing from input dictionary"
                ) from exc
        return cls(lattice_parameters)

    @property
    def lattice_parameters(self) -> tuple[float, ...]:
        """Return all lattice parameters as a tuple."""
        return tuple(getattr(self, name) for name in self.lattice_parameter_keys)

    @property
    def metric(self) -> NDArray[np.float64]:
        """Return the 3x3 metric tensor for this lattice."""
        return metric_tensor(self.lattice_parameters)

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
    def from_cif(cls, filepath: str, data_block: str | None = None) -> "DirectLattice":
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
        lattice_parameters = cif_helpers.get_cif_data(data_items, *data_names)
        return cls(cast(list[float], lattice_parameters))

    def vector(self, uvw: Sequence[float]) -> "DirectLatticeVector":
        """Return a direct lattice vector defined on this lattice.

        Args:
            uvw: Miller indices (u, v, w) of the direct lattice vector.

        Returns:
            A DirectLatticeVector with the given indices attached to this
            lattice.
        """
        return DirectLatticeVector(uvw, self)

    def reciprocal(self) -> "ReciprocalLattice":
        """Return the corresponding reciprocal lattice.

        Returns:
            A ReciprocalLattice with parameters derived from this direct
            lattice using the 2*pi convention.
        """
        reciprocal_lattice_parameters = reciprocalise(self.lattice_parameters)
        return ReciprocalLattice(reciprocal_lattice_parameters)


class ReciprocalLattice(Lattice):
    """Reciprocal lattice defined by six reciprocal lattice parameters.

    Store and expose reciprocal lattice parameters (a_star, b_star, c_star,
    alpha_star, beta_star, gamma_star) using the 2*pi convention. Can be
    constructed from a parameter list, a dictionary, or from a CIF file
    (which reads the direct lattice parameters and converts them).

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

    @classmethod
    def from_cif(
        cls, filepath: str, data_block: str | None = None
    ) -> "ReciprocalLattice":
        """Create a ReciprocalLattice from a CIF file.

        Read direct lattice parameters from the CIF and convert them to
        reciprocal lattice parameters using the 2*pi convention.

        Args:
            filepath: Path to the input CIF file.
            data_block: Data block header to read from, required only when
                the CIF contains multiple data blocks.

        Returns:
            A ReciprocalLattice with parameters derived from the CIF direct
            lattice parameters.

        Raises:
            ValueError: If any required lattice parameter is missing or is
                not valid numerical data.
            TypeError: If the CIF has multiple data blocks but data_block
                is not given.
        """
        data_items = cif_helpers.load_data_block(filepath, data_block)
        data_names = [
            cif_helpers.CIF_NAMES[key]
            for key in ["a", "b", "c", "alpha", "beta", "gamma"]
        ]
        lattice_parameters = cif_helpers.get_cif_data(data_items, *data_names)
        reciprocal_lps = reciprocalise(cast(list[float], lattice_parameters))
        return cls(reciprocal_lps)

    def vector(self, hkl: Sequence[float]) -> "ReciprocalLatticeVector":
        """Return a reciprocal lattice vector defined on this lattice.

        Args:
            hkl: Miller indices (h, k, l) of the reciprocal lattice vector.

        Returns:
            A ReciprocalLatticeVector with the given indices attached to
            this lattice.
        """
        return ReciprocalLatticeVector(hkl, self)

    def direct(self) -> "DirectLattice":
        """Return the corresponding direct lattice.

        Returns:
            A DirectLattice with parameters derived from this reciprocal
            lattice by applying the inverse reciprocalise transform.
        """
        direct_lattice_parameters = reciprocalise(self.lattice_parameters)
        return DirectLattice(direct_lattice_parameters)


def check_lattice(
    operation: Callable[[_VT, _VT], _VT],
) -> Callable[[_VT, _VT], _VT]:
    @wraps(operation)  # TODO: sort error msg when adding direct + recip vector
    def wrapper(self: _VT, other: _VT) -> _VT:
        if self.lattice is None or other.lattice is None:
            raise TypeError(
                f"Cannot perform operation: {self.__class__.__name__} has no"
                " attached lattice"
            )
        if self.lattice != other.lattice:
            raise TypeError(
                f"lattice must be the same for both {self.__class__.__name__}s"
            )
        else:
            return operation(self, other)

    return wrapper


class DirectLatticeVector(np.ndarray):
    """Vector in direct space attached to a DirectLattice.

    Extend numpy ndarray to represent a direct lattice vector [u, v, w].
    The attached lattice provides the metric tensor required for computing
    norms, inner products, and angles in physical units.

    Args:
        uvw: Components (u, v, w) of the direct lattice vector.
        lattice: The DirectLattice this vector is defined on.

    Attributes:
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

    lattice: DirectLattice | None

    def __new__(
        cls, uvw: Sequence[float], lattice: DirectLattice
    ) -> "DirectLatticeVector":
        vector = np.asarray(uvw).view(cls)
        vector.lattice = lattice
        return vector

    def __array_finalize__(
        self, vector: object
    ) -> None:
        self.lattice = getattr(vector, "lattice", None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectLatticeVector):
            return NotImplemented
        return bool(np.array_equal(self, other) and self.lattice == other.lattice)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, DirectLatticeVector):
            return NotImplemented
        return not self.__eq__(other)

    @check_lattice
    def __add__(  # type: ignore[override]
        self, other: "DirectLatticeVector"
    ) -> "DirectLatticeVector":
        return cast("DirectLatticeVector", np.ndarray.__add__(self, other))

    @check_lattice
    def __sub__(  # type: ignore[override]
        self, other: "DirectLatticeVector"
    ) -> "DirectLatticeVector":
        return cast("DirectLatticeVector", np.ndarray.__sub__(self, other))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)}, {self.lattice})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"

    def norm(self) -> float:
        """Calculate the norm (magnitude) of the vector.

        Returns:
            The length of the vector in angstroms.
        """
        if self.lattice is None:
            raise TypeError("Cannot compute norm: vector has no attached lattice")
        return float(np.sqrt(self.dot(self.lattice.metric).dot(self)))

    def inner(self, other: "DirectLatticeVector") -> float:
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
        """
        if self.lattice is None:
            raise TypeError(
                "Cannot compute inner product: vector has no attached lattice"
            )
        #  TODO: is there any way to do this apart from type-checking?
        if type(other) is ReciprocalLatticeVector:
            if other.lattice is None:
                raise TypeError(
                    "Cannot compute inner product: vector has no attached lattice"
                )
            if not np.allclose(
                self.lattice.metric,
                np.linalg.inv(other.lattice.metric / (2 * np.pi) ** 2),
                rtol=1e-2,
            ):
                self_name = self.__class__.__name__
                other_name = other.__class__.__name__
                raise TypeError(
                    f"{self_name} and {other_name} lattices must be reciprocally"
                    " related."
                )
            return float(2 * np.pi * self.dot(other))

        if self.lattice != other.lattice:
            raise TypeError(
                f"lattice must be the same for both {self.__class__.__name__}s"
            )

        return float(self.dot(self.lattice.metric).dot(other))

    def angle(self, other: "DirectLatticeVector") -> float:
        """Calculate the angle between this vector and another.

        Args:
            other: Another lattice vector (direct or reciprocal).

        Returns:
            The angle in degrees.
        """
        u, v = self, other
        return math.degrees(math.acos(u.inner(v) / (u.norm() * v.norm())))


class ReciprocalLatticeVector(DirectLatticeVector):
    """Vector in reciprocal space attached to a ReciprocalLattice.

    Extend DirectLatticeVector to represent a reciprocal lattice vector
    [h, k, l]. The attached reciprocal lattice provides the metric tensor
    for computing norms, inner products, and angles in reciprocal space.

    Args:
        hkl: Miller indices (h, k, l) of the reciprocal lattice vector.
        lattice: The ReciprocalLattice this vector is defined on.

    Attributes:
        lattice: The ReciprocalLattice this vector is attached to.

    Examples:
        Create a reciprocal lattice vector and compute its norm:

        >>> from diffraction import DirectLattice, ReciprocalLatticeVector
        >>> calcite = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])
        >>> rl = calcite.reciprocal()
        >>> u_ = ReciprocalLatticeVector([1, 0, 0], lattice=rl)
        >>> u_.norm()
        4.361842158457823
    """

    lattice: ReciprocalLattice | None  # type: ignore[assignment]

    def __new__(
        cls, hkl: Sequence[float], lattice: ReciprocalLattice
    ) -> "ReciprocalLatticeVector":
        vector = np.asarray(hkl).view(cls)
        vector.lattice = lattice
        return vector

    # TODO: add copies of functions so docstrings aren't inherited using super

    def inner(self, other: "DirectLatticeVector | ReciprocalLatticeVector") -> float:
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
        """
        if self.lattice is None:
            raise TypeError(
                "Cannot compute inner product: vector has no attached lattice"
            )
        #  TODO: is there any way to do this apart from type-checking?
        if type(other) is DirectLatticeVector:
            if other.lattice is None:
                raise TypeError(
                    "Cannot compute inner product: vector has no attached lattice"
                )
            if not np.allclose(
                self.lattice.metric,
                np.linalg.inv(other.lattice.metric) * (2 * np.pi) ** 2,
                rtol=1e-2,
            ):
                self_name = self.__class__.__name__
                other_name = other.__class__.__name__
                raise TypeError(
                    f"{self_name} and {other_name} lattices must be reciprocally"
                    " related."
                )
            return float(2 * np.pi * self.dot(other))

        if self.lattice != other.lattice:
            raise TypeError(
                f"lattice must be the same for both {self.__class__.__name__}s"
            )
        return float(self.dot(self.lattice.metric).dot(other))
