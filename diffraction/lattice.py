"""Direct and reciprocal lattices and vectors.

This module provides objects for performing direct and reciprocal space
lattice calculations. The naming conventions used follow those in the
International Tables of Crystallography.

Classes
-------
DirectLattice, ReciprocalLattice
    Classes to represent direct and reciprocal lattices that
    encapsulate several basic properties of the lattices including the
    direct and reciprocal lattice parameters, unit cell volume, and
    metric tensors. These can be instantiated either just from the
    lattice parameters or from a dictionary or CIF.

DirectLatticeVector, ReciprocalLatticeVector
    Classes for to represent direct and reciprocal lattice vectors that
    can be used for calculations of lengths and angles in direct
    and reciprocal space. These can be instantiated directly by
    supplying the associated lattice in the constructor or by using the
    factory method of the corresponding Lattice object.


Examples
--------
>>> from diffraction import DirectLattice

Create a direct lattice object from given lattice parameters

>>> calcite_lattice = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])

Lattice parameters and computed properties are accessible directly.

>>> calcite_lattice.c
17.002
>>> calcite_lattice.metric
array([[ 24.9001  , -12.45005 ,   0.      ],
       [-12.45005 ,  24.9001  ,   0.      ],
       [  0.      ,   0.      , 289.068004]])

The corresponding reciprocal lattice object can be constructed.

>>> calcite_reciprocal_lattice = calcite_lattice.reciprocal()
>>> calcite_reciprocal_lattice.b_star
1.4539473861596934
>>> calcite_recpirocal_lattice.gamma_star
60.00000000000002

Lattice vector calculations can be done in both direct space...

>>> u = DirectLatticeVector([1, 0, 0], lattice=calcite_lattice)
>>> u.norm()
4.99
>>> v = calcite_lattice.vector([0, 0, 1])
>>> u + v
DirectLatticeVector([ 1, 0,  1])
>>> u.inner(v)
0.0

...in reciprocal space...

>>> u_ = ReciprocalLatticeVector([1, 0, 0], lattice=calcite_reciprocal_lattice)
>>> u_.norm()
4.361842158457823
>>> v_ = calcite_reciprocal_lattice.vector([0, 1, 0])
>>> u_.angle(v_)
59.99999999843518

... and between the two.

>>> u.inner(u_)  # = 2 * pi
6.283185307179586
>>> u.angle(u_)
29.999999999516355
"""

import abc
from functools import wraps
import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .cif.helpers import CIF_NAMES, get_cif_data, load_data_block

LatticeParameters = Sequence[float]

__all__ = ["DirectLattice", "DirectLatticeVector", "ReciprocalLattice",
           "ReciprocalLatticeVector"]


def _to_radians(lattice_parameters: LatticeParameters) -> LatticeParameters:
    """Convert angles in :term:`lattice parameters` from degrees to
    radians.

    Parameters
    ----------
    lattice_parameters
        The :term:`lattice parameters` with angles in units of degrees.

    Returns
    -------
    tuple:
        The :term:`lattice parameters` with angles in units of radians.
    """

    lengths = tuple(lattice_parameters)[:3]
    angles_in_radians = tuple(math.radians(angle)
                              for angle in tuple(lattice_parameters)[3:])
    return lengths + angles_in_radians


def _to_degrees(lattice_parameters: LatticeParameters) -> LatticeParameters:
    """Convert angles in :term:`lattice parameters` from radians to
    degrees.

    Parameters
    ----------
    lattice_parameters: seq
        The lattice parameters with angles in units of radians.

    Returns
    -------
    tuple:
        The lattice parameters with angles in units of degrees.
    """

    lengths = tuple(lattice_parameters)[:3]
    angles_in_degrees = tuple(math.degrees(angle)
                              for angle in tuple(lattice_parameters)[3:])
    return lengths + angles_in_degrees


def metric_tensor(lattice_parameters: LatticeParameters) -> np.ndarray:
    """Calculate the :term:`metric tensor` for a given :term:`lattice`.

    Parameters
    ----------
    lattice_parameters: seq
        The lattice parameters of the lattice with the angles in units
        of degrees.

    Returns
    -------
    ndarray:
        The metric tensor of the lattice.
    """
    a, b, c, al, be, ga = _to_radians(lattice_parameters)
    tensor = np.around([[a ** 2, a * b * np.cos(ga), a * c * math.cos(be)],
                        [a * b * math.cos(ga), b ** 2, b * c * math.cos(al)],
                        [a * c * math.cos(be), b * c * np.cos(al), c ** 2]],
                       10)
    return tensor


def reciprocalise(lattice_parameters: LatticeParameters) -> LatticeParameters:
    """Transform the lattice parameters to those of the reciprocally
    related lattice.

    Transforms the lattice parameters of the given lattice to those of
    the reciprocal lattice. i.e. converts direct lattice parameters to those
    of the reciprocal lattice and vice versa.

    Parameters
    ----------
    lattice_parameters: seq
        The lattice parameters of the lattice, with the angles in units
        of degrees.

    Returns
    -------
    tuple:
        The lattice parameters of the lattice reciprocally related to
        the input lattice, with the angles in units of degrees.
    """
    a, b, c, al, be, ga = _to_radians(lattice_parameters)
    cell_volume = np.sqrt(np.linalg.det(metric_tensor(lattice_parameters)))
    pi, sin, cos, arccos = math.pi, math.sin, math.cos, math.acos

    a_ = 2 * pi * b * c * sin(al) / cell_volume
    b_ = 2 * pi * a * c * sin(be) / cell_volume
    c_ = 2 * pi * a * b * sin(ga) / cell_volume
    alpha_ = arccos((cos(be) * cos(ga) - cos(al)) / (sin(be) * sin(ga)))
    beta_ = arccos((cos(al) * cos(ga) - cos(be)) / (sin(al) * sin(ga)))
    gamma_ = arccos((cos(al) * cos(be) - cos(ga)) / (sin(al) * sin(be)))

    return _to_degrees((a_, b_, c_, alpha_, beta_, gamma_))


class Lattice(abc.ABC):
    """Abstract base class for lattice objects.

    Parameters
    ----------
    lattice_parameters
        The :term:`lattice parameters` of the lattice declared in the
        order [*a*, *b*, *c*, *alpha*, *beta*, *gamma*].

    Attributes
    ----------
    a, b, c: float
        The *a*, *b* and *c* :term:`lattice parameters` describing the
        dimensions of the :term:`unit cell`.
    alpha, beta, gamma: float
        The *alpha*, *beta* and *gamma* :term:`lattice parameters`
        describing the angles of the :term:`unit cell` in degrees.
    lattice_parameters: tuple of float
        The :term:`lattice parameters` in the form
        (*a*, *b*, *c*, *alpha*, *beta*, *gamma*) ]
        with angles in degrees.
    metric: ndarray
        The :term:`metric tensor` of the direct basis.
    unit_cell_volume: float
        The volume of the :term:`unit cell`

    Class Attributes
    ----------------
    lattice_parameter_keys: tuple
    """
    lattice_parameter_keys = None  # type: Tuple[str, str, str, str, str, str]

    def __init__(self, lattice_parameters: LatticeParameters):
        lattice_parameters = self.check_lattice_parameters(lattice_parameters)
        for key, value in zip(self.lattice_parameter_keys, lattice_parameters):
            setattr(self, key, value)

    def check_lattice_parameters(self, lattice_parameters: LatticeParameters
                                 ) -> LatticeParameters:
        """Check given lattice parameters are valid.

        Parameters:
        -----------
        lattice_parameters: seq of float
            The :term:`lattice parameters` in the form
            (*a*, *b*, *c*, *alpha*, *beta*, *gamma*)
            with angles in degrees.

        Raises:
        ------
        ValueError:
            If the input :term:`lattice parameters` are missing any
            any values or if any of the values are invalid.
        """
        if len(lattice_parameters) < 6:
            raise (ValueError("Missing lattice parameter from input"))
        lattice_parameters_ = []
        for key, value in zip(self.lattice_parameter_keys, lattice_parameters):
            try:
                lattice_parameters_.append(float(value))
            except ValueError:
                raise ValueError("Invalid lattice parameter {0}: {1}".format(
                    key, value))
        return lattice_parameters_

    @classmethod
    @abc.abstractmethod
    def from_cif(cls,
                 filepath: str,
                 data_block: Optional[str] = None
                 ) -> "AbstractLattice":
        raise NotImplementedError

    @classmethod
    def from_dict(cls, input_dict: Dict[str, float]) -> "AbstractLattice":
        """Create an AbstractLattice using a dictionary as input

        Raises
        ------
        ValueError:
            If the input dict is missing any :term:`lattice parameters`
        """

        lattice_parameters = []
        for parameter in cls.lattice_parameter_keys:
            try:
                lattice_parameters.append(input_dict[parameter])
            except KeyError:  # TODO: Is OK that reports just 1st missing para?
                raise ValueError("Parameter: '{0}' missing from input "
                                 "dictionary".format(parameter))
        return cls(lattice_parameters)

    @property
    def lattice_parameters(self) -> LatticeParameters:
        return tuple(getattr(self, name)
                     for name in self.lattice_parameter_keys)

    @property
    def metric(self) -> np.ndarray:
        return metric_tensor(self.lattice_parameters)

    @property
    def unit_cell_volume(self) -> float:
        return np.sqrt(np.linalg.det(self.metric))

    def __repr__(self) -> str:
        repr_string = ("{0}([{1!r}, {2!r}, {3!r}, "
                       "{4!r}, {5!r}, {6!r}])")
        rounded_lattice_parameters = [round(parameter, 4) for parameter
                                      in self.lattice_parameters]
        return repr_string.format(self.__class__.__name__,
                                  *rounded_lattice_parameters)

    def __str__(self) -> str:
        return repr(self)


class DirectLattice(Lattice):
    """Class to represent a direct lattice

    Parameters
    ----------
    lattice_parameters: seq of float
        The :term:`lattice parameters` of the lattice declared in the
        order [*a*, *b*, *c*, *alpha*, *beta*, *gamma*]

    Attributes
    ----------
    a, b, c: float
        The *a*, *b* and *c* :term:`lattice parameters` describing the
        dimensions of the :term:`unit cell`.
    alpha, beta, gamma: float
        The *alpha*, *beta* and *gamma* :term:`lattice parameters`
        describing the angles of the :term:`unit cell` in degrees.
    lattice_parameters: tuple of float
        The :term:`lattice parameters` in the form
        (*a*, *b*, *c*, *alpha*, *beta*, *gamma*)
        with angles in degrees.
    metric: ndarray
        The :term:`metric tensor` of the direct basis.
    unit_cell_volume: float
        The volume of the :term:`unit cell`

    Class Attributes
    ----------------
    lattice_parameter_keys: tuple

    Examples
    --------
    >>> from diffraction import DirectLattice

    Create a direct lattice object from given lattice parameters

    >>> calcite_lattice_parameters = [4.99, 4.99, 17.002, 90, 90, 120]
    >>> calcite_lattice = DirectLattice(calcite_lattice_parameters)

    Lattice parameters are accessible individually or as a tuple.

    >>> calcite_lattice.b
    4.99
    >>> calcite_lattice.gamma
    120.0
    >>> calcite_lattice.lattice_parameters
    (4.99, 4.99, 17.002, 90.0, 90.0, 120.0)

    Unit cell volume and the metric tensor are accessible as attributes.

    >>> calcite_lattice.unit_cell_volume
    366.63315390345286
    >>> calcite_lattice.metric
    array([[ 24.9001  , -12.45005 ,   0.      ],
           [-12.45005 ,  24.9001  ,   0.      ],
           [  0.      ,   0.      , 289.068004]])
    """
    lattice_parameter_keys = ("a", "b", "c", "alpha", "beta", "gamma")

    @classmethod
    def from_cif(cls,
                 filepath: str,
                 data_block: Optional[str] = None
                 ) -> "DirectLattice":
        """Create a DirectLattice using a :term:`CIF` as input.

        Parameters
        ----------
        filepath: str
            Filepath to the input CIF
        data_block: str, optional
            The :term:`data block` to generate the DirectLattice from,
            specified by :term:`data block header`. Only required for
            an input :term:`CIF` with multiple data blocks.

        Raises
        ------
        ValueError:
            If the input :term:`CIF` is missing any :term:`lattice
            parameters`.
        ValueError:
            If any of the :term:`lattice parameters` are not valid
            numerical data.
        TypeError:
            If the input CIF has multiple data blocks but data_block is
            not given.
        """

        data_items = load_data_block(filepath, data_block)
        data_names = [CIF_NAMES[key] for key in cls.lattice_parameter_keys]
        lattice_parameters = get_cif_data(data_items, *data_names)
        return cls(lattice_parameters)

    def vector(self, uvw: Sequence[float]) -> "DirectLatticeVector":
        return DirectLatticeVector(uvw, self)

    def reciprocal(self) -> "ReciprocalLattice":
        reciprocal_lattice_parameters = reciprocalise(self.lattice_parameters)
        return ReciprocalLattice(reciprocal_lattice_parameters)


class ReciprocalLattice(Lattice):
    """Class to represent a reciprocal lattice

    Attributes
    ----------
    a_star, b_star, c_star: float
        The *a_star*, *b_star* and *c_star* :term:`lattice parameters`
        describing the dimensions of the :term:`unit cell`.
    alpha_star, beta_star, gamma_star: float
        The *alpha_star*, *beta_star* and *gamma_star* :term:`lattice
        parameters` describing the angles of the reciprocal lattice
        :term:`unit cell`, in degrees.
    lattice_parameters: tuple of float
        The :term:`lattice parameters` with the angles in degrees.
    metric: ndarray
        The :term:`metric tensor` of the reciprocal basis.
    unit_cell_volume: float
        The volume of the :term:`unit cell`

    Class Attributes
    ----------------
    lattice_parameter_keys: tuple

    """
    lattice_parameter_keys = ("a_star", "b_star", "c_star",
                              "alpha_star", "beta_star", "gamma_star")

    @classmethod
    def from_cif(cls,
                 filepath: str,
                 data_block: Optional[str] = None
                 ) -> "ReciprocalLattice":
        """Create an AbstractLattice using a :term:`CIF` as input

        Parameters
        ----------
        filepath: str
            Filepath to the input CIF
        data_block: str, optional
            The :term:`data block` to generate the Lattice from,
            specified by :term:`data block header`. Only required for
            an input :term:`CIF` with multiple data blocks.

        Raises
        ------
        ValueError:
            If the input :term:`CIF` is missing any :term:`lattice
            parameters`.
        ValueError:
            If any of the :term:`lattice parameters` are not valid
            numerical data.
        TypeError:
            If the input CIF has multiple data blocks but data_block is
            not given.
        """

        data_items = load_data_block(filepath, data_block)
        data_names = [CIF_NAMES[key] for key in
                      "a b c alpha beta gamma".split()]
        lattice_parameters = get_cif_data(data_items, *data_names)
        reciprocal_lps = reciprocalise(lattice_parameters)
        return cls(reciprocal_lps)

    def vector(self, hkl: Sequence[float]) -> "ReciprocalLatticeVector":
        """Return a reciprocal lattice vector defined on this
        reciprocal lattice.

        Parameters
        ----------
        hkl:
            The indices of the reciprocal lattice vector, given as a
            sequence.

        Returns
        -------
        ReciprocalLatticeVector:
            A reciprocal lattice vector with the given indices, defined
            on this reciprocal lattice.
        """
        return ReciprocalLatticeVector(hkl, self)

    def direct(self) -> "DirectLattice":
        """Return the corresponding direct lattice object."""
        direct_lattice_parameters = reciprocalise(self.lattice_parameters)
        return DirectLattice(direct_lattice_parameters)


def check_lattice(operation: Callable) -> Callable:
    @wraps(operation)  # TODO: sort error msg when adding direct + recip vector
    def wrapper(self, other):
        if self.lattice != other.lattice:
            raise TypeError("lattice must be the same for both "
                            "{0}s".format(self.__class__.__name__))
        else:
            return operation(self, other)

    return wrapper


class DirectLatticeVector(np.ndarray):
    """Class to represent a direct lattice vector

    Parameters
    ----------
    uvw: array_like
        The u, v, w components of the direct lattice vector.
    lattice: DirectLattice
        The direct lattice the vector is associated with.

    Attributes
    ----------
    lattice: DirectLattice
        The direct lattice the vector is associated with.

    Methods
    -------
    norm:
        Calculate the norm of the vector.
    inner:
        Calculate the inner product of the vector with another direct
        lattice vector.
    angle:
        Calculate the angle between the vector and another direct
        lattice vector.

    """  # TODO: finish docstring
    # TODO: implement __repr__ + tests

    def __new__(cls,
                uvw: Sequence,
                lattice: DirectLattice
                ) -> "DirectLatticeVector":
        vector = np.asarray(uvw).view(cls)
        vector.lattice = lattice
        return vector

    def __array_finalize__(self, vector: "DirectLatticeVector"):
        self.lattice = getattr(vector, "lattice", None)

    def __eq__(self, other: "DirectLatticeVector") -> bool:
        return np.array_equal(self, other) and self.lattice == other.lattice

    def __ne__(self, other: "DirectLatticeVector") -> bool:
        return not self == other

    @check_lattice
    def __add__(self, other: "DirectLatticeVector") -> "DirectLatticeVector":
        return np.ndarray.__add__(self, other)

    @check_lattice
    def __sub__(self, other: "DirectLatticeVector") -> "DirectLatticeVector":
        return np.ndarray.__sub__(self, other)

    def __repr__(self) -> str:
        return "{0}({1}, {2})".format(
            self.__class__.__name__, list(self), self.lattice)

    def __str__(self) -> str:
        return "{0}({1})".format(self.__class__.__name__, list(self))

    def norm(self) -> float:
        """Calculate the norm (or magnitude) of the vector

        Returns
        -------
        float:
            The norm of the vector.
        """

        return np.sqrt(self.dot(self.lattice.metric).dot(self))

    def inner(self, other: "DirectLatticeVector") -> float:
        """Calculate the inner product between the vector and another direct
        lattice vector

        Parameters
        ----------
        other:
            The

        Returns
        -------

        """
        #  TODO: is there any way to do this apart from type-checking?
        if type(other) is ReciprocalLatticeVector:
            if not np.allclose(
                    self.lattice.metric,
                    np.linalg.inv(other.lattice.metric / (2 * np.pi) ** 2),
                    rtol=1e-2):
                raise TypeError("{0} and {1} lattices must be reciprocally "
                                "related.".format(self.__class__.__name__,
                                                  other.__class__.__name__))
            return 2 * np.pi * self.dot(other)

        if self.lattice != other.lattice:
            raise TypeError("lattice must be the same for both "
                            "{0}s".format(self.__class__.__name__))

        return self.dot(self.lattice.metric).dot(other)

    def angle(self, other: "DirectLatticeVector") -> float:
        u, v = self, other
        return math.degrees(math.acos(u.inner(v) / (u.norm() * v.norm())))


class ReciprocalLatticeVector(DirectLatticeVector):  # TODO: Finish docstrings
    """Class to represent a direct lattice vector

    Parameters
    ----------
    hkl: array_like
        The h, k, l components of the reciprocal lattice vector.
    lattice: ReciprocalLattice
        The reciprocal lattice the vector is associated with.

    Attributes
    ----------
    lattice: ReciprocalLattice
        The reciprocal lattice the vector is associated with.

    Methods
    -------
    norm:
        Calculate the norm of the vector.
    inner:
        Calculate the inner product of the vector with another reciprocal
        lattice vector.
    angle:
        Calculate the angle between the vector and another reciprocal
        lattice vector.

    """

    def __new__(cls,
                hkl: Sequence[float],
                lattice: ReciprocalLattice
                ) -> "ReciprocalLatticeVector":
        vector = np.asarray(hkl).view(cls)
        vector.lattice = lattice
        return vector

    # TODO: add copies of functions so docstrings aren't inherited using super

    def inner(self,
              other: Union["DirectLatticeVector", "ReciprocalLatticeVector"]
              ) -> float:
        """Calculate the inner product between the vector and another direct
        lattice vector

        Parameters
        ----------
        other:
            The

        Returns
        -------

        """
        #  TODO: is there any way to do this apart from type-checking?
        if type(other) is DirectLatticeVector:
            if not np.allclose(
                    self.lattice.metric,
                    np.linalg.inv(other.lattice.metric) * (2 * np.pi) ** 2,
                    rtol=1e-2):
                raise TypeError("{0} and {1} lattices must be reciprocally "
                                "related.".format(self.__class__.__name__,
                                                  other.__class__.__name__))
            return 2 * np.pi * self.dot(other)

        if self.lattice != other.lattice:
            raise TypeError("lattice must be the same for both "
                            "{0}s".format(self.__class__.__name__))
        return self.dot(self.lattice.metric).dot(other)
