from functools import wraps

from numpy import arccos, cos, sin, sqrt
import numpy as np

from .cif.helpers import CIF_NAMES, get_cif_data, load_data_block

__all__ = ["DirectLattice", "DirectLatticeVector", "ReciprocalLattice",
           "ReciprocalLatticeVector"]


def _to_radians(lattice_parameters):
    """Convert angles in :term:`lattice parameters` from degrees to
    radians.

    Parameters
    ----------
    lattice_parameters: seq
        The :term:`lattice parameters` with angles in units of degrees.

    Returns
    -------
    tuple:
        The :term:`lattice parameters` with angles in units of radians.
    """

    lengths = tuple(lattice_parameters)[:3]
    angles_in_radians = tuple(np.radians(angle)
                              for angle in tuple(lattice_parameters)[3:])
    return lengths + angles_in_radians


def _to_degrees(lattice_parameters):
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
    angles_in_degrees = tuple(np.degrees(angle)
                              for angle in tuple(lattice_parameters)[3:])
    return lengths + angles_in_degrees


def metric_tensor(lattice_parameters):
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
    tensor = np.around([[a ** 2, a * b * cos(ga), a * c * cos(be)],
                        [a * b * cos(ga), b ** 2, b * c * cos(al)],
                        [a * c * cos(be), b * c * cos(al), c ** 2]],
                       10)
    return tensor


def reciprocalise(lattice_parameters):  # TODO: fix docstring, factor of 2pi?
    """Transform the lattice parameters to those of the reciprocal lattice.

    Transforms the lattice parameters of the given lattice to those of
    the reciprocal lattice. i.e. converts direct lattice parameters to those
    of the reciprocal lattice and vice versa.

    Parameters
    ----------
    lattice_parameters: seq
        The lattice parameters of the lattice with the angles in units
        of degrees.

    Returns
    -------
    tuple:
        The lattice parameters of the +++ASDOIAJFOIas

    """
    a, b, c, al, be, ga = _to_radians(lattice_parameters)
    cell_volume = sqrt(np.linalg.det(metric_tensor(lattice_parameters)))

    a_ = 2 * np.pi * b * c * sin(al) / cell_volume
    b_ = 2 * np.pi * a * c * sin(be) / cell_volume
    c_ = 2 * np.pi * a * b * sin(ga) / cell_volume
    alpha_ = arccos((cos(be) * cos(ga) - cos(al)) / (sin(be) * sin(ga)))
    beta_ = arccos((cos(al) * cos(ga) - cos(be)) / (sin(al) * sin(ga)))
    gamma_ = arccos((cos(al) * cos(be) - cos(ga)) / (sin(al) * sin(be)))

    return _to_degrees((a_, b_, c_, alpha_, beta_, gamma_))


class AbstractLattice:
    """Abstract base class for lattice objects.

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


    """
    lattice_parameter_keys = None

    def __init__(self, lattice_parameters):
        lattice_parameters = self.convert_parameters(lattice_parameters)
        for key, value in zip(self.lattice_parameter_keys, lattice_parameters):
            setattr(self, key, value)

    def convert_parameters(self, lattice_parameters):
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
    def from_cif(cls, filepath, data_block=None):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, input_dict):
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
    def lattice_parameters(self):
        return tuple(getattr(self, name)
                     for name in self.lattice_parameter_keys)

    @property
    def metric(self):
        return metric_tensor(self.lattice_parameters)

    @property
    def unit_cell_volume(self):
        return np.sqrt(np.linalg.det(self.metric))

    def __repr__(self):
        repr_string = ("{0}([{1!r}, {2!r}, {3!r}, "
                       "{4!r}, {5!r}, {6!r}])")
        return repr_string.format(self.__class__.__name__,
                                  *self.lattice_parameters)

    def __str__(self):
        return repr(self)


class DirectLattice(AbstractLattice):
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


    """
    lattice_parameter_keys = ("a", "b", "c", "alpha", "beta", "gamma")

    @classmethod
    def from_cif(cls, filepath, data_block=None):
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
        data_names = [CIF_NAMES[key] for key in
                      "a b c alpha beta gamma".split()]
        lattice_parameters = get_cif_data(data_items, *data_names)
        return cls(lattice_parameters)

    def vector(self, uvw):
        return DirectLatticeVector(uvw, self)

    def reciprocal(self):
        reciprocal_lattice_parameters = reciprocalise(self.lattice_parameters)
        return ReciprocalLattice(reciprocal_lattice_parameters)


class ReciprocalLattice(AbstractLattice):
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
    def from_cif(cls, filepath, data_block=None):
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

    def vector(self, hkl):
        return ReciprocalLatticeVector(hkl, self)

    def direct(self):
        direct_lattice_parameters = reciprocalise(self.lattice_parameters)
        return DirectLattice(direct_lattice_parameters)


def check_lattice(operation):
    @wraps(operation)
    def wrapper(self, other):
        if self.lattice != other.lattice:
            raise TypeError("lattice must be the same for both "
                            "{0}s".format(self.__class__.__name__))
        else:
            return operation(self, other)

    return wrapper


class DirectLatticeVector(np.ndarray):  # TODO: Finish docstrings
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

    """

    def __new__(cls, uvw, lattice):
        vector = np.asarray(uvw).view(cls)
        vector.lattice = lattice
        return vector

    def __array_finalize__(self, vector):
        self.lattice = getattr(vector, "lattice", None)

    def __eq__(self, other):
        return np.array_equal(self, other) and self.lattice == other.lattice

    def __ne__(self, other):
        return not self == other

    @check_lattice
    def __add__(self, other):
        return np.ndarray.__add__(self, other)

    @check_lattice
    def __sub__(self, other):
        return np.ndarray.__sub__(self, other)

    def norm(self):
        """Calculate the norm (or magnitude) of the vector

        Returns
        -------
        float:
            The norm of the vector.
        """

        return np.sqrt(self.dot(self.lattice.metric).dot(self))

    def inner(self, other):
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

    def angle(self, other):
        u, v = self, other
        return np.degrees(np.arccos(u.inner(v) / (u.norm() * v.norm())))


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

    def __new__(cls, hkl, lattice):
        vector = np.asarray(hkl).view(cls)
        vector.lattice = lattice
        return vector

    # TODO: add copies of functions so docstrings aren't inherited using super

    def inner(self, other):
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
