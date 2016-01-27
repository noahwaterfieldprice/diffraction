from functools import wraps

from numpy import arccos, cos, radians, sin, sqrt
import numpy as np

from .cif.helpers import CIF_NAMES, get_cif_data, load_data_block

__all__ = ["DirectLattice", "DirectLatticeVector", "ReciprocalLattice"]


def lattice_parameters_to_radians(lattice_parameters):
    lengths = tuple(lattice_parameters)[:3]
    angles_in_radians = tuple(np.radians(angle)
                              for angle in tuple(lattice_parameters)[3:])
    return lengths + angles_in_radians


def lattice_parameters_to_degrees(lattice_parameters):
    lengths = tuple(lattice_parameters)[:3]
    angles_in_degrees = tuple(np.degrees(angle)
                              for angle in tuple(lattice_parameters)[3:])
    return lengths + angles_in_degrees


def metric_tensor(lattice_parameters):
    a, b, c, al, be, ga = lattice_parameters_to_radians(lattice_parameters)
    tensor = np.around([[a ** 2, a * b * cos(ga), a * c * cos(be)],
                        [a * b * cos(ga), b ** 2, b * c * cos(al)],
                        [a * c * cos(be), b * c * cos(al), c ** 2]],
                       10)
    return tensor


def transform_lattice_parameters(lattice_parameters):
    a, b, c, al, be, ga = lattice_parameters_to_radians(lattice_parameters)
    cell_volume = sqrt(np.linalg.det(metric_tensor(lattice_parameters)))

    a_ = b * c * sin(al) / cell_volume
    b_ = a * c * sin(be) / cell_volume
    c_ = a * b * sin(ga) / cell_volume
    alpha_ = arccos((cos(be) * cos(ga) - cos(al)) / (sin(be) * sin(ga)))
    beta_ = arccos((cos(al) * cos(ga) - cos(be)) / (sin(al) * sin(ga)))
    gamma_ = arccos((cos(al) * cos(be) - cos(ga)) / (sin(al) * sin(be)))

    return lattice_parameters_to_degrees((a_, b_, c_, alpha_, beta_, gamma_))


class AbstractLattice:
    """Parent class to represent lattice object.

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
    lattice_parameters_rad: tuple of float
        The lattice parameters in the form
        (*a*, *b*, *c*, *alpha*, *beta*, *gamma*)
        with angles in radians.
    metric: array_like
        The :term:`metric tensor` of the direct basis.
    unit_cell_volume: float

    """
    lattice_parameter_keys = None

    def __init__(self, lattice_parameters):
        lattice_parameters = self.convert_parameters(lattice_parameters)
        for key, value in zip(self.lattice_parameter_keys, lattice_parameters):
            setattr(self, key, value)

    def convert_parameters(self, lattice_parameters):
        if len(lattice_parameters) < 6:
            raise(ValueError("Missing lattice parameter from input"))
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
            except KeyError:
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
    """Parent class to represent lattice object.

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
    lattice_parameters: list of float
        The :term:`lattice parameters` in the form
        (*a*, *b*, *c*, *alpha*, *beta*, *gamma*)
        with angles in degrees.
    lattice_parameters_rad: tuple of float
        The lattice parameters in the form
        (*a*, *b*, *c*, *alpha*, *beta*, *gamma*)
        with angles in radians.
    metric: array_like
        The :term:`metric tensor` of the direct basis.
    unit_cell_volume: float

    """
    lattice_parameter_keys = ("a", "b", "c", "alpha", "beta", "gamma")

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
        data_names = [CIF_NAMES[key] for key in "a b c alpha beta gamma".split()]
        lattice_parameters = get_cif_data(data_items, *data_names)
        return cls(lattice_parameters)

    def vector(self, uvw):
        return DirectLatticeVector(uvw, self)

    def reciprocal(self):
        reciprocal_lps = transform_lattice_parameters(self.lattice_parameters)
        return ReciprocalLattice(reciprocal_lps)


class ReciprocalLattice(AbstractLattice):
    """Class to asdoiapidjas
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
        data_names = [CIF_NAMES[key] for key in "a b c alpha beta gamma".split()]
        lattice_parameters = get_cif_data(data_items, *data_names)
        reciprocal_lps = transform_lattice_parameters(lattice_parameters)
        return cls(reciprocal_lps)

    # def vector(self, hkl):
    #     return ReciprocalLatticeVector(hkl, self)

    def direct(self, hkl):
        direct_lps = transform_lattice_parameters(self.lattice_parameters)
        return DirectLattice(direct_lps)


def lattice_check(operation):
    @wraps(operation)
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
        The u, v
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
        Calculate the inner product of the vector with another lattice
        vector.
    angle:
        Calculate the angle between the vector and another lattice
        vector.

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

    @lattice_check
    def __add__(self, other):
        return np.ndarray.__add__(self, other)

    @lattice_check
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
        """Calculate the inner product between the vector and another
        lattice vector

        Parameters
        ----------
        other:
            The

        Returns
        -------

        """
        return self.dot(self.lattice.metric).dot(other)

    def angle(self, other):
        u, v = self, other
        return np.degrees(np.arccos(u.inner(v) / (u.norm() * v.norm())))
