from functools import wraps

from numpy import around, array_equal, asarray, cos, ndarray, radians, sqrt
from numpy.linalg import det

from .cif.helpers import CIF_NAMES, get_cif_data, load_data_block

__all__ = ["DirectLattice", "DirectLatticeVector"]

# Lattice parameter names and map to CIF data names
LATTICE_PARAMETER_KEYS = ["a", "b", "c", "alpha", "beta", "gamma"]


class DirectLattice:
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
    lattice_parameters_rad: tuple of float
        The lattice parameters in the form
        (*a*, *b*, *c*, *alpha*, *beta*, *gamma*)
        with angles in radians.
    direct_metric: array_like
        The :term:`metric tensor` of the direct basis.
    unit_cell_volume: float


    Notes
    -----

    Examples
    --------
    >>> from diffraction import Crystal
    >>> calcite = Crystal([4.99, 4.99, 17.002, 90, 90, 120], "R -3 c H")
    >>> calcite.a
    4.99
    >>> calcite.gamma
    120.0
    >>> calcite.space_group
    'R -3 c H'
    """

    def __init__(self, lattice_parameters):
        if len(lattice_parameters) < 6:
            raise(ValueError("Missing lattice parameter from input"))
        for name, value in zip(LATTICE_PARAMETER_KEYS, lattice_parameters):
            try:
                v = float(value)
            except ValueError:
                raise ValueError("Invalid lattice parameter {0}: {1}".format(
                    name, value))
            else:
                setattr(self, name, v)

    @classmethod
    def from_cif(cls, filepath, data_block=None):
        """Create a DirectLattice using a :term:`CIF` as input

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
            parameters` or the :term:`space group`.
        ValueError:
            If any of the :term:`lattice parameters` are not valid
            numerical data.
        TypeError:
            If the input CIF has multiple data blocks but data_block is
            not given.

        Examples
        --------
        >>> from diffraction import Crystal
        >>> calcite = Crystal.from_cif("calcite.cif")
        >>> calcite.a
        4.99
        >>> calcite.gamma
        120.0
        >>> calcite.space_group
        'R -3 c H'
        """

        data_items = load_data_block(filepath, data_block)
        lattice_parameter_names = [CIF_NAMES[key]
                                   for key in LATTICE_PARAMETER_KEYS]
        lattice_parameters = get_cif_data(data_items, *lattice_parameter_names)
        return cls(lattice_parameters)

    @classmethod
    def from_dict(cls, input_dict):
        """Create a DirectLattice using a dictionary as input

        Raises
        ------
        ValueError:
            If the input dict is missing any :term:`lattice parameters`

        Examples
        --------
        >>> from diffraction import DirectLattice
        >>> calcite_lattice = {
        "a": 4.99, "b": 4.99, "c": 17.002,
        "alpha": 90, "beta": 90, "gamma": 120}
        >>> lattice = DirectLattice.from_dict(calcite_parameters)
        >>> lattice.a
        4.99
        >>> lattice.gamma
        120.0
        """

        lattice_parameters = []
        for parameter in LATTICE_PARAMETER_KEYS:
            try:
                lattice_parameters.append(input_dict[parameter])
            except KeyError:
                raise ValueError("Parameter: '{0}' missing from input "
                                 "dictionary".format(parameter))
        return cls(lattice_parameters)

    @property
    def lattice_parameters(self):
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    @property
    def lattice_parameters_rad(self):
        return (self.a, self.b, self.c,
                radians(self.alpha), radians(self.beta), radians(self.gamma))

    @property
    def direct_metric(self):
        a, b, c, al, be, ga = self.lattice_parameters_rad
        tensor = around([[a ** 2, a * b * cos(ga), a * c * cos(be)],
                        [a * b * cos(ga), b ** 2, b * c * cos(al)],
                        [a * c * cos(be), b * c * cos(al), c ** 2]], 10)
        return tensor

    @property
    def unit_cell_volume(self):
        return sqrt(det(self.direct_metric))

    def __repr__(self):
        repr_string = ("{0}([{1.a!r}, {1.b!r}, {1.c!r}, "
                       "{1.alpha!r}, {1.beta!r}, {1.gamma!r}])")
        return repr_string.format(self.__class__.__name__, self)

    def __str__(self):
        return repr(self)


def lattice_check(operation):
    @wraps(operation)
    def wrapper(self, other):
        if self.lattice != other.lattice:
            raise TypeError("lattice must be the same for both "
                            "{0}s".format(self.__class__.__name__))
        else:
            return operation(self, other)
    return wrapper


class DirectLatticeVector(ndarray):

    def __new__(cls, uvw, lattice):
        vector = asarray(uvw).view(cls)
        vector.lattice = lattice
        return vector

    def __array_finalize__(self, vector):
        self.lattice = getattr(vector, "lattice", None)

    def __eq__(self, other):
        return array_equal(self, other) and self.lattice == other.lattice

    def __ne__(self, other):
        return not self == other

    @lattice_check
    def __add__(self, other):
        return ndarray.__add__(self, other)

    @lattice_check
    def __sub__(self, other):
        return ndarray.__sub__(self, other)

    def norm(self):
        return sqrt(self.dot(self.lattice.direct_metric).dot(self))
