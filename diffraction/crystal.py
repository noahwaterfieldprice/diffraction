import re

from .cif import CIF_NAMES, cif_numerical, cif_textual, load_data_block
from .lattice import DirectLattice

__all__ = ["Crystal"]

LATTICE_PARAMETERS = ["a", "b", "c", "alpha", "beta", "gamma"]


class Crystal:  # TODO: Finish docstring and update glossary
    """Class to represent Crystal

    Parameters
    ----------
    lattice_parameters: seq of float
        The lattice parameters of the crystal declared in the order
        [*a*, *b*, *c*, *alpha*, *beta*, *gamma*]
    space_group: str
        The :term:`space group` of the crystal structure.

    Attributes
    ----------
    a, b, c: float
        The *a*, *b* and *c* lattice parameters describing the
        dimensions of the :term:`unit cell`.
    alpha, beta, gamma: float
        The *alpha*, *beta* and *gamma* lattice parameters describing
        the angles of the :term:`unit cell`.
    space_group: str
        The :term:`space group` of the crystal structure.

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
    def __init__(self, lattice_parameters, space_group):
        self.lattice = DirectLattice(lattice_parameters)
        self.space_group = space_group

    @classmethod
    def from_cif(cls, filepath, data_block=None):
        """Create a Crystal using a CIF as input

        Parameters
        ----------
        filepath: str
            Filepath to the input :term:`CIF`
        data_block: str, optional
            The :term:`data block` to generate the Crystal from,
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
        lattice_parameters = []
        for parameter in LATTICE_PARAMETERS:
            data_name = CIF_NAMES[parameter]
            try:
                data_value = data_items[data_name]
                lattice_parameters.append(cif_numerical(data_name, data_value))
            except KeyError:
                raise ValueError("Parameter: '{0}' missing from input "
                                 "CIF".format(data_name))
        try:
            space_group = data_items[CIF_NAMES["space_group"]]
            space_group = cif_textual(space_group)
        except KeyError:
            raise ValueError("Parameter: '{0}' missing from input "
                             "CIF".format(CIF_NAMES["space_group"]))

        return cls(lattice_parameters, space_group)

    @classmethod
    def from_dict(cls, input_dict):
        """Create a Crystal using a dictionary as input

        Raises
        ------
        ValueError:
            If the input :term:`CIF` is missing any :term:`lattice
            parameters` or the :term:`space group`
        TypeError:
            If the input CIF has multiple data blocks but data_block is
            not given.

        Examples
        --------
        >>> from diffraction import Crystal
        >>> calcite_parameters = {
        "a": 4.99, "b": 4.99, "c": 17.002,
        "alpha": 90, "beta": 90, "gamma": 120,
        "space_group": "R -3 c H"}
        >>> calcite = Crystal.from_dict(calcite_parameters)
        >>> calcite.a
        4.99
        >>> calcite.gamma
        120.0
        >>> calcite.space_group
        'R -3 c H'
        """

        lattice_parameters = []
        for parameter in LATTICE_PARAMETERS:
            try:
                lattice_parameters.append(input_dict[parameter])
            except KeyError:
                raise ValueError("Parameter: '{0}' missing from input "
                                 "dictionary".format(parameter))
        try:
            space_group = input_dict["space_group"]
        except KeyError:
            raise ValueError("Parameter: 'space_group' missing from input dictionary")
        return cls(lattice_parameters, space_group)

    def __repr__(self):
        repr_string = ("{0}([{1.a!r}, {1.b!r}, {1.c!r}, "
                       "{1.alpha!r}, {1.beta!r}, {1.gamma!r}], "
                       "{1.space_group!r})")
        return repr_string.format(self.__class__.__name__, self)

    def __str__(self):
        return repr(self)

    def __getattr__(self, name):
        return getattr(self.lattice, name)
