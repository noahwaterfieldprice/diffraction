import re
from collections import abc

from . import load_cif

__all__ = ["Crystal"]

# Lattice parameter names and map to CIF data names
LATTICE_PARAMETERS = ["a", "b", "c", "alpha", "beta", "gamma"]
CIF_NAMES = {
    "a": "cell_length_a",
    "b": "cell_length_b",
    "c": "cell_length_c",
    "alpha": "cell_angle_alpha",
    "beta": "cell_angle_beta",
    "gamma": "cell_angle_gamma",
    "space_group": "symmetry_space_group_name_H-M"
}

# Regular expressions for cif data value matching
CIF_NUMERICAL = re.compile("(\d+\.?\d*)(?:\(\d+\))?$")
CIF_TEXTUAL = re.compile("\'(.*?)\'")


def load_data_block(filepath, data_block=None):
    """Extract the :term:`data items` of a specific :term:`data
    block` from a :term:`CIF`.

    For a multiple data block CIF, the data items of the data block given
    by `data_block` (specified by :term:`data block header`) are returned
    as dictionary. An exception is raised if the data block is not given.

    For a CIF with with only a single data block, the data items of
    that data block are returned automatically.


    Parameters
    ----------
    filepath: str
        Filepath to the input :term:`CIF`.
    data_block: str
        The :term:`data block` to load the data from. Only required
        when the input :term:`CIF` has multiple data blocks.

    Raises
    ------
    TypeError:
        If the input CIF has multiple data blocks but data_block is
        not given.


    Returns
    -------
    dict:
        A dictionary of the :term:`data items` of the :term:`data
        block` in :term:`data name`: :term:`data value` pairs.

    """
    cif = load_cif(filepath)
    if len(cif) == 1:
        (_, data), = cif.items()
    else:
        if data_block is None:
            raise TypeError(
                "__init__() missing keyword argument: 'data_block'. "
                "Required when input CIF has multiple data blocks.")
        else:
            data = cif[data_block]
    return data


def get_input_value(key, dictionary, input_type):
    """Retrieve value of input parameter from dictionary. Inform user
    of missing input data if parameter is not found.
    """
    try:
        value = dictionary[key]
    except KeyError:
        raise ValueError(
            "{0} missing from input {1}".format(key, input_type))
    else:
        return value


def cif_numerical(data_name, data_value):
    """Extract numerical :term:`data value` from raw :term:`CIF` data"""
    if not CIF_NUMERICAL.match(data_value):
        raise ValueError("Invalid numerical value in input CIF {0}: {1}".format(
            data_name, data_value))
    return CIF_NUMERICAL.match(data_value).group(1)


def cif_textual(data_value):
    """Extract textual :term:`data value` from raw :term:`CIF` data"""
    value = CIF_TEXTUAL.match(data_value).group(1)
    return value


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
        if len(lattice_parameters) < 6:
            raise(ValueError("Missing lattice parameter from input"))
        for name, value in zip(LATTICE_PARAMETERS, lattice_parameters):
            try:
                v = float(value)
            except ValueError:
                raise ValueError("Invalid lattice parameter {0}: {1}".format(
                    name, value))
            else:
                setattr(self, name, v)
        self.space_group = space_group

    @classmethod
    def from_cif(cls, filepath, data_block=None):
        """Create a Crystal using a CIF file as input

        Parameters
        ----------
        filepath: str
            Filepath to the input :term:`CIF`
        data_block: str, optional
            The :term:`data block` to generate the Crystal from,
            specified by :term:`data block header`. Only giving a
            :term:`CIF` with multiple data blocks as input.

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
            data_value = get_input_value(data_name, data_items,
                                         input_type="CIF file")
            lattice_parameters.append(cif_numerical(data_name, data_value))
        space_group = get_input_value(CIF_NAMES["space_group"], data_items,
                                      input_type="CIF file")
        space_group = cif_textual(space_group)
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
            value = get_input_value(parameter, input_dict,
                                    input_type="dictionary")
            lattice_parameters.append(value)
        space_group = get_input_value("space_group", input_dict,
                                      input_type="dictionary")
        return cls(lattice_parameters, space_group)

    def __repr__(self):
        repr_string = ("{0}([{1.a!r}, {1.b!r}, {1.c!r}, "
                       "{1.alpha!r}, {1.beta!r}, {1.gamma!r}], "
                       "{1.space_group!r})")
        return repr_string.format(self.__class__.__name__, self)
