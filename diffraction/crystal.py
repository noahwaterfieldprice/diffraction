import re

from . import load_cif


NUMERICAL_DATA_NAMES = (
    "cell_length_a", "cell_length_b", "cell_length_c",
    "cell_angle_alpha", "cell_angle_beta", "cell_angle_gamma")
NUMERICAL_ATTRIBUTES = ("a", "b", "c", "alpha", "beta", "gamma")

TEXTUAL_DATA_NAMES = ("symmetry_space_group_name_H-M", )
TEXTUAL_ATTRIBUTES = ("space_group", )

# Regular expressions for cif data value matching
CIF_NUMERICAL = re.compile("(\d+\.?\d*)(?:\(\d+\))?$")
CIF_TEXTUAL = re.compile("\'(.*?)\'")


def load_data_block(filepath, data_block=None):
    """Extract the :term:`data items` of a specific :term:`data
    block` from a :term:`CIF`.

    For a multiple data block CIF, the data items of the data block
    given by `data_block` (specified by :term:`data block header`)
    are returned as dictionary. An exception is raised if the data
    block is not specified.

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


def numerical_parameter(key, data):
    """Get the value of a numerical parameter from an input
    dictionary"""
    try:
        value = float(data[key])
    except ValueError:
        raise ValueError("Invalid numerical parameter {0}: {1}".format(
            key, data[key]))
    except KeyError:
        raise ValueError(
            "{} missing from input dictionary".format(key))
    else:
        return value


def cif_numerical_parameter(data_name, data_items):
    """Get the value of a numerical :term:`data value` extracted from
    a :term:`CIF`.

    The numerical data value is matched to the pattern #.#(#), where
    # signifies one or more digits and the decimal points and error
    are optional. If present, the error is stripped off and the
    remaining string is converted to a float.

    """
    try:
        data_value = data_items[data_name]
        if not CIF_NUMERICAL.match(data_value):
            raise ValueError("Invalid numerical parameter {0}: {1}".format(
                data_name, data_value))
    except KeyError:
        raise ValueError(
            "{} missing from input CIF file".format(data_name))
    else:
        value = float(CIF_NUMERICAL.match(data_value).group(1))
    return value


def textual_parameter(key, data):
    """Get the value of a numerical parameter from an input
    dictionary"""
    try:
        value = data[key]
    except KeyError:
        raise ValueError(
            "{} missing from input dictionary".format(key))
    else:
        return value


def cif_textual_parameter(data_name, data_items):
    """Get the value of a textual :term:`data value` extracted from
    a :term:`CIF`."""
    try:
        data_value = data_items[data_name]
    except KeyError:
        raise ValueError(
            "{} missing from input CIF file".format(data_name))
    else:
        value = CIF_TEXTUAL.match(data_value).group(1)
        return value


class Crystal:  # TODO: Finish Docstring
    """Class to represent Crystal

    Parameters
    ----------
    crystal_data: str or dict
        Filepath to the input :term:`CIF` or dictionary of parameters
    data_block: str, optional
            The :term:`data block` to generate the Crystal from,
            specified by :term:`data block header`. Only giving a
            :term:`CIF` with multiple data blocks as input.

    Attributes
    ----------
    a, b, c: float
        The *a*, *b* and *c* lattice parameters.
    alpha, beta, gamma: float
        The *alpha*, *beta* and *gamma* lattice parameters.
    space_group: str
        The space group of the crystal structure.

    Raises
    ------
    TypeError:
        If the input CIF has multiple data blocks but data_block is
        not given.

    Notes
    -----

    Examples
    --------

    Crystal can be created from CIF data

    >>> from diffraction import Crystal, load_cif
    >>> calcite_data = load_cif("calcite.cif")
    >>> data_items = calcite_data["data_calcite"]
    >>> calcite = Crystal(data_items)
    >>> calcite.a
    4.99
    >>> calcite.gamma
    120.0
    >>> calcite.space_group
    'R -3 c H'

    Or from a manually created dictionary

    >>> calcite_parameters = {
        "a": 4.99, "b": 4.99, "c": 17.002,
        "alpha": 90, "beta": 90, "gamma": 120,
        "space_group": "R -3 c H"}
    >>> calcite = Crystal(calcite_parameters)
    >>> calcite.a
    4.99
    >>> calcite.gamma
    120.0
    >>> calcite.space_group
    'R -3 c H'



    """
    def __init__(self, crystal_data, data_block=None):
        if isinstance(crystal_data, str):
            self._instantiate_from_cif(crystal_data, data_block)
        elif isinstance(crystal_data, dict):
            self._instantiate_from_dict(crystal_data)

    def _instantiate_from_cif(self, filepath, data_block):
        data = load_data_block(filepath, data_block)
        for data_name, attr in zip(NUMERICAL_DATA_NAMES, NUMERICAL_ATTRIBUTES):
            value = cif_numerical_parameter(data_name, data)
            setattr(self, attr, value)
        for data_name, attr in zip(TEXTUAL_DATA_NAMES, TEXTUAL_ATTRIBUTES):
            value = cif_textual_parameter(data_name, data)
            setattr(self, attr, value)

    def _instantiate_from_dict(self, data):
        for attribute in NUMERICAL_ATTRIBUTES:
            value = numerical_parameter(attribute, data)
            setattr(self, attribute, value)
        for attribute in TEXTUAL_ATTRIBUTES:
            value = textual_parameter(attribute, data)
            setattr(self, attribute, value)
