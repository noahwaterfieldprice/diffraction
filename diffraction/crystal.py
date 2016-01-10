import re

from . import load_cif

NUMERICAL_PARAMETERS = {"cell_length_a": "a",
                        "cell_length_b": "b",
                        "cell_length_c": "c",
                        "cell_angle_alpha": "alpha",
                        "cell_angle_beta": "beta",
                        "cell_angle_gamma": "gamma"}

TEXTUAL_PARAMETERS = {"symmetry_space_group_name_H-M": "space_group"}

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
            The :term:`data block` to load the data from. Only
            required when the input :term:`CIF` has multiple data
            blocks.

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


def numerical_data_value(data_name, data_items):
    """Get the numerical value of a :term:`data value` extracted from
    a :term:`CIF`.

    The numerical data value is matched to the pattern #.#(#), where
    # signifies one or more digits and the decimal points and error
    are optional. If present, the error is stripped off and the
    remaining string is converted to a float.

    Parameters
    ----------
    data_name: str
        The corresponding :term:`data name` of the :term:`data
        value`.
    data: dict
        A dictionary of :term:`data items` for a single :term:`data
        block` extracted from a :term:`CIF`.

    Raises
    ------
    ValueError:
        If the corresponding :term:`data value` is not valid
        numerical data.
    ValueError:
        If `data` does not contain a :term:`data item` with the
        corresponding :term:`data name`.

    Returns
    -------
    float:
        The numerical value of :term:`data value`
    """
    try:
        data_value = data_items[data_name]
        if not CIF_NUMERICAL.match(data_value):
            raise ValueError("Invalid lattice parameter {0}: {1}".format(
                data_name, data_value))
    except KeyError:
        raise ValueError(
            "{} missing from input CIF file".format(data_name))
    else:
        value = float(CIF_NUMERICAL.match(data_value).group(1))
    return value


def textual_data_value(data_name, data_items):
    """Get the string of a :term:`data value` extracted from
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
    filepath: str
        Filepath to the input :term:`CIF`.
    data_block: str
            The :term:`data block` to generate the Crystal from,
            specified by :term:`data block header`. Only required
            when the input :term:`CIF` has multiple data blocks.

    Attributes
    ----------
    a, b, c: float
        The *a*, *b* and *c* lattice parameters.
    alpha, beta, gamma: float
        The *alpha*, *beta* and *gamma* lattice parameters.

    Raises
    ------
    TypeError:
        If the input CIF has multiple data blocks but data_block is
        not given.

    Notes
    -----

    Examples
    --------



    """
    def __init__(self, filepath, data_block=None):
        data = load_data_block(filepath, data_block)
        for data_name, attr in NUMERICAL_PARAMETERS.items():
            value = numerical_data_value(data_name, data)
            setattr(self, attr, value)
        for data_name, attr in TEXTUAL_PARAMETERS.items():
            value = textual_data_value(data_name, data)
            setattr(self, attr, value)
