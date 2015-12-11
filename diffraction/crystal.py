import re

from . import load_cif

__all__ = ["Crystal"]

LATTICE_PARAMETERS = {"cell_length_a": "a",
                      "cell_length_b": "b",
                      "cell_length_c": "c",
                      "cell_angle_alpha": "alpha",
                      "cell_angle_beta": "beta",
                      "cell_angle_gamma": "gamma"}


CIF_NUMERICAL = re.compile("(\d+\.?\d*)(?:\(\d+\))?$")


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
        not specified.

    Notes
    -----

    Examples
    --------



    """
    def __init__(self, filepath, data_block=None):  # TODO: Refactor to methods
        cif_data = load_cif(filepath)
        if len(cif_data) == 1:
            [(_, data)] = cif_data.items()
        else:
            if data_block is None:
                raise TypeError(
                    "__init__() missing keyword argument: 'data_block'. "
                    "Required when input CIF has multiple data blocks.")
            else:
                data = cif_data[data_block]

        for data_name, attr in LATTICE_PARAMETERS.items():
            try:
                data_value = data[data_name]
                if not CIF_NUMERICAL.match(data_value):
                    raise ValueError(
                        "Invalid lattice parameter {0}: {1}".format(data_name,
                                                                    data_value)
                    )
            except KeyError:
                raise ValueError(
                    "{} missing from input CIF file".format(data_name))
            else:
                value = float(CIF_NUMERICAL.match(data_value).group(1))
                setattr(self, attr, value)
