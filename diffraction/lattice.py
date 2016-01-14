from .cif import cif_numerical, load_data_block
__all__ = ["DirectLattice"]

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


class DirectLattice:
    def __init__(self, lattice_parameters):
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
        for parameter in LATTICE_PARAMETERS:
            try:
                lattice_parameters.append(input_dict[parameter])
            except KeyError:
                raise ValueError(
                    "Lattice parameter {0} missing from input "
                    "dictionary".format(parameter))
        return cls(lattice_parameters)

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
        # TODO: write function to retrieve parameters from CIF input
        data_items = load_data_block(filepath, data_block)
        lattice_parameters = []
        for parameter in LATTICE_PARAMETERS:
            data_name = CIF_NAMES[parameter]
            try:
                data_value = data_items[data_name]
                lattice_parameters.append(cif_numerical(data_name, data_value))
            except KeyError:
                raise ValueError("Lattice parameter {0} missing from input "
                                 "CIF".format(data_name))
        return cls(lattice_parameters)

    def __repr__(self):
        repr_string = ("{0}([{1.a!r}, {1.b!r}, {1.c!r}, "
                       "{1.alpha!r}, {1.beta!r}, {1.gamma!r}])")
        return repr_string.format(self.__class__.__name__, self)

    def __str__(self):
        return repr(self)
