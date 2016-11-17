from numpy import array, allclose

from .cif.helpers import CIF_NAMES, get_cif_data, load_data_block
from .lattice import DirectLattice

__all__ = ["Crystal", "Site"]


class Site:
    """Class to represent an atomic site

    Parameters
    ----------
    ion: str
    position: seq
    precision: float, optional

    Attributes
    ----------
    ion: str
    position: array_like
    precision: float

    """

    def __init__(self, ion, position, precision=1E-6):
        self.ion = ion
        self.position = position
        self.precision = precision

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._position = array(new_position)

    def __repr__(self):
        return "{0}({1.ion!r}, {1.position!r})".format(self.__class__.__name__, self)

    def __eq__(self, other):
        return (self.ion == other.ion and
                allclose(self.position, other.position, atol=self.precision))


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
    atoms: dict
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
        self.sites = {}

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
        crystal = cls.__new__(cls)
        crystal.lattice = DirectLattice.from_dict(input_dict)
        try:
            crystal.space_group = input_dict["space_group"]
        except KeyError:
            raise ValueError("Parameter: 'space_group' missing from "
                             "input dictionary")
        if "sites" in input_dict:
            crystal.sites = {}
            crystal.add_sites(input_dict["sites"])
        return crystal

    @classmethod
    def from_cif(cls, filepath, data_block=None, load_sites=True):
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
        crystal = cls.__new__(cls)
        lattice = DirectLattice.from_cif(filepath, data_block)
        data_items = load_data_block(filepath, data_block)
        [space_group] = get_cif_data(data_items, CIF_NAMES["space_group"])
        crystal.lattice, crystal.space_group = lattice, space_group
        if load_sites:
            crystal.sites = {}
            crystal.add_sites_from_cif(filepath, data_block)
        return crystal

    def __repr__(self):
        repr_string = ("{0}([{1.a!r}, {1.b!r}, {1.c!r}, "
                       "{1.alpha!r}, {1.beta!r}, {1.gamma!r}], "
                       "{1.space_group!r})")
        return repr_string.format(self.__class__.__name__, self)

    def __str__(self):
        return repr(self)

    def __getattr__(self, name):  # TODO: Only delegate access for certain variables
        return getattr(self.lattice, name)

    def add_sites_from_cif(self, filepath, data_block=None):
        data_items = load_data_block(filepath, data_block)
        atomic_site_data = get_cif_data(data_items,
                                        "atom_site_label",
                                        "atom_site_type_symbol",
                                        "atom_site_fract_x",
                                        "atom_site_fract_y",
                                        "atom_site_fract_z")
        for label, element, *position in zip(*atomic_site_data):
            self.sites[label] = Site(element, position)

    def add_sites(self, atoms):  # TODO: Finish docstring
        """Add atomic site to crystal

        Parameters
        ----------
        atoms: dict
            A dictionary of atomic positions

        """
        for name, (element, position) in atoms.items():
            self.sites[name] = Site(element, position)
