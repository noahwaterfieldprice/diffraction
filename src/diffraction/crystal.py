from collections.abc import Sequence
from typing import Any

import numpy as np

from . import lattice as lattice_module
from .cif import helpers as cif_helpers

LatticeParameters, Position = Sequence[float], Sequence[float]

__all__ = ["Crystal", "Site"]


class Site:
    """Class to represent an atomic site

    Parameters
    ----------
    ion
        A string denoting ion at the site, e.g. 'Fe3+', 'O2-'
    position
        A sequence of the form (x, y, z) denoting position of the site
        with x, y, z given in :term:`fractional coordinates`.
    precision
        A number representing the precision with which the site is
        positioned. This is used when deciding if the position of two
        sites are the same.

    Attributes
    ----------
    ion: str
    position: array_like
    precision: float

    """

    def __init__(self, ion: str, position: Position, precision: float = 1e-6) -> None:
        self.ion = ion
        self.position = position
        self.precision = precision

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, new_position: Position) -> None:
        self._position = np.array(new_position)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.ion!r}, {self.position!r})"

    def __eq__(self, other: "Site") -> bool:
        return self.ion == other.ion and np.allclose(
            self.position, other.position, atol=self.precision
        )


class Crystal:
    """Represents a crystal structure with lattice parameters and atomic sites.

    A ``Crystal`` wraps a `DirectLattice` with a space group and a set
    of atomic `Site` positions in fractional coordinates. Lattice
    parameter attributes (``a``, ``b``, ``c``, etc.) are delegated to
    the underlying `DirectLattice`.

    Parameters
    ----------
    lattice_parameters: seq of float
        The lattice parameters of the crystal declared in the order
        [*a*, *b*, *c*, *alpha*, *beta*, *gamma*], with angles in
        degrees.
    space_group: str
        The Hermann-Mauguin symbol of the space group.

    Attributes
    ----------
    lattice: DirectLattice
        The direct lattice of the crystal.
    space_group: str
        The Hermann-Mauguin symbol of the space group.
    sites: dict
        A dictionary of atomic sites keyed by label.

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
        self.lattice = lattice_module.DirectLattice(lattice_parameters)
        self.space_group = space_group
        self.sites = {}

    @classmethod
    def from_dict(cls, input_dict: dict[str, float]) -> "Crystal":
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
        crystal.lattice = lattice_module.DirectLattice.from_dict(input_dict)
        try:
            crystal.space_group = input_dict["space_group"]
        except KeyError as exc:
            raise ValueError(
                "Parameter: 'space_group' missing from input dictionary"
            ) from exc
        if "sites" in input_dict:
            crystal.sites = {}
            crystal.add_sites(input_dict["sites"])
        return crystal

    @classmethod
    def from_cif(
        cls,
        filepath: str,
        data_block: str | None = None,
        load_sites: bool = True,
    ) -> "Crystal":
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
        lattice = lattice_module.DirectLattice.from_cif(filepath, data_block)
        data_items = cif_helpers.load_data_block(filepath, data_block)
        [space_group] = cif_helpers.get_cif_data(
            data_items, cif_helpers.CIF_NAMES["space_group"]
        )
        crystal.lattice, crystal.space_group = lattice, space_group
        if load_sites:
            crystal.sites = {}
            crystal.add_sites_from_cif(filepath, data_block)
        return crystal

    def __repr__(self) -> str:
        repr_string = (
            "{0}([{1.a!r}, {1.b!r}, {1.c!r}, "
            "{1.alpha!r}, {1.beta!r}, {1.gamma!r}], "
            "{1.space_group!r})"
        )
        return repr_string.format(self.__class__.__name__, self)

    def __getattr__(
        self, name: str
    ) -> Any:  # TODO: Only delegate access for certain variables
        return getattr(self.lattice, name)

    def add_sites_from_cif(self, filepath: str, data_block: str | None = None) -> None:
        data_items = cif_helpers.load_data_block(filepath, data_block)
        atomic_site_data = cif_helpers.get_cif_data(
            data_items,
            "atom_site_label",
            "atom_site_type_symbol",
            "atom_site_fract_x",
            "atom_site_fract_y",
            "atom_site_fract_z",
        )
        # zip with strict=False: all columns come from same loop so they are
        # equal length by construction; unpacking with *position uses the tail
        for label, element, *position in zip(*atomic_site_data, strict=False):
            self.sites[label] = Site(element, position)

    def add_sites(self, atoms: dict[str, Position]) -> None:
        """Add multiple atomic sites to the crystal.

        Parameters
        ----------
        atoms: dict
            A mapping of ``{label: (element, position)}`` where
            *element* is a string and *position* is a sequence of
            three fractional coordinates.
        """
        for name, (element, position) in atoms.items():
            self.sites[name] = Site(element, position)
