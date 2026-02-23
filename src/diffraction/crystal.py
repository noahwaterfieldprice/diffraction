"""Crystal structure and atomic site representation.

Provide the Crystal and Site classes for building crystal structures from
lattice parameters, space group symbols, and fractional-coordinate atomic
positions. Crystal structures can be constructed from explicit parameters,
dictionaries, or CIF files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from . import lattice as lattice_module
from .cif import helpers as cif_helpers

LatticeParameters: TypeAlias = Sequence[float]
Position: TypeAlias = Sequence[float]

__all__ = ["Crystal", "Site"]

# Lattice parameter attribute names delegated by Crystal.__getattr__.
_LATTICE_PARAMETER_ATTRS: frozenset[str] = frozenset(
    {"a", "b", "c", "alpha", "beta", "gamma"}
)


@dataclass(eq=False)
class Site:
    """Atomic site at a fractional-coordinate position in a crystal.

    Store the ion label, fractional coordinates, and a positional precision
    used when comparing two sites for equality. The ``position`` attribute is
    always stored as a numpy array regardless of the type passed at
    construction or assignment.

    Args:
        ion: Ion label string, e.g. ``'Fe3+'``, ``'O2-'``.
        position: Fractional coordinates (x, y, z) of the site.
        precision: Tolerance used when comparing site positions for
            equality. Defaults to 1e-6.

    Attributes:
        ion: Ion label string.
        position: Fractional coordinates as a numpy array.
        precision: Positional equality tolerance.

    Examples:
        Create a calcium site and inspect its coordinates:

        >>> from diffraction import Site
        >>> ca = Site('Ca2+', [0.0, 0.0, 0.0])
        >>> ca.ion
        'Ca2+'
        >>> ca.position
        array([0., 0., 0.])
    """

    ion: str
    position: Position
    precision: float = 1e-6

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=np.float64)  # type: ignore[assignment]

    def __setattr__(self, name: str, value: object) -> None:
        if name == "position":
            value = np.array(value, dtype=np.float64)
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.ion!r}, {self.position!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Site):
            return NotImplemented
        return self.ion == other.ion and bool(
            np.allclose(
                cast(NDArray[np.float64], self.position),
                cast(NDArray[np.float64], other.position),
                atol=self.precision,
            )
        )


class Crystal:
    """Crystal structure combining a direct lattice, space group, and atomic sites.

    Wrap a DirectLattice with a space group symbol and a dictionary of atomic
    Site objects at fractional-coordinate positions. The six standard lattice
    parameters (``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``) are
    delegated to the underlying DirectLattice via ``__getattr__``. For other
    lattice properties (``metric``, ``unit_cell_volume``, etc.), access them
    via ``crystal.lattice``.

    Args:
        lattice_parameters: Six lattice parameters in the order
            (a, b, c, alpha, beta, gamma) with angles in degrees.
        space_group: Hermann-Mauguin symbol of the space group.

    Attributes:
        lattice: The underlying DirectLattice.
        space_group: Hermann-Mauguin symbol of the space group.
        sites: Dictionary mapping site label strings to Site objects.

    Examples:
        Create a calcite crystal and access its properties:

        >>> from diffraction import Crystal
        >>> calcite = Crystal([4.99, 4.99, 17.002, 90, 90, 120], "R -3 c H")
        >>> calcite.a
        4.99
        >>> calcite.gamma
        120.0
        >>> calcite.space_group
        'R -3 c H'
    """

    def __init__(self, lattice_parameters: LatticeParameters, space_group: str) -> None:
        self.lattice = lattice_module.DirectLattice(lattice_parameters)
        self.space_group = space_group
        self.sites: dict[str, Site] = {}

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> Crystal:
        """Create a Crystal from a parameter dictionary.

        Args:
            input_dict: Mapping containing all six lattice parameter keys
                (``'a'``, ``'b'``, ``'c'``, ``'alpha'``, ``'beta'``,
                ``'gamma'``), a ``'space_group'`` key, and optionally a
                ``'sites'`` key with a ``{label: (element, position)}``
                mapping.

        Returns:
            A Crystal populated from the dictionary.

        Raises:
            ValueError: If any required lattice parameter or space_group
                key is missing from the dictionary.

        Examples:
            >>> from diffraction import Crystal
            >>> params = {
            ...     "a": 4.99, "b": 4.99, "c": 17.002,
            ...     "alpha": 90, "beta": 90, "gamma": 120,
            ...     "space_group": "R -3 c H",
            ... }
            >>> calcite = Crystal.from_dict(params)
            >>> calcite.a
            4.99
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
    ) -> Crystal:
        """Create a Crystal from a CIF file.

        Args:
            filepath: Path to the input CIF file.
            data_block: Data block header to read from, required only when
                the CIF contains multiple data blocks.
            load_sites: Whether to load atomic site data from the CIF.
                Defaults to True.

        Returns:
            A Crystal populated with the lattice, space group, and (if
            requested) atomic sites from the CIF.

        Raises:
            ValueError: If any required lattice parameter or space group is
                missing from the CIF, or if any parameter is not valid
                numerical data.
            TypeError: If the CIF has multiple data blocks but data_block
                is not given.

        Examples:
            >>> from diffraction import Crystal
            >>> calcite = Crystal.from_cif("calcite.cif")
            >>> calcite.a
            4.99
            >>> calcite.space_group
            'R -3 c H'
        """
        crystal = cls.__new__(cls)
        lattice = lattice_module.DirectLattice.from_cif(filepath, data_block)
        data_items = cif_helpers.load_data_block(filepath, data_block)
        [space_group] = cif_helpers.get_cif_data(
            data_items, cif_helpers.CIF_NAMES["space_group"]
        )
        crystal.lattice, crystal.space_group = lattice, cast(str, space_group)
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

    def __getattr__(self, name: str) -> float:
        """Delegate lattice parameter access to the underlying DirectLattice.

        Provides transparent access to the six standard lattice parameters
        (a, b, c, alpha, beta, gamma) via the attached lattice. All other
        attributes raise AttributeError.

        Args:
            name: Attribute name to look up. Must be one of
                ``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``.

        Returns:
            The lattice parameter value as a float.

        Raises:
            AttributeError: If name is not a delegated lattice parameter.
        """
        if name in _LATTICE_PARAMETER_ATTRS:
            return getattr(self.lattice, name)  # type: ignore[no-any-return]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def add_sites_from_cif(self, filepath: str, data_block: str | None = None) -> None:
        """Load and add atomic sites from a CIF file.

        Read atom_site loop data from the CIF and populate ``self.sites``
        with Site objects keyed by atom site label.

        Args:
            filepath: Path to the input CIF file.
            data_block: Data block header to read from, required only when
                the CIF contains multiple data blocks.
        """
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
            self.sites[str(label)] = Site(str(element), position)

    def add_sites(self, atoms: dict[str, tuple[str, Position]]) -> None:
        """Add multiple atomic sites to the crystal.

        Args:
            atoms: Mapping of ``{label: (element, position)}`` where
                *element* is an ion label string and *position* is a
                sequence of three fractional coordinates.
        """
        for name, (element, position) in atoms.items():
            self.sites[name] = Site(element, position)
