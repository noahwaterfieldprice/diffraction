"""CIF data extraction helpers and name mappings.

Provide utility functions for retrieving typed data from parsed CIF
dictionaries, and define the lookup tables that map between diffraction
object attribute names and CIF data names.
"""

import re
from typing import cast

from .cif import load_cif

# CIF data names whose values should be parsed as floating-point numbers.
# The parser stores all raw values as strings; functions in this module
# strip uncertainty suffixes (e.g. "3.456(7)" -> 3.456) before converting.
NUMERICAL_DATA_NAMES = (  # TODO: strip down to only data names used
    "atom_site_fract_x",
    "atom_site_fract_y",
    "atom_site_fract_z",
    "atom_site_B_iso_or_equiv",
    "atom_site_Wyckoff_symbol",
    "atom_site_aniso_U_11",
    "atom_site_aniso_U_12",
    "atom_site_aniso_U_13",
    "atom_site_aniso_U_22",
    "atom_site_aniso_U_23",
    "atom_site_aniso_U_33",
    "atom_site_attached_hydrogens",
    "atom_site_occupancy",
    "atom_site_symmetry_multiplicity",
    "atom_type_oxidation_number",
    "atom_type_radius_bond",
    "cell_angle_alpha",
    "cell_angle_beta",
    "cell_angle_gamma",
    "cell_formula_units_Z",
    "cell_length_a",
    "cell_length_b",
    "cell_length_c",
    "cell_volume",
    "citation_journal_volume",
    "citation_page_first",
    "citation_page_last",
    "citation_year",
    "cod_database_code",
    "database_code_ICSD",
    "diffrn_ambient_temperature",
    "exptl_crystal_density_diffrn",
    "exptl_crystal_density_meas",
    "refine_ls_R_factor_all",
    "refine_ls_R_factor_gt",
    "refine_ls_wR_factor_gt",
    "symmetry_Int_Tables_number",
    "symmetry_equiv_pos_site_id",
)

# CIF data names whose values are returned as raw strings without conversion.
TEXTUAL_DATA_NAMES = (
    "atom_site_aniso_label",
    "atom_site_calc_flag",
    "atom_site_label",
    "atom_site_type_symbol",
    "atom_type_symbol",
    "audit_creation_date",
    "audit_creation_method",
    "audit_update_record",
    "chemical_compound_source",
    "chemical_formula_moiety",
    "chemical_formula_structural",
    "chemical_formula_sum",
    "chemical_name_mineral",
    "chemical_name_structure",
    "chemical_name_systematic",
    "citation_id",
    "citation_journal_full",
    "database_code_CSD",
    "database_code_depnum_ccdc_archive",
    "exptl_crystal_colour",
    "exptl_crystal_description",
    "exptl_special_details",
    "journal_coden",
    "journal_coden_ASTM",
    "journal_name_full",
    "pd_phase_name",
    "publ_author_name",
    "publ_section_title",
    "refine_special_details",
    "symmetry_cell_setting",
    "symmetry_equiv_pos_as_xyz",
    "symmetry_space_group_name_H-M",
    "symmetry_space_group_name_Hall",
)

# Mapping from diffraction object parameter names to CIF data names.
# Used by lattice and crystal constructors to look up CIF values by
# the standard attribute name (e.g. 'a' -> 'cell_length_a').
CIF_NAMES = {
    "a": "cell_length_a",
    "b": "cell_length_b",
    "c": "cell_length_c",
    "alpha": "cell_angle_alpha",
    "beta": "cell_angle_beta",
    "gamma": "cell_angle_gamma",
    "space_group": "symmetry_space_group_name_H-M",
}

NUMERICAL_DATA_VALUE = re.compile(r"(-?\d+\.?\d*)(?:\(\d+\))?$")


def load_data_block(
    filepath: str, data_block: str | None = None
) -> dict[str, str | list[str]]:
    """Extract data items from a specific data block in a CIF file.

    For a CIF with a single data block, return its data items directly.
    For a multi-block CIF, return the data items of the block identified
    by ``data_block``.

    Args:
        filepath: Path to the CIF file.
        data_block: Data block header string (e.g. ``'data_calcite'``),
            required when the CIF contains more than one data block.

    Returns:
        Dictionary mapping data name strings to their extracted values
        (str for scalar items, list[str] for loop items).

    Raises:
        TypeError: If the CIF has multiple data blocks but ``data_block``
            is not provided.
    """
    cif = load_cif(filepath)
    if len(cif) == 1:
        ((_, data),) = cif.items()
    else:
        if data_block is None:
            raise TypeError(
                "__init__() missing keyword argument: 'data_block'. "
                "Required when input CIF has multiple data blocks."
            )
        else:
            data = cif[data_block]
    return data


def get_cif_data(
    data_items: dict[str, str | list[str]], *data_names: str
) -> list[str | float | list[str] | list[float]]:
    """Retrieve data values from a parsed CIF data items dictionary.

    Look up each requested data name in ``data_items`` and return the
    values in the same order. Numerical data names (those in
    ``NUMERICAL_DATA_NAMES``) have their uncertainty suffixes stripped
    and are converted to float.

    Args:
        data_items: Dictionary mapping CIF data name strings to their
            extracted values (str or list[str]).
        *data_names: One or more CIF data name strings to retrieve.

    Returns:
        List of data values in the same order as ``data_names``. Each
        value is a float or list[float] for numerical data names, and a
        str or list[str] for textual data names.

    Raises:
        ValueError: If any requested data name is not present in
            ``data_items``.
        ValueError: If a numerical data value cannot be parsed as a float.
    """
    data: list[str | float | list[str] | list[float]] = []
    for data_name in data_names:
        try:
            data_value = data_items[data_name]
        except KeyError as exc:
            raise ValueError(
                f"Parameter: '{data_name}' missing from input CIF"
            ) from exc
        if data_name in NUMERICAL_DATA_NAMES:
            data.append(cif_numerical(data_name, data_value))
        else:
            data.append(data_value)
    return data


def cif_numerical(data_name: str, data_value: str | list[str]) -> float | list[float]:
    """Parse a numerical CIF data value, stripping any uncertainty suffix.

    Match the value against the pattern ``#.#(#)`` where the decimal point
    and parenthesised uncertainty are optional. Strip the uncertainty and
    return the numeric part as a float. If ``data_value`` is a list,
    apply the conversion element-wise.

    Args:
        data_name: CIF data name, used only in the error message.
        data_value: Raw CIF value string, or a list of such strings.

    Returns:
        Float value (or list of floats) with uncertainty suffix removed.

    Raises:
        ValueError: If the value does not match the expected numeric
            pattern or cannot be converted to float.
    """
    if isinstance(data_value, list):
        return [
            cast(float, cif_numerical(data_name, data_value_element))
            for data_value_element in data_value
        ]
    else:
        try:
            if match := NUMERICAL_DATA_VALUE.match(data_value):
                return float(match.group(1))
            raise ValueError(
                f"Invalid numerical value in input CIF {data_name}: {data_value}"
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numerical value in input CIF {data_name}: {data_value}"
            ) from exc
