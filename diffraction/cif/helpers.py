from collections import Iterable
import re

from .cif import load_cif

# CIF data names corresponding to numerical parameters
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
    "atom_site_fract_x",
    "atom_site_fract_y",
    "atom_site_fract_z",
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

# CIF data names corresponding to textual parameters
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
    "chemical_formula_sum ",
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
    "symmetry_space_group_name_Hall"
)

# Map between diffraction object parameters and CIF data names
CIF_NAMES = {
    "a": "cell_length_a",
    "b": "cell_length_b",
    "c": "cell_length_c",
    "alpha": "cell_angle_alpha",
    "beta": "cell_angle_beta",
    "gamma": "cell_angle_gamma",
    "space_group": "symmetry_space_group_name_H-M"
}

NUMERICAL_DATA_VALUE = re.compile("(-?\d+\.?\d*)(?:\(\d+\))?$")


def load_data_block(filepath, data_block=None):
    """Extract the :term:`data items` of a specific :term:`data
    block` from a :term:`CIF`.

    For a CIF with with only a single data block, the data items of
    that data block are returned automatically. For a multiple data
    block CIF, the data items of the data block given by `data_block`
    (specified by :term:`data block header`) are returned as a
    dictionary. An exception is raised if the data block is not given.
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


def get_cif_data(data_items, *data_names):
    """Retrieve a list of :term:`data values` from dictionary of raw
    :term:`CIF` :term:`data items` given an arbitrary number of
    :term:`data names`.

    Any numerical data values are converted to numerical strings i.e.
    the errors are stripped off. Raises a ValueError if input data does
    not contain any requested data items.


    """
    data = []
    for data_name in data_names:
        try:
            data_value = data_items[data_name]
        except KeyError:
            raise ValueError("Parameter: '{0}' missing from input CIF".format(
                data_name))
        if data_name in NUMERICAL_DATA_NAMES:
            data_value = cif_numerical(data_name, data_value)
        data.append(data_value)
    return data


def cif_numerical(data_name, data_value):
    """Extract numerical :term:`data value` from raw :term:`CIF` data

    The numerical data value is matched to the pattern #.#(#), where #
    signifies one or more digits and the decimal points and error are
    optional. If present, the error is stripped off and the remaining
    string is converted to a float and returned.

    If the numerical data value is a list then each data value in the
    list is converted and the converted list is returned.
    """
    if isinstance(data_value, list):
        data_value = [cif_numerical(data_name, data_value_element)
                      for data_value_element in data_value]
    else:
        try:
            match = NUMERICAL_DATA_VALUE.match(data_value)
            data_value = float(match.group(1))
        except (AttributeError, ValueError):
            raise ValueError("Invalid numerical value in input "
                             "CIF {0}: {1}".format(data_name, data_value))
    return data_value
