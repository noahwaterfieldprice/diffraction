"""Build static space group JSON data files from source text files.

Reads:
  notes/space_group_generators.txt  -- Python dict literal with all 530 settings
  notes/space_groups.txt            -- structured text with xHM-to-number mapping

Writes:
  src/diffraction/static/space_groups/1.json ... 230.json

Run from project root:
  python scripts/build_space_group_data.py
"""

import ast
import json
import re
from fractions import Fraction
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardcoded crystallographic mappings
# ---------------------------------------------------------------------------


def _expand_ranges(
    ranges: list[tuple[int, int, str]],
) -> dict[int, str]:
    """Build a dict from (start, end, value) triples by expanding each range."""
    result: dict[int, str] = {}
    for start, end, value in ranges:
        for n in range(start, end + 1):
            result[n] = value
    return result


# SG number -> point group symbol (verified against ITA)
_SG_POINT_GROUP: dict[int, str] = _expand_ranges([
    (1, 1, "1"),
    (2, 2, "-1"),
    (3, 5, "2"),
    (6, 9, "m"),
    (10, 15, "2/m"),
    (16, 24, "222"),
    (25, 46, "mm2"),
    (47, 74, "mmm"),
    (75, 80, "4"),
    (81, 82, "-4"),
    (83, 88, "4/m"),
    (89, 98, "422"),
    (99, 110, "4mm"),
    (111, 122, "-42m"),
    (123, 142, "4/mmm"),
    (143, 146, "3"),
    (147, 148, "-3"),
    (149, 155, "32"),
    (156, 161, "3m"),
    (162, 167, "-3m"),
    (168, 173, "6"),
    (174, 174, "-6"),
    (175, 176, "6/m"),
    (177, 182, "622"),
    (183, 186, "6mm"),
    (187, 190, "-6m2"),
    (191, 194, "6/mmm"),
    (195, 199, "23"),
    (200, 206, "m-3"),
    (207, 214, "432"),
    (215, 220, "-43m"),
    (221, 230, "m-3m"),
])

# SG number -> crystal system
_SG_CRYSTAL_SYSTEM: dict[int, str] = _expand_ranges([
    (1, 2, "triclinic"),
    (3, 15, "monoclinic"),
    (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"),
    (143, 167, "trigonal"),
    (168, 194, "hexagonal"),
    (195, 230, "cubic"),
])

# SGs that use origin choice 2 as default (not origin choice 1)
ORIGIN_CHOICE_2_SGS: frozenset[int] = frozenset(
    {
        48,
        50,
        59,
        68,
        70,
        85,
        86,
        88,
        125,
        126,
        129,
        130,
        133,
        134,
        137,
        138,
        141,
        142,
        201,
        203,
        222,
        224,
        227,
        228,
    }
)

# R-lattice SG numbers (have H and R settings)
R_GROUP_SGS: frozenset[int] = frozenset({146, 148, 155, 160, 161, 166, 167})


# ---------------------------------------------------------------------------
# xyz operator parser
# ---------------------------------------------------------------------------


def parse_xyz_operator(xyz_str: str) -> tuple[list[list[int]], list[str]]:
    """Parse a coordinate-triplet string into (W, t).

    Args:
        xyz_str: Coordinate string like '-x+1/2,y,-z+3/4'.

    Returns:
        Tuple of (W, t) where W is a 3x3 integer rotation matrix and t is a
        list of 3 fraction strings like ['1/2', '0', '3/4'].
    """
    W = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    t = [Fraction(0), Fraction(0), Fraction(0)]

    axes = {"x": 0, "y": 1, "z": 2}

    for row, term in enumerate(xyz_str.split(",")):
        term = term.strip()
        # Tokenize: split on +/- boundaries, keeping the sign with the token
        tokens = re.findall(r"[+-]?[^+-]+", term)
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            found_var = next((var for var in "xyz" if var in token), None)
            if found_var is not None:
                coeff_str = token.replace(found_var, "").strip()
                if coeff_str in ("", "+"):
                    coeff = 1
                elif coeff_str == "-":
                    coeff = -1
                else:
                    coeff = int(coeff_str)
                W[row][axes[found_var]] = coeff
            else:
                # Pure translation term
                t[row] += Fraction(token)

    return W, [str(ti) for ti in t]


# ---------------------------------------------------------------------------
# Source file parsers
# ---------------------------------------------------------------------------


def load_xhm_to_number(path: Path) -> dict[str, int]:
    """Parse space_groups.txt to build a mapping from xHM symbol to SG number.

    Args:
        path: Path to notes/space_groups.txt.

    Returns:
        Dict mapping stripped xHM symbol string to integer SG number.
    """
    xhm_to_number: dict[str, int] = {}
    current_number: int | None = None

    with path.open() as f:
        in_block = False
        for line in f:
            line = line.strip()
            if line == "begin_spacegroup":
                in_block = True
                current_number = None
            elif line == "end_spacegroup":
                in_block = False
            elif in_block:
                if line.startswith("number "):
                    current_number = int(line.split()[1])
                elif (
                    line.startswith("symbol xHM ")
                    and current_number is not None
                    and (m := re.search(r"'([^']+)'", line))
                ):
                    xhm_to_number[m.group(1)] = current_number

    return xhm_to_number


def load_generators(path: Path) -> dict[str, tuple[list[str], list[str]]]:
    """Parse space_group_generators.txt via ast.literal_eval.

    Args:
        path: Path to notes/space_group_generators.txt.

    Returns:
        Dict mapping xHM key to (primitive_ops_list, centering_vectors_list).
    """
    return ast.literal_eval(path.read_text())


# ---------------------------------------------------------------------------
# Setting key logic
# ---------------------------------------------------------------------------


def determine_setting_key(xhm: str) -> str:
    """Determine the setting key for a given xHM symbol.

    Args:
        xhm: Extended Hermann-Mauguin symbol like 'F d -3 m :2' or 'R 3 c :H'.

    Returns:
        Setting key string: '1', '2', 'H', 'R', or 'standard'.
    """
    if ":1" in xhm:
        return "1"
    if ":2" in xhm:
        return "2"
    if ":H" in xhm:
        return "H"
    if ":R" in xhm:
        return "R"
    return "standard"


def determine_default_setting(sg_number: int, available_keys: set[str]) -> str:
    """Determine the default setting key for a given SG.

    Args:
        sg_number: Space group number 1-230.
        available_keys: Set of setting keys present in the data.

    Returns:
        Default setting key string.
    """
    if "H" in available_keys:
        return "H"
    if "1" in available_keys or "2" in available_keys:
        return "2" if sg_number in ORIGIN_CHOICE_2_SGS else "1"
    return "standard"


def compact_symbol(xhm: str) -> str:
    """Derive compact symbol from xHM: strip spaces and drop ':N' suffix.

    Args:
        xhm: Extended Hermann-Mauguin symbol like 'F d -3 m :2'.

    Returns:
        Compact symbol like 'Fd-3m'.
    """
    base, _, _ = xhm.partition(":")
    return base.replace(" ", "").strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_space_group_data(
    generators_path: Path,
    space_groups_path: Path,
    output_dir: Path,
) -> None:
    """Convert source files to 230 clean JSON files.

    Args:
        generators_path: Path to notes/space_group_generators.txt.
        space_groups_path: Path to notes/space_groups.txt.
        output_dir: Directory to write 1.json ... 230.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source data
    xhm_to_number = load_xhm_to_number(space_groups_path)
    gen_data = load_generators(generators_path)

    # Group settings by SG number
    sg_settings: dict[int, dict[str, dict]] = {n: {} for n in range(1, 231)}

    for xhm, (prim_ops, cen_vecs) in gen_data.items():
        sg_number = xhm_to_number.get(xhm)
        if sg_number is None:
            raise ValueError(f"No SG number found for xHM key: {xhm!r}")

        setting_key = determine_setting_key(xhm)

        # Parse primitive operators
        operators = []
        for xyz in prim_ops:
            W, t = parse_xyz_operator(xyz)
            operators.append({"W": W, "t": t})

        # Parse centering vectors (extract only translation, W must be identity)
        centering_vectors = []
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for cv_xyz in cen_vecs:
            W_cv, t_cv = parse_xyz_operator(cv_xyz)
            assert W_cv == identity, (
                f"Centering vector {cv_xyz!r} for {xhm!r} has non-identity W: {W_cv}"
            )
            centering_vectors.append(t_cv)

        sg_settings[sg_number][setting_key] = {
            "xhm_symbol": xhm,
            "centering_vectors": centering_vectors,
            "operators": operators,
        }

    # Write one JSON file per SG
    for sg_number in range(1, 231):
        settings = sg_settings[sg_number]
        if not settings:
            raise ValueError(f"No settings found for SG {sg_number}")

        available_keys = set(settings.keys())
        default_setting = determine_default_setting(sg_number, available_keys)

        # Derive compact symbol from the default setting's xHM
        default_xhm = settings[default_setting]["xhm_symbol"]
        symbol = compact_symbol(default_xhm)

        data = {
            "number": sg_number,
            "symbol": symbol,
            "point_group": _SG_POINT_GROUP[sg_number],
            "crystal_system": _SG_CRYSTAL_SYSTEM[sg_number],
            "default_setting": default_setting,
            "settings": settings,
        }

        out_path = output_dir / f"{sg_number}.json"
        with out_path.open("w") as f:
            json.dump(data, f, indent=2)

    print(f"Wrote 230 space group JSON files to {output_dir}")


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_all_jsons(output_dir: Path) -> None:
    """Run spot checks on all generated JSON files.

    Args:
        output_dir: Directory containing the generated JSON files.

    Raises:
        AssertionError: If any check fails.
    """
    files = sorted(output_dir.glob("*.json"))
    assert len(files) == 230, f"Expected 230 JSON files, found {len(files)}"

    # Spot checks for specific SGs
    _check_sg1(output_dir)
    _check_sg2(output_dir)
    _check_sg167(output_dir)
    _check_sg225(output_dir)
    _check_sg227(output_dir)

    # Global checks: no floats in translations
    for f in files:
        data = json.loads(f.read_text())
        for sk, sv in data["settings"].items():
            for op in sv["operators"]:
                for ti in op["t"]:
                    assert isinstance(ti, str), (
                        f"{f.name} setting {sk} has non-string in t: {ti!r}"
                    )
            for cv in sv["centering_vectors"]:
                for ci in cv:
                    assert isinstance(ci, str), (
                        f"{f.name} setting {sk} has non-string in centering: {ci!r}"
                    )

    print("All 230 JSONs verified OK")


def _load_sg_json(output_dir: Path, number: int) -> dict:
    return json.loads((output_dir / f"{number}.json").read_text())


def _check_sg1(output_dir: Path) -> None:
    d = _load_sg_json(output_dir, 1)
    assert d["number"] == 1
    assert d["point_group"] == "1"
    assert d["crystal_system"] == "triclinic"
    assert d["default_setting"] == "standard"
    assert len(d["settings"]["standard"]["operators"]) == 1
    assert len(d["settings"]["standard"]["centering_vectors"]) == 1


def _check_sg2(output_dir: Path) -> None:
    d = _load_sg_json(output_dir, 2)
    assert d["number"] == 2
    assert d["point_group"] == "-1"
    assert len(d["settings"]["standard"]["operators"]) == 2


def _check_sg167(output_dir: Path) -> None:
    d = _load_sg_json(output_dir, 167)
    assert d["number"] == 167
    assert d["point_group"] == "-3m"
    assert d["crystal_system"] == "trigonal"
    assert d["default_setting"] == "H"
    assert "H" in d["settings"] and "R" in d["settings"]
    assert len(d["settings"]["H"]["operators"]) == 12
    assert len(d["settings"]["H"]["centering_vectors"]) == 3


def _check_sg225(output_dir: Path) -> None:
    d = _load_sg_json(output_dir, 225)
    assert d["number"] == 225
    assert d["default_setting"] == "standard"
    assert len(d["settings"]["standard"]["operators"]) == 48
    assert len(d["settings"]["standard"]["centering_vectors"]) == 4
    assert d["crystal_system"] == "cubic"
    assert d["point_group"] == "m-3m"


def _check_sg227(output_dir: Path) -> None:
    d = _load_sg_json(output_dir, 227)
    assert d["number"] == 227
    assert d["default_setting"] == "2"
    assert "1" in d["settings"] and "2" in d["settings"]
    assert d["point_group"] == "m-3m"
    assert d["crystal_system"] == "cubic"
    assert len(d["settings"]["2"]["operators"]) == 48


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent
    generators_path = project_root / "notes" / "space_group_generators.txt"
    space_groups_path = project_root / "notes" / "space_groups.txt"
    output_dir = project_root / "src" / "diffraction" / "static" / "space_groups"

    build_space_group_data(generators_path, space_groups_path, output_dir)

    if "--verify" in sys.argv:
        verify_all_jsons(output_dir)
