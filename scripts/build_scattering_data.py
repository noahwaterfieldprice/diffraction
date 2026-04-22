"""Build static element scattering data JSON from the Dans source text file.

Reads:
  Dans Element Properties.txt  (whitespace-delimited, 95 rows, Z=1-95)

Writes:
  src/diffraction/static/elements.json

Usage (from project root):
  python scripts/build_scattering_data.py <source_path>

The **canonical** invocation for CI and reproducible rebuilds is to pass the
source path explicitly; the script's fallback path is a convenience for the
original authoring machine only and is **not** a cross-environment contract.
CONTEXT.md D-01 documents a related convenience path
(``../diffraction_V0/example_libraries/dan/Dans Element Properties.txt``)
that happens to also exist on the authoring machine; neither location is
guaranteed on a fresh clone. If no explicit path is given and the hard-coded
fallback below does not resolve, the script exits with a helpful message
instructing the caller to pass ``<source_path>``.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


def build_element_data(source_path: Path, output_path: Path) -> None:
    """Convert the Dans element properties text file into bundled JSON.

    Args:
        source_path: Path to ``Dans Element Properties.txt`` (whitespace-delimited,
            95 rows covering Z=1-95 with header line).
        output_path: Path to the bundled ``elements.json`` output file. Parent
            directories are created if needed.

    Raises:
        AssertionError: If the source file does not have exactly 95 rows, if any
            Cromer-Mann column contains NaN, or if the set of null-``Coh_b``
            elements drifts from the expected six (Po, At, Rn, Fr, Ac, Pu).
    """
    data = np.genfromtxt(source_path, dtype=None, names=True, encoding="utf-8")
    assert len(data) == 95, f"Expected 95 rows, got {len(data)}"

    cm_cols = ["a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4", "c"]
    for col in cm_cols:
        nan_rows = [int(r["Z"]) for r in data if math.isnan(float(r[col]))]
        assert not nan_rows, f"NaN in Cromer-Mann column {col!r} for Z={nan_rows}"

    elements: dict[str, dict] = {}
    for row in data:
        sym = (
            row["Element"].decode()
            if isinstance(row["Element"], bytes)
            else str(row["Element"])
        )
        name = (
            row["Name"].decode() if isinstance(row["Name"], bytes) else str(row["Name"])
        )
        coh_b_raw = float(row["Coh_b"])
        coh_b = None if math.isnan(coh_b_raw) else coh_b_raw
        if coh_b is None:
            print(f"[INFO] NaN neutron Coh_b for {sym} (Z={int(row['Z'])})")
        elements[sym] = {
            "z": int(row["Z"]),
            "symbol": sym,
            "name": name,
            "cromer_mann": {
                "a": [float(row[f"a{i}"]) for i in range(1, 5)],
                "b": [float(row[f"b{i}"]) for i in range(1, 5)],
                "c": float(row["c"]),
            },
            "neutron_b_coh": coh_b,
        }

    null_neutron = sorted(
        e["z"] for e in elements.values() if e["neutron_b_coh"] is None
    )
    expected_null_zs = [84, 85, 86, 87, 89, 94]  # Po, At, Rn, Fr, Ac, Pu
    assert null_neutron == expected_null_zs, (
        f"Expected null neutron b_coh for Z={expected_null_zs} "
        f"(Po, At, Rn, Fr, Ac, Pu), got Z={null_neutron}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(elements, f, indent=2)
    print(f"Wrote {len(elements)} elements to {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    else:
        source = (
            project_root.parent
            / "example"
            / "diffraction_v0"
            / "example_libraries"
            / "dan"
            / "Dans Element Properties.txt"
        )
    if not source.exists():
        raise SystemExit(
            f"Source file not found at {source}. "
            "Pass an explicit path: "
            "python scripts/build_scattering_data.py <path_to_source_txt>"
        )
    output = project_root / "src" / "diffraction" / "static" / "elements.json"
    build_element_data(source, output)
