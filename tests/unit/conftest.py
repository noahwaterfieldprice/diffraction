"""Unit-test-specific fixtures.

Layered conftest: this file is the unit-level layer. Shared mineral
constants and fixtures are defined in tests/conftest.py and are
automatically available here via pytest fixture discovery.

Unit-specific fixtures can be added here as the test suite grows.
"""

import importlib.util
import sys
from pathlib import Path

# Load the root-level tests/conftest.py as a module so we can re-export
# its constants. pytest adds tests/unit/ to sys.path before tests/, so
# a plain `import conftest` from this directory would resolve here
# (circular). Instead, we load tests/conftest.py by file path and
# re-export the constants unit test files need.
_root_conftest_path = Path(__file__).parent.parent / "conftest.py"
_spec = importlib.util.spec_from_file_location("_root_conftest", _root_conftest_path)
assert _spec is not None and _spec.loader is not None
_root_conftest = importlib.util.module_from_spec(_spec)
sys.modules["_root_conftest"] = _root_conftest
_spec.loader.exec_module(_root_conftest)

CALCITE_LATTICE_PARAMS: tuple[float, ...] = _root_conftest.CALCITE_LATTICE_PARAMS
