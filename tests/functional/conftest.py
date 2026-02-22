"""Functional test fixtures and directory constants.

Provides directory path constants for CIF fixture files used across all
functional tests.
"""

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
VALID_CIFS_DIR = STATIC_DIR / "valid_cifs"
