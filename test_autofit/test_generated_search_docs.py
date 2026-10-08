"""
The generated search docs (``docs/_generate_searches.py``) are up to date with the
search manifest, and the capability matrix has one row per registered search.

Skipped when the ``docs`` directory is absent (an installed-package test run).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from autofit.non_linear.search import registry

DOCS = Path(__file__).resolve().parents[1] / "docs"
GENERATOR = DOCS / "_generate_searches.py"

pytestmark = pytest.mark.skipif(
    not GENERATOR.exists(), reason="docs/ is not part of this checkout"
)


def test_generated_search_pages_are_up_to_date():
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_capability_matrix_renders_one_row_per_search():
    text = (DOCS / "searches" / "index.rst").read_text()
    capabilities = text.split("Capabilities\n------------")[1].split("Objectives\n")[0]

    rows = [line for line in capabilities.splitlines() if line.startswith("   * - :class:")]

    assert len(rows) == len(registry.entries()) == 15
