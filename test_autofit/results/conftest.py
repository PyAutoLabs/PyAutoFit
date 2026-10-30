import pytest


@pytest.fixture(name="results_root")
def make_results_root(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    return root
