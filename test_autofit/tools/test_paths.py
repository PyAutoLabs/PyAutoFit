import os
import shutil
from pathlib import Path

import pytest

import autofit as af

directory = Path(__file__).parent


class PatchPaths(af.DirectoryPaths):
    @property
    def sym_path(self) -> Path:
        return directory / "sym_path"

    @property
    def output_path(self) -> Path:
        return directory / "phase_output_path"


@pytest.fixture(name="paths")
def make_paths():
    return PatchPaths()


def test_restore(paths):
    paths.model = af.Model(af.ex.Gaussian)
    paths.save_all({}, {})

    paths.zip_remove()
    paths.restore()

    assert paths.output_path.exists()
    assert not Path(paths._zip_path).exists()

    shutil.rmtree(paths.output_path)
