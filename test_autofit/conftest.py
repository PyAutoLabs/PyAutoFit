import multiprocessing
import os
import shutil
import sys
from os import path
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from matplotlib import pyplot

from autoconf import conf
from autofit import database as db
from autofit import fixtures
from autofit.database.model import sa
from autofit.non_linear.search import abstract_search

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork")

directory = Path(__file__).parent


@pytest.fixture(autouse=True)
def turn_off_gc(monkeypatch):
    monkeypatch.setattr(abstract_search, "gc", MagicMock())


@pytest.fixture(name="remove_ids")
def make_remove_ids():
    def remove_ids(d):
        if isinstance(d, dict):
            return {k: remove_ids(v) for k, v in d.items() if k != "id"}
        elif isinstance(d, list):
            return [remove_ids(v) for v in d]
        return d

    return remove_ids


@pytest.fixture(name="test_directory", scope="session")
def make_test_directory():
    return directory


@pytest.fixture(name="output_directory", scope="session")
def make_output_directory(test_directory):
    return test_directory / "output"


@pytest.fixture(name="remove_output", scope="session")
def make_remove_output(output_directory):
    def remove_output():
        try:
            for item in os.listdir(output_directory):
                if item != "non_linear":
                    item_path = output_directory / item
                    if item_path.is_dir():
                        shutil.rmtree(
                            item_path,
                            ignore_errors=True,
                        )
                    else:
                        os.remove(item_path)
        except (FileExistsError, FileNotFoundError):
            pass

    return remove_output


@pytest.fixture(autouse=True)
def do_remove_output(output_directory, remove_output):
    yield
    remove_output()


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch


@pytest.fixture(name="session")
def make_session():
    engine = sa.create_engine("sqlite://")
    session = sa.orm.sessionmaker(bind=engine)()
    db.Base.metadata.create_all(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(autouse=True, scope="session")
def remove_logs():
    yield
    for d, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".log"):
                os.remove(path.join(d, file))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance.push(
        new_path=path.join(directory, "config"),
        output_path=path.join(directory, "output"),
    )


@pytest.fixture(name="model_gaussian_x1")
def make_model_gaussian_x1():
    return fixtures.make_model_gaussian_x1()


@pytest.fixture(name="samples_x5")
def make_samples_x5():
    return fixtures.make_samples_x5()
