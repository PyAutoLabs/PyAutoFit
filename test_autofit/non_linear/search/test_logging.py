import logging
import shutil

import pytest

import autofit as af
from autonerves.conf import with_config


class Analysis(af.mock.MockAnalysis):
    def log_likelihood_function(self, instance):
        logging.warning("Loud logging")
        return super().log_likelihood_function(instance)


@pytest.fixture(name="run_search")
def make_run_search():
    def run_search():
        search = af.mock.MockSearch("name", fit_fast=False)
        search.paths.remove_files = False
        analysis = Analysis()
        model = af.Model(af.ex.Gaussian)
        search.fit(model, analysis)
        return search

    return run_search


def test_logging(run_search):
    search = run_search()
    assert (search.paths.output_path / "search.log").exists()


@with_config("output", "search_log", value=False)
def test_logging_off(run_search, output_directory):
    shutil.rmtree(output_directory)
    search = run_search()
    assert not (search.paths.output_path / "search.log").exists()
