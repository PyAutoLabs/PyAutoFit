import logging
from pathlib import Path

import pytest

from autonerves.conf import output_path_for_test

import autofit as af
from autofit import database as m


@pytest.fixture(
    name="fit"
)
def query_fit(session, paths):
    fit, = m.Fit.all(session)
    return fit


output_path = str(
    Path(
        __file__
    ).parent / "temp"
)

@output_path_for_test(
    output_path
)
def test_create():
    m.open_database("test.sqlite")
    assert Path(output_path).exists()


@output_path_for_test(
    output_path
)
def test_make_dirs():
    m.open_database(
        f"a/long/path.sqlite"
    )


@output_path_for_test(
    output_path
)
def test_create_postgres():
    try:
        m.open_database(
            "postgresql://autofit:autofit@localhost/autofit"
        )
    except Exception as e:
        logging.exception(e)
    assert not Path(output_path).exists()


def test_incomplete(paths):
    assert paths.is_complete is False


def test_identifier(
        paths,
        fit
):
    assert fit.id == paths.identifier


def test_completion(
        paths,
        fit
):
    paths.completed()

    assert fit.is_complete
    assert paths.is_complete


def test_object(paths):
    gaussian = af.ex.Gaussian(
        normalization=2.1
    )

    assert paths.is_object(
        "gaussian"
    ) is False

    paths.save_object(
        "gaussian",
        gaussian
    )

    assert paths.is_object(
        "gaussian"
    ) is True
    assert paths.load_object(
        "gaussian"
    ) == gaussian

    paths.remove_object(
        "gaussian"
    )
    assert paths.is_object(
        "gaussian"
    ) is False


def test_save_all(
        paths,
        fit
):
    paths.save_all({
        "key": "value"
    })

    assert fit.model is not None
    assert "search" in fit
    assert fit.info["key"] == "value"
