import os
import shutil
from pathlib import Path

import pytest

import autofit as af
from autonerves.conf import output_path_for_test
from autofit.database import Fit
from autofit.database.sqlalchemy_ import sa

directory = Path(
    __file__
).parent

origin = Path(
    __file__
).parent / "database.sqlite"

copy = Path(
    __file__
).parent / "database_copy.sqlite"


@pytest.fixture(
    autouse=True
)
def manage_files():
    shutil.copy(
        origin,
        copy
    )
    yield
    os.remove(
        copy,

    )


def _check_migration():
    try:
        print(
            af.Aggregator.from_database(
                "database_copy.sqlite"
            ).fits
        )
    except sa.exc.OperationalError as e:
        raise AssertionError(
            "Migration steps are not complete"
        ) from e


@output_path_for_test(
    str(directory),
    remove=False
)
def test():
    """
    Raises an exception if changes to autofit/database/migration/steps.py
    are required to migrate old database files correctly.
    """
    _check_migration()


@output_path_for_test(
    str(directory),
    remove=False
)
def test_fail():
    Fit.random_column = sa.Column(
        sa.String
    )

    with pytest.raises(
            AssertionError
    ):
        _check_migration()
