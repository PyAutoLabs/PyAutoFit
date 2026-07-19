from pathlib import Path

import autofit as af
from autonerves.dictable import to_dict, from_dict


def test_path_prefix():
    path_prefix = Path("test_path_prefix")
    paths = af.DirectoryPaths(
        path_prefix=path_prefix,
    )
    paths = from_dict(to_dict(paths))

    assert paths.path_prefix == path_prefix


def test_identifier():
    paths = af.DirectoryPaths()
    new_paths = from_dict(to_dict(paths))

    assert new_paths.identifier == paths.identifier


def test_identifier_with_model():
    paths = af.DirectoryPaths()
    paths.model = af.Model(af.ex.Gaussian)
    new_paths = from_dict(to_dict(paths))

    assert new_paths.identifier == paths.identifier


def test_serialise_search():
    search = af.DynestyStatic(
        path_prefix=Path("searches"),
    )

    assert "paths" in to_dict(search)["arguments"]


def test_serialise_database_paths(session):
    paths = af.DatabasePaths(session)
    paths = from_dict(to_dict(paths))

    assert paths.session is None
