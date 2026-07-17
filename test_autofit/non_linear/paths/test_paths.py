import pickle
from pathlib import Path

import pytest

import autofit as af
from autoconf.conf import output_path_for_test
from autofit.non_linear.paths.null import NullPaths


def test_null_paths():
    search = af.DynestyStatic()

    assert isinstance(search.paths, NullPaths)


class TestPathDecorator:
    @staticmethod
    def assert_paths_as_expected(paths):
        assert paths.name == "name"
        assert paths.path_prefix == ""

    def test_with_arguments(self):
        search = af.m.MockSearch()
        search.paths = af.DirectoryPaths(name="name")

        self.assert_paths_as_expected(search.paths)

    def test_positional(self):
        search = af.m.MockSearch("name")
        paths = search.paths

        assert paths.name == "name"

    def test_paths_argument(self):
        search = af.m.MockSearch()
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)

    def test_combination_argument(self):
        search = af.m.MockSearch(
            "other",
        )
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)


output_path = Path(__file__).parent / "path"


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian)


@output_path_for_test(output_path)
def test_identifier_file(model):
    paths = af.DirectoryPaths()
    paths.model = model
    paths.search = af.DynestyStatic()
    paths.save_all({}, {})

    assert (output_path / paths.identifier / ".identifier").exists()


def test_serialize(model):
    paths = af.DirectoryPaths()
    paths.model = model

    pickled_paths = pickle.loads(pickle.dumps(paths))

    assert pickled_paths.model is not None


def test_unique_tag():
    paths = af.DirectoryPaths(unique_tag="unique_tag")
    assert "unique_tag" in paths.output_path.parts


class TestTestModeOutputPath:
    """
    PYAUTO_TEST_MODE must namespace the AutoFit output path so smoke
    runs cannot poison the cache for a later real run at the same
    paths. Regression for the "Fit Already Completed" silent skip.
    """

    def test_output_path_contains_test_mode_segment_when_env_set(
        self, monkeypatch
    ):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "2")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" in paths.output_path.parts

    def test_output_path_excludes_segment_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths.output_path.parts

    def test_output_path_excludes_segment_when_level_zero(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "0")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths.output_path.parts

    def test_make_path_contains_segment_when_env_set(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "2")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" in paths._make_path().parts

    def test_make_path_excludes_segment_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths._make_path().parts

    def test_make_path_excludes_segment_when_level_zero(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "0")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths._make_path().parts


def test__preserve_in_zip__file_survives_restore(tmp_path):
    import zipfile

    paths = af.DirectoryPaths(name="preserve_test", path_prefix=str(tmp_path))

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")

    paths.zip_remove()
    assert Path(paths._zip_path).exists()

    # A post-completion cache write: on disk but not in the zip.
    cache_file = files_path / "cache_artifact.json"
    files_path.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"cached": true}')

    paths.preserve_in_zip(cache_file)

    with zipfile.ZipFile(paths._zip_path) as f:
        assert "files/cache_artifact.json" in f.namelist()

    # Idempotent — appending again must not duplicate the member.
    paths.preserve_in_zip(cache_file)
    with zipfile.ZipFile(paths._zip_path) as f:
        assert f.namelist().count("files/cache_artifact.json") == 1

    # The restore cycle (rmtree + re-extract) keeps the preserved file.
    paths.restore()
    assert cache_file.exists()


def test__preserve_in_zip__no_zip_is_a_no_op(tmp_path):
    paths = af.DirectoryPaths(name="preserve_noop_test", path_prefix=str(tmp_path))

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    cache_file = files_path / "cache_artifact.json"
    cache_file.write_text('{"cached": true}')

    paths.preserve_in_zip(cache_file)

    assert not Path(paths._zip_path).exists()
    assert cache_file.exists()
