"""
The archive / resume-state strategies of ``autofit.non_linear.checkpoint`` (search-
extensibility phase A3, decision D5; ``docs/design/checkpointing.md``).
"""
import os
import pickle
from pathlib import Path

import dill
import pytest

import autofit as af
from autofit.non_linear.checkpoint import (
    DillCheckpointer,
    NativeFileCheckpointer,
    PickleCheckpointer,
    checkpointer_kind,
    load_legacy_search_internal,
)
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.paths.null import NullPaths
from autonerves import conf

from test_autofit.non_linear.search.conformance_roster import searches_under_test


class Holder:
    def __init__(self):
        self.value = 1
        self.pool = lambda: None  # a lambda dill can serialise; stripped anyway


@pytest.fixture(name="paths")
def make_paths(tmp_path):
    conf.instance.push(
        new_path=str(Path(__file__).parents[1] / "config"), output_path=str(tmp_path)
    )
    return DirectoryPaths(name="checkpoint_test")


def test_dill_round_trip(paths):
    checkpointer = DillCheckpointer()

    assert not checkpointer.exists(paths)

    checkpointer.save(paths, {"a": 1})

    assert checkpointer.exists(paths)
    assert checkpointer.load(paths) == {"a": 1}
    assert checkpointer.path(paths) == paths.search_internal_path / "search_internal.dill"


def test_dill_strips_and_restores_attributes(paths):
    checkpointer = DillCheckpointer(strip_attributes=("pool",))
    holder = Holder()
    pool = holder.pool

    checkpointer.save(paths, holder)

    assert holder.pool is pool
    assert checkpointer.load(paths).pool is None


def test_pickle_round_trip(paths):
    checkpointer = PickleCheckpointer()

    checkpointer.finalize(paths, {"positions": [1, 2]})

    with open(paths.search_internal_path / "search_internal.pickle", "rb") as f:
        assert pickle.load(f) == {"positions": [1, 2]}
    assert checkpointer.load(paths) == {"positions": [1, 2]}
    assert checkpointer.retain_after_completion


def test_writes_are_atomic(paths, monkeypatch):
    checkpointer = PickleCheckpointer()
    checkpointer.save(paths, {"good": True})

    def broken_dump(obj, f):
        f.write(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(pickle, "dump", broken_dump)

    with pytest.raises(RuntimeError):
        checkpointer.save(paths, {"good": False})

    monkeypatch.undo()
    assert checkpointer.load(paths) == {"good": True}
    assert not any(
        name.endswith(".tmp") for name in os.listdir(paths.search_internal_path)
    )


def test_corrupt_archive_raises_rather_than_being_treated_as_absent(paths):
    path = DillCheckpointer().path(paths)
    path.write_bytes(b"not a dill")

    with pytest.raises(Exception) as error:
        DillCheckpointer().load(paths)

    assert not isinstance(error.value, FileNotFoundError)


def test_native_file(paths):
    checkpointer = NativeFileCheckpointer("state.bin", loader=lambda p: p.read_bytes())

    checkpointer.save(paths, "ignored")
    checkpointer.finalize(paths, "ignored")
    assert not checkpointer.exists(paths)

    with pytest.raises(FileNotFoundError):
        checkpointer.load(paths)

    (paths.search_internal_path / "state.bin").write_bytes(b"abc")

    assert checkpointer.load(paths) == b"abc"

    checkpointer.discard(paths)
    assert not checkpointer.exists(paths)
    checkpointer.discard(paths)  # discarding an absent file is a no-op


@pytest.mark.parametrize(
    "checkpointer",
    [
        DillCheckpointer(),
        PickleCheckpointer(),
        NativeFileCheckpointer("x", loader=lambda p: None),
    ],
)
def test_null_paths_store_nothing(checkpointer):
    paths = NullPaths()

    checkpointer.save(paths, {"a": 1})

    assert checkpointer.path(paths) is None
    assert not checkpointer.exists(paths)
    assert checkpointer.load(paths) is None


def test_database_paths_store_nothing(session):
    paths = af.DatabasePaths(session=session, name="checkpoint_database")

    DillCheckpointer().save(paths, {"a": 1})
    PickleCheckpointer().save(paths, {"a": 1})

    assert DillCheckpointer().load(paths) is None
    assert PickleCheckpointer().load(paths) is None


def test_legacy_detection(paths):
    directory = paths.search_internal_path

    with pytest.raises(FileNotFoundError):
        load_legacy_search_internal(directory)

    with open(directory / "search_internal.dill", "wb") as f:
        dill.dump("dill", f)
    assert load_legacy_search_internal(directory) == "dill"

    with open(directory / "search_internal.pickle", "wb") as f:
        pickle.dump("pickle", f)
    assert load_legacy_search_internal(directory) == "pickle"


def test_archive_falls_back_to_the_legacy_probe(paths):
    """
    An output folder written before its search chose a strategy (here: a pickle
    archive where the search now expects a dill) still loads.
    """
    with open(paths.search_internal_path / "search_internal.pickle", "wb") as f:
        pickle.dump("legacy", f)

    assert DillCheckpointer().load(paths) == "legacy"


def test_checkpointer_kind():
    assert checkpointer_kind(DillCheckpointer()) == "dill"
    assert checkpointer_kind(PickleCheckpointer()) == "pickle"
    assert (
        checkpointer_kind(NativeFileCheckpointer("f.hdf", loader=lambda p: None))
        == "native:f.hdf"
    )
    assert checkpointer_kind(None) is None


# --- per-search declarations -----------------------------------------------------------


@pytest.mark.parametrize(
    "entry", searches_under_test(), ids=lambda entry: entry.name
)
def test_resume_state_implies_resumable(entry):
    missing = entry.missing_requirements()
    if missing:
        pytest.skip(f"{entry.name} needs {missing}")
    cls = entry.resolve()

    if cls.resume_state is not None:
        assert cls.resumable
    assert cls.checkpointer is not None


def test_nuts_and_smc_have_no_resume_state():
    from autofit.non_linear.search.mcmc.blackjax.nuts.search import BlackJAXNUTS
    from autofit.non_linear.search.mcmc.blackjax.smc.search import SMC

    for cls in (BlackJAXNUTS, SMC):
        assert cls.resume_state is None
        assert not cls.resumable
        assert isinstance(cls.checkpointer, PickleCheckpointer)
        assert cls.checkpointer.retain_after_completion


def test_emcee_archive_is_its_hdf_backend():
    assert isinstance(af.Emcee.checkpointer, NativeFileCheckpointer)
    assert af.Emcee.checkpointer.filename == "search_internal.hdf"
    assert af.Emcee.checkpointer.retain_after_completion
    assert af.Emcee.resume_state is af.Emcee.checkpointer


# --- retention precedence ---------------------------------------------------------------


@pytest.fixture(name="search_internal_off")
def make_search_internal_off(tmp_path):
    original_configs = list(conf.instance.configs)
    (tmp_path / "output.yaml").write_text("search_internal: false\n")
    conf.instance.push(new_path=str(tmp_path))
    yield
    conf.instance.configs = original_configs


def test_retention_precedence(search_internal_off):
    """
    ``retain_after_completion`` keeps the archive whatever the config says; otherwise
    ``output.search_internal: false`` removes the folder.
    """
    assert conf.instance["output"]["search_internal"] is False

    assert not af.DynestyStatic().retains_search_internal
    assert not af.Drawer().retains_search_internal
    assert af.Emcee().retains_search_internal


def test_output_search_internal_discards_the_resume_state(paths):
    search = af.DynestyStatic()
    search.paths = paths

    savestate = paths.search_internal_path / "savestate.save"
    savestate.write_bytes(b"resume")

    search.output_search_internal({"archive": True})

    assert not savestate.exists()
    assert DillCheckpointer().load(paths) == {"archive": True}


def test_output_search_internal_keeps_a_resume_state_that_is_the_archive(paths):
    search = af.MultiStartAdam() if hasattr(af, "MultiStartAdam") else None
    if search is None:
        pytest.skip("MultiStartAdam is not exported")
    search.paths = paths

    search.output_search_internal({"params": [1.0]})

    assert DillCheckpointer().load(paths) == {"params": [1.0]}
