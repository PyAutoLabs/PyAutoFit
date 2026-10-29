"""
Artefact integrity (results-library phase 0a): a result zip is complete or
absent, a corrupt zip or checkpoint is quarantined instead of failing every
rerun, and restoring a zip never destroys what it restores from.
"""
import multiprocessing
import os
import signal
import zipfile
from pathlib import Path

import numpy as np
import pytest

import autofit as af
from autonerves import conf
from autofit.non_linear.checkpoint import (
    quarantine_search_internal,
    resume_state_usable,
)
from autofit.non_linear.paths import abstract as paths_abstract
from autofit.non_linear.paths.abstract import zip_is_complete
from autofit.non_linear.timer import Timer
from autofit.tools.util import zip_directory

CONFIG_PATH = Path(__file__).parents[2] / "config"


def _completed_output(paths, n_files=1):
    """
    Write a minimal completed output folder for ``paths``.
    """
    files_path = Path(paths._files_path)
    for index in range(n_files):
        (files_path / f"member_{index}.json").write_text(f'{{"index": {index}}}')
    (Path(paths.output_path) / ".completed").write_text("")


def _paths(tmp_path, name, remove_files):
    paths = af.DirectoryPaths(name=name, path_prefix=str(tmp_path))
    paths.remove_files = remove_files
    return paths


# --- zip_directory ---------------------------------------------------------------------


def test__zip_directory__binary_members_are_stored_and_text_deflated(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("image.png", "data.fits", "checkpoint.hdf5", "samples.json"):
        (source / name).write_bytes(b"0" * 1000)

    zip_directory(source)

    with zipfile.ZipFile(f"{source}.zip") as f:
        compress = {info.filename: info.compress_type for info in f.infolist()}

    assert compress["image.png"] == zipfile.ZIP_STORED
    assert compress["data.fits"] == zipfile.ZIP_STORED
    assert compress["checkpoint.hdf5"] == zipfile.ZIP_STORED
    assert compress["samples.json"] == zipfile.ZIP_DEFLATED


def _zip_and_die(source, output):
    """
    Child process: start archiving ``source`` and SIGKILL itself after the first
    member, as a job killed by the scheduler mid-zip would be.
    """
    original_write = zipfile.ZipFile.write
    written = []

    def write(self, *args, **kwargs):
        original_write(self, *args, **kwargs)
        written.append(args[0])
        if len(written) == 1:
            os.kill(os.getpid(), signal.SIGKILL)

    zipfile.ZipFile.write = write
    zip_directory(source, output)


def test__zip_directory__killed_mid_zip_leaves_the_previous_zip_whole(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(5):
        (source / f"file_{index}.json").write_text("x" * 1000)

    output = tmp_path / "source.zip"
    zip_directory(source, output)
    before = output.read_bytes()

    (source / "new.json").write_text("new")

    process = multiprocessing.get_context("fork").Process(
        target=_zip_and_die, args=(source, output)
    )
    process.start()
    process.join()

    assert process.exitcode == -signal.SIGKILL
    assert output.read_bytes() == before
    with zipfile.ZipFile(output) as f:
        assert f.testzip() is None


# --- restore / is_complete / open_result -----------------------------------------------


def test__restore__corrupt_zip_is_quarantined_and_not_complete(tmp_path):
    paths = _paths(tmp_path, "corrupt", remove_files=True)
    _completed_output(paths)
    paths.zip_remove()

    zip_path = Path(paths._zip_path)
    zip_path.write_bytes(zip_path.read_bytes()[: zip_path.stat().st_size // 2])

    paths.restore()

    assert not zip_path.exists()
    assert len(list(tmp_path.rglob(f"{zip_path.name}.corrupt_*"))) == 1
    assert not paths.is_complete


@pytest.mark.parametrize("remove_files", [True, False])
def test__restore__extracts_and_keeps_zip_unless_owned(tmp_path, remove_files):
    paths = _paths(tmp_path, f"owned_{remove_files}", remove_files=remove_files)
    _completed_output(paths)
    paths.zip_remove()

    if not remove_files:
        import shutil

        shutil.rmtree(paths.output_path)

    paths.restore()

    assert (Path(paths.output_path) / ".completed").exists()
    assert (Path(paths._files_path) / "member_0.json").exists()
    assert Path(paths._zip_path).exists() is (not remove_files)
    assert not list(tmp_path.rglob("*.restore.*"))


def test__restore__leaves_a_complete_directory_untouched(tmp_path):
    paths = _paths(tmp_path, "untouched", remove_files=False)
    _completed_output(paths)
    paths.zip_remove()

    later = Path(paths._files_path) / "later.json"
    later.write_text("written after the zip")

    paths.restore()

    assert later.read_text() == "written after the zip"


def test__is_complete__reads_a_zipped_result(tmp_path):
    paths = _paths(tmp_path, "zipped", remove_files=True)
    _completed_output(paths)
    paths.zip_remove()

    assert not Path(paths.output_path).exists()
    assert paths.is_complete
    assert zip_is_complete(paths._zip_path)

    # A zip without a ``.completed`` member is not a completed result.
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "a.json").write_text("{}")
    zip_directory(incomplete)
    assert not zip_is_complete(f"{incomplete}.zip")


def test__is_complete__cache_follows_a_rewritten_zip(tmp_path):
    paths = _paths(tmp_path, "cached", remove_files=True)
    _completed_output(paths)
    paths.zip_remove()

    zip_path = Path(paths._zip_path)
    assert zip_is_complete(zip_path)

    zip_path.write_bytes(b"not a zip")

    assert not zip_is_complete(zip_path)
    assert not paths.is_complete


def test__open_result__reads_a_zip_without_extracting(tmp_path):
    paths = _paths(tmp_path, "read_only", remove_files=True)
    _completed_output(paths)
    paths.zip_remove()

    zip_path = Path(paths._zip_path)
    before = zip_path.read_bytes()

    root = paths.open_result()

    assert (root / "files" / "member_0.json").read_text() == '{"index": 0}'
    assert zip_path.read_bytes() == before
    assert not Path(paths.output_path).exists()

    assert _paths(tmp_path, "absent", remove_files=True).open_result() is None


def _restore_and_read(paths, barrier, queue):
    barrier.wait()
    paths.restore()
    queue.put(sorted(p.name for p in Path(paths._files_path).iterdir()))


@pytest.mark.parametrize("remove_files", [True, False])
def test__restore__two_processes_restore_one_zip_concurrently(tmp_path, remove_files):
    paths = _paths(tmp_path, f"concurrent_{remove_files}", remove_files=remove_files)
    _completed_output(paths, n_files=200)
    paths.zip_remove()

    import shutil

    shutil.rmtree(paths.output_path, ignore_errors=True)

    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    queue = context.Queue()

    processes = [
        context.Process(target=_restore_and_read, args=(paths, barrier, queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)

    assert [process.exitcode for process in processes] == [0, 0]

    listings = [queue.get(timeout=5) for _ in range(2)]
    expected = sorted(f"member_{index}.json" for index in range(200))
    for listing in listings:
        assert expected == [name for name in listing if name.startswith("member_")]

    assert (Path(paths.output_path) / ".completed").exists()
    assert not list(tmp_path.rglob("*.restore.*"))


# --- checkpoint quarantine -------------------------------------------------------------


def _truncated_hdf5(path: Path):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("points", data=np.arange(10000.0))
    path.write_bytes(path.read_bytes()[:1000])


def test__resume_state_usable__truncated_hdf5_is_quarantined(tmp_path):
    pytest.importorskip("nautilus")

    search = af.Nautilus(name="probe_hdf5", path_prefix=str(tmp_path))
    search.paths.model = af.Model(af.ex.Gaussian)

    checkpoint = Path(search.checkpoint_file)
    _truncated_hdf5(checkpoint)

    assert not resume_state_usable(search.paths, search.resume_state)

    failed = list(checkpoint.parent.parent.glob("search_internal.failed_*"))
    assert len(failed) == 1
    assert (failed[0] / checkpoint.name).exists()
    assert not checkpoint.exists()


def test__resume_state_usable__valid_hdf5_resumes(tmp_path):
    pytest.importorskip("nautilus")
    import h5py

    search = af.Nautilus(name="probe_valid", path_prefix=str(tmp_path))
    search.paths.model = af.Model(af.ex.Gaussian)

    with h5py.File(search.checkpoint_file, "w") as f:
        f.create_dataset("points", data=np.arange(10.0))

    assert resume_state_usable(search.paths, search.resume_state)
    assert Path(search.checkpoint_file).exists()


def test__quarantine__repeated_quarantines_do_not_collide(tmp_path):
    paths = _paths(tmp_path, "collide", remove_files=False)

    for _ in range(2):
        (paths.search_internal_path / "state.dill").write_bytes(b"x")
        assert quarantine_search_internal(paths, reason="test") is not None

    assert len(list(Path(paths._files_path).glob("search_internal.failed_*"))) == 2


# --- fit-level -------------------------------------------------------------------------


@pytest.fixture(name="fit_output")
def make_fit_output(tmp_path, monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")
    conf.instance.push(new_path=str(CONFIG_PATH), output_path=str(tmp_path))
    return tmp_path


def _analysis():
    return af.ex.Analysis(data=np.full(20, 5.0), noise_map=np.full(20, 1.0))


def _count_fits(monkeypatch, cls):
    calls = []
    original = cls._fit

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "_fit", counted)
    return calls


def _prepare_paths(search, model):
    search.paths.model = model
    search.paths.unique_tag = search.unique_tag


class _Killed(BaseException):
    pass


def test__fit__killed_mid_zip_resumes_as_complete(fit_output, monkeypatch):
    calls = _count_fits(monkeypatch, af.Drawer)

    def killed_zip(source_directory, output=None):
        raise _Killed

    with monkeypatch.context() as patch:
        patch.setattr(paths_abstract, "zip_directory", killed_zip)
        search = af.Drawer(name="killed_mid_zip")
        with pytest.raises(_Killed):
            search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert (Path(search.paths.output_path) / ".completed").exists()
    assert not Path(search.paths._zip_path).exists()

    # The debris a SIGKILL (rather than an exception) would leave.
    Path(f"{search.paths._zip_path}.12345.tmp").write_bytes(b"partial")

    resumed = af.Drawer(name="killed_mid_zip")
    result = resumed.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert len(calls) == 1
    assert isinstance(result, af.Result)
    assert zip_is_complete(resumed.paths._zip_path)


def test__fit__corrupt_zip_is_quarantined_and_the_fit_reruns(fit_output, monkeypatch):
    calls = _count_fits(monkeypatch, af.Drawer)

    search = af.Drawer(name="corrupt_zip_rerun")
    search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    zip_path = Path(search.paths._zip_path)
    zip_path.write_bytes(zip_path.read_bytes()[: zip_path.stat().st_size // 2])

    rerun = af.Drawer(name="corrupt_zip_rerun")
    result = rerun.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert len(calls) == 2
    assert isinstance(result, af.Result)
    assert len(list(zip_path.parent.glob(f"{zip_path.name}.corrupt_*"))) == 1
    assert zip_is_complete(zip_path)


def _failed_members(zip_path, filename):
    with zipfile.ZipFile(zip_path) as f:
        return [
            name
            for name in f.namelist()
            if "search_internal.failed_" in name and name.endswith(filename)
        ]


def test__fit__nautilus_truncated_checkpoint_starts_fresh(fit_output, monkeypatch):
    pytest.importorskip("nautilus")

    model = af.Model(af.ex.Gaussian)
    search = af.Nautilus(name="nautilus_truncated", n_live=10)
    _prepare_paths(search, model)
    _truncated_hdf5(Path(search.checkpoint_file))

    result = search.fit(model=model, analysis=_analysis())

    assert isinstance(result, af.Result)
    assert len(_failed_members(search.paths._zip_path, "checkpoint.hdf5")) == 1


def test__fit__nautilus_linalg_error_on_resume_retries_fresh(fit_output, monkeypatch):
    pytest.importorskip("nautilus")
    import h5py

    model = af.Model(af.ex.Gaussian)
    search = af.Nautilus(name="nautilus_linalg", n_live=10)
    _prepare_paths(search, model)

    with h5py.File(search.checkpoint_file, "w") as f:
        f.create_dataset("points", data=np.arange(10.0))

    original = type(search)._run_sampler
    attempts = []

    def run_sampler(self, ctx):
        attempts.append(Path(self.checkpoint_file).exists())
        if len(attempts) == 1:
            raise np.linalg.LinAlgError("singular bound")
        return original(self, ctx)

    monkeypatch.setattr(type(search), "_run_sampler", run_sampler)

    result = search.fit(model=model, analysis=_analysis())

    assert attempts == [True, False]
    assert isinstance(result, af.Result)
    assert len(_failed_members(search.paths._zip_path, "checkpoint.hdf5")) == 1


def test__fit__dynesty_poisoned_savestate_starts_fresh(fit_output):
    pytest.importorskip("dynesty")
    from autofit.non_linear.search.nest.dynesty.search.abstract import (
        DYNESTY_FILENAME,
    )

    model = af.Model(af.ex.Gaussian)
    search = af.DynestyStatic(name="dynesty_poisoned", nlive=10)
    _prepare_paths(search, model)

    checkpoint = Path(search.checkpoint_file)
    checkpoint.write_bytes(b"\x80\x04\x95poisoned dill")

    result = search.fit(model=model, analysis=_analysis())

    assert isinstance(result, af.Result)
    assert len(_failed_members(search.paths._zip_path, DYNESTY_FILENAME)) == 1


# --- Timer -----------------------------------------------------------------------------


def test__timer__writes_to_files_and_reads_legacy_output(tmp_path):
    files = tmp_path / "files"
    legacy = files / "search_internal"
    legacy.mkdir(parents=True)
    (legacy / ".start_time").write_text("100.0")
    (legacy / ".time").write_text("42.0")

    timer = Timer(files)

    assert timer.time == "42.0"

    timer.start()
    assert (files / ".start_time").read_text() == "100.0"

    timer.update()
    assert float((files / ".time").read_text()) > 42.0


def test__timer__fit_writes_timer_outside_search_internal(fit_output):
    search = af.Drawer(name="timer_location")
    search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    with zipfile.ZipFile(search.paths._zip_path) as f:
        names = f.namelist()

    assert "files/.start_time" in names
    assert "files/.time" in names
