"""
The status record of real (cheap) fits: written beside the output on start and
completion, an attempt history across a kill and a resume, a failure record, cadence
writes, SIGUSR1 / SIGTERM in a fitting process, and nothing at all when
``general.hpc.status`` is off.
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.status import MAX_BYTES, FitStatus, StatusWriter, record
from autofit.non_linear.paths.directory import DirectoryPaths
from autonerves import conf

CONFIG = Path(__file__).parents[2] / "config"

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="output", autouse=True)
def make_output(tmp_path):
    conf.instance.push(new_path=str(CONFIG), output_path=str(tmp_path / "output"))
    return tmp_path / "output"


def _analysis():
    return af.ex.Analysis(data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=False)


def _fit(name="status_fit", analysis=None):
    search = af.Drawer(name=name, total_draws=5)
    search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis or _analysis())
    return search


def _writer(search):
    return StatusWriter(search.paths.output_path)


def test_completed_fit_writes_a_small_record_beside_its_output():
    search = _fit()
    writer = _writer(search)
    status = writer.read()

    assert writer.status_path.parent == Path(search.paths.output_path).parent
    assert len(writer.status_path.read_bytes()) <= MAX_BYTES
    assert status["schema"] == 1
    assert status["state"] == "completed"
    assert status["attempt"] == 1
    assert status["failure_class"] is None
    assert status["identifier"] == search.paths.identifier
    assert status["search"] == "Drawer"
    assert status["resources"]["wall_s"] >= 0
    # `remove_files: true` in the test config: only the zip is left, and the record
    # (which sits beside the folder) survives it.
    assert not Path(search.paths.output_path).exists()
    assert status["outputs"]["zip_complete"] is True

    events = [entry["event"] for entry in writer.history()]
    assert events == ["start", "completed"]


def test_rerun_of_a_completed_fit_is_skipped_not_a_new_attempt():
    _fit()
    search = _fit()
    writer = _writer(search)

    assert writer.read()["state"] == "completed"
    history = writer.history()
    assert [entry["event"] for entry in history] == ["start", "completed", "skipped"]
    assert history[-1]["attempt"] == 1


class BadInput(ValueError):
    failure_class = "bad_input"


class FailingAnalysis(af.ex.Analysis):
    def modify_before_fit(self, paths, model):
        raise BadInput("the data are all ones")


def test_exception_writes_a_failure_record_and_is_reraised():
    search = af.Drawer(name="status_failed", total_draws=5)
    analysis = FailingAnalysis(
        data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=False
    )

    with pytest.raises(BadInput, match="all ones"):
        search.fit(model=af.Model(af.ex.Gaussian), analysis=analysis)

    writer = _writer(search)
    status = writer.read()
    assert status["state"] == "failed"
    assert status["failure_class"] == "bad_input"
    assert status["outputs"]["traceback"] == str(writer.traceback_path)
    assert "BadInput: the data are all ones" in writer.traceback_path.read_text()
    assert writer.history()[-1]["event"] == "failed"
    assert search._fit_status is None


def test_failure_before_the_model_is_set_is_not_recorded(output):
    search = _fit(name="status_reused")
    writer = _writer(search)

    class NoParameters(Exception):
        pass

    def check_model(model):
        raise NoParameters()

    search.check_model = check_model
    with pytest.raises(NoParameters):
        search.fit(model=af.Model(af.ex.Exponential), analysis=_analysis())

    # The completed fit's record (the paths' model is still the first fit's) is kept.
    assert writer.read()["state"] == "completed"
    assert not writer.traceback_path.exists()


def test_disabled_writes_nothing_and_leaves_the_output_unchanged(output, monkeypatch):
    on = _fit(name="status_on")
    on_files = sorted(
        str(p.relative_to(output / "status_on"))
        for p in (output / "status_on").rglob("*")
    )

    monkeypatch.setattr(record, "status_enabled", lambda: False)
    off = _fit(name="status_off")
    off_files = sorted(
        str(p.relative_to(output / "status_off"))
        for p in (output / "status_off").rglob("*")
    )

    assert FitStatus.for_search(off) is None
    status_files = {f for f in on_files if ".status." in f or ".traceback." in f}
    assert len(status_files) == 2
    assert [f for f in on_files if f not in status_files] == off_files
    assert not any(".status." in f for f in off_files)


def test_cadence_writes_while_running(tmp_path):
    search = af.Drawer(name="status_cadence")
    search.paths = DirectoryPaths(name="status_cadence")
    search.paths.model = af.Model(af.ex.Gaussian)
    status = FitStatus(search, interval_s=0.05)

    status.start()
    time.sleep(0.5)
    status.close()

    # One write at start, then roughly one per interval.
    assert status.n_writes >= 5
    writes = status.n_writes
    time.sleep(0.15)
    assert status.n_writes == writes
    assert _writer(search).read()["state"] == "running"


def test_update_takes_progress_and_a_signal_flushes_the_checkpoint():
    saved = []

    class ResumeState:
        def save(self, paths, internal):
            saved.append(internal)

    search = af.Drawer(name="status_flush")
    search.paths.model = af.Model(af.ex.Gaussian)
    search.resume_state = ResumeState()
    search.progress = lambda internal: {"n_like": internal, "fraction": internal / 10}
    status = FitStatus(search, interval_s=0)

    status.start()
    status.update(2)
    status.update(4)
    assert _writer(search).read()["progress"]["n_like"] == 4

    status.timeout_pending("SIGUSR1")
    status.close()

    assert saved == [4]
    document = _writer(search).read()
    assert document["state"] == "timeout_pending"
    assert document["progress"]["eta_s"] is not None
    assert _writer(search).history()[-1] == {
        "event": "timeout_pending",
        "attempt": 1,
        "time": document["updated"],
        "signal": "SIGUSR1",
    }


SCRIPT = """
import os, sys, time
import numpy as np
import autofit as af
from autonerves import conf

conf.instance.push(new_path={config!r}, output_path={output!r})


class Slow(af.ex.Analysis):
    def log_likelihood_function(self, instance, shared=None, xp=np):
        time.sleep(float(os.environ.get("STATUS_SLEEP", "0")))
        return super().log_likelihood_function(instance, shared, xp)


search = af.Drawer(name="status_subprocess", total_draws=40)
search.fit(
    model=af.Model(af.ex.Gaussian),
    analysis=Slow(data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=False),
)
"""


def _script(tmp_path, output):
    path = tmp_path / "fit_script.py"
    path.write_text(SCRIPT.format(config=str(CONFIG), output=str(output)))
    return path


def _status_path(output):
    found = list((output / "status_subprocess").glob("*.status.json"))
    return found[0] if found else None


def _launch(script, sleep):
    # The subprocess imports the autofit under test, not whichever is installed.
    source = str(Path(af.__file__).parents[1])
    pythonpath = os.pathsep.join(
        filter(None, [source, os.environ.get("PYTHONPATH")])
    )
    env = dict(os.environ, STATUS_SLEEP=str(sleep), PYTHONPATH=pythonpath)
    return subprocess.Popen(
        [sys.executable, str(script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_for_state(output, state, process, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        path = _status_path(output)
        if path is not None:
            try:
                if json.loads(path.read_text())["state"] == state:
                    return path
            except (OSError, ValueError):
                pass
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise AssertionError(
        f"state {state} never written; stderr: {process.stderr.read()[-2000:]!r}"
        if process.poll() is not None
        else f"state {state} never written"
    )


needs_posix_signals = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="POSIX signals only"
)


@needs_posix_signals
def test_killed_and_resumed_fit_is_attempt_two(tmp_path, output):
    script = _script(tmp_path, output)

    process = _launch(script, sleep=0.5)
    _wait_for_state(output, "running", process)
    process.kill()
    process.wait(timeout=30)

    resumed = _launch(script, sleep=0)
    assert resumed.wait(timeout=300) == 0, resumed.stderr.read()[-2000:]

    path = _status_path(output)
    status = json.loads(path.read_text())
    assert status["state"] == "completed"
    assert status["attempt"] == 2

    history = [
        json.loads(line)
        for line in Path(str(path) + "l").read_text().splitlines()
    ]
    assert [(e["event"], e["attempt"]) for e in history] == [
        ("start", 1),
        ("start", 2),
        ("completed", 2),
    ]


@needs_posix_signals
def test_sigusr1_then_sigterm_in_a_fitting_process(tmp_path, output):
    process = _launch(_script(tmp_path, output), sleep=0.5)
    path = _wait_for_state(output, "running", process)

    process.send_signal(signal.SIGUSR1)
    _wait_for_state(output, "timeout_pending", process)
    # SIGUSR1 does not stop the fit.
    assert process.poll() is None

    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 143

    history = [
        json.loads(line)
        for line in Path(str(path) + "l").read_text().splitlines()
    ]
    assert [e.get("signal") for e in history if e["event"] == "timeout_pending"] == [
        "SIGUSR1",
        "SIGTERM",
    ]
    assert json.loads(path.read_text())["state"] == "timeout_pending"
