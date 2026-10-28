"""
The status record's parts on their own: the state vocabulary, the size-capped atomic
writer and its history, the fail-soft probes and the signal handlers.
"""
import json
import os
import signal
import sys

import pytest

from autofit.non_linear.status import probe, signals
from autofit.non_linear.status.states import (
    COLLECTOR_STATES,
    WRITTEN_STATES,
    FailureClass,
    Product,
    RunState,
    failure_class_of,
)
from autofit.non_linear.status.writer import MAX_BYTES, StatusWriter, compact


def test_vocabulary():
    assert [state.value for state in WRITTEN_STATES] and RunState.RUNNING == "running"
    assert {str(state) for state in COLLECTOR_STATES} == {"timeout", "oom", "node_fail"}
    assert not WRITTEN_STATES & COLLECTOR_STATES
    assert {str(c) for c in FailureClass} == {
        "exception",
        "bad_input",
        "missing_seed",
        "upstream_incomplete",
    }
    assert json.dumps({"state": RunState.TIMEOUT_PENDING}) == '{"state": "timeout_pending"}'
    assert str(Product.STATUS) == ".status.json"


def test_failure_class_of():
    assert failure_class_of(ValueError()) is FailureClass.EXCEPTION

    error = ValueError()
    error.failure_class = "missing_seed"
    assert failure_class_of(error) is FailureClass.MISSING_SEED

    error.failure_class = "not a class"
    assert failure_class_of(error) is FailureClass.EXCEPTION


def test_compact_trims_optional_fields_to_the_cap():
    document = {
        "schema": 1,
        "path_prefix": "x" * 1900,
        "progress": {"n_like": 1, "log_z": 1.0, "n_eff": 2.0, "f_live": 0.1},
        "resources": {"wall_s": 1.0, "gpu": {"model": "y" * 200}, "cpu_model": "z" * 200},
    }
    text = compact(document)

    assert len(text.encode()) <= MAX_BYTES
    loaded = json.loads(text)
    assert loaded["progress"] == {"n_like": 1}
    assert loaded["resources"] == {"wall_s": 1.0}
    assert loaded["path_prefix"] == "x" * 1900


def test_compact_cuts_a_pathological_path_prefix():
    text = compact({"schema": 1, "path_prefix": "a" * 5000 + "tail"})

    assert len(text.encode()) <= MAX_BYTES
    assert json.loads(text)["path_prefix"].endswith("tail")


def test_writer_files_are_siblings_and_history_counts_attempts(tmp_path):
    output_path = tmp_path / "prefix" / "name" / "identifier"
    writer = StatusWriter(output_path)

    assert writer.status_path == tmp_path / "prefix" / "name" / "identifier.status.json"
    assert writer.next_attempt() == 1

    writer.write({"state": "running"})
    assert writer.read() == {"state": "running"}
    assert not list(writer.status_path.parent.glob("*.tmp"))

    writer.append({"event": "start", "attempt": 1})
    writer.append({"event": "failed", "attempt": 1})
    # A line a killed process left truncated is skipped.
    with open(writer.history_path, "a") as f:
        f.write('{"event": "sta')
    assert writer.next_attempt() == 2


def test_writer_gives_up_on_a_held_lock_with_a_timeout(tmp_path):
    writer = StatusWriter(tmp_path / "out")
    writer._lock.acquire()
    try:
        assert writer.write({"state": "running"}, timeout=0.01) is None
        assert writer.append({"event": "x"}, timeout=0.01) is False
    finally:
        writer._lock.release()
    assert not writer.status_path.exists()


SLURM_ENV = {
    "SLURM_JOB_ID": "351102",
    "SLURM_ARRAY_JOB_ID": "351085",
    "SLURM_ARRAY_TASK_ID": "17",
    "SLURMD_NODENAME": "ral-n12",
    "SLURM_CPUS_PER_TASK": "4",
    "SLURM_MEM_PER_CPU": "3072",
    "SLURM_RESTART_COUNT": "1",
    "SLURM_JOB_START_TIME": "1000",
    "SLURM_JOB_END_TIME": "65800",
}


def test_job_from_slurm_env():
    assert probe.job_from_env(SLURM_ENV) == {
        "scheduler": "slurm",
        "job_id": "351085",
        "task_id": 17,
        "node": "ral-n12",
        "cpus": 4,
        "mem_mb": 12288,
        "time_limit_s": 64800,
        "restart_count": 1,
    }


def test_job_outside_a_scheduler_is_none():
    assert probe.job_from_env({}) is None
    assert probe.job_from_env({"SLURM_JOB_ID": ""}) is None


def test_gpu_allocated():
    assert not probe.gpu_allocated({})
    assert not probe.gpu_allocated({"CUDA_VISIBLE_DEVICES": "-1"})
    assert probe.gpu_allocated({"CUDA_VISIBLE_DEVICES": "0"})
    assert probe.gpu_allocated({"SLURM_JOB_GPUS": "0"})


def test_probes_fail_soft(monkeypatch):
    monkeypatch.setattr(probe, "_cgroup_memory_peak", lambda: None)
    rss = probe.peak_rss()
    assert rss["peak_rss_source"] in ("getrusage", None)

    monkeypatch.setattr(probe, "gpu_allocated", lambda env=None: False)
    assert probe.gpu() is None

    resources = probe.ResourceProbe(cpus=2).resources()
    assert set(resources) == {
        "wall_s",
        "cpu_s",
        "cpu_efficiency",
        "peak_rss_mb",
        "peak_rss_source",
        "gpu",
        "cpu_model",
    }
    assert resources["wall_s"] >= 0


def test_cgroup_used_only_inside_a_job(monkeypatch):
    monkeypatch.setattr(probe, "_cgroup_memory_peak", lambda: 3 * 2**30)
    assert probe.peak_rss(use_cgroup=True) == {
        "peak_rss_mb": 3072,
        "peak_rss_source": "cgroup",
    }
    assert probe.peak_rss(use_cgroup=False)["peak_rss_source"] != "cgroup"


class FakeStatus:
    def __init__(self):
        self.signals = []

    def timeout_pending(self, name):
        self.signals.append(name)


needs_usr1 = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="no SIGUSR1 on this platform"
)


@needs_usr1
def test_sigusr1_records_timeout_pending_and_restores_the_previous_handler():
    before = signal.getsignal(signal.SIGUSR1)
    status = FakeStatus()

    restore = signals.install(status)
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert status.signals == ["SIGUSR1"]
    finally:
        restore()

    assert signal.getsignal(signal.SIGUSR1) == before


@needs_usr1
def test_sigterm_records_then_exits_143():
    before = signal.getsignal(signal.SIGTERM)
    status = FakeStatus()

    restore = signals.install(status)
    try:
        with pytest.raises(SystemExit) as exit_info:
            os.kill(os.getpid(), signal.SIGTERM)
            # The handler runs at the next bytecode boundary.
            for _ in range(1000):
                pass
        assert exit_info.value.code == 143
        assert status.signals == ["SIGTERM"]
    finally:
        restore()

    assert signal.getsignal(signal.SIGTERM) == before


def test_no_handlers_outside_the_main_thread():
    import threading

    installed = []

    def target():
        installed.append(signals.can_install())

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    assert installed == [False]
