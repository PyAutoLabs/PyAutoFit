"""
The single vocabulary of run states, failure classes and products shared by the per-fit
status record (``autofit.non_linear.status``), the results library and any campaign
collector.

A fit writes only the states in ``WRITTEN_STATES``. ``timeout``, ``oom`` and
``node_fail`` are assigned by a collector from the scheduler's accounting (e.g.
``sacct``), since a process killed by the scheduler cannot record its own death.

All three enums subclass ``str``, so a member compares equal to (and serialises as) its
plain value: ``RunState.RUNNING == "running"``.
"""
from enum import Enum


class RunState(str, Enum):
    PENDING = "pending"
    """Written by a batch trap before Python starts; never by the fit itself."""
    RUNNING = "running"
    """The fit is executing (written at start and at every cadence write)."""
    COMPLETED = "completed"
    """The fit finished and its output is marked complete."""
    FAILED = "failed"
    """A Python exception left ``fit``; ``failure_class`` says which kind."""
    TIMEOUT_PENDING = "timeout_pending"
    """SIGUSR1 / SIGTERM was received and the checkpoint flushed: the job is about to
    hit its time limit (or be stopped) and should be resubmitted to resume."""
    SKIPPED = "skipped"
    """The fit was already complete, so the search did not run again."""

    # Collector-only states, assigned from scheduler accounting.
    TIMEOUT = "timeout"
    OOM = "oom"
    NODE_FAIL = "node_fail"

    def __str__(self):
        return self.value


WRITTEN_STATES = frozenset(
    {
        RunState.PENDING,
        RunState.RUNNING,
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.TIMEOUT_PENDING,
        RunState.SKIPPED,
    }
)

COLLECTOR_STATES = frozenset({RunState.TIMEOUT, RunState.OOM, RunState.NODE_FAIL})

TERMINAL_STATES = frozenset(
    {
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.SKIPPED,
        RunState.TIMEOUT,
        RunState.OOM,
        RunState.NODE_FAIL,
    }
)


class FailureClass(str, Enum):
    EXCEPTION = "exception"
    """Any exception not classified more precisely (the default)."""
    BAD_INPUT = "bad_input"
    """The inputs of the fit are invalid (data, model or settings)."""
    MISSING_SEED = "missing_seed"
    """A result this fit is chained from (its seed) does not exist."""
    UPSTREAM_INCOMPLETE = "upstream_incomplete"
    """A result this fit is chained from exists but is not complete."""

    def __str__(self):
        return self.value


class ResultState(str, Enum):
    """
    The state of one result on disk, as the results index (``autofit.results``)
    classifies it after the fact. Where a ``RunState`` is what a running fit says about
    itself, a ``ResultState`` is what its output says once read back: it is assigned
    from the status record when one exists and from the output folder or zip otherwise.
    """

    COMPLETED = "completed"
    """The output (folder or zip) carries ``.completed`` and reads back cleanly."""
    RUNNING = "running"
    """The status record says the fit is running and was written recently."""
    INTERRUPTED_VALID_CKPT = "interrupted_valid_ckpt"
    """The fit stopped before completing and left a checkpoint it can resume from."""
    CORRUPT_CKPT = "corrupt_ckpt"
    """The fit stopped before completing and its checkpoint cannot be read (or was
    quarantined to ``search_internal.failed_<utc>/`` with nothing written since)."""
    CORRUPT_ZIP = "corrupt_zip"
    """The output zip cannot be read, or fails its CRC check."""
    FAILED = "failed"
    """The status record says a Python exception ended the fit."""
    SUPERSEDED = "superseded"
    """A completed result that a newer completed result of the same key replaces
    (``Aggregator.latest_per``)."""
    NEVER_STARTED = "never_started"
    """An output folder with no checkpoint and no status record, so a rerun starts the
    search from scratch; with a declared grid (results-library phase 2), also an
    expected result with no output at all."""

    def __str__(self):
        return self.value


class Product(str, Enum):
    """
    The files a fit's status record names, relative to its ``output_path``.
    """

    STATUS = ".status.json"
    """Suffix of ``<output_path>.status.json``, the current status record."""
    HISTORY = ".status.jsonl"
    """Suffix of ``<output_path>.status.jsonl``, the append-only attempt history."""
    TRACEBACK = ".traceback.txt"
    """Suffix of ``<output_path>.traceback.txt``, the traceback of a failed attempt."""
    ZIP = ".zip"
    """Suffix of ``<output_path>.zip``, the fit's archived output."""
    COMPLETED = ".completed"
    """The marker file inside ``output_path`` written when the fit completes."""
    DATASET_IDENTITY = "dataset_identity.json"
    """The record of the fit's inputs (written by the results library)."""

    def __str__(self):
        return self.value


def failure_class_of(exception: BaseException) -> FailureClass:
    """
    The failure class of ``exception``: the ``failure_class`` attribute an exception may
    carry (a ``FailureClass`` or its value), else ``FailureClass.EXCEPTION``.
    """
    value = getattr(exception, "failure_class", None)
    try:
        return FailureClass(value)
    except ValueError:
        return FailureClass.EXCEPTION
