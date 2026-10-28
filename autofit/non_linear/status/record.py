"""
``FitStatus``: the lifecycle of one fit's status record.

``NonLinearSearch.fit`` creates one per fit (``FitStatus.for_search``, ``None`` when
``general.hpc.status`` is off or the paths store nothing on disk) and the search calls
it at each transition:

- ``start`` when the search starts or resumes (``start_resume_fit``): a new attempt,
  state ``running``, the cadence thread and the signal handlers;
- ``update`` at every ``FitContext.update`` (the sampler's full update): the sampler's
  ``progress`` and the internal state a signal flushes;
- ``completed`` when the fit is marked complete (``_complete_fit``);
- ``skipped`` when the fit was already complete;
- ``failed`` when an exception leaves ``fit`` (the exception is re-raised unchanged);
- ``timeout_pending`` on SIGUSR1 / SIGTERM (``autofit.non_linear.status.signals``);
- ``close`` on every exit of ``fit``.

Every write is best effort: a status record that cannot be written logs a warning and
never fails the fit.
"""
import logging
import threading
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from autonerves import conf

from autofit.non_linear.status import probe, signals
from autofit.non_linear.status.states import (
    FailureClass,
    Product,
    RunState,
    failure_class_of,
)
from autofit.non_linear.status.writer import SCHEMA, StatusWriter

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 60.0


def _hpc_setting(name: str, default):
    try:
        value = conf.instance["general"]["hpc"][name]
    except Exception:
        return default
    return default if value is None else value


def status_enabled() -> bool:
    """
    Whether fits write a status record (``general.hpc.status``, default on).
    """
    return bool(_hpc_setting("status", True))


def status_interval_s() -> float:
    """
    Seconds between cadence writes (``general.hpc.status_interval_s``, default 60).
    """
    return float(_hpc_setting("status_interval_s", DEFAULT_INTERVAL_S))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FitStatus:
    @classmethod
    def for_search(cls, search) -> Optional["FitStatus"]:
        """
        The status record of ``search``'s next fit, or ``None`` when status records are
        off (``general.hpc.status: false``) or the search writes no output folder
        (``NullPaths``, ``DatabasePaths``).
        """
        from autofit.non_linear.paths.directory import DirectoryPaths

        if not status_enabled():
            return None
        if not isinstance(getattr(search, "paths", None), DirectoryPaths):
            return None
        return cls(search)

    def __init__(self, search, interval_s: Optional[float] = None):
        """
        The status record of one fit of ``search``. Nothing is written until the first
        transition, since the output path is only known once ``fit`` has set the
        model on the paths.
        """
        self.search = search
        self.interval_s = status_interval_s() if interval_s is None else interval_s
        # A failure before `fit` sets this fit's model on the paths must not be
        # recorded: the paths would still name a previous fit's output.
        self._model_at_entry = getattr(search.paths, "model", None)

        self.writer: Optional[StatusWriter] = None
        self.state: Optional[RunState] = None
        self.attempt: Optional[int] = None
        self.failure_class: Optional[FailureClass] = None
        self.progress: Optional[Dict] = None
        self.last_internal = None
        self.job = None
        self.started: Optional[str] = None
        self.n_writes = 0

        self._probe = None
        self._progress_origin = None
        self._stop = threading.Event()
        self._thread = None
        self._restore_signals = None
        self._keep_existing = False
        self._dirty = False
        # Set while a signal handler runs: writes then wait at most this long for the
        # writer's lock, which the interrupted main thread may hold.
        self._lock_timeout = None

    @property
    def paths(self):
        return self.search.paths

    def _begin(self):
        """
        Open the record for a new attempt: its number (from the history), the job and
        the resource baseline, and the history's ``start`` line.
        """
        if self.writer is not None:
            return
        self.writer = StatusWriter(self.paths.output_path)
        self.attempt = self.writer.next_attempt()
        self.job = probe.job_from_env()
        self._probe = probe.ResourceProbe(
            cpus=(self.job or {}).get("cpus"), in_job=self.job is not None
        )
        self.started = _now()
        self.writer.append(
            {"event": "start", "attempt": self.attempt, "time": self.started, "job": self.job}
        )

    def start(self):
        """
        The search starts (or resumes): write ``running`` and start the cadence writes
        and the signal handlers. Idempotent within one fit.
        """
        if self.state is not None:
            return
        try:
            self._begin()
        except Exception as exception:
            logger.warning(f"Status record not started ({exception}).")
            return

        self.state = RunState.RUNNING
        self.write()

        self._restore_signals = signals.install(self)

        if self.interval_s and self.interval_s > 0:
            self._thread = threading.Thread(
                target=self._beat, name="autofit-status", daemon=True
            )
            self._thread.start()

    def _beat(self):
        while not self._stop.wait(self.interval_s):
            self.write()

    def update(self, search_internal):
        """
        A full update of the run: keep ``search_internal`` (what a signal flushes) and
        the sampler's ``progress`` (``None`` when the search does not implement it),
        then write.
        """
        if self.state is None:
            return
        self.last_internal = search_internal
        self.progress = self._progress_of(search_internal)
        self.write()

    def _transition(self, state: RunState, **entry):
        self.state = state
        self.write()
        if self.writer is not None:
            try:
                self.writer.append(
                    {"event": str(state), "attempt": self.attempt, "time": _now(), **entry},
                    timeout=self._lock_timeout,
                )
            except Exception as exception:
                logger.warning(f"Status history not written ({exception}).")

    def _progress_of(self, search_internal):
        progress = getattr(self.search, "progress", None)
        if progress is None or search_internal is None:
            return self.progress
        try:
            return progress(search_internal)
        except Exception:
            return None

    def completed(self, search_internal=None):
        """
        The fit is marked complete; ``search_internal`` gives the final progress.
        """
        if self.writer is None:
            return
        self.progress = self._progress_of(search_internal)
        if self.progress is not None and self.progress.get("fraction") is not None:
            self.progress["fraction"] = 1.0
            self.progress["eta_s"] = 0
        self._stop.set()
        self._transition(RunState.COMPLETED)

    def skipped(self):
        """
        The fit was already complete, so the search did not run. This is not a new
        attempt: the history gains a ``skipped`` line under the last attempt's number.
        A record that already says ``completed`` is kept (it holds the run's
        resources); any other record (e.g. one written before status records existed)
        becomes ``skipped``.
        """
        try:
            self.writer = StatusWriter(self.paths.output_path)
            self.attempt = max(self.writer.next_attempt() - 1, 1)
            existing = self.writer.read() or {}
            if existing.get("state") == str(RunState.COMPLETED):
                self._keep_existing = True
                self.state = RunState.SKIPPED
                self.writer.append(
                    {"event": str(RunState.SKIPPED), "attempt": self.attempt, "time": _now()}
                )
                return
            self.started = existing.get("started")
        except Exception as exception:
            logger.warning(f"Status record not written ({exception}).")
            return
        self._transition(RunState.SKIPPED)

    def failed(self, exception: BaseException):
        """
        An exception left ``fit``: write its traceback beside the record, then
        ``failed`` with its failure class. The caller re-raises.

        An exception raised before ``fit`` set its model on the paths (e.g. a model
        with no free parameters) has no output path of its own and is not recorded.
        """
        if self.writer is None and (
            getattr(self.paths, "model", None) is self._model_at_entry
        ):
            return
        try:
            self._begin()
        except Exception as error:
            logger.warning(f"Failure record not written ({error}).")
            return
        self._stop.set()
        self.failure_class = failure_class_of(exception)
        try:
            self.writer.write_traceback(
                "".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                )
            )
        except Exception as error:
            logger.warning(f"Traceback not written ({error}).")
        self._transition(
            RunState.FAILED,
            failure_class=str(self.failure_class),
            error=f"{type(exception).__name__}: {exception}"[:200],
        )

    def flush_checkpoint(self):
        """
        Save the search's resume state from the last internal state an update handed
        over (``Checkpointer.save``; a native resume state, which the backend writes
        itself, is a no-op).
        """
        resume_state = getattr(self.search, "resume_state", None)
        if resume_state is None or self.last_internal is None:
            return
        try:
            resume_state.save(self.paths, self.last_internal)
        except Exception as exception:
            logger.warning(f"Checkpoint flush on signal failed ({exception}).")

    def timeout_pending(self, signal_name: str):
        """
        SIGUSR1 / SIGTERM: flush the checkpoint, then write ``timeout_pending``.
        """
        if self.state is None or self.state in (RunState.COMPLETED, RunState.FAILED):
            return
        self._lock_timeout = 1.0
        try:
            self.flush_checkpoint()
            self._transition(RunState.TIMEOUT_PENDING, signal=signal_name)
        finally:
            self._lock_timeout = None

    def close(self):
        """
        End of ``fit`` (every exit): stop the cadence writes and restore the signal
        handlers. A completed record is written once more, since the output is only
        archived to its ``.zip`` after completion.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._restore_signals is not None:
            self._restore_signals()
            self._restore_signals = None
        if self._dirty or (
            self.state in (RunState.COMPLETED, RunState.SKIPPED)
            and not self._keep_existing
        ):
            self._dirty = False
            self.write()

    def _eta_s(self, fraction) -> Optional[int]:
        """
        Seconds left, from this attempt's rate of progress (so a resumed run's earlier
        progress does not inflate the rate).
        """
        now = time.monotonic()
        if self._progress_origin is None:
            self._progress_origin = (now, fraction)
            return None
        t0, f0 = self._progress_origin
        if fraction <= f0 or fraction >= 1:
            return None
        return int((now - t0) * (1 - fraction) / (fraction - f0))

    def _outputs(self) -> Dict:
        output_path = Path(self.paths.output_path)
        zip_path = Path(f"{output_path}{Product.ZIP}")
        outputs = {"zip": None, "zip_complete": None, "bytes": None}
        if zip_path.exists():
            outputs["zip"] = str(zip_path)
            outputs["bytes"] = zip_path.stat().st_size
            try:
                with zipfile.ZipFile(zip_path) as f:
                    outputs["zip_complete"] = str(Product.COMPLETED) in f.namelist()
            except (OSError, zipfile.BadZipFile):
                outputs["zip_complete"] = False
        for candidate in (
            output_path / "files" / str(Product.DATASET_IDENTITY),
            output_path / str(Product.DATASET_IDENTITY),
        ):
            if candidate.exists():
                outputs["dataset_identity"] = str(candidate)
                break
        if self.state == RunState.FAILED and self.writer.traceback_path.exists():
            outputs["traceback"] = str(self.writer.traceback_path)
        return outputs

    def document(self) -> Dict:
        """
        The status document (``schema`` 1) as it stands now.
        """
        paths = self.paths

        def safe(attribute):
            try:
                value = getattr(paths, attribute)
            except Exception:
                return None
            return None if value is None else str(value)

        progress = dict(self.progress) if self.progress else None
        if progress is not None and progress.get("fraction") is not None:
            progress.setdefault("eta_s", self._eta_s(progress["fraction"]))

        return {
            "schema": SCHEMA,
            "identifier": safe("identifier"),
            "name": safe("name"),
            "path_prefix": safe("path_prefix"),
            "search": type(self.search).__name__,
            "state": None if self.state is None else str(self.state),
            "attempt": self.attempt,
            "failure_class": None
            if self.failure_class is None
            else str(self.failure_class),
            "job": self.job,
            "progress": progress,
            "resources": self._probe.resources() if self._probe else None,
            "outputs": self._outputs(),
            "updated": _now(),
            "started": self.started,
        }

    def write(self):
        """
        Write the current document; best effort. Inside a signal handler the wait for
        the writer's lock is bounded (the interrupted main thread may hold it); a write
        that gives up is redone after the write it interrupted.
        """
        if self.writer is None or self.state is None:
            return
        try:
            written = self.writer.write(self.document(), timeout=self._lock_timeout)
        except Exception as exception:
            logger.warning(f"Status record not written ({exception}).")
            return
        if not written:
            self._dirty = True
            return
        self.n_writes += 1
        if self._dirty and self._lock_timeout is None:
            self._dirty = False
            self.write()


@contextmanager
def status_recorded(search):
    """
    Run one ``fit`` of ``search`` inside its status record: the record is kept on
    ``search._fit_status`` for the fit's transitions (``status_notify``), an exception
    leaving the fit is recorded (``failed`` + traceback) and re-raised unchanged, and
    the record is closed on every exit. A fit nested in another fit of the same
    search restores the outer record afterwards.
    """
    outer = getattr(search, "_fit_status", None)
    fit_status = search._fit_status = FitStatus.for_search(search)
    try:
        yield fit_status
    except Exception as exception:
        if fit_status is not None:
            fit_status.failed(exception)
        raise
    finally:
        if fit_status is not None:
            fit_status.close()
        search._fit_status = outer


def status_notify(search, transition: str, *args):
    """
    Call ``transition`` (``start``, ``update``, ``completed``, ``skipped``) on the
    status record of ``search``'s current fit, if it has one.
    """
    fit_status = getattr(search, "_fit_status", None)
    if fit_status is not None:
        getattr(fit_status, transition)(*args)
