import os
import time
from pathlib import Path
from typing import Optional

from autofit.tools.util import open_atomic


class Timer:

    def __init__(self, timer_path: str, legacy_path: Optional[str] = None):
        """Times the run-time of the non-linear searches, by outputting a start-time file to the hard-disk and using
        this to determine the total run time when a `NonLinearSearch` update is performed.

        The files live in ``files/`` (``AbstractPaths.timer_path``), not ``files/search_internal/``: that folder
        is removed when a fit completes without ``output.search_internal`` and moved aside when its checkpoint is
        quarantined, either of which would lose the timing. Output written before the move is still read from
        ``legacy_path``.

        Parameters
        ----------
        timer_path
            The directory in which the timer should save results
        legacy_path
            The directory older output wrote the timer files to (``<timer_path>/search_internal`` by default),
            read when ``timer_path`` holds none.
        """

        self.timer_path = Path(timer_path)
        self.legacy_path = (
            Path(legacy_path) if legacy_path is not None else self.timer_path / "search_internal"
        )
        os.makedirs(
            timer_path,
            exist_ok=True
        )

    def _read(self, name: str) -> Optional[str]:
        for directory in (self.timer_path, self.legacy_path):
            try:
                with open(directory / name, "r") as f:
                    return f.read()
            except FileNotFoundError:
                pass
        return None

    def start(self):
        """
        Record the start time of a `NonLinearSearch` as universal date time, so that the run-time of the search can be
        recorded. A resumed search keeps its original start time, including one written by older output to the
        legacy location, which is copied forward.
        """

        start_time = self.start_time

        try:
            float(start_time)
        except (TypeError, ValueError):
            start_time = str(time.time())

        try:
            with open(self.timer_path / ".start_time") as f:
                if f.read() == start_time:
                    return
        except FileNotFoundError:
            pass

        with open_atomic(self.timer_path / ".start_time", "w+") as f:
            f.write(start_time)

    def update(self):
        """
        Update the timer of the `NonLinearSearch` so it reflections how long the `NonLinearSearch` ahs been running.
        """

        try:
            execution_time = str(time.time() - float(self.start_time))
        except (TypeError, ValueError):
            return

        with open_atomic(self.timer_path / ".time", "w+") as f:
            f.write(execution_time)

    @property
    def start_time(self):
        """
        Load the start time written to hard disk from the .start_time file.
        """
        return self._read(".start_time")

    @property
    def time(self):
        """
        Load the total time of the `NonLinearSearch` written to hard disk fom the .start_time file.
        """
        return self._read(".time")
