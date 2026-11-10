"""Experimental checkpoint batching for the native Nautilus 1.0.5 schema.

Only persistence changes. Full native snapshots include every dirty shell,
bound, cache and RNG state; incremental global metadata is never published over
older shell data. Atomic replacement protects the previous checkpoint against
process interruption. This is not a power-loss or network-filesystem guarantee.
"""

from importlib.metadata import version
from numbers import Integral
import os
from pathlib import Path
import tempfile

from nautilus import Sampler


def validate_interval(interval):
    if isinstance(interval, bool) or not isinstance(interval, Integral) or interval < 1:
        raise ValueError("checkpoint_interval_batches must be a positive integer")
    return int(interval)


class BatchedCheckpointSampler(Sampler):
    """Defer incremental writes, retaining native full-write boundaries.

    An interruption can lose completed batches since the last published
    snapshot plus work in flight. The interval counts native shell-update
    notifications, not wall time or PyAutoFit output updates. No sampler RNG
    calls are made by this policy.
    """

    def __init__(self, *args, checkpoint_interval_batches=5, **kwargs):
        self.checkpoint_interval_batches = validate_interval(checkpoint_interval_batches)
        if self.checkpoint_interval_batches == 1:
            raise ValueError("Use the native Sampler for checkpoint interval 1")
        if version("nautilus-sampler") != "1.0.5":
            raise ValueError("Checkpoint batching is validated only for nautilus-sampler 1.0.5")
        self._pending_checkpoint_batches = 0
        super().__init__(*args, **kwargs)

    def write(self, filepath, overwrite=False):
        """Publish a complete native snapshot without unlinking the good file."""
        filepath = Path(filepath)
        if filepath.suffix not in (".h5", ".hdf5"):
            raise ValueError("File ending must be '.h5' or '.hdf5'.")
        if filepath.exists() and not overwrite:
            raise RuntimeError(f"File {filepath} already exists.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{filepath.name}.", suffix=".hdf5", dir=filepath.parent)
        os.close(fd)
        temporary = Path(name)
        # Native write creates its file exclusively. Reserve a unique name,
        # then let the native writer own creation and HDF5 close/flush.
        temporary.unlink()
        try:
            super().write(temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, filepath)
        finally:
            temporary.unlink(missing_ok=True)
        if self.filepath is not None and filepath.absolute() == Path(self.filepath).absolute():
            self._pending_checkpoint_batches = 0

    def write_shell_update(self, filepath, shell):
        # Never write a single shell alongside global metadata newer than the
        # other deferred shells. An explicit export to another path is immediate.
        if self.filepath is None or Path(filepath).absolute() != Path(self.filepath).absolute():
            return self.write(filepath, overwrite=True)
        self._pending_checkpoint_batches += 1
        if self._pending_checkpoint_batches >= self.checkpoint_interval_batches:
            self.write(filepath, overwrite=True)

    def flush_checkpoint(self):
        if self.filepath is not None and self._pending_checkpoint_batches:
            self.write(self.filepath, overwrite=True)

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        # Deliberately not a finally block: exceptions may leave half-mutated
        # in-memory state. Resume from the last complete published file instead.
        self.flush_checkpoint()
        return result
