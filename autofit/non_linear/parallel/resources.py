"""
The machine probes and the ``number_of_cores`` validation behind ``number_of_cores="auto"``.

- ``validate_number_of_cores`` is the one check every search, ``SettingsSearch``, the grid
  search and sensitivity mapping apply to a requested count: an int >= 1, ``None`` (1) or
  ``"auto"`` (any case).
- ``detected_cores`` is the cores this process may use (``SLURM_CPUS_PER_TASK``, then the
  CPU affinity, then ``os.cpu_count()``; ``autonerves.environment.detect_cores``).
- ``available_memory_gb`` is the memory a fit may still allocate: ``MemAvailable`` from
  ``/proc/meminfo``, lowered to the cgroup v2 headroom (``memory.max - memory.current``)
  when the process runs under a memory limit, falling back to the total RAM. Every file it
  reads may be missing (macOS, Windows, cgroup v1), in which case it falls back quietly.
"""
import logging
import numbers
import os
from typing import Optional, Union

logger = logging.getLogger(__name__)

AUTO = "auto"

_GB = 1024 ** 3


def validate_number_of_cores(value, where: str = "number_of_cores") -> Union[int, str]:
    """
    The validated ``number_of_cores`` request: an int >= 1, or ``"auto"``.

    Parameters
    ----------
    value
        The request: an int >= 1, ``None`` (meaning 1) or ``"auto"`` (case-insensitive).
    where
        What is being configured, as it should read in the error.

    Returns
    -------
    ``value`` as an int, or the string ``"auto"``.

    Raises
    ------
    ValueError
        For anything else (0, negative counts, floats, bools, other strings).
    """
    if value is None:
        return 1

    if isinstance(value, str):
        if value.strip().lower() == AUTO:
            return AUTO
    elif isinstance(value, numbers.Integral) and not isinstance(value, bool):
        if int(value) >= 1:
            return int(value)

    raise ValueError(
        f"{where}: number_of_cores={value!r} is not valid. Use a whole number of "
        f"cores >= 1 (e.g. number_of_cores=4), or number_of_cores=\"auto\" to use "
        f"every core this process may use (the count is resolved when the fit starts)."
    )


def is_auto(value) -> bool:
    """
    Whether ``value`` is the ``"auto"`` request (any case).
    """
    return isinstance(value, str) and value.strip().lower() == AUTO


def detected_cores() -> int:
    """
    The number of CPU cores this process may use (``autonerves.environment.detect_cores``:
    ``SLURM_CPUS_PER_TASK``, then the CPU affinity, then ``os.cpu_count()``; at least 1).
    """
    from autonerves.environment import detect_cores

    return detect_cores()


def resolve_job_cores(value, where: str) -> int:
    """
    A validated job-pool worker count for the grid search and sensitivity mapping, with
    ``"auto"`` resolved to the detected cores (their jobs are whole fits, so no analysis
    is known to cap the count by).
    """
    value = validate_number_of_cores(value, where=where)
    if value == AUTO:
        return detected_cores()
    return value


def _read_meminfo(meminfo_path: str) -> dict:
    """
    ``/proc/meminfo`` as {field: bytes}; empty when the file is missing or unreadable.
    """
    values = {}
    try:
        with open(meminfo_path) as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if not parts:
                    continue
                try:
                    number = float(parts[0])
                except ValueError:
                    continue
                unit = parts[1].lower() if len(parts) > 1 else ""
                values[key.strip()] = number * 1024 if unit == "kb" else number
    except OSError:
        return {}
    return values


def _read_int_file(path: str) -> Optional[int]:
    """
    An integer file such as cgroup ``memory.max``; ``None`` when missing, unreadable or
    ``max`` (no limit).
    """
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _cgroup_headroom_bytes(cgroup_root: str, proc_cgroup_path: str) -> Optional[int]:
    """
    The memory this process's cgroup v2 may still allocate (``memory.max`` minus
    ``memory.current``), or ``None`` when there is no cgroup v2 memory limit.

    The process's own cgroup (the ``0::<path>`` line of ``/proc/self/cgroup``, e.g. a
    Slurm job step) is tried first, then the cgroup root (a container's namespace root).
    """
    candidates = []
    try:
        with open(proc_cgroup_path) as f:
            for line in f:
                if line.startswith("0::"):
                    relative = line[3:].strip().lstrip("/")
                    if relative:
                        candidates.append(os.path.join(cgroup_root, relative))
    except OSError:
        pass
    candidates.append(cgroup_root)

    for directory in candidates:
        limit = _read_int_file(os.path.join(directory, "memory.max"))
        if limit is None:
            continue
        current = _read_int_file(os.path.join(directory, "memory.current")) or 0
        return max(0, limit - current)

    return None


def available_memory_gb(
    meminfo_path: str = "/proc/meminfo",
    cgroup_root: str = "/sys/fs/cgroup",
    proc_cgroup_path: str = "/proc/self/cgroup",
) -> Optional[float]:
    """
    The memory, in GB (GiB), this process may still allocate.

    ``MemAvailable`` from ``/proc/meminfo`` (the kernel's estimate of memory available
    without swapping), lowered to the cgroup v2 headroom when the process runs under a
    lower memory limit (a Slurm job, a container). When ``MemAvailable`` is unknown the
    total RAM is used (``MemTotal``, then ``autonerves.environment.detect_ram_gb``).
    ``None`` when nothing can be read.

    Parameters
    ----------
    meminfo_path, cgroup_root, proc_cgroup_path
        The files read, overridable for tests.
    """
    meminfo = _read_meminfo(meminfo_path)

    available = meminfo.get("MemAvailable")
    if available is None:
        available = meminfo.get("MemTotal")

    cgroup = _cgroup_headroom_bytes(cgroup_root, proc_cgroup_path)
    if cgroup is not None and (available is None or cgroup < available):
        available = cgroup

    if available is not None:
        return available / _GB

    try:
        from autonerves.environment import detect_ram_gb

        return detect_ram_gb()
    except Exception:
        logger.debug("Could not read the total RAM", exc_info=True)
        return None
