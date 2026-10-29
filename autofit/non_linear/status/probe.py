"""
Cheap, fail-soft probes of the resources a fit is using and of the scheduler job it runs
in, for the per-fit status record.

Every probe returns ``None`` (or omits a field) when its source is unavailable -- macOS,
Windows, a laptop outside any scheduler, a container without cgroup v2 -- and never
raises. Nothing here is a hard dependency: NVML (``pynvml``) is used only when it is
importable and a GPU is allocated to the process.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Scheduler environment variables, per scheduler, mapped onto the status record's
# ``job`` fields. A scheduler is detected by the first variable of its mapping
# (``detect``); a new scheduler (PBS, LSF) is one more entry.
SCHEDULER_ENV = {
    "slurm": {
        "detect": "SLURM_JOB_ID",
        "job_id": ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"),
        "task_id": ("SLURM_ARRAY_TASK_ID",),
        "node": ("SLURMD_NODENAME",),
        "cpus": ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"),
        "mem_mb": ("SLURM_MEM_PER_NODE",),
        "mem_per_cpu_mb": ("SLURM_MEM_PER_CPU",),
        "restart_count": ("SLURM_RESTART_COUNT",),
        "start_time": ("SLURM_JOB_START_TIME",),
        "end_time": ("SLURM_JOB_END_TIME",),
        "gpus": ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"),
    },
}


def _first(env, names) -> Optional[str]:
    for name in names:
        value = env.get(name)
        if value not in (None, ""):
            return value
    return None


def _int(value) -> Optional[int]:
    try:
        return int(str(value).split("(")[0])
    except (TypeError, ValueError):
        return None


def job_from_env(env=None) -> Optional[Dict]:
    """
    The scheduler job this process runs in, from the environment (``SCHEDULER_ENV``),
    or ``None`` outside any known scheduler.

    ``time_limit_s`` is derived from the job's start and end times when the scheduler
    exports them (Slurm >= 21.08); ``mem_mb`` falls back to the per-CPU memory times
    the CPUs.
    """
    env = os.environ if env is None else env

    for scheduler, mapping in SCHEDULER_ENV.items():
        if not env.get(mapping["detect"]):
            continue

        def field(name):
            return _first(env, mapping.get(name, ()))

        cpus = _int(field("cpus"))
        mem_mb = _int(field("mem_mb"))
        if mem_mb is None:
            per_cpu = _int(field("mem_per_cpu_mb"))
            if per_cpu is not None and cpus is not None:
                mem_mb = per_cpu * cpus

        start, end = _int(field("start_time")), _int(field("end_time"))
        time_limit_s = end - start if start is not None and end is not None else None

        return {
            "scheduler": scheduler,
            "job_id": field("job_id"),
            "task_id": _int(field("task_id")),
            "node": field("node"),
            "cpus": cpus,
            "mem_mb": mem_mb,
            "time_limit_s": time_limit_s,
            "restart_count": _int(field("restart_count")) or 0,
        }

    return None


def gpu_allocated(env=None) -> bool:
    """
    Whether a GPU is allocated to this process: a scheduler GPU list or a non-empty
    ``CUDA_VISIBLE_DEVICES`` that does not hide every device.
    """
    env = os.environ if env is None else env
    for mapping in SCHEDULER_ENV.values():
        if _first(env, mapping.get("gpus", ())):
            return True
    visible = env.get("CUDA_VISIBLE_DEVICES")
    return visible not in (None, "", "-1", "NoDevFiles")


def available_cpus() -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


_CPU_MODEL = []


def cpu_model() -> Optional[str]:
    """
    The CPU model name from ``/proc/cpuinfo`` (Linux), cached; ``None`` elsewhere.
    """
    if not _CPU_MODEL:
        model = None
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        _CPU_MODEL.append(model)
    return _CPU_MODEL[0]


def cpu_seconds() -> Optional[float]:
    """
    User + system CPU seconds of this process and its reaped children
    (``resource.getrusage``). Children still running (e.g. a live pool's workers) are
    counted only once joined.
    """
    try:
        import resource

        total = 0.0
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
            usage = resource.getrusage(who)
            total += usage.ru_utime + usage.ru_stime
        return total
    except Exception:
        try:
            times = os.times()
            return times.user + times.system + times.children_user + times.children_system
        except Exception:
            return None


def _cgroup_memory_peak() -> Optional[int]:
    """
    The cgroup v2 ``memory.peak`` of this process's cgroup, in bytes (Linux only).
    """
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                if line.startswith("0::"):
                    relative = line.strip()[3:].lstrip("/")
                    peak = Path("/sys/fs/cgroup") / relative / "memory.peak"
                    return int(peak.read_text().strip())
    except (OSError, ValueError):
        pass
    return None


def peak_rss(use_cgroup: bool = True) -> Dict:
    """
    The peak resident memory, in MB, and where it came from: ``cgroup`` (cgroup v2
    ``memory.peak``, which covers every process of the job step) else ``getrusage``
    (the larger of this process and its largest reaped child).

    ``use_cgroup=False`` skips the cgroup: outside a scheduler job a process's cgroup
    is a login session or a whole WSL / container instance, whose peak is not the
    fit's.
    """
    peak = _cgroup_memory_peak() if use_cgroup else None
    if peak is not None:
        return {"peak_rss_mb": round(peak / 2**20), "peak_rss_source": "cgroup"}

    try:
        import resource
        import sys

        maxrss = max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        # Kilobytes on Linux, bytes on macOS.
        scale = 2**20 if sys.platform == "darwin" else 2**10
        return {"peak_rss_mb": round(maxrss / scale), "peak_rss_source": "getrusage"}
    except Exception:
        return {"peak_rss_mb": None, "peak_rss_source": None}


_NVML = []


def gpu() -> Optional[Dict]:
    """
    The first visible GPU's model, used memory (MB) and utilisation (%), through NVML;
    ``None`` when no GPU is allocated or ``pynvml`` is not importable.
    """
    if not gpu_allocated():
        return None
    try:
        import pynvml

        if not _NVML:
            pynvml.nvmlInit()
            _NVML.append(True)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilisation = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "model": name,
            "mem_used_mb": round(memory.used / 2**20),
            "util": utilisation.gpu,
        }
    except Exception:
        return None


class ResourceProbe:
    def __init__(self, cpus: Optional[int] = None, in_job: bool = False):
        """
        Resource use of one fit attempt: wall and CPU seconds are measured from the
        probe's creation (so earlier work in the same process is not counted), peak
        memory is the process's (or the cgroup's) lifetime peak.

        Parameters
        ----------
        cpus
            The CPUs the job was given, for ``cpu_efficiency``; defaults to the CPUs
            this process may run on.
        in_job
            Whether the process runs in a scheduler job, whose cgroup is the job's
            (so its ``memory.peak`` is used).
        """
        self.cpus = cpus or available_cpus()
        self.in_job = in_job
        self._wall_start = time.monotonic()
        self._cpu_start = cpu_seconds()

    def resources(self) -> Dict:
        wall_s = time.monotonic() - self._wall_start
        cpu_now = cpu_seconds()
        cpu_s = (
            cpu_now - self._cpu_start
            if cpu_now is not None and self._cpu_start is not None
            else None
        )
        efficiency = (
            round(cpu_s / (wall_s * self.cpus), 3)
            if cpu_s is not None and self.cpus and wall_s > 0
            else None
        )
        return {
            "wall_s": round(wall_s, 1),
            "cpu_s": None if cpu_s is None else round(cpu_s, 1),
            "cpu_efficiency": efficiency,
            **peak_rss(use_cgroup=self.in_job),
            "gpu": gpu(),
            "cpu_model": cpu_model(),
        }
