"""
The ``Scheduler`` protocol every campaign adapter implements, and the records it
returns.

A scheduler state maps onto the shared run-state vocabulary
(``autofit.non_linear.status.states.RunState``): ``TIMEOUT``, ``OUT_OF_MEMORY`` and
``NODE_FAIL`` become the collector-only states ``timeout``, ``oom`` and ``node_fail``,
which a fit cannot write for itself. A few scheduler outcomes have no run state, because
no fit ran to an end: they keep a ``scheduler_state`` of ``cancelled``, ``requeued`` or
``preempted``.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

from autofit.non_linear.status.states import RunState

# Scheduler outcomes outside the run-state vocabulary.
CANCELLED = "cancelled"
REQUEUED = "requeued"
PREEMPTED = "preempted"
UNKNOWN = "unknown"

_SLURM_STATES = {
    "PENDING": RunState.PENDING,
    "CONFIGURING": RunState.PENDING,
    "REQUEUE_HOLD": RunState.PENDING,
    "REQUEUE_FED": RunState.PENDING,
    "RESIZING": RunState.RUNNING,
    "RUNNING": RunState.RUNNING,
    "COMPLETING": RunState.RUNNING,
    "SUSPENDED": RunState.RUNNING,
    "STOPPED": RunState.RUNNING,
    "SIGNALING": RunState.RUNNING,
    "STAGE_OUT": RunState.RUNNING,
    "COMPLETED": RunState.COMPLETED,
    "FAILED": RunState.FAILED,
    "TIMEOUT": RunState.TIMEOUT,
    "DEADLINE": RunState.TIMEOUT,
    "OUT_OF_MEMORY": RunState.OOM,
    "NODE_FAIL": RunState.NODE_FAIL,
    "BOOT_FAIL": RunState.NODE_FAIL,
    "CANCELLED": CANCELLED,
    "REVOKED": CANCELLED,
    "SPECIAL_EXIT": RunState.FAILED,
    "REQUEUED": REQUEUED,
    "PREEMPTED": PREEMPTED,
}

_SQUEUE_COMPACT = {
    "PD": "PENDING",
    "R": "RUNNING",
    "CG": "COMPLETING",
    "CF": "CONFIGURING",
    "S": "SUSPENDED",
    "ST": "STOPPED",
    "CD": "COMPLETED",
    "F": "FAILED",
    "TO": "TIMEOUT",
    "OOM": "OUT_OF_MEMORY",
    "NF": "NODE_FAIL",
    "BF": "BOOT_FAIL",
    "CA": "CANCELLED",
    "PR": "PREEMPTED",
    "RQ": "REQUEUED",
    "RH": "REQUEUE_HOLD",
    "SE": "SPECIAL_EXIT",
    "DL": "DEADLINE",
}


def run_state_of(scheduler_state: Optional[str]):
    """
    The run state (or ``cancelled`` / ``requeued`` / ``preempted`` / ``unknown``) of a
    Slurm state, long (``CANCELLED by 16309``) or compact (``CA``).
    """
    if not scheduler_state:
        return UNKNOWN
    word = scheduler_state.strip().split()[0].rstrip("+").upper()
    word = _SQUEUE_COMPACT.get(word, word)
    return _SLURM_STATES.get(word, UNKNOWN)


@dataclass
class TaskRecord:
    """
    One scheduler task (an array task or a plain job), as accounting reports it.
    Memory is in MB, times in seconds.
    """

    job_id: str
    task_id: Optional[int] = None
    name: Optional[str] = None
    scheduler_state: Optional[str] = None
    exit_code: Optional[str] = None
    partition: Optional[str] = None
    node: Optional[str] = None
    submit: Optional[datetime] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    elapsed_s: Optional[int] = None
    time_limit_s: Optional[int] = None
    cpus: Optional[int] = None
    req_mem_mb: Optional[float] = None
    max_rss_mb: Optional[float] = None
    total_cpu_s: Optional[float] = None
    workdir: Optional[str] = None
    req_tres: Optional[str] = None
    alloc_tres: Optional[str] = None
    raw_job_id: Optional[str] = None
    attempts: int = 1

    @property
    def key(self) -> str:
        return self.job_id if self.task_id is None else f"{self.job_id}_{self.task_id}"

    @property
    def state(self):
        return run_state_of(self.scheduler_state)

    @property
    def gres(self) -> Optional[str]:
        """
        The GPU part of the request, read from TRES (never from the partition name).
        """
        for tres in (self.req_tres, self.alloc_tres):
            if tres and "gres/gpu" in tres:
                return ",".join(
                    item for item in tres.split(",") if item.startswith("gres/gpu")
                )
        return None

    @property
    def cpu_efficiency(self) -> Optional[float]:
        if not self.elapsed_s or not self.cpus or self.total_cpu_s is None:
            return None
        return self.total_cpu_s / (self.elapsed_s * self.cpus)

    @property
    def cpu_hours(self) -> float:
        if not self.elapsed_s or not self.cpus:
            return 0.0
        return self.elapsed_s * self.cpus / 3600.0


@dataclass
class PartitionCapacity:
    """
    A snapshot of one partition: CPUs by state, and how many jobs are pending on it
    requesting a GPU (``None`` when not measured).
    """

    name: str
    cpus_idle: int = 0
    cpus_allocated: int = 0
    cpus_other: int = 0
    cpus_total: int = 0
    nodes: int = 0
    nodes_unreachable: int = 0
    free_mem_mb: Optional[float] = None
    pending_gpu_jobs: Optional[int] = None


@dataclass
class SubmitRequest:
    """
    One array for one stage: campaign indices ``indices`` (a chunk starting at
    ``offset``), dependencies on upstream arrays as ``{job_id: kind}``.
    """

    campaign: str
    stage: str
    script: str
    offset: int
    indices: List[int]
    resources: "object"
    project_root: str
    manifest: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    after: Dict[str, str] = field(default_factory=dict)
    log_dir: Optional[str] = None

    @property
    def task_ids(self) -> List[int]:
        return [index - self.offset for index in self.indices]

    @property
    def job_name(self) -> str:
        return f"{self.campaign}_{self.stage}"


@runtime_checkable
class Scheduler(Protocol):
    def capacity(
        self, partitions: Optional[Sequence[str]] = None
    ) -> Dict[str, PartitionCapacity]:
        ...

    def query(self, job_ids: Sequence[str]) -> List[TaskRecord]:
        ...

    def submit(self, request: SubmitRequest) -> str:
        ...

    def update_pending(self, job_id: str, **fields) -> None:
        ...

    def cancel(self, job_id: str) -> None:
        ...
