"""
``FakeScheduler``: an in-memory scheduler for tests, with scripted outcomes.

Submitted tasks wait as ``PENDING`` until ``run()``, which settles them in submit order:
a task whose dependency can never be met is cancelled (as ``--kill-on-invalid-dep=yes``
does), every other task takes the outcome its script gives. It lints like the Slurm
adapter, using its own ``capacity`` snapshot.
"""
import copy
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Union

from autofit.campaign.lint import check_partitions, split_partitions
from autofit.campaign.scheduler.base import (
    PartitionCapacity,
    SubmitRequest,
    TaskRecord,
)

Outcome = Dict[str, object]
OutcomeScript = Union[Callable[[str, int, int], Optional[Outcome]], Dict, None]

_EPOCH = datetime(2026, 1, 1)


class FakeScheduler:
    def __init__(
        self,
        outcomes: OutcomeScript = None,
        capacity: Optional[Dict[str, PartitionCapacity]] = None,
        site=None,
    ):
        """
        Parameters
        ----------
        outcomes
            ``outcomes(stage, index, attempt)`` or a dict keyed ``(stage, index)``
            giving the task's ``TaskRecord`` fields, e.g. ``{"scheduler_state":
            "TIMEOUT", "elapsed_s": 3600, "max_rss_mb": 900}``. Missing fields default
            to a completed task using a tenth of its time and half its memory.
        capacity
            The ``capacity()`` snapshot, including ``pending_gpu_jobs``.
        """
        self.outcomes = outcomes
        self._capacity = capacity or {}
        self.site = site
        self.requests: Dict[str, SubmitRequest] = {}
        self.records: Dict[str, List[TaskRecord]] = {}
        self.updates: List[tuple] = []
        self.cancelled: List[str] = []
        self.attempts: Dict[tuple, int] = {}
        self._next_id = 1000

    def capacity(self, partitions: Optional[Sequence[str]] = None) -> Dict[str, PartitionCapacity]:
        if partitions is None:
            return dict(self._capacity)
        return {name: self._capacity[name] for name in partitions if name in self._capacity}

    def _pending_gpu(self, names) -> Optional[int]:
        known = [self._capacity[name].pending_gpu_jobs for name in names if name in self._capacity]
        # Unmeasured counts as unknown, which refuses the lint's exemption.
        if not known or any(value is None for value in known):
            return None
        return sum(known)

    def submit(self, request: SubmitRequest) -> str:
        resources = request.resources
        names = split_partitions(resources.partition)
        check_partitions(
            names,
            cpus=resources.cpus,
            gres=resources.gres,
            throttle=resources.throttle,
            n_tasks=len(request.indices),
            pending_gpu_jobs=self._pending_gpu(names) if names else 0,
            site=self.site,
        ).raise_for_refusal()

        job_id = str(self._next_id)
        self._next_id += 1
        self.requests[job_id] = request
        self.records[job_id] = [
            TaskRecord(
                job_id=job_id,
                task_id=task_id,
                name=request.job_name,
                scheduler_state="PENDING",
                partition=resources.partition,
                cpus=resources.cpus,
                req_mem_mb=resources.mem_mb,
                time_limit_s=resources.time_s,
                workdir=str(request.project_root),
                req_tres=f"cpu={resources.cpus}"
                + (f",gres/{resources.gres.replace(':', '=')}" if resources.gres else ""),
            )
            for task_id in request.task_ids
        ]
        return job_id

    def _outcome(self, stage: str, index: int, attempt: int) -> Outcome:
        if callable(self.outcomes):
            outcome = self.outcomes(stage, index, attempt)
        elif isinstance(self.outcomes, dict):
            outcome = self.outcomes.get((stage, index))
        else:
            outcome = None
        return dict(outcome or {})

    def _dependency_met(self, request: SubmitRequest, task_id: int) -> Optional[bool]:
        for upstream_id, kind in request.after.items():
            upstream = self.records.get(str(upstream_id), [])
            if kind == "aftercorr":
                matching = [r for r in upstream if r.task_id == task_id]
                if not matching or matching[0].scheduler_state != "COMPLETED":
                    return False
            elif kind == "afterok":
                if any(r.scheduler_state != "COMPLETED" for r in upstream):
                    return False
        return True

    def run(self):
        """
        Settle every pending task, in submit order.
        """
        for job_id, records in self.records.items():
            request = self.requests[job_id]
            for record in records:
                if record.scheduler_state != "PENDING":
                    continue
                index = request.offset + record.task_id
                key = (request.stage, index)
                self.attempts[key] = self.attempts.get(key, 0) + 1
                if not self._dependency_met(request, record.task_id):
                    record.scheduler_state = "CANCELLED"
                    record.elapsed_s = 0
                    continue
                limit = record.time_limit_s or 3600
                outcome = self._outcome(request.stage, index, self.attempts[key])
                record.scheduler_state = str(outcome.pop("scheduler_state", "COMPLETED"))
                record.elapsed_s = int(outcome.pop("elapsed_s", limit // 10))
                record.max_rss_mb = outcome.pop(
                    "max_rss_mb", (record.req_mem_mb or 1000) / 2
                )
                record.total_cpu_s = outcome.pop(
                    "total_cpu_s", record.elapsed_s * (record.cpus or 1) * 0.9
                )
                record.start = _EPOCH
                record.end = _EPOCH + timedelta(seconds=record.elapsed_s)
                record.exit_code = "0:0" if record.scheduler_state == "COMPLETED" else "1:0"
                for name, value in outcome.items():
                    setattr(record, name, value)

    def query(self, job_ids: Sequence[str]) -> List[TaskRecord]:
        return [copy.copy(record) for job_id in job_ids for record in self.records.get(str(job_id), [])]

    def update_pending(self, job_id: str, **fields) -> None:
        self.updates.append((str(job_id), fields))
        for record in self.records.get(str(job_id), []):
            if record.scheduler_state != "PENDING":
                continue
            if "time" in fields:
                record.time_limit_s = int(fields["time"])
            if "mem_per_cpu_mb" in fields and record.cpus:
                record.req_mem_mb = float(fields["mem_per_cpu_mb"]) * record.cpus

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(str(job_id))
        for record in self.records.get(str(job_id), []):
            if record.scheduler_state == "PENDING":
                record.scheduler_state = "CANCELLED"
