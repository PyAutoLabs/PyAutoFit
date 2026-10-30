"""
``LocalScheduler``: run a campaign's tasks as subprocesses on this machine (a laptop or
a single workstation), with the same environment a Slurm array task sees.

Each task runs the stage's command in ``project_root`` with ``CAMPAIGN``, ``STAGE``,
``CAMPAIGN_OFFSET``, ``CAMPAIGN_INDEX``, ``SLURM_ARRAY_TASK_ID`` (the task id inside
its array, so Slurm-style scripts work unchanged) and ``MANIFEST`` set. ``aftercorr``
waits for the same task of the upstream array and cancels the task if it did not
complete; ``afterok`` waits for the whole upstream array.
"""
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from autofit.campaign.scheduler.base import (
    PartitionCapacity,
    SubmitRequest,
    TaskRecord,
)


class LocalScheduler:
    def __init__(self, max_workers: Optional[int] = None):
        """
        Parameters
        ----------
        max_workers
            Tasks run at once (default: the number of CPUs).
        """
        self.max_workers = max_workers or os.cpu_count() or 1
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._lock = threading.Lock()
        self._next_id = 1
        self.records: Dict[str, List[TaskRecord]] = {}
        self._futures: Dict[str, Dict[int, Future]] = {}
        self._processes: Dict[str, Dict[int, subprocess.Popen]] = {}

    def capacity(self, partitions: Optional[Sequence[str]] = None) -> Dict[str, PartitionCapacity]:
        running = sum(
            1
            for records in self.records.values()
            for record in records
            if record.scheduler_state == "RUNNING"
        )
        total = os.cpu_count() or 1
        return {
            "local": PartitionCapacity(
                name="local",
                cpus_allocated=running,
                cpus_idle=max(total - running, 0),
                cpus_total=total,
                nodes=1,
                pending_gpu_jobs=0,
            )
        }

    def _command(self, request: SubmitRequest) -> List[str]:
        command = shlex.split(request.script)
        if command and command[0].endswith(".py"):
            command = [sys.executable] + command
        return command

    def _wait_upstream(self, request: SubmitRequest, task_id: int) -> bool:
        for upstream_id, kind in request.after.items():
            futures = self._futures.get(str(upstream_id), {})
            if kind == "aftercorr":
                future = futures.get(task_id)
                if future is None or future.result() != "COMPLETED":
                    return False
            elif kind == "afterok":
                if any(future.result() != "COMPLETED" for future in futures.values()):
                    return False
            else:
                for future in futures.values():
                    future.result()
        return True

    def _run_task(self, job_id: str, request: SubmitRequest, record: TaskRecord) -> str:
        if not self._wait_upstream(request, record.task_id):
            record.scheduler_state = "CANCELLED"
            return record.scheduler_state
        if record.scheduler_state == "CANCELLED":
            return record.scheduler_state

        env = dict(os.environ)
        env.update(request.env)
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "CAMPAIGN": request.campaign,
                "STAGE": request.stage,
                "CAMPAIGN_OFFSET": str(request.offset),
                "CAMPAIGN_INDEX": str(request.offset + record.task_id),
                "SLURM_ARRAY_TASK_ID": str(record.task_id),
            }
        )
        if request.manifest:
            env["MANIFEST"] = request.manifest

        log = None
        if request.log_dir:
            log_dir = Path(request.project_root) / request.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            log = open(log_dir / f"{request.job_name}_{job_id}_{record.task_id}.out", "w")

        record.start = datetime.now()
        record.scheduler_state = "RUNNING"
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                self._command(request),
                cwd=request.project_root,
                env=env,
                stdout=log or subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log else subprocess.DEVNULL,
            )
            self._processes.setdefault(job_id, {})[record.task_id] = process
            timeout = request.resources.time_s
            try:
                code = process.wait(timeout=timeout)
                record.scheduler_state = "COMPLETED" if code == 0 else "FAILED"
                record.exit_code = f"{code}:0"
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
                record.scheduler_state = "TIMEOUT"
        finally:
            if log:
                log.close()
        record.end = datetime.now()
        record.elapsed_s = int(time.monotonic() - started)
        return record.scheduler_state

    def submit(self, request: SubmitRequest) -> str:
        with self._lock:
            job_id = str(self._next_id)
            self._next_id += 1
        records = [
            TaskRecord(
                job_id=job_id,
                task_id=task_id,
                name=request.job_name,
                scheduler_state="PENDING",
                partition="local",
                cpus=request.resources.cpus,
                req_mem_mb=request.resources.mem_mb,
                time_limit_s=request.resources.time_s,
                workdir=str(request.project_root),
            )
            for task_id in request.task_ids
        ]
        self.records[job_id] = records
        self._futures[job_id] = {
            record.task_id: self._pool.submit(self._run_task, job_id, request, record)
            for record in records
        }
        return job_id

    def wait(self, job_ids: Optional[Sequence[str]] = None):
        """
        Block until every task of ``job_ids`` (default: all) has finished.
        """
        for job_id in job_ids or list(self._futures):
            for future in self._futures.get(str(job_id), {}).values():
                future.result()

    def query(self, job_ids: Sequence[str]) -> List[TaskRecord]:
        return [record for job_id in job_ids for record in self.records.get(str(job_id), [])]

    def update_pending(self, job_id: str, **fields) -> None:
        for record in self.records.get(str(job_id), []):
            if record.scheduler_state == "PENDING" and "time" in fields:
                record.time_limit_s = int(fields["time"])

    def cancel(self, job_id: str) -> None:
        for record in self.records.get(str(job_id), []):
            if record.scheduler_state == "PENDING":
                record.scheduler_state = "CANCELLED"
        for process in self._processes.get(str(job_id), {}).values():
            if process.poll() is None:
                process.terminate()
