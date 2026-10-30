"""
Parsing ``sacct --parsable2 --units=M`` output into ``TaskRecord``s.

``sacct`` prints one *allocation* line per job or array task and one line per *step*
(``.batch``, ``.extern``, ``.0``...). Memory high-water marks (``MaxRSS``) are only on
the step lines, so each step is folded into the allocation it belongs to (the last
allocation line with the same ``JobID`` above it, which is how ``sacct`` orders its
output, including under ``--duplicates``).

With ``--duplicates`` a requeued task appears once per attempt; ``parse`` keeps every
attempt (they all used CPU time) and ``latest_per_task`` reduces them to one record per
task with ``attempts`` counted.
"""
import csv
import io
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

from autofit.campaign.scheduler.base import TaskRecord
from autofit.campaign.units import memory_per, parse_memory_mb, parse_time

FIELDS = (
    "JobID",
    "JobIDRaw",
    "JobName%60",
    "Partition",
    "State",
    "ExitCode",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    "Timelimit",
    "AllocCPUS",
    "ReqMem",
    "MaxRSS",
    "AveRSS",
    "TotalCPU",
    "CPUTime",
    "NNodes",
    "NodeList",
    "ReqTRES%80",
    "AllocTRES%80",
    "WorkDir%200",
)


def field_names(fields: Sequence[str] = FIELDS) -> List[str]:
    return [name.split("%", 1)[0] for name in fields]


def _datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or value in ("Unknown", "None", "N/A"):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _memory(value: Optional[str]) -> Optional[float]:
    try:
        return parse_memory_mb(value) if value else None
    except ValueError:
        return None


def _seconds(value: Optional[str]) -> Optional[float]:
    try:
        return parse_time(value) if value else None
    except ValueError:
        return None


def _total_cpu(value: Optional[str]) -> Optional[float]:
    # TotalCPU carries a fractional part ("12-01:02:31" or "00:01.234").
    if not value:
        return None
    whole, _, fraction = value.partition(".")
    try:
        seconds = float(parse_time(whole) if ":" in whole else int(whole))
        return seconds + (float(f"0.{fraction}") if fraction.isdigit() else 0.0)
    except ValueError:
        return None


def _split_job_id(job_id: str):
    """
    ``(base, step)`` of a ``JobID``: ``"342027_1.batch"`` → ``("342027_1", "batch")``.
    """
    if "." in job_id and not job_id.endswith("]"):
        base, step = job_id.split(".", 1)
        return base, step
    return job_id, None


def _array_parts(base: str):
    """
    ``(job_id, task_id)``: ``"342027_1"`` → ``("342027", 1)``; a plain job or a pending
    array placeholder (``"344645_[586,588-594]"``) has no task id.
    """
    if "_" in base:
        job, task = base.split("_", 1)
        if task.isdigit():
            return job, int(task)
        return job, None
    return base, None


def _rows(text: str, fields: Optional[Sequence[str]] = None) -> Iterable[Dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if fields is None:
        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="|", quoting=csv.QUOTE_NONE)
        return list(reader)
    names = field_names(fields)
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter="|", quoting=csv.QUOTE_NONE)
    return [dict(zip(names, row)) for row in reader]


def parse(text: str, fields: Optional[Sequence[str]] = None) -> List[TaskRecord]:
    """
    Every allocation (job or array-task attempt) in ``sacct --parsable2`` output, with
    its steps folded in.

    Parameters
    ----------
    text
        The output; with a header line when ``fields`` is ``None`` (``sacct`` without
        ``--noheader``), else exactly the columns of ``fields``.
    """
    records: List[TaskRecord] = []
    latest: Dict[str, TaskRecord] = {}
    for row in _rows(text, fields):
        job_id = row.get("JobID", "")
        if not job_id:
            continue
        base, step = _split_job_id(job_id)
        if step is not None:
            record = latest.get(base)
            if record is None:
                continue
            rss = _memory(row.get("MaxRSS"))
            if rss is not None:
                record.max_rss_mb = max(record.max_rss_mb or 0.0, rss)
            if record.total_cpu_s is None:
                record.total_cpu_s = _total_cpu(row.get("TotalCPU"))
            if not record.node and row.get("NodeList"):
                record.node = row["NodeList"]
            continue

        job, task = _array_parts(base)
        cpus = _int(row.get("AllocCPUS"))
        req_mem = row.get("ReqMem")
        req_mem_mb = _memory(req_mem)
        if req_mem_mb is not None and memory_per(req_mem) == "cpu" and cpus:
            req_mem_mb *= cpus
        record = TaskRecord(
            job_id=job,
            task_id=task,
            name=row.get("JobName") or None,
            scheduler_state=row.get("State") or None,
            exit_code=row.get("ExitCode") or None,
            partition=row.get("Partition") or None,
            node=row.get("NodeList") or None,
            submit=_datetime(row.get("Submit")),
            start=_datetime(row.get("Start")),
            end=_datetime(row.get("End")),
            elapsed_s=_seconds(row.get("Elapsed")),
            time_limit_s=_seconds(row.get("Timelimit")),
            cpus=cpus,
            req_mem_mb=req_mem_mb,
            max_rss_mb=_memory(row.get("MaxRSS")),
            total_cpu_s=_total_cpu(row.get("TotalCPU")),
            workdir=row.get("WorkDir") or None,
            req_tres=row.get("ReqTRES") or None,
            alloc_tres=row.get("AllocTRES") or None,
            raw_job_id=row.get("JobIDRaw") or None,
        )
        if base.endswith("]"):
            # A pending array's placeholder line: one record for the whole range.
            record.job_id = base
        records.append(record)
        latest[base] = record
    return records


def latest_per_task(records: Iterable[TaskRecord]) -> List[TaskRecord]:
    """
    One record per task, the latest attempt (by start, then submit, then position),
    with ``attempts`` counting every attempt seen.
    """
    by_key: Dict[str, List[TaskRecord]] = {}
    for record in records:
        by_key.setdefault(record.key, []).append(record)
    result = []
    for attempts in by_key.values():
        def order(item):
            position, record = item
            stamp = record.start or record.submit or datetime.min
            return stamp, position

        _, last = max(enumerate(attempts), key=order)
        last.attempts = len(attempts)
        result.append(last)
    return result
