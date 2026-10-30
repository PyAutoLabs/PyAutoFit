"""
The live collector: one pass over a campaign's scheduler accounting and status records,
reduced to the digest.

- **Jobs** come from the campaign's job ledger (``submit``), queried in one call.
- **Status records** (``<output_path>.status.json``, hpc-campaign phase 1) under the
  campaign's ``status_root`` are joined to tasks by the scheduler job and task ids the
  fit recorded for itself.
- **State**: the scheduler decides what a fit cannot say for itself (``timeout``,
  ``oom``, ``node_fail``, ``cancelled``); otherwise the status record's state and
  failure class win over a bare exit code, and a task with neither is classified
  from accounting alone.
- **Success**: a completed task of a stage with a ``success`` check counts as done only
  if its zip exists with ``.completed`` and every listed member inside.
- **Unsubmitted** manifest objects of a submitted stage are reported as ``unsubmitted``.

``collect`` writes the digest files into ``<project_root>/campaign/<name>/`` and returns
the digest; it never submits, updates or cancels anything.
"""
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from autofit.campaign.digest import TaskOutcome, build_digest, write_digest
from autofit.campaign.proposals import propose_updates
from autofit.campaign.scheduler.base import TaskRecord
from autofit.campaign.scheduler.sacct import latest_per_task
from autofit.campaign.submit import ledger_indices, read_ledger
from autofit.non_linear.status.states import Product, RunState

logger = logging.getLogger(__name__)

SCHEDULER_DECIDES = {str(RunState.TIMEOUT), str(RunState.OOM), str(RunState.NODE_FAIL), "cancelled", "preempted"}

StatusIndex = Dict[Tuple[str, Optional[int]], Dict]


def scan_status(root) -> StatusIndex:
    """
    Every status record under ``root``, keyed by the ``(job_id, task_id)`` the fit
    recorded; the newest record wins when a task wrote more than one.
    """
    index: StatusIndex = {}
    root = Path(root)
    if not root.exists():
        return index
    suffix = str(Product.STATUS)
    for directory, _, files in os.walk(root):
        for name in files:
            if not name.endswith(suffix):
                continue
            path = Path(directory) / name
            try:
                document = json.loads(path.read_text())
            except (OSError, ValueError):
                logger.warning(f"Unreadable status record {path}.")
                continue
            job = document.get("job") or {}
            job_id = job.get("job_id")
            if job_id is None:
                continue
            task_id = job.get("task_id")
            key = (str(job_id), None if task_id is None else int(task_id))
            current = index.get(key)
            if current is None or (document.get("updated") or "") >= (current.get("updated") or ""):
                document["_path"] = str(path)
                index[key] = document
    return index


def merge_state(record: TaskRecord, status: Optional[Dict]) -> Tuple[str, Optional[str]]:
    """
    ``(state, failure_class)`` of a task from its accounting record and status record.
    """
    scheduler_state = str(record.state)
    if scheduler_state in SCHEDULER_DECIDES:
        return scheduler_state, None
    if status is not None and status.get("state"):
        state = status["state"]
        if state == str(RunState.SKIPPED):
            return str(RunState.COMPLETED), None
        if state == str(RunState.TIMEOUT_PENDING) and scheduler_state in ("completed", "failed"):
            # Signalled before the limit and then ended by it, or stopped.
            return str(RunState.TIMEOUT), None
        if state in (str(RunState.RUNNING), str(RunState.PENDING)) and scheduler_state in ("completed", "failed"):
            # The process ended without a final status write (killed hard).
            return scheduler_state, None if scheduler_state == "completed" else "unrecorded"
        if state == str(RunState.TIMEOUT_PENDING):
            return str(RunState.RUNNING), None
        return state, status.get("failure_class")
    if scheduler_state == str(RunState.FAILED):
        return scheduler_state, "unrecorded"
    return scheduler_state, None


def check_success(campaign, stage, index: int) -> Optional[bool]:
    """
    Whether task ``index`` of ``stage`` passes the stage's ``success`` check (``None``
    when the stage declares none).
    """
    success = stage.success
    if not success.zip_complete:
        return None
    try:
        path = campaign.path(success.zip_complete.format(**campaign.fields(index)))
    except (KeyError, IndexError):
        return False
    zip_path = path if path.suffix == ".zip" else Path(f"{path}{Product.ZIP}")
    try:
        with zipfile.ZipFile(zip_path) as f:
            names = set(f.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    if str(Product.COMPLETED) not in names:
        return False
    return all(member in names for member in success.members)


def outcome_of(
    stage: str,
    record: TaskRecord,
    status: Optional[Dict],
    index: Optional[int],
    cpu_s_all_attempts: float,
) -> TaskOutcome:
    state, failure_class = merge_state(record, status)
    progress = (status or {}).get("progress") or {}
    outputs = (status or {}).get("outputs") or {}
    resources = (status or {}).get("resources") or {}
    max_rss = record.max_rss_mb or resources.get("peak_rss_mb")
    return TaskOutcome(
        stage=stage,
        key=record.key,
        state=state,
        index=index,
        job_id=record.job_id,
        failure_class=failure_class,
        scheduler_state=record.scheduler_state,
        elapsed_s=record.elapsed_s,
        time_limit_s=record.time_limit_s,
        cpus=record.cpus,
        req_mem_mb=record.req_mem_mb,
        max_rss_mb=max_rss,
        total_cpu_s=record.total_cpu_s,
        partition=record.partition,
        gres=record.gres,
        node=record.node,
        attempts=max(record.attempts, (status or {}).get("attempt") or 1),
        cpu_s_all_attempts=cpu_s_all_attempts,
        bytes=outputs.get("bytes"),
        fraction=progress.get("fraction"),
        start=record.start.isoformat() if record.start else None,
        end=record.end.isoformat() if record.end else None,
    )


def cpu_seconds_by_task(records: Iterable[TaskRecord]) -> Dict[str, float]:
    """
    Allocated CPU-seconds (elapsed × CPUs) of every attempt, summed per task.
    """
    totals: Dict[str, float] = {}
    for record in records:
        if record.elapsed_s and record.cpus:
            totals[record.key] = totals.get(record.key, 0.0) + record.elapsed_s * record.cpus
    return totals


def outcomes_for(campaign, scheduler, status: Optional[StatusIndex] = None) -> List[TaskOutcome]:
    """
    The outcome of every task in the campaign's ledger, one per campaign index and stage
    (the newest array covering it wins), plus ``unsubmitted`` rows.
    """
    ledger = read_ledger(campaign)
    if status is None:
        status = scan_status(campaign.path(campaign.status_root))
    entries = {str(entry["job_id"]): entry for entry in ledger}
    records = scheduler.query(list(entries))
    cpu_s = cpu_seconds_by_task(records)

    outcomes: Dict[Tuple[str, int], TaskOutcome] = {}
    order: Dict[Tuple[str, int], int] = {}
    position = {job_id: i for i, job_id in enumerate(entries)}
    for record in latest_per_task(records):
        entry = entries.get(str(record.job_id))
        if entry is None or record.task_id is None:
            continue
        index = entry["offset"] + record.task_id
        stage_name = entry["stage"]
        key = (stage_name, index)
        rank = position[str(record.job_id)]
        if key in order and order[key] > rank:
            # An older array of a task already seen from its resubmit.
            outcomes[key].attempts += record.attempts
            outcomes[key].cpu_s_all_attempts += cpu_s.get(record.key, 0.0)
            continue
        outcome = outcome_of(
            stage_name,
            record,
            status.get((str(record.job_id), record.task_id)),
            index,
            cpu_s.get(record.key, 0.0),
        )
        previous = outcomes.get(key)
        if previous is not None:
            # A resubmit: the earlier array's attempts and CPU time count too.
            outcome.attempts += previous.attempts
            outcome.cpu_s_all_attempts += previous.cpu_s_all_attempts
        if outcome.state == str(RunState.COMPLETED) and stage_name in campaign.stages:
            outcome.success = check_success(campaign, campaign.stages[stage_name], index)
        outcomes[key] = outcome
        order[key] = rank

    submitted: Dict[str, set] = {}
    for entry in ledger:
        submitted.setdefault(entry["stage"], set()).update(ledger_indices(entry))
    manifest = set(campaign.indices())
    for stage_name in submitted:
        for index in sorted(manifest - submitted[stage_name]):
            outcomes[(stage_name, index)] = TaskOutcome(
                stage=stage_name, key=f"-_{index}", state="unsubmitted", index=index
            )
    return [outcomes[key] for key in sorted(outcomes)]


def collect(campaign, scheduler, write: bool = True) -> Dict:
    """
    Collect the campaign's state into its digest (and write the digest files).
    """
    outcomes = outcomes_for(campaign, scheduler)
    site = campaign.site if campaign.site_name else None
    partitions = sorted(
        {name for stage in campaign.stages.values() for name in (stage.resources.partition or "").split(",") if name}
    )
    try:
        capacity = scheduler.capacity(partitions or None)
    except Exception as exception:
        logger.warning(f"Capacity snapshot failed: {exception}")
        capacity = None
    digest = build_digest(str(campaign.name), outcomes, site=site, capacity=capacity)
    for name, stage_digest in digest["stages"].items():
        if name in campaign.stages:
            updates = propose_updates(
                [o for o in outcomes if o.stage == name], stage_digest, campaign.stages[name], site=site
            )
            if updates:
                stage_digest["update_pending"] = updates
    if write:
        write_digest(campaign.directory, digest, outcomes)
    return digest
