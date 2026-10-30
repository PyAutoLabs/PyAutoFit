"""
Retrospective mode: the digest of a finished campaign from a ``sacct`` dump alone.

A campaign that ran before ``campaign.yaml`` existed (the Euclid DR1 campaign) has no
job ledger and few status records, but the scheduler kept accounting for every attempt.
``retro`` reads one ``sacct --parsable2 --units=M --duplicates`` dump, selects the
campaign's jobs by working directory and / or job name, names each job's stage from its
job name, optionally joins status records, and builds the same digest as the live
collector, plus the campaign's period and peak concurrency.

What it can and cannot say: each row is a *scheduler task*, not an object. A task that
timed out in one array and was resubmitted in another is two tasks here, because
nothing in accounting links them; the digest therefore reports task outcomes and
CPU-hours, never "objects done". Pending-array placeholder lines are counted as
``pending`` once per line.
"""
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from autofit.campaign.collector import cpu_seconds_by_task, outcome_of, scan_status
from autofit.campaign.digest import TaskOutcome, build_digest
from autofit.campaign.scheduler import sacct
from autofit.campaign.scheduler.base import TaskRecord


def under(path: Optional[str], prefixes: Sequence[str]) -> bool:
    """
    Whether ``path`` is one of ``prefixes`` or inside one (a path boundary, so
    ``/a/euclid_dr1`` does not match ``/a/euclid_dr1_old``).
    """
    if not prefixes:
        return True
    if not path:
        return False
    path = path.rstrip("/")
    for prefix in prefixes:
        prefix = prefix.rstrip("/")
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def stage_of(name: Optional[str], stage_regex: Optional[str], other: Optional[str] = None) -> str:
    """
    The stage named by a job name: the regex's ``stage`` group (or first group) when
    it matches, else ``other`` when given, else the job name itself.
    """
    if not name:
        return other or "unnamed"
    if stage_regex:
        match = re.search(stage_regex, name)
        if match:
            groups = match.groupdict()
            if groups.get("stage"):
                return groups["stage"]
            if match.groups():
                return match.group(1)
        if other:
            return other
    return name


def select(
    records: Iterable[TaskRecord],
    workdir_prefixes: Sequence[str] = (),
    name_regex: Optional[str] = None,
) -> List[TaskRecord]:
    pattern = re.compile(name_regex) if name_regex else None
    return [
        record
        for record in records
        if under(record.workdir, workdir_prefixes)
        and (pattern is None or pattern.search(record.name or ""))
    ]


def peak_concurrency(records: Iterable[TaskRecord]) -> Tuple[int, Optional[str]]:
    """
    The most CPUs the campaign held at once, and when (from start / end times).
    """
    events = []
    for record in records:
        if record.start and record.end and record.cpus and record.end > record.start:
            events.append((record.start, record.cpus))
            events.append((record.end, -record.cpus))
    events.sort(key=lambda event: (event[0], event[1]))
    current = peak = 0
    when = None
    for stamp, delta in events:
        current += delta
        if current > peak:
            peak, when = current, stamp
    return peak, when.isoformat() if when else None


def retro_outcomes(
    records: List[TaskRecord],
    stage_regex: Optional[str] = None,
    status_root: Optional[str] = None,
    other_stage: Optional[str] = None,
) -> List[TaskOutcome]:
    status = scan_status(status_root) if status_root else {}
    cpu_s = cpu_seconds_by_task(records)
    outcomes = []
    for record in sacct.latest_per_task(records):
        outcomes.append(
            outcome_of(
                stage_of(record.name, stage_regex, other_stage),
                record,
                status.get((str(record.job_id), record.task_id)),
                None,
                cpu_s.get(record.key, 0.0),
            )
        )
    return outcomes


def retro(
    sacct_path,
    workdir_prefixes: Sequence[str] = (),
    name_regex: Optional[str] = None,
    stage_regex: Optional[str] = None,
    status_root: Optional[str] = None,
    site=None,
    name: str = "retro",
    other_stage: Optional[str] = None,
) -> Tuple[Dict, List[TaskOutcome]]:
    """
    The digest and outcomes of the campaign selected from the ``sacct`` dump at
    ``sacct_path`` (read only).

    Parameters
    ----------
    workdir_prefixes
        Keep jobs whose ``WorkDir`` is one of these directories or inside one.
    name_regex
        Keep jobs whose name matches.
    stage_regex
        Name each job's stage from its job name (``stage`` group or first group).
    other_stage
        The stage of every job ``stage_regex`` does not match (default: its own name).
    status_root
        Join status records found under this directory.
    site
        The ``autonerves.site_config.Site`` (to recognise GPU-only partitions).
    """
    records = sacct.parse(Path(sacct_path).read_text())
    selected = select(records, workdir_prefixes, name_regex)
    outcomes = retro_outcomes(selected, stage_regex, status_root, other_stage)

    submits = [r.submit for r in selected if r.submit]
    ends = [r.end for r in selected if r.end]
    peak, peak_at = peak_concurrency(selected)
    extra = {
        "mode": "retro",
        "source": {
            "sacct": Path(sacct_path).name,
            "rows": len(records),
            "allocations_selected": len(selected),
            "workdir_prefixes": list(workdir_prefixes),
            "name_regex": name_regex,
            "stage_regex": stage_regex,
            "other_stage": other_stage,
            "jobs": len({r.job_id for r in selected}),
        },
        "period": {
            "first_submit": min(submits).isoformat() if submits else None,
            "last_end": max(ends).isoformat() if ends else None,
        },
        "peak_cpus": {"cpus": peak, "at": peak_at},
        "partitions": dict(Counter(r.partition for r in selected if r.partition)),
    }
    return build_digest(name, outcomes, site=site, extra=extra), outcomes
