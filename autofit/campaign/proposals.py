"""
Resubmit and pending-update proposals, sized from the tasks' own measurements.

A proposal is only ever written down: ``python -m autofit.campaign resubmit`` writes a
``resubmit_<stage>_<class>.json`` and prints the command that would act on it
(``submit --proposal <file>``), which a human runs once they approve it (ruling F5).

Sizing, per the stage's ``retry`` policy (defaults in brackets):

- ``timeout`` [``time: p95+25%``]: the larger of the completed tasks' p95 wall time and
  the longest timed-out run, plus 25 %, capped at the partition's ``max_time``;
- ``oom`` [``mem: peak*1.5``]: the larger of the OOM tasks' peak memory and their
  request, times 1.5, as ``--mem-per-cpu``, capped at the partition's memory per node;
- ``node_fail`` / ``preempted``: the same request again;
- a Python failure (``exception``, ``bad_input``) is not resubmitted: its cause needs
  fixing first; ``missing_seed`` / ``upstream_incomplete`` follow the upstream stage.

A task already tried ``max`` more times than its first attempt is left out, with the
reason.
"""
import copy
import re
from typing import Dict, List

from autofit.campaign.digest import TaskOutcome, percentile
from autofit.campaign.lint import split_partitions
from autofit.campaign.units import array_spec, format_time, parse_time

RESUBMITTABLE = {"timeout", "oom", "node_fail", "preempted"}
NOT_RESUBMITTED = {
    "exception": "a Python exception: fix the cause, then resubmit",
    "bad_input": "invalid inputs: fix the data or settings first",
    "missing_seed": "the upstream result is missing: resubmit the upstream stage",
    "upstream_incomplete": "the upstream result is incomplete: resubmit the upstream stage",
    "unrecorded": "failed without a status record: read the job log first",
    "cancelled": "cancelled (by a user or an unmet dependency): resubmit the upstream stage",
}

_TIME_RULE = re.compile(r"^\s*p(\d+)\s*\+\s*(\d+)\s*%\s*$")
_MEM_RULE = re.compile(r"^\s*peak\s*\*\s*([0-9.]+)\s*$")


def _site_limits(resources, site):
    max_time = None
    mem_node = None
    if site is None:
        return max_time, mem_node
    for name in split_partitions(resources.partition):
        partition = site.partitions.get(name)
        if partition is None:
            continue
        if partition.max_time:
            limit = parse_time(partition.max_time)
            max_time = limit if max_time is None else min(max_time, limit)
        if partition.mem_per_node_mb:
            mem_node = partition.mem_per_node_mb if mem_node is None else min(mem_node, partition.mem_per_node_mb)
    return max_time, mem_node


def propose_resubmit(
    stage,
    failure: str,
    outcomes: List[TaskOutcome],
    site=None,
) -> Dict:
    """
    The resubmit proposal for ``stage``'s tasks of failure class ``failure``.

    Parameters
    ----------
    stage
        The campaign ``Stage`` (its resources and ``retry`` policy).
    failure
        ``timeout``, ``oom``, ``node_fail``, ``preempted`` (or a class that is never
        resubmitted, for which the proposal says why).
    outcomes
        The stage's outcomes (all of them: completed tasks size a timeout).
    """
    tasks = [o for o in outcomes if o.stage == stage.name and o.failure_list == failure]
    proposal = {
        "stage": stage.name,
        "class": failure,
        "tasks": len(tasks),
        "indices": None,
        "resources": None,
        "changes": {},
        "excluded": [],
        "reason": None,
    }
    if not tasks:
        proposal["reason"] = "no tasks of this class"
        return proposal
    if failure not in RESUBMITTABLE:
        proposal["reason"] = NOT_RESUBMITTED.get(failure, "not resubmitted automatically")
        return proposal

    policy = stage.retry.get(failure)
    if policy in ("none", None) and failure in stage.retry:
        proposal["reason"] = f"the stage's retry policy for {failure} is none"
        return proposal
    policy = policy or {}
    maximum = policy.get("max")

    eligible = []
    for task in tasks:
        if maximum is not None and task.attempts > int(maximum):
            proposal["excluded"].append(
                {"key": task.key, "reason": f"{task.attempts} attempts, retry max {maximum}"}
            )
        elif task.index is None:
            proposal["excluded"].append({"key": task.key, "reason": "no campaign index"})
        else:
            eligible.append(task)

    resources = copy.deepcopy(stage.resources)
    max_time, mem_node = _site_limits(resources, site)

    if failure == "timeout":
        rule = _TIME_RULE.match(str(policy.get("time", "p95+25%")))
        if rule is None:
            raise ValueError(f"Unknown timeout rule {policy.get('time')!r}; use pNN+MM%.")
        q, extra = int(rule.group(1)), int(rule.group(2))
        completed = [o.elapsed_s for o in outcomes if o.stage == stage.name and o.state == "completed" and o.elapsed_s]
        base = max(
            [percentile(completed, q) or 0.0] + [t.elapsed_s or 0.0 for t in tasks]
        )
        new_time = base * (1 + extra / 100.0)
        if max_time is not None and new_time > max_time:
            new_time = max_time
            proposal["changes"]["note"] = f"capped at the partition max_time {format_time(max_time)}"
        old = resources.time_s
        if old is not None and new_time <= old:
            proposal["changes"]["note"] = (
                "cannot raise the time limit further; the tasks need to resume from "
                "their checkpoints, so resubmitting continues them"
            )
            new_time = old
        resources.time_s = int(new_time)
        proposal["changes"]["time"] = [format_time(old), format_time(resources.time_s)]

    elif failure == "oom":
        rule = _MEM_RULE.match(str(policy.get("mem", "peak*1.5")))
        if rule is None:
            raise ValueError(f"Unknown OOM rule {policy.get('mem')!r}; use peak*X.")
        factor = float(rule.group(1))
        peak = max(max(t.max_rss_mb or 0.0, t.req_mem_mb or 0.0) for t in tasks)
        new_mem = peak * factor
        if mem_node is not None and new_mem > mem_node:
            new_mem = mem_node
            proposal["changes"]["note"] = f"capped at the partition's {mem_node} MB per node"
        old = resources.mem_mb
        resources.mem_per_cpu_mb = int(-(-new_mem // resources.cpus))
        proposal["changes"]["mem_mb"] = [old, resources.mem_mb]

    if not eligible:
        proposal["reason"] = "every task has exhausted its retries"
        return proposal

    proposal["indices"] = array_spec([t.index for t in eligible])
    proposal["tasks"] = len(eligible)
    proposal["resources"] = {
        key: value for key, value in vars(resources).items() if value is not None
    }
    return proposal


def propose_updates(outcomes: List[TaskOutcome], digest_stage: Dict, stage, site=None) -> List[Dict]:
    """
    ``scontrol update`` proposals for a stage's arrays that still have pending tasks,
    from its drift flags: a raised time limit after timeouts, a corrected memory
    request after tight or slack memory.
    """
    pending_jobs = sorted({o.job_id for o in outcomes if o.state == "pending" and o.job_id})
    if not pending_jobs:
        return []
    flags = {flag["flag"] for flag in digest_stage.get("flags", [])}
    fields: Dict = {}
    max_time, mem_node = _site_limits(stage.resources, site)
    if "timeouts" in flags and digest_stage["wall_s"]["p95"]:
        base = max(digest_stage["wall_s"]["p95"], max((o.elapsed_s or 0) for o in outcomes if o.state == "timeout"))
        new_time = int(base * 1.25)
        if max_time is not None:
            new_time = min(new_time, max_time)
        fields["time"] = format_time(new_time)
    if flags & {"memory_tight", "memory_slack"} and digest_stage["rss_mb"]["p95"]:
        new_mem = digest_stage["rss_mb"]["p95"] * 1.5
        if mem_node is not None:
            new_mem = min(new_mem, mem_node)
        fields["mem_per_cpu_mb"] = int(-(-new_mem // (stage.resources.cpus or 1)))
    if not fields:
        return []
    mapping = {"time": "TimeLimit", "mem_per_cpu_mb": "MinMemoryCPU"}
    return [
        {
            "job_id": job_id,
            "fields": fields,
            "command": f"scontrol update JobId={job_id} "
            + " ".join(f"{mapping[k]}={v}" for k, v in fields.items()),
        }
        for job_id in pending_jobs
    ]
