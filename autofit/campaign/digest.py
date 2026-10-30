"""
The campaign digest: what a campaign's tasks did, per stage, in one small document.

Both the live collector (``autofit.campaign.collector``) and the retrospective mode
(``autofit.campaign.retro``) reduce their inputs to ``TaskOutcome``s and call
``build_digest``. ``write_digest`` writes:

- ``digest.json``: the per-stage summary (counts by state and failure class, p50 / p95 /
  max wall time and peak memory against the request, CPU efficiency, CPU-hours split
  into successful and failed or repeated work, disk, ETA, drift flags and proposals);
- ``digest.md``: the same, as at most ``MAX_MD_BYTES`` of markdown, which is what an
  agent reads;
- ``tasks.jsonl``: one row per task, for tools (``resubmit`` reads it);
- ``failed_<class>.txt``: one task per line for every failure class present.

Every proposal is a proposal: nothing here submits, updates or cancels a job.
"""
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from autofit.campaign.lint import gpu_partitions_of, split_partitions
from autofit.campaign.units import format_time, requests_gpu
from autofit.non_linear.status.states import RunState
from autofit.tools.util import open_atomic

SCHEMA = 1
MAX_MD_BYTES = 2048

FINISHED_BAD = ("timeout", "oom", "node_fail", "failed", "cancelled", "preempted")

# Drift thresholds.
TIMEOUT_RATE = 0.05
MEMORY_HIGH = 0.9
MEMORY_LOW = 0.4
CPU_EFFICIENCY_LOW = 0.5
MIN_SAMPLE = 20


@dataclass
class TaskOutcome:
    """
    One task's outcome, from scheduler accounting joined with its status record.
    Memory is in MB, times in seconds; ``cpu_s_all_attempts`` is the allocated
    CPU time of every attempt (what the cluster spent).
    """

    stage: str
    key: str
    state: str
    index: Optional[int] = None
    job_id: Optional[str] = None
    failure_class: Optional[str] = None
    scheduler_state: Optional[str] = None
    elapsed_s: Optional[float] = None
    time_limit_s: Optional[float] = None
    cpus: Optional[int] = None
    req_mem_mb: Optional[float] = None
    max_rss_mb: Optional[float] = None
    total_cpu_s: Optional[float] = None
    partition: Optional[str] = None
    gres: Optional[str] = None
    node: Optional[str] = None
    attempts: int = 1
    cpu_s_all_attempts: float = 0.0
    bytes: Optional[int] = None
    success: Optional[bool] = None
    fraction: Optional[float] = None
    start: Optional[str] = None
    end: Optional[str] = None

    @property
    def cpu_efficiency(self) -> Optional[float]:
        if not self.elapsed_s or not self.cpus or self.total_cpu_s is None:
            return None
        return self.total_cpu_s / (self.elapsed_s * self.cpus)

    @property
    def failure_list(self) -> Optional[str]:
        """
        The ``failed_<class>`` list this task belongs on, if any.
        """
        if self.state == str(RunState.FAILED):
            return self.failure_class or "unrecorded"
        if self.state in FINISHED_BAD:
            return self.state
        if self.state == str(RunState.COMPLETED) and self.success is False:
            return "success_check"
        return None


def percentile(values: List[float], q: float) -> Optional[float]:
    """
    The ``q``-th percentile (0-100) of ``values`` by linear interpolation.
    """
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = (len(values) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _median(values) -> Optional[float]:
    return percentile(list(values), 50)


def _round(value, digits=0):
    if value is None:
        return None
    return round(value, digits) if digits else int(round(value))


def _summary(values: List[float]) -> Dict:
    return {
        "p50": _round(percentile(values, 50)),
        "p95": _round(percentile(values, 95)),
        "max": _round(max(values)) if values else None,
        "n": len(values),
    }


def _gpu_misuse(outcome: TaskOutcome, site) -> bool:
    names = split_partitions(outcome.partition)
    return bool(gpu_partitions_of(names, site=site)) and not requests_gpu(outcome.gres)


def stage_digest(outcomes: List[TaskOutcome], site=None) -> Dict:
    """
    The digest of one stage's outcomes (see the module docstring).
    """
    states = Counter(outcome.state for outcome in outcomes)
    failure_classes = Counter(
        outcome.failure_class or "unrecorded"
        for outcome in outcomes
        if outcome.state == str(RunState.FAILED)
    )
    completed = [o for o in outcomes if o.state == str(RunState.COMPLETED)]
    finished = [o for o in outcomes if o.state not in ("pending", "running")]
    remaining = [o for o in outcomes if o.state in ("pending", "running")]

    walls = [o.elapsed_s for o in completed if o.elapsed_s is not None]
    rss = [o.max_rss_mb for o in completed if o.max_rss_mb]
    efficiency = [o.cpu_efficiency for o in completed if o.cpu_efficiency is not None]
    time_request = _median(o.time_limit_s for o in outcomes if o.time_limit_s)
    mem_request = _median(o.req_mem_mb for o in outcomes if o.req_mem_mb)
    cpus = _median(o.cpus for o in outcomes if o.cpus)

    cpu_h_success = sum(o.elapsed_s * o.cpus for o in completed if o.elapsed_s and o.cpus) / 3600
    cpu_h_all = sum(o.cpu_s_all_attempts for o in outcomes) / 3600

    digest = {
        "tasks": len(outcomes),
        "states": dict(sorted(states.items())),
        "failure_classes": dict(sorted(failure_classes.items())),
        "success_check_failed": sum(1 for o in completed if o.success is False),
        "attempts": sum(o.attempts for o in outcomes),
        "request": {
            "cpus": _round(cpus),
            "mem_mb": _round(mem_request),
            "time_s": _round(time_request),
            "partitions": sorted({o.partition for o in outcomes if o.partition}),
        },
        "wall_s": _summary(walls),
        "rss_mb": _summary(rss),
        "cpu_efficiency_p50": _round(_median(efficiency), 3),
        "cpu_h": {
            "successful": _round(cpu_h_success, 1),
            "all_attempts": _round(cpu_h_all, 1),
            "failed_or_repeated": _round(max(cpu_h_all - cpu_h_success, 0.0), 1),
        },
    }

    misuse = [o for o in outcomes if _gpu_misuse(o, site)]
    if misuse:
        digest["gpu_partition_cpu_tasks"] = {
            "tasks": len(misuse),
            "cpu_h": _round(sum(o.cpu_s_all_attempts for o in misuse) / 3600, 1),
        }

    sizes = [o.bytes for o in completed if o.bytes]
    if sizes:
        mean = sum(sizes) / len(sizes)
        digest["disk"] = {
            "bytes_used": int(sum(o.bytes or 0 for o in outcomes)),
            "bytes_per_completed": int(mean),
            "bytes_forecast_remaining": int(mean * len(remaining)),
        }

    running_cpus = sum(o.cpus or 0 for o in outcomes if o.state == "running")
    p50 = digest["wall_s"]["p50"]
    if remaining and running_cpus and p50 and cpus:
        remaining_cpu_s = 0.0
        for o in remaining:
            done = (o.fraction or 0.0) if o.state == "running" else 0.0
            remaining_cpu_s += p50 * (1 - min(done, 1.0)) * (o.cpus or cpus)
        digest["eta_s"] = _round(remaining_cpu_s / running_cpus)

    digest["flags"] = _flags(digest, outcomes, completed, finished, remaining)
    return digest


def _flags(digest, outcomes, completed, finished, remaining) -> List[Dict]:
    flags = []
    request = digest["request"]
    timeouts = sum(1 for o in finished if o.state == str(RunState.TIMEOUT))
    if finished and timeouts / len(finished) >= TIMEOUT_RATE:
        flags.append(
            {
                "flag": "timeouts",
                "detail": f"{timeouts}/{len(finished)} finished tasks hit the time limit",
                "propose": "raise the time limit of pending tasks (p95 + 25 %) or resubmit timeouts",
            }
        )
    rss_p95 = digest["rss_mb"]["p95"]
    if rss_p95 and request["mem_mb"] and len(completed) >= MIN_SAMPLE:
        ratio = rss_p95 / request["mem_mb"]
        if ratio >= MEMORY_HIGH:
            flags.append(
                {
                    "flag": "memory_tight",
                    "detail": f"p95 peak memory is {ratio:.0%} of the request",
                    "propose": f"mem {_round(rss_p95 * 1.5)} MB per task",
                }
            )
        elif ratio <= MEMORY_LOW:
            flags.append(
                {
                    "flag": "memory_slack",
                    "detail": f"p95 peak memory is {ratio:.0%} of the request (strands cores)",
                    "propose": f"mem {_round(rss_p95 * 1.5)} MB per task",
                }
            )
    efficiency = digest["cpu_efficiency_p50"]
    if efficiency is not None and efficiency < CPU_EFFICIENCY_LOW and len(completed) >= MIN_SAMPLE:
        flags.append(
            {
                "flag": "cpu_efficiency",
                "detail": f"median CPU efficiency {efficiency:.0%}",
                "propose": "fewer CPUs per task",
            }
        )
    if digest.get("gpu_partition_cpu_tasks"):
        flags.append(
            {
                "flag": "cpu_on_gpu_partition",
                "detail": f"{digest['gpu_partition_cpu_tasks']['tasks']} CPU-only tasks ran on a GPU partition",
                "propose": "never again: the partition lint refuses this at submit",
            }
        )
    if digest["success_check_failed"]:
        flags.append(
            {
                "flag": "success_check",
                "detail": f"{digest['success_check_failed']} completed tasks fail the stage's success check",
                "propose": "inspect failed_success_check.txt",
            }
        )
    return flags


def build_digest(
    name: str,
    outcomes: Iterable[TaskOutcome],
    site=None,
    capacity: Optional[Dict] = None,
    extra: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> Dict:
    """
    The digest of a campaign's outcomes, stage by stage, plus totals and a capacity
    snapshot.
    """
    outcomes = list(outcomes)
    by_stage: Dict[str, List[TaskOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_stage[outcome.stage].append(outcome)
    stages = {stage: stage_digest(items, site=site) for stage, items in by_stage.items()}

    totals_states = Counter(o.state for o in outcomes)
    digest = {
        "schema": SCHEMA,
        "campaign": name,
        "updated": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals": {
            "tasks": len(outcomes),
            "states": dict(sorted(totals_states.items())),
            "cpu_h_all_attempts": _round(sum(o.cpu_s_all_attempts for o in outcomes) / 3600, 1),
            "cpu_h_successful": _round(
                sum(s["cpu_h"]["successful"] or 0 for s in stages.values()), 1
            ),
        },
        "stages": stages,
    }
    if capacity:
        digest["capacity"] = {
            name: {
                "cpus_idle": c.cpus_idle,
                "cpus_total": c.cpus_total,
                "pending_gpu_jobs": c.pending_gpu_jobs,
            }
            for name, c in capacity.items()
        }
    if extra:
        digest.update(extra)
    return digest


def _hours(seconds) -> str:
    return "-" if seconds is None else format_time(seconds)


def digest_markdown(digest: Dict, max_bytes: int = MAX_MD_BYTES) -> str:
    """
    ``digest`` as markdown of at most ``max_bytes`` bytes: a header, one table row per
    stage, then flags; flags are dropped from the end, then stages, until it fits.
    """
    totals = digest["totals"]
    header = [
        f"# {digest['campaign']} — {digest['updated']}",
        "",
        f"{totals['tasks']} tasks: "
        + ", ".join(f"{k} {v}" for k, v in totals["states"].items())
        + f". CPU-h {totals['cpu_h_all_attempts']} (successful {totals['cpu_h_successful']}).",
    ]
    if digest.get("period"):
        header.append(f"Period {digest['period']['first_submit']} → {digest['period']['last_end']}.")
    rows = [
        "",
        "| stage | done/all | fail | wall p50/p95 | req time | RSS p95/req MB | eff | CPU-h ok/all |",
        "|---|---|---|---|---|---|---|---|",
    ]
    flags = []
    for name, stage in digest["stages"].items():
        states = stage["states"]
        bad = sum(states.get(k, 0) for k in FINISHED_BAD)
        rows.append(
            f"| {name} | {states.get('completed', 0)}/{stage['tasks']} | {bad} | "
            f"{_hours(stage['wall_s']['p50'])}/{_hours(stage['wall_s']['p95'])} | "
            f"{_hours(stage['request']['time_s'])} | "
            f"{stage['rss_mb']['p95'] or '-'}/{stage['request']['mem_mb'] or '-'} | "
            f"{stage['cpu_efficiency_p50'] if stage['cpu_efficiency_p50'] is not None else '-'} | "
            f"{stage['cpu_h']['successful']}/{stage['cpu_h']['all_attempts']} |"
        )
        for flag in stage["flags"]:
            flags.append(f"- {name}: {flag['flag']} — {flag['detail']}; propose {flag['propose']}")
    if digest.get("capacity"):
        rows.append("")
        rows.append(
            "Idle CPUs: "
            + ", ".join(f"{k} {v['cpus_idle']}/{v['cpus_total']}" for k, v in digest["capacity"].items())
        )
    flag_block = ["", "Flags:"] + flags if flags else []

    def render(rows_, flags_):
        return "\n".join(header + rows_ + flags_) + "\n"

    text = render(rows, flag_block)
    while len(text.encode()) > max_bytes and len(flag_block) > 2:
        flag_block = flag_block[:-1]
        text = render(rows, flag_block + ["- (more flags in digest.json)"])
    if len(text.encode()) > max_bytes:
        flag_block = []
    while len(text.encode()) > max_bytes and len(rows) > 4:
        rows = rows[:-1]
        text = render(rows + ["| … | more stages in digest.json |"], flag_block)
    text = text if len(text.encode()) <= max_bytes else text.encode()[: max_bytes - 4].decode(errors="ignore") + "\n…\n"
    return text


def outcome_row(outcome: TaskOutcome) -> Dict:
    return {key: value for key, value in asdict(outcome).items() if value is not None}


def read_outcomes(path) -> List[TaskOutcome]:
    """
    The ``TaskOutcome``s of a ``tasks.jsonl``.
    """
    outcomes = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            outcomes.append(TaskOutcome(**json.loads(line)))
    return outcomes


def write_digest(directory, digest: Dict, outcomes: Iterable[TaskOutcome]) -> Path:
    """
    Write ``digest.json``, ``digest.md``, ``tasks.jsonl`` and the ``failed_<class>.txt``
    lists into ``directory`` (atomically, each), removing lists no longer current.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outcomes = list(outcomes)

    with open_atomic(directory / "digest.json", "w") as f:
        json.dump(digest, f, indent=1, default=str)
    with open_atomic(directory / "digest.md", "w") as f:
        f.write(digest_markdown(digest))
    with open_atomic(directory / "tasks.jsonl", "w") as f:
        for outcome in outcomes:
            f.write(json.dumps(outcome_row(outcome), default=str) + "\n")

    lists: Dict[str, List[str]] = defaultdict(list)
    for outcome in outcomes:
        name = outcome.failure_list
        if name:
            index = "-" if outcome.index is None else outcome.index
            lists[name].append(f"{outcome.stage}\t{index}\t{outcome.key}")
    for stale in directory.glob("failed_*.txt"):
        if stale.stem[len("failed_"):] not in lists:
            stale.unlink()
    for name, lines in lists.items():
        with open_atomic(directory / f"failed_{name}.txt", "w") as f:
            f.write("\n".join(sorted(lines)) + "\n")
    return directory
