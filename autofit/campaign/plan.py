"""
``campaign plan``: what a campaign will need, before it is submitted, from evidence.

For every stage the plan reads two kinds of evidence:

- a **pilot**: the outcomes of tasks of this campaign (or a previous run of it) on this
  site, as a ``tasks.jsonl`` written by ``status`` or ``retro``. Its numbers are graded
  ``estimated`` (measured, but on a sample);
- the **published summaries** (``autofit.campaign.evidence``): the cost of one
  likelihood call, host memory, and whole-stage wall times measured by the profiling
  and inference projects, graded by how well their keys fit the stage.

and proposes, per stage, the request a task should make:

- **CPUs**: kept, unless the pilot's median CPU efficiency is below
  ``CPU_EFFICIENCY_SHRINK``; then the largest power of two not above the CPUs it used;
- **memory**: the pilot's peak RSS × ``MEMORY_MARGIN``, rounded up to ``MEMORY_STEP_MB``
  (memory is a hard limit, so the maximum, not a quantile, sets it);
- **wall time**: the p95 of finished tasks × (1 + ``TIME_MARGIN``), per size bin when
  the stage names a manifest ``size_field``, rounded up to ``TIME_STEP_S`` and capped at
  the partition's ``max_time``. Timed-out tasks are kept at their limit (a lower bound),
  so a p95 at the old limit is reported as censored rather than trusted;
- **throttle**: at most ``THROTTLE_FRACTION`` of the idle CPUs, in tasks.

plus ETA (remaining CPU-hours ÷ the CPUs the throttle allows, times the pilot's
rerun factor), a disk forecast, and an energy placeholder that phase 4 (carbon)
fills. ``plan.json`` lists every record used with its grade; ``plan.md`` is the same
for a human to approve. Nothing is submitted.

Every number is a ``Graded`` value: ``{"value": ..., "grade": ..., "sources": [...]}``.
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from autofit.campaign.digest import TaskOutcome, percentile, read_outcomes
from autofit.campaign.evidence import (
    GRADES,
    EvidenceSet,
    designation_for,
    describe_limitation,
    matches,
    per_call_s,
    worst,
)
from autofit.campaign.lint import split_partitions
from autofit.campaign.units import format_time, parse_time
from autofit.tools.util import open_atomic

SCHEMA = 1

CPU_EFFICIENCY_SHRINK = 0.6
MEMORY_MARGIN = 1.25
MEMORY_STEP_MB = 4096
TIME_MARGIN = 0.25
TIME_STEP_S = 900
THROTTLE_FRACTION = 0.5
SIZE_BINS = 3


def graded(value, grade: str, sources: Sequence[str] = (), note: Optional[str] = None) -> Dict:
    """
    A number with its evidence grade and the records or files it came from.
    """
    entry = {"value": value, "grade": grade, "sources": list(sources)}
    if note:
        entry["note"] = note
    return entry


def unsupported(note: str) -> Dict:
    return graded(None, "unsupported", [], note)


def _round_up(value: float, step: float) -> int:
    return int(math.ceil(value / step) * step)


def read_pilot(path) -> List[TaskOutcome]:
    """
    The outcomes of a pilot: a ``tasks.jsonl``, or a directory holding one (the output
    of ``status`` or ``retro``).
    """
    path = Path(path)
    return read_outcomes(path / "tasks.jsonl" if path.is_dir() else path)


def pilot_stats(outcomes: List[TaskOutcome], stage: str, source: str) -> Optional[Dict]:
    """
    What a stage's pilot tasks measured: wall time over finished tasks (timeouts kept
    at their limit, counted as censored), peak memory and CPU efficiency over completed
    tasks, and the rerun factor (CPU-hours of every attempt over successful ones).
    """
    tasks = [o for o in outcomes if o.stage == stage]
    completed = [o for o in tasks if o.state == "completed"]
    if not completed:
        return None
    timeouts = [o for o in tasks if o.state == "timeout"]
    finished = completed + timeouts
    walls = [o.elapsed_s for o in finished if o.elapsed_s]
    rss = [o.max_rss_mb for o in completed if o.max_rss_mb]
    efficiency = [o.cpu_efficiency for o in completed if o.cpu_efficiency is not None]
    cpus = percentile([o.cpus for o in tasks if o.cpus], 50)
    successful = sum(o.elapsed_s * o.cpus for o in completed if o.elapsed_s and o.cpus)
    every = sum(o.cpu_s_all_attempts for o in tasks)
    sizes = [o.bytes for o in completed if o.bytes]
    limit = min((o.time_limit_s for o in timeouts if o.time_limit_s), default=None)
    wall_p95 = percentile(walls, 95)
    return {
        "source": f"{source}#{stage}",
        "tasks": len(tasks),
        "completed": len(completed),
        "timeouts": len(timeouts),
        "cpus": cpus,
        "wall_s": {
            "p50": percentile(walls, 50),
            "p95": wall_p95,
            "max": max(walls) if walls else None,
        },
        "wall_censored": bool(limit and wall_p95 and wall_p95 >= limit),
        "rss_mb": {
            "p50": percentile(rss, 50),
            "p95": percentile(rss, 95),
            "max": max(rss) if rss else None,
        },
        "cpu_efficiency_p50": percentile(efficiency, 50),
        "rerun_factor": (every / successful) if successful and every else None,
        "bytes_per_result": (sum(sizes) / len(sizes)) if sizes else None,
        "by_index": {
            o.index: o.elapsed_s for o in finished if o.index is not None and o.elapsed_s
        },
        "limit_by_index": {
            o.index: o.time_limit_s
            for o in timeouts
            if o.index is not None and o.time_limit_s
        },
    }


def _best_group(found):
    """
    The matches sharing the best grade of ``found`` (best first), in a deterministic
    order: by record id.
    """
    if not found:
        return []
    return sorted((m for m in found if m.grade == found[0].grade), key=lambda m: m.record.id)


def _spread(values) -> Optional[Dict]:
    if len(values) < 2:
        return None
    return {"n": len(values), "min": min(values), "max": max(values)}


def evidence_stats(
    keys: Dict[str, str],
    evidence: Sequence[EvidenceSet],
    designations: Sequence[Dict] = (),
    evals: Optional[float] = None,
) -> Dict:
    """
    What the published summaries say about a stage with match keys ``keys``: the best
    graded per-call cost, host memory, whole-stage wall time, and their scaling.
    """
    stats = {"keys": keys}
    used: List[Dict] = []

    def note(match):
        used.append(
            {
                "id": match.record.id,
                "schema": match.record.schema,
                "grade": match.grade,
                "differs": {k: list(v) for k, v in match.differs.items()},
                "limitations": [describe_limitation(t) for t in match.record.limitations],
            }
        )

    runtime = [m for m in matches(evidence, keys, "runtime", designations) if per_call_s(m.record)]
    if runtime:
        best = runtime[0]
        note(best)
        stats["per_call_s"] = graded(per_call_s(best.record), best.grade, [best.record.id])
    else:
        stats["per_call_s"] = unsupported("no profiling runtime record matches the stage keys")

    # Memory and stage wall time: several records may match (seeds, legs). The plan
    # takes the largest of those sharing the best grade, deterministically, and keeps
    # every contributing id and the spread.
    memory = _best_group(
        [m for m in matches(evidence, keys, "memory_host", designations) if m.record.values]
    )
    fallback = _best_group([m for m in runtime if m.record.extra.get("peak_rss_gb")])
    if memory:
        for match in memory:
            note(match)

        def mb(record):
            value = next(iter(record.values.values()))
            return value * 1.048576 if (record.unit or "").lower() == "mib" else value

        values = [mb(m.record) for m in memory]
        stats["host_peak_rss_mb"] = graded(
            round(max(values)), memory[0].grade, [m.record.id for m in memory],
            _spread_note(values, "MB"),
        )
    elif fallback:
        # A runtime record's own resources block (sacct MaxRSS of the profiling leg).
        values = [m.record.extra["peak_rss_gb"] * 1024 for m in fallback]
        stats["host_peak_rss_mb"] = graded(
            round(max(values)), fallback[0].grade, [m.record.id for m in fallback],
            "; ".join(
                filter(None, ["peak RSS recorded with the runtime measurement", _spread_note(values, "MB")])
            ),
        )
    else:
        stats["host_peak_rss_mb"] = unsupported("no profiling memory record matches")

    walltime = _best_group(
        [m for m in matches(evidence, keys, "stage_walltime", designations) if m.record.values.get("wall_s")]
    )
    if walltime:
        for match in walltime:
            note(match)
        chosen = max(walltime, key=lambda m: (m.record.values["wall_s"], m.record.id))
        ids = [m.record.id for m in walltime]
        walls = [m.record.values["wall_s"] for m in walltime]
        values = chosen.record.values
        stats["stage_wall_s"] = graded(values["wall_s"], chosen.grade, ids, _spread_note(walls, "s"))
        if values.get("cores"):
            stats["stage_wall_s"]["cores"] = int(values["cores"])
        stats["likelihood_evals"] = graded(
            max((m.record.values.get("likelihood_evals") or 0 for m in walltime)) or None,
            chosen.grade, ids,
        )
        implied = [m.record.values["implied_s_per_eval"] for m in walltime if m.record.values.get("implied_s_per_eval")]
        stats["implied_s_per_eval"] = graded(max(implied) if implied else None, chosen.grade, ids)
        if values.get("cores"):
            stats["implied_s_per_eval"]["cores"] = int(values["cores"])
    else:
        stats["stage_wall_s"] = unsupported("no walltime-validation record matches the stage")

    # Scaling: evaluations × cost per evaluation is an estimate whatever its inputs.
    n_evals = (
        graded(evals, "estimated", ["campaign.yaml evidence.evals"])
        if evals
        else stats.get("likelihood_evals")
    )
    implied = stats.get("implied_s_per_eval")
    cost = implied if implied and implied.get("value") is not None else stats["per_call_s"]
    if n_evals and n_evals.get("value") and cost.get("value"):
        stats["scaled_wall_s"] = graded(
            n_evals["value"] * cost["value"],
            worst(n_evals["grade"], cost["grade"], "estimated"),
            n_evals["sources"] + cost["sources"],
            "likelihood evaluations × cost per evaluation",
        )
        if cost.get("cores"):
            stats["scaled_wall_s"]["cores"] = cost["cores"]
    else:
        stats["scaled_wall_s"] = unsupported("needs both a number of evaluations and a cost per evaluation")

    designation = designation_for(keys, designations)
    stats["designation"] = (
        {"decision": designation["id"], "optimal": designation.get("optimal")}
        if designation
        else None
    )
    stats["records"] = used
    return stats


def _spread_note(values, unit: str) -> Optional[str]:
    spread = _spread(values)
    if spread is None:
        return None
    return (
        f"largest of {spread['n']} matching records "
        f"({spread['min']:.0f}–{spread['max']:.0f} {unit})"
    )


def _size_bins(campaign, stage, pilot, n_bins: int = SIZE_BINS) -> Optional[List[Dict]]:
    field_name = stage.evidence.get("size_field")
    if not field_name or not pilot or not pilot["by_index"]:
        return None
    sizes = {}
    for index in campaign.indices():
        try:
            sizes[index] = float(campaign.fields(index)[field_name])
        except (KeyError, ValueError):
            continue
    if not sizes:
        return None
    ordered = sorted(sizes.values())
    edges = [ordered[min(len(ordered) - 1, int(len(ordered) * i / n_bins))] for i in range(1, n_bins)]
    bins = defaultdict(list)
    population = defaultdict(int)

    def bin_of(size):
        return sum(1 for edge in edges if size > edge)

    for index, size in sizes.items():
        population[bin_of(size)] += 1
    limits = defaultdict(list)
    for index, wall in pilot["by_index"].items():
        if index in sizes:
            bins[bin_of(sizes[index])].append(wall)
    for index, limit in pilot.get("limit_by_index", {}).items():
        if index in sizes:
            limits[bin_of(sizes[index])].append(limit)
    lows = [ordered[0]] + edges
    highs = edges + [ordered[-1]]
    result = []
    for number in range(n_bins):
        walls = bins.get(number, [])
        p95 = percentile(walls, 95)
        result.append(
            {
                "bin": number,
                "size_range": [lows[number], highs[number]],
                "objects": population.get(number, 0),
                "pilot_tasks": len(walls),
                "wall_p95_s": p95,
                # As in ``pilot_stats``: timeouts are kept at their limit, so a p95 at
                # or above the bin's smallest limit is a lower bound.
                "censored": bool(p95 and limits.get(number) and p95 >= min(limits[number])),
            }
        )
    return result


def _max_time(stage, site) -> Optional[int]:
    if site is None:
        return None
    limits = []
    for name in split_partitions(stage.resources.partition):
        partition = site.partitions.get(name)
        if partition is not None and partition.max_time:
            limits.append(parse_time(partition.max_time))
    return min(limits) if limits else None


def _proposed_time(wall_p95, max_time) -> int:
    seconds = _round_up(wall_p95 * (1 + TIME_MARGIN), TIME_STEP_S)
    return min(seconds, max_time) if max_time else seconds


def propose(stage, pilot: Optional[Dict], evidence: Dict, site=None, idle_cpus: Optional[Dict] = None) -> Dict:
    """
    The proposed request for ``stage`` (see the module docstring for the rules), each
    value graded by what it rests on.
    """
    request = stage.resources
    proposal = {}
    pilot_source = [pilot["source"]] if pilot else []

    # CPUs.
    efficiency = pilot["cpu_efficiency_p50"] if pilot else None
    if pilot and efficiency is not None and efficiency < CPU_EFFICIENCY_SHRINK:
        used = (pilot["cpus"] or request.cpus) * efficiency
        cpus = 2 ** int(math.floor(math.log2(max(used, 1.0))))
        proposal["cpus"] = graded(
            cpus, "estimated", pilot_source,
            f"median CPU efficiency {efficiency:.2f} < {CPU_EFFICIENCY_SHRINK}: {used:.1f} CPUs used; "
            f"wall times were measured at {pilot['cpus'] or request.cpus:g} CPUs, so re-pilot at {cpus}",
        )
    elif pilot:
        proposal["cpus"] = graded(int(pilot["cpus"] or request.cpus), "estimated", pilot_source)
    else:
        proposal["cpus"] = graded(request.cpus, "unsupported", [], "the stage's own request; no pilot")
    cpus = proposal["cpus"]["value"]

    # Memory.
    peak = pilot["rss_mb"]["max"] if pilot else None
    if peak:
        mem = _round_up(peak * MEMORY_MARGIN, MEMORY_STEP_MB)
        proposal["mem_mb"] = graded(mem, "estimated", pilot_source, f"peak {peak:.0f} MB × {MEMORY_MARGIN}")
    elif evidence["host_peak_rss_mb"]["value"]:
        host = evidence["host_peak_rss_mb"]
        mem = _round_up(host["value"] * MEMORY_MARGIN, MEMORY_STEP_MB)
        proposal["mem_mb"] = graded(mem, worst(host["grade"], "estimated"), host["sources"], "host peak × margin")
    else:
        proposal["mem_mb"] = graded(request.mem_mb, "unsupported", [], "the stage's own request")
    if proposal["mem_mb"]["value"]:
        proposal["mem_per_cpu_mb"] = dict(
            proposal["mem_mb"], value=int(-(-proposal["mem_mb"]["value"] // cpus))
        )

    # Wall time.
    max_time = _max_time(stage, site)
    if pilot and pilot["wall_s"]["p95"]:
        seconds = _proposed_time(pilot["wall_s"]["p95"], max_time)
        note = f"p95 {format_time(pilot['wall_s']['p95'])} + {TIME_MARGIN:.0%}"
        if pilot["wall_censored"]:
            note += "; censored: the p95 is at the old time limit, so the real p95 is higher — re-pilot at the new limit"
        proposal["time_s"] = graded(seconds, "estimated", pilot_source, note)
    else:
        basis = evidence_wall(evidence, cpus)
        if basis["value"]:
            seconds = _proposed_time(basis["value"], max_time)
            proposal["time_s"] = graded(
                seconds, worst(basis["grade"], "estimated"), basis["sources"],
                "; ".join(
                    filter(None, ["measured stage wall + margin: no spread is known, so re-pilot", basis.get("note")])
                ),
            )
            if basis.get("limitations"):
                proposal["time_s"]["limitations"] = basis["limitations"]
        else:
            proposal["time_s"] = graded(request.time_s, "unsupported", [], "the stage's own request")
    if max_time and proposal["time_s"]["value"] and proposal["time_s"]["value"] >= max_time:
        proposal["time_s"]["note"] = (proposal["time_s"].get("note", "") + f"; capped at the partition max_time").strip("; ")
    proposal["time"] = dict(proposal["time_s"], value=format_time(proposal["time_s"]["value"]))

    # Throttle.
    idle = idle_cpus or {}
    partition = split_partitions(request.partition)[0] if request.partition else None
    idle_entry = idle.get(partition) or idle.get("*")
    if idle_entry and idle_entry["value"]:
        throttle = max(1, int(idle_entry["value"] * THROTTLE_FRACTION // cpus))
        proposal["throttle"] = graded(
            throttle, worst(idle_entry["grade"], "estimated"), idle_entry["sources"],
            f"{THROTTLE_FRACTION:.0%} of {idle_entry['value']} idle CPUs",
        )
    else:
        proposal["throttle"] = graded(request.throttle, "unsupported", [], "no capacity snapshot")
    return proposal


def evidence_wall(evidence: Dict, cpus: int) -> Dict:
    """
    The evidence's stage wall time (a measured stage, else evaluations × cost) for a
    task of ``cpus`` CPUs. A wall time measured on another number of cores is scaled by
    cores / cpus — perfect scaling, which is optimistic when ``cpus`` is the smaller —
    graded ``estimated`` and marked with a ``cores_mismatch`` limitation.
    """
    basis = evidence["stage_wall_s"] if evidence["stage_wall_s"]["value"] else evidence["scaled_wall_s"]
    cores = basis.get("cores")
    if not basis["value"] or not cores or not cpus or cores == cpus:
        return basis
    limitation = (
        f"cores_mismatch: measured at {cores} cores, requested {cpus}; "
        f"wall scaled by {cores}/{cpus} assuming perfect scaling"
        + (" (optimistic)" if cpus < cores else "")
    )
    return dict(
        basis,
        value=basis["value"] * cores / cpus,
        grade=worst(basis["grade"], "estimated"),
        note=limitation,
        limitations=[limitation],
        measured_wall_s=basis["value"],
    )


def _eta(n_tasks, pilot, evidence, proposal) -> Dict:
    cpus = proposal["cpus"]["value"]
    rerun = (pilot or {}).get("rerun_factor") or 1.0
    core_s = None
    if pilot and pilot["wall_s"]["p50"]:
        wall = graded(pilot["wall_s"]["p50"], "estimated", [pilot["source"]])
    else:
        wall = evidence_wall(evidence, cpus)
        if wall.get("cores") and wall["value"]:
            # CPU-hours from what was measured (wall × the cores it ran on), not from
            # a scaled wall time.
            core_s = wall.get("measured_wall_s", wall["value"]) * wall["cores"]
    if not wall["value"] or not cpus:
        return {"cpu_h": unsupported("no wall time"), "eta_s": unsupported("no wall time")}
    sources = wall["sources"]
    if core_s is not None:
        cpu_h = n_tasks * core_s * rerun / 3600
        how = f"{n_tasks} tasks × measured wall × {wall['cores']} measured cores × rerun factor {rerun:.2f}"
    else:
        cpu_h = n_tasks * wall["value"] * cpus * rerun / 3600
        how = f"{n_tasks} tasks × p50 wall × {cpus} CPUs × rerun factor {rerun:.2f}"
    result = {
        "cpu_h": graded(round(cpu_h, 1), worst(wall["grade"], "estimated"), sources, how),
    }
    if wall.get("limitations"):
        result["cpu_h"]["limitations"] = wall["limitations"]
    throttle = proposal["throttle"]
    if throttle["value"]:
        concurrent = throttle["value"] * cpus
        result["eta_s"] = graded(
            round(cpu_h * 3600 / concurrent),
            worst(result["cpu_h"]["grade"], throttle["grade"]),
            sources + throttle["sources"],
            f"at {concurrent} CPUs at once",
        )
    else:
        result["eta_s"] = unsupported("no throttle: no capacity snapshot")
    return result


def _disk(n_tasks, pilot, results_index: Optional[Dict], stage) -> Dict:
    if pilot and pilot.get("bytes_per_result"):
        per = graded(pilot["bytes_per_result"], "estimated", [pilot["source"]])
    else:
        names = set(stage.evidence.get("result_names") or [stage.name])
        sizes = (results_index or {}).get("sizes", {})
        found = [size for name in names for size in sizes.get(name, [])]
        if not found:
            return {"bytes": unsupported("no pilot sizes and no results index for the stage")}
        per = graded(sum(found) / len(found), "estimated", [f"{results_index['source']} ({len(found)} results)"])
    return {
        "bytes_per_result": dict(per, value=int(per["value"])),
        "bytes": dict(per, value=int(per["value"] * n_tasks)),
    }


def read_results_index(path) -> Dict:
    """
    Result sizes per search name from a results index (``python -m autofit.results
    index``): its per-result JSON records under ``records/``, read as plain JSON.
    """
    path = Path(path)
    sizes = defaultdict(list)
    for record_path in sorted((path / "records").glob("*.json")):
        try:
            record = json.loads(record_path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("state") == "completed" and record.get("bytes"):
            sizes[record.get("name")].append(record["bytes"])
    return {"source": str(path), "sizes": dict(sizes)}


def build_plan(
    campaign,
    evidence: Sequence[EvidenceSet] = (),
    pilot: Optional[List[TaskOutcome]] = None,
    pilot_source: Optional[str] = None,
    designations: Sequence[Dict] = (),
    idle_cpus: Optional[Dict] = None,
    results_index: Optional[Dict] = None,
    decisions_source: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict:
    """
    The plan of ``campaign`` (see the module docstring).

    Parameters
    ----------
    evidence
        Summaries read by ``autofit.campaign.evidence.read_evidence``.
    pilot
        Pilot outcomes (``read_pilot``) and the file they came from.
    designations
        Pulse designation decisions (``read_designations``).
    idle_cpus
        Idle CPUs per partition (or ``"*"``) as graded values, from a live capacity
        snapshot or stated by hand.
    results_index
        Result sizes per search name (``read_results_index``), for the disk forecast.
    """
    site = campaign.site if campaign.site_name else None
    n_tasks = len(campaign.indices())
    stages = {}
    records: Dict[str, Dict] = {}
    for name in campaign.order():
        stage = campaign.stages[name]
        keys = dict((stage.evidence or {}).get("keys") or {})
        stage_pilot = pilot_stats(pilot, name, pilot_source or "pilot") if pilot else None
        ev = evidence_stats(keys, evidence, designations, stage.evidence.get("evals"))
        proposal = propose(stage, stage_pilot, ev, site=site, idle_cpus=idle_cpus)
        eta = _eta(n_tasks, stage_pilot, ev, proposal)
        bins = _size_bins(campaign, stage, stage_pilot)
        flags = []
        if bins:
            max_time = _max_time(stage, site)
            wrapped = []
            for entry in bins:
                if entry["wall_p95_s"]:
                    time = graded(
                        format_time(_proposed_time(entry["wall_p95_s"], max_time)),
                        "estimated", [stage_pilot["source"]],
                        "censored: a lower bound — re-pilot at a longer limit" if entry["censored"] else None,
                    )
                else:
                    time = unsupported("no pilot task in this bin")
                if entry["censored"]:
                    flags.append(f"censored wall time in size bin {entry['bin']}")
                wrapped.append(
                    {
                        "measured": graded(
                            entry, "estimated", [stage_pilot["source"]],
                            "pilot tasks per size bin; sizes and object counts from the manifest",
                        ),
                        "time": time,
                    }
                )
            bins = wrapped
        if stage_pilot and stage_pilot["wall_censored"]:
            flags.append("censored wall time: the pilot p95 is at the old time limit")
        for entry in (proposal["time_s"], eta["cpu_h"]):
            for limitation in entry.get("limitations") or []:
                if limitation not in flags:
                    flags.append(limitation)
        cpu_h = eta["cpu_h"]
        mem = proposal["mem_mb"]["value"]
        energy = {
            "status": "placeholder: hpc-campaign phase 4 (carbon) fills energy from these",
            "cpu_h": cpu_h,
            "mem_gb_h": graded(
                round(cpu_h["value"] / proposal["cpus"]["value"] * mem / 1024, 1)
                if cpu_h["value"] and mem else None,
                worst(cpu_h["grade"], proposal["mem_mb"]["grade"]),
                cpu_h["sources"] + proposal["mem_mb"]["sources"],
            ),
        }
        for record in ev["records"]:
            records[record["id"]] = record
        stages[name] = {
            "keys": keys,
            "request": {
                "cpus": stage.resources.cpus,
                "mem_mb": stage.resources.mem_mb,
                "time": format_time(stage.resources.time_s),
                "partition": stage.resources.partition,
                "throttle": stage.resources.throttle,
            },
            "pilot": graded(
                {
                    k: v
                    for k, v in stage_pilot.items()
                    if k not in ("by_index", "limit_by_index", "source")
                },
                "estimated", [stage_pilot["source"]], "pilot measurements",
            )
            if stage_pilot
            else None,
            "flags": flags,
            "evidence": {k: v for k, v in ev.items() if k != "records"},
            "proposal": proposal,
            "size_bins": bins,
            "eta": eta,
            "disk": _disk(n_tasks, stage_pilot, results_index, stage),
            "energy": energy,
        }

    graded_values = list(_graded_values(stages))
    counts = {grade: sum(1 for g in graded_values if g == grade) for grade in GRADES}
    etas = [s["eta"]["eta_s"] for s in stages.values()]
    total_eta = (
        graded(
            sum(e["value"] for e in etas),
            worst(*(e["grade"] for e in etas)),
            sorted({src for e in etas for src in e["sources"]}),
            "stages one after another: an upper bound, since aftercorr overlaps them",
        )
        if etas and all(e["value"] for e in etas)
        else unsupported("a stage has no ETA")
    )
    return {
        "schema": SCHEMA,
        "campaign": campaign.name,
        "created": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "objects": n_tasks,
        "inputs": {
            "evidence": [
                {"source": e.source, "schema": f"{e.schema}@{e.version}", "generated_at": e.generated_at,
                 "records": len(e.records)}
                for e in evidence
            ],
            "pilot": pilot_source,
            "decisions": decisions_source,
            "designations": len(designations),
        },
        "stages": stages,
        "totals": {
            "eta_s": total_eta,
            "cpu_h": graded(
                round(sum(s["eta"]["cpu_h"]["value"] or 0 for s in stages.values()), 1),
                worst(*(s["eta"]["cpu_h"]["grade"] for s in stages.values())) if stages else "unsupported",
                sorted({src for s in stages.values() for src in s["eta"]["cpu_h"]["sources"]}),
            ),
        },
        "grade_counts": counts,
        "records": sorted(records.values(), key=lambda r: r["id"]),
    }


def _graded_values(node):
    if isinstance(node, dict):
        if "grade" in node and "value" in node:
            yield node["grade"]
            return
        for value in node.values():
            yield from _graded_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _graded_values(value)


def _cell(entry: Optional[Dict], render=lambda v: v) -> str:
    if not entry:
        return "-"
    if entry.get("value") is None:
        return f"— ({entry['grade']})"
    return f"{render(entry['value'])} ({entry['grade']})"


def _hours(seconds) -> str:
    return f"{seconds / 3600:.1f} h"


def plan_markdown(plan: Dict) -> str:
    """
    ``plan`` for a human to read and approve.
    """
    lines = [
        f"# Plan: {plan['campaign']} — {plan['created']}",
        "",
        f"{plan['objects']} objects. Every number carries its evidence grade "
        f"({', '.join(GRADES)}); nothing here is submitted.",
        "",
        "| stage | CPUs | memory | time | throttle | CPU-h | ETA |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, stage in plan["stages"].items():
        proposal = stage["proposal"]
        lines.append(
            f"| {name} | {_cell(proposal['cpus'])} | "
            f"{_cell(proposal['mem_mb'], lambda v: f'{v / 1024:.0f} GB')} | "
            f"{_cell(proposal['time'])} | {_cell(proposal['throttle'])} | "
            f"{_cell(stage['eta']['cpu_h'], lambda v: f'{v:,.0f}')} | "
            f"{_cell(stage['eta']['eta_s'], _hours)} |"
        )
    totals = plan["totals"]
    lines += [
        "",
        f"Total: {_cell(totals['cpu_h'], lambda v: f'{v:,.0f} CPU-h')}; "
        f"ETA {_cell(totals['eta_s'], _hours)} ({totals['eta_s'].get('note', '')}).",
        "",
        "Grades used: " + ", ".join(f"{g} {n}" for g, n in plan["grade_counts"].items() if n) + ".",
    ]
    for name, stage in plan["stages"].items():
        lines += ["", f"## {name}", ""]
        request = stage["request"]
        lines.append(
            f"Requested now: {request['cpus']} CPUs, "
            f"{(request['mem_mb'] or 0) / 1024:.0f} GB, {request['time']}, "
            f"partition {request['partition']}."
        )
        for key, entry in stage["proposal"].items():
            if key in ("time_s", "mem_per_cpu_mb"):
                continue
            note = f" — {entry['note']}" if entry.get("note") else ""
            lines.append(f"- {key}: {_cell(entry)}{note}")
        ev = stage["evidence"]
        lines.append(
            f"- evidence: per-call {_cell(ev['per_call_s'], lambda v: f'{v:.3g} s')}, "
            f"host memory {_cell(ev['host_peak_rss_mb'], lambda v: f'{v} MB')}, "
            f"stage wall {_cell(ev['stage_wall_s'], _hours)}, "
            f"scaled wall {_cell(ev['scaled_wall_s'], _hours)}"
        )
        for flag in stage.get("flags") or []:
            lines.append(f"- **flag** {flag}")
        if ev["designation"]:
            lines.append(
                f"- optimal setup per Pulse decision {ev['designation']['decision']}: "
                f"{ev['designation']['optimal']}"
            )
        else:
            lines.append("- designation: none recorded for these keys")
        if stage["size_bins"]:
            lines.append(
                "- by size: "
                + "; ".join(
                    f"{m['size_range'][0]:g}–{m['size_range'][1]:g} ({m['objects']} objects): {_cell(b['time'])}"
                    for b in stage["size_bins"]
                    for m in [b["measured"]["value"]]
                )
            )
        disk = stage["disk"]
        lines.append(f"- disk: {_cell(disk['bytes'], lambda v: f'{v / 1e9:.1f} GB')}")
    if plan["records"]:
        lines += ["", "## Records used", ""]
        for record in plan["records"]:
            differs = ", ".join(
                f"{k} not recorded (wanted {v[0]})" if v[1] is None else f"{k} {v[1]} not {v[0]}"
                for k, v in record["differs"].items()
            )
            lines.append(
                f"- `{record['id']}` ({record['schema']}, {record['grade']})"
                + (f" — differs: {differs}" if differs else "")
            )
    return "\n".join(lines) + "\n"


def write_plan(directory, plan: Dict) -> Path:
    """
    Write ``plan.json`` and ``plan.md`` into ``directory`` (atomically, each).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open_atomic(directory / "plan.json", "w") as f:
        json.dump(plan, f, indent=1, default=str)
    with open_atomic(directory / "plan.md", "w") as f:
        f.write(plan_markdown(plan))
    return directory
