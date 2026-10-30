"""
Energy and carbon of a campaign, estimated with the Green Algorithms model
(Lannelongue, Grealey & Inouye 2021, Adv. Sci.; arXiv 2007.07610):

    E [kWh] = t × (n_c × P_c × u_c + n_m × P_m + n_g × P_g) × PUE / 1000
    C [gCO2e] = E × CI

with ``t`` the wall time in hours, ``n_c`` the CPUs, ``P_c`` the power per CPU (W),
``u_c`` the CPU usage (0-1), ``n_m`` the memory in GB, ``P_m`` = 0.3725 W/GB, ``n_g``
the GPUs, ``P_g`` their power (W, at full use) and ``CI`` the grid carbon intensity.

This is an **estimate**: CPU power is not measured on a shared node (RAPL is root-only
and package-wide; Slurm energy accounting is off at most sites), so the per-CPU power,
the PUE and the carbon intensity come from the site file (``autonerves.site_config``)
as low / central / high values. Every estimate is given as four cases:

- ``low``: low power, PUE and carbon intensity; measured CPU usage; peak memory;
- ``central``: central inputs; measured CPU usage; requested memory (the usage basis);
- ``allocation``: central inputs; every reserved CPU charged at full use (``u = 1``)
  and requested memory (the allocation basis, which is what reserving costs);
- ``high``: high inputs on the allocation basis — the upper bound.

The carbon-intensity range is the spread of the site's bases (e.g. regional, national,
DESNZ; ruling F6): the central value is the site's ``ci_basis``, the bounds the lowest
and highest of the bases' central values. Half-hourly spreads within one basis are not
used, since a campaign averages over many half-hours.

Nothing falls back to Green Algorithms' world-average defaults (PUE 1.67, 475 g/kWh,
10-12 W per core), which overstate a modern site by an order of magnitude: an input the
site does not give is reported missing and the figure that needs it is left out.

The network is never used unless asked: ``neso_intensity`` fetches the GB grid's
recorded carbon intensity for a campaign's period from the NESO Carbon Intensity API
(cached on disk), and ``begin_hint`` suggests a start time for a short, deferrable array
from a 48 h forecast.
"""
import json
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from autofit.campaign.digest import TaskOutcome
from autofit.non_linear.status.states import RunState

SCHEMA = 1

P_MEMORY_W_PER_GB = 0.3725
"""Green Algorithms' power draw of memory, W per GB."""

CASES = ("low", "central", "allocation", "high")

NESO_API = "https://api.carbonintensity.org.uk"
NESO_MAX_DAYS = 13
"""The API serves at most 14 days per request."""


@dataclass
class Bounds:
    """
    A low / central / high triple of one input.
    """

    low: Optional[float]
    central: Optional[float]
    high: Optional[float]

    @classmethod
    def of(cls, value) -> Optional["Bounds"]:
        """
        From an ``autonerves.site_config.SiteValue``, a number, a 2- or 3-sequence or
        ``None``.
        """
        if value is None:
            return None
        if isinstance(value, Bounds):
            return value
        if isinstance(value, (int, float)):
            return cls(float(value), float(value), float(value))
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                return cls(value[0], (value[0] + value[1]) / 2, value[1])
            return cls(*value)
        central = value.central
        if central is None:
            return None
        low, high = value.range
        return cls(
            low if low is not None else central,
            central,
            high if high is not None else central,
        )

    def case(self, case: str) -> Optional[float]:
        """
        The value a case uses: ``low`` → low, ``high`` → high, else central.
        """
        return {"low": self.low, "high": self.high}.get(case, self.central)

    def as_list(self) -> List[Optional[float]]:
        return [self.low, self.central, self.high]


@dataclass
class CarbonInputs:
    """
    The site constants of an estimate, each a ``Bounds``.
    """

    tdp_w_per_cpu: Dict[str, Bounds] = field(default_factory=dict)
    """Power per Slurm CPU (W), by partition."""
    gpu_w: Dict[str, Bounds] = field(default_factory=dict)
    """Power per GPU at full use (W), by partition."""
    pue: Optional[Bounds] = None
    ci_g_per_kwh: Optional[Bounds] = None
    ci_basis: Optional[str] = None
    ci_bases: Dict[str, float] = field(default_factory=dict)
    """The central carbon intensity of every basis the site gives."""
    ci_source: Optional[str] = None
    ci_missing_reason: Optional[str] = None
    """Why there is no carbon intensity, when the site has a table but no usable value."""
    unverified: List[str] = field(default_factory=list)
    site: Optional[str] = None

    @classmethod
    def from_site(
        cls,
        site=None,
        ci_basis: Optional[str] = None,
        pue=None,
        tdp_w_per_cpu=None,
        ci=None,
    ) -> "CarbonInputs":
        """
        The inputs from an ``autonerves`` ``Site`` (may be ``None``), with optional
        overrides: ``pue``, ``tdp_w_per_cpu`` (all partitions) and ``ci`` (gCO2e/kWh),
        each a number, a ``(low, high)`` or a ``(low, central, high)``.
        """
        inputs = cls(site=getattr(site, "name", None))
        if site is not None:
            for name, partition in site.partitions.items():
                tdp = Bounds.of(partition.values.get("tdp_w_per_cpu"))
                if tdp is not None:
                    inputs.tdp_w_per_cpu[name] = tdp
                gpu = Bounds.of(partition.values.get("gpu_tdp_w"))
                if gpu is not None:
                    inputs.gpu_w[name] = gpu
            inputs.pue = Bounds.of(site.pue)
            table = getattr(site, "carbon_intensity", {}) or {}
            basis = ci_basis or site.ci_basis
            bases = {
                name: float(value.central)
                for name, value in table.items()
                if value.central is not None
            }
            if basis in bases:
                inputs.ci_basis = basis
                inputs.ci_bases = bases
                centrals = list(bases.values())
                inputs.ci_g_per_kwh = Bounds(min(centrals), bases[basis], max(centrals))
                inputs.ci_source = (
                    f"site {site.name}: {basis} basis, bounds the spread of bases "
                    f"{', '.join(f'{k} {v:g}' for k, v in bases.items())}"
                )
            elif table:
                # The chosen basis has no value: no carbon figure rather than one from
                # a basis nobody chose.
                inputs.ci_missing_reason = (
                    f"site {site.name} gives no value for the {basis!r} "
                    f"carbon_intensity basis"
                )
            inputs.unverified = list(site.unverified())
        if pue is not None:
            inputs.pue = Bounds.of(pue)
        if tdp_w_per_cpu is not None:
            bounds = Bounds.of(tdp_w_per_cpu)
            inputs.tdp_w_per_cpu = {name: bounds for name in inputs.tdp_w_per_cpu}
            inputs.tdp_w_per_cpu["*"] = bounds
        if ci is not None:
            inputs.ci_g_per_kwh = Bounds.of(ci)
            inputs.ci_basis = "given"
            inputs.ci_bases = {}
            inputs.ci_source = "given explicitly"
            inputs.ci_missing_reason = None
        return inputs

    def tdp_for(self, partition: Optional[str]) -> Tuple[Optional[Bounds], bool]:
        """
        The per-CPU power of ``partition`` and whether it is a fallback: a partition
        the site does not list (or a task on several) takes the widest range of the
        partitions it does list.
        """
        if partition in self.tdp_w_per_cpu:
            return self.tdp_w_per_cpu[partition], False
        if "*" in self.tdp_w_per_cpu:
            return self.tdp_w_per_cpu["*"], False
        if not self.tdp_w_per_cpu:
            return None, True
        values = list(self.tdp_w_per_cpu.values())
        return (
            Bounds(
                min(v.low for v in values),
                sum(v.central for v in values) / len(values),
                max(v.high for v in values),
            ),
            True,
        )

    def missing(self) -> List[str]:
        missing = []
        if not self.tdp_w_per_cpu:
            missing.append("tdp_w_per_cpu")
        if self.pue is None:
            missing.append("pue")
        if self.ci_g_per_kwh is None:
            missing.append("carbon_intensity")
        return missing

    def to_dict(self) -> Dict:
        return {
            "site": self.site,
            "tdp_w_per_cpu": {k: v.as_list() for k, v in self.tdp_w_per_cpu.items()},
            "gpu_w": {k: v.as_list() for k, v in self.gpu_w.items()},
            "pue": None if self.pue is None else self.pue.as_list(),
            "ci_g_per_kwh": None
            if self.ci_g_per_kwh is None
            else self.ci_g_per_kwh.as_list(),
            "ci_basis": self.ci_basis,
            "ci_bases": self.ci_bases,
            "ci_source": self.ci_source,
            "ci_missing_reason": self.ci_missing_reason,
            "unverified": self.unverified,
            "missing": self.missing(),
        }


def power_w(
    cpus: float,
    tdp_w_per_cpu: float,
    usage: float,
    memory_gb: float,
    gpus: float = 0,
    gpu_w: float = 0.0,
) -> float:
    """
    The Green Algorithms power draw (W) of one task, before PUE.
    """
    return cpus * tdp_w_per_cpu * usage + memory_gb * P_MEMORY_W_PER_GB + gpus * gpu_w


def gpu_count(gres: Optional[str]) -> int:
    """
    The number of GPUs a ``--gres`` / TRES request asks for (``gres/gpu=2``,
    ``gpu:a100:2``, ``gpu:1``; a bare ``gpu`` is one).
    """
    if not gres:
        return 0
    total = 0
    for item in str(gres).replace(" ", ",").split(","):
        item = item.strip().lower()
        if not item:
            continue
        name = item.split("=", 1)[0].split(":", 1)[0]
        if name not in ("gpu", "gpus", "gres/gpu") and not name.startswith("gres/gpu"):
            continue
        if "=" in item:
            count = item.split("=", 1)[1]
        else:
            parts = item.split(":")
            count = parts[-1] if len(parts) > 1 else "1"
        try:
            total += int(count)
        except ValueError:
            total += 1
    return total


def _empty() -> Dict[str, float]:
    return {case: 0.0 for case in CASES}


IN_PROGRESS_STATES = ("pending", "running", str(RunState.TIMEOUT_PENDING))
"""Tasks not finished yet: their energy so far is neither successful nor wasted."""

BUCKETS = ("successful", "in_progress", "failed_or_repeated")


@dataclass
class _Tally:
    kwh: Dict[str, float] = field(default_factory=_empty)
    buckets: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {bucket: _empty() for bucket in BUCKETS}
    )
    tasks: int = 0
    completed: int = 0
    in_progress: int = 0

    def add(self, energy: Dict[str, float], bucket: str):
        for case in CASES:
            self.kwh[case] += energy[case]
            self.buckets[bucket][case] += energy[case]


def _usage(outcome: TaskOutcome) -> Optional[float]:
    efficiency = outcome.cpu_efficiency
    if efficiency is None:
        return None
    return min(max(efficiency, 0.0), 1.0)


def _attempt_energy(
    hours: float,
    cpus: float,
    usage: float,
    memory_requested_gb: float,
    memory_peak_gb: float,
    gpus: int,
    tdp: Bounds,
    gpu: Optional[Bounds],
    pue: Bounds,
) -> Dict[str, float]:
    energy = {}
    for case in CASES:
        u = 1.0 if case in ("allocation", "high") else usage
        memory = memory_peak_gb if case == "low" else memory_requested_gb
        gpu_w = gpu.case(case) if (gpu is not None and gpus) else 0.0
        watts = power_w(cpus, tdp.case(case), u, memory, gpus, gpu_w)
        energy[case] = hours * watts * pue.case(case) / 1000.0
    return energy


def _carbon(kwh: Dict[str, float], ci: Optional[Bounds]) -> Optional[Dict[str, float]]:
    if ci is None:
        return None
    return {case: kwh[case] * ci.case(case) / 1000.0 for case in CASES}


def _round_cases(values: Optional[Dict[str, float]], digits=6):
    if values is None:
        return None
    return {case: round(value, digits) for case, value in values.items()}


def _summary(tally: _Tally, ci: Optional[Bounds]) -> Dict:
    successful = tally.buckets["successful"]
    failed = tally.buckets["failed_or_repeated"]
    # The share of *finished* work that was failed or repeated: energy of tasks still
    # in progress is neither, so it is left out of both sides.
    finished = successful["central"] + failed["central"]
    summary = {
        "tasks": tally.tasks,
        "completed": tally.completed,
        "in_progress": tally.in_progress,
        "kwh": _round_cases(tally.kwh),
        "kg_co2e": _round_cases(_carbon(tally.kwh, ci)),
        "successful_kwh": _round_cases(successful),
        "in_progress_kwh": _round_cases(tally.buckets["in_progress"]),
        "failed_or_repeated_kwh": _round_cases(failed),
        "failed_or_repeated_share": round(failed["central"] / finished, 3)
        if finished > 0
        else None,
    }
    if tally.completed:
        per_task = {case: tally.kwh[case] / tally.completed for case in CASES}
        summary["per_completed_task"] = {
            "kwh": _round_cases(per_task, 6),
            "g_co2e": _round_cases(
                None
                if ci is None
                else {c: per_task[c] * ci.case(c) for c in CASES},
                1,
            ),
        }
    return summary


def estimate(
    outcomes: Iterable[TaskOutcome],
    inputs: CarbonInputs,
    objects: Optional[int] = None,
) -> Dict:
    """
    The energy and carbon of a campaign's tasks (live or retrospective), per stage and
    in total, in the four cases (module docstring), split into successful work (the
    final attempt of each completed task), work in progress (the latest attempt of a
    pending, running or ``timeout_pending`` task, so far) and failed or repeated work
    (every earlier attempt and every finished task that did not complete).
    ``failed_or_repeated_share`` is of finished work only.

    Earlier attempts of a task are known only by their allocated CPU time; they are
    charged with the final attempt's CPUs, memory and usage. A task with no CPU time
    recorded is charged at full use (``u = 1``) in every case and counted in
    ``assumptions``. GPUs are charged at full use, as Green Algorithms does.

    Parameters
    ----------
    objects
        The number of objects (e.g. lenses) the campaign set out to analyse; when
        given, the totals are also divided per object.
    """
    missing = [m for m in inputs.missing() if m != "carbon_intensity"]
    result = {
        "schema": SCHEMA,
        "model": "Green Algorithms (Lannelongue et al. 2021)",
        "cases": {
            "low": "low inputs, measured CPU usage, peak memory",
            "central": "central inputs, measured CPU usage, requested memory (usage basis)",
            "allocation": "central inputs, every reserved CPU at full use, requested memory (allocation basis)",
            "high": "high inputs on the allocation basis",
        },
        "inputs": inputs.to_dict(),
    }
    if missing:
        result["error"] = (
            f"no energy estimate: the site gives no {', '.join(missing)} "
            f"(pass them explicitly; no world-average default is used)"
        )
        return result

    ci = inputs.ci_g_per_kwh
    stages: Dict[str, _Tally] = {}
    total = _Tally()
    counts = {"usage_missing": 0, "partition_fallback": 0, "gpu_tasks": 0, "no_time": 0}

    for outcome in outcomes:
        tally = stages.setdefault(outcome.stage, _Tally())
        tally.tasks += 1
        total.tasks += 1
        completed = outcome.state == str(RunState.COMPLETED)
        in_progress = outcome.state in IN_PROGRESS_STATES
        if completed:
            tally.completed += 1
            total.completed += 1
        if in_progress:
            tally.in_progress += 1
            total.in_progress += 1
        cpus = outcome.cpus or 0
        if not cpus or outcome.elapsed_s is None:
            counts["no_time"] += 1
            continue

        usage = _usage(outcome)
        if usage is None:
            counts["usage_missing"] += 1
            usage = 1.0
        tdp, fallback = inputs.tdp_for(outcome.partition)
        if fallback:
            counts["partition_fallback"] += 1
        gpus = gpu_count(outcome.gres)
        if gpus:
            counts["gpu_tasks"] += 1
        gpu = inputs.gpu_w.get(outcome.partition) if gpus else None

        requested_gb = (outcome.req_mem_mb or 0.0) / 1024.0
        peak_gb = (
            outcome.max_rss_mb / 1024.0 if outcome.max_rss_mb else requested_gb
        )

        # The final attempt, then any earlier ones (known by allocated CPU time only).
        earlier_cpu_s = (outcome.cpu_s_all_attempts or 0.0) - outcome.elapsed_s * cpus
        if completed and outcome.success is not False:
            final = "successful"
        elif in_progress:
            final = "in_progress"
        else:
            final = "failed_or_repeated"
        attempts = [(outcome.elapsed_s / 3600.0, final)]
        if earlier_cpu_s > 1e-6:
            attempts.append((earlier_cpu_s / cpus / 3600.0, "failed_or_repeated"))
        for hours, bucket in attempts:
            energy = _attempt_energy(
                hours, cpus, usage, requested_gb, peak_gb, gpus, tdp, gpu, inputs.pue
            )
            tally.add(energy, bucket)
            total.add(energy, bucket)

    result["total"] = _summary(total, ci)
    result["stages"] = {name: _summary(t, ci) for name, t in sorted(stages.items())}
    if objects:
        result["objects"] = objects
        result["per_object"] = {
            "kwh": _round_cases({c: total.kwh[c] / objects for c in CASES}, 6),
            "g_co2e": None
            if ci is None
            else _round_cases(
                {c: total.kwh[c] * ci.case(c) / objects for c in CASES}, 1
            ),
        }
    result["assumptions"] = counts
    if ci is None:
        reason = inputs.ci_missing_reason or "the site gives no carbon_intensity"
        result["note"] = (
            f"energy only: {reason}, so no carbon figure "
            f"(pass --ci; no world-average default is used)"
        )
    return result


@dataclass
class StageForecast:
    """
    A planned stage for ``forecast``: ``tasks`` tasks of ``hours`` (low, central, high)
    wall time on ``cpus`` CPUs at CPU ``usage``, holding ``memory_gb`` (low = peak,
    central = requested, high = the largest reservation considered).
    """

    name: str
    tasks: int
    hours: Sequence[float]
    cpus: int
    usage: float
    memory_gb: Sequence[float]
    gpus: int = 0
    partition: Optional[str] = None


def forecast(stages: Iterable[StageForecast], inputs: CarbonInputs) -> Dict:
    """
    The energy and carbon of a planned campaign, in the same four cases as
    ``estimate``: ``low`` takes each stage's low hours and memory at its usage,
    ``central`` the central ones, ``allocation`` the central ones at full use and
    ``high`` the high ones at full use. Used by planning, and to reproduce the worked
    estimate of the hpc-campaign report (§1.4).
    """
    missing = [m for m in inputs.missing() if m != "carbon_intensity"]
    if missing:
        return {"error": f"no energy forecast: the site gives no {', '.join(missing)}"}
    ci = inputs.ci_g_per_kwh
    result = {"stages": {}, "inputs": inputs.to_dict()}
    total = _empty()
    for stage in stages:
        tdp, _ = inputs.tdp_for(stage.partition)
        gpu = inputs.gpu_w.get(stage.partition) if stage.gpus else None
        hours = Bounds.of(tuple(stage.hours))
        memory = Bounds.of(tuple(stage.memory_gb))
        kwh = {}
        for case in CASES:
            u = 1.0 if case in ("allocation", "high") else stage.usage
            h, m = hours.case(case), memory.case(case)
            gpu_w = gpu.case(case) if gpu is not None else 0.0
            watts = power_w(stage.cpus, tdp.case(case), u, m, stage.gpus, gpu_w)
            kwh[case] = stage.tasks * h * watts * inputs.pue.case(case) / 1000.0
            total[case] += kwh[case]
        result["stages"][stage.name] = {
            "watts_central": round(
                power_w(
                    stage.cpus,
                    tdp.central,
                    stage.usage,
                    memory.central,
                    stage.gpus,
                    gpu.central if gpu is not None else 0.0,
                ),
                2,
            ),
            "kwh": _round_cases(kwh),
            "kg_co2e": _round_cases(_carbon(kwh, ci), 2),
        }
    result["total"] = {"kwh": _round_cases(total), "kg_co2e": _round_cases(_carbon(total, ci), 2)}
    return result


# --- NESO Carbon Intensity API (network only when asked) -------------------------


def _fetch_json(url: str, timeout: float = 30.0) -> Dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _points(document: Dict) -> List[float]:
    data = document.get("data")
    if isinstance(data, dict):
        data = data.get("data", [])
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "data" in data[0]:
        data = [point for region in data for point in region.get("data", [])]
    values = []
    for point in data or []:
        intensity = point.get("intensity") or {}
        value = intensity.get("actual")
        if value is None:
            value = intensity.get("forecast")
        if value is not None:
            values.append(float(value))
    return values


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "autofit" / "carbon"


def neso_intensity(
    start: datetime,
    end: datetime,
    region_id: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    fetch: Callable[[str], Dict] = _fetch_json,
) -> Dict:
    """
    The mean GB grid carbon intensity (gCO2e/kWh) over ``start``-``end`` from the NESO
    Carbon Intensity API (CC BY 4.0, no key): national actuals, or the regional series
    of ``region_id`` (12 is South England). Requests are split into windows the API
    accepts and each response is cached under ``cache_dir`` (default
    ``~/.cache/autofit/carbon``), so a period is fetched once.

    Returns ``{"mean": ..., "points": n, "basis": ..., "start": ..., "end": ...}``.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    values = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + timedelta(days=NESO_MAX_DAYS), end)
        span = f"{_iso(window_start)}/{_iso(window_end)}"
        if region_id is None:
            url = f"{NESO_API}/intensity/{span}"
        else:
            url = f"{NESO_API}/regional/intensity/{span}/regionid/{region_id}"
        cache = cache_dir / (url[len(NESO_API) + 1 :].replace("/", "_").replace(":", "") + ".json")
        if cache.exists():
            document = json.loads(cache.read_text())
        else:
            document = fetch(url)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(document))
        values += _points(document)
        window_start = window_end
    if not values:
        raise ValueError(f"NESO returned no intensity for {start} → {end}")
    return {
        "mean": sum(values) / len(values),
        "points": len(values),
        "basis": "national (NESO actual)" if region_id is None else f"NESO region {region_id}",
        "start": _iso(start),
        "end": _iso(end),
    }


def begin_hint(
    series: Sequence[Tuple[datetime, float]],
    duration_s: float,
    now: Optional[datetime] = None,
) -> Optional[Dict]:
    """
    The start, within a forecast ``series`` of ``(half-hour start, gCO2e/kWh)``, that
    minimises the mean forecast intensity over a job of ``duration_s``, against
    starting now. Only worth acting on for a short array that can wait: over a
    campaign of days the window averages out (hpc-campaign report §1.4).
    """
    series = sorted(series)
    if not series:
        return None
    slots = max(1, math.ceil(duration_s / 1800.0))
    if len(series) < slots:
        return None
    now = now or series[0][0]

    def mean_from(i):
        return sum(v for _, v in series[i : i + slots]) / slots

    first = next((i for i, (t, _) in enumerate(series) if t >= now), 0)
    best = min(range(first, len(series) - slots + 1), key=mean_from)
    return {
        "begin": series[best][0].strftime("%Y-%m-%dT%H:%M:%S"),
        "mean_g_per_kwh": round(mean_from(best), 1),
        "now_mean_g_per_kwh": round(mean_from(first), 1),
        "slurm": f"--begin={series[best][0].strftime('%Y-%m-%dT%H:%M:%S')}",
    }


def neso_forecast(
    region_id: Optional[int] = None,
    now: Optional[datetime] = None,
    fetch: Callable[[str], Dict] = _fetch_json,
) -> List[Tuple[datetime, float]]:
    """
    The 48 h carbon-intensity forecast from now (national, or ``region_id``).
    """
    now = now or datetime.now(timezone.utc)
    if region_id is None:
        url = f"{NESO_API}/intensity/{_iso(now)}/fw48h"
    else:
        url = f"{NESO_API}/regional/intensity/{_iso(now)}/fw48h/regionid/{region_id}"
    document = fetch(url)
    data = document.get("data")
    if isinstance(data, dict):
        data = data.get("data", [])
    series = []
    for point in data or []:
        value = (point.get("intensity") or {}).get("forecast")
        if value is None or not point.get("from"):
            continue
        moment = datetime.strptime(point["from"], "%Y-%m-%dT%H:%MZ").replace(
            tzinfo=timezone.utc
        )
        series.append((moment, float(value)))
    return series


def apply_neso(inputs: CarbonInputs, measured: Dict) -> CarbonInputs:
    """
    ``inputs`` with the central carbon intensity replaced by a NESO period mean; the
    bounds widen to include it.
    """
    mean = measured["mean"]
    ci = inputs.ci_g_per_kwh
    low = mean if ci is None else min(ci.low, mean)
    high = mean if ci is None else max(ci.high, mean)
    inputs.ci_g_per_kwh = Bounds(low, mean, high)
    inputs.ci_basis = measured["basis"]
    inputs.ci_bases = {**inputs.ci_bases, measured["basis"]: round(mean, 1)}
    inputs.ci_source = (
        f"{measured['basis']} mean over {measured['start']} → {measured['end']} "
        f"({measured['points']} half-hours)"
    )
    return inputs


def period_of(outcomes: Iterable[TaskOutcome]) -> Optional[Tuple[datetime, datetime]]:
    """
    The first start and last end of ``outcomes`` (naive times taken as UTC).
    """
    starts, ends = [], []
    for outcome in outcomes:
        for value, into in ((outcome.start, starts), (outcome.end, ends)):
            if not value:
                continue
            try:
                moment = datetime.fromisoformat(value)
            except ValueError:
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            into.append(moment)
    if not starts or not ends:
        return None
    return min(starts), max(ends)
