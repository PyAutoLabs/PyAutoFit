"""
The end-of-campaign report: ``python -m autofit.campaign report``.

Built from a campaign's digest folder (``digest.json`` + ``tasks.jsonl``, written by
``status`` or ``retro``) and the site, it writes ``report.md`` and ``report.json``:

- compute: tasks, period, CPU-hours (successful and failed or repeated), peak CPUs;
- energy and carbon (``autofit.campaign.carbon``): the four cases, the share spent on
  failed or repeated work, per stage, per completed task and, given the number of
  objects, per object, with every input, its source and whether it is unverified;
- the failure census and a per-stage table of wall time, memory and CPU efficiency;
- results: the counts of a results index (``python -m autofit.results index``) when
  one is given;
- a paste-ready paragraph for a paper, which calls the figures an estimate, gives the
  range and names the carbon-intensity basis.

The report describes what was used. It compares against nothing and claims no
reduction: a comparison belongs to the reader, against the same analysis on other
settings.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from autofit.campaign.carbon import CarbonInputs, estimate, period_of
from autofit.campaign.digest import read_outcomes
from autofit.campaign.units import format_time
from autofit.tools.util import open_atomic


def _significant(value: float) -> str:
    return f"{value:.2g}" if value < 10 else f"{value:,.0f}"


def _scaled(values: Dict[str, float], small: str, large: str) -> Dict[str, str]:
    """
    The central, low and high of ``values`` in one unit: ``large`` (a thousand
    ``small``) when the central value reaches a thousand, else ``small``.
    """
    scale, unit = (1000.0, large) if values["central"] >= 1000 else (1.0, small)
    return {
        case: _significant(values[case] / scale)
        for case in ("low", "central", "high")
    } | {"unit": unit}


def _number(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3g}" if abs(value) < 10 else f"{value:,.1f}"
    return f"{int(value):,}"


def results_counts(index_dir) -> Optional[Dict]:
    """
    The result-state counts of a results index (``<index_dir>/index.json``), or
    ``None`` if there is none.
    """
    if index_dir is None:
        return None
    path = Path(index_dir) / "index.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    return {
        "index": str(Path(index_dir)),
        "updated": manifest.get("updated"),
        "counts": manifest.get("counts", {}),
    }


def paper_paragraph(digest: Dict, carbon: Dict) -> Optional[str]:
    """
    A paragraph for the paper's computing statement, or ``None`` without an energy
    estimate.
    """
    if "error" in carbon:
        return None
    totals = digest["totals"]
    inputs = carbon["inputs"]
    kwh = carbon["total"]["kwh"]
    kg = carbon["total"]["kg_co2e"]
    share = carbon["total"]["failed_or_repeated_share"]
    period = digest.get("period") or {}
    when = (
        f" between {period['first_submit'][:10]} and {period['last_end'][:10]}"
        if period.get("first_submit") and period.get("last_end")
        else ""
    )
    tdp = sorted({tuple(v) for v in inputs["tdp_w_per_cpu"].values()})
    tdp_text = "; ".join(f"{c:g} W (range {lo:g}–{hi:g} W)" for lo, c, hi in tdp)
    pue = inputs["pue"]
    site = f" on the {inputs['site']} site" if inputs.get("site") else ""

    text = (
        f"The {digest['campaign']} campaign ran {totals['tasks']:,} scheduler tasks"
        f"{site}{when}, allocating {totals['cpu_h_all_attempts']:,.0f} CPU-hours. "
        f"Following the Green Algorithms methodology (Lannelongue et al. 2021), with "
        f"wall time, CPU time and requested memory from scheduler accounting, a power "
        f"draw of {tdp_text} per scheduler CPU, a data-centre PUE of {pue[1]:g} "
        f"(range {pue[0]:g}–{pue[2]:g})"
    )
    if kg is not None:
        ci = inputs["ci_g_per_kwh"]
        bases = ", ".join(
            f"{k} {v:g}" for k, v in inputs["ci_bases"].items() if k != inputs["ci_basis"]
        )
        text += (
            f" and a grid carbon intensity of {ci[1]:g} gCO2e/kWh ({inputs['ci_basis']} "
            f"basis; range {ci[0]:g}–{ci[2]:g} across bases"
            + (f": {bases}" if bases else "")
            + ")"
        )
    energy = _scaled(kwh, "kWh", "MWh")
    text += (
        f", we estimate the campaign used ≈{energy['central']} {energy['unit']} "
        f"(range {energy['low']}–{energy['high']} {energy['unit']})"
    )
    if kg is not None:
        mass = _scaled(kg, "kgCO2e", "tCO2e")
        text += (
            f", or ≈{mass['central']} {mass['unit']} (range {mass['low']}–"
            f"{mass['high']} {mass['unit']})"
        )
    per_object = carbon.get("per_object")
    if per_object and per_object.get("g_co2e"):
        text += (
            f", ≈{per_object['g_co2e']['central'] / 1000:.2g} kgCO2e per object "
            f"over {carbon['objects']:,} objects"
        )
    text += "."
    if share is not None:
        text += (
            f" About {share:.0%} of the central estimate was spent on failed, "
            f"timed-out or repeated runs."
        )
    text += (
        " The upper bound charges every reserved CPU at full use and the requested "
        "memory (allocation basis) with the high inputs. The estimate excludes embodied "
        "hardware emissions, storage, idle capacity held for the campaign and "
        "development runs."
    )
    if inputs.get("unverified"):
        text += (
            " Site constants not yet confirmed against the cluster are used as ranges."
        )
    return text


def _cases_row(name: str, kwh: Dict, kg: Optional[Dict]) -> str:
    cells = [name]
    for case in ("low", "central", "allocation", "high"):
        value = _number(kwh[case])
        if kg is not None:
            value += f" / {_number(kg[case])}"
        cells.append(value)
    return "| " + " | ".join(cells) + " |"


def build_report(
    directory,
    site=None,
    inputs: Optional[CarbonInputs] = None,
    objects: Optional[int] = None,
    results_index=None,
) -> Dict:
    """
    The report of the campaign whose digest is in ``directory``.
    """
    directory = Path(directory)
    digest = json.loads((directory / "digest.json").read_text())
    outcomes = read_outcomes(directory / "tasks.jsonl")
    inputs = inputs or CarbonInputs.from_site(site)
    carbon = estimate(outcomes, inputs, objects=objects)
    period = period_of(outcomes)
    return {
        "campaign": digest.get("campaign"),
        "digest": digest,
        "carbon": carbon,
        "results": results_counts(results_index),
        "period": None
        if period is None
        else [period[0].isoformat(), period[1].isoformat()],
        "paragraph": paper_paragraph(digest, carbon),
    }


def report_markdown(report: Dict) -> str:
    digest = report["digest"]
    carbon = report["carbon"]
    totals = digest["totals"]
    lines: List[str] = [f"# Campaign report: {digest['campaign']}", ""]

    period = digest.get("period") or {}
    lines += ["## Compute", ""]
    lines.append(
        f"- {totals['tasks']:,} scheduler tasks: "
        + ", ".join(f"{k} {v:,}" for k, v in totals["states"].items())
    )
    if period:
        lines.append(f"- Period: {period.get('first_submit')} → {period.get('last_end')}")
    cpu_all, cpu_ok = totals["cpu_h_all_attempts"], totals["cpu_h_successful"]
    share = 1 - cpu_ok / cpu_all if cpu_all else 0
    lines.append(
        f"- CPU-hours allocated: {cpu_all:,.0f}, of which {cpu_ok:,.0f} in completed "
        f"tasks ({share:.0%} failed, timed-out or repeated)"
    )
    peak = digest.get("peak_cpus")
    if isinstance(peak, dict) and peak.get("cpus"):
        lines.append(f"- Peak CPUs held: {peak['cpus']:,} ({peak.get('at')})")
    lines.append("")

    lines += ["## Energy and carbon (estimate)", ""]
    if "error" in carbon:
        lines += [f"Not estimated: {carbon['error']}.", ""]
    else:
        has_carbon = carbon["total"]["kg_co2e"] is not None
        unit = "kWh / kgCO2e" if has_carbon else "kWh"
        lines.append(f"| stage ({unit}) | low | central | allocation | high |")
        lines.append("|---|---|---|---|---|")
        for name, stage in carbon["stages"].items():
            lines.append(_cases_row(name, stage["kwh"], stage["kg_co2e"]))
        lines.append(
            _cases_row("**total**", carbon["total"]["kwh"], carbon["total"]["kg_co2e"])
        )
        lines.append("")
        for case, meaning in carbon["cases"].items():
            lines.append(f"- **{case}**: {meaning}")
        total = carbon["total"]
        lines.append(
            f"- Failed or repeated work: {_number(total['failed_or_repeated_kwh']['central'])} "
            f"kWh of {_number(total['kwh']['central'])} central "
            f"({total['failed_or_repeated_share']:.0%})"
        )
        rows = ["", "| stage | per completed task, central | failed or repeated share |", "|---|---|---|"]
        for name, stage in carbon["stages"].items():
            per = stage.get("per_completed_task")
            if per:
                text = f"{_number(per['kwh']['central'])} kWh"
                if per["g_co2e"]:
                    text += f", {_number(per['g_co2e']['central'])} gCO2e"
            else:
                text = "-"
            share_text = (
                "-"
                if stage["failed_or_repeated_share"] is None
                else f"{stage['failed_or_repeated_share']:.0%}"
            )
            rows.append(f"| {name} | {text} | {share_text} |")
        lines += rows
        if carbon.get("per_object"):
            per = carbon["per_object"]
            text = f"{_number(per['kwh']['central'])} kWh"
            if per["g_co2e"]:
                text += f", {_number(per['g_co2e']['central'])} gCO2e"
            lines += ["", f"Per object ({carbon['objects']:,} objects), central: {text}."]
        if carbon.get("note"):
            lines += ["", carbon["note"] + "."]
        lines.append("")

        inputs = carbon["inputs"]
        lines += ["### Inputs", "", "| input | low | central | high | source |", "|---|---|---|---|---|"]
        for name, bounds in inputs["tdp_w_per_cpu"].items():
            lines.append(
                f"| W per CPU ({name}) | {bounds[0]} | {bounds[1]} | {bounds[2]} | site {inputs['site']} |"
            )
        for name, bounds in inputs["gpu_w"].items():
            lines.append(f"| W per GPU ({name}) | {bounds[0]} | {bounds[1]} | {bounds[2]} | site {inputs['site']} |")
        if inputs["pue"]:
            p = inputs["pue"]
            lines.append(f"| PUE | {p[0]} | {p[1]} | {p[2]} | site {inputs['site']} |")
        if inputs["ci_g_per_kwh"]:
            c = inputs["ci_g_per_kwh"]
            lines.append(f"| gCO2e/kWh | {c[0]} | {c[1]} | {c[2]} | {inputs['ci_source']} |")
        lines.append("| memory | | 0.3725 W/GB | | Green Algorithms |")
        lines.append("")
        if inputs.get("unverified"):
            lines.append(f"Unverified site values: {', '.join(inputs['unverified'])}.")
        assumptions = carbon.get("assumptions", {})
        notes = []
        if assumptions.get("usage_missing"):
            notes.append(f"{assumptions['usage_missing']:,} tasks without CPU time charged at full use")
        if assumptions.get("partition_fallback"):
            notes.append(f"{assumptions['partition_fallback']:,} tasks on partitions the site does not list (widest range used)")
        if assumptions.get("gpu_tasks"):
            notes.append(f"{assumptions['gpu_tasks']:,} GPU tasks, GPUs charged at full power")
        if assumptions.get("no_time"):
            notes.append(f"{assumptions['no_time']:,} tasks with no wall time (not charged)")
        if notes:
            lines.append("Assumptions: " + "; ".join(notes) + ".")
        lines.append("")

    lines += ["## Failure census", "", "| stage | tasks | states | failure classes |", "|---|---|---|---|"]
    for name, stage in digest["stages"].items():
        states = ", ".join(f"{k} {v:,}" for k, v in stage["states"].items())
        classes = ", ".join(f"{k} {v:,}" for k, v in stage.get("failure_classes", {}).items()) or "-"
        lines.append(f"| {name} | {stage['tasks']:,} | {states} | {classes} |")
    lines.append("")

    lines += [
        "## Stages",
        "",
        "| stage | wall p50 / p95 / max | requested time | peak memory p95 / requested (MB) | CPU efficiency p50 | CPU-h ok / all |",
        "|---|---|---|---|---|---|",
    ]
    for name, stage in digest["stages"].items():
        wall = stage["wall_s"]
        lines.append(
            f"| {name} | {format_time(wall['p50']) or '-'} / {format_time(wall['p95']) or '-'} / "
            f"{format_time(wall['max']) or '-'} | {format_time(stage['request']['time_s']) or '-'} | "
            f"{_number(stage['rss_mb']['p95'])} / {_number(stage['request']['mem_mb'])} | "
            f"{stage['cpu_efficiency_p50'] if stage['cpu_efficiency_p50'] is not None else '-'} | "
            f"{_number(stage['cpu_h']['successful'])} / {_number(stage['cpu_h']['all_attempts'])} |"
        )
    flags = [
        f"- {name}: {flag['flag']} — {flag['detail']}"
        for name, stage in digest["stages"].items()
        for flag in stage.get("flags", [])
    ]
    if flags:
        lines += ["", "Flags:"] + flags
    lines.append("")

    if report.get("results"):
        results = report["results"]
        lines += ["## Results", ""]
        lines.append(
            f"Results index `{results['index']}` (updated {results['updated']}): "
            + ", ".join(f"{k} {v:,}" for k, v in results["counts"].items())
        )
        lines.append("")

    if report.get("paragraph"):
        lines += ["## For the paper", "", report["paragraph"], ""]
    return "\n".join(lines)


def write_report(directory, report: Dict, output=None) -> Path:
    """
    Write ``report.md`` and ``report.json`` (the report without the digest, which is
    beside it) into ``output`` (default: the digest folder).
    """
    output = Path(output or directory)
    output.mkdir(parents=True, exist_ok=True)
    with open_atomic(output / "report.md", "w") as f:
        f.write(report_markdown(report))
    with open_atomic(output / "report.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "digest"}, f, indent=1, default=str)
    return output
