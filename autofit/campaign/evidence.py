"""
Evidence for planning a campaign: measured costs read from the profiling and inference
summaries, each number graded by how well it fits the campaign's stage.

Three published summaries are read, by path or URL, never modified:

- ``profiling-summary@3`` (``autolens_profiling/dashboard/catalogue.json``, or the
  PyAutoPulse snapshot that wraps it under ``summary``): per-likelihood measurements
  (``runtime``, ``memory_host``, ``compile`` ...) per setup and host class;
- ``walltime-validation@1`` (``autolens_inference/dashboard/walltime_validation.json``):
  whole Nautilus stages, with wall time, likelihood evaluations and the implied cost
  per evaluation;
- ``inference-summary@2`` (``autolens_inference/dashboard/catalogue.json``): full
  pipeline runs, with their execution status and timings.

A stage names what it is in ``campaign.yaml`` under ``evidence.keys``. The keys
(``dataset_type``, ``variant``, ``instrument``, ``regime``, ``setup_id``,
``host_class``, ``model_variant``, ``stage``) are opaque to PyAutoFit: they are compared
for equality with the same fields of each record and nothing is inferred from them.

Every number carries one **grade**, best first:

- ``measured-current``: a current measurement with exactly the stage's keys (a
  walltime-validation or completed inference run, or a profiling record whose class
  is designated by a Pulse decision);
- ``provisional``: current and exact, but its class is not designated (profiling
  records are ``secondary`` with ``provisional_optimal`` until a human designation
  decision exists, which today is all of them);
- ``analogue``: current, but the instrument or regime differs (which is said);
- ``superseded``: an exact match whose record a newer one replaced;
- ``estimated``: from a pilot or a previous run of the campaign itself, or scaled
  from other numbers (e.g. evaluations × cost per evaluation);
- ``unsupported``: no record says anything.

The word *optimal* is never derived here. A setup is called optimal only when a Pulse
designation decision (``decisions.yaml``, ``kind: designation``) names it for the
stage's likelihood, regime and hardware class; ``designation_for`` finds one.
"""
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import yaml

GRADES = (
    "measured-current",
    "provisional",
    "analogue",
    "superseded",
    "estimated",
    "unsupported",
)

KEYS = (
    "dataset_type",
    "variant",
    "instrument",
    "regime",
    "setup_id",
    "host_class",
    "model_variant",
    "stage",
)

ANALOGUE_KEYS = ("instrument", "regime")
"""The keys that may differ for an ``analogue`` match; every other declared key must
be equal."""

HARDWARE_CLASS = {
    "hpc_ral_cpu": "cpu_many_core",
    "hpc_a100": "gpu_a100",
    "local_gpu": "gpu_a100",
    "local_laptop": "laptop",
}
"""Profiling host classes to the hardware classes of Pulse designation decisions."""

PER_CALL_KEYS = (
    "warm_iid_median_s",
    "vmap.per_call",
    "single_jit_median",
    "single_jit_block",
)
"""Which ``values`` of a profiling ``runtime`` record is the cost of one likelihood
call, in order of preference when the record names none."""


def grade_rank(grade: str) -> int:
    return GRADES.index(grade)


def worst(*grades: str) -> str:
    """
    The weakest of ``grades`` (a number built from several is no better than its
    weakest input).
    """
    return max(grades, key=grade_rank)


@dataclass
class EvidenceRecord:
    """
    One record of a summary, reduced to what planning reads.
    """

    id: str
    schema: str
    keys: Dict[str, Optional[str]]
    status: str
    measurement: Optional[str] = None
    values: Dict[str, float] = field(default_factory=dict)
    unit: Optional[str] = None
    record_class: Optional[str] = None
    decision_id: Optional[str] = None
    limitations: List[str] = field(default_factory=list)
    measured_at: Optional[str] = None
    source: Optional[str] = None
    extra: Dict = field(default_factory=dict)

    @property
    def provisional_class(self) -> bool:
        return self.record_class != "optimal" or not self.decision_id


@dataclass
class EvidenceSet:
    schema: str
    version: int
    source: str
    generated_at: Optional[str]
    records: List[EvidenceRecord]


@dataclass
class Match:
    record: EvidenceRecord
    grade: str
    differs: Dict[str, Tuple[Optional[str], Optional[str]]] = field(default_factory=dict)

    def describe(self) -> str:
        text = f"{self.record.id} [{self.grade}]"
        if self.differs:
            text += " (" + ", ".join(
                f"{key} not recorded" if got is None else f"{key} {got} not {wanted}"
                for key, (wanted, got) in self.differs.items()
            ) + ")"
        return text


def _read(source: Union[str, Path]) -> Dict:
    source = str(source)
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(source).read_text())


def _number_values(values) -> Dict[str, float]:
    if not isinstance(values, dict):
        return {}
    return {
        k: float(v) for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _profiling_record(raw: Dict, source: str) -> EvidenceRecord:
    likelihood = raw.get("likelihood") or {}
    resources = raw.get("resources") or {}
    throughput = raw.get("projected_node_throughput") or {}
    return EvidenceRecord(
        id=raw["id"],
        schema="profiling-summary",
        keys={
            "dataset_type": likelihood.get("dataset_type"),
            "variant": likelihood.get("variant"),
            "instrument": raw.get("instrument"),
            "regime": raw.get("regime"),
            "setup_id": raw.get("setup_id"),
            "host_class": raw.get("host_class"),
            "model_variant": raw.get("model_variant"),
            "stage": None,
        },
        status=raw.get("status") or "current",
        measurement=raw.get("measurement"),
        values=_number_values(raw.get("values")),
        unit=raw.get("unit"),
        record_class=raw.get("class"),
        decision_id=raw.get("decision_id"),
        limitations=[str(item) for item in raw.get("limitations") or []],
        measured_at=raw.get("measured_at"),
        source=source,
        extra={
            "metric": raw.get("metric"),
            "per_call_s": throughput.get("per_call_s"),
            "per_call_source": throughput.get("per_call_source"),
            "cpus_per_eval": resources.get("cpus_per_eval"),
            "peak_rss_gb": resources.get("peak_rss_gb"),
        },
    )


def _walltime_record(raw: Dict, source: str) -> EvidenceRecord:
    likelihood = raw.get("likelihood") or {}
    values = {
        key: raw[key]
        for key in ("wall_s", "likelihood_evals", "implied_s_per_eval", "cores")
        if isinstance(raw.get(key), (int, float))
    }
    return EvidenceRecord(
        id=raw["id"],
        schema="walltime-validation",
        keys={
            "dataset_type": likelihood.get("dataset_type"),
            "variant": likelihood.get("variant"),
            "instrument": raw.get("instrument"),
            "regime": None,
            "setup_id": raw.get("setup_id"),
            "host_class": raw.get("host_class"),
            "model_variant": raw.get("model_variant"),
            "stage": raw.get("stage"),
        },
        status="current",
        measurement="stage_walltime",
        values=_number_values(values),
        unit="s",
        source=source,
        extra={
            "sampler": raw.get("sampler"),
            "release_pin": raw.get("release_pin"),
            "source_pixels": raw.get("source_pixels"),
            "profiling_status": raw.get("profiling_status"),
        },
    )


def _inference_record(raw: Dict, source: str) -> Optional[EvidenceRecord]:
    execution = raw.get("execution") or {}
    if execution.get("completed") is not True:
        # A failed or unknown run measures nothing a plan can use.
        return None
    likelihood = raw.get("likelihood") or {}
    dataset = raw.get("dataset") or {}
    timings = raw.get("timings") or {}
    work = raw.get("work") or {}
    configuration = raw.get("configuration") or {}
    values = {}
    for key in ("sampling_s", "wall_s", "fit_s"):
        if isinstance(timings.get(key), (int, float)):
            values[key] = timings[key]
    if isinstance(work.get("likelihood_evaluations"), (int, float)):
        values["likelihood_evaluations"] = work["likelihood_evaluations"]
    return EvidenceRecord(
        id=raw["id"],
        schema="inference-summary",
        keys={
            "dataset_type": likelihood.get("dataset_type"),
            "variant": likelihood.get("variant"),
            "instrument": dataset.get("instrument"),
            "regime": None,
            "setup_id": configuration.get("config_name"),
            "host_class": raw.get("host_class"),
            "model_variant": raw.get("model_variant"),
            "stage": raw.get("stage"),
        },
        status="superseded" if raw.get("archived") else "current",
        measurement="pipeline_run",
        values=_number_values(values),
        unit="s",
        measured_at=raw.get("measured_at"),
        source=source,
    )


def read_evidence(source: Union[str, Path]) -> EvidenceSet:
    """
    The records of a ``profiling-summary@3``, ``walltime-validation@1`` or
    ``inference-summary@2`` file or URL (a PyAutoPulse snapshot is unwrapped).

    Raises
    ------
    ValueError
        If the document is none of the three, or another version of one of them.
    """
    document = _read(source)
    if "summary" in document and isinstance(document["summary"], dict):
        document = document["summary"]
    schema = document.get("schema")
    version = document.get("version")
    supported = {
        ("profiling-summary", 3): _profiling_record,
        ("walltime-validation", 1): _walltime_record,
        ("inference-summary", 2): _inference_record,
    }
    reader = supported.get((schema, version))
    if reader is None:
        raise ValueError(
            f"{source}: {schema}@{version} is not a supported evidence schema "
            f"({', '.join(f'{s}@{v}' for s, v in supported)})."
        )
    records = []
    for raw in document.get("records") or []:
        record = reader(raw, str(source))
        if record is not None:
            records.append(record)
    return EvidenceSet(
        schema=schema,
        version=version,
        source=str(source),
        generated_at=document.get("generated_at"),
        records=records,
    )


def read_designations(source: Optional[Union[str, Path]]) -> List[Dict]:
    """
    The ``kind: designation`` entries of a PyAutoPulse ``decisions.yaml`` (none when
    ``source`` is ``None`` or the index holds none).
    """
    if source is None:
        return []
    raw = yaml.safe_load(Path(source).read_text()) or {}
    return [
        entry
        for entry in raw.get("decisions") or []
        if isinstance(entry, dict) and entry.get("kind") == "designation"
    ]


def designation_for(keys: Dict[str, str], designations: Sequence[Dict]) -> Optional[Dict]:
    """
    The designation decision covering ``keys`` (likelihood, regime and hardware class),
    or ``None``. Only through this may a plan call a setup optimal.
    """
    likelihood = f"{keys.get('dataset_type')}/{keys.get('variant')}"
    hardware = HARDWARE_CLASS.get(keys.get("host_class") or "", keys.get("host_class"))
    current = [
        d
        for d in designations
        if d.get("likelihood") == likelihood
        and d.get("regime") == keys.get("regime")
        and d.get("hardware_class") == hardware
    ]
    replaced = {d.get("supersedes") for d in current if d.get("supersedes")}
    current = [d for d in current if d.get("id") not in replaced]
    return current[-1] if current else None


def grade_of(
    record: EvidenceRecord,
    wanted: Dict[str, str],
    designations: Sequence[Dict] = (),
) -> Optional[Match]:
    """
    The match of ``record`` against the stage keys ``wanted`` (only the keys the stage
    declares are compared), or ``None`` when it does not match even as an analogue.
    """
    differs = {}
    for key, value in wanted.items():
        if value is None or key not in KEYS:
            continue
        got = record.keys.get(key)
        if got == value:
            continue
        if got is None and key == "stage" and record.schema == "profiling-summary":
            # A profiling record is the cost of one likelihood call, which holds for
            # every stage that runs that likelihood.
            continue
        if key in ANALOGUE_KEYS:
            # Includes a schema that does not carry the key (walltime records have no
            # regime): not a mismatch, but not a match either.
            differs[key] = (value, got)
            continue
        return None

    if record.status in ("superseded", "archived"):
        return None if differs else Match(record, "superseded")
    if differs:
        return Match(record, "analogue", differs)
    return Match(record, _current_grade(record, designations))


def _current_grade(record: EvidenceRecord, designations: Sequence[Dict]) -> str:
    if record.schema != "profiling-summary":
        return "measured-current"
    # The record's decision must be the current (not superseded) designation for the
    # record's own likelihood, regime and hardware class, and name its setup.
    designation = designation_for(record.keys, designations) if record.decision_id else None
    if (
        designation is not None
        and designation.get("id") == record.decision_id
        and record.record_class == "optimal"
        and designation.get("optimal") == record.keys.get("setup_id")
    ):
        return "measured-current"
    return "provisional"


def matches(
    evidence: Iterable[EvidenceSet],
    wanted: Dict[str, str],
    measurement: Optional[str] = None,
    designations: Sequence[Dict] = (),
) -> List[Match]:
    """
    Every record of ``evidence`` matching ``wanted`` (optionally one measurement kind),
    best grade first, then newest. A stage that declares no keys matches nothing: with
    nothing compared, any record would match.
    """
    if not any(value is not None for key, value in wanted.items() if key in KEYS):
        return []
    found = []
    for evidence_set in evidence:
        for record in evidence_set.records:
            if measurement is not None and record.measurement != measurement:
                continue
            match = grade_of(record, wanted, designations)
            if match is not None:
                found.append(match)
    # Newest first (undated last), then a stable sort by grade.
    found.sort(key=lambda m: m.record.measured_at or "", reverse=True)
    found.sort(key=lambda m: grade_rank(m.grade))
    return found


def per_call_s(record: EvidenceRecord) -> Optional[float]:
    """
    The cost of one likelihood call recorded by a profiling ``runtime`` record.
    """
    if record.extra.get("per_call_s"):
        return float(record.extra["per_call_s"])
    for key in PER_CALL_KEYS:
        if key in record.values:
            return record.values[key]
    return None


def describe_limitation(text: str) -> str:
    """
    A record's limitation as a plan may quote it: Pulse's ``provisional_optimal`` marker
    is restated without the word, since only a designation decision may use it.
    """
    if text.strip().startswith("provisional_optimal"):
        return "provisional: no designation decision exists for this key"
    return text
