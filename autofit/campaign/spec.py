"""
``campaign.yaml``: the single declaration of a modelling campaign.

.. code-block:: yaml

    campaign: euclid_dr1_next10k
    project_root: /mnt/ral/jnightin/euclid_dr1
    manifest: hpc/run_manifests/next10k.txt   # 0-based line = campaign index
    manifest_fields: [sample, tile]           # optional names for a line's fields
    chunk: 1000                               # tasks per array, ≤ the site's max_array_size
    site: ral                                 # autonerves site layer
    scheduler: slurm                          # slurm (default) or local
    status_root: output                       # where fits write *.status.json
    log_dir: hpc/logs
    stages:
      vis_lp:
        script: scripts/initial_lens_model.py --stage=vis_lp
        env: {JAX_PLATFORMS: cpu}
        resources: {cpus: 4, mem_per_cpu_mb: 3072, time: "04:00:00",
                    partition: ral, throttle: 400, nice: 1000}
        success: {zip_complete: "output/{sample}/{tile}/initial_lens_model/vis_lp",
                  members: [files/samples_summary.json]}
        retry: {timeout: {time: "p95+25%", max: 1}, oom: {mem: "peak*1.5", max: 1}}
      vis_pix:
        after: {vis_lp: aftercorr}
        script: scripts/pixelization.py

Each manifest line is one object; its 0-based line number is its *campaign index* and
is the same in every stage. The manifest is split into arrays of at most ``chunk``
tasks; inside an array the Slurm task id runs from 0 and the batch script exports
``CAMPAIGN_INDEX = CAMPAIGN_OFFSET + SLURM_ARRAY_TASK_ID``, so per-index dependencies
(``aftercorr``) line up between stages chunk by chunk.

``retry`` is a policy for *proposals* only: the collector proposes resubmits sized
from the failed tasks' own measurements and a human approves every one.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

from autofit.campaign.lint import check_partitions, split_partitions
from autofit.campaign.units import parse_memory_mb, parse_time

CAMPAIGN_FIELDS = {
    "campaign",
    "project_root",
    "manifest",
    "manifest_fields",
    "manifest_separator",
    "chunk",
    "site",
    "scheduler",
    "status_root",
    "log_dir",
    "stages",
    "note",
}
STAGE_FIELDS = {"script", "env", "resources", "after", "seed", "success", "retry", "note"}
RESOURCE_FIELDS = {
    "cpus",
    "mem_per_cpu_mb",
    "mem_mb",
    "time",
    "partition",
    "gres",
    "throttle",
    "nice",
    "qos",
    "account",
}
DEPENDENCY_KINDS = {"aftercorr", "afterok", "afterany"}
RETRY_CLASSES = {
    "timeout",
    "oom",
    "node_fail",
    "exception",
    "bad_input",
    "missing_seed",
    "upstream_incomplete",
}
SCHEDULERS = {"slurm", "local"}


class CampaignException(Exception):
    pass


@dataclass
class Resources:
    cpus: int = 1
    mem_per_cpu_mb: Optional[int] = None
    time_s: Optional[int] = None
    partition: Optional[str] = None
    gres: Optional[str] = None
    throttle: Optional[int] = None
    nice: Optional[int] = None
    qos: Optional[str] = None
    account: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "Resources":
        raw = dict(raw or {})
        cpus = int(raw.get("cpus", 1))
        mem_per_cpu = raw.get("mem_per_cpu_mb")
        if mem_per_cpu is None and raw.get("mem_mb") is not None:
            # One knob: memory follows the CPU count (`--mem-per-cpu`), so a total is
            # turned into a per-CPU value.
            mem_per_cpu = -(-parse_memory_mb(raw["mem_mb"]) // cpus)
        return cls(
            cpus=cpus,
            mem_per_cpu_mb=None
            if mem_per_cpu is None
            else int(parse_memory_mb(mem_per_cpu)),
            time_s=parse_time(raw.get("time")),
            partition=raw.get("partition"),
            gres=raw.get("gres"),
            throttle=None if raw.get("throttle") is None else int(raw["throttle"]),
            nice=None if raw.get("nice") is None else int(raw["nice"]),
            qos=raw.get("qos"),
            account=raw.get("account"),
        )

    @property
    def mem_mb(self) -> Optional[int]:
        if self.mem_per_cpu_mb is None:
            return None
        return self.mem_per_cpu_mb * self.cpus


@dataclass
class Success:
    zip_complete: Optional[str] = None
    members: List[str] = field(default_factory=list)


@dataclass
class Stage:
    name: str
    script: str
    resources: Resources
    env: Dict[str, str] = field(default_factory=dict)
    after: Dict[str, str] = field(default_factory=dict)
    seed: Optional[str] = None
    success: Success = field(default_factory=Success)
    retry: Dict[str, Dict] = field(default_factory=dict)


class Campaign:
    def __init__(self, raw: Dict, source: Optional[Path] = None):
        """
        A ``campaign.yaml`` (``Campaign.load``). Construction does not validate; call
        ``validate`` (``load`` does) for the list of problems.
        """
        self.raw = raw
        self.source = source
        self.name = raw.get("campaign")
        root = raw.get("project_root")
        if root is None and source is not None:
            root = Path(source).parent
        self.project_root = Path(root) if root is not None else Path.cwd()
        self.manifest_path = raw.get("manifest")
        self.manifest_fields = list(raw.get("manifest_fields") or [])
        self.manifest_separator = raw.get("manifest_separator")
        self.chunk = raw.get("chunk")
        self.site_name = raw.get("site")
        self.scheduler = raw.get("scheduler", "slurm")
        self.status_root = raw.get("status_root", "output")
        self.log_dir = raw.get("log_dir", "hpc/logs")
        self.stages: Dict[str, Stage] = {}
        for name, stage in (raw.get("stages") or {}).items():
            stage = stage or {}
            if not isinstance(stage, dict):
                continue
            success = stage.get("success") or {}
            self.stages[name] = Stage(
                name=name,
                script=stage.get("script", ""),
                resources=Resources.from_dict(stage.get("resources")),
                env={k: str(v) for k, v in (stage.get("env") or {}).items()},
                after=dict(stage.get("after") or {}),
                seed=stage.get("seed"),
                success=Success(
                    zip_complete=success.get("zip_complete"),
                    members=list(success.get("members") or []),
                ),
                retry=dict(stage.get("retry") or {}),
            )
        self._site = None
        self._manifest = None

    @classmethod
    def load(cls, path: Union[str, Path], site=None) -> "Campaign":
        """
        Load and validate the campaign file at ``path``.

        Parameters
        ----------
        site
            The ``autonerves.site_config.Site`` to validate against; by default the
            file's ``site:`` is loaded through the autonerves site layer.

        Raises
        ------
        CampaignException
            If the file cannot be read or fails ``validate``.
        """
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text())
        except OSError as exception:
            raise CampaignException(f"Campaign file {path} cannot be read: {exception}")
        except yaml.YAMLError as exception:
            raise CampaignException(f"Campaign file {path} is not valid YAML: {exception}")
        if not isinstance(raw, dict):
            raise CampaignException(f"Campaign file {path} must be a mapping.")
        campaign = cls(raw, source=path)
        if site is not None:
            campaign._site = site
        errors = campaign.validate()
        if errors:
            raise CampaignException(
                f"Campaign file {path} is invalid:\n  " + "\n  ".join(errors)
            )
        return campaign

    @property
    def site(self):
        """
        The site layer named by ``site:`` (``None`` when the file names none).
        """
        if self._site is None and self.site_name:
            from autonerves import conf
            from autonerves.site_config import load_site, site_path

            paths = getattr(conf.instance, "paths", ())
            self._site = load_site(site_path(str(self.site_name), config_paths=paths))
        return self._site

    @property
    def directory(self) -> Path:
        """
        Where the campaign keeps its job ledger and digest:
        ``<project_root>/campaign/<name>/``.
        """
        return self.project_root / "campaign" / str(self.name)

    def path(self, value: Union[str, Path]) -> Path:
        value = Path(value)
        return value if value.is_absolute() else self.project_root / value

    def manifest(self) -> List[str]:
        """
        The manifest's lines (blank lines and ``#`` comments are not objects but keep
        their line number, so an index is always a line number).
        """
        if self._manifest is None:
            text = self.path(self.manifest_path).read_text()
            self._manifest = text.splitlines()
        return self._manifest

    def indices(self) -> List[int]:
        return [
            index
            for index, line in enumerate(self.manifest())
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def fields(self, index: int) -> Dict[str, str]:
        """
        The template fields of manifest line ``index``: ``index``, ``line`` and, with
        ``manifest_fields``, each named field of the split line.
        """
        line = self.manifest()[index].strip()
        values = {"index": str(index), "line": line}
        if self.manifest_fields:
            parts = line.split(self.manifest_separator)
            for name, part in zip(self.manifest_fields, parts):
                values[name] = part.strip()
        return values

    def chunk_size(self) -> int:
        size = self.chunk
        maximum = self.max_array_size()
        if size is None:
            size = maximum or 1000
        return int(size)

    def max_array_size(self) -> Optional[int]:
        site = self._site_or_none()
        if site is None:
            return None
        sizes = [
            site.partitions[name].max_array_size
            for stage in self.stages.values()
            for name in split_partitions(stage.resources.partition)
            if name in site.partitions
            and site.partitions[name].max_array_size is not None
        ]
        return min(sizes) if sizes else None

    def chunks(self, indices: Optional[List[int]] = None) -> List[Tuple[int, List[int]]]:
        """
        ``(offset, indices)`` per array: campaign indices grouped by
        ``offset = (index // chunk) * chunk``, so one index is always in the same array
        in every stage and its Slurm task id is ``index - offset``.
        """
        size = self.chunk_size()
        indices = self.indices() if indices is None else sorted(set(indices))
        chunks: Dict[int, List[int]] = {}
        for index in indices:
            chunks.setdefault((index // size) * size, []).append(index)
        return sorted(chunks.items())

    def order(self) -> List[str]:
        """
        The stage names in dependency order (a stage after every stage it waits on).

        Raises
        ------
        CampaignException
            On a dependency cycle or an unknown stage.
        """
        ordered: List[str] = []
        visiting = set()

        def visit(name, trail):
            if name in ordered:
                return
            if name not in self.stages:
                raise CampaignException(f"Stage {trail[-1]!r} waits on unknown stage {name!r}.")
            if name in visiting:
                raise CampaignException(
                    f"Stage dependency cycle: {' -> '.join(trail + [name])}."
                )
            visiting.add(name)
            for upstream in self.stages[name].after:
                visit(upstream, trail + [name])
            visiting.discard(name)
            ordered.append(name)

        for name in self.stages:
            visit(name, [name])
        return ordered

    def _site_or_none(self):
        try:
            return self.site
        except Exception:
            return None

    def validate(self) -> List[str]:
        """
        Every problem with the campaign, as messages (empty when valid).
        """
        raw = self.raw
        errors = []
        unknown = set(raw) - CAMPAIGN_FIELDS
        if unknown:
            errors.append(f"unknown campaign keys {sorted(unknown)}")
        for key in ("campaign", "manifest", "stages"):
            if not raw.get(key):
                errors.append(f"{key}: required")
        if self.scheduler not in SCHEDULERS:
            errors.append(f"scheduler: {self.scheduler!r} is not one of {sorted(SCHEDULERS)}")
        if self.chunk is not None and (not isinstance(self.chunk, int) or self.chunk < 1):
            errors.append(f"chunk: must be a positive integer, not {self.chunk!r}")

        site = None
        if self.site_name:
            try:
                site = self.site
            except Exception as exception:
                errors.append(f"site: {exception}")

        maximum = self.max_array_size() if site is not None else None
        if maximum is not None and self.chunk is not None and self.chunk > maximum:
            errors.append(
                f"chunk: {self.chunk} is above the site's max_array_size {maximum}"
            )

        for name, raw_stage in (raw.get("stages") or {}).items():
            where = f"stages.{name}"
            if not isinstance(raw_stage, dict):
                errors.append(f"{where}: must be a mapping")
                continue
            unknown = set(raw_stage) - STAGE_FIELDS
            if unknown:
                errors.append(f"{where}: unknown keys {sorted(unknown)}")
            if not raw_stage.get("script"):
                errors.append(f"{where}.script: required")
            resources = raw_stage.get("resources") or {}
            unknown = set(resources) - RESOURCE_FIELDS
            if unknown:
                errors.append(f"{where}.resources: unknown keys {sorted(unknown)}")
            try:
                Resources.from_dict(resources)
            except (ValueError, TypeError) as exception:
                errors.append(f"{where}.resources: {exception}")
                continue
            for upstream, kind in (raw_stage.get("after") or {}).items():
                if upstream not in (raw.get("stages") or {}):
                    errors.append(f"{where}.after: unknown stage {upstream!r}")
                if kind not in DEPENDENCY_KINDS:
                    errors.append(
                        f"{where}.after.{upstream}: {kind!r} is not one of "
                        f"{sorted(DEPENDENCY_KINDS)}"
                    )
            for failure, policy in (raw_stage.get("retry") or {}).items():
                if failure not in RETRY_CLASSES:
                    errors.append(f"{where}.retry: unknown failure class {failure!r}")
                elif policy not in (None, "none") and not isinstance(policy, dict):
                    errors.append(f"{where}.retry.{failure}: must be a mapping or none")

            if name in self.stages and self.scheduler == "slurm":
                errors += self._resource_errors(where, self.stages[name], site)

        if not errors:
            try:
                self.order()
            except CampaignException as exception:
                errors.append(str(exception))
        return errors

    def _resource_errors(self, where: str, stage: Stage, site) -> List[str]:
        errors = []
        resources = stage.resources
        if not resources.partition:
            errors.append(f"{where}.resources.partition: required for a Slurm campaign")
            return errors
        names = split_partitions(resources.partition)
        if site is not None:
            for partition_name in names:
                if partition_name not in site.partitions:
                    errors.append(
                        f"{where}.resources.partition: site {site.name!r} has no "
                        f"partition {partition_name!r}"
                    )
                    continue
                partition = site.partitions[partition_name]
                max_time = parse_time(partition.max_time) if partition.max_time else None
                if max_time and resources.time_s and resources.time_s > max_time:
                    errors.append(
                        f"{where}.resources.time: above {partition_name!r} max_time "
                        f"{partition.max_time}"
                    )
                mem_node = partition.mem_per_node_mb
                if mem_node and resources.mem_mb and resources.mem_mb > mem_node:
                    errors.append(
                        f"{where}.resources: {resources.mem_mb} MB per task is above "
                        f"{partition_name!r} mem_per_node_mb {mem_node}"
                    )
        # Static half of the partition lint: a mixed or oversized request can never
        # qualify for the exemption; the pending-GPU check runs again at submit time.
        verdict = check_partitions(
            names,
            cpus=resources.cpus,
            gres=resources.gres,
            throttle=resources.throttle,
            n_tasks=None,
            pending_gpu_jobs=0,
            site=site,
        )
        if not verdict.ok:
            errors.append(f"{where}.resources.partition: " + "; ".join(verdict.reasons))
        return errors
