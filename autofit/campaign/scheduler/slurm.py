"""
The Slurm adapter: batched ``sacct`` / ``squeue`` / ``sinfo`` queries and ``sbatch``
script generation.

Nothing here runs at import. Every command goes through a *runner*,
``runner(arguments, input=None) -> stdout``, which by default calls ``subprocess`` and
in tests is a fake; a login node reached over ssh is just a runner that prefixes
``ssh <alias>``.

Every ``submit`` and every ``update_pending`` that changes a partition is linted
(``autofit.campaign.lint``) against a live count of GPU jobs pending on the target
partition: CPU-only work never goes on a GPU partition.
"""
import logging
import shlex
import subprocess
from typing import Callable, Dict, List, Optional, Sequence

from autofit.campaign.lint import check_partitions, split_partitions
from autofit.campaign.scheduler import sacct
from autofit.campaign.scheduler.base import (
    PartitionCapacity,
    SubmitRequest,
    TaskRecord,
)
from autofit.campaign.units import array_spec, format_time, requests_gpu

logger = logging.getLogger(__name__)

Runner = Callable[..., str]

SIGNAL = "B:USR1@600"


def subprocess_runner(arguments: Sequence[str], input: Optional[str] = None) -> str:
    completed = subprocess.run(
        list(arguments), input=input, capture_output=True, text=True, check=True
    )
    return completed.stdout


def _export(key: str, value: str) -> str:
    # A value naming a variable (``$SLURM_CPUS_PER_TASK``) is expanded by the job.
    if "$" in value:
        return f'export {key}="{value}"'
    return f"export {key}={shlex.quote(value)}"


def sbatch_script(request: SubmitRequest) -> str:
    """
    The batch script of one array.

    The job runs the stage's command in the background so the batch shell can forward
    the ``--signal=B:USR1@600`` pre-timeout warning to it (the fit flushes its
    checkpoint and writes ``timeout_pending``); the shell exits with the command's
    status.
    """
    resources = request.resources
    lines = ["#!/bin/bash", f"#SBATCH --job-name={request.job_name}"]
    if resources.partition:
        lines.append(f"#SBATCH --partition={resources.partition}")
    lines.append(f"#SBATCH --array={array_spec(request.task_ids, resources.throttle)}")
    lines.append(f"#SBATCH --cpus-per-task={resources.cpus}")
    if resources.mem_per_cpu_mb is not None:
        lines.append(f"#SBATCH --mem-per-cpu={int(resources.mem_per_cpu_mb)}")
    if resources.time_s is not None:
        lines.append(f"#SBATCH --time={format_time(resources.time_s)}")
    if resources.gres:
        lines.append(f"#SBATCH --gres={resources.gres}")
    if resources.nice is not None:
        lines.append(f"#SBATCH --nice={resources.nice}")
    if resources.qos:
        lines.append(f"#SBATCH --qos={resources.qos}")
    if resources.account:
        lines.append(f"#SBATCH --account={resources.account}")
    if request.after:
        dependency = ",".join(f"{kind}:{job_id}" for job_id, kind in request.after.items())
        lines.append(f"#SBATCH --dependency={dependency}")
        lines.append("#SBATCH --kill-on-invalid-dep=yes")
    lines.append(f"#SBATCH --signal={SIGNAL}")
    if request.log_dir:
        lines.append(f"#SBATCH --output={request.log_dir}/%x_%A_%a.out")
        lines.append(f"#SBATCH --error={request.log_dir}/%x_%A_%a.err")

    env = {
        "PYTHONUNBUFFERED": "1",
        "CAMPAIGN": request.campaign,
        "STAGE": request.stage,
        "CAMPAIGN_OFFSET": str(request.offset),
    }
    if request.manifest:
        env["MANIFEST"] = request.manifest
    env.update(request.env)

    command = shlex.split(request.script)
    if command and command[0].endswith(".py"):
        command = ["python"] + command

    lines += [
        "",
        "set -euo pipefail",
        *(_export(key, value) for key, value in env.items()),
        'export CAMPAIGN_INDEX=$((CAMPAIGN_OFFSET + SLURM_ARRAY_TASK_ID))',
        f"cd {shlex.quote(str(request.project_root))}",
        "",
        f"{' '.join(shlex.quote(part) for part in command)} &",
        "child=$!",
        "trap 'kill -USR1 \"$child\" 2>/dev/null || true' USR1",
        "trap 'kill -TERM \"$child\" 2>/dev/null || true' TERM",
        # A trapped signal ends `wait` early (128 + signal); wait again until the fit
        # itself has exited and keep only its own status, so a fit that checkpoints on
        # USR1 and then finishes cleanly exits 0, not 138.
        "while true; do",
        "    status=0",
        '    wait "$child" || status=$?',
        '    kill -0 "$child" 2>/dev/null || break',
        "done",
        'exit "$status"',
        "",
    ]
    return "\n".join(lines)


class SlurmScheduler:
    def __init__(self, runner: Optional[Runner] = None, user: Optional[str] = None, site=None):
        """
        A ``Scheduler`` over the Slurm command-line tools.

        Parameters
        ----------
        runner
            ``runner(arguments, input=None) -> stdout``; ``subprocess`` by default.
        user
            Whose jobs ``squeue`` lists (default: the current user, ``--me``).
        site
            The ``autonerves.site_config.Site`` naming the GPU-only partitions.
        """
        self.runner = runner or subprocess_runner
        self.user = user
        self.site = site

    def _run(self, arguments: List[str], input: Optional[str] = None) -> str:
        logger.debug(f"slurm: {' '.join(arguments)}")
        return self.runner(arguments, input=input) if input is not None else self.runner(arguments)

    def capacity(self, partitions: Optional[Sequence[str]] = None) -> Dict[str, PartitionCapacity]:
        """
        CPUs by state per partition (one ``sinfo``) and GPU-requesting jobs pending on
        each (one ``squeue``).
        """
        arguments = ["sinfo", "-N", "-h", "-o", "%P|%n|%t|%C|%m|%e"]
        if partitions:
            arguments += ["-p", ",".join(partitions)]
        result: Dict[str, PartitionCapacity] = {}
        seen = set()
        for line in self._run(arguments).splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            partition, node, state, cpus = parts[:4]
            partition = partition.rstrip("*")
            if (partition, node) in seen:
                continue
            seen.add((partition, node))
            capacity = result.setdefault(partition, PartitionCapacity(name=partition))
            capacity.nodes += 1
            # `*` marks an unreachable node: its CPUs are not free.
            if state.endswith("*") or state.startswith(("down", "drain", "fail")):
                capacity.nodes_unreachable += 1
            try:
                allocated, idle, other, total = (int(v) for v in cpus.split("/"))
            except ValueError:
                continue
            if state.endswith("*"):
                other, idle = other + idle, 0
            capacity.cpus_allocated += allocated
            capacity.cpus_idle += idle
            capacity.cpus_other += other
            capacity.cpus_total += total
            if len(parts) > 5 and parts[5].strip().isdigit():
                capacity.free_mem_mb = (capacity.free_mem_mb or 0) + float(parts[5])

        pending = self._run(["squeue", "-h", "-r", "-t", "PD", "-o", "%P|%b|%i"])
        counts: Dict[str, int] = {name: 0 for name in result}
        for line in pending.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            if requests_gpu(parts[1]):
                for partition in split_partitions(parts[0]):
                    counts[partition] = counts.get(partition, 0) + 1
        for name, capacity in result.items():
            capacity.pending_gpu_jobs = counts.get(name, 0)
        return result

    def pending_gpu_jobs(self, partitions: Sequence[str]) -> int:
        capacity = self.capacity(partitions)
        return sum(capacity[name].pending_gpu_jobs or 0 for name in partitions if name in capacity)

    def query(self, job_ids: Sequence[str]) -> List[TaskRecord]:
        """
        Every allocation of ``job_ids`` from one ``sacct`` call (steps folded in,
        attempts kept; see ``sacct.latest_per_task``).
        """
        if not job_ids:
            return []
        arguments = [
            "sacct",
            "-j",
            ",".join(str(job_id) for job_id in job_ids),
            "--parsable2",
            "--noheader",
            "--units=M",
            "--duplicates",
            "-o",
            ",".join(sacct.FIELDS),
        ]
        return sacct.parse(self._run(arguments), fields=sacct.FIELDS)

    def queue(self) -> List[Dict[str, str]]:
        """
        The user's queued tasks, one row per array task (``squeue -r``).
        """
        arguments = ["squeue", "-h", "-r", "-o", "%i|%F|%K|%j|%T|%r|%P|%C|%m|%b|%Q|%y"]
        arguments += ["-u", self.user] if self.user else ["--me"]
        names = ["id", "array_job", "array_task", "name", "state", "reason", "partition",
                 "cpus", "min_memory", "gres", "priority", "nice"]
        return [
            dict(zip(names, line.strip().split("|")))
            for line in self._run(arguments).splitlines()
            if line.strip()
        ]

    def check(self, request: SubmitRequest):
        resources = request.resources
        names = split_partitions(resources.partition)
        pending = self.pending_gpu_jobs(names) if names else 0
        verdict = check_partitions(
            names,
            cpus=resources.cpus,
            gres=resources.gres,
            throttle=resources.throttle,
            n_tasks=len(request.indices),
            pending_gpu_jobs=pending,
            site=self.site,
        )
        verdict.raise_for_refusal()
        return verdict

    def submit(self, request: SubmitRequest) -> str:
        """
        Lint, then ``sbatch --parsable`` the array's script on stdin; returns the job
        id.

        Raises
        ------
        PartitionLintError
            If the request would put CPU-only work on a GPU partition.
        """
        self.check(request)
        output = self._run(["sbatch", "--parsable"], input=sbatch_script(request))
        return output.strip().split(";")[0]

    def update_pending(self, job_id: str, **fields) -> None:
        """
        ``scontrol update`` of a job's still-pending tasks: ``time`` (Slurm time or
        seconds), ``mem_per_cpu_mb``, ``throttle``, ``nice``, ``partition`` (linted).
        Started tasks keep their request (Slurm splits them off).
        """
        mapping = {
            "time": "TimeLimit",
            "mem_per_cpu_mb": "MinMemoryCPU",
            "throttle": "ArrayTaskThrottle",
            "nice": "Nice",
            "partition": "Partition",
        }
        # `cpus` and `gres` describe the job for the lint; they are not updated.
        cpus = fields.pop("cpus", None)
        gres = fields.pop("gres", None)
        unknown = set(fields) - set(mapping)
        if unknown:
            raise ValueError(f"Cannot update {sorted(unknown)}; allowed {sorted(mapping)}.")
        if "partition" in fields:
            names = split_partitions(fields["partition"])
            verdict = check_partitions(
                names,
                cpus=cpus,
                gres=gres,
                throttle=fields.get("throttle"),
                pending_gpu_jobs=self.pending_gpu_jobs(names),
                site=self.site,
            )
            verdict.raise_for_refusal()
        arguments = ["scontrol", "update", f"JobId={job_id}"]
        for key, value in fields.items():
            if key == "time" and not isinstance(value, str):
                value = format_time(value)
            arguments.append(f"{mapping[key]}={int(value) if key == 'mem_per_cpu_mb' else value}")
        self._run(arguments)

    def cancel(self, job_id: str) -> None:
        self._run(["scancel", str(job_id)])
