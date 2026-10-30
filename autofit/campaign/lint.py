"""
The partition lint: CPU-only work never goes on a GPU partition.

On 2026-09-30 CPU-only arrays submitted to RAL with ``--partition=ral,gpu`` filled every
CPU of both A100 nodes and left all eight GPUs idle and unschedulable for hours. Every
submit, resubmit and pending-job update in ``autofit.campaign`` passes through
``check_partitions``, which refuses a partition list containing a GPU partition for a
request that asks for no GPU.

A GPU partition is one the site file marks ``gpu_only`` (``autonerves.site_config``);
with no site file, any partition whose name contains ``gpu``. Whether the request asks
for a GPU is read from its ``--gres`` / ``--gpus`` / TRES request, never from the
partition name.

The one exemption is a small, bounded CPU timing leg on a GPU partition on its own
(never a mixed list such as ``ral,gpu``), and only when all three hold at submit time:

- at most 8 CPUs per task;
- an array throttle of at most ``%2`` (or at most two tasks in all);
- no job pending on that partition that requests a GPU.

When the pending-GPU count is unknown the exemption is refused: the rule fails closed.
"""
import re
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from autofit.campaign.units import parse_array_spec, requests_gpu

EXEMPT_MAX_CPUS = 8
EXEMPT_MAX_THROTTLE = 2


class PartitionLintError(Exception):
    pass


@dataclass
class LintResult:
    ok: bool
    partitions: List[str]
    gpu_partitions: List[str]
    requests_gpu: bool
    exempt: bool = False
    reasons: List[str] = field(default_factory=list)

    def raise_for_refusal(self):
        if not self.ok:
            raise PartitionLintError("; ".join(self.reasons))


def split_partitions(partitions: Union[str, Sequence[str], None]) -> List[str]:
    if partitions is None:
        return []
    if isinstance(partitions, str):
        partitions = partitions.split(",")
    return [p.strip() for p in partitions if p and p.strip()]


def gpu_partitions_of(partitions: Sequence[str], site=None) -> List[str]:
    """
    The members of ``partitions`` that are GPU partitions: ``gpu_only`` in ``site``
    when it names the partition, else any whose name contains ``gpu``.
    """
    found = []
    for name in partitions:
        site_partition = None
        if site is not None:
            site_partition = getattr(site, "partitions", {}).get(name)
        if site_partition is not None:
            if site_partition.gpu_only:
                found.append(name)
        elif "gpu" in name.lower():
            found.append(name)
    return found


def check_partitions(
    partitions: Union[str, Sequence[str], None],
    cpus: Optional[int] = None,
    gres: Optional[str] = None,
    throttle: Optional[int] = None,
    n_tasks: Optional[int] = None,
    pending_gpu_jobs: Optional[int] = None,
    site=None,
) -> LintResult:
    """
    Lint one request's partition list against the GPU-partition rule.

    Parameters
    ----------
    partitions
        A partition name, a Slurm list (``"ral,gpu"``) or a list of names.
    cpus
        CPUs per task.
    gres
        The request's GPU / generic-resource request (``gpu:1``, ``gres/gpu=1``).
    throttle
        The array throttle (``%N``), if any.
    n_tasks
        How many tasks the request launches (1 for a plain job).
    pending_gpu_jobs
        How many jobs are pending on the GPU partition requesting a GPU, measured at
        submit time; ``None`` when unknown (refuses the exemption).
    site
        The ``autonerves.site_config.Site`` naming which partitions are ``gpu_only``.

    Returns
    -------
    The verdict, with the reasons for a refusal.
    """
    names = split_partitions(partitions)
    gpu_names = gpu_partitions_of(names, site=site)
    gpu_request = requests_gpu(gres)
    result = LintResult(
        ok=True, partitions=names, gpu_partitions=gpu_names, requests_gpu=gpu_request
    )
    if not gpu_names or gpu_request:
        return result

    listed = ",".join(names)
    if len(names) > len(gpu_names):
        result.ok = False
        result.reasons.append(
            f"partition list {listed!r} mixes GPU partition(s) {gpu_names} with CPU "
            f"partitions for a request with no GPU: CPU-only work never goes on a GPU "
            f"partition; use {[n for n in names if n not in gpu_names]} alone"
        )
        return result

    failures = []
    if cpus is None or cpus > EXEMPT_MAX_CPUS:
        failures.append(f"{cpus} CPUs per task (exemption needs ≤{EXEMPT_MAX_CPUS})")
    bounded = (throttle is not None and throttle <= EXEMPT_MAX_THROTTLE) or (
        n_tasks is not None and n_tasks <= EXEMPT_MAX_THROTTLE
    )
    if not bounded:
        failures.append(
            f"throttle {throttle if throttle is not None else 'none'} over "
            f"{n_tasks if n_tasks is not None else 'unknown'} tasks "
            f"(exemption needs %≤{EXEMPT_MAX_THROTTLE})"
        )
    if pending_gpu_jobs is None:
        failures.append("pending GPU jobs unknown (exemption needs none pending)")
    elif pending_gpu_jobs > 0:
        failures.append(f"{pending_gpu_jobs} GPU job(s) pending on {gpu_names}")

    if failures:
        result.ok = False
        result.reasons.append(
            f"partition {listed!r} is a GPU partition and the request asks for no GPU; "
            f"the timing-leg exemption does not apply: " + "; ".join(failures)
        )
        return result

    result.exempt = True
    return result


_SBATCH = re.compile(r"^\s*#SBATCH\s+(.*)$")


@dataclass
class SbatchRequest:
    partition: Optional[str] = None
    cpus: Optional[int] = None
    gres: Optional[str] = None
    throttle: Optional[int] = None
    n_tasks: Optional[int] = None


def _apply_option(request: SbatchRequest, option: str, value: Optional[str]):
    if option in ("--partition", "-p"):
        request.partition = value
    elif option in ("--cpus-per-task", "-c"):
        request.cpus = int(value)
    elif option in ("--gres", "--tres-per-node", "--tres-per-task"):
        request.gres = value if request.gres is None else f"{request.gres},{value}"
    elif option in ("--gpus", "-G", "--gpus-per-node", "--gpus-per-task"):
        gres = f"gpu:{value}"
        request.gres = gres if request.gres is None else f"{request.gres},{gres}"
    elif option in ("--array", "-a"):
        indices, throttle = parse_array_spec(value)
        request.n_tasks = len(indices)
        request.throttle = throttle


def _parse_options(tokens: List[str], request: SbatchRequest):
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--") and "=" in token:
            option, value = token.split("=", 1)
        elif token.startswith("-"):
            option = token
            value = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 1
        else:
            i += 1
            continue
        _apply_option(request, option, value)
        i += 1


def sbatch_request(script: str, arguments: Sequence[str] = ()) -> SbatchRequest:
    """
    The partition, CPUs, GPU request and array of a batch script's ``#SBATCH``
    directives, with command-line ``arguments`` (``sbatch -p gpu ...``) overriding
    them as ``sbatch`` does.
    """
    request = SbatchRequest(n_tasks=1)
    for line in script.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            # sbatch reads directives only up to the first command.
            break
        match = _SBATCH.match(line)
        if match:
            _parse_options(shlex.split(match.group(1), comments=True), request)
    _parse_options(list(arguments), request)
    return request


def check_sbatch(
    script: str,
    arguments: Sequence[str] = (),
    pending_gpu_jobs: Optional[int] = None,
    site=None,
) -> LintResult:
    """
    Lint a batch script (and any ``sbatch`` command-line overrides) against the
    GPU-partition rule; see ``check_partitions``.
    """
    request = sbatch_request(script, arguments)
    return check_partitions(
        request.partition,
        cpus=request.cpus if request.cpus is not None else 1,
        gres=request.gres,
        throttle=request.throttle,
        n_tasks=request.n_tasks,
        pending_gpu_jobs=pending_gpu_jobs,
        site=site,
    )
