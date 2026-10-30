"""
Submitting a campaign: one array per stage per manifest chunk, chained chunk by chunk,
recorded in the campaign's job ledger ``<project_root>/campaign/<name>/jobs.json``.

The ledger is how the collector finds the campaign's jobs later; it is written
atomically after every successful submit, so an interrupted submit leaves a ledger that
names exactly the arrays that exist.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from autofit.campaign.lint import check_partitions
from autofit.campaign.scheduler.base import SubmitRequest
from autofit.campaign.spec import Campaign, CampaignException
from autofit.campaign.units import array_spec, parse_array_spec
from autofit.tools.util import open_atomic

logger = logging.getLogger(__name__)

LEDGER = "jobs.json"


def ledger_path(campaign: Campaign) -> Path:
    return campaign.directory / LEDGER


def read_ledger(campaign: Campaign) -> List[Dict]:
    """
    The campaign's submitted arrays: ``{stage, job_id, offset, indices, submitted,
    resources}`` per array, oldest first (empty before the first submit).
    """
    path = ledger_path(campaign)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _write_ledger(campaign: Campaign, entries: List[Dict]):
    path = ledger_path(campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_atomic(path, "w") as f:
        json.dump(entries, f, indent=1)


def ledger_indices(entry: Dict) -> List[int]:
    return parse_array_spec(entry["indices"])[0]


def requests_for(
    campaign: Campaign,
    stages: Optional[Sequence[str]] = None,
    indices: Optional[Sequence[int]] = None,
    resources: Optional[Dict[str, object]] = None,
) -> List[SubmitRequest]:
    """
    The arrays a submit of ``stages`` (default: all, in dependency order) over
    ``indices`` (default: the whole manifest) would create, before dependencies are
    resolved.
    """
    order = campaign.order()
    names = [name for name in order if stages is None or name in stages]
    unknown = set(stages or ()) - set(order)
    if unknown:
        raise CampaignException(f"Unknown stages {sorted(unknown)}.")
    requests = []
    for name in names:
        stage = campaign.stages[name]
        for offset, chunk in campaign.chunks(indices):
            requests.append(
                SubmitRequest(
                    campaign=str(campaign.name),
                    stage=name,
                    script=stage.script,
                    offset=offset,
                    indices=chunk,
                    resources=(resources or {}).get(name, stage.resources),
                    project_root=str(campaign.project_root),
                    manifest=str(campaign.path(campaign.manifest_path)),
                    env=dict(stage.env),
                    log_dir=campaign.log_dir,
                )
            )
    return requests


def submit(
    campaign: Campaign,
    scheduler,
    stages: Optional[Sequence[str]] = None,
    indices: Optional[Sequence[int]] = None,
    resources: Optional[Dict[str, object]] = None,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Submit ``stages`` over ``indices``, chaining each array to the array of its
    upstream stages that covers the same chunk (submitted now, or earlier and found
    in the ledger).

    Parameters
    ----------
    resources
        Per-stage ``Resources`` overriding the campaign's (an approved resubmit).
    dry_run
        Return the requests (with ``"script"`` holding the batch script for a Slurm
        scheduler) without submitting or writing the ledger.

    Returns
    -------
    The ledger entries of the arrays submitted (or that would be).
    """
    ledger = [] if dry_run else read_ledger(campaign)
    submitted: List[Dict] = []
    # (stage, offset) -> job id, newest wins: an upstream resubmit supersedes the
    # original array for the dependency of a later downstream submit.
    jobs = {(entry["stage"], entry["offset"]): entry["job_id"] for entry in ledger}

    for request in requests_for(campaign, stages, indices, resources):
        stage = campaign.stages[request.stage]
        for upstream, kind in stage.after.items():
            job_id = jobs.get((upstream, request.offset))
            if job_id is None:
                if dry_run:
                    job_id = f"<{upstream}@{request.offset}>"
                else:
                    raise CampaignException(
                        f"Stage {request.stage!r} chunk {request.offset} waits on "
                        f"{upstream!r}, which has no array for that chunk; submit "
                        f"{upstream!r} first."
                    )
            request.after[str(job_id)] = kind

        entry = {
            "stage": request.stage,
            "offset": request.offset,
            "indices": array_spec(request.indices),
            "after": dict(request.after),
            "resources": {
                key: value
                for key, value in vars(request.resources).items()
                if value is not None
            },
        }
        if dry_run:
            from autofit.campaign.scheduler.slurm import SlurmScheduler, sbatch_script

            if isinstance(scheduler, SlurmScheduler) or scheduler is None:
                entry["script"] = sbatch_script(request)
            # A dry run never queries the cluster, so the pending-GPU count is
            # unknown: only the lint's static half can pass here.
            resources = request.resources
            verdict = check_partitions(
                resources.partition,
                cpus=resources.cpus,
                gres=resources.gres,
                throttle=resources.throttle,
                n_tasks=len(request.indices),
                pending_gpu_jobs=None,
                site=campaign.site if campaign.site_name else None,
            )
            entry["lint"] = "ok" if verdict.ok else "; ".join(verdict.reasons)
            entry["job_id"] = f"<{request.stage}@{request.offset}>"
        else:
            entry["job_id"] = scheduler.submit(request)
            entry["submitted"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ledger.append(entry)
            _write_ledger(campaign, ledger)
            logger.info(
                f"Submitted {request.stage} chunk {request.offset} "
                f"({len(request.indices)} tasks) as {entry['job_id']}."
            )
        jobs[(request.stage, request.offset)] = entry["job_id"]
        submitted.append(entry)
    return submitted
