"""
Scheduler adapters for ``autofit.campaign``: the ``Scheduler`` protocol, an in-memory
``FakeScheduler`` for tests, a subprocess ``LocalScheduler`` and the ``SlurmScheduler``.
None of them runs a command at import.
"""
from autofit.non_linear.status.states import RunState
from autofit.campaign.scheduler.base import (
    CANCELLED,
    PREEMPTED,
    REQUEUED,
    UNKNOWN,
    PartitionCapacity,
    Scheduler,
    SubmitRequest,
    TaskRecord,
    run_state_of,
)
from autofit.campaign.scheduler.fake import FakeScheduler
from autofit.campaign.scheduler.local import LocalScheduler
from autofit.campaign.scheduler.slurm import SlurmScheduler, sbatch_script


def scheduler_for(campaign, runner=None):
    """
    The scheduler a campaign's ``scheduler:`` names (``slurm`` or ``local``).
    """
    if campaign.scheduler == "local":
        return LocalScheduler()
    return SlurmScheduler(runner=runner, site=campaign.site if campaign.site_name else None)
