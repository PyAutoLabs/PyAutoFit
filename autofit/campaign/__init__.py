"""
Campaigns: run many fits on a scheduler from one declaration (hpc-campaign phase 2).

- ``spec``: ``campaign.yaml`` (stages, manifest, chunks, resources, dependencies,
  success checks, propose-only retry policy, the autonerves site);
- ``scheduler``: the ``Scheduler`` protocol and its Fake, Local and Slurm adapters;
- ``lint``: the partition lint (CPU-only work never goes on a GPU partition);
- ``submit``: arrays per stage and chunk, chained per index, with a job ledger.

Command line: ``python -m autofit.campaign {validate,submit}``; in Python,
``autofit.campaign.submit.submit(campaign, scheduler)``.

Run states and failure classes come from ``autofit.non_linear.status.states``.
"""
from autofit.campaign.lint import (
    LintResult,
    PartitionLintError,
    check_partitions,
    check_sbatch,
)
from autofit.campaign.spec import Campaign, CampaignException, Resources, Stage
from autofit.campaign.submit import read_ledger
