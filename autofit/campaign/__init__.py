"""
Campaigns: run many fits on a scheduler from one declaration (hpc-campaign phase 2).

- ``spec``: ``campaign.yaml`` (stages, manifest, chunks, resources, dependencies,
  success checks, propose-only retry policy, the autonerves site);
- ``scheduler``: the ``Scheduler`` protocol and its Fake, Local and Slurm adapters;
- ``lint``: the partition lint (CPU-only work never goes on a GPU partition);
- ``submit``: arrays per stage and chunk, chained per index, with a job ledger;
- ``collector``: accounting joined with status records → ``digest``
  (``digest.json``, a ≤2 KB ``digest.md``, ``tasks.jsonl``, ``failed_<class>.txt``);
- ``proposals``: resubmit and pending-update proposals (never acted on);
- ``retro``: the digest of a finished campaign from a ``sacct`` dump alone;
- ``evidence``: profiling / inference summaries, read and graded per stage;
- ``plan``: proposed resources, wall time, ETA and disk, every number graded;
- ``pilot``: a stratified subset of the manifest to measure first.

Command line: ``python -m autofit.campaign {validate,submit,status,resubmit,retro,plan,pilot}``;
in Python, ``autofit.campaign.submit.submit(campaign, scheduler)`` and
``autofit.campaign.retro.retro(dump, ...)``.

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
from autofit.campaign.collector import collect
from autofit.campaign.digest import TaskOutcome, build_digest, digest_markdown
from autofit.campaign.proposals import propose_resubmit
