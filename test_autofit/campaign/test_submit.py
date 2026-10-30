"""
Submitting a campaign end to end on the FakeScheduler: arrays per chunk, per-index
dependencies, the job ledger, and the lint.
"""
import pytest

from autofit.campaign import Campaign, PartitionLintError, read_ledger
from autofit.campaign.submit import submit
from autofit.campaign.scheduler import FakeScheduler, PartitionCapacity
from autofit.campaign.spec import CampaignException, Resources


def _capacity(pending=0):
    return {
        "ral": PartitionCapacity("ral", cpus_idle=100, cpus_total=256, pending_gpu_jobs=0),
        "gpu": PartitionCapacity("gpu", cpus_idle=10, cpus_total=124, pending_gpu_jobs=pending),
    }


def test_submit_chains_chunks(campaign_file, ral):
    campaign = Campaign.load(campaign_file)
    scheduler = FakeScheduler(
        outcomes={("vis_lp", 5): {"scheduler_state": "FAILED"}},
        capacity=_capacity(),
        site=ral,
    )
    entries = submit(campaign, scheduler)
    assert [(e["stage"], e["offset"]) for e in entries] == [
        ("vis_lp", 0), ("vis_lp", 4), ("vis_lp", 8),
        ("vis_pix", 0), ("vis_pix", 4), ("vis_pix", 8),
        ("sed", 0), ("sed", 4), ("sed", 8),
    ]
    jobs = {(e["stage"], e["offset"]): e["job_id"] for e in entries}
    assert entries[4]["after"] == {jobs[("vis_lp", 4)]: "aftercorr"}
    assert read_ledger(campaign) == entries

    scheduler.run()
    records = {(r.name, int(r.job_id), r.task_id): r.scheduler_state for job in jobs.values() for r in scheduler.query([job])}
    # Index 5 is task 1 of the offset-4 arrays: its vis_lp failed, so the dependent
    # vis_pix and sed tasks are cancelled (kill-on-invalid-dep); every other one ran.
    assert records[("demo_vis_lp", int(jobs[("vis_lp", 4)]), 1)] == "FAILED"
    assert records[("demo_vis_pix", int(jobs[("vis_pix", 4)]), 1)] == "CANCELLED"
    assert records[("demo_sed", int(jobs[("sed", 4)]), 1)] == "CANCELLED"
    assert records[("demo_vis_pix", int(jobs[("vis_pix", 4)]), 0)] == "COMPLETED"


def test_downstream_alone_uses_ledger(campaign_file, ral):
    campaign = Campaign.load(campaign_file)
    scheduler = FakeScheduler(capacity=_capacity(), site=ral)
    with pytest.raises(CampaignException, match="submit 'vis_lp' first"):
        submit(campaign, scheduler, stages=["vis_pix"])
    first = submit(campaign, scheduler, stages=["vis_lp"], indices=[1, 9])
    second = submit(campaign, scheduler, stages=["vis_pix"], indices=[1, 9])
    assert [e["after"] for e in second] == [{e["job_id"]: "aftercorr"} for e in first]
    assert len(read_ledger(campaign)) == 4


def test_override_resources_is_linted(campaign_file, ral):
    campaign = Campaign.load(campaign_file)
    scheduler = FakeScheduler(capacity=_capacity(), site=ral)
    bad = Resources.from_dict({"cpus": 4, "partition": "ral,gpu"})
    with pytest.raises(PartitionLintError):
        submit(campaign, scheduler, stages=["vis_lp"], resources={"vis_lp": bad})
    assert read_ledger(campaign) == []


def test_dry_run(campaign_file):
    campaign = Campaign.load(campaign_file)
    entries = submit(campaign, None, dry_run=True)
    assert len(entries) == 9
    assert "#SBATCH --dependency=aftercorr:<vis_lp@4>" in entries[4]["script"]
    assert entries[0]["lint"] == "ok"
    assert not campaign.directory.exists()
