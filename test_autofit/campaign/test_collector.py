"""
The collector end to end on the FakeScheduler: injected timeout, OOM, exception and
missing-seed tasks come out in the right classes, lists and proposals.
"""
import json
import zipfile

import pytest

from autofit.campaign import Campaign, collect, propose_resubmit
from autofit.campaign.__main__ import main
from autofit.campaign.digest import MAX_MD_BYTES, read_outcomes
from autofit.campaign.scheduler import FakeScheduler, PartitionCapacity
from autofit.campaign.submit import submit

LIMIT = 4 * 3600


def _outcomes(stage, index, attempt):
    if stage != "vis_lp":
        return None
    if index == 1:
        return {"scheduler_state": "TIMEOUT", "elapsed_s": LIMIT, "max_rss_mb": 5000}
    if index == 2:
        return {"scheduler_state": "OUT_OF_MEMORY", "elapsed_s": 600, "max_rss_mb": 12288}
    if index in (3, 5):
        return {"scheduler_state": "FAILED", "elapsed_s": 60}
    return {"elapsed_s": 3000 + 100 * index, "max_rss_mb": 6000 + 10 * index}


def _status(root, job_id, task_id, state, failure_class=None):
    path = root / "output" / f"{job_id}_{task_id}.status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "state": state,
                "failure_class": failure_class,
                "attempt": 1,
                "job": {"job_id": job_id, "task_id": task_id},
                "outputs": {"bytes": 1000},
                "updated": "2026-10-30T00:00:00Z",
            }
        )
    )


def _zip(root, tile, members=("files/samples_summary.json", ".completed")):
    path = root / "output" / "dr1" / tile / "vis_lp.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as f:
        for member in members:
            f.writestr(member, "{}")


@pytest.fixture(name="collected")
def make_collected(campaign_file, ral):
    root = campaign_file.parent
    campaign = Campaign.load(campaign_file)
    capacity = {"ral": PartitionCapacity("ral", cpus_idle=52, cpus_total=252, pending_gpu_jobs=0)}
    scheduler = FakeScheduler(outcomes=_outcomes, capacity=capacity, site=ral)
    entries = submit(campaign, scheduler, stages=["vis_lp"])
    scheduler.run()
    jobs = {entry["offset"]: entry["job_id"] for entry in entries}
    _status(root, jobs[0], 3, "failed", "exception")
    _status(root, jobs[4], 1, "failed", "missing_seed")
    for index in (0, 4, 6, 7, 8, 9):
        _zip(root, f"Tile{index:03d}")
    # Index 0 lacks a required member: completed by the scheduler, not by the check.
    _zip(root, "Tile000", members=(".completed",))
    # vis_pix waits on vis_lp and is not run yet: pending tasks for update proposals.
    submit(campaign, scheduler, stages=["vis_pix"])
    return campaign, scheduler, collect(campaign, scheduler)


def test_states_and_classes(collected):
    campaign, _, digest = collected
    vis_lp = digest["stages"]["vis_lp"]
    assert vis_lp["states"] == {"completed": 6, "failed": 2, "oom": 1, "timeout": 1}
    assert vis_lp["failure_classes"] == {"exception": 1, "missing_seed": 1}
    assert vis_lp["success_check_failed"] == 1
    assert digest["stages"]["vis_pix"]["states"] == {"pending": 10}
    assert digest["capacity"]["ral"]["cpus_idle"] == 52
    assert vis_lp["wall_s"]["max"] == 3900 and vis_lp["wall_s"]["n"] == 6
    assert vis_lp["cpu_h"]["all_attempts"] > vis_lp["cpu_h"]["successful"]


def test_files(collected):
    campaign, _, digest = collected
    directory = campaign.directory
    lists = {p.name: p.read_text().split("\n")[0] for p in directory.glob("failed_*.txt")}
    assert set(lists) == {
        "failed_timeout.txt",
        "failed_oom.txt",
        "failed_exception.txt",
        "failed_missing_seed.txt",
        "failed_success_check.txt",
    }
    assert lists["failed_timeout.txt"].startswith("vis_lp\t1\t")
    assert lists["failed_missing_seed.txt"].startswith("vis_lp\t5\t")
    md = (directory / "digest.md").read_text()
    assert len(md.encode()) <= MAX_MD_BYTES
    assert "| vis_lp | 6/10 | 4 |" in md
    assert json.loads((directory / "digest.json").read_text())["campaign"] == "demo"
    assert len(read_outcomes(directory / "tasks.jsonl")) == 20


def test_proposals(collected, ral):
    campaign, _, _ = collected
    outcomes = read_outcomes(campaign.directory / "tasks.jsonl")
    stage = campaign.stages["vis_lp"]

    timeout = propose_resubmit(stage, "timeout", outcomes, site=ral)
    assert timeout["indices"] == "1"
    assert timeout["changes"]["time"] == ["04:00:00", "05:00:00"]

    oom = propose_resubmit(stage, "oom", outcomes, site=ral)
    assert oom["indices"] == "2"
    assert oom["changes"]["mem_mb"] == [12288, 18432]
    assert oom["resources"]["mem_per_cpu_mb"] == 4608

    exception = propose_resubmit(stage, "exception", outcomes)
    assert exception["indices"] is None and "fix the cause" in exception["reason"]
    seed = propose_resubmit(stage, "missing_seed", outcomes)
    assert seed["indices"] is None and "upstream" in seed["reason"]


def test_retry_max_excludes(collected):
    campaign, _, _ = collected
    outcomes = read_outcomes(campaign.directory / "tasks.jsonl")
    for outcome in outcomes:
        if outcome.state == "timeout":
            outcome.attempts = 2
    proposal = propose_resubmit(campaign.stages["vis_lp"], "timeout", outcomes)
    assert proposal["indices"] is None and proposal["excluded"][0]["reason"] == "2 attempts, retry max 1"


def test_resubmit_round_trip(collected, monkeypatch, capsys, ral):
    campaign, scheduler, _ = collected
    import autofit.campaign.scheduler as schedulers

    monkeypatch.setattr(schedulers, "scheduler_for", lambda campaign, runner=None: scheduler)
    path = str(campaign.source)
    assert main(["resubmit", path, "--stage", "vis_lp", "--class", "timeout"]) == 0
    out = capsys.readouterr().out
    assert "Proposal only" in out
    # Nothing was submitted by the proposal.
    jobs_before = len(scheduler.requests)
    assert main(["submit", path, "--proposal", str(campaign.directory / "resubmit_vis_lp_timeout.json")]) == 0
    assert len(scheduler.requests) == jobs_before + 1
    request = list(scheduler.requests.values())[-1]
    assert request.indices == [1] and request.resources.time_s == 5 * 3600

    scheduler.run()
    assert main(["status", path]) == 0
    digest = json.loads((campaign.directory / "digest.json").read_text())
    vis_lp = digest["stages"]["vis_lp"]
    # The resubmitted task 1 timed out again: one task, two attempts.
    assert vis_lp["tasks"] == 10 and vis_lp["attempts"] == 11
    assert not (campaign.directory / "failed_none.txt").exists()


def test_update_pending_proposals(collected):
    campaign, _, digest = collected
    assert "update_pending" not in digest["stages"]["vis_pix"]
    vis_lp = digest["stages"]["vis_lp"]
    assert {flag["flag"] for flag in vis_lp["flags"]} >= {"timeouts", "success_check"}
