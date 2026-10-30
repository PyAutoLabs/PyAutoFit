from autofit.campaign.digest import (
    MAX_MD_BYTES,
    TaskOutcome,
    build_digest,
    digest_markdown,
    percentile,
    write_digest,
)
from autofit.campaign.proposals import propose_updates
from autofit.campaign.spec import Stage, Resources


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
    assert percentile([], 50) is None
    assert percentile(list(range(101)), 95) == 95


def _outcome(i, stage="fit", state="completed", **kwargs):
    defaults = dict(job_id="1", elapsed_s=1000 + i, cpus=4, total_cpu_s=(1000 + i) * 4 * 0.8, req_mem_mb=16000,
                    max_rss_mb=4000, time_limit_s=7200, partition="ral", cpu_s_all_attempts=(1000 + i) * 4)
    defaults.update(kwargs)
    return TaskOutcome(stage=stage, key=f"1_{i}", state=state, index=i, **defaults)


def test_stage_statistics_and_slack_flag():
    outcomes = [_outcome(i) for i in range(30)] + [_outcome(99, state="pending", job_id="2")]
    digest = build_digest("c", outcomes)
    stage = digest["stages"]["fit"]
    assert stage["wall_s"] == {"p50": 1014, "p95": 1028, "max": 1029, "n": 30}
    assert stage["cpu_efficiency_p50"] == 0.8
    assert [f["flag"] for f in stage["flags"]] == ["memory_slack"]
    updates = propose_updates(outcomes, stage, Stage("fit", "x", Resources(cpus=4)))
    assert updates == [{"job_id": "2", "fields": {"mem_per_cpu_mb": 1500},
                        "command": "scontrol update JobId=2 MinMemoryCPU=1500"}]


def test_gpu_partition_misuse(ral):
    outcomes = [_outcome(i, partition="gpu") for i in range(3)] + [
        _outcome(5, partition="gpu", gres="gres/gpu=1")
    ]
    stage = build_digest("c", outcomes, site=ral)["stages"]["fit"]
    assert stage["gpu_partition_cpu_tasks"]["tasks"] == 3
    assert "cpu_on_gpu_partition" in [f["flag"] for f in stage["flags"]]


def test_markdown_capped():
    outcomes = []
    for s in range(40):
        outcomes += [_outcome(i, stage=f"a_very_long_stage_name_number_{s}", state="timeout") for i in range(3)]
    text = digest_markdown(build_digest("c" * 50, outcomes))
    assert len(text.encode()) <= MAX_MD_BYTES
    assert "more stages in digest.json" in text


def test_write_digest_replaces_stale_lists(tmp_path):
    outcomes = [_outcome(0, state="timeout")]
    write_digest(tmp_path, build_digest("c", outcomes), outcomes)
    assert (tmp_path / "failed_timeout.txt").exists()
    outcomes = [_outcome(0)]
    write_digest(tmp_path, build_digest("c", outcomes), outcomes)
    assert not (tmp_path / "failed_timeout.txt").exists()
