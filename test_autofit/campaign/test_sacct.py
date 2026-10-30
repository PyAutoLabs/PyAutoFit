from autofit.campaign.scheduler import RunState, run_state_of
from autofit.campaign.scheduler.sacct import latest_per_task, parse

HEADER = "JobID|JobIDRaw|JobName|Partition|State|ExitCode|Submit|Start|End|Elapsed|Timelimit|AllocCPUS|ReqMem|MaxRSS|AveRSS|TotalCPU|CPUTime|NNodes|NodeList|ReqTRES|AllocTRES|WorkDir"

DUMP = "\n".join(
    [
        HEADER,
        "342027_1|342029|vis_lp|ral|COMPLETED|0:0|2026-08-30T02:54:31|2026-08-30T02:54:31|2026-09-01T00:04:55|1-21:10:24|2-00:00:00|8|98304M|||12-01:02:31|15-01:23:12|1|n21|billing=8,cpu=8,mem=98304M,node=1|billing=8,cpu=8,mem=98304M,node=1|/w/euclid_dr1/hpc",
        "342027_1.batch|342029.batch|batch||COMPLETED|0:0|2026-08-30T02:54:31|2026-08-30T02:54:31|2026-09-01T00:04:55|1-21:10:24||8||29823.80M|29823.80M|12-01:02:31|15-01:23:12|1|n21||cpu=8,mem=98304M,node=1|",
        "342027_1.extern|342029.extern|extern||COMPLETED|0:0|2026-08-30T02:54:31|2026-08-30T02:54:31|2026-09-01T00:04:55|1-21:10:24||8||0.50M|0.50M|00:00.001|15-01:23:12|1|n21||cpu=8,mem=98304M,node=1|",
        "342027_2|342030|vis_lp|ral|REQUEUED|0:0|2026-08-30T02:54:31|2026-08-30T02:55:00|2026-08-30T03:00:00|00:05:00|2-00:00:00|8|98304M|||00:30:00|00:40:00|1|n3|billing=8|billing=8|/w/euclid_dr1/hpc",
        "342027_2.batch|342030.batch|batch||CANCELLED|0:15|2026-08-30T02:54:31|2026-08-30T02:55:00|2026-08-30T03:00:00|00:05:00||8||100M|100M|00:30:00|00:40:00|1|n3||cpu=8|",
        "342027_2|342030|vis_lp|ral|TIMEOUT|0:0|2026-08-30T03:01:00|2026-08-30T03:02:00|2026-09-01T03:02:00|2-00:00:00|2-00:00:00|8|98304M|||15-00:00:00|16-00:00:00|1|n4|billing=8|billing=8|/w/euclid_dr1/hpc",
        "342027_2.batch|342030.batch|batch||CANCELLED|0:15|2026-08-30T03:01:00|2026-08-30T03:02:00|2026-09-01T03:02:00|2-00:00:00||8||50000M|50000M|15-00:00:00|16-00:00:00|1|n4||cpu=8|",
        "350000|350000|sed|gpu|OUT_OF_MEMORY|0:125|2026-09-30T10:00:00|2026-09-30T10:00:01|2026-09-30T10:10:01|00:10:00|08:00:00|4|3072Mc|||00:30:00|00:40:00|1|g1|billing=4,cpu=4,gres/gpu=1|billing=4,cpu=4|/w/euclid_dr1/hpc",
        "350000.batch|350000.batch|batch||OUT_OF_MEMORY|0:125|2026-09-30T10:00:00|2026-09-30T10:00:01|2026-09-30T10:10:01|00:10:00||4||12500M|12500M|00:30:00|00:40:00|1|g1||cpu=4|",
        "344645_[586,588-594,...]|344645|vis_pix|ral|PENDING|0:0|2026-09-02T17:39:47|Unknown|Unknown|00:00:00|2-00:00:00|8|98304M|||00:00:00|00:00:00|1|None assigned|billing=8|billing=8|/w/euclid_dr1/hpc",
    ]
)


def test_parse_folds_steps():
    records = parse(DUMP)
    assert len(records) == 5
    first = records[0]
    assert (first.job_id, first.task_id, first.name) == ("342027", 1, "vis_lp")
    assert first.max_rss_mb == 29823.8
    assert first.elapsed_s == 162624 and first.time_limit_s == 172800
    assert first.total_cpu_s == 12 * 86400 + 3751
    assert first.cpu_efficiency == first.total_cpu_s / (162624 * 8)
    assert first.state == RunState.COMPLETED


def test_requeue_attempts_and_latest():
    tasks = {record.key: record for record in latest_per_task(parse(DUMP))}
    task = tasks["342027_2"]
    assert task.attempts == 2
    assert task.state == RunState.TIMEOUT
    assert task.max_rss_mb == 50000


def test_per_cpu_memory_and_gpu_tres():
    record = parse(DUMP)[3]
    assert record.req_mem_mb == 3072 * 4
    assert record.state == RunState.OOM
    assert record.gres == "gres/gpu=1"
    assert record.task_id is None


def test_pending_placeholder():
    record = parse(DUMP)[-1]
    assert record.job_id.startswith("344645_[") and record.state == RunState.PENDING
    assert record.start is None


def test_noheader_with_fields():
    fields = ["JobID", "JobName", "State", "Elapsed", "AllocCPUS"]
    records = parse("7_0|fit|COMPLETED|00:10:00|2\n7_0.batch|batch|COMPLETED|00:10:00|2\n", fields=fields)
    assert len(records) == 1 and records[0].cpu_hours == 20 / 60


def test_state_mapping():
    assert run_state_of("CANCELLED by 16309") == "cancelled"
    assert run_state_of("NODE_FAIL") == RunState.NODE_FAIL
    assert run_state_of("PD") == RunState.PENDING
    assert run_state_of("COMPLETED+") == RunState.COMPLETED
    assert run_state_of("WEIRD") == "unknown"
