"""
The Slurm adapter over an injected runner: no command ever reaches a cluster.
"""
import subprocess
import sys

import pytest

from autofit.campaign import PartitionLintError
from autofit.campaign.scheduler import SlurmScheduler, SubmitRequest, sbatch_script
from autofit.campaign.spec import Resources


class Runner:
    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = outputs or {}

    def __call__(self, arguments, input=None):
        self.calls.append((list(arguments), input))
        return self.outputs.get(arguments[0], "")


SINFO = "\n".join(
    [
        "ral*|n1|mix|200/52/0/252|920000|300000",
        "ral*|n2|idle|0/252/0/252|920000|900000",
        "ral*|n3|down*|0/252/0/252|920000|0",
        "gpu|g1|mix|60/2/0/62|500000|1000",
        "gpu|g2|mix|62/0/0/62|500000|1000",
    ]
)
SQUEUE_PENDING = "gpu|gres/gpu:1|413161_[0-5%16]\nral|N/A|413200_[0-99]\ngpu|N/A|413300\n"


def _request(**resources):
    defaults = dict(cpus=4, mem_per_cpu_mb=3072, time="04:00:00", partition="ral", throttle=400, nice=1000)
    defaults.update(resources)
    return SubmitRequest(
        campaign="dr1",
        stage="vis_lp",
        script="scripts/fit.py --stage=vis_lp",
        offset=1000,
        indices=list(range(1000, 1010)),
        resources=Resources.from_dict(defaults),
        project_root="/mnt/ral/x/euclid_dr1",
        manifest="/mnt/ral/x/euclid_dr1/m.txt",
        env={"OMP_NUM_THREADS": "$SLURM_CPUS_PER_TASK", "NAME": "a b"},
        after={"999": "aftercorr"},
        log_dir="hpc/logs",
    )


def test_sbatch_script():
    script = sbatch_script(_request())
    for line in [
        "#SBATCH --job-name=dr1_vis_lp",
        "#SBATCH --partition=ral",
        "#SBATCH --array=0-9%400",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem-per-cpu=3072",
        "#SBATCH --time=04:00:00",
        "#SBATCH --nice=1000",
        "#SBATCH --dependency=aftercorr:999",
        "#SBATCH --kill-on-invalid-dep=yes",
        "#SBATCH --signal=B:USR1@600",
        "#SBATCH --output=hpc/logs/%x_%A_%a.out",
        "set -euo pipefail",
        "export PYTHONUNBUFFERED=1",
        "export CAMPAIGN_OFFSET=1000",
        'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        "export NAME='a b'",
        "export CAMPAIGN_INDEX=$((CAMPAIGN_OFFSET + SLURM_ARRAY_TASK_ID))",
        "python scripts/fit.py --stage=vis_lp &",
        "trap 'kill -USR1 \"$child\" 2>/dev/null || true' USR1",
    ]:
        assert line in script.splitlines(), line
    assert "--gres" not in script


def test_sbatch_script_passes_its_own_lint(ral):
    from autofit.campaign.lint import check_sbatch

    assert check_sbatch(sbatch_script(_request()), site=ral, pending_gpu_jobs=0).ok
    assert not check_sbatch(sbatch_script(_request(partition="ral,gpu")), site=ral, pending_gpu_jobs=0).ok


def test_capacity():
    scheduler = SlurmScheduler(runner=Runner({"sinfo": SINFO, "squeue": SQUEUE_PENDING}))
    capacity = scheduler.capacity()
    ral, gpu = capacity["ral"], capacity["gpu"]
    assert (ral.cpus_idle, ral.cpus_allocated, ral.cpus_total, ral.nodes, ral.nodes_unreachable) == (304, 200, 756, 3, 1)
    assert ral.cpus_other == 252
    assert (gpu.cpus_idle, gpu.pending_gpu_jobs) == (2, 1)
    assert ral.pending_gpu_jobs == 0


def test_submit_lints_then_sbatches(ral):
    runner = Runner({"sinfo": SINFO, "squeue": SQUEUE_PENDING, "sbatch": "413500;cluster\n"})
    scheduler = SlurmScheduler(runner=runner, site=ral)
    assert scheduler.submit(_request()) == "413500"
    arguments, script = runner.calls[-1]
    assert arguments == ["sbatch", "--parsable"]
    assert "#SBATCH --array=0-9%400" in script


@pytest.mark.parametrize("partition", ["ral,gpu", "gpu,ral", "gpu"])
def test_submit_refuses_cpu_on_gpu(partition, ral):
    runner = Runner({"sinfo": SINFO, "squeue": SQUEUE_PENDING, "sbatch": "1"})
    with pytest.raises(PartitionLintError):
        SlurmScheduler(runner=runner, site=ral).submit(_request(partition=partition))
    assert not any(call[0][0] == "sbatch" for call in runner.calls)


def test_timing_leg_refused_while_gpu_jobs_pend(ral):
    runner = Runner({"sinfo": SINFO, "squeue": SQUEUE_PENDING, "sbatch": "1"})
    request = _request(partition="gpu", cpus=8, throttle=2)
    with pytest.raises(PartitionLintError, match="1 GPU job"):
        SlurmScheduler(runner=runner, site=ral).submit(request)
    runner.outputs["squeue"] = "ral|N/A|1\n"
    assert SlurmScheduler(runner=runner, site=ral).submit(request) == "1"


def test_query_is_one_sacct_call():
    runner = Runner({"sacct": "5_0|5|fit|ral|COMPLETED|0:0|||||01:00:00|2|4096M||||||||||\n"})
    records = SlurmScheduler(runner=runner).query(["5", "6"])
    (arguments, _), = runner.calls
    assert arguments[:3] == ["sacct", "-j", "5,6"]
    assert "--parsable2" in arguments and "--units=M" in arguments and "--duplicates" in arguments
    assert records[0].key == "5_0" and records[0].cpus == 2


def test_update_pending(ral):
    runner = Runner({"sinfo": SINFO, "squeue": SQUEUE_PENDING})
    scheduler = SlurmScheduler(runner=runner, site=ral)
    scheduler.update_pending("413500", time=5 * 3600, mem_per_cpu_mb=4096.4, throttle=200)
    assert runner.calls[-1][0] == [
        "scontrol", "update", "JobId=413500", "TimeLimit=05:00:00", "MinMemoryCPU=4096", "ArrayTaskThrottle=200"
    ]
    with pytest.raises(PartitionLintError):
        scheduler.update_pending("413500", partition="ral,gpu", cpus=4)
    with pytest.raises(ValueError):
        scheduler.update_pending("413500", gpus=4)


def test_cancel():
    runner = Runner()
    SlurmScheduler(runner=runner).cancel("413500")
    assert runner.calls == [(["scancel", "413500"], None)]


def test_no_command_at_import():
    # Import autofit first (it is not ours to police), then forbid any command while
    # the campaign package and its adapters import.
    code = (
        "import subprocess, autofit\n"
        "def refuse(*a, **k): raise SystemExit('a command ran at import')\n"
        "subprocess.run = subprocess.Popen = subprocess.check_output = refuse\n"
        "import autofit.campaign, autofit.campaign.scheduler, autofit.campaign.__main__\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2000:]
