"""
The GPU-partition rule (human, 2026-09-30): every branch.
"""
import pytest

from autofit.campaign.lint import PartitionLintError, check_partitions, check_sbatch


def _check(partitions, ral, **kwargs):
    defaults = dict(cpus=4, gres=None, throttle=None, n_tasks=1000, pending_gpu_jobs=0)
    defaults.update(kwargs)
    return check_partitions(partitions, site=ral, **defaults)


def test_cpu_partition_is_fine(ral):
    result = _check("ral", ral)
    assert result.ok and not result.exempt and result.gpu_partitions == []


@pytest.mark.parametrize("partitions", ["ral,gpu", "gpu,ral", ["ral", "gpu"], " gpu , ral "])
def test_mixed_list_always_refused(partitions, ral):
    # Even a request that would meet the exemption: the incident form is never allowed.
    result = _check(partitions, ral, cpus=1, throttle=1, n_tasks=1, pending_gpu_jobs=0)
    assert not result.ok
    assert "mixes GPU partition" in result.reasons[0]
    with pytest.raises(PartitionLintError):
        result.raise_for_refusal()


def test_bulk_cpu_array_on_gpu_refused(ral):
    result = _check("gpu", ral, cpus=8, throttle=400)
    assert not result.ok
    assert "throttle 400" in result.reasons[0]


def test_timing_leg_exemption(ral):
    result = _check("gpu", ral, cpus=8, throttle=2, pending_gpu_jobs=0)
    assert result.ok and result.exempt


def test_exemption_small_plain_job(ral):
    assert _check("gpu", ral, cpus=2, throttle=None, n_tasks=1).exempt


def test_exemption_too_many_cpus(ral):
    result = _check("gpu", ral, cpus=9, throttle=2)
    assert not result.ok and "9 CPUs per task" in result.reasons[0]


def test_exemption_gpu_jobs_pending(ral):
    result = _check("gpu", ral, cpus=8, throttle=2, pending_gpu_jobs=3)
    assert not result.ok and "3 GPU job(s) pending" in result.reasons[0]


def test_exemption_fails_closed_when_pending_unknown(ral):
    result = _check("gpu", ral, cpus=8, throttle=2, pending_gpu_jobs=None)
    assert not result.ok and "unknown" in result.reasons[0]


def test_gpu_request_may_use_gpu(ral):
    result = _check("gpu", ral, cpus=32, gres="gpu:1", throttle=None, pending_gpu_jobs=5)
    assert result.ok and result.requests_gpu and not result.exempt


def test_gpu_request_on_mixed_list_is_not_linted(ral):
    assert _check("ral,gpu", ral, gres="gpu:a100:1").ok


def test_without_site_names_containing_gpu_are_gpu_partitions():
    result = check_partitions("cpu,gpu_a100", cpus=4, n_tasks=10, pending_gpu_jobs=0)
    assert not result.ok and result.gpu_partitions == ["gpu_a100"]


def test_site_decides_over_name(ral):
    # `ral` is gpu_only: false in the site file; `gpu` is gpu_only: true.
    assert _check("ral", ral).gpu_partitions == []
    assert _check("gpu", ral, throttle=2).gpu_partitions == ["gpu"]


SCRIPT = """#!/bin/bash
#SBATCH --job-name=euclid_dr1_vis_lp
#SBATCH --partition=ral
#SBATCH --array=0-999%400
#SBATCH --cpus-per-task=4
python fit.py
"""


def test_sbatch_script_lint(ral):
    assert check_sbatch(SCRIPT, site=ral, pending_gpu_jobs=0).ok


@pytest.mark.parametrize("arguments", [["-p", "gpu"], ["--partition=ral,gpu"], ["--partition", "gpu,ral"]])
def test_sbatch_command_line_override_is_linted(arguments, ral):
    assert not check_sbatch(SCRIPT, arguments, site=ral, pending_gpu_jobs=0).ok


def test_sbatch_directive_mixed_list_refused(ral):
    script = SCRIPT.replace("--partition=ral", "--partition=ral,gpu")
    assert not check_sbatch(script, site=ral, pending_gpu_jobs=0).ok


def test_sbatch_timing_leg(ral):
    script = SCRIPT.replace("--partition=ral", "-p gpu").replace("0-999%400", "0-5%2").replace(
        "--cpus-per-task=4", "-c 8"
    )
    result = check_sbatch(script, site=ral, pending_gpu_jobs=0)
    assert result.ok and result.exempt


def test_sbatch_gpu_job(ral):
    script = SCRIPT.replace("--partition=ral", "--partition=gpu\n#SBATCH --gpus=1")
    result = check_sbatch(script, site=ral, pending_gpu_jobs=7)
    assert result.ok and result.requests_gpu


def test_sbatch_directive_after_command_is_ignored(ral):
    # sbatch ignores directives after the first command, so a late --gpus does not
    # make a CPU job a GPU job.
    script = SCRIPT.replace("--partition=ral", "--partition=gpu") + "#SBATCH --gpus=1\n"
    assert not check_sbatch(script, site=ral, pending_gpu_jobs=0).ok
