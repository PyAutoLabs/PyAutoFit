"""
The Green Algorithms estimate: the worked DR1 example of the hpc-campaign report
(§1.4), the four cases, failed or repeated work, missing inputs, GPUs and the NESO
helpers (with the network mocked).
"""
from datetime import datetime, timedelta, timezone

import pytest

from autofit.campaign.carbon import (
    Bounds,
    CarbonInputs,
    StageForecast,
    apply_neso,
    begin_hint,
    estimate,
    forecast,
    gpu_count,
    neso_intensity,
    power_w,
)
from autofit.campaign.digest import TaskOutcome

# Report §1.4 / survey carbon.md §6: low 2.8 W, PUE 1.20, 100 g; central 3.5 W, 1.31,
# 130 g; high 4.4 W, 1.45, 190 g.
REPORT_INPUTS = dict(pue=(1.20, 1.31, 1.45), tdp_w_per_cpu=(2.8, 3.5, 4.4), ci=(100, 130, 190))

DR1_FORECAST = [
    StageForecast("vis_pix", 4600, (9.1, 9.1, 9.1), 8, 0.83, (11, 11, 64)),
    StageForecast("vis_lp", 5000, (1.0, 1.5, 2.0), 8, 0.53, (4, 12, 64)),
    StageForecast("sed", 5000, (11 / 60,) * 3, 8, 0.72, (4, 8, 16)),
]


def _outcome(state="completed", elapsed_s=3600, cpus=8, total_cpu_s=None, **kwargs):
    defaults = dict(
        stage="vis_lp",
        key="1_0",
        state=state,
        elapsed_s=elapsed_s,
        cpus=cpus,
        req_mem_mb=12288.0,
        max_rss_mb=4096.0,
        total_cpu_s=total_cpu_s if total_cpu_s is not None else 0.5 * elapsed_s * cpus,
        partition="ral",
        cpu_s_all_attempts=elapsed_s * cpus,
    )
    defaults.update(kwargs)
    return TaskOutcome(**defaults)


def test_power_is_the_green_algorithms_formula():
    # Report §1.4: vis_pix central draw 8 × 3.5 × 0.83 + 11 × 0.3725 = 27.3 W.
    assert power_w(8, 3.5, 0.83, 11) == pytest.approx(27.34, abs=0.01)
    assert power_w(4, 3.5, 1.0, 0, gpus=1, gpu_w=300) == pytest.approx(314.0)


def test_report_worked_dr1_estimate_is_reproduced():
    result = forecast(DR1_FORECAST, CarbonInputs.from_site(None, **REPORT_INPUTS))

    stages = result["stages"]
    assert stages["vis_pix"]["watts_central"] == pytest.approx(27.3, abs=0.05)
    assert stages["vis_lp"]["watts_central"] == pytest.approx(19.3, abs=0.05)
    assert stages["sed"]["watts_central"] == pytest.approx(23.1, abs=0.05)

    # kWh low / central / high per stage and in total, as in the report's table.
    expected = {
        "vis_pix": (1140, 1499, 3584),
        "vis_lp": (80, 190, 856),
        "sed": (19, 28, 55),
    }
    for name, (low, central, high) in expected.items():
        kwh = stages[name]["kwh"]
        assert (kwh["low"], kwh["central"], kwh["high"]) == pytest.approx(
            (low, central, high), abs=1
        )
    total = result["total"]
    assert (total["kwh"]["low"], total["kwh"]["central"], total["kwh"]["high"]) == pytest.approx(
        (1239, 1717, 4494), abs=1
    )
    # ≈0.22 tCO2e, range 0.12–0.85 t.
    kg = total["kg_co2e"]
    assert (kg["low"], kg["central"], kg["high"]) == pytest.approx((124, 223, 854), abs=1)
    # The allocation basis of vis_lp (u = 1) is the survey's 319 kWh.
    assert stages["vis_lp"]["kwh"]["allocation"] == pytest.approx(319, abs=1)


def test_cases_order_and_bases():
    inputs = CarbonInputs.from_site(None, **REPORT_INPUTS)
    result = estimate([_outcome()], inputs)
    kwh = result["total"]["kwh"]

    assert kwh["low"] < kwh["central"] < kwh["allocation"] < kwh["high"]
    # Usage basis: 1 h × (8 × 3.5 × 0.5 + 12 × 0.3725) × 1.31.
    assert kwh["central"] == pytest.approx((14 + 4.47) * 1.31 / 1000, abs=1e-4)
    # Allocation basis charges every reserved CPU at full use.
    assert kwh["allocation"] == pytest.approx((28 + 4.47) * 1.31 / 1000, abs=1e-4)
    # Low uses peak memory (4 GB).
    assert kwh["low"] == pytest.approx((8 * 2.8 * 0.5 + 4 * 0.3725) * 1.20 / 1000, abs=1e-4)
    assert result["total"]["kg_co2e"]["central"] == pytest.approx(
        kwh["central"] * 0.130, abs=1e-4
    )


def test_failed_and_repeated_work_is_split_out():
    inputs = CarbonInputs.from_site(None, **REPORT_INPUTS)
    once = _outcome(key="a")
    twice = _outcome(key="b", attempts=2, cpu_s_all_attempts=2 * 3600 * 8)
    timeout = _outcome(key="c", state="timeout")
    failed_check = _outcome(key="d", success=False)

    result = estimate([once, twice, timeout, failed_check], inputs)
    total = result["total"]
    per_attempt = estimate([once], inputs)["total"]["kwh"]["central"]

    assert total["kwh"]["central"] == pytest.approx(5 * per_attempt, abs=1e-3)
    assert total["successful_kwh"]["central"] == pytest.approx(2 * per_attempt, abs=1e-3)
    assert total["failed_or_repeated_kwh"]["central"] == pytest.approx(3 * per_attempt, abs=1e-3)
    assert total["failed_or_repeated_share"] == pytest.approx(0.6)
    assert total["completed"] == 3


def test_per_stage_per_task_and_per_object():
    inputs = CarbonInputs.from_site(None, **REPORT_INPUTS)
    outcomes = [_outcome(key=str(i)) for i in range(4)] + [
        _outcome(stage="sed", key="s", elapsed_s=600)
    ]
    result = estimate(outcomes, inputs, objects=2)

    assert set(result["stages"]) == {"vis_lp", "sed"}
    assert result["stages"]["vis_lp"]["per_completed_task"]["kwh"]["central"] == pytest.approx(
        result["stages"]["vis_lp"]["kwh"]["central"] / 4, abs=1e-3
    )
    assert result["per_object"]["kwh"]["central"] == pytest.approx(
        result["total"]["kwh"]["central"] / 2, abs=1e-3
    )


def test_missing_cpu_time_is_charged_at_full_use():
    inputs = CarbonInputs.from_site(None, **REPORT_INPUTS)
    outcome = _outcome()
    outcome.total_cpu_s = None
    result = estimate([outcome], inputs)

    assert result["assumptions"]["usage_missing"] == 1
    assert result["total"]["kwh"]["central"] == result["total"]["kwh"]["allocation"]


def test_gpu_tasks_add_their_gpus():
    site_like = CarbonInputs.from_site(None, **REPORT_INPUTS)
    site_like.gpu_w["gpu"] = Bounds(250, 300, 400)
    cpu = estimate([_outcome(partition="gpu")], site_like)
    gpu = estimate([_outcome(partition="gpu", gres="gres/gpu=1")], site_like)

    assert gpu["assumptions"]["gpu_tasks"] == 1
    assert gpu["total"]["kwh"]["central"] - cpu["total"]["kwh"]["central"] == pytest.approx(
        300 * 1.31 / 1000, abs=1e-3
    )
    assert gpu_count("gres/gpu=2") == 2
    assert gpu_count("gpu:a100:4") == 4
    assert gpu_count("cpu=8,mem=12G") == 0
    assert gpu_count(None) == 0


def test_ral_site_inputs(ral):
    inputs = CarbonInputs.from_site(ral)

    assert inputs.missing() == []
    assert inputs.ci_basis == "regional"
    # Ruling F6: regional central, the other bases as the bounds.
    assert inputs.ci_g_per_kwh.as_list() == [129, 188, 188]
    assert inputs.pue.as_list() == [1.14, 1.31, 1.45]
    assert inputs.tdp_w_per_cpu["ral"].as_list() == [2.8, 3.5, 4.4]
    assert inputs.gpu_w["gpu"].central == 300
    assert "carbon_intensity.regional" in inputs.unverified

    national = CarbonInputs.from_site(ral, ci_basis="national")
    assert national.ci_g_per_kwh.central == 129


def test_no_site_means_no_estimate_and_no_world_default():
    result = estimate([_outcome()], CarbonInputs.from_site(None))
    assert "error" in result
    assert "tdp_w_per_cpu" in result["error"] and "pue" in result["error"]

    energy_only = estimate(
        [_outcome()], CarbonInputs.from_site(None, pue=1.3, tdp_w_per_cpu=3.5)
    )
    assert energy_only["total"]["kwh"]["central"] > 0
    assert energy_only["total"]["kg_co2e"] is None
    assert "no world-average default" in energy_only["note"]


def test_unlisted_partition_takes_the_widest_range(ral):
    inputs = CarbonInputs.from_site(ral)
    result = estimate([_outcome(partition="other")], inputs)
    assert result["assumptions"]["partition_fallback"] == 1


def test_neso_intensity_is_cached_and_chunked(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return {"data": [{"intensity": {"forecast": 150, "actual": 100}}, {"intensity": {"forecast": 200, "actual": None}}]}

    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=30)
    measured = neso_intensity(start, end, cache_dir=tmp_path, fetch=fetch)

    assert len(calls) == 3  # 13 + 13 + 4 days
    assert measured["mean"] == pytest.approx(150)
    assert measured["points"] == 6

    neso_intensity(start, end, cache_dir=tmp_path, fetch=fetch)
    assert len(calls) == 3  # served from the cache

    regional = neso_intensity(
        start,
        start + timedelta(days=1),
        region_id=12,
        cache_dir=tmp_path,
        fetch=lambda url: {"data": [{"regionid": 12, "data": [{"intensity": {"forecast": 180}}]}]},
    )
    assert regional["mean"] == 180
    assert regional["basis"] == "NESO region 12"


def test_apply_neso_widens_the_bounds():
    inputs = CarbonInputs.from_site(None, ci=(129, 188, 188))
    apply_neso(inputs, {"mean": 210.0, "points": 4, "basis": "NESO region 12", "start": "a", "end": "b"})
    assert inputs.ci_g_per_kwh.as_list() == [129, 210.0, 210.0]
    assert inputs.ci_basis == "NESO region 12"


def test_begin_hint_picks_the_lowest_window():
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    values = [200, 190, 120, 110, 130, 210, 220, 230]
    series = [(now + timedelta(minutes=30 * i), v) for i, v in enumerate(values)]

    hint = begin_hint(series, duration_s=3600, now=now)

    assert hint["begin"] == "2026-10-30T13:00:00"
    assert hint["mean_g_per_kwh"] == 115.0
    assert hint["now_mean_g_per_kwh"] == 195.0
    assert hint["slurm"] == "--begin=2026-10-30T13:00:00"
    assert begin_hint([], 3600) is None
