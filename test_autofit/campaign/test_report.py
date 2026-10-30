"""
``campaign report`` and ``campaign carbon`` over a digest folder: the report's
sections, the paper paragraph's wording and the command line (offline).
"""
import json
import re

from autofit.campaign.__main__ import main
from autofit.campaign.digest import TaskOutcome, build_digest, write_digest
from autofit.campaign.report import build_report, paper_paragraph, report_markdown


def _digest_folder(tmp_path):
    outcomes = [
        TaskOutcome(
            stage="vis_lp",
            key=f"1_{i}",
            state="completed",
            elapsed_s=5400,
            cpus=8,
            req_mem_mb=12288.0,
            max_rss_mb=4000.0,
            total_cpu_s=0.53 * 5400 * 8,
            partition="ral",
            cpu_s_all_attempts=5400 * 8,
            start="2026-09-11T10:00:00",
            end="2026-09-11T11:30:00",
        )
        for i in range(30)
    ] + [
        TaskOutcome(
            stage="sed",
            key="2_0",
            state="timeout",
            elapsed_s=10800,
            cpus=8,
            req_mem_mb=16384.0,
            max_rss_mb=3000.0,
            total_cpu_s=0.3 * 10800 * 8,
            partition="ral",
            cpu_s_all_attempts=10800 * 8,
            start="2026-09-12T10:00:00",
            end="2026-09-12T13:00:00",
        )
    ]
    digest = build_digest("demo", outcomes)
    digest["period"] = {"first_submit": "2026-09-11T10:00:00", "last_end": "2026-09-12T13:00:00"}
    folder = tmp_path / "digest"
    write_digest(folder, digest, outcomes)
    return folder


def test_report_sections_and_paragraph(tmp_path, ral):
    folder = _digest_folder(tmp_path)
    index = tmp_path / "index"
    index.mkdir()
    (index / "index.json").write_text(
        json.dumps({"updated": "2026-10-30", "counts": {"completed": 29, "corrupt_zip": 1}})
    )

    report = build_report(folder, site=ral, objects=30, results_index=index)
    text = report_markdown(report)

    for heading in (
        "## Compute",
        "## Energy and carbon (estimate)",
        "### Inputs",
        "## Failure census",
        "## Stages",
        "## Results",
        "## For the paper",
    ):
        assert heading in text
    assert "corrupt_zip 1" in text
    assert "Per object (30 objects)" in text

    paragraph = report["paragraph"]
    assert "estimate" in paragraph
    assert "Green Algorithms" in paragraph
    assert "regional basis" in paragraph
    assert re.search(r"range [\d.,]+–[\d.,]+ (k|M)Wh", paragraph)
    assert "failed, timed-out or repeated" in paragraph
    assert "saving" not in text.lower()


def test_paragraph_without_carbon_intensity(tmp_path):
    folder = _digest_folder(tmp_path)
    from autofit.campaign.carbon import CarbonInputs

    report = build_report(folder, inputs=CarbonInputs.from_site(None, pue=1.3, tdp_w_per_cpu=3.5))
    assert "gCO2e" not in report["paragraph"]
    assert "energy only" in report_markdown(report)

    no_inputs = build_report(folder)
    assert no_inputs["paragraph"] is None
    assert "Not estimated" in report_markdown(no_inputs)
    assert paper_paragraph(no_inputs["digest"], no_inputs["carbon"]) is None


def test_cli_report_and_carbon(tmp_path, capsys):
    folder = _digest_folder(tmp_path)
    out = tmp_path / "out"

    assert main(["report", str(folder), "--site", "ral", "--output", str(out)]) == 0
    assert (out / "report.md").exists() and (out / "report.json").exists()
    assert "## For the paper" in capsys.readouterr().out

    assert main(["carbon", str(folder), "--site", "ral", "--ci-basis", "national"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["inputs"]["ci_g_per_kwh"][1] == 129

    assert main(["carbon", str(folder), "--pue", "1.2,1.3,1.4", "--tdp", "3.5"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total"]["kg_co2e"] is None

    assert main(["carbon", str(folder)]) == 1
    capsys.readouterr()


def test_cli_report_from_campaign_file(campaign_file, capsys):
    from autofit.campaign.spec import Campaign

    campaign = Campaign.load(campaign_file)
    outcomes = [
        TaskOutcome(stage="vis_lp", key="1_0", state="completed", elapsed_s=3600, cpus=4,
                    req_mem_mb=12288.0, total_cpu_s=7200.0, partition="ral", cpu_s_all_attempts=14400.0)
    ]
    write_digest(campaign.directory, build_digest("demo", outcomes), outcomes)

    assert main(["report", str(campaign_file)]) == 0
    assert "regional basis" in capsys.readouterr().out
    assert (campaign.directory / "report.md").exists()


def _live_folder(tmp_path, outcomes):
    folder = tmp_path / "live"
    write_digest(folder, build_digest("live", outcomes), outcomes)
    return folder


def test_report_with_only_unstarted_or_running_tasks(tmp_path, ral):
    # Review finding 2: zero finished energy must not crash the report.
    pending = TaskOutcome(stage="vis_lp", key="1_0", state="pending", cpus=8, partition="ral")
    report = build_report(_live_folder(tmp_path, [pending]), site=ral)
    text = report_markdown(report)
    assert "Failed or repeated work: 0 kWh of 0 central" in text
    assert "of finished work" not in text

    running = TaskOutcome(
        stage="vis_lp", key="1_1", state="running", elapsed_s=1800, cpus=8,
        req_mem_mb=12288.0, total_cpu_s=7200.0, partition="ral", cpu_s_all_attempts=14400.0,
    )
    report = build_report(_live_folder(tmp_path / "b", [running]), site=ral)
    text = report_markdown(report)
    assert "In progress: 1 unfinished tasks" in text
    # Review finding 3: never claims running work was wasted.
    assert "was spent on failed" not in report["paragraph"]
    assert "still running" in report["paragraph"]


def test_paragraph_with_given_ci_names_no_site_bases(tmp_path, ral):
    from autofit.campaign.carbon import CarbonInputs

    folder = _digest_folder(tmp_path)
    inputs = CarbonInputs.from_site(ral, ci=(100, 130, 190))
    paragraph = build_report(folder, site=ral, inputs=inputs)["paragraph"]
    assert "130 gCO2e/kWh (given; range 100–190)" in paragraph
    assert "national" not in paragraph and "regional" not in paragraph


def test_cli_without_source_is_a_usage_error(capsys):
    import pytest

    for command in ("report", "carbon"):
        with pytest.raises(SystemExit) as exit_info:
            main([command])
        assert exit_info.value.code == 2
        assert "give a digest folder or campaign.yaml" in capsys.readouterr().err
