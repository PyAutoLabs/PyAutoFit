from autofit.campaign.__main__ import main
from autofit.campaign.retro import peak_concurrency, retro, stage_of, under
from autofit.campaign.scheduler.sacct import parse
from test_autofit.campaign.test_sacct import HEADER

ROWS = [
    "1_0|1|euclid_dr1_vis_lp_n5k_p01|ral|COMPLETED|0:0|2026-09-01T00:00:00|2026-09-01T00:00:00|2026-09-01T02:00:00|02:00:00|04:00:00|4|12288M|||07:00:00|08:00:00|1|n1|cpu=4|cpu=4|/m/euclid_dr1/hpc/batch_cpu",
    "1_0.batch|1.batch|batch||COMPLETED|0:0|2026-09-01T00:00:00|2026-09-01T00:00:00|2026-09-01T02:00:00|02:00:00||4||6000M|6000M|07:00:00|08:00:00|1|n1||cpu=4|",
    "1_1|2|euclid_dr1_vis_lp_n5k_p01|ral|TIMEOUT|0:0|2026-09-01T00:00:00|2026-09-01T01:00:00|2026-09-01T05:00:00|04:00:00|04:00:00|4|12288M|||15:00:00|16:00:00|1|n1|cpu=4|cpu=4|/m/euclid_dr1/hpc/batch_cpu",
    "1_1.batch|2.batch|batch||CANCELLED|0:15|2026-09-01T00:00:00|2026-09-01T01:00:00|2026-09-01T05:00:00|04:00:00||4||9000M|9000M|15:00:00|16:00:00|1|n1||cpu=4|",
    "3_0|3|euclid_dr1_sed_v2g|gpu|COMPLETED|0:0|2026-09-30T00:00:00|2026-09-30T00:00:00|2026-09-30T01:00:00|01:00:00|08:00:00|8|8192M|||06:00:00|08:00:00|1|g1|cpu=8|cpu=8|/m/active/euclid_dr1/hpc/batch_cpu",
    "4_0|4|other|ral|COMPLETED|0:0|2026-09-01T00:00:00|2026-09-01T00:00:00|2026-09-01T01:00:00|01:00:00|08:00:00|8|8192M|||06:00:00|08:00:00|1|n2|cpu=8|cpu=8|/m/euclid_dr1_old/hpc",
]
STAGES = r"euclid_dr1_(?P<stage>vis_lp|vis_pix|sed)"


def _dump(tmp_path):
    path = tmp_path / "dump.psv"
    path.write_text("\n".join([HEADER] + ROWS) + "\n")
    return path


def test_under_is_a_path_boundary():
    assert under("/m/euclid_dr1/hpc", ["/m/euclid_dr1"])
    assert under("/m/euclid_dr1", ["/m/euclid_dr1/"])
    assert not under("/m/euclid_dr1_old/hpc", ["/m/euclid_dr1"])
    assert under("/anything", [])


def test_stage_of():
    assert stage_of("euclid_dr1_vis_pix_v2g_p05", STAGES) == "vis_pix"
    assert stage_of("subval_source_pix", STAGES) == "subval_source_pix"
    assert stage_of("euclid_dr1_release_package", STAGES, other="tooling") == "tooling"


def test_peak_concurrency(tmp_path):
    peak, at = peak_concurrency(parse(_dump(tmp_path).read_text()))
    # 1_0 and 4_0 start together; 4_0 ends as 1_1 starts (end before start).
    assert peak == 12 and at == "2026-09-01T00:00:00"


def test_retro(tmp_path, ral):
    digest, outcomes = retro(
        _dump(tmp_path),
        workdir_prefixes=["/m/euclid_dr1", "/m/active/euclid_dr1"],
        stage_regex=STAGES,
        site=ral,
        name="dr1",
    )
    assert digest["source"]["allocations_selected"] == 3
    assert set(digest["stages"]) == {"vis_lp", "sed"}
    vis_lp = digest["stages"]["vis_lp"]
    assert vis_lp["states"] == {"completed": 1, "timeout": 1}
    assert vis_lp["rss_mb"]["max"] == 6000
    assert vis_lp["cpu_h"] == {"successful": 8.0, "all_attempts": 24.0, "failed_or_repeated": 16.0}
    assert digest["stages"]["sed"]["gpu_partition_cpu_tasks"] == {"tasks": 1, "cpu_h": 8.0}
    assert digest["period"] == {"first_submit": "2026-09-01T00:00:00", "last_end": "2026-09-30T01:00:00"}


def test_retro_cli(tmp_path, capsys):
    output = tmp_path / "out"
    assert main(["retro", "--sacct", str(_dump(tmp_path)), "--workdir-prefix", "/m/euclid_dr1",
                 "--stage-regex", STAGES, "--site", "ral", "--name", "dr1", "--output", str(output)]) == 0
    assert "# dr1" in capsys.readouterr().out
    assert (output / "failed_timeout.txt").read_text().startswith("vis_lp\t-\t1_1")
    assert sorted(p.name for p in output.iterdir()) == ["digest.json", "digest.md", "failed_timeout.txt", "tasks.jsonl"]
