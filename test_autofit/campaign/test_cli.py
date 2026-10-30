from autofit.campaign.__main__ import main


def test_validate(campaign_file, capsys):
    assert main(["validate", str(campaign_file)]) == 0
    out = capsys.readouterr().out
    assert "10 objects, 3 chunk(s) of ≤4, stages vis_lp → vis_pix → sed" in out
    assert "unverified" in out


def test_validate_refuses(campaign_file, capsys):
    campaign_file.write_text(campaign_file.read_text().replace("partition: ral, throttle", "partition: \"gpu,ral\", throttle"))
    assert main(["validate", str(campaign_file)]) == 1
    assert "mixes GPU partition" in capsys.readouterr().err


def test_submit_dry_run(campaign_file, capsys):
    assert main(["submit", str(campaign_file), "--dry-run", "--stage", "vis_lp", "--indices", "0-5"]) == 0
    out = capsys.readouterr().out
    assert out.count("#!/bin/bash") == 2
    assert "#SBATCH --array=0-1%400" in out
