import pytest

from autofit.campaign.spec import Campaign, CampaignException
from test_autofit.campaign.conftest import CAMPAIGN, write_campaign


def test_load(campaign_file):
    campaign = Campaign.load(campaign_file)
    assert campaign.name == "demo"
    assert campaign.order() == ["vis_lp", "vis_pix", "sed"]
    vis_lp = campaign.stages["vis_lp"].resources
    assert (vis_lp.cpus, vis_lp.mem_per_cpu_mb, vis_lp.time_s, vis_lp.mem_mb) == (4, 3072, 14400, 12288)
    # A total `mem_mb` becomes one per-CPU knob.
    assert campaign.stages["vis_pix"].resources.mem_per_cpu_mb == 4096
    assert campaign.stages["vis_lp"].success.members == ["files/samples_summary.json"]
    assert campaign.site.name == "ral"


def test_chunks_and_fields(campaign_file):
    campaign = Campaign.load(campaign_file)
    assert campaign.chunks() == [(0, [0, 1, 2, 3]), (4, [4, 5, 6, 7]), (8, [8, 9])]
    assert campaign.chunks([9, 1]) == [(0, [1]), (8, [9])]
    assert campaign.fields(3) == {"index": "3", "line": "dr1 Tile003", "sample": "dr1", "tile": "Tile003"}


def test_comments_keep_line_numbers(tmp_path):
    path = write_campaign(tmp_path / "p")
    (tmp_path / "p" / "manifest.txt").write_text("# header\na\n\nb\n")
    assert Campaign.load(path).indices() == [1, 3]


def _errors(tmp_path, old, new):
    path = write_campaign(tmp_path / "p", CAMPAIGN.replace(old, new))
    with pytest.raises(CampaignException) as info:
        Campaign.load(path)
    return str(info.value)


def test_cpu_stage_on_gpu_refused(tmp_path):
    message = _errors(tmp_path, "time: \"12:00:00\", partition: ral}", "time: \"12:00:00\", partition: \"ral,gpu\"}")
    assert "mixes GPU partition" in message


def test_bulk_cpu_stage_on_gpu_alone_refused(tmp_path):
    message = _errors(tmp_path, "partition: ral, throttle: 400", "partition: gpu, throttle: 400")
    assert "stages.vis_lp.resources.partition" in message and "throttle 400" in message


def test_unknown_partition(tmp_path):
    assert "no partition 'cpu'" in _errors(tmp_path, "partition: ral, throttle", "partition: cpu, throttle")


def test_unknown_keys_and_dependencies(tmp_path):
    message = _errors(tmp_path, "after: {{vis_lp: aftercorr}}\n    script: scripts/fit.py --stage=vis_pix",
                      "after: {{nope: afterwards}}\n    script: scripts/fit.py --stage=vis_pix\n    walltime: 3")
    assert "unknown stage 'nope'" in message
    assert "'afterwards' is not one of" in message
    assert "unknown keys ['walltime']" in message


def test_chunk_above_max_array_size(tmp_path):
    assert "max_array_size 1001" in _errors(tmp_path, "chunk: 4", "chunk: 5000")


def test_unknown_retry_class(tmp_path):
    assert "unknown failure class 'segfault'" in _errors(tmp_path, "retry: {{timeout", "retry: {{segfault: none, timeout")


def test_cycle(tmp_path):
    text = CAMPAIGN.replace("    script: scripts/fit.py --stage=vis_lp", "    after: {{sed: aftercorr}}\n    script: scripts/fit.py --stage=vis_lp")
    message = _errors(tmp_path, CAMPAIGN, text)
    assert "cycle" in message


def test_local_campaign_needs_no_partition(tmp_path):
    text = (
        "campaign: local_demo\nproject_root: {root}\nmanifest: manifest.txt\nscheduler: local\n"
        "stages:\n  fit:\n    script: fit.py\n"
    )
    campaign = Campaign.load(write_campaign(tmp_path / "p", text))
    assert campaign.chunks() == [(0, list(range(10)))]
