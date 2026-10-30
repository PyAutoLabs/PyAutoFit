import textwrap
from pathlib import Path

import pytest

from autonerves.site_config import load_site, site_path


@pytest.fixture(name="ral")
def make_ral():
    return load_site(site_path("ral"))


CAMPAIGN = """
campaign: demo
project_root: {root}
manifest: manifest.txt
manifest_fields: [sample, tile]
chunk: 4
site: ral
log_dir: logs
stages:
  vis_lp:
    script: scripts/fit.py --stage=vis_lp
    env: {{JAX_PLATFORMS: cpu, OMP_NUM_THREADS: "$SLURM_CPUS_PER_TASK"}}
    resources: {{cpus: 4, mem_per_cpu_mb: 3072, time: "04:00:00", partition: ral, throttle: 400, nice: 1000}}
    success: {{zip_complete: "output/{{sample}}/{{tile}}/vis_lp", members: [files/samples_summary.json]}}
    retry: {{timeout: {{time: "p95+25%", max: 1}}, oom: {{mem: "peak*1.5", max: 1}}}}
  vis_pix:
    after: {{vis_lp: aftercorr}}
    script: scripts/fit.py --stage=vis_pix
    resources: {{cpus: 4, mem_mb: 16384, time: "12:00:00", partition: ral}}
  sed:
    after: {{vis_lp: aftercorr}}
    script: scripts/fit.py --stage=sed
    resources: {{cpus: 2, mem_per_cpu_mb: 4096, time: "06:00:00", partition: ral}}
"""


def write_campaign(root: Path, text: str = CAMPAIGN, n: int = 10) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.txt").write_text(
        "\n".join(f"dr1 Tile{index:03d}" for index in range(n)) + "\n"
    )
    path = root / "campaign.yaml"
    path.write_text(textwrap.dedent(text.format(root=root)))
    return path


@pytest.fixture(name="campaign_file")
def make_campaign_file(tmp_path):
    return write_campaign(tmp_path / "project")
