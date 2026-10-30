import sys
import textwrap

from autofit.campaign import Campaign
from autofit.campaign.submit import submit
from autofit.campaign.scheduler import LocalScheduler
from test_autofit.campaign.conftest import write_campaign

TEXT = """
campaign: local_demo
project_root: {root}
manifest: manifest.txt
scheduler: local
chunk: 2
stages:
  first:
    script: task.py first
    resources: {{time: "00:01:00"}}
  second:
    after: {{first: aftercorr}}
    script: task.py second
"""


def test_local_campaign(tmp_path):
    root = tmp_path / "p"
    path = write_campaign(root, TEXT, n=4)
    (root / "task.py").write_text(
        textwrap.dedent(
            """
            import os, sys
            index = int(os.environ["CAMPAIGN_INDEX"])
            assert index == int(os.environ["CAMPAIGN_OFFSET"]) + int(os.environ["SLURM_ARRAY_TASK_ID"])
            if sys.argv[1] == "first" and index == 2:
                sys.exit(3)
            open(f"{sys.argv[1]}_{index}.done", "w").write(os.environ["MANIFEST"])
            """
        )
    )
    campaign = Campaign.load(path)
    scheduler = LocalScheduler(max_workers=2)
    entries = submit(campaign, scheduler)
    scheduler.wait()
    states = {
        (record.name, record.task_id + entry["offset"]): record.scheduler_state
        for entry in entries
        for record in scheduler.query([entry["job_id"]])
    }
    assert states[("local_demo_first", 2)] == "FAILED"
    assert states[("local_demo_second", 2)] == "CANCELLED"
    assert states[("local_demo_second", 3)] == "COMPLETED"
    assert sorted(p.name for p in root.glob("*.done")) == [
        "first_0.done", "first_1.done", "first_3.done",
        "second_0.done", "second_1.done", "second_3.done",
    ]
