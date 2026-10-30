"""
Evidence grades, ``campaign plan`` and ``campaign pilot``.

The DR1 repack test rebuilds the measurements behind the 2026-09-25/27 RAL repack
(vis_lp peak 4.1 GB at 53 % of 8 CPUs; vis_pix 11 GB at 83 %; SED 3.7 GB at 72 %) as a
``sacct`` dump and checks that ``plan`` proposes what the human chose by hand (vis_lp
12 GB / 4 CPUs, vis_pix 16 GB / 8, SED 8 GB / 8) within the stated tolerance: CPUs
exact, memory within one ``MEMORY_STEP_MB`` rounding step.
"""
import json
import re

import pytest

from autofit.campaign.__main__ import main
from autofit.campaign.digest import write_digest
from autofit.campaign.evidence import (
    designation_for,
    grade_of,
    matches,
    read_designations,
    read_evidence,
    worst,
)
from autofit.campaign.pilot import pilot_indices, write_pilot
from autofit.campaign.plan import (
    MEMORY_STEP_MB,
    build_plan,
    plan_markdown,
    read_pilot,
    write_plan,
)
from autofit.campaign.retro import retro
from autofit.campaign.spec import Campaign
from test_autofit.campaign.conftest import write_campaign
from test_autofit.campaign.test_sacct import HEADER

PROFILING = {
    "schema": "profiling-summary",
    "version": 3,
    "generated_at": "2026-10-29T09:46:25Z",
    "records": [
        {
            "id": "lifecycle/euclid-mge-current",
            "class": "secondary",
            "decision_id": None,
            "status": "current",
            "measurement": "runtime",
            "metric": "single_jit_block",
            "unit": "s",
            "likelihood": {"dataset_type": "imaging", "variant": "mge"},
            "instrument": "euclid",
            "regime": "standard",
            "setup_id": "jax_cpu_dense_fp64",
            "host_class": "hpc_ral_cpu",
            "model_variant": "default",
            "limitations": ["provisional_optimal: true"],
            "measured_at": "2026-10-20T00:00:00Z",
            "values": {"single_jit_block": 0.2, "vmap.per_call": 0.05},
            "projected_node_throughput": {"per_call_s": 0.05, "per_call_source": "vmap.per_call"},
            "resources": {"cpus_per_eval": 1, "peak_rss_gb": 2.0},
        },
        {
            "id": "lifecycle/euclid-mge-old",
            "class": "secondary",
            "status": "superseded",
            "measurement": "runtime",
            "unit": "s",
            "likelihood": {"dataset_type": "imaging", "variant": "mge"},
            "instrument": "euclid",
            "regime": "standard",
            "setup_id": "jax_cpu_dense_fp64",
            "host_class": "hpc_ral_cpu",
            "model_variant": "default",
            "values": {"vmap.per_call": 0.09},
        },
        {
            "id": "lifecycle/hst-mge-current",
            "class": "secondary",
            "status": "current",
            "measurement": "runtime",
            "unit": "s",
            "likelihood": {"dataset_type": "imaging", "variant": "mge"},
            "instrument": "hst",
            "regime": "standard",
            "setup_id": "jax_cpu_dense_fp64",
            "host_class": "hpc_ral_cpu",
            "model_variant": "default",
            "values": {"vmap.per_call": 0.08},
        },
        {
            "id": "lifecycle/euclid-mge-memory",
            "class": "secondary",
            "status": "current",
            "measurement": "memory_host",
            "metric": "host_peak_rss",
            "unit": "MiB",
            "likelihood": {"dataset_type": "imaging", "variant": "mge"},
            "instrument": "euclid",
            "regime": "standard",
            "setup_id": "jax_cpu_dense_fp64",
            "host_class": "hpc_ral_cpu",
            "model_variant": "default",
            "values": {"host_peak_rss": 3000.0},
        },
        {
            "id": "lifecycle/euclid-rect-gpu",
            "class": "secondary",
            "status": "current",
            "measurement": "runtime",
            "unit": "s",
            "likelihood": {"dataset_type": "imaging", "variant": "rectangular"},
            "instrument": "euclid",
            "regime": "standard",
            "setup_id": "jax_gpu_dense_fp64",
            "host_class": "hpc_a100",
            "model_variant": "default",
            "values": {"vmap.per_call": 0.01},
        },
    ],
}

WALLTIME = {
    "schema": "walltime-validation",
    "version": 1,
    "generated_at": "2026-10-28T00:00:00Z",
    "records": [
        {
            "id": "imaging/mge/hst/default/jax_cpu_dense_fp64/hpc_ral_cpu/nautilus:source_lp[1]/seed_1",
            "likelihood": {"dataset_type": "imaging", "variant": "mge"},
            "instrument": "hst",
            "setup_id": "jax_cpu_dense_fp64",
            "host_class": "hpc_ral_cpu",
            "model_variant": "default",
            "stage": "source_lp[1]",
            "wall_s": 3600.0,
            "likelihood_evals": 40000,
            "implied_s_per_eval": 0.09,
            "cores": 8,
        }
    ],
}

INFERENCE = {
    "schema": "inference-summary",
    "version": 2,
    "records": [
        {"id": "run/failed", "execution": {"completed": None}, "likelihood": {}, "dataset": {}},
        {
            "id": "run/done",
            "execution": {"completed": True},
            "likelihood": {"dataset_type": "imaging", "variant": "mge"},
            "dataset": {"instrument": "euclid"},
            "host_class": "hpc_ral_cpu",
            "model_variant": "default",
            "stage": "source_lp[1]",
            "configuration": {"config_name": "jax_cpu_dense_fp64"},
            "timings": {"sampling_s": 1200.0},
            "work": {"likelihood_evaluations": None},
        },
    ],
}

DESIGNATIONS = """
decisions:
- id: some-other-decision
  title: not a designation
- id: imaging-mge-standard-cpu
  kind: designation
  likelihood: imaging/mge
  regime: standard
  hardware_class: cpu_many_core
  optimal: jax_cpu_dense_fp64
  secondary: []
  evidence: [lifecycle/euclid-mge-current]
  rule: one sentence
  supersedes: null
"""

KEYS = {
    "dataset_type": "imaging",
    "variant": "mge",
    "instrument": "euclid",
    "regime": "standard",
    "setup_id": "jax_cpu_dense_fp64",
    "host_class": "hpc_ral_cpu",
    "model_variant": "default",
}


def _write(tmp_path, name, document):
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return path


@pytest.fixture(name="evidence")
def make_evidence(tmp_path):
    return [
        read_evidence(_write(tmp_path, "profiling.json", PROFILING)),
        read_evidence(_write(tmp_path, "walltime.json", WALLTIME)),
        read_evidence(_write(tmp_path, "inference.json", INFERENCE)),
    ]


def test_read_evidence_schemas_and_pulse_snapshot(tmp_path, evidence):
    assert [e.schema for e in evidence] == [
        "profiling-summary",
        "walltime-validation",
        "inference-summary",
    ]
    # A failed inference run measures nothing.
    assert [r.id for r in evidence[2].records] == ["run/done"]
    snapshot = read_evidence(_write(tmp_path, "lens.json", {"instance": "lens", "summary": PROFILING}))
    assert len(snapshot.records) == 5
    with pytest.raises(ValueError, match="not a supported evidence schema"):
        read_evidence(_write(tmp_path, "v2.json", dict(PROFILING, version=2)))


def test_grades(evidence):
    found = {m.record.id: m for m in matches(evidence, KEYS)}
    assert found["lifecycle/euclid-mge-current"].grade == "provisional"
    assert found["lifecycle/euclid-mge-old"].grade == "superseded"
    hst = found["lifecycle/hst-mge-current"]
    assert hst.grade == "analogue"
    assert hst.differs == {"instrument": ("euclid", "hst")}
    # Setup and host class are not analogue keys: a GPU record does not match.
    assert "lifecycle/euclid-rect-gpu" not in found
    # Best first.
    assert matches(evidence, KEYS, "runtime")[0].record.id == "lifecycle/euclid-mge-current"


def test_stage_key_and_missing_regime(evidence):
    keys = dict(KEYS, stage="source_lp[1]")
    found = {m.record.id: m for m in matches(evidence, keys)}
    # A profiling record is per call, so the stage key does not exclude it.
    assert found["lifecycle/euclid-mge-current"].grade == "provisional"
    # Walltime records carry no regime and are HST here: analogue, saying both.
    walltime = found[WALLTIME["records"][0]["id"]]
    assert walltime.grade == "analogue"
    assert set(walltime.differs) == {"instrument", "regime"}
    # The completed inference run matches exactly except the regime it lacks.
    assert found["run/done"].grade == "analogue"


def test_designation_turns_provisional_into_measured_current(tmp_path, evidence):
    designations = read_designations(_write_text(tmp_path, "decisions.yaml", DESIGNATIONS))
    assert [d["id"] for d in designations] == ["imaging-mge-standard-cpu"]
    assert designation_for(KEYS, designations)["optimal"] == "jax_cpu_dense_fp64"
    assert designation_for(dict(KEYS, host_class="hpc_a100"), designations) is None

    record = evidence[0].records[0]
    record.record_class, record.decision_id = "optimal", "imaging-mge-standard-cpu"
    assert grade_of(record, KEYS, designations).grade == "measured-current"
    assert grade_of(record, KEYS, []).grade == "provisional"


def test_no_keys_match_nothing(evidence):
    assert matches(evidence, {}) == []
    assert matches(evidence, {"colour": "red"}) == []


def test_worst():
    assert worst("provisional", "analogue", "measured-current") == "analogue"
    assert worst("estimated", "unsupported") == "unsupported"


def _write_text(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


# --- the DR1 repack, from a sacct fixture -------------------------------------------

REPACK = {
    # stage: (tasks, peak RSS MB at the largest task, CPU efficiency, wall h)
    "vis_lp": (40, 4198, 0.53, 1.5),
    "vis_pix": (40, 11264, 0.83, 9.1),
    "sed": (40, 3789, 0.72, 0.2),
}
CHOSEN = {"vis_lp": (4, 12288), "vis_pix": (8, 16384), "sed": (8, 8192)}


def _clock(seconds):
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    clock = f"{rest // 3600:02d}:{rest % 3600 // 60:02d}:{rest % 60:02d}"
    return f"{days}-{clock}" if days else clock


def repack_dump(path):
    rows = [HEADER]
    job = 100
    for stage, (n, peak, efficiency, hours) in REPACK.items():
        job += 1
        for task in range(n):
            scale = 0.7 + 0.3 * task / (n - 1)  # the last task is the largest
            wall = hours * 3600 * scale
            rss = peak * scale
            cpu = wall * 8 * efficiency
            raw = job * 1000 + task
            common = f"|ral|COMPLETED|0:0|2026-09-25T00:00:00|2026-09-25T00:00:00|2026-09-25T10:00:00|{_clock(wall)}"
            rows.append(
                f"{job}_{task}|{raw}|euclid_dr1_{stage}_n5k{common}|1-00:00:00|8|65536M|||"
                f"{_clock(cpu)}|{_clock(wall * 8)}|1|n1|cpu=8|cpu=8|/m/euclid_dr1/hpc"
            )
            rows.append(
                f"{job}_{task}.batch|{raw}.batch|batch{common}||8||{rss:.0f}M|{rss:.0f}M|"
                f"{_clock(cpu)}|{_clock(wall * 8)}|1|n1||cpu=8|"
            )
    path.write_text("\n".join(rows) + "\n")
    return path


REPACK_CAMPAIGN = """
campaign: dr1_repack
project_root: {root}
manifest: manifest.txt
manifest_fields: [sample, tile, pixels]
chunk: 1000
site: ral
stages:
  vis_lp:
    script: scripts/fit.py --stage=vis_lp
    resources: {{cpus: 8, mem_mb: 65536, time: "18:00:00", partition: ral}}
  vis_pix:
    after: {{vis_lp: aftercorr}}
    script: scripts/fit.py --stage=vis_pix
    resources: {{cpus: 8, mem_mb: 65536, time: "24:00:00", partition: ral}}
  sed:
    after: {{vis_lp: aftercorr}}
    script: scripts/fit.py --stage=sed
    resources: {{cpus: 8, mem_mb: 65536, time: "03:00:00", partition: ral}}
"""


@pytest.fixture(name="repack")
def make_repack(tmp_path, ral):
    root = tmp_path / "project"
    write_campaign(root, REPACK_CAMPAIGN)
    (root / "manifest.txt").write_text(
        "\n".join(f"dr1 Tile{i:03d} {100 + i}" for i in range(200)) + "\n"
    )
    campaign = Campaign.load(root / "campaign.yaml", site=ral)
    digest, outcomes = retro(
        repack_dump(tmp_path / "dump.psv"),
        stage_regex=r"euclid_dr1_(?P<stage>vis_lp|vis_pix|sed)",
        site=ral,
    )
    write_digest(tmp_path / "pilot", digest, outcomes)
    return campaign, tmp_path / "pilot"


def test_plan_reproduces_the_dr1_repack(repack):
    campaign, pilot_dir = repack
    plan = build_plan(campaign, pilot=read_pilot(pilot_dir), pilot_source=str(pilot_dir))
    for stage, (cpus, mem_mb) in CHOSEN.items():
        proposal = plan["stages"][stage]["proposal"]
        assert proposal["cpus"]["value"] == cpus, stage
        assert abs(proposal["mem_mb"]["value"] - mem_mb) <= MEMORY_STEP_MB, stage
        assert proposal["mem_mb"]["grade"] == "estimated"
        assert proposal["mem_mb"]["sources"] == [f"{pilot_dir}#{stage}"]
    # The two the human sized at measured peak × ~1.5 come out exactly.
    assert plan["stages"]["vis_pix"]["proposal"]["mem_mb"]["value"] == 16384
    assert plan["stages"]["sed"]["proposal"]["mem_mb"]["value"] == 8192
    # p95 of the tasks (9.1 h × 0.985) + 25 % = 11.2 h, rounded up to 15 min.
    assert plan["stages"]["vis_pix"]["proposal"]["time"]["value"] == "11:15:00"


def test_every_number_is_graded_and_no_optimal_without_a_designation(repack, evidence, tmp_path):
    campaign, pilot_dir = repack
    campaign.stages["vis_lp"].evidence = {"keys": KEYS}
    plan = build_plan(
        campaign,
        evidence=evidence,
        pilot=read_pilot(pilot_dir),
        pilot_source=str(pilot_dir),
        idle_cpus={"ral": {"value": 3400, "grade": "estimated", "sources": ["snapshot"]}},
    )
    write_plan(tmp_path / "out", plan)
    text = (tmp_path / "out" / "plan.md").read_text() + (tmp_path / "out" / "plan.json").read_text()
    # The source records carry "provisional_optimal: true"; it is never repeated.
    assert not re.search("optimal", text, re.IGNORECASE)

    def check(node, path="plan"):
        if isinstance(node, dict):
            if "value" in node and "grade" in node:
                assert node["grade"] in (
                    "measured-current", "provisional", "analogue", "superseded", "estimated", "unsupported",
                ), path
                assert node["value"] is None or node["grade"] != "unsupported" or "note" in node, path
                return
            for key, value in node.items():
                check(value, f"{path}.{key}")
        elif isinstance(node, list):
            for value in node:
                check(value, path)

    for stage in plan["stages"].values():
        for key in ("proposal", "eta", "disk", "energy"):
            check(stage[key])
    vis_lp = plan["stages"]["vis_lp"]
    assert vis_lp["evidence"]["per_call_s"] == {
        "value": 0.05,
        "grade": "provisional",
        "sources": ["lifecycle/euclid-mge-current"],
    }
    assert vis_lp["evidence"]["host_peak_rss_mb"]["value"] == round(3000 * 1.048576)
    assert {r["id"] for r in plan["records"]} >= {"lifecycle/euclid-mge-current"}
    assert vis_lp["proposal"]["throttle"]["value"] == 3400 // 2 // 4
    assert vis_lp["eta"]["eta_s"]["value"] > 0
    assert plan["totals"]["eta_s"]["value"] == sum(
        s["eta"]["eta_s"]["value"] for s in plan["stages"].values()
    )


def test_designation_is_the_only_way_to_say_optimal(repack, evidence, tmp_path):
    campaign, pilot_dir = repack
    campaign.stages["vis_lp"].evidence = {"keys": KEYS}
    designations = read_designations(_write_text(tmp_path, "decisions.yaml", DESIGNATIONS))
    plan = build_plan(campaign, evidence=evidence, designations=designations)
    assert plan["stages"]["vis_lp"]["evidence"]["designation"] == {
        "decision": "imaging-mge-standard-cpu",
        "optimal": "jax_cpu_dense_fp64",
    }
    assert "optimal setup per Pulse decision imaging-mge-standard-cpu" in plan_markdown(plan)


def test_evidence_only_plan_scales_evaluations(repack, evidence):
    campaign, _ = repack
    campaign.stages["vis_lp"].evidence = {"keys": dict(KEYS, stage="source_lp[1]")}
    plan = build_plan(campaign, evidence=evidence)
    ev = plan["stages"]["vis_lp"]["evidence"]
    # The HST walltime record: its wall time, and its evals × implied cost.
    assert ev["stage_wall_s"]["grade"] == "analogue"
    assert ev["scaled_wall_s"]["value"] == pytest.approx(40000 * 0.09)
    assert ev["scaled_wall_s"]["grade"] == "estimated"
    proposal = plan["stages"]["vis_lp"]["proposal"]
    assert proposal["time"]["value"] == "01:15:00"
    assert proposal["time"]["grade"] == "estimated"
    # No pilot: CPUs and memory fall back to the request / host memory, graded so.
    assert proposal["cpus"]["grade"] == "unsupported"
    assert proposal["mem_mb"]["grade"] == "estimated"
    # A stage with no evidence at all says so.
    assert plan["stages"]["sed"]["proposal"]["time"]["grade"] == "unsupported"


def test_censored_wall_time_is_reported(tmp_path, ral):
    from autofit.campaign.digest import TaskOutcome
    from autofit.campaign.plan import pilot_stats

    outcomes = [
        TaskOutcome(stage="sed", key=f"1_{i}", state="completed", elapsed_s=600, cpus=8,
                    max_rss_mb=3000, total_cpu_s=600 * 8 * 0.7, cpu_s_all_attempts=600 * 8)
        for i in range(10)
    ] + [
        TaskOutcome(stage="sed", key=f"1_{i}", state="timeout", elapsed_s=10800, time_limit_s=10800,
                    cpus=8, cpu_s_all_attempts=10800 * 8)
        for i in range(10, 13)
    ]
    stats = pilot_stats(outcomes, "sed", "pilot")
    assert stats["timeouts"] == 3
    assert stats["wall_censored"]
    assert stats["rerun_factor"] > 1


def test_size_bins(repack):
    campaign, pilot_dir = repack
    campaign.stages["vis_pix"].evidence = {"size_field": "pixels"}
    outcomes = read_pilot(pilot_dir)
    for outcome in outcomes:
        # Pilot tasks 0..39 are campaign indices 0..39; big wall for big index.
        outcome.index = int(outcome.key.split("_")[1])
    plan = build_plan(campaign, pilot=outcomes, pilot_source="pilot")
    bins = plan["stages"]["vis_pix"]["size_bins"]
    assert len(bins) == 3
    measured = [b["measured"]["value"] for b in bins]
    assert all(b["measured"]["grade"] == "estimated" for b in bins)
    assert sum(m["objects"] for m in measured) == 200
    assert measured[0]["pilot_tasks"] == 40 and bins[0]["time"]["grade"] == "estimated"
    assert not measured[0]["censored"]
    assert bins[2]["time"]["grade"] == "unsupported"


def test_pilot_is_stratified_and_never_submitted(repack, tmp_path):
    campaign, _ = repack
    pilot = pilot_indices(campaign, fraction=0.05, size_field="pixels", bins=5, seed=1)
    assert len(pilot["indices"]) == 10
    assert [b["chosen"] for b in pilot["bins"]] == [2] * 5
    assert pilot["bins"][0]["size_range"][0] == 100 and pilot["bins"][-1]["size_range"][1] == 299
    assert pilot == pilot_indices(campaign, fraction=0.05, size_field="pixels", bins=5, seed=1)

    uniform = pilot_indices(campaign, fraction=0.01)
    assert len(uniform["indices"]) == 2 and uniform["stratified_by"] is None

    path = write_pilot(tmp_path / "out", campaign, pilot, "campaign.yaml")
    document = json.loads(path.read_text())
    assert document["submit_command"].startswith("python -m autofit.campaign submit campaign.yaml --indices ")
    assert "Not submitted" in document["note"]


def test_cli_plan_and_pilot(repack, tmp_path, capsys):
    campaign, pilot_dir = repack
    out = tmp_path / "plan_out"
    profiling = _write(tmp_path, "p.json", PROFILING)
    assert main([
        "plan", str(campaign.source), "--evidence", str(profiling), "--pilot", str(pilot_dir),
        "--idle-cpus", "ral=1000", "--capacity-source", "sinfo 2026-10-30", "--output", str(out),
    ]) == 0
    text = capsys.readouterr().out
    assert text.startswith("# Plan: dr1_repack")
    plan = json.loads((out / "plan.json").read_text())
    assert plan["stages"]["sed"]["proposal"]["throttle"]["sources"] == ["sinfo 2026-10-30"]
    assert main(["pilot", str(campaign.source), "--size-field", "pixels", "--output", str(out)]) == 0
    assert json.loads((out / "pilot.json").read_text())["tasks"] == 5


def test_spec_validates_evidence(tmp_path):
    root = tmp_path / "project"
    text = REPACK_CAMPAIGN.replace(
        'script: scripts/fit.py --stage=sed',
        'script: scripts/fit.py --stage=sed\n    evidence: {{keys: {{colour: red}}, evals: -1, bad: 1}}',
    )
    write_campaign(root, text)
    errors = Campaign(__import__("yaml").safe_load((root / "campaign.yaml").read_text())).validate()
    assert any("evidence: unknown keys ['bad']" in e for e in errors)
    assert any("evidence.keys: unknown keys ['colour']" in e for e in errors)
    assert any("evidence.evals: must be a positive number" in e for e in errors)


def test_host_memory_falls_back_to_the_runtime_record(tmp_path):
    from autofit.campaign.plan import evidence_stats

    document = dict(PROFILING, records=[r for r in PROFILING["records"] if r["measurement"] != "memory_host"])
    stats = evidence_stats(KEYS, [read_evidence(_write(tmp_path, "p.json", document))])
    assert stats["host_peak_rss_mb"] == {
        "value": 2048,
        "grade": "provisional",
        "sources": ["lifecycle/euclid-mge-current"],
        "note": "peak RSS recorded with the runtime measurement",
    }


# --- review findings -------------------------------------------------------------

DELAUNAY_KEYS = {
    "dataset_type": "imaging",
    "variant": "delaunay",
    "instrument": "hst",
    "setup_id": "numba_cpu_sparse_fp64",
    "host_class": "hpc_ral_cpu",
    "stage": "mass_total[1]",
}


def _walltime_seed(seed, wall, evals, cores=8):
    return {
        "id": f"imaging/delaunay/hst/default/numba_cpu_sparse_fp64/hpc_ral_cpu/nautilus:mass_total[1]/seed_{seed}",
        "likelihood": {"dataset_type": "imaging", "variant": "delaunay"},
        "instrument": "hst",
        "setup_id": "numba_cpu_sparse_fp64",
        "host_class": "hpc_ral_cpu",
        "model_variant": "default",
        "stage": "mass_total[1]",
        "wall_s": wall,
        "likelihood_evals": evals,
        "implied_s_per_eval": wall / evals,
        "cores": cores,
    }


@pytest.fixture(name="seeds")
def make_seeds(tmp_path):
    document = {
        "schema": "walltime-validation",
        "version": 1,
        # File order puts the faster seed first: the plan must not depend on it.
        "records": [_walltime_seed(2, 4564.0, 20000), _walltime_seed(1, 8648.0, 38660)],
    }
    return [read_evidence(_write(tmp_path, "w.json", document))]


def _one_stage_campaign(tmp_path, ral, cpus):
    root = tmp_path / "one"
    text = REPACK_CAMPAIGN.split("  vis_pix:")[0].replace(
        "resources: {{cpus: 8, mem_mb: 65536", f"resources: {{{{cpus: {cpus}, mem_mb: 8192"
    )
    write_campaign(root, text, n=10)
    campaign = Campaign.load(root / "campaign.yaml", site=ral)
    campaign.stages["vis_lp"].evidence = {"keys": DELAUNAY_KEYS}
    return campaign


def test_review_1_cores_mismatch_scales_wall_and_cpu_hours(tmp_path, ral, seeds):
    campaign = _one_stage_campaign(tmp_path, ral, cpus=1)
    plan = build_plan(campaign, evidence=seeds)
    stage = plan["stages"]["vis_lp"]
    time = stage["proposal"]["time_s"]
    # 8648 s measured at 8 cores, requested 1: scaled ×8 (perfect scaling), + 25 %.
    assert time["value"] == -(-8648 * 8 * 1.25 // 900) * 900
    assert time["grade"] == "estimated"
    assert time["limitations"] == [
        "cores_mismatch: measured at 8 cores, requested 1; wall scaled by 8/1 assuming perfect scaling (optimistic)"
    ]
    # CPU-hours from measured core-seconds: 10 × 8648 s × 8 cores.
    assert stage["eta"]["cpu_h"]["value"] == round(10 * 8648 * 8 / 3600, 1)
    assert any(f.startswith("cores_mismatch") for f in stage["flags"])
    assert "**flag** cores_mismatch: measured at 8 cores, requested 1" in plan_markdown(plan)

    same = build_plan(_one_stage_campaign(tmp_path / "b", ral, cpus=8), evidence=seeds)
    assert "limitations" not in same["stages"]["vis_lp"]["proposal"]["time_s"]
    assert same["stages"]["vis_lp"]["flags"] == []


def test_review_2_largest_of_the_best_grade_with_every_id_and_spread(seeds):
    from autofit.campaign.plan import evidence_stats

    stats = evidence_stats(DELAUNAY_KEYS, seeds)
    wall = stats["stage_wall_s"]
    assert wall["value"] == 8648.0
    assert wall["grade"] == "measured-current"
    assert sorted(wall["sources"]) == sorted(r["id"] for r in [_walltime_seed(1, 1, 1), _walltime_seed(2, 1, 1)])
    assert wall["note"] == "largest of 2 matching records (4564–8648 s)"
    assert stats["likelihood_evals"]["value"] == 38660
    # Same answer whatever the file order.
    seeds[0].records.reverse()
    assert evidence_stats(DELAUNAY_KEYS, seeds)["stage_wall_s"] == wall


def test_review_3_cost_falls_back_to_per_call_when_implied_is_missing(tmp_path):
    from autofit.campaign.plan import evidence_stats

    walltime = {"schema": "walltime-validation", "version": 1, "records": [
        dict(WALLTIME["records"][0], instrument="euclid", implied_s_per_eval=None),
    ]}
    evidence = [
        read_evidence(_write(tmp_path, "p.json", PROFILING)),
        read_evidence(_write(tmp_path, "w.json", walltime)),
    ]
    stats = evidence_stats(dict(KEYS, stage="source_lp[1]"), evidence)
    assert stats["implied_s_per_eval"]["value"] is None
    # 40000 evaluations × the profiling per-call cost (0.05 s).
    assert stats["scaled_wall_s"]["value"] == pytest.approx(40000 * 0.05)


def test_review_4_every_number_in_plan_json_is_graded(repack, evidence, tmp_path):
    campaign, pilot_dir = repack
    campaign.stages["vis_lp"].evidence = {"keys": KEYS}
    campaign.stages["vis_pix"].evidence = {"size_field": "pixels"}
    outcomes = read_pilot(pilot_dir)
    for outcome in outcomes:
        outcome.index = int(outcome.key.split("_")[1])
    plan = build_plan(
        campaign, evidence=evidence, pilot=outcomes, pilot_source="pilot",
        idle_cpus={"*": {"value": 100, "grade": "estimated", "sources": ["x"]}},
    )
    write_plan(tmp_path / "out", plan)
    document = json.loads((tmp_path / "out" / "plan.json").read_text())

    bare = []

    def walk(node, path, inside):
        if isinstance(node, dict):
            inside = inside or ("grade" in node and "value" in node)
            for key, value in node.items():
                walk(value, f"{path}.{key}", inside)
        elif isinstance(node, list):
            for number, value in enumerate(node):
                walk(value, f"{path}[{number}]", inside)
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and not inside:
            bare.append(path)

    for name, stage in document["stages"].items():
        # The stage's own request is the campaign file's declaration, not evidence.
        walk({k: v for k, v in stage.items() if k != "request"}, f"stages.{name}", False)
    walk(document["totals"], "totals", False)
    assert bare == []


def test_review_4_censored_size_bin(repack):
    from autofit.campaign.digest import TaskOutcome

    campaign, _ = repack
    campaign.stages["sed"].evidence = {"size_field": "pixels"}
    outcomes = [
        TaskOutcome(stage="sed", key=f"1_{i}", index=i, state="completed", elapsed_s=600, cpus=8,
                    max_rss_mb=3000, total_cpu_s=600 * 8 * 0.7, cpu_s_all_attempts=600 * 8)
        for i in range(5)
    ] + [
        TaskOutcome(stage="sed", key=f"1_{i}", index=i, state="timeout", elapsed_s=10800,
                    time_limit_s=10800, cpus=8, cpu_s_all_attempts=10800 * 8)
        for i in range(5, 10)
    ]
    plan = build_plan(campaign, pilot=outcomes, pilot_source="pilot")
    first = plan["stages"]["sed"]["size_bins"][0]
    assert first["measured"]["value"]["censored"]
    assert "censored" in first["time"]["note"]
    assert "censored wall time in size bin 0" in plan["stages"]["sed"]["flags"]


def test_designation_must_cover_the_record_and_be_current(tmp_path, evidence):
    text = DESIGNATIONS + """
- id: imaging-mge-standard-cpu-v2
  kind: designation
  likelihood: imaging/mge
  regime: standard
  hardware_class: cpu_many_core
  optimal: jax_cpu_dense_fp64
  secondary: []
  evidence: [lifecycle/euclid-mge-current]
  rule: one sentence
  supersedes: imaging-mge-standard-cpu
"""
    designations = read_designations(_write_text(tmp_path, "decisions.yaml", text))
    record = evidence[0].records[0]
    record.record_class = "optimal"
    # Citing the superseded decision is not enough.
    record.decision_id = "imaging-mge-standard-cpu"
    assert grade_of(record, KEYS, designations).grade == "provisional"
    record.decision_id = "imaging-mge-standard-cpu-v2"
    assert grade_of(record, KEYS, designations).grade == "measured-current"
    # A designation for another hardware class does not cover this record.
    record.keys["host_class"] = "hpc_a100"
    assert grade_of(record, dict(KEYS, host_class="hpc_a100"), designations).grade == "provisional"
