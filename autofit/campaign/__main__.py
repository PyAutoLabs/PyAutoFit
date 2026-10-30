"""
Command-line entry point: ``python -m autofit.campaign <command>``.

Commands
--------
validate <campaign.yaml>
    Check the campaign file against its site; print every problem.
submit <campaign.yaml> [--stage S ...] [--indices SPEC] [--proposal FILE] [--dry-run]
    Submit the stages (default: all) over the manifest (or ``--indices``, a Slurm
    array spec of campaign indices), chunk by chunk. Every array is linted against
    the GPU-partition rule first. ``--proposal`` submits an approved resubmit proposal
    (its stage, indices and resources). ``--dry-run`` prints the batch scripts and
    submits nothing.
status <campaign.yaml>
    Collect the campaign (one accounting query, one capacity snapshot, the status
    records), write the digest files and print ``digest.md``.
resubmit <campaign.yaml> --stage S --class C
    Write ``resubmit_<S>_<C>.json``, a proposal sized from the failed tasks' own
    measurements (from the last ``status``), and print it. Nothing is submitted.
retro --sacct DUMP [--workdir-prefix DIR ...] [--name-regex RE] [--stage-regex RE]
      [--other-stage NAME] [--status-root DIR] [--site NAME] [--name NAME] [--output DIR]
    The digest of a finished campaign from a ``sacct --parsable2`` dump alone.
plan <campaign.yaml> [--evidence SRC ...] [--pilot TASKS] [--decisions YAML]
     [--idle-cpus [PARTITION=]N ...] [--capacity-source TEXT] [--results-index DIR] [--output DIR]
    Propose each stage's resources, wall time, ETA and disk from a pilot and the
    profiling / inference summaries, every number graded (``autofit.campaign.plan``).
pilot <campaign.yaml> [--fraction F] [--size-field NAME | --size-path TEMPLATE] [--bins N]
      [--seed S] [--output DIR]
    Write ``pilot.json``: a stratified subset of the manifest to measure first. Nothing
    is submitted.
"""
import argparse
import json
import sys

from autofit.campaign.lint import PartitionLintError
from autofit.campaign.spec import Campaign, CampaignException
from autofit.campaign.units import parse_array_spec


def _validate(arguments) -> int:
    try:
        campaign = Campaign.load(arguments.campaign)
    except CampaignException as exception:
        print(exception, file=sys.stderr)
        return 1
    chunks = campaign.chunks()
    print(
        f"{campaign.name}: valid — {len(campaign.indices())} objects, "
        f"{len(chunks)} chunk(s) of ≤{campaign.chunk_size()}, "
        f"stages {' → '.join(campaign.order())}"
    )
    if campaign.site_name and campaign.site is not None and campaign.site.unverified():
        print(f"site {campaign.site.name}: unverified values {campaign.site.unverified()}")
    return 0


def _submit(arguments) -> int:
    from autofit.campaign.scheduler import scheduler_for
    from autofit.campaign.submit import submit

    try:
        campaign = Campaign.load(arguments.campaign)
        indices = parse_array_spec(arguments.indices)[0] if arguments.indices else None
        stages = arguments.stage
        resources = None
        if arguments.proposal:
            stages, indices, resources = _proposal(campaign, arguments.proposal)
        scheduler = None if arguments.dry_run else scheduler_for(campaign)
        entries = submit(
            campaign,
            scheduler,
            stages=stages,
            indices=indices,
            resources=resources,
            dry_run=arguments.dry_run,
        )
    except (CampaignException, PartitionLintError) as exception:
        print(f"refused: {exception}", file=sys.stderr)
        return 1
    if arguments.dry_run:
        for entry in entries:
            print(f"# {entry['stage']} chunk {entry['offset']} (lint: {entry['lint']})")
            print(entry.get("script", json.dumps(entry, indent=1)))
        return 0
    for entry in entries:
        print(f"{entry['stage']} chunk {entry['offset']}: job {entry['job_id']}")
    return 0


def _proposal(campaign, path):
    from autofit.campaign.spec import Resources

    proposal = json.loads(open(path).read())
    if not proposal.get("indices") or not proposal.get("resources"):
        raise CampaignException(f"Proposal {path} has nothing to submit: {proposal.get('reason')}")
    raw = dict(proposal["resources"])
    resources = Resources(**raw)
    return [proposal["stage"]], parse_array_spec(proposal["indices"])[0], {proposal["stage"]: resources}


def _status(arguments) -> int:
    from autofit.campaign.collector import collect
    from autofit.campaign.digest import digest_markdown
    from autofit.campaign.scheduler import scheduler_for

    try:
        campaign = Campaign.load(arguments.campaign)
    except CampaignException as exception:
        print(exception, file=sys.stderr)
        return 1
    digest = collect(campaign, scheduler_for(campaign))
    print(digest_markdown(digest), end="")
    return 0


def _resubmit(arguments) -> int:
    from autofit.campaign.digest import read_outcomes
    from autofit.campaign.proposals import propose_resubmit

    try:
        campaign = Campaign.load(arguments.campaign)
    except CampaignException as exception:
        print(exception, file=sys.stderr)
        return 1
    if arguments.stage not in campaign.stages:
        print(f"Unknown stage {arguments.stage!r}.", file=sys.stderr)
        return 1
    tasks = campaign.directory / "tasks.jsonl"
    if not tasks.exists():
        print(f"No {tasks}: run `status` first.", file=sys.stderr)
        return 1
    proposal = propose_resubmit(
        campaign.stages[arguments.stage],
        getattr(arguments, "class"),
        read_outcomes(tasks),
        site=campaign.site if campaign.site_name else None,
    )
    path = campaign.directory / f"resubmit_{arguments.stage}_{getattr(arguments, 'class')}.json"
    path.write_text(json.dumps(proposal, indent=1, default=str))
    print(json.dumps(proposal, indent=1, default=str))
    if proposal["indices"]:
        print(
            f"\nProposal only. To act on it once approved:\n"
            f"  python -m autofit.campaign submit {arguments.campaign} --proposal {path}"
        )
    return 0


def _retro(arguments) -> int:
    from autofit.campaign.digest import digest_markdown, write_digest
    from autofit.campaign.retro import retro

    site = None
    if arguments.site:
        from autonerves.site_config import load_site, site_path

        site = load_site(site_path(arguments.site))
    digest, outcomes = retro(
        arguments.sacct,
        workdir_prefixes=arguments.workdir_prefix or (),
        name_regex=arguments.name_regex,
        stage_regex=arguments.stage_regex,
        status_root=arguments.status_root,
        site=site,
        name=arguments.name,
        other_stage=arguments.other_stage,
    )
    if arguments.output:
        write_digest(arguments.output, digest, outcomes)
    print(digest_markdown(digest), end="")
    return 0


def _idle_cpus(values, source):
    from autofit.campaign.plan import graded

    idle = {}
    for value in values or ():
        partition, _, number = value.rpartition("=")
        idle[partition or "*"] = graded(int(number), "estimated", [source or "stated on the command line"])
    return idle


def _plan(arguments) -> int:
    from autofit.campaign.evidence import read_designations, read_evidence
    from autofit.campaign.plan import build_plan, plan_markdown, read_pilot, read_results_index, write_plan

    try:
        campaign = Campaign.load(arguments.campaign)
    except CampaignException as exception:
        print(exception, file=sys.stderr)
        return 1
    plan = build_plan(
        campaign,
        evidence=[read_evidence(source) for source in arguments.evidence or ()],
        pilot=read_pilot(arguments.pilot) if arguments.pilot else None,
        pilot_source=arguments.pilot,
        designations=read_designations(arguments.decisions),
        idle_cpus=_idle_cpus(arguments.idle_cpus, arguments.capacity_source),
        results_index=read_results_index(arguments.results_index) if arguments.results_index else None,
        decisions_source=arguments.decisions,
    )
    write_plan(arguments.output or campaign.directory, plan)
    print(plan_markdown(plan), end="")
    return 0


def _pilot(arguments) -> int:
    from autofit.campaign.pilot import pilot_indices, write_pilot

    try:
        campaign = Campaign.load(arguments.campaign)
    except CampaignException as exception:
        print(exception, file=sys.stderr)
        return 1
    pilot = pilot_indices(
        campaign,
        fraction=arguments.fraction,
        size_field=arguments.size_field,
        size_path=arguments.size_path,
        bins=arguments.bins,
        seed=arguments.seed,
    )
    path = write_pilot(arguments.output or campaign.directory, campaign, pilot, arguments.campaign)
    print(path.read_text())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autofit.campaign")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Check a campaign file.")
    validate.add_argument("campaign")
    validate.set_defaults(handler=_validate)

    submit = commands.add_parser("submit", help="Submit a campaign's stages.")
    submit.add_argument("campaign")
    submit.add_argument("--stage", action="append", help="Stage to submit (repeatable).")
    submit.add_argument("--indices", help="Campaign indices, as a Slurm array spec.")
    submit.add_argument("--proposal", help="Submit an approved resubmit proposal.")
    submit.add_argument("--dry-run", action="store_true", help="Print, do not submit.")
    submit.set_defaults(handler=_submit)

    status = commands.add_parser("status", help="Collect the campaign into its digest.")
    status.add_argument("campaign")
    status.set_defaults(handler=_status)

    resubmit = commands.add_parser("resubmit", help="Propose a resubmit (never submits).")
    resubmit.add_argument("campaign")
    resubmit.add_argument("--stage", required=True)
    resubmit.add_argument("--class", required=True, help="timeout, oom, node_fail, ...")
    resubmit.set_defaults(handler=_resubmit)

    retro = commands.add_parser("retro", help="Digest a finished campaign from sacct.")
    retro.add_argument("--sacct", required=True, help="A sacct --parsable2 dump.")
    retro.add_argument("--workdir-prefix", action="append", help="Keep jobs under this WorkDir.")
    retro.add_argument("--name-regex", help="Keep jobs whose name matches.")
    retro.add_argument("--stage-regex", help="Stage from job name (group 'stage').")
    retro.add_argument("--other-stage", help="Stage of jobs the stage regex misses.")
    retro.add_argument("--status-root", help="Join status records under this directory.")
    retro.add_argument("--site", help="Site layer (GPU-only partitions).")
    retro.add_argument("--name", default="retro", help="Campaign name in the digest.")
    retro.add_argument("--output", help="Write the digest files here.")
    retro.set_defaults(handler=_retro)

    plan = commands.add_parser("plan", help="Propose resources from evidence (never submits).")
    plan.add_argument("campaign")
    plan.add_argument("--evidence", action="append", help="A profiling / inference summary (path or URL).")
    plan.add_argument("--pilot", help="Pilot outcomes: a tasks.jsonl or the directory holding one.")
    plan.add_argument("--decisions", help="PyAutoPulse decisions.yaml (designations).")
    plan.add_argument("--idle-cpus", action="append", help="[PARTITION=]N idle CPUs (repeatable).")
    plan.add_argument("--capacity-source", help="Where the idle-CPU figure comes from.")
    plan.add_argument("--results-index", help="A results index, for the disk forecast.")
    plan.add_argument("--output", help="Write plan.json / plan.md here.")
    plan.set_defaults(handler=_plan)

    pilot = commands.add_parser("pilot", help="Choose a stratified pilot (never submits).")
    pilot.add_argument("campaign")
    pilot.add_argument("--fraction", type=float, default=0.02)
    pilot.add_argument("--size-field", help="Manifest field holding each object's size.")
    pilot.add_argument("--size-path", help="Path template whose file size is the object's size.")
    pilot.add_argument("--bins", type=int, default=5)
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--output", help="Write pilot.json here.")
    pilot.set_defaults(handler=_pilot)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
