"""
Command-line entry point: ``python -m autofit.campaign <command>``.

Commands
--------
validate <campaign.yaml>
    Check the campaign file against its site; print every problem.
submit <campaign.yaml> [--stage S ...] [--indices SPEC] [--dry-run]
    Submit the stages (default: all) over the manifest (or ``--indices``, a Slurm
    array spec of campaign indices), chunk by chunk. Every array is linted against
    the GPU-partition rule first. ``--dry-run`` prints the batch scripts and submits
    nothing.
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
        scheduler = None if arguments.dry_run else scheduler_for(campaign)
        entries = submit(
            campaign,
            scheduler,
            stages=arguments.stage,
            indices=indices,
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
    submit.add_argument("--dry-run", action="store_true", help="Print, do not submit.")
    submit.set_defaults(handler=_submit)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
