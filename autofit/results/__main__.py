"""
``python -m autofit.results``: index, classify and count results from the shell.

    python -m autofit.results index output/ [more_roots/ ...] [--index-dir DIR]
    python -m autofit.results status output/ [--index-dir DIR] [--state corrupt_zip]
    python -m autofit.results count output/ [--index-dir DIR] [--state completed]

``index`` brings the index up to date (only new or changed results are opened);
``status`` prints one line per result state, or with ``--state`` the results in that
state; ``count`` prints a number. ``status`` and ``count`` read the existing index and
never open a result; run ``index`` first.
"""
import argparse
import json
import sys
import time

from autofit.non_linear.status.states import ResultState
from autofit.results.index import ResultsIndex


def _index(arguments) -> ResultsIndex:
    return ResultsIndex(
        arguments.roots,
        index_dir=arguments.index_dir,
        verify=not getattr(arguments, "no_verify", False),
        stale_after_s=getattr(arguments, "stale_after_s", 600.0),
    )


def _states(value):
    if value is None:
        return None
    states = [state.strip() for state in value.split(",") if state.strip()]
    for state in states:
        ResultState(state)
    return states


def command_index(arguments) -> int:
    index = _index(arguments)
    start = time.perf_counter()
    counts = index.update()
    elapsed = time.perf_counter() - start
    print(
        f"indexed {counts['scanned']} results in {elapsed:.1f} s: "
        f"{counts['new']} new, {counts['changed']} changed, "
        f"{counts['unchanged']} unchanged, {counts['removed']} removed "
        f"(index: {index.index_dir})"
    )
    return 0


def _require_index(index: ResultsIndex) -> bool:
    if index.manifest() is None:
        print(
            f"No index at {index.index_dir}; run `python -m autofit.results index` "
            f"first.",
            file=sys.stderr,
        )
        return False
    return True


def command_status(arguments) -> int:
    index = _index(arguments)
    if not _require_index(index):
        return 1
    records = index.records()
    states = _states(arguments.state)
    if arguments.json:
        if states:
            records = [record for record in records if record.state in states]
            print(json.dumps([record.to_dict() for record in records], indent=1))
        else:
            print(json.dumps(index.counts(records), indent=1))
        return 0
    if states:
        for record in records:
            if record.state in states:
                reason = f"  ({record.reason})" if record.reason else ""
                print(f"{record.state:24s} {record.output_path}{reason}")
        return 0
    manifest = index.manifest()
    total = len(records)
    print(f"{total} results (index updated {manifest.get('updated')})")
    for state, count in index.counts(records).items():
        print(f"  {state:24s} {count}")
    mismatched = sum(1 for record in records if record.location_mismatch)
    if mismatched:
        print(f"  {'location_mismatch':24s} {mismatched} (declared path differs)")
    return 0


def command_count(arguments) -> int:
    index = _index(arguments)
    if not _require_index(index):
        return 1
    states = _states(arguments.state)
    records = index.records()
    if states:
        records = [record for record in records if record.state in states]
    print(len(records))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autofit.results",
        description="Index, classify and count PyAutoFit results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, function, help_):
        command = commands.add_parser(name, help=help_)
        command.add_argument("roots", nargs="+", help="Directories holding results.")
        command.add_argument(
            "--index-dir",
            default=None,
            help="Where the index is kept (default <root>/.autofit_index; required "
            "with several roots or a read-only tree).",
        )
        command.set_defaults(function=function)
        return command

    index = add("index", command_index, "Bring the index up to date.")
    index.add_argument(
        "--no-verify",
        action="store_true",
        help="Do not CRC-check new or changed zips.",
    )
    index.add_argument(
        "--stale-after-s",
        type=float,
        default=600.0,
        help="Age after which a 'running' status record is not trusted.",
    )

    for name, function, help_ in (
        ("status", command_status, "Results per state, or the results in a state."),
        ("count", command_count, "How many results (in a state)."),
    ):
        command = add(name, function, help_)
        command.add_argument(
            "--state", default=None, help="Comma-separated result states."
        )
        if name == "status":
            command.add_argument("--json", action="store_true")

    arguments = parser.parse_args(argv)
    return arguments.function(arguments)


if __name__ == "__main__":
    sys.exit(main())
