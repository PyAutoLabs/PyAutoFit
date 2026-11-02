"""
``python -m autofit.results``: index, classify and count results from the shell.

    python -m autofit.results index output/ [more_roots/ ...] [--index-dir DIR]
    python -m autofit.results status output/ [--index-dir DIR] [--state corrupt_zip]
    python -m autofit.results count output/ [--index-dir DIR] [--state completed]
    python -m autofit.results check library.yaml [--no-update] [--output-dir DIR]
    python -m autofit.results enrich output/ --fn module:function --name NAME \
        [--index-dir DIR] [--shard i/N] [--overwrite]

``index`` brings the index up to date (only new or changed results are opened);
``status`` prints one line per result state, or with ``--state`` the results in that
state; ``count`` prints a number. ``status`` and ``count`` read the existing index and
never open a result; run ``index`` first.

``check`` compares a declared library (``library.yaml``, see
``autofit.results.library``) with the index and writes ``<name>.missing.csv``; it
exits 1 when anything expected is not done. ``enrich`` adds the product
``function(search_output)`` returns to every completed result as ``files/<NAME>``
(``autofit.results.enrich``); ``--shard i/N`` takes one of N disjoint parts.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

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
        f"{counts['unchanged']} unchanged, {counts['removed']} removed, "
        f"{counts['unreadable']} unreadable (retried next time) "
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


def command_check(arguments) -> int:
    from autofit.results.library import ResultsLibrary

    # 1 means "something expected is not done", so every error is 2.
    try:
        library = ResultsLibrary.from_yaml(arguments.library)
        report = library.check(update=not arguments.no_update)
        path = report.write_missing_csv(
            arguments.output_dir or Path(arguments.library).parent
        )
    except Exception as exception:
        print(f"check failed: {type(exception).__name__}: {exception}", file=sys.stderr)
        return 2
    print(report.summary())
    print(f"gaps: {path}")
    return 0 if report.is_complete else 1


def _function(spec: str):
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"--fn is module:function, not {spec!r}.")
    function = importlib.import_module(module)
    for part in name.split("."):
        function = getattr(function, part)
    return function


def command_enrich(arguments) -> int:
    from autofit.aggregator.aggregator import Aggregator

    sys.path.insert(0, str(Path.cwd()))
    try:
        function = _function(arguments.fn)
    except (ImportError, AttributeError, ValueError) as exception:
        print(f"Cannot load {arguments.fn}: {exception}", file=sys.stderr)
        return 2
    try:
        aggregator = Aggregator.from_index(
            arguments.roots, index_dir=arguments.index_dir
        )
        report = aggregator.enrich(
            function,
            name=arguments.name,
            overwrite=arguments.overwrite,
            shard=arguments.shard,
        )
    except Exception as exception:
        print(f"enrich failed: {type(exception).__name__}: {exception}", file=sys.stderr)
        return 2
    print(report)
    for output_path, error in report.errors:
        print(f"  error {output_path}: {error}", file=sys.stderr)
    return 1 if report.errors else 0


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

    check = commands.add_parser(
        "check", help="Completeness of a declared library; writes <name>.missing.csv."
    )
    check.add_argument("library", help="The library.yaml.")
    check.add_argument(
        "--no-update",
        action="store_true",
        help="Use the index as it stands instead of updating it first.",
    )
    check.add_argument(
        "--output-dir",
        default=None,
        help="Where <name>.missing.csv goes (default: beside library.yaml).",
    )
    check.set_defaults(function=command_check)

    enrich = add("enrich", command_enrich, "Add a product to every completed result.")
    enrich.add_argument(
        "--fn", required=True, help="module:function, called with each search output."
    )
    enrich.add_argument("--name", required=True, help="Written to files/<name>.")
    enrich.add_argument("--shard", default=None, help="i/N: shard i of N.")
    enrich.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the product where a result already holds it.",
    )

    arguments = parser.parse_args(argv)
    return arguments.function(arguments)


if __name__ == "__main__":
    sys.exit(main())
