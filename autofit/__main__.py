"""
Command-line entry point: ``python -m autofit <command>``.

Commands
--------
search-manifest [--json]
    Print the declarative search registry. ``--json`` emits the versioned manifest
    (``search-manifest@1``), the only cross-repo format for search facts; without it a
    plain-text table is printed.
"""

import argparse
import json
import sys


def _search_manifest(arguments) -> int:
    from autofit.non_linear.search.registry import manifest

    data = manifest()

    if arguments.json:
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"{data['schema']} (autofit {data['autofit_version']})")
    for search in data["searches"]:
        capabilities = search["capabilities"]
        print(
            f"{search['name']:<20} {search['family']:<5} "
            f"jax={capabilities['jax_use']:<9} gradient={capabilities['gradient']:<5} "
            f"posterior={capabilities['posterior_kind']:<9} "
            f"evidence={capabilities['produces_evidence']!s:<5} "
            f"status={capabilities['status']}"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autofit")
    commands = parser.add_subparsers(dest="command", required=True)

    search_manifest = commands.add_parser(
        "search-manifest",
        help="Print the declarative search registry (search-manifest@1).",
    )
    search_manifest.add_argument(
        "--json", action="store_true", help="Emit the versioned JSON manifest."
    )
    search_manifest.set_defaults(handler=_search_manifest)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
