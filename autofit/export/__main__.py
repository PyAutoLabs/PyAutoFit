"""
``python -m autofit.export``: package, verify and compare from the shell.

    python -m autofit.export package LIBRARY DIST [--cap-gb 10 | --n-parts 5] [--title T]
        [--repo PATH ...] [--replan]
    python -m autofit.export verify DIST
    python -m autofit.export compare A.csv B.csv [--key id] [--exact col,...]
        [--gate col,...] [--threshold 3] [--floor 0] [--rel-floor 0] [--report parity.md]

``verify`` and ``compare`` exit 0 when the package is whole / the tables agree, 1
when they do not, and 2 on an error.
"""
import argparse
import sys


def _list(value):
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def command_package(arguments) -> int:
    from autofit.export.package import package

    try:
        report = package(
            arguments.library,
            arguments.dist,
            cap_gb=arguments.cap_gb,
            n_parts=arguments.n_parts,
            title=arguments.title,
            repos=arguments.repo or (),
            replan=arguments.replan,
        )
    except Exception as exception:
        print(f"package failed: {type(exception).__name__}: {exception}", file=sys.stderr)
        return 2
    print(report)
    return 0


def command_verify(arguments) -> int:
    from autofit.export.verify import verify

    try:
        report = verify(arguments.dist)
    except Exception as exception:
        print(f"verify failed: {type(exception).__name__}: {exception}", file=sys.stderr)
        return 2
    print(report)
    return 0 if report.ok else 1


def command_compare(arguments) -> int:
    from autofit.export.parity import compare

    try:
        report = compare(
            arguments.a,
            arguments.b,
            key=_list(arguments.key),
            exact=_list(arguments.exact),
            gate=_list(arguments.gate),
            threshold=arguments.threshold,
            floor=arguments.floor,
            rel_floor=arguments.rel_floor,
        )
        markdown = report.to_markdown()
        if arguments.report:
            with open(arguments.report, "w") as f:
                f.write(markdown)
    except Exception as exception:
        print(f"compare failed: {type(exception).__name__}: {exception}", file=sys.stderr)
        return 2
    print(markdown)
    return 0 if report.passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m autofit.export",
        description="Package, verify and compare PyAutoFit results libraries.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    package = commands.add_parser("package", help="Package a library folder.")
    package.add_argument("library")
    package.add_argument("dist")
    size = package.add_mutually_exclusive_group()
    size.add_argument("--cap-gb", type=float, default=None)
    size.add_argument("--n-parts", type=int, default=None)
    package.add_argument("--title", default=None)
    package.add_argument("--repo", action="append", help="A git checkout to record.")
    package.add_argument("--replan", action="store_true")
    package.set_defaults(function=command_package)

    verify = commands.add_parser("verify", help="Check a package is whole.")
    verify.add_argument("dist")
    verify.set_defaults(function=command_verify)

    compare = commands.add_parser("compare", help="Diff two results tables.")
    compare.add_argument("a", help="The table under test.")
    compare.add_argument("b", help="The reference table.")
    compare.add_argument("--key", default="id", help="Comma-separated key columns.")
    compare.add_argument("--exact", default="", help="Columns that must be equal.")
    compare.add_argument("--gate", default="", help="Value columns that gate.")
    compare.add_argument("--threshold", type=float, default=3.0)
    compare.add_argument("--floor", type=float, default=0.0)
    compare.add_argument("--rel-floor", type=float, default=0.0)
    compare.add_argument("--report", default=None, help="Write the markdown here.")
    compare.set_defaults(function=command_compare)

    arguments = parser.parse_args(argv)
    return arguments.function(arguments)


if __name__ == "__main__":
    sys.exit(main())
