"""
The ``README.md`` and ``KNOWN_ISSUES.md`` of a package (``package``).

``README.md`` is regenerated on every build from ``package.json``: what is in the
package, how to check and unpack it, and its provenance (when, from which code, from
which results). ``KNOWN_ISSUES.md`` is written once as a template and then left to its
maintainer — it is where a release states what it knowingly gets wrong.
"""
from pathlib import Path

from autofit.tools.util import open_atomic

KNOWN_ISSUES_TEMPLATE = """# Known issues — {title}

List what this release knowingly gets wrong or leaves out, one line each, with the
objects affected and whether a later release fixes it. A reader should never have to
discover one of these from the data.

- (none recorded yet)
"""


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def write_readme(dist: Path, document: dict, title: str, notes: str = ""):
    zips = document["zips"]
    objects = document["objects"]
    provenance = document["provenance"]
    parts = sorted(name for name in zips if name.startswith("part_"))
    total = sum(entry["bytes"] for entry in zips.values())
    csv_rows = {
        name: rows
        for entry in zips.values()
        for name, rows in (entry.get("csv_rows") or {}).items()
    }

    lines = [
        f"# {title}",
        "",
        f"{len(objects)} objects in {len(parts)} part zips, plus `tables.zip` "
        f"({_size(total)} in all).",
        "",
    ]
    if notes:
        lines += [notes.strip(), ""]
    lines += [
        "## Contents",
        "",
        "| file | holds | size |",
        "|---|---|---|",
        f"| `tables.zip` | the tables ({len(zips['tables.zip']['members'])} files) | "
        f"{_size(zips['tables.zip']['bytes'])} |",
    ]
    for name in parts:
        n = sum(1 for part in objects.values() if part == name)
        lines.append(f"| `{name}` | {n} objects | {_size(zips[name]['bytes'])} |")
    lines += [
        "| `parts_manifest.csv` | which part holds each object | |",
        "| `package.json` | members, checksums and row counts, for `verify` | |",
        "| `SHA256SUMS` | the checksum of every file above | |",
        "",
    ]
    if csv_rows:
        lines += ["## Tables", "", "| table | rows |", "|---|---|"]
        lines += [f"| `{name}` | {rows} |" for name, rows in sorted(csv_rows.items())]
        lines.append("")
    lines += [
        "## Check and unpack",
        "",
        "```bash",
        "sha256sum -c SHA256SUMS",
        "python -m autofit.export verify .   # checksums, CRCs, empty files, table rows",
        "for z in *.zip; do unzip -q \"$z\"; done",
        "```",
        "",
        "Only the parts holding the objects you need are required: `parts_manifest.csv`",
        "names each object's part.",
        "",
        "## Provenance",
        "",
        f"- built: {provenance.get('created')}",
        f"- autofit: {provenance.get('autofit_version')}",
    ]
    if provenance.get("index_hash"):
        lines.append(f"- results index: `{provenance['index_hash']}`")
    for repo in provenance.get("repos") or []:
        dirty = " (uncommitted changes)" if repo.get("dirty") else ""
        lines.append(f"- `{Path(repo['repo']).name}`: `{repo.get('revision')}`{dirty}")
    lines += ["", "See `KNOWN_ISSUES.md` for what this release knowingly gets wrong.", ""]

    with open_atomic(dist / "README.md", "w") as f:
        f.write("\n".join(lines))
    known_issues = dist / "KNOWN_ISSUES.md"
    if not known_issues.exists():
        known_issues.write_text(KNOWN_ISSUES_TEMPLATE.format(title=title))
