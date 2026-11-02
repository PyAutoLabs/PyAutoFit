"""
``verify``: check a package (``package``) is whole before it is published or after it
is downloaded — the questions a reader of the release cannot answer by eye.

1. every file in ``SHA256SUMS`` exists and matches its checksum, and every zip
   ``package.json`` names is in ``SHA256SUMS``;
2. every zip passes its CRC check and holds exactly the members ``package.json``
   records;
3. no member is empty (a zero-byte FITS or CSV is a failed write, not data), except
   names matched by ``allow_empty``;
4. every CSV member parses, every row has as many fields as the header, and its row
   count is the one recorded when the package was built;
5. every object is in exactly one part as the zips actually hold them — the part
   ``package.json`` and ``parts_manifest.csv`` both name.
"""
import csv
import fnmatch
import json
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from autofit.export.package import MANIFEST, PACKAGE_JSON, SUMS, csv_records, sha256


@dataclass
class VerifyReport:
    dist: Path
    problems: List[str] = field(default_factory=list)
    zips: int = 0
    members: int = 0
    csv_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self):
        head = (
            f"verify {self.dist}: {'OK' if self.ok else 'FAILED'} — {self.zips} zips, "
            f"{self.members} members, {self.csv_rows} table rows checked"
        )
        return "\n".join([head] + [f"  {problem}" for problem in self.problems])


def verify(dist, allow_empty: Sequence[str] = (".completed", "*/.completed")) -> VerifyReport:
    """
    Check the package in ``dist`` (see the module docstring).

    Parameters
    ----------
    allow_empty
        ``fnmatch`` patterns of member names allowed to be empty (marker files).
    """
    dist = Path(dist)
    report = VerifyReport(dist=dist)
    problems = report.problems

    try:
        document = json.loads((dist / PACKAGE_JSON).read_text())
    except (OSError, ValueError) as exception:
        problems.append(f"{PACKAGE_JSON} cannot be read: {exception}")
        return report

    sums = {}
    try:
        for number, line in enumerate((dist / SUMS).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                problems.append(f"{SUMS} line {number} is not '<sha256>  <file>': {line!r}")
                continue
            sums[fields[1].strip().lstrip("*")] = fields[0]
    except OSError as exception:
        problems.append(f"{SUMS} cannot be read: {exception}")
    for name, digest in sorted(sums.items()):
        path = dist / name
        if not path.exists():
            problems.append(f"{name}: listed in {SUMS} but missing")
        elif sha256(path) != digest:
            problems.append(f"{name}: SHA256 does not match {SUMS}")

    zips = document.get("zips") or {}
    root = document.get("root")
    seen = {}
    for name, entry in sorted(zips.items()):
        if name not in sums:
            problems.append(f"{name}: not listed in {SUMS}")
        path = dist / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        report.zips += 1
        try:
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad is not None:
                    problems.append(f"{name}: CRC failure on {bad}")
                infos = [info for info in archive.infolist() if not info.is_dir()]
                names = sorted(info.filename for info in infos)
                if name.startswith("part_"):
                    for member in names:
                        parts = member.split("/")
                        if len(parts) >= 3 and parts[0] == root:
                            seen.setdefault(parts[1], set()).add(name)
                expected = sorted(member[0] for member in entry.get("members") or [])
                if names != expected:
                    extra = sorted(set(names) - set(expected))[:3]
                    missing = sorted(set(expected) - set(names))[:3]
                    problems.append(
                        f"{name}: members differ from {PACKAGE_JSON} "
                        f"(extra {extra}, missing {missing})"
                    )
                csv_rows = entry.get("csv_rows") or {}
                for info in infos:
                    report.members += 1
                    if info.file_size == 0 and not any(
                        fnmatch.fnmatch(info.filename, pattern) for pattern in allow_empty
                    ):
                        problems.append(f"{name}: {info.filename} is empty")
                    if info.filename.endswith(".csv"):
                        problems += _check_csv(archive, info, name, csv_rows, report)
        except (zipfile.BadZipFile, OSError, zlib.error, EOFError, RuntimeError) as exception:
            problems.append(f"{name}: unreadable zip: {type(exception).__name__}: {exception}")

    for path in sorted(dist.glob("*.zip")):
        if path.name not in zips:
            problems.append(
                f"{path.name}: a zip {PACKAGE_JSON} does not list (stale or foreign); "
                f"unpacking every *.zip would include it"
            )

    objects = document.get("objects") or {}
    for item, part in sorted(objects.items()):
        holders = seen.get(item, set())
        if holders != {part}:
            problems.append(
                f"{item}: {PACKAGE_JSON} names {part}, found in {sorted(holders) or 'no part'}"
            )
    for item in sorted(set(seen) - set(objects)):
        problems.append(f"{item}: in {sorted(seen[item])} but not in {PACKAGE_JSON}")
    try:
        with open(dist / MANIFEST, newline="") as f:
            manifest = {row["object"]: row["part_zip"] for row in csv.DictReader(f)}
        if manifest != objects:
            differ = sorted(
                item
                for item in set(manifest) | set(objects)
                if manifest.get(item) != objects.get(item)
            )
            problems.append(f"{MANIFEST} disagrees with {PACKAGE_JSON} for {differ[:5]}")
    except (OSError, KeyError) as exception:
        problems.append(f"{MANIFEST} cannot be read: {exception}")

    return report


def _check_csv(archive, info, name, csv_rows, report) -> List[str]:
    problems = []
    try:
        with archive.open(info) as handle:
            rows = csv_records(handle)
    except csv.Error as exception:
        return [f"{name}: {info.filename} does not parse: {exception}"]
    if not rows:
        return problems
    width = len(rows[0])
    ragged = [i + 1 for i, row in enumerate(rows[1:], start=1) if len(row) != width]
    if ragged:
        problems.append(
            f"{name}: {info.filename} rows {ragged[:5]} do not have {width} fields"
        )
    n = len(rows) - 1
    report.csv_rows += n
    expected = csv_rows.get(info.filename)
    if expected is not None and expected != n:
        problems.append(
            f"{name}: {info.filename} has {n} rows, {PACKAGE_JSON} records {expected}"
        )
    return problems
