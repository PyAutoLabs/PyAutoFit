"""
``enrich``: add a new product to results that already exist, computed from each saved
result, without rerunning a fit.

``fn(search_output)`` returns the product's content, written to ``files/<name>``:

- a ``dict`` or ``list``: JSON (``files/<name>.json`` when ``name`` has no suffix);
- ``str`` or ``bytes``: the content as is;
- an ``os.PathLike``: the file at that path;
- ``None``: nothing (``fn`` declines this result).

A zipped result gets the member atomically (``write_zip_member``): every other member
keeps its content and a write killed part-way leaves the zip as it was. A result kept
as a folder gets the file written atomically. A result with both (the default, without
``remove_files``) gets both, the zip first; a product counts as present only when every
copy holds it, so rerunning a stopped pass finishes it.

Each write rewrites the whole zip, so two passes must not write to the same results at
the same time (two products at once, or a pass while those searches run): one of the
members would be lost. Run one pass after another, or give each its own shard.

With ``overwrite=False`` (the default) a result that already holds the product is left
alone, so an enrich pass that was stopped can simply be run again. ``shard`` (``"i/N"``
or ``(i, N)``) takes one of ``N`` disjoint parts of the results
(``autofit.results.shard``), so the pass can be shared between Slurm array tasks or a
process pool. A result whose ``fn`` or write fails is recorded in ``errors`` and the pass
goes on.

``enrich`` replaces running a search again with ``force_pickle_overwrite=True`` (and a
reload job on the cluster) to add a product such as ``coolest.json`` or ``wcs.json``.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from autofit.non_linear.status.states import Product
from autofit.results.shard import ShardLike, in_shard, parse_shard
from autofit.tools.util import NumpyEncoder, open_atomic, write_zip_member

logger = logging.getLogger(__name__)


@dataclass
class EnrichReport:
    name: str
    """The member written, ``files/<name>``."""
    written: List[str] = field(default_factory=list)
    """The output paths that got the product."""
    exists: List[str] = field(default_factory=list)
    """The output paths that already held it (``overwrite=False``)."""
    declined: List[str] = field(default_factory=list)
    """The output paths ``fn`` returned ``None`` for."""
    errors: List[Tuple[str, str]] = field(default_factory=list)
    """``(output_path, error)`` of every result whose ``fn`` or write failed."""
    skipped_shard: int = 0
    """Results in another shard."""

    def counts(self):
        return {
            "written": len(self.written),
            "exists": len(self.exists),
            "declined": len(self.declined),
            "errors": len(self.errors),
            "other_shards": self.skipped_shard,
        }

    def __str__(self):
        counts = self.counts()
        return f"enrich {self.name}: " + ", ".join(
            f"{count} {key}" for key, count in counts.items()
        )


def member_name(name: str) -> str:
    """
    The member ``files/<name>`` a product named ``name`` is written to.
    """
    name = str(name).strip("/")
    if not name or name.startswith(".") or ".." in Path(name).parts:
        raise ValueError(f"{name!r} is not a product name.")
    return f"files/{name}"


def _content(value, name: str) -> Tuple[Optional[str], Optional[bytes], Optional[Path]]:
    """
    ``(arcname, data, file_path)`` for the value ``fn`` returned.
    """
    arcname = member_name(name)
    if isinstance(value, (dict, list)):
        if not Path(arcname).suffix:
            arcname = f"{arcname}.json"
        data = json.dumps(value, indent=4, cls=NumpyEncoder).encode("utf-8")
        return arcname, data, None
    if isinstance(value, str):
        return arcname, value.encode("utf-8"), None
    if isinstance(value, (bytes, bytearray)):
        return arcname, bytes(value), None
    if isinstance(value, os.PathLike):
        return arcname, None, Path(value)
    raise TypeError(
        f"fn returned a {type(value).__name__}; return a dict, list, str, bytes, a "
        f"path or None."
    )


def _locations(search_output) -> Tuple[Optional[Path], Optional[Path]]:
    """
    ``(zip_path, folder)`` a search output is stored in (either may be ``None``).
    """
    from autofit.aggregator.zip_member import ZipMember

    directory = search_output.directory
    if isinstance(directory, ZipMember):
        # Without ``remove_files`` the folder stays beside the zip, and a rerun of the
        # search re-zips the folder: the product must go into both.
        zip_path = directory.zip_path
        folder = zip_path.with_name(zip_path.name[: -len(str(Product.ZIP))])
        return zip_path, (folder if folder.is_dir() else None)
    folder = Path(directory)
    zip_path = Path(f"{folder}{Product.ZIP}")
    return (zip_path if zip_path.exists() else None), folder


def _holds(zip_path: Optional[Path], folder: Optional[Path], arcname: str) -> bool:
    """
    Whether every copy of the result (zip and folder, where they exist) holds
    ``arcname``: a pass stopped between the two writes is finished by the next one.
    """
    from autofit.aggregator.zip_member import zip_names

    if zip_path is not None and arcname not in zip_names(zip_path):
        return False
    if folder is not None and not (folder / arcname).is_file():
        return False
    return zip_path is not None or folder is not None


def _write(zip_path: Optional[Path], folder: Optional[Path], arcname, data, file_path):
    # The zip first: it is what a ``remove_files`` resume restores from, and a folder
    # copy without one is finished by rerunning the pass (``_holds``).
    if zip_path is not None:
        write_zip_member(zip_path, arcname, data=data, file_path=file_path)
    if folder is not None and folder.is_dir():
        with open_atomic(folder / arcname, "wb") as f:
            if data is not None:
                f.write(data)
            else:
                f.write(Path(file_path).read_bytes())


def enrich(
    search_outputs,
    fn: Callable,
    name: str,
    overwrite: bool = False,
    shard: ShardLike = None,
) -> EnrichReport:
    """
    Add the product ``fn`` computes to each of ``search_outputs`` (see the module
    docstring).

    Parameters
    ----------
    fn
        ``fn(search_output)`` returns the product's content, or ``None`` to skip.
    name
        The product's name; it is written to ``files/<name>``.
    overwrite
        Rewrite the product where a result already holds it.
    shard
        ``"i/N"`` or ``(i, N)``: only the results in shard ``i`` of ``N``.
    """
    shard = parse_shard(shard)
    report = EnrichReport(name=member_name(name))
    for search_output in search_outputs:
        if getattr(search_output, "_temporary_directory", None) is not None:
            raise ValueError(
                "enrich cannot write to results extracted to a temporary directory "
                "(Aggregator.from_directory(unzip_temporary=True)); build the "
                "aggregator with Aggregator.from_index or from_directory instead."
            )
        zip_path, folder = _locations(search_output)
        output_path = str(folder if folder is not None else zip_path)
        if output_path.endswith(str(Product.ZIP)):
            output_path = output_path[: -len(str(Product.ZIP))]
        if not in_shard(output_path, shard):
            report.skipped_shard += 1
            continue
        try:
            arcname = member_name(name)
            if not overwrite and (
                _holds(zip_path, folder, arcname)
                or _holds(zip_path, folder, f"{arcname}.json")
            ):
                report.exists.append(output_path)
                continue
            value = fn(search_output)
            if value is None:
                report.declined.append(output_path)
                continue
            arcname, data, file_path = _content(value, name)
            report.name = arcname
            _write(zip_path, folder, arcname, data, file_path)
        except Exception as exception:
            logger.warning(f"enrich {name}: {output_path}: {exception}")
            report.errors.append((output_path, f"{type(exception).__name__}: {exception}"))
            continue
        report.written.append(output_path)
    logger.info(str(report))
    return report
