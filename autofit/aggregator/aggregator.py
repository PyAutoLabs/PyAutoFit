#!/usr/bin/env python

"""
Filter and collate search outputs in all subdirectories.

Usage:

./aggregator.py (root_directory) [pipeline=pipeline phase=phase dataset=dataset]

Example:

./aggregator.py ../output pipeline=data_mass_x1_source_x1_positions
"""

import logging
import os
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from shutil import rmtree
from typing import List, Union, Iterator, Optional, Tuple

from autonerves.test_mode import is_test_mode

from .predicate import AttributePredicate
from .search_output import (
    SearchOutput,
    GridSearchOutput,
    GridSearch,
    TemporaryExtraction,
)

logger = logging.getLogger(__name__)

ON_ERROR_VALUES = ("skip", "raise")
"""
What ``Aggregator.from_directory`` does with an unreadable zip: ``"skip"`` it and
record it in ``Aggregator.errors``, or ``"raise"``.
"""


def _check_on_error(on_error: str):
    if on_error not in ON_ERROR_VALUES:
        raise ValueError(
            f"on_error must be one of {ON_ERROR_VALUES}, not {on_error!r}."
        )


ZIP_READ_ERRORS = (zipfile.BadZipFile, EOFError, zlib.error)
"""
What reading one damaged result zip raises: ``BadZipFile`` for a missing or bad
central directory or a CRC mismatch, ``EOFError`` and ``zlib.error`` for a member
truncated or corrupted part-way through.
"""


def _bad_zip_error(zip_path: Path, error: Exception) -> zipfile.BadZipFile:
    return zipfile.BadZipFile(
        f"File is not a readable zip file ({type(error).__name__}: {error}): \n "
        f"{zip_path.parent} \n"
        f"{zip_path.name}"
    )


class AggregatorGroup:
    def __init__(self, groups: ["AggregatorGroup"]):
        """
        A group of aggregators produced by grouping search_outputs on a field.

        Parameters
        ----------
        groups
            Groups, each with a common value in the metadata file
        """
        self.groups = groups

    def filter(self, *predicates) -> "AggregatorGroup":
        """
        Apply predicates to the underlying groups whilst maintaining the total number of groups.

        Parameters
        ----------
        predicates
            Predicates that evaluate to `True` or `False` for any given phase.

        Returns
        -------
        A collection of groups of the same length with each group having the same or fewer members.
        """
        return AggregatorGroup([group.filter(*predicates) for group in self.groups])

    def __getitem__(self, item):
        return self.groups[item]

    def __len__(self):
        return len(self.groups)

    def values(self, name: str, parser=lambda o: o) -> List[List]:
        """
        Extract a list of lists values with a certain name from the output objects in
        this group.

        Parameters
        ----------
        name
            The name of the attribute to be extracted
        parser
            A function used to parse the output

        Returns
        -------
        A list of lists of values.
        """
        return [group.values(name, parser=parser) for group in self.groups]


def unzip_directory(
    directory: str, on_error: str = "raise"
) -> List[Tuple[Path, Exception]]:
    """
    Unzip all zip files in a directory recursively.

    Parameters
    ----------
    directory
        The directory searched recursively for zips.
    on_error
        ``"raise"`` (the default) raises on the first unreadable zip; ``"skip"``
        logs it, carries on, and returns it among the ``(path, exception)`` pairs.

    Returns
    -------
    The ``(path, exception)`` of every zip that could not be read.
    """
    _check_on_error(on_error)

    errors = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".zip"):
                zip_path = Path(root) / filename
                try:
                    with zipfile.ZipFile(zip_path, "r") as f:
                        f.extractall(Path(root) / filename[:-4])
                except ZIP_READ_ERRORS as e:
                    if on_error == "raise":
                        raise _bad_zip_error(zip_path, e) from e
                    logger.warning(f"Skipping unreadable zip {zip_path}: {e}")
                    errors.append((zip_path, e))

    return errors


def _is_search_output(root, dirs, filenames) -> bool:
    """
    Whether an ``os.walk`` step has landed on a search output directory.

    The sentinel is ``files/search.json``, which ``save_all`` always writes; the
    ``"files" in dirs`` guard keeps the walk from returning the ``files``
    directory itself as a search output. A bare ``metadata`` file is still
    accepted so output folders (and archived zips) written before the sentinel
    changed continue to aggregate.
    """
    if "files" in dirs and (Path(root) / "files" / "search.json").exists():
        return True
    return "metadata" in filenames


def is_relative_to(path_a, path_b):
    """Return True if the path is relative to another path or False."""
    try:
        path_a.relative_to(path_b)
        return True
    except ValueError:
        return False


class Aggregator:
    def __init__(
        self,
        search_outputs: List[SearchOutput],
        grid_search_outputs: List[GridSearchOutput],
        temporary_directory: Optional[TemporaryExtraction] = None,
        errors: Optional[List[Tuple[Path, Exception]]] = None,
    ):
        """
        Class to aggregate phase results for all subdirectories in a given directory.

        Parameters
        ----------
        search_outputs
            A list of search_outputs
        temporary_directory
            The temporary extraction zipped outputs were unpacked into, if
            ``from_directory`` was called with ``unzip_temporary=True``. Held so
            ``close`` can remove it eagerly; the search outputs keep it alive
            regardless.
        errors
            The ``(path, exception)`` of every output ``from_directory`` could not
            read and skipped (``on_error="skip"``), exposed as ``errors``.
        """
        if len(search_outputs) > 20:
            logger.info(
                "When aggregating many searches it can be more efficient to use the database.\n"
                "Checkout the database cookbook at this URL: "
                "https://pyautofit.readthedocs.io/en/latest/cookbooks/database.html"
            )
        self.search_outputs = search_outputs
        self.grid_search_outputs = grid_search_outputs
        # Set here rather than lazily: ``__getattr__`` returns an
        # ``AttributePredicate`` for any name it does not find.
        self._temporary_directory = temporary_directory
        self.errors = list(errors or [])

    def grid_searches(self):
        """
        A list of grid search outputs
        """
        return [
            GridSearch(
                output,
                [
                    search_output
                    for search_output in self.search_outputs
                    if is_relative_to(search_output.directory, output.directory)
                ],
            )
            for output in self.grid_search_outputs
        ]

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, os.PathLike],
        completed_only=False,
        reference: Optional[dict] = None,
        unzip_temporary: bool = False,
        on_error: str = "skip",
    ) -> "Aggregator":
        """
        Aggregate phase results for all subdirectories in a given directory.

        The whole directory structure is traversed and a Phase object created for each directory that contains a
        ``files/search.json`` file (or, for legacy output, a ``metadata`` file).

        Zipped search outputs are extracted as they are encountered, in the same traversal. A zip whose
        extracted directory already exists is skipped, so repeat calls do not pay the extraction cost again;
        delete the extracted directory to force re-extraction.

        By default the extraction is written next to the zip and left there, which doubles the disk a results
        tree occupies. Pass ``unzip_temporary=True`` to extract into a temporary directory instead, which is
        removed once the aggregator and its search outputs are released (or immediately, via ``close``).

        Parameters
        ----------
        directory
            A directory in which the outputs of search_outputs are kept. This is searched recursively.
        completed_only
            If `True` only search_outputs with a .completed file (indicating the phase was completed)
            are included in the aggregator.
        reference
            A dictionary mapping paths to types to be used when loading models from disk.
        unzip_temporary
            If `True` zips are extracted into a temporary directory that is cleaned up automatically,
            leaving the aggregated directory untouched. A zip whose extracted directory already sits
            beside it and carries a ``.completed`` file still uses that directory and is not extracted
            again; one whose sibling directory has no ``.completed`` file is extracted, because such a
            directory is written after the search was archived and does not stand for the search.
        on_error
            What to do with a zip that cannot be read (``ZIP_READ_ERRORS``): ``"skip"`` (the default) logs
            a warning, leaves it out and records ``(path, exception)`` in the returned aggregator's
            ``errors``, so one corrupt archive does not hide every other result; ``"raise"`` raises.

        Returns
        -------
        An aggregator whose search outputs are ordered by path, so that the same results tree gives
        the same order on every machine regardless of the filesystem's directory-entry order.
        """
        _check_on_error(on_error)

        logger.info("Aggregator loading search_outputs... could take some time.")

        errors = []
        temporary_directory = None
        temporary_roots = []
        incomplete_count = 0

        def temporary_path(zip_path: Path, scan_root: Path) -> Path:
            """
            Where to extract ``zip_path`` inside the shared temporary directory.

            The scanned layout is mirrored so that two zips of the same name in
            different directories cannot collide.
            """
            nonlocal temporary_directory
            if temporary_directory is None:
                temporary_directory = TemporaryExtraction()
            relative = zip_path.parent.relative_to(scan_root)
            return temporary_directory.path / relative / zip_path.stem

        def scan(scan_directory, extract_temporary=False, owner=None):
            """
            Walk ``scan_directory``, extracting zips and collecting search outputs.

            ``extract_temporary`` sends extractions to the shared temporary
            directory rather than beside the zip; ``owner`` is the temporary
            extraction the outputs found here should keep alive.
            """
            scan_directory = Path(scan_directory)
            search_outputs = []
            grid_search_outputs = []

            nonlocal incomplete_count

            for root, dirs, filenames in os.walk(scan_directory, topdown=True):
                for filename in filenames:
                    if filename.endswith(".zip"):
                        zip_path = Path(root) / filename
                        sibling = Path(root) / filename[:-4]
                        extracted = sibling
                        if sibling.exists():
                            if (sibling / ".completed").exists():
                                continue
                            # A directory beside the zip that carries no
                            # ``.completed`` is not the extraction of a finished
                            # search: post-completion writers (cache artifacts
                            # derived from a finished result) recreate
                            # ``<search>/files/`` after the real directory was
                            # zipped and removed. Preferring it over the zip
                            # silently loses the search under ``completed_only``,
                            # so the zip wins and its contents are authoritative.
                            logger.warning(
                                f"Aggregator: {sibling} has no .completed file but "
                                f"{zip_path.name} does; the zip was used. The "
                                f"directory holds files written after the search "
                                f"was archived."
                            )
                        if extract_temporary:
                            extracted = temporary_path(zip_path, scan_directory)
                        existed = extracted.exists()
                        try:
                            with zipfile.ZipFile(zip_path, "r") as f:
                                f.extractall(extracted)
                        except ZIP_READ_ERRORS as e:
                            if on_error == "raise":
                                raise _bad_zip_error(zip_path, e) from e
                            logger.warning(
                                f"Aggregator: skipping unreadable zip {zip_path}: {e}"
                            )
                            errors.append((zip_path, e))
                            if not existed:
                                # A partial extraction is not a search output.
                                rmtree(extracted, ignore_errors=True)
                            continue
                        if extract_temporary:
                            # Outside the tree being walked, so it is scanned separately.
                            temporary_roots.append(extracted)
                            if filename[:-4] in dirs:
                                # The incomplete sibling is superseded by the
                                # temporary extraction; walking it too would
                                # yield the same search a second time.
                                dirs.remove(filename[:-4])
                        elif filename[:-4] not in dirs:
                            dirs.append(filename[:-4])

                def should_add():
                    return not completed_only or ".completed" in filenames

                if _is_search_output(root, dirs, filenames):
                    if should_add():
                        search_outputs.append(
                            SearchOutput(
                                Path(root),
                                reference=reference,
                                temporary_directory=owner,
                            )
                        )
                    else:
                        incomplete_count += 1
                if ".is_grid_search" in filenames:
                    if should_add():
                        grid_search_outputs.append(
                            GridSearchOutput(
                                Path(root),
                                temporary_directory=owner,
                            )
                        )

            return search_outputs, grid_search_outputs

        def scan_all(scan_directory):
            """
            Scan a directory and then every temporary extraction it produced.

            Temporary extractions sit outside the walked tree, so each is scanned in
            turn. Any zip nested inside one extracts beside itself, which is still
            inside the temporary directory and removed with it.

            The collected outputs are sorted by path before they are returned.
            ``os.walk`` yields directory entries in whatever order the filesystem
            reports — hash order on ext4, insertion order on tmpfs — so without
            this an aggregator's outputs come back in a machine-dependent order,
            and anything that pairs a row to an output by index (the summary CSV
            writers, ``add_label_column``) pairs them differently on different
            machines. The sort is applied to the collected lists rather than to
            ``dirs`` inside the walk because the temporary extractions are
            scanned separately and so never pass through that walk.
            """
            search_outputs, grid_search_outputs = scan(
                scan_directory,
                extract_temporary=unzip_temporary,
            )
            while temporary_roots:
                temporary_root = temporary_roots.pop()
                extra_search_outputs, extra_grid_search_outputs = scan(
                    temporary_root,
                    owner=temporary_directory,
                )
                search_outputs.extend(extra_search_outputs)
                grid_search_outputs.extend(extra_grid_search_outputs)

            search_outputs.sort(key=lambda output: str(output.directory))
            grid_search_outputs.sort(key=lambda output: str(output.directory))

            return search_outputs, grid_search_outputs

        search_outputs, grid_search_outputs = scan_all(directory)

        # Under test mode the searches wrote their results beneath an inserted
        # ``test_mode`` segment (``output/test_mode/<prefix>``) rather than the
        # real-run location the caller points at (``output/<prefix>``); see
        # ``_test_mode_segment`` in ``non_linear/paths/abstract.py``. If nothing
        # was found there, retry once against that sibling so aggregator-based
        # tutorials (e.g. features/interpolate) run cleanly under test mode.
        if len(search_outputs) == 0 and is_test_mode():
            directory = Path(directory)
            test_mode_directory = directory.parent / "test_mode" / directory.name
            if test_mode_directory.exists():
                incomplete_count = 0
                search_outputs, grid_search_outputs = scan_all(test_mode_directory)

        if len(search_outputs) == 0:
            # Loud: an empty aggregator otherwise flows on into empty tables and
            # plots that look like a result.
            logger.warning(
                f"Aggregator found no search outputs in {directory} "
                f"(completed_only={completed_only}, unzip_temporary={unzip_temporary}, "
                f"excluded as incomplete={incomplete_count}, unreadable={len(errors)})."
            )
        else:
            logger.info(
                f"A total of {str(len(search_outputs))} search_outputs and results were found."
            )

        # A search dropped for having no ``.completed`` file is invisible in the
        # count above, so a run that aggregates a fraction of its searches looks
        # like a complete one. Say how many were left out.
        if completed_only and incomplete_count > 0:
            logger.warning(
                f"{incomplete_count} further search_outputs were excluded because "
                f"they have no .completed file."
            )

        if errors:
            logger.warning(
                f"{len(errors)} unreadable output(s) were skipped; see Aggregator.errors."
            )

        return cls(
            search_outputs,
            grid_search_outputs,
            temporary_directory=temporary_directory,
            errors=errors,
        )

    @classmethod
    def from_index(
        cls,
        roots: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
        index_dir: Optional[Union[str, os.PathLike]] = None,
        states: Tuple[str, ...] = ("completed",),
        reference: Optional[dict] = None,
        update: bool = True,
    ) -> "Aggregator":
        """
        Aggregate the results a results index (``autofit.results.ResultsIndex``) lists,
        reading zipped results in place: nothing is extracted to disk.

        Parameters
        ----------
        roots
            The directory (or directories) holding the results.
        index_dir
            Where the index is kept (default ``<root>/.autofit_index``; required with
            several roots or a read-only tree).
        states
            The result states to include. The default, ``("completed",)``, leaves out
            superseded results (an older sibling of a newer completed fit); add
            ``"superseded"`` to keep them.
        reference
            A dictionary mapping paths to types to be used when loading models from disk.
        update
            Bring the index up to date first (only new or changed results are opened).
            ``False`` uses the index as it stands.
        """
        from autofit.results import ResultsIndex

        index = ResultsIndex(roots, index_dir=index_dir)
        if update:
            index.update()

        states = {str(state) for state in states}
        search_outputs = []
        for record in index.records():
            if record.state not in states:
                continue
            output_path = Path(record.output_path)
            if record.has_zip and record.state != "corrupt_zip":
                search_outputs.append(
                    SearchOutput.from_zip(f"{output_path}.zip", reference=reference)
                )
            elif record.has_dir:
                search_outputs.append(SearchOutput(output_path, reference=reference))

        logger.info(
            f"Aggregator.from_index: {len(search_outputs)} search outputs in states "
            f"{sorted(states)}."
        )
        if not search_outputs:
            logger.warning(
                f"Aggregator.from_index found no search outputs in states "
                f"{sorted(states)} (index: {index.index_dir}, counts: {index.counts()})."
            )
        return cls(search_outputs, [])

    def add_directory(
        self,
        directory: Union[str, Path],
        unzip_temporary: bool = False,
        on_error: str = "skip",
    ):
        """
        Add a directory to the aggregator.

        Parameters
        ----------
        directory
            A directory searched recursively for search outputs.
        unzip_temporary
            If `True` zips are extracted into a temporary directory that is cleaned
            up automatically, as in ``from_directory``. The added search outputs
            keep that directory alive; ``close`` does not remove it.
        on_error
            As in ``from_directory``; skipped outputs are added to ``errors``.
        """
        aggregator = Aggregator.from_directory(
            directory,
            unzip_temporary=unzip_temporary,
            on_error=on_error,
        )
        self.search_outputs.extend(aggregator.search_outputs)
        self.grid_search_outputs.extend(aggregator.grid_search_outputs)
        self.errors.extend(aggregator.errors)

    def close(self):
        """
        Remove the temporary directory zipped outputs were extracted into, if there
        is one, without waiting for garbage collection.

        Search outputs read out of it cannot be loaded afterwards. Only the
        aggregator returned by ``from_directory`` holds the directory; one produced
        by slicing or querying it does not, so ``close`` there does nothing.
        """
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> "Aggregator":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def remove_unzipped(self):
        """
        Removes the unzipped output directory for each phase.

        Only relevant to the default extraction mode, which unpacks each zip beside
        itself and leaves it there. Outputs extracted with ``unzip_temporary=True``
        are cleaned up on their own.
        """
        for phase in self.search_outputs:
            split_path = Path(phase.directory).parent

            rmtree(split_path, ignore_errors=True)

    def __getitem__(self, item: Union[slice, int]) -> Union["Aggregator", SearchOutput]:
        """
        If an index is passed in then a specific phase output is returned.

        If a slice is passed in then an aggregator comprising several search_outputs is returned.

        Parameters
        ----------
        item
            A slice or index

        Returns
        -------
        An aggregator or phase
        """
        if isinstance(item, slice):
            return Aggregator(
                self.search_outputs[item],
                self.grid_search_outputs,
                errors=self.errors,
            )
        return self.search_outputs[item]

    def __len__(self):
        return len(self.search_outputs)

    def __iter__(self):
        return iter(self.search_outputs)

    def __getattr__(self, item):
        return AttributePredicate(item)

    def query(self, *predicates) -> "Aggregator":
        """
        Filter phase outputs by predicates. A predicate is created using a conditional
        operator.

        Another aggregator object is returned.

        Parameters
        ----------
        predicates
            Objects representing predicates that may evaluate to `True` or `False` for any
            given phase output.

        Returns
        -------
        An aggregator comprising all search_outputs that evaluate to `True` for all predicates.
        """
        search_outputs = self.search_outputs
        for predicate in predicates:
            search_outputs = predicate.filter(search_outputs)
        search_outputs = list(search_outputs)
        logger.info(f"Filter found a total of {str(len(search_outputs))} results")
        return Aggregator(
            search_outputs=list(search_outputs),
            grid_search_outputs=self.grid_search_outputs,
            errors=self.errors,
        )

    def latest_per(
        self,
        key: Tuple[str, ...] = ("path_prefix", "unique_tag", "name"),
        by: str = "completed_at",
    ) -> "Aggregator":
        """
        Keep, for each value of ``key``, only the search output with the greatest
        ``by``: the latest of several fits of the same search to the same object
        (sibling hashes, e.g. a rerun after a model or search setting changed).

        ``completed_at`` comes from the status record or the ``.completed`` member of
        the result's zip, never the zip's modification time, which copying or
        re-zipping resets. Outputs with no ``by`` value lose to any that have one;
        remaining ties go to the greater path, so the choice is the same on every
        machine.

        Parameters
        ----------
        key
            Attributes of a search output that together name one fit of one object.
            The default suits both layouts in use: the object in ``path_prefix``
            (``unique_tag`` shared) or in ``unique_tag`` (``path_prefix`` shared).
        by
            The attribute that orders the outputs within a key.
        """

        def value_of(search_output, attribute):
            value = getattr(search_output, attribute, None)
            return None if value is None else str(value)

        latest = {}
        for search_output in self.search_outputs:
            group = tuple(value_of(search_output, k) for k in key)
            order = (
                value_of(search_output, by) is not None,
                value_of(search_output, by) or "",
                str(search_output.directory),
            )
            if group not in latest or order > latest[group][0]:
                latest[group] = (order, search_output)

        search_outputs = sorted(
            (search_output for _, search_output in latest.values()),
            key=lambda output: str(output.directory),
        )
        logger.info(
            f"latest_per kept {len(search_outputs)} of {len(self.search_outputs)} "
            f"search outputs."
        )
        return Aggregator(
            search_outputs=search_outputs,
            grid_search_outputs=self.grid_search_outputs,
            errors=self.errors,
        )

    def values(self, name: str, parser=lambda o: o) -> Iterator:
        """
        Get values from outputs with a given name.

        A list the same length as the number of search_outputs is returned
        where each item is the value of the attribute for a given
        phase.

        Parameters
        ----------
        name
            The name of an attribute expected to be associated with
            phase output. If a pickle file with this name is in the
            phase output directory then that pickle will be loaded.
        parser
            A function used to parse the output

        Returns
        -------
        A generator of values for the attribute
        """
        for value in map(lambda phase: getattr(phase, name), self.search_outputs):
            yield parser(value)

    def child_values(self, name: str) -> Iterator[List]:
        """
        Get values with a given name from the child analyses of each search in
        this aggregator.

        Parameters
        ----------
        name
            The name of an attribute expected to be associated with
            child analysis output. If a pickle file with this name
            is in the child analysis output directory then that pickle
            will be loaded.

        Returns
        -------
        A generator of values for the attribute
        """
        return (phase.child_values(name) for phase in self.search_outputs)

    def map(self, func):
        """
        Map some function onto the aggregated output objects.

        Parameters
        ----------
        func
            A function

        Returns
        -------
        A generator of results
        """
        return map(func, self.search_outputs)

    def group_by(self, field: str) -> AggregatorGroup:
        """
        Group the search_outputs by a field, e.g. pipeline.

        The object returned still permits filtering and attribute querying.

        Parameters
        ----------
        field
            The field by which to group

        Returns
        -------
        An object comprising lists of grouped fields
        """
        group_dict = defaultdict(list)
        for phase in self.search_outputs:
            group_dict[getattr(phase, field)].append(phase)
        return AggregatorGroup(list(map(Aggregator, group_dict.values())))

    @property
    def model_results(self) -> str:
        """
        A string joining headers and results for all included search_outputs.
        """
        return "\n\n".join(
            "{}\n\n{}".format(phase.header, phase.model_results)
            for phase in self.search_outputs
        )
