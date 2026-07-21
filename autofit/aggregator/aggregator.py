#!/usr/bin/env python

"""
Filter and collate search outputs in all subdirectories.

Usage:

./aggregator.py (root_directory) [pipeline=pipeline phase=phase dataset=dataset]

Example:

./aggregator.py ../output pipeline=data_mass_x1_source_x1_positions
"""

import os
import zipfile
from collections import defaultdict
from pathlib import Path
from shutil import rmtree
from typing import List, Union, Iterator, Optional

from autonerves.test_mode import is_test_mode

from .predicate import AttributePredicate
from .search_output import SearchOutput, GridSearchOutput, GridSearch


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


def unzip_directory(directory: str):
    """
    Unzip all zip files in a directory recursively.
    """
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".zip"):
                try:
                    with zipfile.ZipFile(Path(root) / filename, "r") as f:
                        f.extractall(Path(root) / filename[:-4])
                except zipfile.BadZipFile:
                    raise zipfile.BadZipFile(
                        f"File is not a zip file: \n " f"{root} \n" f"{filename}"
                    )


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
    ):
        """
        Class to aggregate phase results for all subdirectories in a given directory.

        Parameters
        ----------
        search_outputs
            A list of search_outputs
        """
        if len(search_outputs) > 20:
            print(
                "When aggregating many searches it can be more efficient to use the database.\n"
                "Checkout the database cookbook at this URL: "
                "https://pyautofit.readthedocs.io/en/latest/cookbooks/database.html"
            )
        self.search_outputs = search_outputs
        self.grid_search_outputs = grid_search_outputs

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
    ) -> "Aggregator":
        """
        Aggregate phase results for all subdirectories in a given directory.

        The whole directory structure is traversed and a Phase object created for each directory that contains a
        metadata file.

        Zipped search outputs are extracted as they are encountered, in the same traversal. A zip whose
        extracted directory already exists is skipped, so repeat calls do not pay the extraction cost again;
        delete the extracted directory to force re-extraction.

        Parameters
        ----------
        directory
            A directory in which the outputs of search_outputs are kept. This is searched recursively.
        completed_only
            If `True` only search_outputs with a .completed file (indicating the phase was completed)
            are included in the aggregator.
        reference
            A dictionary mapping paths to types to be used when loading models from disk.
        """
        print("Aggregator loading search_outputs... could take some time.")

        def scan(scan_directory):
            search_outputs = []
            grid_search_outputs = []

            for root, dirs, filenames in os.walk(scan_directory, topdown=True):
                for filename in filenames:
                    if filename.endswith(".zip"):
                        extracted = Path(root) / filename[:-4]
                        if extracted.exists():
                            continue
                        try:
                            with zipfile.ZipFile(Path(root) / filename, "r") as f:
                                f.extractall(extracted)
                        except zipfile.BadZipFile:
                            raise zipfile.BadZipFile(
                                f"File is not a zip file: \n " f"{root} \n" f"{filename}"
                            )
                        dirs.append(filename[:-4])

                def should_add():
                    return not completed_only or ".completed" in filenames

                if "metadata" in filenames:
                    if should_add():
                        search_outputs.append(
                            SearchOutput(
                                Path(root),
                                reference=reference,
                            )
                        )
                if ".is_grid_search" in filenames:
                    if should_add():
                        grid_search_outputs.append(
                            GridSearchOutput(
                                Path(root),
                            )
                        )

            return search_outputs, grid_search_outputs

        search_outputs, grid_search_outputs = scan(directory)

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
                search_outputs, grid_search_outputs = scan(test_mode_directory)

        if len(search_outputs) == 0:
            print(f"\nNo search_outputs found in {directory}\n")
        else:
            print(
                f"\n A total of {str(len(search_outputs))} search_outputs and results were found."
            )

        return cls(search_outputs, grid_search_outputs)

    def add_directory(self, directory: Union[str, Path]):
        """
        Add a directory to the aggregator.
        """
        aggregator = Aggregator.from_directory(directory)
        self.search_outputs.extend(aggregator.search_outputs)
        self.grid_search_outputs.extend(aggregator.grid_search_outputs)

    def remove_unzipped(self):
        """
        Removes the unzipped output directory for each phase.
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
        print(f"Filter found a total of {str(len(search_outputs))} results")
        return Aggregator(
            search_outputs=list(search_outputs),
            grid_search_outputs=self.grid_search_outputs,
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
