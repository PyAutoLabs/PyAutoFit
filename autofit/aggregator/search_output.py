import csv
import json
import logging
import pickle
from abc import ABC
from os import path
from pathlib import Path
from typing import Generator, Tuple, Optional, List, cast, Type

import dill
from PIL import Image

from autoconf import cached_property
from autoconf.class_path import get_class
from autofit.non_linear.samples import Samples

from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.aggregator.file_output import (
    JSONOutput,
    FileOutput,
)
from autofit.mapper.identifier import Identifier
from autofit.non_linear.samples.sample import samples_from_iterator
from autoconf.dictable import from_dict
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.non_linear.samples.util import simple_model_for_kwargs
from . import fit_interface

# noinspection PyProtectedMember
original_create_file_handle = dill._dill._create_filehandle


def _create_file_handle(*args, **kwargs):
    """
    Handle FileNotFoundError when attempting to deserialize pickles
    using dill and return None instead.
    """
    try:
        return original_create_file_handle(*args, **kwargs)
    except pickle.UnpicklingError as e:
        if not isinstance(e.args[0], FileNotFoundError):
            raise e
        logging.warning(
            f"Could not create a handler for {e.args[0].filename} as it does not exist"
        )
        return None


dill._dill._create_filehandle = _create_file_handle


class AbstractSearchOutput(ABC):
    def __init__(self, directory: Path, reference: Optional[dict] = None):
        self.directory = directory
        self._reference = reference

    @property
    def is_complete(self) -> bool:
        """
        Whether the search has completed
        """
        return (self.directory / ".completed").exists()

    @property
    def parent_identifier(self) -> Optional[str]:
        """
        Read the parent identifier for a fit in a directory.

        Defaults to None if no .parent_identifier file is found.
        """
        try:
            return (self.directory / ".parent_identifier").read_text()
        except FileNotFoundError:
            return None

    @property
    def files_path(self):
        return self.directory / "files"

    def _outputs(self, suffix):
        return self._outputs_in_directory("files", suffix) + self._outputs_in_directory(
            "image", suffix
        )

    def _outputs_in_directory(self, name: str, suffix: str):
        files_path = self.directory / name
        outputs = []
        for file_path in files_path.rglob(f"*{suffix}"):
            name = ".".join(file_path.relative_to(files_path).with_suffix("").parts)
            outputs.append(FileOutput(name, file_path))
        return outputs

    @cached_property
    def jsons(self) -> List[JSONOutput]:
        """
        The json files in the search output files directory
        """
        return cast(List[JSONOutput], self._outputs(".json"))

    @cached_property
    def arrays(self):
        """
        The csv files in the search output files directory
        """
        return self._outputs(".csv")

    @cached_property
    def pickles(self):
        """
        The pickle files in the search output files directory
        """
        return self._outputs(".pickle")

    @cached_property
    def fits(self):
        """
        The fits files in the search output files directory
        """
        return self._outputs(".fits")

    @property
    def max_log_likelihood(self) -> Optional[float]:
        """
        The log likelihood of the maximum log likelihood sample
        """
        try:
            return self.samples.max_log_likelihood_sample.log_likelihood
        except AttributeError:
            return None

    def __getattr__(self, name):
        """
        Attempt to load a pickle by the same name from the search output directory.

        dataset.pickle, meta_dataset.pickle etc.
        """
        return self.value(name)

    def image(self, name: str) -> Image.Image:
        """
        Load an image from the files directory for the search.

        Parameters
        ----------
        name
            The name of the file to load without a file suffix.

        Returns
        -------
        The loaded image
        """
        return Image.open(self.directory / "image" / f"{name}.png")

    def value(self, name: str):
        """
        Load the value of some object in the files directory for the search.

        This may be a pickle, json, csv or fits file.

        If the JSON has a specified type it is parsed as that type. See dictable.py
        in autoconf.

        Returns None if the file does not exist.

        Parameters
        ----------
        name
            The name of the file to load without a file suffix.

        Returns
        -------
        The loaded object
        """
        for item in self.jsons:
            if item.name == name:
                return item.value_using_reference(self._reference)
        for item in self.pickles + self.arrays + self.fits:
            if item.name == name:
                return item.value

        return None


class SearchOutput(AbstractSearchOutput, fit_interface.Fit):
    """
    @DynamicAttrs
    """

    is_grid_search = False

    def __init__(self, directory: Path, reference: dict = None):
        """
        Represents the output of a single search. Comprises a metadata file and other dataset files.

        Parameters
        ----------
        directory
            The directory of the search
        """
        super().__init__(directory, reference)
        self.__search = None
        self.__model = None
        self._samples = None
        self._latent_samples = None

        self.directory = directory

        self.file_path = directory / "metadata"

        try:
            with open(self.file_path) as f:
                self.text = f.read()
                pairs = [
                    line.split("=") for line in self.text.split("\n") if "=" in line
                ]
                self.__dict__.update({pair[0]: pair[1] for pair in pairs})
        except FileNotFoundError:
            pass

    @property
    def samples_summary(self) -> SamplesSummary:
        """
        The summary of the samples, which includes the maximum log likelihood sample and the log evidence.

        This is loaded from a JSON file.
        """
        summary = self.value("samples_summary")
        summary.model = self.model
        return summary

    @property
    def latent_summary(self) -> SamplesSummary:
        """
        The summary of the samples, which includes the maximum log likelihood sample and the log evidence.

        This is loaded from a JSON file.
        """
        summary = self.value("latent.latent_summary")
        summary.model = self.model
        return summary

    @property
    def instance(self):
        """
        The instance of the maximum log likelihood sample i.e. the instance
        with the greatest likelihood.

        None if samples cannot be loaded.
        """
        try:
            return self.samples.max_log_likelihood()
        except (AttributeError, NotImplementedError):
            return self.samples_summary.instance

    @property
    def id(self) -> str:
        """
        The unique identifier of the search.

        This is used as a directory name and as a database identifier.
        """
        return str(Identifier([self.search, self.model, self.unique_tag]))

    @property
    def model(self):
        """
        The model used by the search
        """
        if self.__model is None:
            self.__model = self.value("model")
        return self.__model

    @property
    def samples(self) -> SamplesPDF:
        """
        The samples of the search, parsed from a CSV containing individual samples
        and a JSON containing metadata.
        """
        if not self._samples:
            self._samples = self._load_samples(
                model=self.model,
            )
        return self._samples

    @property
    def latent_samples(self):
        """
        The latent variables of the search, parsed from a CSV file.
        """
        if not self._latent_samples:
            self._latent_samples = self._load_samples("latent")
        return self._latent_samples

    def _load_samples(self, name=None, model=None):
        if name:
            directory = self.files_path / name
        else:
            directory = self.files_path
        try:
            info_json = JSONOutput("info", directory / "samples_info.json").dict

            with open(directory / "samples.csv") as f:
                sample_list = samples_from_iterator(csv.reader(f))

            if model is None:
                try:
                    model = simple_model_for_kwargs(sample_list[0].kwargs)
                except IndexError:
                    model = None

            cls = cast(
                Type[Samples],
                get_class(info_json["class_path"]),
            )

            return cls.from_list_info_and_model(
                sample_list=sample_list,
                samples_info=info_json,
                model=model,
            )
        except FileNotFoundError:
            raise AttributeError(f"No {name} found")

    def names_and_paths(
        self,
        suffix: str,
    ) -> Generator[Tuple[str, Path], None, None]:
        """
        Get the names and paths of files with a given suffix.

        Parameters
        ----------
        suffix
            The suffix of the files to retrieve (e.g. ".json")

        Returns
        -------
        A generator of tuples of the form (name, path) where name is the path to the file
        joined by . without the suffix and path is the path to the file
        """
        for file in list(self.files_path.rglob(f"*{suffix}")):
            name = ".".join(file.relative_to(self.files_path).with_suffix("").parts)
            yield name, file

    @property
    def children(self):
        """
        A list of child analyses loaded from the analyses directory
        """
        return list(map(SearchOutput, Path(self.directory).glob("analyses/*")))

    @property
    def model_results(self) -> str:
        """
        Reads the model.results file
        """
        with open(self.directory / "model.results") as f:
            return f.read()

    @property
    def mask(self):
        """
        A pickled mask object
        """
        with open(self.files_path / "mask.pickle", "rb") as f:
            return dill.load(f)

    @property
    def header(self) -> str:
        """
        A header created by joining the search name
        """
        phase = self.phase or ""
        dataset_name = self.dataset_name or ""
        return path.join(phase, dataset_name)

    @property
    def search(self):
        """
        The search object that was used in this phase
        """
        if self.__search is None:
            try:
                with open(self.files_path / "search.json") as f:
                    self.__search = from_dict(json.load(f))
            except (FileNotFoundError, ModuleNotFoundError):
                try:
                    with open(self.files_path / "search.pickle", "rb") as f:
                        self.__search = pickle.load(f)
                except (FileNotFoundError, ModuleNotFoundError):
                    logging.warning("Could not load search")
        return self.__search

    def child_values(self, name):
        """
        Get the values of a given key for all children
        """
        return [getattr(child, name) for child in self.children]

    @property
    def path_prefix(self):
        return self.search.paths.path_prefix

    @property
    def name(self):
        """
        The name of the search
        """
        return self.search.name

    @property
    def unique_tag(self):
        """
        The unique tag of the search
        """
        return self.search.unique_tag

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<PhaseOutput {}>".format(self)


class GridSearchOutput(AbstractSearchOutput):
    is_grid_search = True

    @property
    def unique_tag(self) -> str:
        """
        The unique tag of the grid search.
        """
        with open(self.directory / ".is_grid_search") as f:
            return f.read()

    @property
    def id(self) -> str:
        """
        Use the unique tag of the grid search as an identifier.
        """
        return self.unique_tag


class GridSearch:
    def __init__(
        self,
        grid_search_output: GridSearchOutput,
        children: List[SearchOutput],
    ):
        """
        Represents the output of a grid search. Comprises overall information from the grid search
        and output from each individual search.

        Parameters
        ----------
        grid_search_output
            The output of the grid search
        children
            The outputs of each individual search performed as part of the grid search
        """
        self.grid_search_output = grid_search_output
        self.children = children

    @property
    def best_fit(self) -> SearchOutput:
        """
        The output for the search in the grid search that had the greatest log likelihood
        """
        return max(self.children, key=lambda x: x.instance.log_likelihood)

    def __getattr__(self, item):
        return getattr(self.grid_search_output, item)
