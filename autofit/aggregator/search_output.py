import csv
import json
import logging
import pickle
import tempfile
import weakref
import zipfile
from abc import ABC
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from shutil import rmtree
from typing import Generator, Tuple, Optional, List, cast, Type

import dill
from PIL import Image

from autonerves import cached_property
from autonerves.class_path import get_class
from autofit.non_linear.samples import Samples

from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.aggregator.file_output import (
    JSONOutput,
    FileOutput,
)
from autofit.aggregator.zip_member import ZipMember
from autofit.mapper.identifier import Identifier
from autofit.non_linear.samples.sample import samples_from_iterator
from autonerves.dictable import from_dict
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

_KINDS = (".json", ".pickle", ".csv", ".fits")
"""
The file kinds ``value`` loads, in the order a bare name resolves them.
"""

_warned_ambiguous = set()


class TemporaryExtraction:
    """
    A temporary directory into which the aggregator extracts zipped search outputs.

    Every search output read out of the directory holds a reference to the
    ``TemporaryExtraction`` that owns it, so the extracted files live exactly as
    long as something still points at them and are removed once the last such
    object is garbage collected. ``cleanup`` removes them eagerly.

    A ``weakref.finalize`` is used rather than ``tempfile.TemporaryDirectory``
    because the latter warns (``ResourceWarning``) whenever it is cleaned up
    implicitly, and implicit cleanup is the normal path here.
    """

    def __init__(self, prefix: str = "autofit_aggregator_"):
        self.path = Path(tempfile.mkdtemp(prefix=prefix))
        # The callback must not close over ``self`` or the object would never
        # become unreachable; ``self.path`` is bound as an argument instead.
        self._finalizer = weakref.finalize(self, rmtree, self.path, True)

    @property
    def is_alive(self) -> bool:
        """
        Whether the extracted files are still on disk.
        """
        return self._finalizer.alive

    def cleanup(self):
        """
        Remove the extracted files now, without waiting for garbage collection.

        Search outputs still pointing into the directory cannot be read afterwards.
        """
        self._finalizer()


class AbstractSearchOutput(ABC):
    def __init__(
        self,
        directory: Path,
        reference: Optional[dict] = None,
        temporary_directory: Optional[TemporaryExtraction] = None,
    ):
        self.directory = directory
        self._reference = reference
        # Keeps a temporary extraction alive for as long as this output can be
        # read from it. ``None`` for outputs read from a real directory.
        # Must be set here: ``__getattr__`` would otherwise try to load a
        # pickle of this name from disk.
        self._temporary_directory = temporary_directory

    @property
    def is_complete(self) -> bool:
        """
        Whether the search has completed
        """
        return (self.directory / ".completed").exists()

    @cached_property
    def completed_at(self) -> Optional[str]:
        """
        When the search completed (ISO 8601), or ``None`` if it has not.

        Taken from the status record beside the output (``<output_path>.status.json``)
        when it says ``completed``, else from the ``.completed`` member of the zip
        beside the output, else from the ``.completed`` file's modification time. The
        zip's own modification time is never used: copying or re-zipping resets it.
        An output extracted to a temporary directory has neither sibling, so its
        ``.completed`` time is the extraction time; index the tree
        (``python -m autofit.results index``) to compare such outputs.
        """
        if isinstance(self.directory, ZipMember):
            zip_path = self.directory.zip_path
            output_path = zip_path.with_name(zip_path.name[: -len(".zip")])
        else:
            output_path = Path(self.directory)
            zip_path = Path(f"{output_path}.zip")
        try:
            with open(f"{output_path}.status.json") as f:
                status = json.load(f)
            if status.get("state") == "completed" and status.get("updated"):
                return status["updated"]
        except (OSError, ValueError, AttributeError):
            pass
        try:
            with zipfile.ZipFile(zip_path) as f:
                return datetime(*f.getinfo(".completed").date_time).isoformat()
        except (OSError, KeyError, zipfile.BadZipFile):
            pass
        if isinstance(self.directory, ZipMember):
            return None
        try:
            mtime = (output_path / ".completed").stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(mtime, timezone.utc).isoformat(
            timespec="seconds"
        )

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

    @cached_property
    def _outputs_by_suffix(self) -> dict:
        """
        All output files in the files and image directories, grouped by suffix.

        A single traversal of each directory serves every suffix — building the
        json/pickle/csv/fits lists separately makes eight sweeps per search output,
        which dominates load time when aggregating many results.
        """
        outputs = {".json": [], ".pickle": [], ".csv": [], ".fits": []}
        for directory_name in ("files", "image"):
            files_path = self.directory / directory_name
            for file_path in files_path.rglob("*"):
                if file_path.suffix in outputs:
                    name = ".".join(
                        file_path.relative_to(files_path).with_suffix("").parts
                    )
                    outputs[file_path.suffix].append(FileOutput(name, file_path))
        return outputs

    @cached_property
    def jsons(self) -> List[JSONOutput]:
        """
        The json files in the search output files directory
        """
        return cast(List[JSONOutput], self._outputs_by_suffix[".json"])

    @cached_property
    def arrays(self):
        """
        The csv files in the search output files directory
        """
        return self._outputs_by_suffix[".csv"]

    @cached_property
    def pickles(self):
        """
        The pickle files in the search output files directory
        """
        return self._outputs_by_suffix[".pickle"]

    @cached_property
    def fits(self):
        """
        The fits files in the search output files directory
        """
        return self._outputs_by_suffix[".fits"]

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
        return Image.open((self.directory / "image" / f"{name}.png").open("rb"))

    def open_member(self, path: str, mode: str = "rb"):
        """
        Open a file of this output by its path relative to the output (e.g.
        ``"files/tracer.fits"`` or ``"image/fit.png"``), whether the output is a folder
        or a zip read in place.

        Parameters
        ----------
        path
            The file's path inside the output, with ``/`` separators.
        mode
            ``"rb"`` (the default) or ``"r"``.
        """
        return (self.directory / path).open(mode)

    def value(self, name: str, kind: Optional[str] = None):
        """
        Load the value of some object in the files directory for the search.

        This may be a pickle, json, csv or fits file.

        If the JSON has a specified type it is parsed as that type. See dictable.py
        in autonerves.

        Returns None if the file does not exist.

        Parameters
        ----------
        name
            The name of the file to load without a file suffix.
        kind
            Which file to load when several share ``name``: ``"json"``, ``"pickle"``,
            ``"csv"`` or ``"fits"``. Without it a JSON file wins, then pickle, csv and
            fits, and a warning names the files ``name`` could mean (e.g.
            ``tracer.json`` and ``tracer.fits``), so pass ``kind`` to be explicit.

        Returns
        -------
        The loaded object
        """
        if kind is not None:
            suffix = f".{kind.lstrip('.')}"
            if suffix not in _KINDS:
                raise ValueError(
                    f"kind must be one of {sorted(k[1:] for k in _KINDS)}, not {kind!r}."
                )
            for item in self._outputs_by_suffix[suffix]:
                if item.name == name:
                    if suffix == ".json":
                        return item.value_using_reference(self._reference)
                    return item.value
            return None

        matches = [
            item
            for suffix in _KINDS
            for item in self._outputs_by_suffix[suffix]
            if item.name == name
        ]
        if not matches:
            return None
        if len(matches) > 1 and name not in _warned_ambiguous:
            _warned_ambiguous.add(name)
            logging.warning(
                f"'{name}' names several files ({', '.join(m.path.name for m in matches)}); "
                f"the {matches[0].path.suffix} file is loaded. Pass kind= to choose."
            )
        item = matches[0]
        if isinstance(item, JSONOutput):
            return item.value_using_reference(self._reference)
        return item.value


class SearchOutput(AbstractSearchOutput, fit_interface.Fit):
    """
    @DynamicAttrs
    """

    is_grid_search = False

    def __init__(
        self,
        directory: Path,
        reference: dict = None,
        temporary_directory: Optional[TemporaryExtraction] = None,
    ):
        """
        Represents the output of a single search. Comprises the ``files`` directory
        written by the search (including ``search.json``) and other dataset files.

        Parameters
        ----------
        directory
            The directory of the search
        temporary_directory
            The temporary extraction this output was read from, if any. Held so the
            extracted files outlive the scan that produced them.
        """
        super().__init__(directory, reference, temporary_directory)
        self.__search = None
        self.__model = None
        self._samples = None
        self._latent_samples = None
        self.__samples_summary = None
        self.__latent_summary = None

        self.directory = directory

    @classmethod
    def from_zip(cls, zip_path, reference: Optional[dict] = None) -> "SearchOutput":
        """
        The output archived in the zip at ``zip_path``, read in place: every file is
        read straight out of the zip when it is needed and nothing is extracted.
        """
        return cls(ZipMember(zip_path), reference=reference)

    @property
    def samples_summary(self) -> SamplesSummary:
        """
        The summary of the samples, which includes the maximum log likelihood sample and the log evidence.

        This is loaded from a JSON file.
        """
        if self.__samples_summary is None:
            summary = self.value("samples_summary")
            summary.model = self.model
            self.__samples_summary = summary
        return self.__samples_summary

    @property
    def latent_summary(self) -> SamplesSummary:
        """
        The summary of the samples, which includes the maximum log likelihood sample and the log evidence.

        This is loaded from a JSON file.
        """
        if self.__latent_summary is None:
            summary = self.value("latent.latent_summary")
            summary.model = self.model
            self.__latent_summary = summary
        return self.__latent_summary

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

    @cached_property
    def id(self) -> str:
        """
        The unique identifier of the search.

        This is used as a directory name and as a database identifier.

        It is the identifier the search wrote, when the output persisted one: the
        ``.identifier`` file holds the identifier's description (its ``hash_list``, one
        entry per line), whose md5 is the identifier. Recomputing it from today's
        loaded search and model would give an old output a new identity whenever the
        hashing rules change (e.g. the search-level seed, which reads main's default
        BlackJAXNUTS / SMC ``seed: 42`` differently). Outputs without the file are
        recomputed, as before.
        """
        persisted = self.persisted_identifier
        if persisted is not None:
            return persisted
        return str(Identifier([self.search, self.model, self.unique_tag]))

    @property
    def persisted_identifier(self) -> Optional[str]:
        """
        The identifier recorded in the output's ``.identifier`` file, or ``None`` when
        the output has none (``DirectoryPaths.save_identifier`` writes
        ``"\\n".join(hash_list)``; ``Identifier.__str__`` is the md5 of
        ``".".join(hash_list)``).
        """
        try:
            description = (self.directory / ".identifier").read_text()
        except (FileNotFoundError, NotADirectoryError):
            return None
        if not description:
            return None
        return md5(".".join(description.split("\n")).encode("utf-8")).hexdigest()

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

            with (directory / "samples.csv").open() as f:
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
        return list(map(SearchOutput, self.directory.glob("analyses/*")))

    @property
    def model_results(self) -> str:
        """
        Reads the model.results file
        """
        with (self.directory / "model.results").open() as f:
            return f.read()

    @property
    def mask(self):
        """
        A pickled mask object
        """
        with (self.files_path / "mask.pickle").open("rb") as f:
            return dill.load(f)

    @property
    def header(self) -> str:
        """
        A header created by joining the search name
        """
        phase = self.phase or ""
        dataset_name = self.dataset_name or ""
        return str(Path(phase) / dataset_name) if dataset_name else phase

    @property
    def search(self):
        """
        The search object that was used in this phase
        """
        if self.__search is None:
            try:
                with (self.files_path / "search.json").open() as f:
                    self.__search = from_dict(json.load(f))
            except (FileNotFoundError, ModuleNotFoundError):
                try:
                    with (self.files_path / "search.pickle").open("rb") as f:
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

    @property
    def non_linear_search(self) -> Optional[str]:
        """
        Lower-cased class name of the search (e.g. "emcee"), read from
        ``files/search.json``.

        Legacy output folders which predate that sentinel are read from the
        ``non_linear_search=`` line of their ``metadata`` file instead.
        """
        search_path = self.directory / "files" / "search.json"
        try:
            with search_path.open() as f:
                class_path = json.load(f)["class_path"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
        else:
            return class_path.rsplit(".", 1)[-1].lower()

        try:
            text = (self.directory / "metadata").read_text()
        except FileNotFoundError:
            return None
        for line in text.split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                if key.strip() == "non_linear_search":
                    return value.strip()
        return None

    def __str__(self):
        return f"<SearchOutput {self.directory}>"

    def __repr__(self):
        return str(self)


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
