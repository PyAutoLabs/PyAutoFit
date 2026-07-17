import logging
import os
import re
import shutil
import zipfile
from abc import ABC, abstractmethod
from configparser import NoSectionError
from pathlib import Path
from typing import Optional

import numpy as np

from autoconf import conf
from autoconf.test_mode import is_test_mode
from autofit.mapper.identifier import Identifier, IdentifierField
from autofit.non_linear.samples.summary import SamplesSummary

from autofit.text import text_util
from autofit.tools.util import open_, zip_directory

logger = logging.getLogger(__name__)

pattern = re.compile(r"(?<!^)(?=[A-Z])")


def _test_mode_segment() -> Optional[str]:
    """
    Returns ``"test_mode"`` when ``PYAUTO_TEST_MODE`` has an active
    test-mode level, else ``None``.

    Inserted into the output-path composition so that smoke runs land
    under ``output/test_mode/...`` instead of sharing a directory with
    real runs. Without this, a cached test-mode result short-circuits
    a later real run at the same paths ("Fit Already Completed").
    """
    return "test_mode" if is_test_mode() else None


class AbstractPaths(ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[os.PathLike] = None,
        is_identifier_in_paths=True,
        parent: Optional["AbstractPaths"] = None,
        unique_tag: Optional[str] = None,
        identifier: str = None,
        image_path_suffix: str = "",
    ):
        """
        Manages the path structure for `NonLinearSearch` output, for analyses both not using and using the search
        API. Use via non-linear searches requires manual input of paths, whereas the search API manages this using the
        search attributes.

        The output path within which the *Paths* objects path structure is contained is set via PyAutoConf, using the
        command:

        from autoconf import conf
        conf.instance = conf.Config(output_path="path/to/output")

        If we assume all the input strings above are used with the following example names:

        name = "name"
        path_prefix = "folder_0/folder_1"

        The output path of the `NonLinearSearch` results will be:

        /path/to/output/folder_0/folder_1/name

        If ``PYAUTO_TEST_MODE`` has an active level greater than zero, a
        ``test_mode`` segment is inserted directly after the output root:

        /path/to/output/test_mode/folder_0/folder_1/name

        This keeps smoke-test artefacts in a sibling tree so they
        cannot be picked up by a later real run with the same paths.

        Parameters
        ----------
        name
            The name of the non-linear search, which is used as a folder name after the ``path_prefix``. For searchs
            this name is the ``name``.
        path_prefix
            A prefixed path that appears after the output_path but before the name variable.
        is_identifier_in_paths
            If True output path and symlink path terminate with an identifier generated from the
            search and model
        parent
            The parent paths object of this paths object.
        unique_tag
            A unique tag for the search, used to differentiate between searches with the same name.
        identifier
            A custom identifier for the search, if this is not None it will be used instead of the automatically
            generated identifier
        image_path_suffix
            A suffix which is appended to the image path. This is used to differentiate between different
            image outputs, for example the image of the starting point of an MLE.
        """

        self.name = name or ""
        self.path_prefix = path_prefix or ""

        self.unique_tag = unique_tag

        self._non_linear_name = None
        self.__custom_identifier = identifier
        self.__identifier = None

        self.is_identifier_in_paths = is_identifier_in_paths

        self._parent = None
        self.parent = parent

        try:
            self.remove_files = conf.instance["general"]["output"]["remove_files"]

            if conf.instance["general"]["hpc"]["hpc_mode"]:
                self.remove_files = True
        except NoSectionError as e:
            logger.exception(e)

        self.image_path_suffix = image_path_suffix

    @property
    @abstractmethod
    def samples(self):
        pass

    def save_parent_identifier(self):
        pass

    def save_unique_tag(self, is_grid_search=False):
        pass

    def __str__(self):
        return str(self.output_path)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    @property
    def parent(self) -> "AbstractPaths":
        """
        The search performed before this search. For example, a search
        that is then compared to searches during a grid search.
        """
        return self._parent

    @parent.setter
    @abstractmethod
    def parent(self, parent: "AbstractPaths"):
        pass

    @property
    @abstractmethod
    def is_grid_search(self) -> bool:
        pass

    def for_sub_analysis(self, analysis_name: str):
        return self.create_child(name=analysis_name)

    @abstractmethod
    def create_child(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        is_identifier_in_paths: Optional[bool] = None,
        identifier: Optional[str] = None,
    ) -> "AbstractPaths":
        """
        Create a paths object which is the child of some parent
        paths object. This is done during a GridSearch so that
        results can be stored in the correct directory.

        Parameters
        ----------
        name
        path_prefix
        is_identifier_in_paths
            If False then this path's identifier will not be
            added to its output path.
        identifier

        Returns
        -------
        A new paths object
        """

    search = IdentifierField()
    model = IdentifierField()
    unique_tag = IdentifierField()

    @property
    def non_linear_name(self):
        if self._non_linear_name is None:
            if self.search is not None:
                self._non_linear_name = pattern.sub(
                    "_", type(self.search).__name__
                ).lower()
        return self._non_linear_name

    @property
    def _identifier(self):
        if self.__custom_identifier is not None:
            return self.__custom_identifier
        if self.__identifier is None:
            if None in (self.model, self.search):
                logger.debug(
                    "Generating identifier without both model and search having been set."
                )

            identifier_list = [self.search, self.model]

            if self.unique_tag is not None:
                identifier_list.append(self.unique_tag)
            self.__identifier = Identifier(identifier_list)

        return self.__identifier

    @_identifier.setter
    def _identifier(self, identifier):
        self.__identifier = identifier

    @property
    def identifier(self):
        return str(self._identifier)

    def save_identifier(self):
        with open_(self.output_path / ".identifier", "w+") as f:
            f.write(self._identifier.description)

    @property
    def search_internal_path(self) -> Path:
        """
        The path to the samples folder.
        """

        os.makedirs(self._files_path / "search_internal", exist_ok=True)

        return self._files_path / "search_internal"

    @property
    def image_path(self) -> Path:
        """
        The path to the image folder.
        """

        if not (self.output_path / f"image{self.image_path_suffix}").exists():
            os.makedirs(self.output_path / f"image{self.image_path_suffix}")

        return self.output_path / f"image{self.image_path_suffix}"

    @property
    def profile_path(self) -> Path:
        """
        The path to the profile folder.
        """
        return self.output_path / "profile"

    @property
    def output_path(self) -> Path:
        """
        The path to the output information for a search.
        """

        strings = list(
            filter(
                None,
                [
                    str(conf.instance.output_path),
                    _test_mode_segment(),
                    str(self.path_prefix),
                    self.unique_tag,
                    str(self.name),
                ],
            )
        )

        if self.is_identifier_in_paths:
            strings.append(self.identifier)

        return Path(*strings) if strings else Path("")

    @property
    def _files_path(self) -> Path:
        """
        This is private for a reason, use the save_json etc. methods to save and load json
        """
        files_path = self.output_path / "files"
        try:
            os.makedirs(files_path, exist_ok=True)
        except FileExistsError:
            pass
        return files_path

    def zip_remove(self):
        """
        Copy files from the sym linked search folder then remove the sym linked folder.
        """

        self._zip()

    def _zip(self):
        try:
            zip_directory(self.output_path, self._zip_path)

            if self.remove_files:
                shutil.rmtree(
                    self.output_path,
                    ignore_errors=True,
                )

        except FileNotFoundError:
            pass

    def preserve_in_zip(self, file_path):
        """
        Add a file (already inside this search's output directory) to the
        search's ``.zip`` archive so it survives the resume cycle.

        ``restore()`` deletes the output directory and re-extracts the zip, so
        any file written into e.g. ``files/`` *after* a search completed — such
        as a cache artifact derived from the finished result — would be
        destroyed by the next resume unless it is also a member of the zip.

        No-op when the zip does not exist (e.g. the search is still running,
        so the file will be zipped with everything else at completion) and
        when the member is already present.

        Parameters
        ----------
        file_path
            Absolute path of the file to preserve; must live under
            ``output_path``, from which its archive name is derived.
        """
        if not Path(self._zip_path).exists():
            return

        arcname = str(Path(file_path).relative_to(self.output_path))

        with zipfile.ZipFile(self._zip_path, "a") as f:
            if arcname not in f.namelist():
                f.write(file_path, arcname)

    def restore(self):
        """
        Copy files from the ``.zip`` file to the samples folder.
        """

        if Path(self._zip_path).exists():
            shutil.rmtree(self.output_path, ignore_errors=True)

            try:
                try:
                    with zipfile.ZipFile(self._zip_path, "r") as f:
                        f.extractall(self.output_path)
                except FileExistsError:
                    pass
            except zipfile.BadZipFile as e:
                raise zipfile.BadZipFile(
                    f"Unable to restore the zip file at the path {self._zip_path}"
                ) from e

            try:
                os.remove(self._zip_path)
            except FileNotFoundError:
                pass

    def __eq__(self, other):
        return isinstance(other, AbstractPaths) and all(
            [
                self.path_prefix == other.path_prefix,
                self.name == other.name,
                self.non_linear_name == other.non_linear_name,
            ]
        )

    @property
    def _zip_path(self) -> str:
        return f"{self.output_path}.zip"

    @abstractmethod
    def save_json(self, name, object_dict: dict, prefix: str = ""):
        pass

    @abstractmethod
    def load_json(self, name, prefix: str = "") -> dict:
        pass

    @abstractmethod
    def save_array(self, name, array: np.ndarray):
        pass

    @abstractmethod
    def load_array(self, name) -> np.ndarray:
        pass

    @abstractmethod
    def save_fits(self, name: str, fits, prefix: str = ""):
        pass

    @abstractmethod
    def load_fits(self, name: str, prefix: str = ""):
        pass

    @abstractmethod
    def save_object(self, name: str, obj: object, prefix: str = ""):
        pass

    @abstractmethod
    def load_object(self, name: str, prefix: str = ""):
        pass

    @abstractmethod
    def remove_object(self, name: str):
        pass

    @abstractmethod
    def is_object(self, name: str) -> bool:
        pass

    def save_search_internal(self, obj):
        raise NotImplementedError

    def load_search_internal(self):
        raise NotImplementedError

    def remove_search_internal(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @abstractmethod
    def completed(self):
        pass

    @abstractmethod
    def save_all(self, search_config_dict=None, info=None):
        pass

    @abstractmethod
    def load_samples(self):
        """
        Load samples from the database
        """

    @abstractmethod
    def save_samples(self, samples):
        """
        Save samples to the database
        """

    def save_samples_summary(
        self,
        samples_summary: SamplesSummary,
        name: str = "",
    ):
        """
        Save samples summary to the database.
        """

    def load_samples_summary(self) -> SamplesSummary:
        """
        Load samples summary from the database.
        """

    @abstractmethod
    def save_latent_samples(self, latent_samples):
        """
        Save latent variables. These are values computed from an instance and output
        during analysis.
        """

    @abstractmethod
    def load_samples_info(self):
        pass

    def save_summary(
        self,
        samples,
        latent_samples,
        log_likelihood_function_time,
        visualization_time = None,
    ):
        result_info = text_util.result_info_from(
            samples=samples,
        )
        self.output_model_results(result_info=result_info)

        if latent_samples:
            result_info = text_util.result_info_from(
                samples=latent_samples,
            )
            filename = self.output_path / "latent.results"
            with open_(filename, "w") as f:
                f.write(result_info)

        text_util.search_summary_to_file(
            samples=samples,
            log_likelihood_function_time=log_likelihood_function_time,
            visualization_time=visualization_time,
            filename=self.output_path / "search.summary",
        )

    @property
    def _samples_file(self) -> Path:
        return self._files_path / "samples.csv"

    @property
    def _latent_variables_file(self) -> Path:
        return self._files_path / "latent.csv"

    @property
    def _covariance_file(self) -> Path:
        return self._files_path / "covariance.csv"

    @property
    def _info_file(self) -> Path:
        return self._files_path / "samples_info.json"

    def output_model_results(self, result_info):

        filename = self.output_path / "model.results"

        with open_(filename, "w") as f:
            f.write(result_info)
