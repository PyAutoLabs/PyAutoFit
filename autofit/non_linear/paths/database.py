import shutil
from typing import Optional, Union

from autoconf.output import conditional_output, should_output
from autofit.database.sqlalchemy_ import sa
from .abstract import AbstractPaths
import numpy as np

from autofit.database.model import Fit
from autoconf.dictable import to_dict, from_dict
from autofit.database.aggregator.info import Info
from autofit.non_linear.samples.summary import SamplesSummary


class DatabasePaths(AbstractPaths):
    def __init__(
        self,
        session,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        is_identifier_in_paths=True,
        parent=None,
        save_all_samples=False,
        unique_tag: Optional["str"] = None,
    ):
        super().__init__(
            name=name,
            path_prefix=path_prefix,
            is_identifier_in_paths=is_identifier_in_paths,
            parent=parent,
        )
        self.session = session
        self._fit = None
        self.save_all_samples = save_all_samples
        self.unique_tag = unique_tag

    __nullify_fields__ = ("session",)

    parent: "DatabasePaths"

    @AbstractPaths.parent.setter
    def parent(self, parent: "DatabasePaths"):
        """
        The search performed before this search. For example, a search
        that is then compared to searches during a grid search.

        For database paths the parent must also be database paths.
        """
        if not (parent is None or isinstance(parent, DatabasePaths)):
            raise TypeError(
                "The parent of search that uses the database must also use the database"
            )
        self._parent = parent

    @property
    def is_grid_search(self) -> bool:
        return self.fit.is_grid_search

    def create_child(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        is_identifier_in_paths: Optional[bool] = None,
        identifier: Optional[str] = None,
    ) -> "DatabasePaths":
        """
        Create a paths object which is the child of some parent
        paths object. This is done during a GridSearch so that
        results can be stored in the correct directory. It also
        allows database fit objects to be related correctly.

        If no instance is set the prior median model is used
        to ensure that the parent object is queryable.

        Parameters
        ----------
        identifier
        name
        path_prefix
        is_identifier_in_paths
            If False then this path's identifier will not be
            added to its output path.

        Returns
        -------
        A new paths object
        """
        self.fit.is_grid_search = True
        if self.fit.instance is None:
            self.fit.instance = self.model.instance_from_prior_medians()
        child = type(self)(
            session=self.session,
            name=name or self.name,
            path_prefix=path_prefix or self.path_prefix,
            is_identifier_in_paths=(
                is_identifier_in_paths
                if is_identifier_in_paths is not None
                else self.is_identifier_in_paths
            ),
            parent=self,
        )
        child.model = self.model
        child.search = self.search
        child._identifier = identifier
        return child

    def zip_remove(self):
        """
        Remove files from both the symlinked folder and the output directory
        """
        self.session.commit()
        Info(self.session).write()

        if self.remove_files:
            shutil.rmtree(self.output_path, ignore_errors=True)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["session"]
        return d

    @conditional_output
    def save_json(self, name, object_dict: Union[dict, list], prefix: str = ""):
        """
        Save a dictionary as a json file in the database

        Parameters
        ----------
        name
            The name of the json
        object_dict
            The dictionary to save
        """
        self.fit.set_json(name, object_dict)

    def load_json(self, name: str, prefix: str = "") -> Union[dict, list]:
        """
        Load a json file from the database

        Parameters
        ----------
        name
            The name of the json

        Returns
        -------
        The loaded dictionary
        """
        return self.fit.get_json(name)

    @property
    def samples(self):
        return self.fit.samples

    @conditional_output
    def save_array(self, name, array: np.ndarray):
        """
        Save an array as a json file in the database

        Parameters
        ----------
        name
            The name of the array
        array
            The array to save
        """
        self.fit.set_array(name, array)

    def load_array(self, name: str) -> np.ndarray:
        """
        Load an array from the database

        Parameters
        ----------
        name
            The name of the array

        Returns
        -------
        The loaded array
        """
        return self.fit.get_array(name)

    @conditional_output
    def save_fits(self, name: str, fits, prefix: str = ""):
        """
        Save a fits file in the database

        Parameters
        ----------
        name
            The name of the fits file
        fits
            The fits file to save
        """
        self.fit.set_fits(name, fits)

    def load_fits(self, name: str, prefix: str = ""):
        """
        Load a fits file from the database

        Parameters
        ----------
        name
            The name of the fits file

        Returns
        -------
        The loaded hdu
        """
        return self.fit.get_fits(name)

    @conditional_output
    def save_object(self, name: str, obj: object, prefix: str = ""):
        self.fit[name] = obj

    def load_object(self, name: str, prefix: str = ""):
        return self.fit[name]

    def remove_object(self, name: str):
        del self.fit[name]

    def is_object(self, name: str) -> bool:
        return name in self.fit

    def save_search_internal(self, obj):
        pass

    def load_search_internal(self):
        pass

    def remove_search_internal(self):
        pass

    @property
    def fit(self) -> Fit:
        if self._fit is None:
            try:
                self._fit = (
                    self.session.query(Fit).filter(Fit.id == self.identifier).one()
                )
            except sa.orm.exc.NoResultFound:
                self._fit = Fit(
                    id=self.identifier,
                    is_complete=False,
                    unique_tag=self.unique_tag,
                    path_prefix=self.path_prefix,
                    name=self.name,
                )
                self.session.add(self._fit)

        if self.parent is not None:
            self._fit.parent = self.parent.fit
        return self._fit

    @property
    def is_complete(self) -> bool:
        return self.fit.is_complete

    def completed(self):
        self.fit.is_complete = True

    def save_summary(
        self,
        samples,
        latent_samples,
        log_likelihood_function_time,
        visualization_time = None,
    ):
        self.fit.instance = samples.max_log_likelihood()
        self.fit.max_log_likelihood = samples.max_log_likelihood_sample.log_likelihood

    def save_samples(self, samples):
        if not self.save_all_samples:
            samples = samples.minimise()

        self.fit.samples = samples
        self.fit.set_json("samples_info", samples.samples_info)

    def save_samples_summary(
        self, samples_summary: SamplesSummary, name="samples_summary"
    ):
        model = samples_summary.model

        filter_args = tuple(
            arg_name
            for arg_name in (
                "errors_at_sigma_1",
                "errors_at_sigma_3",
                "values_at_sigma_1",
                "values_at_sigma_3",
                "max_log_likelihood_sample",
                "median_pdf_sample",
            )
            if not should_output(arg_name)
        )

        samples_summary.model = None
        self.fit.set_json(
            name or "samples_summary",
            to_dict(samples_summary, filter_args=filter_args),
        )
        samples_summary.model = model

    def load_samples_summary(self) -> SamplesSummary:
        try:
            summary_dict = self.fit.get_json("samples_summary")
        except KeyError:
            raise FileNotFoundError(
                f"No samples_summary saved for fit {self.identifier}"
            )
        samples_summary = from_dict(summary_dict)
        samples_summary.model = self.model

        return samples_summary

    def save_latent_samples(self, latent_samples):
        if not self.save_all_samples:
            latent_samples = latent_samples.minimise()
        self.fit.latent_samples = latent_samples

    def load_latent_samples(self):
        return self.fit.latent_samples

    def _load_samples(self):
        samples = self.fit.samples
        samples.model = self.model
        return samples

    def load_samples(self):
        return self._load_samples().sample_list

    def load_samples_info(self):
        return self._load_samples().samples_info

    def save_all(self, info, *_, **kwargs):
        self.fit.info = info
        self.fit.model = self.model
        if info:
            self.save_json("info", info)
        self.save_json("search", to_dict(self.search))
        self.save_json("model", to_dict(self.model))

        self.session.commit()
        Info(self.session).write()
