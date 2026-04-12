from __future__ import annotations
import copy
import gc
import logging
import multiprocessing as mp
import numpy as np
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Tuple, List, Dict

import psutil

if TYPE_CHECKING:
    from autofit.non_linear.result import Result

from autoconf import conf, cached_property

from autoconf.output import should_output

from autofit import exc
from autofit.database.sqlalchemy_ import sa
from autofit.graphical import (
    MeanField,
    AnalysisFactor,
    _HierarchicalFactor,
    FactorApproximation,
)
from autofit.graphical.utils import Status
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.model import ModelInstance
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.parallel import SneakyPool, SneakierPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.paths.sub_directory_paths import SubDirectoryPaths
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.non_linear.timer import Timer
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.paths.null import NullPaths
from autofit.graphical.declarative.abstract import PriorFactor
from autofit.graphical.expectation_propagation import AbstractFactorOptimiser

from autofit.non_linear.fitness import get_timeout_seconds
from autofit.non_linear.test_mode import is_test_mode, test_mode_level, skip_fit_output

logger = logging.getLogger(__name__)


def check_cores(func):
    """
    Checks how many cores the search has been configured to
    use and then returns None instead of calling the pool
    creation function in the case that only one core has
    been set.

    Parameters
    ----------
    func
        A function that creates a pool

    Returns
    -------
    None or a pool
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.number_of_cores == 1:
            return None
        return func(self, *args, **kwargs)

    return wrapper


def configure_handler(func):
    """
    Add a file handler for logging during the course of the search.

    Optionally outputs 'search.log' to the search's output directory. Can be
    turned on or off in the output.yaml file.

    Parameters
    ----------
    func
        Some function for which logging should be output to file

    Returns
    -------
    A decorated version of the function
    """
    root_logger = logging.getLogger()

    def decorated(self, *args, **kwargs):
        if not should_output("search_log"):
            return func(self, *args, **kwargs)
        if self.disable_output:
            return func(self, *args, **kwargs)
        try:
            os.makedirs(
                self.paths.output_path,
                exist_ok=True,
            )
            handler = logging.FileHandler(self.paths.output_path / "search.log")
            root_logger.addHandler(handler)
        except AttributeError:
            return func(self, *args, **kwargs)

        try:
            return func(self, *args, **kwargs)
        finally:
            root_logger.removeHandler(handler)

    return decorated


class NonLinearSearch(AbstractFactorOptimiser, ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Initializer = None,
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = 1,
        session: Optional[sa.orm.Session] = None,
        paths: Optional[AbstractPaths] = None,
        **kwargs,
    ):
        """
        Abstract base class for non-linear searches.L

        This class sets up the file structure for the non-linear search, which are standardized across all non-linear
        searches.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        session
            An SQLAlchemy session instance so the results of the model-fit are written to an SQLite database.
        """
        super().__init__()

        if name is None and path_prefix is None:
            self.disable_output = True
        else:
            self.disable_output = False

        from autofit.non_linear.paths.database import DatabasePaths

        if name:
            path_prefix = Path(path_prefix or "")

        self.path_prefix = path_prefix

        self.path_prefix_no_unique_tag = path_prefix

        self._logger = None

        self.unique_tag = unique_tag

        if paths:
            self.paths = paths
        elif session is not None:
            logger.debug("Session found. Using database.")
            self.paths = DatabasePaths(
                name=name,
                path_prefix=path_prefix,
                session=session,
                save_all_samples=kwargs.get("save_all_samples", False),
                unique_tag=unique_tag,
            )
        elif name is not None or path_prefix:
            logger.debug("Session not found. Using directory output.")
            self.paths = DirectoryPaths(
                name=name, path_prefix=path_prefix, unique_tag=unique_tag
            )
        else:
            self.paths = NullPaths()

        self.force_pickle_overwrite = conf.instance["general"]["output"][
            "force_pickle_overwrite"
        ]

        self.force_visualize_overwrite = conf.instance["general"]["output"][
            "force_visualize_overwrite"
        ]

        if initializer is None:
            self.logger.debug("Creating initializer ")
            self.initializer = Initializer.from_config(config=self._config)
        else:
            self.initializer = initializer

        self.iterations_per_quick_update = float((iterations_per_quick_update or
            conf.instance["general"]["updates"]["iterations_per_quick_update"]))
        

        self.iterations_per_full_update = float((iterations_per_full_update or
            conf.instance["general"]["updates"]["iterations_per_full_update"]))

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.iterations_per_quick_update = float(conf.instance["general"]["hpc"][
                "iterations_per_quick_update"
            ])
            self.iterations_per_full_update = float(conf.instance["general"]["hpc"][
                "iterations_per_full_update"
            ])

        self.iterations = 0

        self.should_profile = conf.instance["general"]["profiling"]["should_profile"]

        self.silence = self._config("printing", "silence")

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.silence = True

        self.kwargs = kwargs

        for key, value in self.config_dict_search.items():
            setattr(self, key, value)

        try:
            for key, value in self.config_dict_run.items():
                setattr(self, key, value)
        except KeyError:
            pass

        self.number_of_cores = number_of_cores

        if number_of_cores > 1 and any(
            os.environ.get(key) != "1"
            for key in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            if conf.instance["general"]["parallel"]["warn_environment_variables"]:
                warnings.warn(exc.SearchWarning(""))
                logger.warning(
                    """
                        The non-linear search is using multiprocessing (number_of_cores>1). 
    
                        However, the following environment variables have not been set to 1:
    
                        OPENBLAS_NUM_THREADS
                        MKL_NUM_THREADS
                        OMP_NUM_THREADS
                        VECLIB_MAXIMUM_THREADS
                        NUMEXPR_NUM_THREADS
    
                        This can lead to performance issues, because both the non-linear search and libraries that may be
                        used in your `log_likelihood_function` evaluation (e.g. NumPy, SciPy, scikit-learn) may attempt to
                        parallelize over all cores available.
    
                        This will lead to slow-down, due to overallocation of tasks over the CPUs.
    
                        To mitigate this, set the environment variables to 1 via the following command on your
                        bash terminal / command line:
    
                        export OPENBLAS_NUM_THREADS=1
                        export MKL_NUM_THREADS=1
                        export OMP_NUM_THREADS=1
                        export VECLIB_MAXIMUM_THREADS=1
                        export NUMEXPR_NUM_THREADS=1
    
                        This means only the non-linear search is parallelized over multiple cores.
    
                        If you "know what you are doing" and do not want these environment variables to be set to one, you 
                        can disable this warning by changing the following entry in the config files:
    
                        `config -> general.yaml -> parallel: -> warn_environment_variables=False`
                        """
                )

        self.optimisation_counter = Counter()

    __identifier_fields__ = tuple()

    def optimise(
        self,
        factor_approx: FactorApproximation,
        status: Status = Status(),
    ) -> Tuple[MeanField, Status]:
        """
        Perform optimisation for expectation propagation. Currently only
        applicable for ModelFactors created by the declarative interface.

        1. Analysis and model classes are extracted from the factor.
        2. Priors are updated from the mean field.
        3. Analysis and model are fit as usual.
        4. A new mean field is constructed with the (posterior) 'linking' priors.
        5. Projection is performed to produce an updated EPMeanField object.

        Output directories are generated according to the factor and the number
        of the search. For example a factor called "factor" would output:

        factor/optimization_0/<identifier>
        factor/optimization_1/<identifier>
        factor/optimization_2/<identifier>

        For the first, second and third optimizations respectively.

        Parameters
        ----------
        factor_approx
            A collection of messages defining the current best approximation to
            some global model
        status

        Returns
        -------
        An updated approximation to the model having performed optimisation on
        a single factor.
        """

        factor = factor_approx.factor

        _ = status
        if not isinstance(factor, (AnalysisFactor, PriorFactor, _HierarchicalFactor)):
            raise NotImplementedError(
                f"Optimizer {self.__class__.__name__} can only be applied to"
                f" AnalysisFactors, HierarchicalFactors and PriorFactors"
            )

        model = factor.prior_model.mapper_from_prior_arguments(
            {
                prior: prior.with_message(message)
                for prior, message in factor_approx.cavity_dist.arguments.items()
            }
        )

        analysis = factor.analysis

        number = self.optimisation_counter[factor.name]

        self.optimisation_counter[factor.name] += 1

        self.paths = SubDirectoryPaths(
            parent=self.paths,
            analysis_name=f"{factor.name}/optimization_{number}",
            is_flat=True,
        )

        result = self.fit(model=model, analysis=analysis)

        new_model_dist = MeanField.from_priors(result.projected_model.priors)

        status.result = result

        return new_model_dist, status

    @property
    def name(self):
        return self.paths.name

    def __getstate__(self):
        """
        Remove the logger for pickling
        """
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        if "paths" in state:
            del state["paths"]
        return state

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = None
        if self._logger is None:
            logger_ = logging.getLogger(self.name)
            self._logger = logger_
        return self._logger

    @property
    def timer(self) -> Optional[Timer]:
        """
        Returns the timer of the search, which is used to output informaiton such as how long the search took and
        how much parallelization sped up the search time.

        If the search is running in `NullPaths` mode, meaning that no output is written to the hard-disk, the timer
        is disabled and a `None` is returned.

        Returns
        -------
        An object which times the non-linear search.
        """
        try:
            return Timer(self.paths.search_internal_path)
        except TypeError:
            pass

    @property
    def paths(self) -> Optional[AbstractPaths]:
        return self._paths

    @paths.setter
    def paths(self, paths: Optional[AbstractPaths]):
        if paths is not None:
            paths.search = self
        self._paths = paths

    def copy_with_paths(self, paths):
        self.logger.debug(f"Creating a copy of {self._paths.name}")
        search_instance = copy.copy(self)
        search_instance.paths = paths
        search_instance._logger = None

        return search_instance

    def fit(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        info: Optional[Dict] = None,
    ) -> Union[Result, List[Result]]:
        """
        Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness.

        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The search
        produces instances of the model by picking points in an N dimensional space.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        info
            Optional dictionary containing information about the fit that can be saved in the `files` folder
            (e.g. as `files/info.json`) and can be loaded via the database.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.

        Raises
        ------
        AssertionError
            If the model has 0 dimensions.
        """
        self.check_model(model=model)

        if getattr(analysis, "_use_jax", False):
            try:
                import jax
                devices = jax.devices()
                device = devices[0]
                backend = device.platform.upper()
                device_name = getattr(device, "device_kind", backend)
                logger.info(
                    f"Starting non-linear search with JAX ({backend}: {device_name})."
                )
            except Exception:
                logger.info("Starting non-linear search with JAX.")
        else:
            logger.info(f"Starting non-linear search with {self.number_of_cores} cores.")
        self._log_process_state()

        model = analysis.modify_model(model)
        self.paths.model = model
        self.paths.unique_tag = self.unique_tag

        self.paths.restore()

        model.freeze()
        analysis = analysis.modify_before_fit(paths=self.paths, model=model)
        model.unfreeze()

        if not skip_fit_output():
            self.pre_fit_output(
                analysis=analysis,
                model=model,
                info=info,
            )
        else:
            # Skip mode still needs the metadata + identifier files written
            # so downstream aggregator scraping can discover the search
            # directory. `save_all` is lightweight (a handful of JSON dumps)
            # and skips the expensive `analysis.save_attributes` /
            # `visualize_before_fit` calls that `pre_fit_output` would add.
            if hasattr(self.paths, "save_all"):
                self.paths.save_all(
                    search_config_dict=self.config_dict_search,
                    info=info,
                )

        if not self.paths.is_complete:
            result = self.start_resume_fit(
                analysis=analysis,
                model=model,
            )
        else:
            result = self.result_via_completed_fit(
                analysis=analysis,
                model=model,
            )

        if not skip_fit_output():
            analysis = analysis.modify_after_fit(
                paths=self.paths, model=model, result=result
            )

            self.post_fit_output(
                search_internal=result.search_internal,
            )

        gc.collect()

        self.logger.info("Search complete, returning result")

        return result

    @staticmethod
    def _log_process_state():
        total_files = 0

        for process in psutil.process_iter(attrs=["pid"]):
            try:
                proc_info = process.as_dict(attrs=["pid"])
                logger.debug(
                    f"Process ID: {proc_info['pid']} has the following open files:"
                )

                open_files = process.open_files()
                for file in open_files:
                    logger.debug(file)
                    total_files += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if conf.instance["logging"]["total_files_open"]:
            logger.info(f"Total Files Open: {total_files}")

    def pre_fit_output(
        self, analysis: Analysis, model: AbstractPriorModel, info: Optional[Dict] = None
    ):
        """
        Outputs attributes of fit before the non-linear search begins.

        The following attributes of a fit may be output before the search begins:

        - The model composition, which is output as a .json file (`files/model.json`).

        - The non-linear search settings, which are output as a .json file (`files/search.json`).

        - Custom attributes of the analysis defined via the `save_attributes` method of the analysis class, for
        example the data (e.g. `files/data.json`).

        - Custom Visualization associated with the analysis, defined via the `visualize_before_fit`
        and `visualize_before_fit_combined` methods. This is typically quantities that do not change during the
        model-fit (e.g. the data).

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        info
            Optional dictionary containing information about the fit that can be saved in the `files` folder
            (e.g. as `files/info.json`) and can be loaded via the database.
        """

        if not self.disable_output:
            self.logger.info(f"The output path of this fit is {self.paths.output_path}")
        else:
            self.logger.info(
                "Output to hard-disk disabled, input a search name to enable."
            )

        if not self.paths.is_complete or self.force_pickle_overwrite:
            if not self.disable_output:
                self.logger.info(
                    f"Outputting pre-fit files (e.g. model.info, visualization)."
                )

            self.paths.save_all(
                search_config_dict=self.config_dict_search,
                info=info,
            )
            analysis.save_attributes(paths=self.paths)

        if analysis.should_visualize(paths=self.paths):
            analysis.visualize_before_fit(
                paths=self.paths,
                model=model,
            )
            analysis.visualize_before_fit_combined(
                paths=self.paths,
                model=model,
            )

        timeout_seconds = get_timeout_seconds()

        if timeout_seconds is not None:
            logger.info(
                f"\n\n ***Log Likelihood Function timeout is "
                f"turned on and set to {timeout_seconds} seconds.***\n"
            )

    @configure_handler
    def start_resume_fit(self, analysis: Analysis, model: AbstractPriorModel) -> Result:
        """
        Start a non-linear search from scratch, or resumes one which was previously terminated mid-way through.

        If the search is resumed, the model-fit will begin by loading the samples from the previous search and
        from where it left off.

        After the search is completed, a `.completed` file is output so that if the search is resumed in the future
        it is not repeated and results are loaded via the `update_completed_fit` method.

        Results are also output to hard-disk in the `files` folder via the `save_results` method of the analysis class.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.

        Returns
        -------
        The result of the non-linear search, which includes the best-fit model instance and best-fit log likelihood
        and errors on the model parameters.
        """
        if not isinstance(self.paths, DatabasePaths) and not isinstance(
            self.paths, NullPaths
        ):
            self.timer.start()

        mode = test_mode_level()
        if mode >= 2:
            return self._fit_bypass_test_mode(
                model=model,
                analysis=analysis,
                call_likelihood=(mode == 2),
            )

        model.freeze()
        search_internal, fitness = self._fit(
            model=model,
            analysis=analysis,
        )

        samples = self.perform_update(
            model=model,
            analysis=analysis,
            search_internal=search_internal,
            fitness=fitness,
            during_analysis=False,
        )

        result = analysis.make_result(
            samples_summary=samples.summary(),
            paths=self.paths,
            samples=samples,
            search_internal=search_internal,
        )

        analysis.save_results(paths=self.paths, result=result)
        analysis.save_results_combined(paths=self.paths, result=result)

        model.unfreeze()

        self.paths.completed()

        return result

    def result_via_completed_fit(
        self,
        analysis: Analysis,
        model: AbstractPriorModel,
    ) -> Result:
        """
        Returns the result of the non-linear search of a completed model-fit.

        The result contains the non-linear search samples summary, which contains the maximum log likelihood instance
        that is used for visualization and prior passing via the search chaining API.

        This funciton may also load the full samples of the completed fit, for example if visualization of the
        seatch chains (e.g. a corner plot) is performed. This task is optional and be slow due to loading times.

        Optional tasks can be performed to update the results of the model-fit on hard-disk depending on the following
        entries of the `general.yaml` config file's `output` section:

        ` `force_visualize_overwrite=True`: the visualization of the model-fit is performed again (e.g. to
        add new visualizations or replot figures with a different source code).

        - `force_pickle_overwrite=True`: the output files of the model-fit are recreated (e.g. to add a new attribute
        that was previously not output).

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.

        Returns
        -------
        The result of the non-linear search, which includes the best-fit model instance and best-fit log likelihood
        and errors on the model parameters.
        """

        model.freeze()
        samples_summary = self.paths.load_samples_summary()
        try:
            samples = self.paths.samples
        except FileNotFoundError:
            samples = None

        result = analysis.make_result(
            samples_summary=samples_summary,
            samples=samples,
            paths=self.paths,
        )

        self.logger.info(f"Fit Already Completed: skipping non-linear search.")

        if self.force_visualize_overwrite:
            self.perform_visualization(
                model=model,
                analysis=analysis,
                samples_summary=samples_summary,
                during_analysis=False,
            )

        if self.force_pickle_overwrite:
            self.logger.info("Forcing pickle overwrite")

            analysis.save_results(paths=self.paths, result=result)
            analysis.save_results_combined(paths=self.paths, result=result)

        model.unfreeze()

        return result

    def post_fit_output(self, search_internal):
        """
        Cleans up the output folderds after a completed non-linear search.

        The main task this performs is removing the folder containing the results of a non-linear search such that only
        its corresponding `.zip` file is left. This is use for supercomputers, where users often have a file limit on
        the number of files they can store in their home directory, so storing them all in just a .zip file is
        advantageous.

        This only occurs if `remove_files=False` in the `general.yaml` config file's `output` section.

        Parameters
        ----------
        search_internal
            The internal search.
        """
        if not conf.instance["output"]["search_internal"]:
            self.logger.info("Removing search internal folder.")
            self.paths.remove_search_internal()
        elif search_internal is not None:
            self.output_search_internal(search_internal=search_internal)

        if not self.disable_output:
            self.logger.info("Removing all files except for .zip file")

        self.paths.zip_remove()

    def _fit_bypass_test_mode(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        call_likelihood: bool = True,
    ):
        """
        Bypass the sampler entirely in test mode (levels 2 and 3).

        Generates fake samples and writes all expected output files so that
        downstream code sees a complete result folder.

        Parameters
        ----------
        model
            The model being fitted.
        analysis
            The analysis object with the log likelihood function.
        call_likelihood
            If True (mode 2), call the likelihood function once to verify it
            works. If False (mode 3), skip the likelihood call entirely.
        """
        from autofit.non_linear.samples.pdf import SamplesPDF
        from autofit.non_linear.samples.sample import Sample

        mode = test_mode_level()
        if mode == 2:
            logger.warning(
                "TEST MODE 2 (bypass + likelihood): Skipping sampler, "
                "calling likelihood function once to verify it works."
            )
        else:
            logger.warning(
                "TEST MODE 3 (full bypass): Skipping sampler and likelihood "
                "entirely for maximum speed. No likelihood verification."
            )

        model.freeze()

        unit_vector = [0.5] * model.prior_count
        parameter_vector = [
            float(v) for v in model.vector_from_unit_vector(
                unit_vector=unit_vector,
            )
        ]

        log_likelihood = -1.0e99
        if call_likelihood:
            instance = model.instance_from_vector(vector=parameter_vector)
            log_likelihood = float(
                analysis.log_likelihood_function(instance)
            )

        sample_list = self._build_fake_samples(
            model=model,
            parameter_vector=parameter_vector,
            log_likelihood=log_likelihood,
        )

        # Stub log_evidence in samples_info so downstream arithmetic
        # (grid search log_evidences, subhalo Bayesian model comparison,
        # scrape aggregator assertions) doesn't crash on None. SamplesPDF
        # reads log_evidence from samples_info.
        samples = SamplesPDF(
            model=model,
            sample_list=sample_list,
            samples_info={
                "total_iterations": 1,
                "time": 0.0,
                "log_evidence": log_likelihood,
            },
        )

        samples_summary = samples.summary()

        # Persist samples + summary to disk so downstream code that reads
        # from the output folder (database scrape, paths.load_samples_summary)
        # sees a complete result. Matches the docstring's promise. NullPaths
        # and DatabasePaths both handle these calls safely.
        self.paths.save_samples_summary(samples_summary=samples_summary)
        self.paths.save_samples(samples=samples)

        result = analysis.make_result(
            samples_summary=samples_summary,
            paths=self.paths,
            samples=samples,
            search_internal=None,
        )

        model.unfreeze()

        return result

    @staticmethod
    def _build_fake_samples(model, parameter_vector, log_likelihood):
        """
        Build a minimal list of fake Sample objects for test mode bypass.

        Creates two samples: the "best" at the prior median and a second
        with slightly perturbed parameters and worse likelihood, so that
        SamplesPDF methods like median_pdf work correctly.
        """
        from autofit.non_linear.samples.sample import Sample

        perturbed = [
            p * 1.001 if p != 0.0 else 0.001 for p in parameter_vector
        ]

        return Sample.from_lists(
            model=model,
            parameter_lists=[parameter_vector, perturbed],
            log_likelihood_list=[log_likelihood, log_likelihood - 1.0],
            log_prior_list=[0.0, 0.0],
            weight_list=[1.0, 0.5],
        )

    @abstractmethod
    def _fit(self, model: AbstractPriorModel, analysis: Analysis):
        pass

    def check_model(self, model: AbstractPriorModel):
        if model is not None and model.prior_count == 0:
            raise AssertionError("Model has no priors! Cannot fit a 0 dimension model.")

    def config_dict_test_mode_from(self, config_dict: Dict) -> Dict:
        raise NotImplementedError

    @property
    def _class_config(self) -> Dict:
        return self.config_type[self.__class__.__name__]

    @cached_property
    def config_dict_search(self) -> Dict:
        config_dict = copy.deepcopy(self._class_config["search"])

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        return config_dict

    @cached_property
    def config_dict_run(self) -> Dict:
        config_dict = copy.deepcopy(self._class_config["run"])

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        if is_test_mode():
            logger.warning(
                "TEST MODE 1 (reduced iterations): Sampler will run with "
                "minimal iterations for faster completion."
            )

            config_dict = self.config_dict_test_mode_from(config_dict=config_dict)

        return config_dict

    @property
    def config_dict_settings(self) -> Dict:
        return self._class_config["settings"]

    @property
    def config_type(self):
        raise NotImplementedError()

    def _config(self, section, attribute_name):
        """
        Get a config field from this search's section in non_linear.ini by a key and value type.

        Parameters
        ----------
        attribute_name: str
            The analysis_path of the field

        Returns
        -------
        attribute
            An attribute for the key with the specified type.
        """
        return self._class_config[section][attribute_name]

    def output_search_internal(self, search_internal):
        self.paths.save_search_internal(
            obj=search_internal,
        )

    def perform_update(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        during_analysis: bool,
        fitness: Optional[Fitness] = None,
        search_internal=None,
    ) -> Samples:
        """
        Perform an update of the non-linear search's model-fitting results.

        This occurs every `iterations_per_full_update` of the non-linear search and once it is complete.

        The update performs the following tasks (if the settings indicate they should be performed):

        1) Visualize the search results.
        2) Visualize the maximum log likelihood model using model-specific visualization implented via the `Analysis`
           object.
        3) Perform profiling of the analysis object `log_likelihood_function` and ouptut run-time information.
        4) Output the `search.summary` file which contains information on model-fitting so far.
        5) Output the `model.results` file which contains a concise text summary of the model results so far.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        during_analysis
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
            of updates and only a subset of visualization may be performed.
        """
        self.iterations += self.iterations_per_full_update

        if not self.disable_output:
            self.logger.info(
                f"""Fit Running: Updating results (see output folder)."""
            )

        if not isinstance(self.paths, DatabasePaths) and not isinstance(
            self.paths, NullPaths
        ):
            self.timer.update()

        samples = self.samples_from(model=model, search_internal=search_internal)
        samples_summary = samples.summary()

        try:
            instance = samples_summary.instance
        except exc.FitException:
            return samples

        self.paths.save_samples_summary(samples_summary=samples_summary)

        samples_save = samples

        log_message = True

        if during_analysis:
            log_message = False
        elif self.disable_output:
            log_message = False

        samples_save = samples_save.samples_above_weight_threshold_from(
            log_message=log_message
        )
        self.paths.save_samples(samples=samples_save)

        latent_samples = None

        if (during_analysis and conf.instance["output"]["latent_during_fit"]) or (
            not during_analysis and conf.instance["output"]["latent_after_fit"]
        ):

            if conf.instance["output"]["latent_draw_via_pdf"]:

                total_draws = conf.instance["output"]["latent_draw_via_pdf_size"]

                logger.info(f"Creating latent samples by drawing {total_draws} from the PDF.")

                try:
                    latent_samples = samples.samples_drawn_randomly_via_pdf_from(total_draws=total_draws)
                except AttributeError:
                    latent_samples = samples_save
                    logger.info(
                        "Drawing via PDF not available for this search, "
                        "using all samples above the samples weight threshold instead."
                        "")

            else:

                logger.info(f"Creating latent samples using all samples above the samples weight threshold.")

                latent_samples = samples_save

            latent_samples = analysis.compute_latent_samples(
                latent_samples,
                batch_size=fitness.batch_size
            )

            if latent_samples:
                if not conf.instance["output"]["latent_draw_via_pdf"]:
                    self.paths.save_latent_samples(latent_samples)
                self.paths.save_samples_summary(
                    latent_samples.summary(),
                    "latent/latent_summary",
                )

        start = time.time()

        self.perform_visualization(
            model=model,
            analysis=analysis,
            samples_summary=samples_summary,
            during_analysis=during_analysis,
            search_internal=search_internal,
        )

        visualization_time = time.time() - start

        if self.should_profile:

            self.logger.debug("Profiling Maximum Likelihood Model")

            analysis.profile_log_likelihood_function(
                paths=self.paths,
                instance=instance,
            )

        self.logger.debug("Outputting model result")

        try:

            parameters = samples.max_log_likelihood(as_instance=False)

            start = time.time()
            figure_of_merit = fitness.call_wrap(parameters)

            # account for asynchronous JAX calls
            np.array(figure_of_merit)

            log_likelihood_function_time = time.time() - start

            self.paths.save_summary(
                samples=samples,
                latent_samples=latent_samples,
                log_likelihood_function_time=log_likelihood_function_time,
                visualization_time=visualization_time,
            )

        except exc.FitException:
            pass

        self._log_process_state()

        return samples

    def perform_visualization(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        during_analysis: bool,
        samples_summary: Optional[SamplesSummary] = None,
        instance: Optional[ModelInstance] = None,
        paths_override: Optional[AbstractPaths] = None,
        search_internal=None,
    ):
        """
        Perform visualization of the non-linear search's model-fitting results.

        This occurs every `iterations_per_full_update` of the non-linear search, when the search is complete and can
        also be forced to occur even though a search is completed on a rerun, to update the visualization
        with different `matplotlib` settings.

        The update performs the following tasks (if the settings indicate they should be performed):

        1) Visualize the maximum log likelihood model using model-specific visualization implented via the `Analysis`
           object.
        2) Visualize the search results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        samples_summary
            The summary of the samples of the non-linear search, which are used for visualization.
        during_analysis
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
            of updates and only a subset of visualization may be performed.
        instance
            The instance of the model that is used for visualization. If not input, the maximum log likelihood
            instance from the samples is used.
        """

        self.logger.debug("Visualizing")

        paths = paths_override or self.paths

        if instance is None and samples_summary is None:
            raise AssertionError(
                """
                The search's perform_visualization method has been called without an input instance or 
                samples_summary. 

                This should not occur, please ensure one of these inputs is provided.
                """
            )

        if instance is None:
            instance = samples_summary.instance

        if analysis.should_visualize(paths=paths, during_analysis=during_analysis):
            analysis.visualize(
                paths=paths,
                instance=instance,
                during_analysis=during_analysis,
            )
            analysis.visualize_combined(
                paths=paths,
                instance=instance,
                during_analysis=during_analysis,
            )

        if analysis.should_visualize(paths=paths, during_analysis=during_analysis):
            if not isinstance(paths, NullPaths):
                try:
                    samples = self.samples_from(
                        model=model, search_internal=search_internal
                    )

                    self.plot_results(samples=samples)
                except FileNotFoundError:
                    pass

    @property
    def should_plot_start_point(self) -> bool:
        return conf.instance["output"]["start_point"]

    def plot_start_point(
        self,
        parameter_vector: List[float],
        model: AbstractPriorModel,
        analysis: Analysis,
    ):
        """
        Visualize the starting point of the non-linear search, using an instance of the model at the starting point
        of the maximum likelihood estimator.

        Plots are output to a folder named `image_start` in the output path, so that the starting point model
        can be compared to the final model inferred by the non-linear search.

        Parameters
        ----------
        model
            The model used by the non-linear search
        analysis
            The analysis which contains the visualization methods which plot the starting point model.

        Returns
        -------

        """

        if not self.should_plot_start_point:
            return

        self.logger.info(f"Visualizing Starting Point Model in image_start folder.")

        instance = model.instance_from_vector(vector=parameter_vector)
        paths = copy.copy(self.paths)
        paths.image_path_suffix = "_start"

        self.perform_visualization(
            model=model,
            analysis=analysis,
            instance=instance,
            during_analysis=False,
            paths_override=paths,
        )

    @property
    def samples_cls(self):
        raise NotImplementedError()

    def samples_from(self, model: AbstractPriorModel, search_internal=None) -> Samples:
        """
        Loads the samples of a non-linear search from its output files.

        The samples can be loaded from one of two files, which are attempted to be loading in the following order:

        1) Load via the internal results of the non-linear search, which are specified to that search's outputs
           (e.g. the .hdf file output by the MCMC method `emcee`).

        2) Load via the `samples.csv` and `samples_info.json` files of the search, which are outputs that are the
           same for all non-linear searches as they are homogenized by autofit.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        """
        try:
            return self.samples_via_internal_from(
                model=model, search_internal=search_internal
            )
        except (FileNotFoundError, NotImplementedError, AttributeError):
            return self.paths.samples

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        raise NotImplementedError

    @check_cores
    def make_pool(self):
        """Make the pool instance used to parallelize a `NonLinearSearch` alongside a set of unique ids for every
        process in the pool. If the specified number of cores is 1, a pool instance is not made and None is returned.

        The pool cannot be set as an attribute of the class itself because this prevents pickling, thus it is generated
        via this function before calling the non-linear search.

        The pool instance is also set up with a list of unique pool ids, which are used during model-fitting to
        identify a 'master core' (the one whose id value is lowest) which handles model result output, visualization,
        etc."""
        self.logger.info("...using pool")
        return mp.Pool(processes=self.number_of_cores)

    @check_cores
    def make_sneaky_pool(self, fitness: Fitness) -> Optional[SneakyPool]:
        """
        Create a pool for multiprocessing that uses slight-of-hand
        to avoid copying the fitness function between processes
        multiple times.

        Parameters
        ----------
        fitness
            An instance of a fitness class used to evaluate the
            likelihood that a particular model is correct

        Returns
        -------
        An implementation of a multiprocessing pool
        """

        self.logger.warning(
            "...using SneakyPool. This copies the likelihood function "
            "to each process on instantiation to avoid copying multiple "
            "times."
        )
        return SneakyPool(
            processes=self.number_of_cores, paths=self.paths, fitness=fitness
        )

    def make_sneakier_pool(self, fitness_function: Fitness, **kwargs) -> SneakierPool:

        self.logger.info(f"number of cores == {self.number_of_cores}")

        if self.number_of_cores > 1:
            self.logger.info("Creating SneakierPool...")
        else:
            self.logger.info("Creating multiprocessing Pool of size 1...")

        pool = SneakierPool(
            processes=self.number_of_cores, fitness=fitness_function, **kwargs
        )

        return pool

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__

    def plot_results(self, samples):
        raise NotImplementedError
