import configparser
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional

import numpy as np

from autofit import exc
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.parallel import SneakyPool

logger = logging.getLogger(__name__)


IDENTICAL_FIGURES_OF_MERIT_MESSAGE = """
                The initial samples all have the same figure of merit (e.g. log likelihood values).

                The non-linear search will therefore not progress correctly.

                Possible causes for this behaviour are:

                - The `log_likelihood_function` of the analysis class is defined incorrectly.
                - The model parameterization creates numerically inaccurate log likelihoods.
                - The model is so tightly constrained that every drawn point is effectively the
                  same point. This is the usual cause when the search is a factor optimiser
                  inside an outer loop that updates its priors, e.g. expectation propagation.

                Note that this is a check for *identical* figures of merit, made with
                `np.allclose`, which is `False` for `nan`. `nan` draws are also discarded
                by `figure_of_metric` before they ever reach the check. An all-`nan`
                `log_likelihood_function` therefore cannot raise this exception and is not
                a possible cause of it.
                """


class AbstractInitializer(ABC):

    @abstractmethod
    def _generate_unit_parameter_list(self, model):
        pass

    def info_from_model(self, model : AbstractPriorModel) -> str:
        raise NotImplementedError

    @staticmethod
    def figure_of_metric(args) -> Optional[float]:
        fitness, parameter_list = args
        try:
            figure_of_merit = fitness(parameters=parameter_list)

            if np.isnan(figure_of_merit) or figure_of_merit < -1e98:
                return None

            return float(figure_of_merit)
        except exc.FitException:
            return None

    def samples_from_model(
            self,
            total_points: int,
            model: AbstractPriorModel,
            fitness,
            paths: AbstractPaths,
            use_prior_medians: bool = False,
            test_mode_samples: bool = True,
            n_cores: int = 1,
    ):
        """
        Generate the initial points of the non-linear search, by randomly drawing unit values from a uniform
        distribution between the ball_lower_limit and ball_upper_limit values.

        Parameters
        ----------
        total_points
            The number of points in non-linear paramemter space which initial points are created for.
        model
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.
        """

        if is_test_mode() and test_mode_samples:
            return self.samples_in_test_mode(total_points=total_points, model=model)

        if n_cores == 1:
            return self.samples_jax(
                total_points=total_points,
                model=model,
                fitness=fitness,
                use_prior_medians=use_prior_medians
            )

        unit_parameter_lists = []
        parameter_lists = []
        figures_of_merit_list = []

        sneaky_pool = SneakyPool(n_cores, fitness, paths)

        logger.info(f"Generating initial samples of model using {n_cores} cores")

        while len(figures_of_merit_list) < total_points:
            remaining_points = total_points - len(figures_of_merit_list)
            batch_size = min(remaining_points, n_cores)
            parameter_lists_ = []
            unit_parameter_lists_ = []

            for _ in range(batch_size):
                if not use_prior_medians:
                    unit_parameter_list = self._generate_unit_parameter_list(model)
                else:
                    unit_parameter_list = [0.5] * model.prior_count

                parameter_list = model.vector_from_unit_vector(
                    unit_vector=unit_parameter_list
                )

                parameter_lists_.append(parameter_list)
                unit_parameter_lists_.append(unit_parameter_list)

            for figure_of_merit, unit_parameter_list, parameter_list in zip(
                    sneaky_pool.map(
                        function=self.figure_of_metric,
                        args_list=[(fitness, parameter_list) for parameter_list in parameter_lists_],
                        log_info=False
                    ),
                    unit_parameter_lists_,
                    parameter_lists_,
            ):
                if figure_of_merit is not None:
                    unit_parameter_lists.append(unit_parameter_list)
                    parameter_lists.append(parameter_list)
                    figures_of_merit_list.append(figure_of_merit)

        if total_points > 1 and np.allclose(
                a=figures_of_merit_list[0], b=figures_of_merit_list[1:]
        ):
            raise exc.InitializerException(IDENTICAL_FIGURES_OF_MERIT_MESSAGE)

        logger.info(f"Initial samples generated, starting non-linear search")

        return unit_parameter_lists, parameter_lists, figures_of_merit_list

    def samples_jax(
            self,
            total_points: int,
            model: AbstractPriorModel,
            fitness,
            use_prior_medians: bool = False,
    ):
        """
        Generate the initial points of the non-linear search, by randomly drawing unit values from a uniform
        distribution between the ball_lower_limit and ball_upper_limit values.

        Parameters
        ----------
        total_points
            The number of points in non-linear paramemter space which initial points are created for.
        model
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.
        """

        unit_parameter_lists = []
        parameter_lists = []
        figures_of_merit_list = []

        logger.info(f"Generating initial samples of model using JAX LH Function cores")

        while len(figures_of_merit_list) < total_points:

            if not use_prior_medians:
                unit_parameter_list = self._generate_unit_parameter_list(model)
            else:
                unit_parameter_list = [0.5] * model.prior_count

            parameter_list = model.vector_from_unit_vector(
                unit_vector=unit_parameter_list
            )

            figure_of_merit = self.figure_of_metric((fitness, parameter_list))

            if figure_of_merit is not None:
                unit_parameter_lists.append(unit_parameter_list)
                parameter_lists.append(parameter_list)
                figures_of_merit_list.append(figure_of_merit)

        if total_points > 1 and np.allclose(
                a=figures_of_merit_list[0], b=figures_of_merit_list[1:]
        ):
            raise exc.InitializerException(IDENTICAL_FIGURES_OF_MERIT_MESSAGE)

        logger.info(f"Initial samples generated, starting non-linear search")

        return unit_parameter_lists, parameter_lists, figures_of_merit_list

    def samples_in_test_mode(self, total_points: int, model: AbstractPriorModel):
        """
        Generate the initial points of the non-linear search in test mode. Like normal, test model draws points, by
        randomly drawing unit values from a uniform distribution between the ball_lower_limit and ball_upper_limit
        values.

        However, the log likelihood function is bypassed and all likelihoods are returned with a value -1.0e99. This
        is so that integration testing of large-scale model-fitting projects can be performed efficiently by bypassing
        sampling of points using the `log_likelihood_function`.

        Parameters
        ----------
        total_points
            The number of points in non-linear paramemter space which initial points are created for.
        model
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.
        """

        logger.warning(
            "TEST MODE 1 (reduced iterations): Initial samples assigned "
            "arbitrary large likelihoods to accelerate sampler convergence."
        )

        unit_parameter_lists = []
        parameter_lists = []
        figure_of_merit_list = []

        point_index = 0

        figure_of_merit = -1.0e99

        while point_index < total_points:
            try:
                unit_parameter_list = self._generate_unit_parameter_list(model)
                parameter_list = model.vector_from_unit_vector(
                    unit_vector=unit_parameter_list
                )
                model.instance_from_vector(vector=parameter_list)
                unit_parameter_lists.append(unit_parameter_list)
                parameter_lists.append(parameter_list)
                figure_of_merit_list.append(figure_of_merit)
                figure_of_merit *= 10.0
                point_index += 1
            except exc.FitException:
                pass

        return unit_parameter_lists, parameter_lists, figure_of_merit_list


class InitializerParamBounds(AbstractInitializer):
    def __init__(
        self,
        parameter_dict: Dict[Prior, Tuple[float, float]],
        lower_limit=0.0,
        upper_limit=1.0,
    ):
        """
        Initializer which uses the bounds on input parameters as the starting point for the search (e.g. where
        an MLE optimization starts or MCMC walkers are initialized).

        Parameters
        ----------
        parameter_dict
            A dictionary mapping each parameter path to bounded ranges of physical values that
            are where the search begins.
        lower_limit
            A default, unit lower limit used when a prior is not specified
        upper_limit
            A default, unit upper limit used when a prior is not specified
        """
        
        self.parameter_dict = parameter_dict
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        self._generated_warnings = set()

    def _generate_unit_parameter_list(self, model: AbstractPriorModel) -> List[float]:
        """
        Generate a unit vector for the model. The default limits are used for any
        priors which the model has but are not found in the parameter dict.

        Parameters
        ----------
        model
            A model for which initial points are required

        Returns
        -------
        A unit vector
        """

        unit_parameter_list = []
        for prior in model.priors_ordered_by_id:

            try:
                lower, upper = map(prior.unit_value_for, self.parameter_dict[prior])
                value = random.uniform(lower, upper)
            except KeyError:
                key = ".".join(model.path_for_prior(prior))
                if key not in self._generated_warnings:
                    logger.warning(
                        f"Range for {key} not set in the InitializerParamBounds. "
                        f"Using defaults."
                    )
                    self._generated_warnings.add(key)

                lower = self.lower_limit
                upper = self.upper_limit

                value = prior.unit_value_for(prior.random(lower, upper))

            unit_parameter_list.append(value)

        return unit_parameter_list

    def info_from_model(self, model : AbstractPriorModel) -> str:
        """
        Returns a string showing the bounds of the parameters in the initializer.
        """
        info = "Total Free Parameters = " + str(model.prior_count) + "\n"
        info += "Total Starting Points = " + str(len(self.parameter_dict)) + "\n\n"
        for prior in model.priors_ordered_by_id:

            key = ".".join(model.path_for_prior(prior))

            try:

                value = self.info_value_from(self.parameter_dict[prior])

                info += f"{key}: Start[{value}]\n"

            except KeyError:

                info += f"{key}: {prior})\n"

        return info

    def info_value_from(self, value : Tuple[float, float]) -> Tuple[float, float]:
        """
        Returns the value that is used to display the bounds of the parameters in the initializer.

        This function simply returns the input value, but it can be overridden in subclasses for diffferent
        initializers.

        Parameters
        ----------
        value
            The value to be displayed in the initializer info which is a tuple of the lower and upper bounds of the
            parameter.
        """
        return value


class InitializerParamStartPoints(InitializerParamBounds):
    def __init__(
        self,
        parameter_dict: Dict[Prior, float],
    ):
        """
        Initializer which input values of the parameters as the starting point for the search (e.g. where
        an MLE optimization starts or MCMC walkers are initialized).

        Parameters
        ----------
        parameter_dict
            A dictionary mapping each parameter path to the starting point physical values that
            are where the search begins.
        lower_limit
            A default, unit lower limit used when a prior is not specified
        upper_limit
            A default, unit upper limit used when a prior is not specified
        """
        parameter_dict_new = {}

        for key, value in parameter_dict.items():
            parameter_dict_new[key] = (value - 1.0e-8, value + 1.0e-8)

        super().__init__(parameter_dict=parameter_dict_new)

        # Set only by `from_result` when `n_points > 1` (or `jitter != 0.0`).
        # `None` preserves the exact single-point behaviour above unchanged.
        self._point_dicts: Optional[List[Dict[Prior, float]]] = None
        self._point_index: int = 0

    def info_value_from(self, value : Tuple[float, float]) -> float:
        """
        Returns the value that is used to display the starting point of the parameters in the initializer.

        This function returns the mean of the input value, as the starting point is a single value in the center of the
        bounds.

        Parameters
        ----------
        value
            The value to be displayed in the initializer info which is a tuple of the lower and upper bounds of the
            parameter.
        """
        return (value[1] + value[0]) / 2.0

    @classmethod
    def from_result(
        cls,
        result,
        model: Optional[AbstractPriorModel] = None,
        point: str = "max_log_likelihood",
        n_points: int = 1,
        jitter: float = 0.0,
        seed: int = 0,
    ) -> "InitializerParamStartPoints":
        """
        Build an initializer whose starting point(s) are taken from a previous `Result`, for example to
        warm-start a search (e.g. `BlackJAXNUTS`) from an earlier fit's best-fit point(s).

        This maps the previous result's inferred parameter values onto the (possibly different) target
        model **by prior path**, never by list index or by touching prior objects directly. This is
        deliberately distinct from `Result.model_centred*`, which *rewrite* priors (e.g. to `GaussianPrior`s
        centred on the previous best-fit); `from_result` only ever reads values off `result.samples` and
        writes a `{Prior: float}` starting-point dictionary, so the scientific priors of `model` (or, if not
        given, `result.model`) are left completely untouched.

        Parameters
        ----------
        result
            The previous `Result` (or any object exposing `.samples` and `.model`) start points are drawn
            from.
        model
            The model the start point(s) are mapped onto. If `None`, `result.model` is used (the common
            case: continuing to fit the same model with a different / warm-started search).
        point
            Which point of `result.samples` to draw the starting values from: `"max_log_likelihood"` (the
            default) or `"median_pdf"`.
        n_points
            The number of starting points to generate (e.g. one per NUTS chain). `n_points=1` (the default)
            reproduces the exact single-point `InitializerParamStartPoints` behaviour (a fixed +-1e-8
            unit-space jitter).
        jitter
            For `n_points > 1` (or when explicitly non-zero), each of the `n_points` starting values is
            offset in *physical* parameter space by `jitter * sigma * N(0, 1)`, where `sigma` is the
            corresponding diagonal entry of `result.samples.covariance_matrix` (its square root) when that
            covariance is available and finite; otherwise `jitter` is interpreted as an absolute physical
            offset (`jitter * N(0, 1)`). `jitter=0.0` (the default) still applies a fixed, tiny (1e-8)
            offset so that `n_points > 1` starting points are not bit-identical.
        seed
            Seed for the `numpy.random.RandomState` used to draw the (reproducible) jitter offsets.

        Returns
        -------
        InitializerParamStartPoints
            An initializer that draws its `n_points` starting vectors, in order, from the previous result.

        Raises
        ------
        InitializerException
            If `result.samples` is unavailable, if `point` is not a recognised option, or if the source
            and target models cannot be matched by path (dimension mismatch or an unmatched path).
        """
        samples = result.samples

        if samples is None:
            raise exc.InitializerException(
                "InitializerParamStartPoints.from_result: `result.samples` is None "
                "(no samples available in memory or on disk) -- cannot build start points."
            )

        if point == "max_log_likelihood":
            vector = samples.max_log_likelihood(as_instance=False)
        elif point == "median_pdf":
            vector = samples.median_pdf(as_instance=False)
        else:
            raise exc.InitializerException(
                "InitializerParamStartPoints.from_result: `point` must be "
                f"'max_log_likelihood' or 'median_pdf', got {point!r}."
            )

        # The source model is `samples.model` (not `result.model`, which for a `Result` is a *freshly
        # constructed* mapper via `mapper_via_defaults_from()` and therefore has priors with different
        # `id`s / ordering). `vector` above is ordered consistently with `samples.model.priors_ordered_by_id`
        # (both derive from the same `Samples` object), so path-matching must start from that same model.
        source_model = samples.model
        source_priors = source_model.priors_ordered_by_id

        if len(source_priors) != len(vector):
            raise exc.InitializerException(
                "InitializerParamStartPoints.from_result: the source result's samples vector has "
                f"{len(vector)} entries but its model has {len(source_priors)} priors -- cannot map by path."
            )

        value_by_source_path = {}
        for prior, value in zip(source_priors, vector):
            path = source_model.path_for_prior(prior)
            if path is None:
                raise exc.InitializerException(
                    "InitializerParamStartPoints.from_result: no path found for a prior in the source "
                    "result's model; cannot map its value by path."
                )
            value_by_source_path[path] = float(value)

        target_model = model if model is not None else result.model
        target_priors = target_model.priors_ordered_by_id

        base_dict = {}
        missing_paths = []
        for prior in target_priors:
            path = target_model.path_for_prior(prior)
            if path is None or path not in value_by_source_path:
                missing_paths.append(path)
                continue
            base_dict[prior] = value_by_source_path[path]

        if missing_paths:
            raise exc.InitializerException(
                "InitializerParamStartPoints.from_result: could not match the following target-model "
                f"prior paths to the source result's ({point}) values: {missing_paths}. The target "
                "model's free parameters must live at the same paths as the source result's model."
            )

        if n_points < 1:
            raise exc.InitializerException(
                f"InitializerParamStartPoints.from_result: n_points must be >= 1, got {n_points}."
            )

        if n_points == 1 and jitter == 0.0:
            return cls(parameter_dict=base_dict)

        # Multi-point and/or explicitly-jittered path: draw `n_points` starting vectors, each offset in
        # physical space, deterministically from `seed`. Priors are only ever read (`.id`, `path_for_prior`)
        # -- never mutated.
        sigma_by_source_path = {}
        covariance = None
        try:
            covariance = np.asarray(samples.covariance_matrix)
        except Exception:
            covariance = None

        if (
            covariance is not None
            and covariance.ndim == 2
            and covariance.shape[0] == covariance.shape[1] == len(source_priors)
        ):
            diagonal = np.diag(covariance)
            for index, prior in enumerate(source_priors):
                path = source_model.path_for_prior(prior)
                sigma_value = diagonal[index]
                sigma_by_source_path[path] = (
                    float(np.sqrt(sigma_value))
                    if np.isfinite(sigma_value) and sigma_value > 0
                    else None
                )

        # `jitter == 0.0` still uses a tiny fixed magnitude so `n_points > 1` points differ from one
        # another (matching the single-point class's own +-1e-8 default jitter).
        magnitude = jitter if jitter != 0.0 else 1.0e-8
        random_state = np.random.RandomState(seed)

        point_dicts: List[Dict[Prior, float]] = []
        for _ in range(n_points):
            point_dict = {}
            for prior in target_priors:
                path = target_model.path_for_prior(prior)
                value = base_dict[prior]
                sigma = sigma_by_source_path.get(path)
                if sigma is not None:
                    value = value + magnitude * sigma * random_state.normal()
                else:
                    value = value + magnitude * random_state.normal()
                point_dict[prior] = value
            point_dicts.append(point_dict)

        initializer = cls(parameter_dict=base_dict)
        initializer._point_dicts = point_dicts
        initializer._point_index = 0
        return initializer

    def _generate_unit_parameter_list(self, model: AbstractPriorModel) -> List[float]:
        """
        Generate a unit vector for the model.

        When this initializer was built via `from_result(..., n_points > 1)` (or a non-zero `jitter`),
        successive calls cycle through the `n_points` pre-computed starting vectors, one per call (i.e. one
        per requested starting point / chain), wrapping around if more points are requested than were
        generated. Otherwise this defers to the base `InitializerParamBounds` behaviour (a fixed +-1e-8
        unit-space jitter around the single stored `parameter_dict`).
        """
        if self._point_dicts is None:
            return super()._generate_unit_parameter_list(model)

        point_dict = self._point_dicts[self._point_index % len(self._point_dicts)]
        self._point_index += 1

        unit_parameter_list = []
        for prior in model.priors_ordered_by_id:
            try:
                value = point_dict[prior]
            except KeyError:
                key = ".".join(model.path_for_prior(prior))
                if key not in self._generated_warnings:
                    logger.warning(
                        f"Range for {key} not set in the InitializerParamStartPoints. "
                        f"Using defaults."
                    )
                    self._generated_warnings.add(key)
                value = prior.random(self.lower_limit, self.upper_limit)

            unit_parameter_list.append(prior.unit_value_for(value))

        return unit_parameter_list

    def samples_from_model(self, *args, **kwargs):
        """
        As `AbstractInitializer.samples_from_model`, but first resets the per-point cycling counter used by
        `_generate_unit_parameter_list` (when this initializer was built via `from_result(n_points > 1)`) so
        repeated fits/resumes start again from point 0.
        """
        if self._point_dicts is not None:
            self._point_index = 0

        return super().samples_from_model(*args, **kwargs)


class Initializer(AbstractInitializer):
    def __init__(self, lower_limit: float, upper_limit: float):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        `NonLinearSearch` to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    @classmethod
    def from_config(cls, config):
        """
        Load the Initializer from a non_linear config file.
        """

        try:
            initializer = config("initialize", "method")

        except configparser.NoSectionError:
            return None

        if initializer in "prior":
            return InitializerPrior()

        elif initializer in "ball":
            ball_lower_limit = config("initialize", "ball_lower_limit")
            ball_upper_limit = config("initialize", "ball_upper_limit")

            return InitializerBall(
                lower_limit=ball_lower_limit, upper_limit=ball_upper_limit
            )

    def _generate_unit_parameter_list(self, model):
        return model.random_unit_vector_within_limits(
            lower_limit=self.lower_limit, upper_limit=self.upper_limit
        )


class InitializerPrior(Initializer):
    def __init__(self):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        `NonLinearSearch` to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.

        The InitializerPrior class generates from the priors, by drawing all values as unit values between 0.0 and 1.0
        and mapping them to physical values via the prior.
        """
        super().__init__(lower_limit=0.0, upper_limit=1.0)


class InitializerBall(Initializer):
    def __init__(self, lower_limit: float, upper_limit: float):
        """
        The Initializer creates the initial set of samples in non-linear parameter space that can be passed into a
        `NonLinearSearch` to define where to begin sampling.

        Although most non-linear searches have in-built functionality to do this, some do not cope well with parameter
        resamples that are raised as FitException's. Thus, PyAutoFit uses its own initializer to bypass these problems.

        The InitializerBall class generates the samples in a small compact volume or 'ball' in parameter space, which is
        the recommended initialization strategy for the MCMC `NonLinearSearch` Emcee.

        Parameters
        ----------
        lower_limit
            The lower limit of the uniform distribution unit values are drawn from when initializing walkers in a small
            compact ball.
        upper_limit
            The upper limit of the uniform distribution unit values are drawn from when initializing walkers in a small
            compact ball.
        """
        super().__init__(lower_limit=lower_limit, upper_limit=upper_limit)
