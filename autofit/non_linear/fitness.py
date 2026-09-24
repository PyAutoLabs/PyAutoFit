import logging
import numpy as np
import os
import time

from timeout_decorator import timeout
from typing import Optional

from autonerves import conf
from autonerves import cached_property

from autofit import exc

from autofit.text import text_util


from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.jax_compile import log_on_first_compile
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.analysis import Analysis

#: Iteration cadences at or above this mean "never". The packaged config default
#: for both ``iterations_per_quick_update`` and ``iterations_per_full_update`` is
#: the inf-like ``1e99`` sentinel documented on ``_steps_until_full_update`` --
#: kept as a float so ``search.json`` stores a readable ``1e99`` rather than a
#: 99-digit integer. Compared against a threshold rather than ``1e99`` exactly so
#: a hand-set ``1e100`` in a workspace config reads as "never" too.
ITERATIONS_NEVER = 1e90


def get_timeout_seconds():

    try:
        return conf.instance["general"]["test"]["lh_timeout_seconds"]
    except KeyError:
        pass

logger = logging.getLogger(__name__)
timeout_seconds = get_timeout_seconds()


def _exception_override() -> bool:
    """
    Whether `general.test.exception_override` is set, which disables assertion checking.

    Read once per `Fitness` rather than per call: the traced assertion penalty branches on this
    with a Python `if`, which must resolve at trace time. A config that never mentions the key
    means no override.
    """
    try:
        return bool(conf.instance["general"]["test"]["exception_override"])
    except KeyError:
        return False

#: Ceiling used when ``general.test.log_likelihood_ceiling`` is absent from the config (e.g. a
#: workspace whose ``general.yaml`` pre-dates the key). ``inf`` -- i.e. the guard is **off** --
#: because the packaged default is off, and a config that never mentions the key must inherit
#: today's default rather than the one that shipped in PyAutoFit 2025.x.
LOG_LIKELIHOOD_CEILING_DEFAULT = float("inf")

#: Set the first time the magnitude guard actually rejects a log likelihood on the numpy path, so
#: the warning below is emitted once per process rather than once per sample. Module level, not
#: per-`Fitness`: a search rebuilds its fitness object on resume, and one warning per run is the
#: point.
_log_likelihood_ceiling_warning_emitted = False


def _warn_log_likelihood_ceiling_fired(log_likelihood, ceiling: float):
    """
    Warn, once per process, that the magnitude guard has rejected a log likelihood.

    The guard is deliberately silent about *which* of the two things it caught: numerical garbage
    (what it exists for), or a legitimate likelihood on a dataset whose noise-map units make
    ``|log_likelihood|`` genuinely enormous (what makes the ceiling unsafe as a default -- see
    `get_log_likelihood_ceiling`). Only the user knows their units, so the warning names the value
    and the ceiling and leaves the judgement to them.

    **Numpy path only.** `Fitness.call` also runs under ``jax.jit`` / ``jax.vmap``, where the
    rejection is an ``xp.where`` on a *tracer*: there is no Python-visible moment at which the guard
    "fires", so this cannot be called, and those paths reject silently by construction. Nor is a
    fire *counter* offered for them -- incrementing one from traced code needs a host callback or a
    donated buffer, which perturbs the very code being profiled and does not survive ``vmap`` /
    ``grad``. A counter that only ever counted the numpy path would read ``0`` on a jitted run that
    rejected every sample, which is worse than no counter at all.
    """
    global _log_likelihood_ceiling_warning_emitted

    if _log_likelihood_ceiling_warning_emitted:
        return

    _log_likelihood_ceiling_warning_emitted = True

    logger.warning(
        f"A log likelihood of {log_likelihood} exceeded the configured magnitude ceiling of "
        f"{ceiling} and was replaced by the resample figure of merit, so the search will not "
        "sample this model. This is usually numerical garbage (e.g. an fp64 Cholesky on a "
        "non-positive-definite matrix), but a log likelihood scales with the noise-map units, so a "
        "badly-scaled dataset can exceed the ceiling legitimately -- check the units before "
        "trusting the rejection. Set general.test.log_likelihood_ceiling higher, or blank to "
        "disable the guard. This warning is issued once per run; JAX-traced runs (jit / vmap) "
        "cannot issue it at all."
    )


def get_log_likelihood_ceiling() -> float:
    """
    The largest ``|log_likelihood|`` a fitness function accepts before treating the value as
    numerical garbage and substituting the resample figure of merit.

    **Off by default.** The packaged ``general.yaml`` ships this blank, so the guard is disabled
    unless a config opts in. It is off because the ceiling is a bare magnitude in *unspecified
    units*: a log likelihood is ``-0.5 * chi_squared - 0.5 * noise_normalization``, and both terms
    scale with the noise map -- ``chi_squared`` as ``noise ** -2``, ``noise_normalization``
    linearly in the number of pixels. A dataset whose noise map is expressed in a small unit can
    therefore produce a *legitimate* log likelihood above any fixed threshold, and every model
    would be silently rejected, leaving the search nothing but the resample sentinel. The guard
    cannot tell that case from the one below, because a magnitude is the only signal it reads.

    What it exists for, when a config does opt in: `Fitness.call` maps ``NaN`` and ``inf`` log
    likelihoods to ``resample_figure_of_merit``, but a *finite* value of, say, ``3e+303`` passes
    straight through and the search accepts it as its best point. That is not hypothetical: a
    non-positive-definite regularization matrix makes an fp64 Cholesky return finite garbage, a
    nested sampler treats it as the peak of the posterior, its shell log evidence explodes to
    ``~1e56`` and the termination criterion never fires -- the run burns its wall clock without ever
    converging. `autolens_profiling` enables the ceiling for exactly this reason (run ``341908_5``);
    a science analysis should enable it only once its own units are known to be safe.

    The value is read **once** from
    ``conf.instance["general"]["test"]["log_likelihood_ceiling"]`` and returned as a static Python
    float, because the guard it feeds runs inside JAX-traced code, where a Python ``if`` on a traced
    value is illegal and the comparison must be a static-vs-traced ``xp.where``.

    A bare ``1e20`` in YAML parses as a *string*, not a float (PyYAML requires a decimal point or a
    signed exponent), so the value is coerced with ``float`` rather than trusted as read.

    Returns
    -------
    The ceiling as a float. ``inf`` disables the guard, which is what an absent, ``null``,
    non-positive or unparseable config entry maps to, since ``abs(x) > inf`` is never true.
    """
    try:
        ceiling = conf.instance["general"]["test"]["log_likelihood_ceiling"]
    except KeyError:
        return LOG_LIKELIHOOD_CEILING_DEFAULT

    if ceiling is None:
        return float("inf")

    try:
        ceiling = float(ceiling)
    except (TypeError, ValueError):
        logger.warning(
            f"general.test.log_likelihood_ceiling = {ceiling!r} could not be read as a float. "
            "The log likelihood magnitude guard is disabled for this run."
        )
        return float("inf")

    if not ceiling > 0.0:
        logger.warning(
            f"general.test.log_likelihood_ceiling = {ceiling} is not positive, so every log "
            "likelihood would be rejected. The log likelihood magnitude guard is disabled for "
            "this run."
        )
        return float("inf")

    return ceiling


class Fitness:
    def __init__(
        self,
        model : AbstractPriorModel,
        analysis : Analysis,
        paths : Optional[AbstractPaths] = None,
        fom_is_log_likelihood: bool = True,
        resample_figure_of_merit: float = None,
        convert_to_chi_squared: bool = False,
        store_history: bool = False,
        use_jax_vmap : bool = False,
        use_jax_jit : bool = False,
        batch_size : Optional[int] = None,
        iterations_per_quick_update: Optional[int] = None,
        background_quick_update: bool = False,
        live_visual_update: bool = False,
    ):
        """
        Interfaces with any non-linear search to fit the model to the data and return a log likelihood via
        the analysis.

        The interface of a non-linear search and fitness function is summarised as follows:

        1) The non-linear search samples a new set of model parameters, which are passed to the fitness
        function's `__call__` method.

        2) The list of parameter values are mapped to an instance of the model.

        3) The instance is passed to the analysis class's log likelihood function, which fits the model to the
        data and returns the log likelihood.

        4) A final figure-of-merit is computed and returned to the non-linear search, which is either the log
        likelihood or log posterior (e.g. adding the log prior to the log likelihood).

        Certain searches (commonly nested samplers) require the parameters to be mapped from unit values to physical
        values, which is performed internally by the fitness object in step 2.

        Certain searches require the returned figure of merit to be a log posterior (often MCMC methods) whereas
        others require it to be a log likelihood (often nested samples which account for priors internally) in step 4.
        Which values is returned by the `fom_is_log_likelihood` bool.

        Some searches require a chi-squared value (which they minimized), given by the log likelihood multiplied
        by -2.0. This is returned by the fitness if the `convert_to_chi_squared` bool is `True`.

        If a model-fit raises an exception or returns a `np.nan`, a `resample_figure_of_merit` value is returned
        instead. The appropriate value depends on the search, but is typically either `None`, `-np.inf` or `1.0e99`.
        All values indicate to the non-linear search that the model-fit should be resampled or ignored.

        Many searches do not store the history of the parameters and log likelihood values, often to save
        memory on large model-fits. However, this can be useful, for example to plot the results of a model-fit
        versus iteration number. If the `store_history` bool is `True`, the parameters and log likelihoods are stored
        in the `parameters_history_list` and `figure_of_merit_history_list` attribute of the fitness object.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        paths
            The paths of the search, which if the search is being resumed from an old run is used to check that
            the likelihood function has not changed from the previous run.
        fom_is_log_likelihood
            If `True`, the figure of merit returned by the fitness function is the log likelihood. If `False`, the
            figure of merit is the log posterior.
        resample_figure_of_merit
            The figure of merit returned if the model-fit raises an exception or returns a `np.nan`.
        convert_to_chi_squared
            If `True`, the figure of merit returned is the log likelihood multiplied by -2.0, such that it is a
            chi-squared value that is minimized.
        store_history
            If `True`, the parameters and log likelihood values of every model-fit are stored in lists.
        """

        self.analysis = analysis
        self.model = model
        self.paths = paths
        self.fom_is_log_likelihood = fom_is_log_likelihood

        self.resample_figure_of_merit = resample_figure_of_merit or -self._xp.inf

        # Static Python float, read once here rather than per-call: `call` compares it against a
        # traced value under `jax.jit` / `jax.vmap`, which requires the ceiling to be a compile-time
        # constant.
        self.log_likelihood_ceiling = get_log_likelihood_ceiling()

        self._set_traced_assertions()

        self.convert_to_chi_squared = convert_to_chi_squared
        self.store_history = store_history

        self.parameters_history_list = []
        self.log_likelihood_history_list = []

        self.use_jax_vmap = use_jax_vmap
        self.use_jax_jit = use_jax_jit

        if getattr(self.analysis, "_use_jax", False):
            from autofit.jax.pytrees import enable_pytrees, register_model

            enable_pytrees()
            register_model(self.model)

        self._call = self.call

        if self.use_jax_vmap:
            self._call = self._vmap
        elif self.use_jax_jit:
            self._call = self._jit

        self.batch_size = batch_size
        self.iterations_per_quick_update = iterations_per_quick_update
        self.live_visual_update = live_visual_update
        self.quick_update_max_lh_parameters = None
        self.quick_update_max_lh = -self._xp.inf
        self.quick_update_count = 0

        self._background_quick_update = None
        self._live_display = None

        if background_quick_update and self.iterations_per_quick_update is not None:
            from autofit.non_linear.quick_update import BackgroundQuickUpdate

            convert_jax = (
                getattr(self.analysis, "_use_jax", False)
                and not getattr(self.analysis, "supports_jax_visualization", False)
            )

            self._background_quick_update = BackgroundQuickUpdate(
                convert_jax=convert_jax,
                live_visual_update=self.live_visual_update,
            )
        elif self.live_visual_update and self.iterations_per_quick_update is not None:
            # Synchronous quick-update path: BackgroundQuickUpdate is off
            # but the user still asked for live visuals. Manage display
            # surfaces via a standalone LiveDisplay; the rendering itself
            # still runs on the main thread inside `manage_quick_update`.
            from autofit.non_linear.quick_update import LiveDisplay

            self._live_display = LiveDisplay(live_visual_update=True)

        if self.paths is not None:
            self.check_log_likelihood(fitness=self)

        if (
            self.iterations_per_quick_update is not None
            and self._xp.__name__.startswith("jax")
        ):
            self._warmup_visualization()

    def _warmup_visualization(self):
        """Pre-compile the JAX operations used by ``fit_for_visualization``.

        The first call to ``fit_for_visualization`` triggers ~200 small
        per-function JAX JIT compilations (one per profile method per
        decorator). Running them here moves that cost to search setup
        so every quick update during sampling is fast.
        """
        logger.info(
            "Warming up visualization (one-time JAX compilation)..."
        )
        try:
            instance = self.model.instance_from_prior_medians()
            fit = self.analysis.fit_for_visualization(instance=instance)
            _ = fit.model_data
        except Exception:
            logger.warning(
                "Visualization warm-up failed (non-fatal); "
                "first quick update may be slow."
            )
        else:
            logger.info("Visualization warm-up complete.")

    @property
    def _xp(self):
        return self.analysis._xp

    @property
    def _is_jax(self) -> bool:
        """
        Whether the array backend is JAX, and therefore whether `call` may be running on tracers.

        Read by the log-likelihood magnitude guard, which warns from Python and so must stay on the
        numpy path — under `jax.jit` / `jax.vmap` a Python branch on a traced value is illegal.
        """
        return self._xp.__name__.startswith("jax")

    def call(self, parameters):
        """
        A private method that calls the fitness function with the given parameters and additional keyword arguments.
        This method is intended for internal use only.

        Model assertions (`AbstractPriorModel.add_assertion`) are enforced here too, and how
        depends on the backend. On numpy `instance_from_vector` raises a `FitException` and the
        `except` below returns `resample_figure_of_merit`. Under JAX a `raise` cannot happen
        inside a trace, so the instance is built with `ignore_assertions=True` and the assertions
        -- gathered from the whole model tree once, in `__init__`, so a child-attached assertion
        is enforced exactly as it is on numpy -- are instead evaluated as a **traced boolean** by
        `assertions_satisfied_from_vector` and applied with an `xp.where` to the **final figure of
        merit**, mapping a violating model to `resample_figure_of_merit`. Applying it at the end
        is what makes the two backends agree exactly: numpy returns its sentinel from an early
        `return`, before the log prior is added and before the chi-squared multiply. That
        makes the JAX path exception-free as required, at the cost of the same **value-only**
        caveat the NaN guards carry below: under `jax.grad` the `where` still differentiates the
        rejected branch. Whether the penalty is applied at all is a static Python bool decided in
        `__init__`, so the branch resolves at trace time under `jax.jit` / `jax.vmap`.

        The NaN/inf/magnitude guards below protect the **value only, never the gradient**.

        A model whose likelihood is NaN, inf, or larger in magnitude than `self.log_likelihood_ceiling` is mapped
        to `resample_figure_of_merit`, so searches that read only the figure of merit (nested samplers, MCMC) get
        the resample sentinel and never select the point. The magnitude ceiling exists because the NaN/inf checks
        miss the failure mode that actually kills long runs: an fp64 Cholesky on a non-positive-definite matrix
        returns *finite* garbage (log likelihoods up to `3e+303`), which a nested sampler happily accepts as its
        best point and then never terminates on. It is **off unless a config opts in** — the threshold is a bare
        magnitude and a log likelihood scales with the noise-map units, so a fixed ceiling can reject a legitimate
        fit (`get_log_likelihood_ceiling` carries the argument). It is a static Python float, so the default `inf`
        makes the `where` a no-op and the comparison stays legal under `jax.jit` / `jax.vmap`. When it does fire,
        the numpy path warns once per process; the traced paths cannot warn at all. Gradient
        consumers get no such protection: under `jax.grad`, reverse-mode differentiates *both* branches of an
        `xp.where` and multiplies the unselected one by zero, so if the likelihood's derivative is also non-finite
        the guard yields `0 * NaN = NaN` and the returned gradient is NaN even though the value looks handled.

        This bites only when the masked branch's *derivative* is non-finite — not merely its value. `sqrt(x)` at
        x < 0 and `cholesky(A)` for non-positive-definite `A` are NaN in both value and derivative, so they trigger
        it; `log(x)` at x < 0 is NaN in value but its derivative `1/x` stays finite, so it does not.

        A guard here **cannot** repair this. By the time this method receives `log_likelihood` the non-finite
        derivative is already recorded on the autodiff tape, and no transformation of the output can remove it —
        an output-side "double-where" does not work. Gradient-safety must be established at the site that creates
        the NaN, by never *evaluating* the offending operation at the invalid input.

        See autolens_workspace_developer#104, where this was diagnosed, and
        `autofit_workspace_test/scripts/jax_assertions/fitness_nan_gradient_contract.py`, which pins the behaviour
        described here.

        Parameters
        ----------
        parameters
            The parameters (typically a list) chosen by a non-linear search, which are mapped to an instance of the
            model via its priors and fitted to the data.
        kwargs
            Additional key-word arguments that may be necessary for specific non-linear searches.

        Returns
        -------
        The figure of merit returned to the non-linear search, which is either the log likelihood or log posterior.
        """
        # `None` on the numpy path, and on the JAX path when the model has no assertions: a Python
        # `is None` on it below is therefore a static branch, legal under `jax.jit` / `jax.vmap`.
        assertions_satisfied = None

        if self._is_jax:

            # Get instance from model. Assertions are skipped here because they signal failure by
            # raising, which is illegal inside a trace; they are applied below as a traced value.
            instance = self.model.instance_from_vector(
                vector=parameters, ignore_assertions=True, xp=self._xp
            )

            # Evaluate log likelihood (must be side-effect free and exception-free)
            log_likelihood = self.analysis.log_likelihood_function(instance=instance)

            if self._apply_assertions_traced:
                assertions_satisfied = self.model.assertions_satisfied_from_vector(
                    parameters,
                    xp=self._xp,
                    assertions=self._traced_assertions,
                )

        else:

            try:
                instance = self.model.instance_from_vector(vector=parameters, xp=self._xp)
                log_likelihood = self.analysis.log_likelihood_function(instance=instance)
            except exc.FitException:
                return self.resample_figure_of_merit

        # Penalize NaNs in the log-likelihood. Value-only: under jax.grad these `where`s still differentiate the
        # masked branch, so a non-finite derivative propagates as `0 * NaN = NaN`. See the contract in the
        # docstring above -- gradient-safety belongs at the site that creates the NaN, not here.
        log_likelihood = self._xp.where(self._xp.isnan(log_likelihood), self.resample_figure_of_merit, log_likelihood)
        log_likelihood = self._xp.where(self._xp.isinf(log_likelihood), self.resample_figure_of_merit, log_likelihood)

        # Penalize finite-but-impossible log-likelihoods (e.g. 3e+303 out of an fp64 Cholesky on a
        # non-positive-definite matrix), which the isnan/isinf guards above let through. Same
        # value-only caveat. The ceiling is a static float, and it is `inf` unless a config opts in
        # (see `get_log_likelihood_ceiling`), so by default this is a no-op; when it is set, the
        # resample sentinels (-inf, -1e99, -1e30) map to themselves.
        raw_log_likelihood = log_likelihood

        over_ceiling = self._xp.abs(log_likelihood) > self.log_likelihood_ceiling

        log_likelihood = self._xp.where(
            over_ceiling,
            self.resample_figure_of_merit,
            log_likelihood,
        )

        # Tell the user the first time the guard actually rejects something, because a rejection is
        # indistinguishable from "the model is bad" from inside the search. Numpy path only:
        # `over_ceiling` is a tracer under jit / vmap, so a Python `if` on it is illegal there and
        # those runs reject silently (documented in `_warn_log_likelihood_ceiling_fired`). The
        # ceiling test comes first so the default (disabled) config short-circuits before touching
        # the array at all, and `_warning_emitted` takes this off the hot path after the first fire.
        if (
            self.log_likelihood_ceiling != np.inf
            and not _log_likelihood_ceiling_warning_emitted
            and not self._is_jax
        ):
            if bool(np.any(over_ceiling)):
                _warn_log_likelihood_ceiling_fired(
                    log_likelihood=raw_log_likelihood,
                    ceiling=self.log_likelihood_ceiling,
                )

        # Determine final figure of merit
        if self.fom_is_log_likelihood:
            figure_of_merit = log_likelihood
        else:
            # Ensure prior list is compatible with JAX (must return a JAX array, not list)
            log_prior_array = self._xp.array(self.model.log_prior_list_from_vector(vector=parameters, xp=self._xp))
            figure_of_merit = log_likelihood + self._xp.sum(log_prior_array)

        # Convert to chi-squared scale if requested
        if self.convert_to_chi_squared:
            figure_of_merit *= -2.0

        # Reject models that violate an assertion, *after* the conversions above. The numpy path
        # returns `resample_figure_of_merit` from an early `return`, so neither the log prior nor
        # the chi-squared multiply ever touches its sentinel; applying this to the log likelihood
        # instead would let both rewrite it here, and `convert_to_chi_squared` flips its sign --
        # turning the most-rejected point in the space into the most attractive one for a
        # minimizer. Same value-only caveat as the guards above.
        if assertions_satisfied is not None:
            figure_of_merit = self._xp.where(
                assertions_satisfied,
                figure_of_merit,
                self.resample_figure_of_merit,
            )

        return figure_of_merit

    def log_likelihood_from(self, figure_of_merit, parameters):
        """
        Invert the figure-of-merit convention to recover the log likelihood.

        `call` maps a log likelihood to the figure of merit (FoM) the search consumes: it adds the summed log prior
        when `fom_is_log_likelihood` is `False` (giving a log posterior) and multiplies by `-2.0` when
        `convert_to_chi_squared` is `True` (giving a chi-squared). This method applies the exact inverse, so any
        code holding a FoM can get back the log likelihood on the scale that `Samples` persist.

        Both callers need that inverse. `call_wrap` uses it for the quick-update / history bookkeeping, which is
        defined on log likelihoods. `check_log_likelihood` uses it to compare a freshly computed value against the
        log likelihood stored in a previous run's samples summary; without it, that check compares a stored log
        likelihood against a value in the search's own FoM convention and every resume of a non-log-likelihood
        search (e.g. `MultiStartAdam`, `LBFGS`, `Emcee`) fails its sanity check on an unchanged likelihood
        function.

        Parameters
        ----------
        figure_of_merit
            The figure of merit returned by `call`, in this fitness's own convention.
        parameters
            The parameter vector the figure of merit was computed for, needed to evaluate the log priors that
            `fom_is_log_likelihood=False` folded in.

        Returns
        -------
        The log likelihood, on the same scale as `Sample.log_likelihood`.
        """
        if self.convert_to_chi_squared:
            log_likelihood = -0.5 * figure_of_merit
        else:
            log_likelihood = figure_of_merit

        if not self.fom_is_log_likelihood:
            log_prior_list = np.array(self.model.log_prior_list_from_vector(vector=parameters, xp=np))
            log_prior_sum = np.sum(log_prior_list)
            # A non-finite prior sum marks a rejected out-of-support point (the strict
            # priors return -inf outside their bounds, PyAutoFit#1489). The figure of
            # merit there is -inf too, and subtracting -inf from -inf is NaN — keep the
            # -inf so history / quick-update bookkeeping stays comparable.
            if np.isfinite(log_prior_sum):
                log_likelihood = log_likelihood - log_prior_sum

        return log_likelihood

    def call_wrap(self, parameters):
        """
        Wrapper around a JAX-jitted likelihood function that optionally stores
        the history of evaluated parameters and likelihood values.

        Depending on whether the figure of merit
        (FoM) is defined as a log-likelihood (`self.fom_is_log_likelihood`), it
        either uses the FoM directly or subtracts the summed log-prior to obtain
        the log-likelihood.

        If `self.store_history` is True, both the input parameters and the
        corresponding log-likelihood are appended to internal history lists
        (`self.parameters_history_list`, `self.log_likelihood_history_list`).

        Parameters
        ----------
        parameters
            A vector of model parameters to evaluate.

        Returns
        -------
        float
            The computed figure of merit for the input parameters. This is either
            the log-likelihood itself or another objective function value,
            depending on configuration.
        """

        if self.use_jax_vmap:
            if len(np.array(parameters).shape) == 1:
                parameters = np.array(parameters)[None, :]

        figure_of_merit = self._call(parameters)

        if self.use_jax_jit:
            figure_of_merit = float(figure_of_merit)

        log_likelihood = self.log_likelihood_from(
            figure_of_merit=figure_of_merit, parameters=parameters
        )

        self.manage_quick_update(parameters=parameters, log_likelihood=log_likelihood)

        if self.store_history:

            self.parameters_history_list.append(np.array(parameters))
            self.log_likelihood_history_list.append(np.array(log_likelihood))

        return figure_of_merit

    def manage_quick_update(self, parameters, log_likelihood):
        """
        Manage quick updates during the non-linear search.

        A "quick update" is a lightweight visualization of the current best-fit
        (maximum likelihood) model parameters. This provides fast feedback on the
        progress of the fit without waiting for the full analysis to complete.

        It does not require leaving the active non-linear search, and is
        therefore faster than the full analysis visualization.

        Workflow:
        ----------
        1. Track the number of likelihood evaluations since the last quick update.
        2. Identify the maximum log-likelihood from the current batch of evaluations.
           - If `log_likelihood` is an array (batched evaluations), find the best
             index with `argmax`.
           - If it’s just a scalar (single evaluation), treat it as one update.
        3. If a new maximum likelihood is found, update:
           - `self.quick_update_max_lh` (best log-likelihood value so far).
           - `self.quick_update_max_lh_parameters` (corresponding parameter vector).
        4. Once the number of evaluations exceeds
           `self.iterations_per_quick_update`, generate a quick visualization of
           the current max-likelihood model via
           `self.analysis.perform_quick_update()`.

        Parameters
        ----------
        parameters : array-like
            The parameter vectors evaluated in this batch. Shape is typically
            (n_batch, n_param).
        log_likelihood : float or array-like
            The corresponding log-likelihood(s). If batched, must have shape
            (n_batch,).

        Notes
        -----
        - Quick updates are optional and controlled by
          `self.iterations_per_quick_update`.
        - If the `analysis` class does not implement
          `perform_quick_update`, the update is silently skipped.
        - If the current maximum log likelihood parameters cannot be turned
          into a model instance (e.g. they are outside a model component's
          physical domain), the visual is skipped with a logged warning and
          the search continues -- a quick update never terminates a fit. The
          `model.results` text, which is formatted from the parameter vector
          alone, is still written.
        - This mechanism is intended for fast, coarse visualization only,
          not detailed science-quality outputs.
        """

        # Quick updates disabled (the packaged 1e99 default): return before any
        # per-call bookkeeping. The running max / count only ever feed the quick
        # update body, which can never fire at this cadence, and this method is
        # called once per likelihood evaluation.
        if (
            self.iterations_per_quick_update is None
            or self.iterations_per_quick_update >= ITERATIONS_NEVER
        ):
            return

        try:

            best_idx = self._xp.argmax(log_likelihood)
            best_log_likelihood = log_likelihood[best_idx]
            best_parameters = parameters[best_idx]
            total_updates = log_likelihood.shape[0]

        except (AttributeError, IndexError, TypeError):

            best_log_likelihood = log_likelihood
            best_parameters = parameters
            total_updates = 1

        if best_log_likelihood > self.quick_update_max_lh:
            self.quick_update_max_lh = best_log_likelihood
            self.quick_update_max_lh_parameters = best_parameters

        self.quick_update_count += total_updates

        if self.quick_update_count >= self.iterations_per_quick_update:

            from IPython.display import clear_output

            # Clearing a notebook cell removes the display_id target used by
            # LiveDisplay.update_display, leaving later image updates invisible.
            # Keep that target alive for both synchronous and background renders.
            if not self.live_visual_update:
                clear_output(wait=True)

            start_time = time.time()

            logger.info("Performing quick update of maximum log likelihood fit image and model.results")

            # A quick update is a convenience render, so nothing inside it may end
            # the search. The max-likelihood vector a sampler is carrying is not
            # guaranteed to map to a physical instance: PyAutoGalaxy raises
            # `ModelParameterException` (a `ValueError` / `af.exc.FitException`)
            # straight out of a profile constructor, and an out-of-disk `ell_comps`
            # unwound through this call and killed a 36 h Nautilus run at its very
            # first quick update (PyAutoFit#1567). `Exception` is caught rather than
            # `FitException` alone so the synchronous path is as protected as the
            # background worker (`BackgroundQuickUpdate._process_pending`).
            try:
                instance = self.model.instance_from_vector(
                    vector=self.quick_update_max_lh_parameters, xp=self._xp
                )
            except Exception:
                logger.exception(
                    "Quick update skipped: the current maximum log likelihood "
                    "parameters do not map to a valid model instance. The search "
                    "continues; the model.results text is still written."
                )
                instance = None

            if instance is not None:
                if self._background_quick_update is not None:
                    self._background_quick_update.submit(
                        self.analysis, self.paths, instance,
                    )
                else:
                    try:
                        self.analysis.perform_quick_update(self.paths, instance)
                    except NotImplementedError:
                        pass
                    except Exception:
                        logger.exception(
                            "Quick update visual raised an exception (ignored)."
                        )
                    else:
                        if self._live_display is not None:
                            try:
                                self._live_display.update(self.paths)
                            except Exception:
                                logger.exception(
                                    "Live display update raised an exception (ignored)."
                                )

            # Searches hand their parameters over in whatever type they hold them:
            # ndarray (Nautilus), JAX array, or a plain Python list (Dynesty's
            # initializer). `np.asarray` normalizes all three -- calling `.tolist()`
            # directly assumed the array case, which held only while Nautilus was
            # the sole search wired up to quick updates (PyAutoFit#1434).
            result_info = text_util.result_max_lh_info_from(
                max_log_likelihood_sample=np.asarray(
                    self.quick_update_max_lh_parameters
                ).tolist(),
                max_log_likelihood=self.quick_update_max_lh,
                model=self.model,
            )
            result_info = "\n".join(result_info)

            logger.info(result_info)
            self.paths.output_model_results(result_info=result_info)

            self.quick_update_count = 0

            logger.info(f"Quick update complete in {time.time() - start_time} seconds.")

    def shutdown_quick_update(self):
        """Shut down the background quick-update worker and any live
        display surfaces (matplotlib viewer subprocess) that were spawned
        for this fitness instance."""
        if self._background_quick_update is not None:
            self._background_quick_update.shutdown()
            self._background_quick_update = None
        if self._live_display is not None:
            self._live_display.shutdown()
            self._live_display = None

    @timeout(timeout_seconds)
    def __call__(self, parameters, *kwargs):
        """
        Interfaces with any non-linear in order to fit a model to the data and return a log likelihood via
        an `Analysis` class.

        The interface is described in full in the `__init__` docstring above.

        Parameters
        ----------
        parameters
            The parameters (typically a list) chosen by a non-linear search, which are mapped to an instance of the
            model via its priors and fitted to the data.
        kwargs
            Addition key-word arguments that may be necessary for specific non-linear searches.

        Returns
        -------
        The figure of merit returned to the non-linear search, which is either the log likelihood or log posterior.
        """
        return self.call_wrap(parameters)

    def _set_traced_assertions(self):
        """
        Gather the model's assertions and decide, once, whether the traced penalty applies.

        Both are static by construction. The walk over the model tree is a Python loop over Python
        objects, so it cannot happen inside a trace, and there is no reason to repeat it per
        likelihood evaluation; `_apply_assertions_traced` is read by `call` as a Python `if`, which
        must resolve at trace time under `jax.jit` / `jax.vmap`.

        It is `False` when the model tree carries no assertions (the `where` would be a no-op) or
        when `general.test.exception_override` is set, which is what disables assertions on the
        numpy path too.

        Called from `__init__` **and** from `__setstate__`: a `Fitness` pickled before the traced
        penalty existed carries neither attribute, and defaulting them to "no assertions" would
        leave a resumed JAX search quietly sampling models the user forbade. The assertions live
        on the restored model either way, so they are recomputed rather than defaulted.
        """
        # `hasattr` because some tests pass a stand-in for the model rather than a real one.
        self._traced_assertions = (
            self.model.gathered_assertions()
            if hasattr(self.model, "gathered_assertions")
            else []
        )
        self._apply_assertions_traced = (
            bool(self._traced_assertions) and not _exception_override()
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Strip JAX-compiled callables: jax.jit / jax.vmap / jax.grad return
        # functions tied to C++ XLA state that can't roundtrip through pickle.
        # cached_property values lazily recompile on first access after unpickle.
        for attr in ("_call", "_jit", "_vmap", "_grad"):
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Fitness objects pickled before the magnitude guard existed carry no ceiling; give them the
        # configured one rather than letting `call` raise `AttributeError` on resume.
        self.__dict__.setdefault("log_likelihood_ceiling", get_log_likelihood_ceiling())
        # `Fitness` objects pickled before the traced assertion penalty existed carry neither
        # assertion attribute. Recompute them from the restored model rather than defaulting to
        # "no assertions", which would silently stop enforcing them on resume.
        if (
            "_traced_assertions" not in self.__dict__
            or "_apply_assertions_traced" not in self.__dict__
        ):
            self._set_traced_assertions()
        self._call = self.call
        if getattr(self, "use_jax_vmap", False):
            self._call = self._vmap
        elif getattr(self, "use_jax_jit", False):
            self._call = self._jit

    @cached_property
    def _vmap(self):
        """
        Vectorized and JIT-compiled likelihood function.

        This wraps the base likelihood function (`self.call`) so that it evaluates a whole batch of
        parameter vectors at once, as `jax.jit(jax.vmap(self.call))`. The order is jit **of** vmap:
        the conventional composition, and the one `analysis/latent.py` already uses.

        That order means the batch is one XLA program. The outer jit traces `vmap(call)` once, keyed
        on the shape of the batch it was given, and caches the compiled executable; a second call at
        the same batch shape is a cache hit served from jit's C++ fast path, with no Python batching
        trace in between. The previous composition, `jax.vmap(jax.jit(self.call))`, re-entered
        Python's batching machinery on every call and dispatched the inner `pjit` eagerly underneath
        the batching trace, which is where the `autogalaxy_workspace_test#118` hangs are parked.

        The inner jit is dropped rather than kept: a jit nested inside a jit is inlined, so it adds a
        call boundary and nothing else.

        This is **not** a fix for the XLA CPU Eigen-pool FFT deadlock (`PyAutoFit#1530`). That fires
        at execution time, and execution still happens once per call whatever the tracing order is.
        The only measured effect of the ordering is the A/B run of 2026-08-23 on `rectangular_mge.py`
        (control 8/10 stalls against 3/10 for jit-of-vmap, Fisher exact p=0.070), recorded there as
        contributory rather than causal.

        The compiled executable is still specialised per batch *length*, so a search that varies its
        batch size recompiles per distinct length (see the PyAutoMind draft
        `vmap_jit_recompiles_per_nautilus_batch_length.md`). This change neither fixes nor worsens
        that.

        Because this is a `cached_property`, the compiled function is stored after its first
        creation, avoiding repeated JIT compilation overhead.
        """
        import jax

        return log_on_first_compile(
            jax.jit(jax.vmap(self.call)),
            "vectorized (vmap) likelihood function",
        )

    @cached_property
    def _jit(self):
        """
        JIT-compiled likelihood function.

        This wraps the base likelihood function (`self.call`) with `jax.jit`,
        producing a compiled version optimized for repeated evaluation on a
        single set of parameters. The first call triggers compilation, while
        later calls benefit from the compiled execution.

        As a `cached_property`, the compiled function is cached after its
        first use, so JIT compilation only occurs once.
        """
        import jax

        return log_on_first_compile(
            jax.jit(self.call),
            "likelihood function",
        )

    @cached_property
    def _grad(self):
        """
        Gradient of the JIT-compiled likelihood function.

        This wraps the JIT-compiled likelihood function (`self._call`) with
        `jax.grad`, returning a function that computes gradients of the
        likelihood with respect to its input parameters. Useful for gradient-
        based optimization and inference methods.

        Since this is a `cached_property`, the gradient function is compiled
        and cached on first access, ensuring that expensive setup is done
        only once.
        """
        import jax

        return log_on_first_compile(
            jax.grad(self.call),
            "likelihood function gradient",
        )

    def grad(self, *args, **kwargs):
        return self._grad(*args, **kwargs)

    def check_log_likelihood(self, fitness):
        """
        Changes to the PyAutoGalaxy source code may inadvertantly change the numerics of how a log likelihood is
        computed. Equally, one may set off a model-fit that resumes from previous results, but change the settings of
        the pixelization or inversion in a way that changes the log likelihood function.

        This function performs an optional sanity check, which raises an exception if the log likelihood calculation
        changes, to ensure a model-fit is not resumed with a different likelihood calculation to the previous run.

        If the model-fit has not been performed before (e.g. it is not a resume) this function outputs
        the `figure_of_merit` (e.g. the log likelihood) of the maximum log likelihood model at the end of the model-fit.

        If the model-fit is a resume, it loads this `figure_of_merit` and compares it against a new value computed for
        the resumed run (again using the maximum log likelihood model inferred). If the two likelihoods do not agree
        and therefore the log likelihood function has changed, an exception is raised and the code execution terminated.

        Parameters
        ----------
        paths
            certain searches the non-linear search outputs are stored,
            visualization, and pickled objects used by the database and aggregator.
        result
            The result containing the maximum log likelihood fit of the model.
        """
        import numpy as np

        from autofit.non_linear.test_mode import skip_fit_output
        if skip_fit_output():
            return

        if not conf.instance["general"]["test"]["check_likelihood_function"]:
            return

        try:
            samples_summary = self.paths.load_samples_summary()
        except FileNotFoundError:
            return
        except ValueError as e:
            # A CORRUPT previous summary means the same thing as an ABSENT one
            # for this check: there is no trustworthy old likelihood to compare
            # against. Returning early is what the FileNotFoundError branch
            # above already does for the no-previous-run case.
            #
            # `ValueError` is the catch because `json.JSONDecodeError`
            # subclasses it -- which is exactly why this was missed by both the
            # `FileNotFoundError` above and the `(FileNotFoundError, TypeError,
            # KeyError)` guard on the multi-start resume path. A half-written
            # file therefore aborted the whole run from inside an OPTIONAL
            # sanity check -- and it stayed aborted on every subsequent run of
            # the same search name, since nothing rewrites the file until a run
            # gets far enough to finish.
            #
            # Warned rather than passed over in silence: an unreadable file is
            # a real event, unlike a missing one, and the user is the only one
            # who can decide whether the old results mattered.
            logger.warning(
                f"Could not read the previous samples summary while resuming "
                f"({type(e).__name__}: {e}). It is missing or corrupt, most "
                f"likely because an earlier run of this search was interrupted "
                f"while writing its output. The likelihood-function sanity "
                f"check is being SKIPPED for this run, and results are being "
                f"recomputed. Delete the search's output directory if you want "
                f"a guaranteed-clean start."
            )
            return

        try:
            max_log_likelihood_sample = samples_summary.max_log_likelihood_sample
        except AttributeError:
            return
        log_likelihood_old = samples_summary.max_log_likelihood_sample.log_likelihood

        parameters = max_log_likelihood_sample.parameter_lists_for_model(model=self.model)

        # `fitness(...)` returns the figure of merit in this search's own convention, which is only the log
        # likelihood when `fom_is_log_likelihood=True` and `convert_to_chi_squared=False`. The stored value is
        # always a log likelihood (`Sample.log_likelihood`), so the fresh value is converted back onto that scale
        # before comparison -- otherwise every resume of a log-posterior / chi-squared search fails this check on
        # an unchanged likelihood function.
        log_likelihood_new = self.log_likelihood_from(
            figure_of_merit=fitness(parameters=parameters), parameters=parameters
        )

        if not np.isclose(log_likelihood_old, log_likelihood_new):
            raise exc.SearchException(
                f"""
                Log likelihood sanity check failed.

                This means that the existing results of a model fit used a different
                likelihood function compared to the one implemented now.

                Both values below are log likelihoods, converted out of this search's
                figure-of-merit convention, so they are directly comparable.

                Old Log Likelihood = {log_likelihood_old}
                New Log Likelihood = {log_likelihood_new}
                """
            )
