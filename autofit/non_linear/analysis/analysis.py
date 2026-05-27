import inspect
import logging
from abc import ABC
import functools
import numpy as np
import time
from typing import Optional, Dict

from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.non_linear.result import Result
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.samples.sample import Sample

from .visualize import Visualizer
from ..samples.util import simple_model_for_kwargs

logger = logging.getLogger(__name__)


class Analysis(ABC):
    """
    Protocol for an analysis. Defines methods that can or
    must be implemented to define a class that compute the
    likelihood that some instance fits some data.
    """

    Result = Result
    Visualizer = Visualizer

    LATENT_KEYS = []

    # Strategy used by `compute_latent_samples` when `use_jax=True`.
    #
    # - "vmap" (default): wrap `compute_latent_variables` in
    #   `jax.jit(jax.vmap(...))` so all posterior samples are evaluated in a
    #   single batched call. Fastest when the function is fully vmap-safe
    #   (no Python control flow that depends on traced shapes, no calls into
    #   external libraries that don't support `jax.vmap`).
    #
    # - "jit": wrap in plain `jax.jit(...)` and loop in Python over samples.
    #   The JIT compile cache is reused across samples, so this is much
    #   faster than per-sample NumPy but slower than vmap. Use this when the
    #   inner function calls JAX code that documents vmap incompatibility
    #   (e.g. `jax_zero_contour.ZeroSolver` which uses `lax.cond` /
    #   `lax.while_loop` early-termination not safe under vmap).
    #
    # Subclasses override this attribute when their `compute_latent_variables`
    # implementation depends on vmap-incompatible primitives. For example,
    # `autogalaxy.AnalysisDataset` sets `LATENT_BATCH_MODE = "jit"` because
    # the lensing latents (Einstein radius via zero-contour) route through
    # `ZeroSolver`.
    LATENT_BATCH_MODE = "vmap"

    def __init__(
        self,
        use_jax: bool = False,
        **kwargs,
    ):
        import os
        if os.environ.get("PYAUTO_DISABLE_JAX") == "1":
            use_jax = False

        if use_jax:
            import importlib.util
            import warnings
            if importlib.util.find_spec("jax") is None:
                warnings.warn(
                    "\n"
                    "+----------------------------------------------------------------------+\n"
                    "|  use_jax=True was requested but JAX is not installed.                |\n"
                    "|                                                                      |\n"
                    "|  Falling back to numpy. The fit will run, but JAX acceleration       |\n"
                    "|  (typically 10-100x for large lens models) is unavailable.           |\n"
                    "|                                                                      |\n"
                    "|  To enable JAX, install on Python 3.11+ via your library's [jax]     |\n"
                    "|  extra, e.g.:  pip install autolens[jax]                             |\n"
                    "+----------------------------------------------------------------------+",
                    UserWarning,
                    stacklevel=2,
                )
                use_jax = False

        self._use_jax = use_jax
        self.kwargs = kwargs

    def fit_for_visualization(self, instance):
        """
        Build the fit used by the visualizer.

        Delegates to ``self.fit_from(instance)``. When ``use_jax=True``,
        the profile evaluations inside ``fit_from`` dispatch to JAX via
        the decorator chain. The per-function JIT caches warm up on the
        first call and are reused on all subsequent quick updates.
        """
        return self.fit_from(instance=instance)

    def __getattr__(self, item: str):
        """
        If a method starts with 'visualize_' then we assume it is associated with
        the Visualizer and forward the call to the visualizer.

        It may be desirable to remove this behaviour as the visualizer component of
        the system becomes more sophisticated.
        """
        if item.startswith("visualize") or item.startswith("should_visualize"):
            _method = getattr(self.Visualizer, item)
        else:
            raise AttributeError(f"Analysis has no attribute {item}")

        def method(*args, **kwargs):
            parameters = inspect.signature(_method).parameters
            if "analyses" in parameters:
                logger.debug(f"Skipping {item} as this is not a combined analysis")
                return
            return _method(self, *args, **kwargs)

        return method

    @property
    def _xp(self):
        if self._use_jax:
            import jax.numpy as jnp
            return jnp
        return np

    def compute_latent_samples(self, samples: Samples, batch_size : Optional[int] = None) -> Optional[Samples]:
        """
        Compute latent variables from a model instance.

        A latent variable is not itself a free parameter of the model but can be derived from it.
        Latent variables may provide physically meaningful quantities that help interpret a model
        fit, and their values (with errors) are stored in `latent.csv` in parallel with `samples.csv`.

        This implementation is designed to be compatible with both NumPy and JAX:

        - It is written to be side-effect free, so it can be JIT-compiled with `jax.jit`.
        - It can be vectorized over many parameter sets at once using `jax.vmap`, enabling efficient
          batched evaluation of latent variables for multiple samples.
        - Returned values should be simple JAX/NumPy scalars or arrays (no Python objects), so they
          can be stacked into arrays of shape `(n_samples, n_latents)` for batching.
        - Any NaNs introduced (e.g. from invalid model states) can be masked or replaced downstream.

        Parameters
        ----------
        parameters : array-like
            The parameter vector of the model sample. This will typically come from the non-linear search.
            Inside this method it is mapped back to a model instance via `model.instance_from_vector`.
        model : Model
            The model object defining how the parameter vector is mapped to an instance. Passed explicitly
            so that this function can be used inside JAX transforms (`vmap`, `jit`) with `functools.partial`.

        Returns
        -------
        tuple of (float or jax.numpy scalar)
            A tuple containing the latent variables in a fixed order:
            `(intensity_total, magnitude, angle)`. Each entry may be NaN if the corresponding component
            of the model is not present.
        """
        batch_size = batch_size or 10

        try:

            start_latent = time.time()

            compute_latent_for_model = functools.partial(self.compute_latent_variables, model=samples.model)

            if self._use_jax:
                import jax
                import jax.numpy as jnp
                start = time.time()
                if self.LATENT_BATCH_MODE == "vmap":
                    logger.info("JAX: Applying vmap and jit to likelihood function for latent variables -- may take a few seconds.")
                    batched_compute_latent = jax.jit(jax.vmap(compute_latent_for_model))
                elif self.LATENT_BATCH_MODE == "jit":
                    logger.info("JAX: Applying per-sample jit to latent variables (LATENT_BATCH_MODE='jit') -- may take a few seconds on first sample.")
                    jitted_compute_latent = jax.jit(compute_latent_for_model)

                    def batched_compute_latent(parameters_batch):
                        # Per-sample jit returns one (l1, l2, ..., lN) tuple per
                        # sample. Transpose to a tuple of N batched arrays so the
                        # downstream `jnp.stack(latent_values_batch, axis=-1)`
                        # works identically to the vmap path.
                        sample_results = [
                            jitted_compute_latent(p) for p in parameters_batch
                        ]
                        n_latents = len(sample_results[0])
                        return tuple(
                            jnp.stack([s[i] for s in sample_results])
                            for i in range(n_latents)
                        )
                else:
                    raise ValueError(
                        f"Unknown LATENT_BATCH_MODE={self.LATENT_BATCH_MODE!r}; expected 'vmap' or 'jit'."
                    )
                logger.info(f"JAX: {self.LATENT_BATCH_MODE} dispatch applied in {time.time() - start} seconds.")
            else:
                n_latents = len(self.LATENT_KEYS)
                nan_row = np.full(n_latents, np.nan)

                def _safe_compute(xx):
                    try:
                        return compute_latent_for_model(xx)
                    except exc.FitException:
                        return nan_row

                def batched_compute_latent(x):
                    return np.array([_safe_compute(xx) for xx in x])

            parameter_array = np.array(samples.parameter_lists)
            latent_samples = []

            # process in batches
            for i in range(0, len(parameter_array), batch_size):

                batch = parameter_array[i:i + batch_size]
                batch_samples = samples.sample_list[i:i + batch_size]

                # batched JAX call on this chunk
                latent_values_batch = batched_compute_latent(batch)

                if self._use_jax:
                    import jax.numpy as jnp
                    latent_values_batch = jnp.stack(latent_values_batch, axis=-1)  # (batch, n_latents)
                    mask = jnp.all(jnp.isfinite(latent_values_batch), axis=0)
                    latent_values_batch = latent_values_batch[:, mask]
                else:
                    # Drop samples whose latent computation failed (e.g. FitException from
                    # model assertions surfaced as a NaN row in _safe_compute). This leaves
                    # the per-latent column mask to continue handling degenerate latent
                    # dimensions that produce NaN for all remaining samples.
                    row_mask = np.all(np.isfinite(latent_values_batch), axis=1)
                    latent_values_batch = latent_values_batch[row_mask]
                    batch_samples = [s for s, keep in zip(batch_samples, row_mask) if keep]

                    if len(latent_values_batch):
                        col_mask = np.all(np.isfinite(latent_values_batch), axis=0)
                        latent_values_batch = latent_values_batch[:, col_mask]

                for sample, values in zip(batch_samples, latent_values_batch):

                    kwargs = {k: float(v) for k, v in zip(self.LATENT_KEYS, values)}

                    latent_samples.append(
                        Sample(
                            log_likelihood=sample.log_likelihood,
                            log_prior=sample.log_prior,
                            weight=sample.weight,
                            kwargs=kwargs,
                        )
                    )

            print(f"Time to compute latent variables: {time.time() - start_latent} seconds for {len(samples)} samples.")

            return type(samples)(
                sample_list=latent_samples,
                model=simple_model_for_kwargs(latent_samples[0].kwargs),
                samples_info=samples.samples_info,
            )

        except NotImplementedError:
            return None

    def compute_latent_variables(self, parameters, model) -> Dict[str, float]:
        """
        Override to compute latent variables from the instance.

        Latent variables are expressed as a dictionary:
        {"name": value}

        More complex models can be expressed by separating variables
        names by '.'
        {"name.attribute": value}

        Parameters
        ----------
        instance
            An instance of the model.

        Returns
        -------
        The computed latent variables.
        """
        raise NotImplementedError()

    def with_model(self, model):
        """
        Associate an explicit model with this analysis. Instances of the model
        will be used to compute log likelihood in place of the model passed
        from the search.

        Parameters
        ----------
        model
            A model to associate with this analysis

        Returns
        -------
        An analysis for that model
        """
        from .model_analysis import ModelAnalysis

        return ModelAnalysis(analysis=self, model=model)

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def save_attributes(self, paths: AbstractPaths):
        pass

    def save_results(self, paths: AbstractPaths, result: Result):
        pass

    def save_results_combined(self, paths: AbstractPaths, result: Result):
        pass

    def modify_before_fit(self, paths: AbstractPaths, model: AbstractPriorModel):
        """
        Overwrite this method to modify the attributes of the `Analysis` class before the non-linear search begins.

        An example use-case is using properties of the model to alter the `Analysis` class in ways that can speed up
        the fitting performed in the `log_likelihood_function`.
        """
        return self

    def modify_model(self, model):
        return model

    def modify_after_fit(
        self, paths: AbstractPaths, model: AbstractPriorModel, result: Result
    ):
        """
        Overwrite this method to modify the attributes of the `Analysis` class before the non-linear search begins.

        An example use-case is using properties of the model to alter the `Analysis` class in ways that can speed up
        the fitting performed in the `log_likelihood_function`.
        """
        return self

    def make_result(
        self,
        samples_summary: SamplesSummary,
        paths: AbstractPaths,
        samples: Optional[SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ) -> Result:
        """
        Returns the `Result` of the non-linear search after it is completed.

        The result type is defined as a class variable in the `Analysis` class. It can be manually overwritten
        by a user to return a user-defined result object, which can be extended with additional methods and attributes
        specific to the model-fit.

        The standard `Result` object may include:

        - The samples summary, which contains the maximum log likelihood instance and median PDF model.

        - The paths of the search, which are used for loading the samples and search internal below when a search
        is resumed.

        - The samples of the non-linear search (e.g. MCMC chains) also stored in `samples.csv`.

        - The non-linear search used for the fit in its internal representation, which is used for resuming a search
        and making bespoke visualization using the search's internal results.

        - The analysis used to fit the model (default disabled to save memory, but option may be useful for certain
        projects).

        Parameters
        ----------
        samples_summary
            The summary of the samples of the non-linear search, which include the maximum log likelihood instance and
            median PDF model.
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        samples
            The samples of the non-linear search, for example the chains of an MCMC run.
        search_internal
            The internal representation of the non-linear search used to perform the model-fit.
        analysis
            The analysis used to fit the model.

        Returns
        -------
        Result
            The result of the non-linear search, which is defined as a class variable in the `Analysis` class.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=analysis,
        )

    @property
    def supports_background_update(self) -> bool:
        """Whether this analysis supports background quick updates."""
        return False

    @property
    def supports_jax_visualization(self) -> bool:
        """Whether the visualizer can work directly with JAX arrays."""
        return self._use_jax

    def perform_quick_update(self, paths, instance):
        raise NotImplementedError

    def print_vram_use(self, model, batch_size : int) -> str:
        """
        Print JAX VRAM use for a given batch size.

        Parameters
        ----------
        batch_size
            The batch size to profile, which is the number of model evaluations JAX will perform simultaneously.
        """
        from autofit.non_linear.test_mode import skip_fit_output

        if skip_fit_output():
            return

        if not self._use_jax:
            print("use_jax=False for this analysis, therefore does not use GPU and VRAM use cannot be profiled.")
            return

        import jax
        import jax.numpy as jnp

        from autofit.non_linear.fitness import Fitness

        fitness = Fitness(
            model=model,
            analysis=self,
            fom_is_log_likelihood=True,
            use_jax_vmap=True,
            batch_size=batch_size,
        )

        parameters = np.zeros((batch_size, model.total_free_parameters))

        for i in range(batch_size):
            parameters[i, :] = model.physical_values_from_prior_medians

        parameters = jnp.array(parameters)

        batched_call = jax.jit(jax.vmap(fitness.call))
        lowered = batched_call.lower(parameters)
        compiled = lowered.compile()
        memory_analysis = compiled.memory_analysis()

        vram_bytes = (
                memory_analysis.output_size_in_bytes
                + memory_analysis.temp_size_in_bytes
        )

        if vram_bytes == 0:
            print(
                "VRAM USE = 0.000 GB "
                "(this likely means JAX is running in CPU-only mode)"
            )
        else:
            print(
                f"VRAM USE = {vram_bytes / 1024 ** 3:.3f} GB"
            )