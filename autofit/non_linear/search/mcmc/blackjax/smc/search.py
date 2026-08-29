from __future__ import annotations

import logging
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from autonerves import conf

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import Initializer, InitializerPrior
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mcmc.blackjax.chains import (
    InverseMassMatrixSpec,
    inverse_mass_matrix_kind_from,
    resolve_inverse_mass_matrix,
    stack_initial_positions,
)
from autofit.non_linear.search.mcmc.blackjax.smc.samples import SamplesSMC
from autofit.non_linear.test_mode import is_test_mode
from autofit.non_linear.samples.sample import Sample

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa

logger = logging.getLogger(__name__)

#: The unit-cube quantiles used to measure each prior's "natural" width (the central 68% interval, i.e. one
#: standard deviation either side of the median for a Gaussian). Mapping a *uniform* unit vector through
#: ``model.vector_from_unit_vector`` sends each entry through its own prior's inverse CDF, so this gives a
#: per-parameter scale for every prior type (Uniform, LogUniform, Gaussian, TruncatedGaussian) without any
#: per-prior special-casing.
_QUANTILE_LO = 0.15865
_QUANTILE_HI = 0.84135

#: The target width (in whitened units, where the *prior* has width 1) the auto step size aims at for a **cold**
#: run. Cold, we have no measurement of the posterior width, so this is a fixed fraction of the prior width and
#: reproduces the validated prototype's cold default. Cold SMC on a likelihood far sharper than the prior is not
#: the representative regime -- see the ``SMC`` class docstring.
_COLD_TARGET_WIDTH = 0.1

VALID_KERNELS = ("mala", "hmc")


class SMC(AbstractMCMC):
    __identifier_fields__ = (
        "num_particles",
        "kernel",
        "num_mcmc_steps",
        "num_integration_steps",
        "target_ess",
        "inverse_mass_matrix",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        num_particles: int = 256,
        kernel: str = "mala",
        num_mcmc_steps: int = 5,
        num_integration_steps: int = 8,
        target_ess: float = 0.5,
        step_size: Optional[float] = None,
        whiten_inflate: float = 2.0,
        max_smc_steps: int = 200,
        batch_size: int = 0,
        seed: int = 42,
        initializer: Optional[Initializer] = None,
        inverse_mass_matrix: InverseMassMatrixSpec = None,
        auto_correlation_settings: AutoCorrelationsSettings = AutoCorrelationsSettings(
            check_for_convergence=False
        ),
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: Optional[int] = None,
        number_of_cores: int = 1,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A BlackJAX adaptive tempered Sequential Monte Carlo (SMC) non-linear search with a **gradient** inner
        kernel.

        SMC anneals a cloud of ``num_particles`` particles along a tempering path from a normalised starting
        distribution (``lambda = 0``) to the posterior (``lambda = 1``), moving the particles at each temperature
        with an inner MCMC kernel and reweighting/resampling between temperatures. Two properties make it the lead
        sampler of the JAX-native wave:

        1. It yields the **log evidence for free** -- the sum of every step's tempering
           ``log_likelihood_increment`` is ``log Z``, on the same scale a nested sampler reports.
        2. Its inner kernel is gradient-based (MALA by default, HMC optionally), so it exploits the JAX-native
           differentiable likelihood rather than random-walking.

        The autofit ``Analysis`` must therefore be constructed with ``use_jax=True`` (and JAX run in float64,
        ``JAX_ENABLE_X64=True``); a clear error is raised at fit time otherwise.

        For BlackJAX, see https://github.com/blackjax-devs/blackjax.

        **Whitening -- and why it is the whole game.** The inner kernel takes a single *scalar* step size, so the
        target must be close to isotropic and unit-scale for any step to work. The search therefore samples in a
        whitened space ``z``, related to the physical parameters by the affine map ``x = shift + L @ z``, and this
        is what ``inverse_mass_matrix`` sets:

        - **Prior-whitening does not whiten the posterior.** Measured on a 15-parameter lens model, the posterior
          is 269x anisotropic in prior-whitened coordinates (one parameter's prior scale is 8.0 against a
          posterior std of 2e-4). A scalar step tuned to the mean of that spread is ~88x too large for the
          tightest parameter and acceptance pins at exactly 0.000.
        - **Diagonal whitening is not enough.** The posterior is correlated (condition number 568, correlation
          coefficient 0.95 between two parameters); diagonal whitening leaves a thin *tilted* ridge a spherical
          proposal walks off.
          Whitening uses the **Cholesky factor of the full covariance**.
        - **Cold, there is nothing better than the prior scale**, which is what ``inverse_mass_matrix=None``
          falls back to.

        **Warm starting -- and how the evidence survives it.** These samplers are meant to be handed a starting
        point near the maximum-likelihood solution by a JAX optimizer; cold-benchmarking them is not a fair test
        (on the lens model above, every cold arm sat ~190,000 log-units below the optimizer's solution). To warm
        start::

            search = af.SMC(
                initializer=result.start_point_from(),
                inverse_mass_matrix=result,
            )

        Simply dropping the particles at the previous best fit would **destroy the log evidence**, which is only
        valid when tempering starts from a *normalised* distribution. So the warm path does not shortcut
        ``lambda``: it still starts the tempering at ``lambda = 0``, but from a normalised Gaussian reference
        ``g = N(shift, L L^T)`` centred on the warm-start point rather than from the prior, bridging

        .. code::

            log target_lambda = log g + lambda * (log prior + log L - log g)

        onto BlackJAX's ``logprior_fn + lambda * loglikelihood_fn``. Because ``g`` integrates to 1, the
        accumulated increments still estimate ``log Z = log integral(prior * L)`` -- the true evidence, directly
        comparable to Nautilus. The prior must carry its normalisation here (it cancels in a Metropolis ratio but
        **not** in the evidence), and the whitening Jacobian ``log|det L|`` is added back.

        **Judging a run.** Never judge an SMC fit by "it reached ``lambda = 1``". When acceptance collapses to ~0,
        adaptive tempering can force a single huge increment straight to ``lambda = 1`` and report a meaningless
        ``log_evidence`` while still looking converged. Judge it by the per-step acceptance trace and the
        max-log-likelihood progression, both recorded in ``search_internal`` and surfaced on `SamplesSMC` as
        ``acceptance_rate_list`` / ``lambda_list`` / ``ess_list``.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output to.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            A unique tag for this model-fit, used as a folder between the path prefix and the search name and as
            the SQLite identifier.
        num_particles
            The number of SMC particles carried along the tempering path. This is also the number of posterior
            samples the fit returns.
        kernel
            The gradient inner kernel used to move the particles at each temperature: ``"mala"`` (default) or
            ``"hmc"``. HMC holds acceptance far higher as ``lambda -> 1`` but costs ~4x the likelihood/gradient
            evaluations; measured on the lens model, plain MALA matched its max log likelihood and log evidence at
            a quarter of the evaluations.
        num_mcmc_steps
            The number of inner-kernel updates applied to every particle at each temperature.
        num_integration_steps
            Leapfrog steps per HMC trajectory. Ignored when ``kernel="mala"``.
        target_ess
            The effective sample size, as a fraction of ``num_particles``, that adaptive tempering targets when
            choosing each ``lambda`` increment. Higher means a finer schedule (more, smaller steps).
        step_size
            The inner kernel's step size, in whitened units. ``None`` (default) auto-scales it from the target
            width (see below). **Units trap:** MALA proposes ``x + eps*grad + sqrt(2*eps)*xi``, so its
            ``step_size`` is a *squared* length and the proposal length is ``sqrt(2*eps)``; HMC's step size is a
            length. The auto-scaling applies the optimal MALA rule ``ell = 2.38 * d^(-1/6) * sigma``,
            ``eps = ell^2 / 2``, and ``sigma * d^(-1/4)`` for HMC, so a hand-set ``step_size`` must respect the
            same convention. There is deliberately **no per-temperature step adaptation**: BlackJAX's
            ``inner_kernel_tuning`` was measured to *collapse* acceptance (0.04-0.12, worse than a fixed step's
            0.84 -> 0.15 decay) on this problem.
        whiten_inflate
            The warm-start Gaussian reference is deliberately widened by this factor so it *covers* the posterior
            (the tempering path misses mass if the reference is narrower than the target). It also sets the auto
            step size: in whitened units the reference has width 1 but the posterior has width ``1 /
            whiten_inflate``, so the step targets the **posterior** width. Targeting the reference width instead
            overshoots by ``whiten_inflate^2`` and pins acceptance at zero (measured: ``eps=1.148 -> acc 0.00``,
            ``eps=0.1 -> acc 0.94``).
        max_smc_steps
            A hard cap on the number of SMC temperatures, so a pathological run terminates instead of annealing
            forever. A run that hits it is reported as not converged.
        batch_size
            Forwarded to BlackJAX: when ``> 0``, particles are processed in sequential batches of this size via
            ``jax.lax.map`` instead of one full ``jax.vmap``, trading speed for peak GPU memory. ``0`` (default)
            keeps the full ``vmap``.
        seed
            Integer seed passed to ``jax.random.PRNGKey``.
        initializer
            Generates the initial particles. Defaults to `InitializerPrior` -- **cold SMC's evidence requires the
            particles to be drawn from the prior**, so the `InitializerBall` default the other MCMC searches use
            would silently invalidate ``log_evidence``. When warm-started (``inverse_mass_matrix`` given), the
            initializer supplies only the *centre* of the Gaussian reference and the particles are drawn from that
            reference instead -- pass ``result.start_point_from()`` (equivalently
            ``InitializerParamStartPoints.from_result(result)``).
        inverse_mass_matrix
            The covariance the sampling space is whitened by (named for API parity with `BlackJAXNUTS`, where the
            same specification seeds window adaptation's metric):

            - ``None`` (default): cold. Whiten by the per-parameter prior width.
            - a 1-D ``numpy`` array of shape ``(n_dim,)``: a diagonal covariance.
            - a 2-D ``numpy`` array of shape ``(n_dim, n_dim)``: a full covariance, Cholesky-factorised.
            - a `Result` or `Samples` object: its ``samples.covariance_matrix``. Raises ``ValueError`` if that
              covariance looks MLE-only (too few samples, or non-finite / identity) -- pass an explicit array in
              that case, e.g. from a Laplace approximation.

            The ``"diagonal"`` / ``"dense"`` strings `BlackJAXNUTS` accepts are **rejected** here: they name an
            adaptation strategy, and SMC does not adapt its metric, so they would silently give a cold run.
        auto_correlation_settings
            Kept for API parity with the other MCMC searches. SMC particles are not a chain, so no
            auto-correlation diagnostics are computed and ``check_for_convergence`` defaults to ``False``.
        iterations_per_full_update
            Inherited from the autonerves config when ``None``. SMC calls ``perform_update`` once per
            temperature (SMC steps are coarse and few), so this does not chunk the run.
        number_of_cores
            Used only when generating the initial particles. The tempering loop itself runs vmapped on a single
            device.
        silence
            If True, the default print output of the non-linear search is silenced.
        session
            An SQLalchemy session instance.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer or InitializerPrior(),
            auto_correlation_settings=auto_correlation_settings,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs,
        )

        if kernel not in VALID_KERNELS:
            raise ValueError(
                f"SMC kernel must be one of {VALID_KERNELS}, got {kernel!r}"
            )

        if isinstance(inverse_mass_matrix, str):
            raise ValueError(
                "SMC does not adapt its metric, so the 'diagonal' / 'dense' strings BlackJAXNUTS accepts "
                "carry no information here and would silently give a cold (prior-whitened) run. Pass a "
                "Result/Samples object or an explicit covariance array to warm start, or None to run cold."
            )

        self.num_particles = num_particles
        self.kernel = kernel
        self.num_mcmc_steps = num_mcmc_steps
        self.num_integration_steps = num_integration_steps
        self.target_ess = target_ess
        self.step_size = step_size
        self.whiten_inflate = whiten_inflate
        self.max_smc_steps = max_smc_steps
        self.batch_size = batch_size
        self.seed = seed

        # As in `BlackJAXNUTS`: the raw specification is kept private (it is resolved into an actual covariance
        # at fit time, when `n_dim` is known) and the public attribute is the small descriptive string used by
        # `__identifier_fields__` and persisted onto `search_internal` / `samples_info`. Validated eagerly here so
        # a bad value fails at construction, not mid-fit.
        self._inverse_mass_matrix_spec = inverse_mass_matrix
        self.inverse_mass_matrix = inverse_mass_matrix_kind_from(inverse_mass_matrix)

        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating SMC Search")

        conf.instance["output"]["search_internal"] = True

    def apply_test_mode(self):
        logger.warning(
            "TEST MODE 1 (reduced iterations): SMC will run with num_particles<=16, num_mcmc_steps=2, "
            "max_smc_steps=5 for faster completion."
        )
        self.num_particles = min(self.num_particles, 16)
        self.num_mcmc_steps = 2
        self.max_smc_steps = 5

    @property
    def is_warm_start(self) -> bool:
        """
        Whether the run whitens (and tempers) from a warm-start covariance rather than from the prior. Set by
        passing a `Result` / `Samples` / covariance array as ``inverse_mass_matrix``.
        """
        return self._inverse_mass_matrix_spec is not None

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using BlackJAX adaptive tempered SMC. The autofit ``Analysis`` must be constructed with
        ``use_jax=True`` so the log-likelihood is JAX-traceable and differentiable.

        Returns
        -------
        (search_internal, fitness): tuple
            ``search_internal`` is the dict pickled under ``search_internal/``; ``fitness`` is the autofit
            Fitness wrapper.
        """
        import jax
        import jax.numpy as jnp
        import blackjax
        import blackjax.smc.resampling as resampling

        # JAX is mandatory: MALA/HMC need gradients of the log-density. Refuse cleanly if the analysis was built
        # without ``use_jax=True``.
        xp = getattr(analysis, "_xp", None)
        if xp is None or not xp.__name__.startswith("jax"):
            raise ValueError(
                "SMC requires an Analysis built with use_jax=True. Its inner kernel (MALA/HMC) is "
                "gradient-based and the log-likelihood must flow through jax.grad. Construct "
                "Analysis(..., use_jax=True) and call enable_pytrees() / register_model(model) before fit. "
                "See autofit_workspace_test/scripts/searches/BlackJAXNUTS.py for a worked example of the "
                "sibling BlackJAX search."
            )

        n_dim = model.prior_count

        # `fom_is_log_likelihood=True`: SMC needs the log likelihood and the log prior as *separate* functions
        # (blackjax tempers `logprior_fn + lambda * loglikelihood_fn`), so the log-posterior form the other
        # gradient searches use would double-count the prior.
        #
        # The resample sentinel is the library-standard `-1e99` rather than `-inf`: blackjax's ESS root solver
        # multiplies the log-likelihood by a candidate `delta` that can be exactly 0.0, and `0 * -inf` is NaN.
        # A large finite negative value gives the same zero weight without that hazard.
        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=-1.0e99,
            iterations_per_quick_update=self.iterations_per_quick_update,
            background_quick_update=self.quick_update_background,
            live_visual_update=self.live_visual_update,
        )

        # ---- Whitening ---------------------------------------------------
        #
        # x = shift + L @ z. Warm: L is the (inflated) Cholesky factor of the warm-start covariance and shift is
        # the warm-start point. Cold: L is the diagonal of prior widths and shift is the prior median.
        _, covariance_seed = resolve_inverse_mass_matrix(
            self._inverse_mass_matrix_spec, n_dim=n_dim
        )

        prior_scales = prior_scales_from(model=model)
        lower_limits, upper_limits = prior_limits_from(model=model)

        # ---- Initial points ----------------------------------------------
        #
        # Cold, the drawn points *are* the particles, and they must be prior draws for the evidence to be valid.
        # Warm, only their centre is needed (the particles are drawn from the Gaussian reference itself), and a
        # start-point initializer returns the same point every time -- which `samples_from_model` rejects when
        # asked for more than one point.
        n_start_points = 1 if self.is_warm_start else self.num_particles

        if not self.is_warm_start and not isinstance(self.initializer, InitializerPrior):
            logger.warning(
                f"SMC is running cold with a {type(self.initializer).__name__} initializer. Cold SMC's "
                "log_evidence is only valid when the initial particles are drawn from the prior; use the "
                "default InitializerPrior, or warm start by passing inverse_mass_matrix=result."
            )

        if self.is_warm_start and isinstance(self.initializer, InitializerPrior):
            logger.warning(
                "SMC is warm-starting from a covariance but the default InitializerPrior, so the Gaussian "
                "reference will be centred on a single random prior draw rather than on the previous best fit. "
                "Pass initializer=result.start_point_from() alongside inverse_mass_matrix=result."
            )

        _, parameter_lists, _ = self.initializer.samples_from_model(
            total_points=n_start_points,
            model=model,
            fitness=fitness,
            paths=self.paths,
            n_cores=self.number_of_cores,
        )

        self.plot_start_point(
            parameter_vector=parameter_lists[0],
            model=model,
            analysis=analysis,
        )

        start_points = stack_initial_positions(parameter_lists)

        if self.is_warm_start:
            shift = start_points.mean(axis=0)
        else:
            shift = np.asarray(
                model.vector_from_unit_vector(unit_vector=[0.5] * n_dim), dtype=float
            )

        whitening, whitening_kind = whitening_from(
            covariance_seed=covariance_seed,
            prior_scales=prior_scales,
            inflate=self.whiten_inflate,
        )
        whitening_inverse = np.linalg.inv(whitening)

        # log|det L| -- the Jacobian of x = shift + L @ z. See `_log_evidence_from` for when it is applied.
        log_jacobian = float(np.linalg.slogdet(whitening)[1])

        shift_jax = jnp.asarray(shift)
        whitening_jax = jnp.asarray(whitening)
        lower_jax = jnp.asarray(lower_limits)
        upper_jax = jnp.asarray(upper_limits)

        def physical_from_z(z):
            return shift_jax + whitening_jax @ z

        def log_prior_z(z):
            # `-inf` outside a bounded prior's support, which is what rejects an out-of-bounds proposal.
            parameters = physical_from_z(z)
            return jnp.sum(
                jnp.asarray(
                    model.log_prior_list_from_vector(vector=parameters, xp=jnp)
                )
            )

        def log_likelihood_z(z):
            # Clip to the prior bounds before the likelihood so an out-of-bounds proposal returns a *finite*
            # value with zero gradient (`clip` has zero gradient outside its bounds); `log_prior_z` above is what
            # rejects it. Without this the likelihood may be evaluated on a nonsensical instance and return NaN,
            # whose derivative is already on the autodiff tape by the time any guard sees it.
            parameters = jnp.clip(physical_from_z(z), lower_jax, upper_jax)
            return fitness.call(parameters)

        # ---- The tempering path ------------------------------------------
        if self.is_warm_start:
            # The reference g is EXACTLY the standard normal in these coordinates -- we whitened by its own
            # scale -- so it is normalised by construction, which is what keeps the evidence valid.
            log_prior_normalisation = log_prior_normalisation_from(model=model)
            reference_log_normalisation = -0.5 * n_dim * math.log(2.0 * math.pi)

            def log_reference_z(z):
                return reference_log_normalisation - 0.5 * jnp.sum(z**2)

            def smc_logprior_fn(z):
                return log_reference_z(z)

            def smc_loglikelihood_fn(z):
                return (
                    log_prior_z(z)
                    + log_prior_normalisation
                    + log_likelihood_z(z)
                    - log_reference_z(z)
                )

        else:
            log_prior_normalisation = 0.0
            smc_logprior_fn = log_prior_z
            smc_loglikelihood_fn = log_likelihood_z

        # ---- Inner kernel -------------------------------------------------
        effective_step_size = self.effective_step_size(n_dim=n_dim)

        mcmc_step_fn, mcmc_init_fn = inner_kernel_from(
            kernel=self.kernel,
            n_dim=n_dim,
            num_integration_steps=self.num_integration_steps,
        )

        # A leading dimension of 1 tells blackjax this parameter is shared across every particle.
        mcmc_parameters = {"step_size": jnp.asarray([effective_step_size])}

        smc = blackjax.adaptive_tempered_smc(
            logprior_fn=smc_logprior_fn,
            loglikelihood_fn=smc_loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=mcmc_parameters,
            resampling_fn=resampling.systematic,
            target_ess=self.target_ess,
            num_mcmc_steps=self.num_mcmc_steps,
            batch_size=self.batch_size,
        )

        rng_key = jax.random.PRNGKey(self.seed)
        rng_key, init_key = jax.random.split(rng_key)

        if self.is_warm_start:
            # Required for the bridge's log Z: at lambda = 0 the particles must be distributed as the reference.
            initial_particles = jax.random.normal(
                init_key, shape=(self.num_particles, n_dim)
            )
        else:
            initial_particles = jnp.asarray(
                (start_points - shift) @ whitening_inverse.T
            )

        state = smc.init(initial_particles)

        vmapped_log_likelihood = jax.jit(jax.vmap(log_likelihood_z))

        self.logger.info(
            f"SMC: adaptive tempered SMC ({self.num_particles} particles, kernel={self.kernel}, "
            f"num_mcmc_steps={self.num_mcmc_steps}, target_ess={self.target_ess}, "
            f"step_size={effective_step_size:.4g}, whitening={whitening_kind}, "
            f"start={'warm' if self.is_warm_start else 'cold'})"
        )

        lambda_list: List[float] = []
        log_likelihood_increment_list: List[float] = []
        acceptance_rate_list: List[float] = []
        ess_list: List[float] = []
        max_log_likelihood_list: List[float] = []

        log_evidence_bridge = 0.0
        n_smc_steps = 0

        search_internal = None

        while float(state.tempering_param) < 1.0 and n_smc_steps < self.max_smc_steps:
            rng_key, step_key = jax.random.split(rng_key)
            state, info = jax.block_until_ready(smc.step(step_key, state))

            n_smc_steps += 1
            log_evidence_bridge += float(info.log_likelihood_increment)

            step_log_likelihood = vmapped_log_likelihood(state.particles)

            lambda_list.append(float(state.tempering_param))
            log_likelihood_increment_list.append(float(info.log_likelihood_increment))
            acceptance_rate_list.append(
                float(jnp.mean(info.update_info.acceptance_rate))
            )
            ess_list.append(ess_from_weights(np.asarray(state.weights)))
            max_log_likelihood_list.append(float(jnp.max(step_log_likelihood)))

            self.logger.info(
                f"SMC: step {n_smc_steps} lambda={lambda_list[-1]:.4f} "
                f"acceptance={acceptance_rate_list[-1]:.3f} ess={ess_list[-1]:.1f} "
                f"max log L={max_log_likelihood_list[-1]:.4f} "
                f"log Z (running)={log_evidence_bridge + (log_jacobian if self.is_warm_start else 0.0):.4f}"
            )

            # Persist the particles de-whitened, in physical parameter space, and clipped to the prior support
            # so the record is always a valid model instance. Only the warm path's initial reference draw can
            # sit outside a bounded prior, and those particles carry zero weight from the first temperature
            # onward -- so the clip never moves a particle that contributes to the posterior, it just keeps
            # `model.log_prior_list_from` finite for the interim `perform_update`.
            physical_particles = np.clip(
                np.asarray(state.particles) @ whitening.T + shift,
                lower_limits,
                upper_limits,
            )

            search_internal = _build_search_internal(
                particles=physical_particles,
                weights=np.asarray(state.weights),
                log_likelihood_list=np.asarray(step_log_likelihood),
                lambda_list=lambda_list,
                log_likelihood_increment_list=log_likelihood_increment_list,
                acceptance_rate_list=acceptance_rate_list,
                ess_list=ess_list,
                max_log_likelihood_list=max_log_likelihood_list,
                log_evidence=_log_evidence_from(
                    log_evidence_bridge=log_evidence_bridge,
                    log_jacobian=log_jacobian,
                    is_warm_start=self.is_warm_start,
                ),
                log_jacobian=log_jacobian,
                num_particles=self.num_particles,
                kernel=self.kernel,
                num_mcmc_steps=self.num_mcmc_steps,
                num_integration_steps=self.num_integration_steps,
                target_ess=self.target_ess,
                step_size=effective_step_size,
                whitening_kind=whitening_kind,
                is_warm_start=self.is_warm_start,
                warm_start_source=type(self.initializer).__name__,
                inverse_mass_matrix_kind=self.inverse_mass_matrix,
                n_smc_steps=n_smc_steps,
                max_smc_steps=self.max_smc_steps,
            )

            self.output_search_internal(search_internal=search_internal)

            if float(state.tempering_param) < 1.0 and n_smc_steps < self.max_smc_steps:
                self.perform_update(
                    model=model,
                    analysis=analysis,
                    search_internal=search_internal,
                    fitness=fitness,
                    during_analysis=True,
                )

        if search_internal is None:
            raise RuntimeError(
                "SMC terminated before taking a single tempering step -- max_smc_steps must be at least 1."
            )

        if not search_internal["converged"]:
            logger.warning(
                f"SMC hit max_smc_steps={self.max_smc_steps} at lambda="
                f"{lambda_list[-1]:.4f} without reaching 1.0; log_evidence is a partial-path value and the "
                "particles are not posterior samples."
            )

        return search_internal, fitness

    def effective_step_size(self, n_dim: int) -> float:
        """
        The inner kernel's step size in whitened units: ``step_size`` if given, otherwise auto-scaled from the
        target width.

        Warm-started, the target width is the **posterior** width, which in whitened units is
        ``1 / whiten_inflate`` (the reference is inflated to cover the posterior, so scaling to the reference's
        own width of 1 overshoots by ``whiten_inflate^2``). Cold, the posterior width is unknown and
        `_COLD_TARGET_WIDTH` is used.

        Parameters
        ----------
        n_dim
            The number of free model parameters, which sets the dimension scaling of the optimal step.
        """
        if self.step_size is not None:
            return float(self.step_size)

        target_width = (
            1.0 / self.whiten_inflate if self.is_warm_start else _COLD_TARGET_WIDTH
        )

        return auto_step_size_from(
            target_width=target_width, n_dim=n_dim, kernel=self.kernel
        )

    # ------------------------------------------------------------------
    # Persistence + samples
    # ------------------------------------------------------------------

    @property
    def backend_filename(self):
        return self.paths.search_internal_path / "search_internal.pickle"

    @property
    def backend(self) -> dict:
        """Load the pickled search-internal dict written by ``_fit``."""
        if not Path(self.backend_filename).is_file():
            raise FileNotFoundError(
                f"search_internal.pickle does not exist at {self.paths.search_internal_path}"
            )
        with open(self.backend_filename, "rb") as f:
            return pickle.load(f)

    def output_search_internal(self, search_internal):
        """
        Pickle the search-internal dict.

        BlackJAX has no native on-disk format (cf. emcee's HDFBackend), so we round-trip the particle cloud +
        tempering diagnostics via pickle, bypassing ``self.paths.save_search_internal`` for the same reason
        `BlackJAXNUTS` does: the autofit dill path chokes on a few numpy/jax-backed members, and a direct pickle
        of already-numpy data is robust.

        ``NullPaths`` (no ``name``/``path_prefix``) sets ``search_internal_path`` to ``None`` to suppress disk
        output -- skip silently in that case.
        """
        if self.paths.search_internal_path is None:
            return
        os.makedirs(self.paths.search_internal_path, exist_ok=True)
        with open(self.backend_filename, "wb") as f:
            pickle.dump(search_internal, f)

    def _test_mode_samples_info(self) -> dict:
        return {
            "num_particles": int(self.num_particles),
            "kernel": self.kernel,
            "num_mcmc_steps": int(self.num_mcmc_steps),
            "target_ess": float(self.target_ess),
            "log_evidence": None,
            "lambda_list": [],
            "acceptance_rate_list": [],
            "ess_list": [],
            "max_log_likelihood_list": [],
            "n_smc_steps": 0,
            "converged": False,
            "warm_start_source": type(self.initializer).__name__,
            "inverse_mass_matrix_kind": self.inverse_mass_matrix,
            "total_samples": 0,
        }

    def samples_info_from(self, search_internal=None):
        search_internal = (
            search_internal if search_internal is not None else self.backend
        )

        return {
            "num_particles": int(search_internal["num_particles"]),
            "kernel": search_internal["kernel"],
            "num_mcmc_steps": int(search_internal["num_mcmc_steps"]),
            "num_integration_steps": int(search_internal["num_integration_steps"]),
            "target_ess": float(search_internal["target_ess"]),
            "step_size": float(search_internal["step_size"]),
            "log_evidence": float(search_internal["log_evidence"]),
            "log_jacobian": float(search_internal["log_jacobian"]),
            "lambda_list": [float(x) for x in search_internal["lambda_list"]],
            "log_likelihood_increment_list": [
                float(x) for x in search_internal["log_likelihood_increment_list"]
            ],
            "acceptance_rate_list": [
                float(x) for x in search_internal["acceptance_rate_list"]
            ],
            "ess_list": [float(x) for x in search_internal["ess_list"]],
            "max_log_likelihood_list": [
                float(x) for x in search_internal["max_log_likelihood_list"]
            ],
            "n_smc_steps": int(search_internal["n_smc_steps"]),
            "converged": bool(search_internal["converged"]),
            "is_warm_start": bool(search_internal["is_warm_start"]),
            "whitening_kind": search_internal["whitening_kind"],
            "warm_start_source": search_internal.get("warm_start_source", "none"),
            "inverse_mass_matrix_kind": search_internal.get(
                "inverse_mass_matrix_kind", "none"
            ),
            "total_samples": int(search_internal["num_particles"]),
            "time": self.timer.time if self.timer else None,
        }

    def samples_via_internal_from(self, model, search_internal=None):
        """
        Convert the particle cloud pickled under ``search_internal/`` into a `SamplesSMC`.

        SMC particles are **weighted** samples: each carries the normalised importance weight blackjax
        assigns it at the current temperature, so the weights (not a uniform 1.0) are what the PDF, medians
        and errors are computed from.
        """
        search_internal = (
            search_internal if search_internal is not None else self.backend
        )

        parameter_lists = np.asarray(search_internal["particles"]).tolist()
        log_likelihood_list = [
            float(x) for x in np.asarray(search_internal["log_likelihood_list"])
        ]
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)

        weights = np.asarray(search_internal["weights"], dtype=float)
        weight_sum = weights.sum()
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            weights = np.full(weights.shape, 1.0 / max(weights.size, 1))
        else:
            weights = weights / weight_sum

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weights.tolist(),
        )

        return SamplesSMC(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
        )


# --------------------------------------------------------------------------
# Module-level helpers (kept outside the class so jax.vmap / jax.jit do not
# capture ``self`` and trip pytree complaints)
# --------------------------------------------------------------------------


def prior_scales_from(model: AbstractPriorModel) -> np.ndarray:
    """
    The per-parameter "natural" width of the prior: half the central 68% quantile interval.

    A *uniform* unit vector maps through ``model.vector_from_unit_vector`` one prior at a time, so this measures
    every prior type (Uniform, LogUniform, Gaussian, TruncatedGaussian) on its own terms with no special-casing:
    for a Gaussian it recovers ``sigma``, for a Uniform it is 0.68 of the half-width.

    This is the whitening a **cold** run falls back on. It is the best available with no posterior information,
    but note it is *not* a posterior whitening -- see the `SMC` docstring on the measured 269x anisotropy.
    """
    n_dim = model.prior_count

    lower = np.asarray(
        model.vector_from_unit_vector(unit_vector=[_QUANTILE_LO] * n_dim), dtype=float
    )
    upper = np.asarray(
        model.vector_from_unit_vector(unit_vector=[_QUANTILE_HI] * n_dim), dtype=float
    )

    scales = 0.5 * (upper - lower)

    # A degenerate prior (zero or non-finite width) would make the whitening singular.
    return np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)


def prior_limits_from(model: AbstractPriorModel) -> Tuple[np.ndarray, np.ndarray]:
    """
    The per-parameter ``(lower_limit, upper_limit)`` arrays of the model's priors, used to clip a proposal to the
    prior's support before the likelihood is evaluated.
    """
    lower = np.asarray(
        [float(prior.lower_limit) for prior in model.priors_ordered_by_id], dtype=float
    )
    upper = np.asarray(
        [float(prior.upper_limit) for prior in model.priors_ordered_by_id], dtype=float
    )
    return lower, upper


def log_prior_normalisation_from(model: AbstractPriorModel) -> float:
    """
    The total additive constant that ``Prior.log_prior_from_value`` drops.

    That constant is irrelevant to posterior *shape* (it cancels in a Metropolis ratio) but it does **not** cancel
    in the evidence: ``log Z = log integral(prior * L)`` requires ``prior`` to be a normalised density. The warm
    bridge divides by a normalised Gaussian reference, so it must add this back; a cold run does not need it (the
    SMC estimator is a ratio in which it cancels).
    """
    return float(
        sum(prior.log_normalisation() for prior in model.priors_ordered_by_id)
    )


def whitening_from(
    covariance_seed: Optional[np.ndarray],
    prior_scales: np.ndarray,
    inflate: float,
) -> Tuple[np.ndarray, str]:
    """
    Build the whitening matrix ``L`` of the affine map ``x = shift + L @ z``.

    Parameters
    ----------
    covariance_seed
        The covariance resolved from ``inverse_mass_matrix`` by `chains.resolve_inverse_mass_matrix`: ``None``
        for a cold run, a 1-D array for a diagonal covariance, or a 2-D array for a full covariance.
    prior_scales
        The per-parameter prior widths from `prior_scales_from`, used when ``covariance_seed`` is ``None``.
    inflate
        The factor the warm-start reference is widened by so it covers the posterior.

    Returns
    -------
    (whitening, kind)
        The ``(n_dim, n_dim)`` matrix and a short string naming how it was built, recorded on ``search_internal``.
    """
    if covariance_seed is None:
        return np.diag(prior_scales), "prior-scale diagonal (cold)"

    if covariance_seed.ndim == 1:
        return np.diag(np.sqrt(covariance_seed) * inflate), "diagonal covariance"

    return np.linalg.cholesky(covariance_seed) * inflate, "full-covariance Cholesky"


def auto_step_size_from(target_width: float, n_dim: int, kernel: str) -> float:
    """
    The inner-kernel step size that moves a particle by ``target_width`` in whitened units.

    **Units trap.** MALA proposes ``x + eps*grad + sqrt(2*eps)*xi``, so its ``step_size`` ``eps`` is a *squared*
    length and the proposal length is ``sqrt(2*eps)``, not ``eps``. Setting ``eps ~ sigma`` therefore overshoots by
    ~``1/sigma`` and gives zero acceptance at every temperature (measured: ``eps=0.0029`` against ``sigma=0.005``
    gave a proposal ~16x wider than the posterior and acceptance 0.000). The optimal MALA scaling is
    ``ell = 2.38 * d^(-1/6) * sigma`` with ``eps = ell^2 / 2``. HMC's ``step_size`` *is* a length (the leapfrog
    step), so it scales as ``sigma`` directly, with the standard ``d^(-1/4)`` dimension penalty.
    """
    if kernel == "mala":
        length = 2.38 * n_dim ** (-1.0 / 6.0) * target_width
        return float(0.5 * length * length)

    return float(target_width * n_dim ** (-0.25))


def inner_kernel_from(kernel: str, n_dim: int, num_integration_steps: int):
    """
    Build the ``(mcmc_step_fn, mcmc_init_fn)`` pair blackjax's SMC machinery vmaps across particles.

    MALA takes a scalar ``step_size`` and no metric argument -- the whitening, not an array step or a mass
    matrix, is what preconditions it. HMC additionally takes an inverse mass matrix (the identity here, since the
    space is already whitened) and a leapfrog trajectory length.
    """
    import jax.numpy as jnp
    import blackjax.mcmc.hmc as hmc
    import blackjax.mcmc.mala as mala

    if kernel == "mala":
        mala_kernel = mala.build_kernel()

        def mcmc_step_fn(rng_key, state, logdensity_fn, step_size):
            return mala_kernel(rng_key, state, logdensity_fn, step_size)

        return mcmc_step_fn, mala.init

    hmc_kernel = hmc.build_kernel()
    inverse_mass_matrix = jnp.ones(n_dim)

    def mcmc_step_fn(rng_key, state, logdensity_fn, step_size):
        return hmc_kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        )

    return mcmc_step_fn, hmc.init


def ess_from_weights(weights: np.ndarray) -> float:
    """
    The effective sample size of a set of importance weights, ``1 / sum(w^2)`` for normalised ``w``.

    Adaptive tempering chooses each ``lambda`` increment to hold this at ``target_ess * num_particles``, so a
    recorded value far below that target means the schedule solver could not meet its target and the step was a
    forced jump.
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()

    if not np.isfinite(total) or total <= 0.0:
        return float("nan")

    normalised = weights / total
    return float(1.0 / np.sum(normalised**2))


def _log_evidence_from(
    log_evidence_bridge: float, log_jacobian: float, is_warm_start: bool
) -> float:
    """
    Convert the accumulated SMC increments into the physical log evidence.

    The SMC estimator is the *ratio* ``log[integral(pi_0 L) / integral(pi_0)]`` in whitened coordinates, so which
    constants survive depends on what ``pi_0`` is:

    - **Cold**, ``pi_0(z) = prior(x(z))``, which integrates to ``1/|det L|`` over ``z``. Both the whitening
      Jacobian and the prior's normalisation appear in the numerator and the denominator and cancel exactly, so
      the increments already sum to the physical ``log Z``.
    - **Warm**, ``pi_0(z) = N(0, I)(z)``, which integrates to 1. The numerator is ``integral(prior(x(z))
      L(x(z)) dz) = Z / |det L|``, so ``log|det L|`` has to be added back (and the prior's normalisation, which no
      longer cancels, is already carried inside the bridge weight).
    """
    if is_warm_start:
        return float(log_evidence_bridge + log_jacobian)
    return float(log_evidence_bridge)


def _build_search_internal(
    particles,
    weights,
    log_likelihood_list,
    lambda_list,
    log_likelihood_increment_list,
    acceptance_rate_list,
    ess_list,
    max_log_likelihood_list,
    log_evidence,
    log_jacobian,
    num_particles,
    kernel,
    num_mcmc_steps,
    num_integration_steps,
    target_ess,
    step_size,
    whitening_kind,
    is_warm_start,
    warm_start_source,
    inverse_mass_matrix_kind,
    n_smc_steps,
    max_smc_steps,
):
    """
    Glue the SMC loop's output into the persistence dict pickled under ``search_internal/search_internal.pickle``.

    ``particles`` are in **physical** parameter space (de-whitened) so the record can be read without knowing the
    whitening, and ``weights`` are their normalised importance weights at the current temperature.

    ``converged`` records only that ``lambda`` reached 1.0. It is a weak criterion on its own -- adaptive
    tempering can force a single huge increment to 1.0 once acceptance has collapsed -- so the acceptance and
    max-log-likelihood traces are persisted beside it and are what a run should be judged by.
    """
    return {
        "particles": np.asarray(particles),
        "weights": np.asarray(weights),
        "log_likelihood_list": np.asarray(log_likelihood_list),
        "lambda_list": list(lambda_list),
        "log_likelihood_increment_list": list(log_likelihood_increment_list),
        "acceptance_rate_list": list(acceptance_rate_list),
        "ess_list": list(ess_list),
        "max_log_likelihood_list": list(max_log_likelihood_list),
        "log_evidence": log_evidence,
        "log_jacobian": log_jacobian,
        "num_particles": num_particles,
        "kernel": kernel,
        "num_mcmc_steps": num_mcmc_steps,
        "num_integration_steps": num_integration_steps,
        "target_ess": target_ess,
        "step_size": step_size,
        "whitening_kind": whitening_kind,
        "is_warm_start": is_warm_start,
        "warm_start_source": warm_start_source,
        "inverse_mass_matrix_kind": inverse_mass_matrix_kind,
        "n_smc_steps": n_smc_steps,
        "max_smc_steps": max_smc_steps,
        "converged": bool(lambda_list) and float(lambda_list[-1]) >= 1.0,
    }
