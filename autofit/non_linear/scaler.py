"""
Per-parameter step scaling for the gradient searches.

The problem
-----------

``MultiStartGradient`` steps in **physical** parameter space with a single global
step scale, while prior box widths across one model routinely span orders of
magnitude. On the reference ``imaging/mge`` lens model they span **40x**::

    mass.einstein_radius   UniformPrior(0, 8)        width 8.00
    shear.gamma_1/2        UniformPrior(-0.3, 0.3)   width 0.60
    bulge.centre_0/1       UniformPrior(-0.1, 0.1)   width 0.20

A step that is sensible for ``einstein_radius`` is a wall-crossing for
``bulge.centre``. Lanes are not initialised at the edges -- ``_broad_starts``
draws in the unit cube at ``(0.15, 0.85)``, the middle 70% of every prior -- they
*walk* there, and on that cell clipping fires on 31.5% of lane-steps
(autolens_profiling#131).

Every other search family already handles this
---------------------------------------------

The multi-start gradient family is the **only** search in the library that steps
without adapting to per-parameter scale:

===========================  =========================================  ==============
search                       step mechanism                             adapts scale?
===========================  =========================================  ==============
Nautilus / Dynesty           proposals in the unit cube                 by construction
Emcee                        affine-invariant stretch move              yes, provably
Zeus                         ensemble slice sampling (``tune=True``)    yes, per direction
BlackJAXNUTS                 ``blackjax.window_adaptation``             yes -- step size AND mass matrix
LBFGS / BFGS                 quasi-Newton inverse-Hessian estimate      partial -- starts from the identity
Drawer                       independent draws from the priors          n/a, no stepping
MultiStart{Adam,...,Prodigy} fixed / global step in physical space      **no**
===========================  =========================================  ==============

The reason this family is the exception is specific, and it is not an oversight
that can be fixed by tuning: Adam-family rules normalise per-coordinate by
**gradient magnitude**, not **parameter scale**. Since ``m_hat / sqrt(v_hat) ~ 1``
for a consistent gradient, every coordinate moves by roughly ``learning_rate`` in
*physical* units -- equalising precisely the wrong thing when the natural scales
differ 40x. Prodigy inherits that per-coordinate behaviour and adds a single
**global** estimate ``d``, so it cannot repair a per-coordinate problem either.

A ``Scaler`` is therefore the direct analogue of NUTS's **mass matrix** and
L-BFGS's **inverse Hessian**: a diagonal preconditioner, an entirely established
idea that this one family happens to lack.

Static, and what that costs
---------------------------

NUTS *learns* its preconditioner from the geometry during warmup. A ``Scaler``
is **static** and derived from the priors, which is a deliberate trade: it costs
one array multiply, needs no warmup, and cannot itself introduce a failure mode.
The price is that **a prior width is a proxy for the posterior scale**, and where
the likelihood is far tighter than the prior the proxy is poor -- the coordinate
is then merely under-scaled rather than mis-scaled, but the benefit shrinks. A
diagonal learned from the observed gradient/step history is the natural successor
if this proves too blunt.

What "one scaled unit" does and does not mean
---------------------------------------------

:class:`ScalerPriorWidth` mixes ``width`` for uniform priors with ``sigma`` for
Gaussian ones, so **one scaled unit does not carry a single consistent
probabilistic interpretation** across coordinates. That is a real limitation and
it is stated rather than papered over.

It is acceptable because the goal here is an **engineering** one -- stop a step
that suits a wide coordinate from crossing a narrow coordinate's wall -- and for
that, order-of-magnitude commensurability is what matters. It is *not* a claim to
prior-standardised coordinates.

If genuinely prior-standardised coordinates are ever wanted, the unified
definitions to reach for, in rough order of sophistication, are: the prior's
actual **standard deviation**; a **robust central quantile width** (e.g. the
inter-quantile range between the 15.9% and 84.1% points, which degrades
gracefully for heavy-tailed and bounded priors alike); or the **local inverse-CDF
derivative** ``dtheta/du`` evaluated at a reference point, which is the
continuous generalisation of what the uniform and log-uniform rules already do.
Each is a strictly better-defined quantity than the mixture above.

None of them is implemented, deliberately. The simple width/sigma version is the
one to measure first: if it does not move the clip rate, a more principled scale
definition will not either, and the diagnosis behind the whole feature is wrong.

Why a linear change of variables, and not the unit cube
-------------------------------------------------------

Stepping in the unit cube would make the step automatically commensurate with
every box width, and it was rejected -- deliberately, and the objections stand:

- a **logit** reparameterisation sends an optimum that genuinely sits on a
  boundary to infinity, and the reference cell demonstrably has such optima
  (6 of 16 lanes end pinned to a bound);
- the **inverse-CDF** transform for non-uniform priors has ``dtheta/du -> inf``
  at the cube faces, trading one numerical hazard for another;
- it invalidates every stored benchmark.

Scaling by a constant diagonal buys the same normalisation and dodges all three:
the map is **linear**, so boundary optima stay finite and no face singularity
exists.

The invariance that must not be broken
--------------------------------------

The search runs in ``phi = theta / s`` and evaluates the objective at
``theta = s * phi`` -- the **physical-space** posterior, at the transformed point.
The Jacobian of a constant diagonal map is constant, so it adds a constant to the
log-density and **cannot move the MAP**.

Optimising the density **of ``phi``** instead would fold a non-constant Jacobian
into the objective and move the answer -- and it would fail *silently*, since
every counter and diagnostic would look healthy. That is why the objective is
composed as ``fitness(s * phi)`` rather than the prior being re-expressed, and
why ``test_scaler.py`` pins the invariance directly.

The scale must also be applied as a **change of variables**, never as a post-hoc
``params += s * updates``: Prodigy estimates its step scale from the distance
actually travelled (``params0``, ``grad_sum``), so rescaling its update
externally would leave its own estimate inconsistent with the trajectory it
believes it took. Run the rule in ``phi`` and it needs no changes at all.

Composition with ``Clipper``
----------------------------

A ``Scaler`` and a :mod:`~autofit.non_linear.clipper` are **complementary, not
alternatives**, and this stays true however well scaling works. Even if scaling
drove the clip rate from 31% to 0.1%, projection would be retained as the
last-line invariant.

The reason is not hygiene, it is **constrained-optimizer semantics**. A search
that advertises itself as optimising a posterior with hard prior support is
solving a *constrained* problem, and a state outside that support is not an
inconveniently poor candidate -- it is **infeasible**. Projection onto the box:

- does not alter the constrained optimum;
- makes that invariant explicit rather than incidental;
- prevents silent invalid trajectories;
- prevents prior exits from masking later pathologies;
- makes the search diagnostics interpretable at all.

Scaling treats the *cause* -- it reduces how often a lane reaches a wall -- and
can only ever make overshoot rarer, never impossible: a large gradient, bad
curvature, or a grown step scale can still cross. Without clipping that crossing
is silent. And where the likelihood genuinely prefers a value outside the prior,
**the clipped lane sitting on the bound is the correct MAP answer under the
declared prior** -- only clipping can express it.

``AbstractClipper.project`` accepts the scale so it can clip in ``phi`` against
``bounds / s``.

On the clipper's default
------------------------

Hard-support enforcement is the **intended** default for the gradient/MLE
searches on those semantics. It is not the default yet, and the hesitation is
empirical breadth rather than mathematics: everything measured so far is
``MultiStartProdigy`` on a single lens cell, and ``MultiStartAdam`` /
``MultiStartLion`` / ``MultiStartADABelief`` remain unmeasured -- which matters,
because the fixed-rate rules are *more* exposed to the underlying problem, not
less (Adam steps by roughly ``learning_rate`` in physical units in every
coordinate; Lion, being sign-based, by exactly it).

When the default does flip, it must not be sold as a long-budget accuracy
improvement. It is not one: measured at 16x3000 the answer does not move at all,
and at 105 steps the finite-budget gain was 114 nats. The case is correctness of
the optimisation problem being solved.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple

import numpy as np

from autofit.mapper.prior.gaussian import GaussianPrior
from autofit.mapper.prior.log_gaussian import LogGaussianPrior
from autofit.mapper.prior.log_uniform import LogUniformPrior
from autofit.mapper.prior.truncated_gaussian import TruncatedGaussianPrior
from autofit.mapper.prior.uniform import UniformPrior

logger = logging.getLogger(__name__)


class ScaleTerm(NamedTuple):
    """
    One parameter's contribution to the scale vector, kept in a struct so that
    :meth:`AbstractScaler.scale_from_model` and
    :meth:`AbstractScaler.info_from_model` are computed from **one** source
    rather than deriving the numbers twice and drifting apart -- the same
    discipline ``ClipperPriorBox._limits_from_model`` exists for.

    Parameters
    ----------
    basis
        The name of the prior quantity the scale was read from (``"width"``,
        ``"sigma"``, ``"median x log-ratio"``, ``"fallback"``). Reported to the
        user, because *which* quantity was used is the part of this that is a
        judgement call.
    basis_value
        The value of that quantity.
    scale
        The un-normalised scale.
    """

    basis: str
    basis_value: float
    scale: float


class AbstractScaler(ABC):
    """
    A strategy supplying a per-parameter step scale for a search that steps in
    physical parameter space.

    Subclasses are pluggable per-search, exactly like
    :class:`~autofit.non_linear.clipper.AbstractClipper`, and the default is
    :class:`ScalerNone` so behaviour is unchanged until a user opts in.
    """

    @abstractmethod
    def scale_from_model(self, model) -> np.ndarray:
        """
        The ``(n_params,)`` scale vector, in **physical parameter order** (that
        is, ``model.priors_ordered_by_id`` -- the same order
        :meth:`~autofit.non_linear.clipper.AbstractClipper.bounds_from_model`
        returns, so the two compose without a reindex).

        Every entry is strictly positive and finite. That is a guarantee, not an
        expectation: the vector is used as a divisor, so a zero would send the
        whole population to infinity and a ``NaN`` would poison every lane.
        """

    def info_from_model(self, model) -> str:
        """
        A human-readable summary of the scale vector, appended to the written
        ``model.info`` file.

        The default is empty, which is what causes nothing to be appended for a
        search that does no scaling.
        """
        return ""


class ScalerNone(AbstractScaler):
    """
    The no-op scaler, and **the default**.

    The scale is all ones, so a search configured with this is bit-identical to
    one with no scaler concept at all. Searches should test for it and skip the
    change of variables entirely rather than multiplying by ones, so that the
    compiled step is unchanged too.
    """

    def scale_from_model(self, model) -> np.ndarray:
        return np.ones(model.prior_count, dtype=float)


class ScalerPriorWidth(AbstractScaler):
    """
    A static diagonal preconditioner derived from the priors.

    The rule, per prior, is **the physical extent of a unit-cube-sized step**:

    - ``UniformPrior(lo, hi)`` -> ``hi - lo``. ``dtheta/du`` is constant and is
      exactly the width.

    - ``LogUniformPrior(lo, hi)`` -> ``sqrt(lo * hi) * log(hi / lo)``, the same
      quantity evaluated at the prior median, since ``theta(u) = lo * (hi/lo)**u``
      gives ``dtheta/du = theta * log(hi / lo)``. The **physical width is not
      usable here** and the error is not marginal: for ``LogUniform(1e-4, 1e4)``
      the width is ``1e4`` against a true local scale of ``18.4``, a factor of
      543. A log-spaced coordinate's natural step is multiplicative, and this is
      the linear-space image of it.

    - ``GaussianPrior(mu, sigma)`` -> ``sigma``. The prior is unbounded, so no
      width exists.

    - ``TruncatedGaussianPrior(mu, sigma, lo, hi)`` -> ``sigma``, **not** the
      truncation width. The two differ by a lot in practice rather than in
      principle: ``ell_comps`` is ``sigma=0.3`` inside a ``[-1, 1]`` box, so the
      width overstates its scale six-fold.

    - ``LogGaussianPrior(mu, sigma)`` -> ``exp(mu) * sigma``, the Gaussian rule
      carried through the exponential at the median (``mu`` and ``sigma`` are in
      **natural log** space, so ``exp(mu)`` is the median and ``sigma`` its
      fractional spread).

    - anything else, or a non-finite or non-positive result -> ``1.0``, logged.

    A deliberate inconsistency, recorded rather than hidden
    ------------------------------------------------------

    The Uniform rule uses the full ``dtheta/du``; the Gaussian rules use
    ``sigma``, which is ``dtheta/du`` at the median divided by ``sqrt(2 * pi)``.
    So a Gaussian coordinate is scaled about **2.5x smaller** relative to a
    Uniform one than the strict unit-step principle alone would give. That is the
    intended behaviour: ``sigma`` is the quantity a user actually reasons about
    when they write a Gaussian prior, and matching the constant would inflate
    every Gaussian coordinate's step by a factor nobody asked for. It is written
    down here because an undocumented factor of 2.5 between two branches of one
    rule is exactly the kind of thing that later reads as a bug.

    Normalisation
    -------------

    The vector is normalised so its **geometric mean is exactly 1**. That leaves
    the *global* step magnitude untouched and alters only the *ratios*, so an A/B
    against an unscaled run measures rescaling alone and is not confounded with
    an effective learning-rate change. The geometric mean is the right centre for
    a set of quantities that are compared multiplicatively; the arithmetic mean
    would be dominated by the single widest prior.
    """

    def _terms_from_model(self, model) -> List[ScaleTerm]:
        """
        One :class:`ScaleTerm` per prior, in ``model.priors_ordered_by_id`` order,
        **before** normalisation.

        Priors are matched by type rather than by reading ``lower_limit`` /
        ``upper_limit`` off every one of them (which is how
        ``ClipperPriorBox._limits_from_model`` works). The two want different
        things: the clipper wants *the support*, which every prior exposes
        uniformly, whereas this wants *the scale*, which lives in a different
        attribute per prior family and for ``TruncatedGaussianPrior`` is
        specifically **not** the support.
        """
        terms = []

        for prior in model.priors_ordered_by_id:
            term = self._term_for(prior)

            if not np.isfinite(term.scale) or term.scale <= 0.0:
                logger.warning(
                    f"ScalerPriorWidth: {type(prior).__name__} produced a "
                    f"non-positive or non-finite scale ({term.scale}); falling "
                    f"back to 1.0 for this parameter. The step scale for it is "
                    f"therefore unchanged, not zero."
                )
                term = ScaleTerm(basis="fallback", basis_value=term.scale, scale=1.0)

            terms.append(term)

        return terms

    @staticmethod
    def _term_for(prior) -> ScaleTerm:
        """The scale for a single prior, before the fallback guard."""
        if isinstance(prior, UniformPrior):
            width = float(prior.upper_limit) - float(prior.lower_limit)
            return ScaleTerm(basis="width", basis_value=width, scale=width)

        if isinstance(prior, LogUniformPrior):
            lower = float(prior.lower_limit)
            upper = float(prior.upper_limit)
            # Guarded before the logs rather than after: `log` of a non-positive
            # limit is `nan`/`-inf` and would reach the fallback as a warning
            # about arithmetic rather than about the prior, plus a NumPy
            # RuntimeWarning on the way. A LogUniformPrior with a non-positive
            # limit is malformed, not merely awkward to scale.
            if lower <= 0.0 or upper <= 0.0:
                return ScaleTerm(basis="log-ratio", basis_value=np.nan, scale=np.nan)
            log_ratio = np.log(upper / lower)
            median = np.sqrt(lower * upper)
            return ScaleTerm(
                basis="median x log-ratio",
                basis_value=float(log_ratio),
                scale=float(median * log_ratio),
            )

        # TruncatedGaussianPrior is checked before GaussianPrior purely for
        # readability -- they are sibling subclasses of `Prior`, not parent and
        # child, so the order is not load-bearing today. It is written this way
        # so that it does not silently become load-bearing if that ever changes.
        if isinstance(prior, TruncatedGaussianPrior):
            sigma = float(prior.sigma)
            return ScaleTerm(basis="sigma", basis_value=sigma, scale=sigma)

        if isinstance(prior, GaussianPrior):
            sigma = float(prior.sigma)
            return ScaleTerm(basis="sigma", basis_value=sigma, scale=sigma)

        if isinstance(prior, LogGaussianPrior):
            sigma = float(prior.sigma)
            median = float(np.exp(prior.mean))
            return ScaleTerm(
                basis="median x sigma",
                basis_value=sigma,
                scale=median * sigma,
            )

        logger.warning(
            f"ScalerPriorWidth: no scale rule for {type(prior).__name__}; using "
            f"1.0 for this parameter."
        )
        return ScaleTerm(basis="fallback", basis_value=np.nan, scale=1.0)

    @staticmethod
    def _normalised(scales: np.ndarray) -> np.ndarray:
        """
        ``scales`` divided by their geometric mean, so the returned vector has a
        geometric mean of exactly 1.

        Computed in log space (``exp(mean(log(s)))``) rather than as
        ``prod(s) ** (1/n)``: the product of fifteen widths spanning 40x
        underflows or overflows for no reason, whereas the sum of their logs does
        not. Every entry is guaranteed positive and finite by
        :meth:`_terms_from_model` before this is called.
        """
        return scales / np.exp(np.mean(np.log(scales)))

    def scale_from_model(self, model) -> np.ndarray:
        scales = np.array(
            [term.scale for term in self._terms_from_model(model)], dtype=float
        )

        if scales.size == 0:
            return scales

        return self._normalised(scales)

    def terms_and_scale_from_model(self, model) -> Tuple[List[ScaleTerm], np.ndarray]:
        """
        The per-prior terms alongside the normalised scale vector, for callers
        that want to report *why* each scale is what it is (the ``model.info``
        block) without recomputing it.
        """
        terms = self._terms_from_model(model)
        scales = np.array([term.scale for term in terms], dtype=float)

        if scales.size == 0:
            return terms, scales

        return terms, self._normalised(scales)

    def info_from_model(self, model) -> str:
        """
        The ``model.info`` block: one line per parameter giving its prior type,
        the quantity the scale was read from, and the normalised scale.

        Formatted as aligned ``path  ...`` lines, matching how the initializer
        already reports start points into ``model.start``
        (:meth:`~autofit.non_linear.initializer.InitializerParamBounds.info_from_model`)
        rather than as a ``TextFormatter`` tree, because this is a flat
        per-parameter table and the tree adds nothing to it.
        """
        terms, scales = self.terms_and_scale_from_model(model)

        if len(terms) == 0:
            return ""

        rows = []
        for prior, term, scale in zip(model.priors_ordered_by_id, terms, scales):
            rows.append(
                (
                    ".".join(str(path) for path in model.path_for_prior(prior)),
                    type(prior).__name__,
                    term.basis,
                    f"{term.basis_value:.4e}",
                    f"{scale:.4e}",
                )
            )

        widths = [max(len(row[column]) for row in rows) for column in range(4)]

        info = f"Per-Parameter Step Scaling ({type(self).__name__})\n\n"
        for path, prior_name, basis, basis_value, scale in rows:
            info += (
                f"{path:<{widths[0]}}   {prior_name:<{widths[1]}}   "
                f"{basis:<{widths[2]}} {basis_value:>{widths[3]}}   "
                f"scale {scale}\n"
            )

        info += (
            f"\nScales are normalised so their geometric mean is 1.0, which "
            f"changes only the RATIOS between parameters and leaves the global "
            f"step magnitude unchanged.\n"
        )

        return info
