"""
Search-local enforcement of prior support.

The gradient searches step in **physical** parameter space against an objective
that folds in the prior::

    fom = -2 * (log_likelihood + sum(log_prior_list))

A hard-edged prior (``UniformPrior``, ``TruncatedGaussianPrior``, ...) is
``-inf`` outside its box, so a step that leaves that box makes the objective
non-finite. Nothing in the step itself pulls the parameter back: because
``log_prior = -inf`` is *constant* outside the box its derivative is zero, so the
total gradient is the finite **likelihood** gradient and ``optax.apply_if_finite``
never fires. The lane is marked dead but keeps stepping for the rest of the run,
paying full likelihood-and-gradient cost with its output discarded. On the
reference ``imaging/mge`` cell this accounted for 60.25% of lane-steps and 14 of
16 lane deaths, while the likelihood itself never went non-finite once (see
autolens_profiling#128).

The samplers that are *not* exposed are instructive: the nested samplers already
work in unit-cube coordinates, and the MCMC samplers reject ``-inf`` proposals so
the walker simply stays put. **Rejection is the restoring mechanism the gradient
methods lack**, and a ``Clipper`` supplies it.

(Historical correction, PyAutoFit#1489: the rejection sentence above was written
while it was FALSE on the NumPy path — between release 2025.10.16.1, which removed
``assert_within_limits`` / ``PriorLimitException``, and the strict
``log_prior_from_value`` bounds restored for #1489, a NumPy-path ``UniformPrior``
returned ``0.0`` outside its box, so there was no ``-inf`` for the MCMC samplers
to reject and walkers escaped the declared support. It was true before
2025.10.16.1, via exception-driven resampling, and is true again now via the
strict priors. The gradient-method reasoning below was unaffected.)

Two consumers, one source of truth
----------------------------------

Unlike :mod:`autofit.non_linear.initializer`, which this module is otherwise
modelled on, a ``Clipper`` has two structurally different consumers:

- ``MultiStartGradient`` enforces the constraint itself after every update, so it
  wants an imperative :meth:`AbstractClipper.project`;
- ``LBFGS`` hands box bounds to scipy and lets *scipy* enforce them, so it wants a
  declarative :meth:`AbstractClipper.bounds_from_model`.

Both read one private ``_limits_from_model`` so the declarative and imperative
views can never drift apart.

``project`` returns **which coordinates it clipped**, not just the new vector.
That mask is what lets a caller zero optimiser momentum along clipped directions:
projecting the parameters while the accumulated optimiser state keeps pushing
outward leaves lanes pinned to a bound. The ``Clipper`` cannot fix that itself
(it does not own ``opt_state``), so it exposes enough for the search to.

A soft-wall strategy would belong here too, as a ``Clipper`` — **never** as a
change to the ``Prior`` classes, which would silently alter the objective for the
nested samplers, where the hard box currently works correctly.

Bound kinds
-----------

The inset applied to keep a projected value strictly inside its support is keyed
on **what kind of bound it is**, never on unguarded ``upper - lower`` arithmetic.
That distinction is load-bearing rather than fussy — see
:class:`ClipperPriorBox` for the three cases and why collapsing them breaks.

Composition with a :mod:`~autofit.non_linear.bijector`
--------------------------------------------------------

``AbstractClipper.project`` also accepts a ``bijector``, alongside (never
together with) ``scale``, so a search stepping in ``phi = bijector.forward
(theta)`` can clip in that same space. Every bijector kind is elementwise
**strictly monotone increasing**, so forward and clip commute: clipping
``forward(theta)`` against ``forward(lower_inset)`` / ``forward(upper_inset)``
gives exactly ``forward(clip(theta, lower_inset, upper_inset))``.

**The physical inset is wrong for a ``log``-kind coordinate, and the error is
not marginal.** The two-sided inset above is a *relative-to-physical-width*
margin, ``lower + margin * (upper - lower)``. For
``LogUniformPrior(1e-6, 1e6)`` with the default ``margin=1e-6`` that is
``1e-6 + 1e-6 * (1e6 - 1e-6) ~ 1e-6 + 1.0 ~ 1.0`` — a PHYSICAL inset of order
1, on a prior whose entire point is that it is uniform across twelve decades.
Clipping (or mapping through a ``log`` bijector) against that bound fences off
every ``lambda < 1``, silently, for a margin whose entire purpose is to be
negligible. The bug is in the inset arithmetic, not in the bijector: a
``log``-kind coordinate's natural margin is a fraction of its **log-ratio**,
``margin * log(upper / lower)``, mapped back through ``exp`` — the same
"physical width is not usable here" correction
:class:`~autofit.non_linear.scaler.ScalerPriorWidth` already makes for its own
``LogUniformPrior`` scale rule. ``ClipperPriorBox`` applies this whenever a
``bijector`` reports a coordinate's kind as ``"log"`` (see
:meth:`ClipperPriorBox._inset_from_model`); the plain, no-bijector
``bounds_from_model`` path is unaffected — every existing caller keeps its
current (physical) inset exactly.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AbstractClipper(ABC):
    """
    A strategy for keeping a search's parameter vector inside the support of the
    model's priors.

    Subclasses are pluggable per-search, exactly like
    :class:`~autofit.non_linear.initializer.AbstractInitializer`, and the default
    is :class:`ClipperNone` so that behaviour is unchanged until a user opts in.
    """

    @abstractmethod
    def bounds_from_model(self, model) -> Tuple[np.ndarray, np.ndarray]:
        """
        The ``(lower, upper)`` box, in **physical** parameter order.

        Unbounded coordinates are ``-inf`` / ``+inf``. The two arrays are returned
        separately because that is the shape :meth:`project` broadcasts against.

        Note that this is **not** the shape ``scipy.optimize.minimize`` accepts:
        it reads a ``(lower, upper)`` tuple as a *sequence of ``(min, max)``
        pairs*, which for a two-parameter model silently produces a wrong fit
        rather than an error. Callers handing these to scipy must build an
        explicit ``scipy.optimize.Bounds`` — see
        :meth:`~autofit.non_linear.search.mle.bfgs.search.AbstractBFGS._bounds_from`.

        Parameters
        ----------
        model
            The model whose priors define the support.
        """

    @abstractmethod
    def project(self, vector, model, xp=np, scale=None, bijector=None):
        """
        Project ``vector`` onto the prior support.

        Parameters
        ----------
        vector
            A parameter vector, either a single ``(n_params,)`` vector or a
            batched ``(n_starts, n_params)`` array of them. Broadcasting handles
            both, so no ``vmap`` is required of the caller. Physical unless
            ``scale`` or ``bijector`` is given, in which case it is in that
            change of variables' coordinates.
        model
            The model whose priors define the support.
        xp
            The array module, ``numpy`` or ``jax.numpy``.
        scale
            The per-parameter step scale from a
            :mod:`~autofit.non_linear.scaler`, when the caller is stepping in
            ``phi = theta / scale`` rather than in physical parameters. The
            **bounds** are divided by it, so the projection happens in the
            caller's own coordinates and no round-trip through physical space is
            needed. Scales are strictly positive, so dividing preserves the
            ordering of each ``(lower, upper)`` pair and ``+/-inf`` stay
            ``+/-inf``. Mutually exclusive with ``bijector``.
        bijector
            A resolved :class:`~autofit.non_linear.bijector.AbstractBijector`
            (``.from_model`` already called), when the caller is stepping in
            ``phi = bijector.forward(theta)``. The inset bounds are computed in
            physical space (kind-aware — see the module docstring) and then
            mapped through ``bijector.bounds_forward``, which commutes with
            clipping because every bijector kind is monotone increasing.
            Mutually exclusive with ``scale``.

        Returns
        -------
        A ``(projected, clipped_mask)`` pair. ``clipped_mask`` is boolean, the same
        shape as ``vector``, and ``True`` exactly where a coordinate was moved.
        """


class ClipperNone(AbstractClipper):
    """
    The no-op clipper, and **the default**.

    Bounds are ``±inf`` and ``project`` is the identity, so a search configured
    with this is bit-identical to one with no clipper concept at all. Searches
    should test for it and skip the projection entirely rather than applying it as
    a no-op, so that the compiled step is unchanged too.
    """

    def bounds_from_model(self, model) -> Tuple[np.ndarray, np.ndarray]:
        n = model.prior_count
        return np.full(n, -np.inf), np.full(n, np.inf)

    def project(self, vector, model, xp=np, scale=None, bijector=None):
        return vector, xp.zeros_like(vector, dtype=bool)


class ClipperPriorBox(AbstractClipper):
    """
    Hard projection onto the prior support, inset to stay strictly inside it.

    The inset is keyed on the **kind** of each bound. Collapsing these three cases
    into one relative ``margin * (upper - lower)`` is the obvious implementation
    and it is wrong in two separate ways, both silent:

    - **Two-sided finite** (``UniformPrior``, ``LogUniformPrior``,
      ``TruncatedGaussianPrior``) — inset by ``margin`` *relative* to the box
      width. This is deliberately **not** for prior support: those priors are
      *inclusive* at the limit (``log_prior`` is finite exactly on the bound), so
      clipping onto the bound would already be valid. It is to avoid parking a
      lane exactly **on** a prior edge, where the model's own transforms are prone
      to singularities (``arctan2`` / ``sqrt`` at exactly 0) — the same reason
      ``MultiStartGradient`` draws its broad starts from the interior
      ``(0.15, 0.85)`` of the unit cube rather than from ``(0, 1)``.

    - **Unbounded** (``GaussianPrior``) — no inset, and critically **no width
      arithmetic at all**. ``lower + margin * (upper - lower)`` evaluates
      ``-inf + inf``, which is ``NaN``; clipping against ``NaN`` bounds turns the
      coordinate into ``NaN`` and the whole objective with it. That failure looks
      exactly like the lane death this class exists to prevent, so the guarantee
      that an unbounded coordinate passes through untouched holds only if the
      width is never computed for it.

    - **Half-open and exclusive** (``LogGaussianPrior``, whose support is
      ``(0, inf)``) — inset by an *absolute* ``strict_epsilon``. A relative margin
      is identically zero here, there being no finite width, so it would clip
      exactly onto ``0.0``, where the support is strict and ``log_prior`` is
      ``-inf``. Only an absolute nudge lands strictly inside. Which bounds are
      exclusive is read off the prior's ``lower_limit_strict`` /
      ``upper_limit_strict``, never inferred from its type.

    Parameters
    ----------
    margin
        The relative inset applied to two-sided finite bounds, as a fraction of
        the box width.
    strict_epsilon
        The absolute inset applied to half-open bounds whose support excludes the
        limit itself. Deliberately small: the floor it produces is a very
        low-density point, but it is *finite*, and the resulting steep prior
        gradient is what pushes the lane back into the bulk rather than leaving it
        dead.
    """

    def __init__(self, margin: float = 1.0e-6, strict_epsilon: float = 1.0e-12):
        self.margin = float(margin)
        self.strict_epsilon = float(strict_epsilon)

    def _limits_from_model(self, model):
        """
        The raw ``(lower, upper, lower_strict, upper_strict)`` arrays for ``model``,
        in physical parameter order.

        Limits are read off ``prior.lower_limit`` / ``prior.upper_limit``, and
        strictness off ``prior.lower_limit_strict`` / ``prior.upper_limit_strict``.
        Both resolve for **every** prior type without a type switch: ``UniformPrior``
        and ``LogUniformPrior`` shadow the limits with their own attributes,
        ``LogGaussianPrior`` declares its ``(0, inf)`` support and flags the lower
        bound strict, ``TruncatedGaussianPrior`` picks up real limits from
        ``TruncatedNormalMessage``, and ``GaussianPrior`` falls through
        ``Prior.__getattr__`` to its message's ``±inf``.

        This clipper used to declare ``LogGaussianPrior``'s support itself, because
        that prior reported ``(-inf, inf)`` while ``log_prior_from_value`` returned
        ``-inf`` for ``value <= 0``. That was a workaround for a defect in the prior,
        fixed in PyAutoFit#1526: any consumer of ``lower_limit`` — not just this one —
        was being told a strictly positive parameter could go negative.

        The ``strict`` flags mark bounds the support *excludes* (``value > limit``
        rather than ``value >= limit``); only those need the absolute inset.
        """
        lower, upper, lower_strict, upper_strict = [], [], [], []

        for prior in model.priors_ordered_by_id:
            low = float(prior.lower_limit)
            high = float(prior.upper_limit)
            low_strict = bool(prior.lower_limit_strict)
            high_strict = bool(prior.upper_limit_strict)

            lower.append(low)
            upper.append(high)
            lower_strict.append(low_strict)
            upper_strict.append(high_strict)

        return (
            np.array(lower, dtype=float),
            np.array(upper, dtype=float),
            np.array(lower_strict, dtype=bool),
            np.array(upper_strict, dtype=bool),
        )

    def _inset_from_model(
        self, model, kinds: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The inset ``(lower, upper)`` box, in physical parameter order, aware of
        an optional per-coordinate ``kinds`` list (from
        :attr:`~autofit.non_linear.bijector.AbstractBijector.kinds`).

        ``kinds=None`` (the default, and what :meth:`bounds_from_model` passes)
        reproduces the original physical-relative-margin rule for every
        two-sided finite coordinate, unchanged. Where ``kinds[i] == "log"`` the
        margin is instead taken as a fraction of the LOG-ratio and mapped back
        through ``exp`` — see the module docstring for why the physical margin
        is wrong there (it is not a small correction: it can be O(1) against a
        prior spanning many decades).
        """
        lower, upper, lower_strict, upper_strict = self._limits_from_model(model)

        two_sided = np.isfinite(lower) & np.isfinite(upper)

        is_log = np.zeros(len(lower), dtype=bool)
        if kinds is not None:
            is_log = np.array([kind == "log" for kind in kinds], dtype=bool)
            # A "log" kind is only ever assigned (by the bijector classes) to a
            # coordinate with a strictly positive, two-sided-finite physical
            # bound; this additional guard is defence-in-depth so a stray or
            # malformed `kinds` entry can never reach `np.log` of a
            # non-positive or non-finite value below.
            is_log = is_log & two_sided & (lower > 0.0)

        # PHYSICAL (linear-space) margin, for every two-sided finite coordinate
        # that is NOT log-kind -- identical to the original rule. The width is
        # evaluated ONLY where both bounds are finite and the coordinate is a
        # linear target. `np.where` alone would not be enough -- it evaluates
        # both branches, so `inf - -inf` would still be computed and still be
        # NaN. The finite substitution has to happen before the subtraction,
        # not after it.
        linear_target = two_sided & ~is_log
        safe_lower = np.where(linear_target, lower, 0.0)
        safe_upper = np.where(linear_target, upper, 0.0)
        relative = self.margin * (safe_upper - safe_lower)

        lower_inset = lower + np.where(linear_target, relative, 0.0)
        upper_inset = upper - np.where(linear_target, relative, 0.0)

        # LOG-space margin, for every log-kind coordinate: a fraction of the
        # log-ratio, mapped back through `exp`. `safe_log_lower` /
        # `safe_log_upper` substitute `1.0` for every non-log-kind coordinate
        # so `np.log` is never evaluated outside its domain, the same
        # before-the-call substitution `bounds_from_model` above already uses.
        safe_log_lower = np.where(is_log, lower, 1.0)
        safe_log_upper = np.where(is_log, upper, 1.0)
        log_relative = self.margin * np.log(safe_log_upper / safe_log_lower)

        log_lower_inset = np.log(safe_log_lower) + log_relative
        log_upper_inset = np.log(safe_log_upper) - log_relative

        lower_inset = np.where(is_log, np.exp(log_lower_inset), lower_inset)
        upper_inset = np.where(is_log, np.exp(log_upper_inset), upper_inset)

        # Half-open bounds get an absolute nudge, since `relative` is identically
        # zero for them.
        lower_inset = lower_inset + np.where(
            lower_strict & ~two_sided, self.strict_epsilon, 0.0
        )
        upper_inset = upper_inset - np.where(
            upper_strict & ~two_sided, self.strict_epsilon, 0.0
        )

        return lower_inset, upper_inset

    def bounds_from_model(self, model) -> Tuple[np.ndarray, np.ndarray]:
        return self._inset_from_model(model, kinds=None)

    def project(self, vector, model, xp=np, scale=None, bijector=None):
        if scale is not None and bijector is not None:
            raise ValueError(
                f"{type(self).__name__}.project received both `scale` and "
                "`bijector` -- they are two different changes of variables for "
                "the same step, and a caller must pick exactly one (a search "
                "already enforces this at construction; see "
                "AbstractMultiStartGradient.__init__)."
            )

        if bijector is not None:
            # Inset in PHYSICAL space, kind-aware, then mapped through the
            # bijector -- never the other way round. Every kind is monotone
            # increasing, so this commutes with clipping (see the module
            # docstring), and computing the inset in physical space is what
            # lets the log-kind correction above be expressed in terms even a
            # non-bijector-aware caller (`bounds_from_model`) already
            # understands.
            lower, upper = self._inset_from_model(model, kinds=bijector.kinds)
            lower, upper = bijector.bounds_forward(lower, upper)
        else:
            lower, upper = self.bounds_from_model(model)

            if scale is not None:
                # Divided AFTER the insets are applied, not before. The inset is
                # relative to the box width, so scaling the raw limits first and
                # insetting afterwards gives the identical box -- but the
                # half-open `strict_epsilon` is ABSOLUTE, and dividing it by the
                # scale would shrink the nudge for a large-scale coordinate
                # until it no longer lands strictly inside a support that
                # excludes its limit. Insetting in physical space and then
                # mapping the finished bounds keeps the inset meaning what it
                # says in the space the prior is declared in.
                scale = np.asarray(scale, dtype=float)
                lower = lower / scale
                upper = upper / scale

        dtype = getattr(vector, "dtype", None)
        lower = xp.asarray(lower, dtype=dtype)
        upper = xp.asarray(upper, dtype=dtype)

        projected = xp.clip(vector, lower, upper)

        return projected, projected != vector
