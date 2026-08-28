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

That split is also what bounds :class:`ClipperPriorBoxJoint`, the box-plus-ball
subclass: a ball has an imperative projection but no declarative ``Bounds``, so
it is available to the first consumer and explicitly refused by the second (see
:meth:`~autofit.non_linear.search.mle.bfgs.search.AbstractBFGS._bounds_from`)
rather than quietly degraded to its box.

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

A **ball** composes with the same maps under a narrower condition, resolved per
pair rather than per map: it survives a common linear rescale of both its
members and nothing else. See :class:`ClipperPriorBoxJoint` ("Composition with
a scaler/bijector") and :meth:`ClipperPriorBoxJoint.pairs_in_stepped_coordinates`.
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

    # Pinned explicitly rather than left to the identifier's fallback, which
    # infers the fields from `__init__`'s argspec and therefore produces the
    # identical `{margin, strict_epsilon}` today. The pin is what keeps it
    # identical: a subclass that takes a further constructor argument would
    # otherwise silently re-key every stored `ClipperPriorBox` result whose
    # search shares the identifier machinery. Any subclass adding a parameter
    # that changes where a lane can sit must extend this tuple deliberately.
    __identifier_fields__ = ("margin", "strict_epsilon")

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


class ClipperPriorBoxJoint(ClipperPriorBox):
    """
    The prior box, and then the class-declared **balls** inside it.

    A box prior on each of a pair of coordinates is a square, and where the model
    means a disk the corners of that square are not merely unlikely, they are
    **non-physical**. The canonical case is `ell_comps`: two independent
    ``[-1, 1]`` priors whose valid region is ``e0**2 + e1**2 < 1``, so ``1 - pi/4``
    — 21.5% — of the declared prior area describes no ellipse at all. Nothing in
    a box clipper can see that, because no per-coordinate bound can: ``(0.8, 0.8)``
    is inside both boxes and outside the disk.

    That region is not hypothetical. 20.1% of MultiStart lane best points across
    the recorded `autolens_profiling` campaign end at ``|e| >= 1``, and 0 of the
    246 lanes that reach the target basin do — ending outside the disk is a
    property of *failed* lanes, so the 20% is wasted budget rather than a wasted
    answer (autolens_profiling#182). The reason the lanes are never pulled back
    is the same reason :class:`ClipperPriorBox` exists at all: the conversion to
    an axis ratio *saturates* past the clamp, so the objective is flat out there
    and the gradient has nothing to say.

    Which is why this clipper **projects** rather than penalises: the projection
    does not need a gradient to exist, and it moves the lane in one step from a
    region where no gradient exists to the nearest point where one does.

    Where the geometry comes from
    -----------------------------

    Nothing here knows what ``ell_comps`` is. The radius and the coordinate pair
    are read off the model, which reads them off the declaring class's
    ``__model_ball_constraints__`` (see
    :mod:`autofit.mapper.prior_model.constraint`) — so PyAutoGalaxy states its own
    geometry and PyAutoFit projects onto whatever geometry it is told about.

    The order of the two projections
    --------------------------------

    The box is applied first and the balls second, never the reverse. The ball is
    a *subset* of the box for any radius the box contains, so projecting onto the
    ball last is what leaves the point inside both; projecting onto the box last
    could push a point that was on the ball's surface back off it. It is not a
    true joint projection onto the intersection — that would require iterating —
    but for the case this exists for (a ball inscribed in, or well inside, its
    box) the two-step projection lands in the intersection in one pass.

    Jittability
    -----------

    The radial shrink is written to survive ``jit``, ``vmap`` and ``grad``:

    - the factor is built with ``xp.where`` rather than a Python branch, so the
      pair's radius is never a concrete condition;
    - the radius is compared **squared**, so the ``sqrt`` is only ever needed on
      the branch that is actually taken;
    - the ``sqrt`` argument is itself passed through a ``where`` that substitutes
      ``1.0`` on the untaken branch — the "double where" idiom. A lane sitting
      exactly at the origin (where a *spherical* profile's linked components sit,
      and where many priors start) would otherwise evaluate ``sqrt(0)``, whose
      derivative is infinite; ``jax.grad`` computes **both** branches of a
      ``where`` and multiplies the untaken one by zero, and ``0 * inf`` is
      ``NaN``. The forward value is unaffected either way; only the gradient is,
      and only silently. Substituting before the ``sqrt`` is the only thing that
      keeps it finite;
    - the divisor is additionally floored at ``tiny``, so a degenerate
      ``radius=0`` declaration cannot divide by zero;
    - the factors are assembled as a full multiplicative vector and applied with
      a single multiply, rather than with ``.at[...].set(...)`` scatter updates,
      so the traced program is a fixed handful of ops independent of how many
      balls the model declares and works unchanged under ``numpy``.

    ``clipped_mask`` is set on **both** members of a projected pair, even though
    a shrink of a factor ``1.0 - 1e-16`` moves one of them imperceptibly. The
    mask's consumer is ``MultiStartGradient``'s momentum reset: a lane pushed out
    of the disk carries outward momentum in *both* coordinates, and zeroing only
    one of them leaves the pair spiralling back out on the next step.

    Composition with a scaler/bijector
    ---------------------------------

    A ball is a statement about **physical** coordinates, so a search stepping
    in some other coordinates can only be given a ball projection if the ball
    is still a ball there. That is a question about the *pair*, not about the
    map as a whole, and it has a clean answer:

    - both members ``identity``-kind under a **common** linear scale ``s``: the
      step coordinates are ``(theta_i / s, theta_j / s)``, so
      ``theta_i**2 + theta_j**2 <= R**2`` is exactly
      ``phi_i**2 + phi_j**2 <= (R / s)**2``. A disk of radius ``R`` in physical
      coordinates **is** a disk of radius ``R / s`` in the stepped ones, and
      the radial shrink below — which is scale-covariant, being a pure
      multiply — projects onto it correctly. The pair is accepted, with its
      radius divided;
    - anything else — ``log``, ``logit``, or two ``identity`` members with
      *different* scales — is not a disk. Unequal scales give an **ellipse**,
      whose nearest-point projection is not a radial shrink at all (it is the
      root of a quartic), and the non-linear kinds give a region with no closed
      form. Projecting a circle onto those coordinates would enforce a
      different, silently wrong constraint, so :meth:`project` raises instead,
      naming the offending index pair and the kinds it found.

    This is deliberately resolved **per pair** rather than per map. A model
    reparameterised through ``log`` on, say, an ``einstein_radius`` while its
    ``ell_comps`` stay linear is the common case, and refusing it wholesale is
    what left those lanes at the box corner ``|e| = 1.414`` — inside every prior
    box, outside the disk, and in the flat region the clipper exists to escape.

    The projection is **not** round-tripped through the bijector (map back to
    physical, project, map forward). That would be correct for every kind, but
    it would also drag the ``logit`` epsilon clamps across coordinates the ball
    has nothing to do with, breaking the bit-identity the interior-point path
    promises, saturating gradients at the clamped edges, and adding traced ops
    to every step. Dividing the radius costs nothing and is exact.
    """

    # Divisor floor for the radial shrink. Small enough to be irrelevant for any
    # coordinate the projection actually moves (a pair is only shrunk when
    # `r > radius`, and every declared radius is orders of magnitude larger than
    # this), and large enough that `r / tiny` cannot overflow float32.
    _TINY = 1.0e-30

    def pairs_in_stepped_coordinates(self, pairs, scale=None, bijector=None):
        """
        The declared ball pairs, restated in whatever coordinates the search is
        actually stepping in — or a ``ValueError`` naming the pair that cannot
        be restated.

        Parameters
        ----------
        pairs
            ``(index_0, index_1, radius)`` triples as
            :meth:`~autofit.mapper.prior_model.abstract.AbstractPriorModel.
            ball_constraint_index_pairs` returns them, in **physical**
            coordinates.
        scale
            The per-parameter linear scale a ``scaler`` is applying, or
            ``None``.
        bijector
            A **resolved** bijector (``from_model`` already called) the search
            is stepping through, or ``None``. Mutually exclusive with ``scale``.

        Returns
        -------
        The same triples with each ``radius`` divided by that pair's common
        linear scale — unchanged when neither a scaler nor a bijector is in
        play, since the common scale is then ``1.0``.

        Raises
        ------
        ValueError
            When a pair is not a disk in the stepped coordinates. See the class
            docstring's "Composition with a scaler/bijector" for which pairs
            those are and why the projection cannot be defined for them.

        This is a **model-resolution-time** check, separable from
        :meth:`project` on purpose: ``MultiStartGradient`` calls it once, as
        soon as it has resolved its bijector against the model and before it
        compiles a step, so a combination that cannot work dies there rather
        than a minute into a multi-hour fit.
        """
        if scale is None and bijector is None:
            return list(pairs)

        if scale is not None:
            # A raw scaler is linear by construction, so every coordinate is
            # `identity`-kind; only the equality of the two scales is in doubt.
            scales = [float(value) for value in np.asarray(scale, dtype=float)]
            kinds = ["identity"] * len(scales)
        else:
            scales = bijector.identity_scales
            kinds = bijector.kinds

        resolved = []

        for index_0, index_1, radius in pairs:
            scale_0 = scales[index_0]
            scale_1 = scales[index_1]

            common = (
                scale_0 is not None
                and scale_1 is not None
                and scale_0 == scale_1
                and scale_0 > 0.0
                and np.isfinite(scale_0)
            )

            if not common:
                raise ValueError(
                    f"{type(self).__name__} cannot project onto the ball "
                    f"declared on parameter indices ({index_0}, {index_1}) "
                    "while the search steps in scaled or transformed "
                    f"coordinates: that pair maps as ({kinds[index_0]}, "
                    f"{kinds[index_1]}) with linear scales ({scale_0}, "
                    f"{scale_1}), which is not a disk. A ball survives a "
                    "COMMON linear scale `s` (radius `R` becomes `R / s`) and "
                    "is composed with; unequal scales give an ellipse and a "
                    "log/logit coordinate a region with no closed form, and "
                    "projecting a circle onto either would enforce a "
                    "different, silently wrong constraint. Leave this pair "
                    "identity-mapped under one common scale (e.g. a "
                    "`BijectorPerPath` carrying the non-linear kinds on the "
                    "OTHER paths), or use ClipperPriorBox instead."
                )

            resolved.append((index_0, index_1, radius / scale_0))

        return resolved

    def project(self, vector, model, xp=np, scale=None, bijector=None):
        projected, clipped_mask = super().project(
            vector=vector,
            model=model,
            xp=xp,
            scale=scale,
            bijector=bijector,
        )

        pairs = model.ball_constraint_index_pairs()

        if not pairs:
            return projected, clipped_mask

        # Restated in the coordinates `projected` is already in (the box above
        # returns the STEPPED vector on both the scaler and the bijector path),
        # so the radial shrink below is unchanged -- it just shrinks onto a
        # radius that has been divided by the pair's common scale.
        pairs = self.pairs_in_stepped_coordinates(
            pairs=pairs,
            scale=scale,
            bijector=bijector,
        )

        # One multiplicative factor per parameter, defaulting to 1.0, so the
        # whole projection is a single elementwise multiply and every
        # non-ball coordinate passes through exactly (`x * 1.0 == x`).
        factor = xp.ones_like(projected)
        shrunk = xp.zeros_like(projected, dtype=bool)

        for index_0, index_1, radius in pairs:
            value_0 = projected[..., index_0]
            value_1 = projected[..., index_1]

            r_squared = value_0**2 + value_1**2
            outside = r_squared > radius**2

            # See the class docstring: the substitution has to happen BEFORE the
            # `sqrt`, not after it, or `grad` differentiates `sqrt` at zero on
            # the branch it then discards and the NaN survives the discard.
            r = xp.sqrt(xp.where(outside, r_squared, xp.ones_like(r_squared)))

            pair_factor = xp.where(
                outside,
                radius / xp.maximum(r, self._TINY),
                xp.ones_like(r),
            )

            # Built by broadcasting a one-hot selector rather than by scattering
            # into `factor`, so the same code runs under numpy and under a JAX
            # trace (where arrays are immutable) with no `.at` and no
            # index-dependent control flow.
            for index in (index_0, index_1):
                selector = xp.asarray(
                    np.arange(projected.shape[-1]) == index,
                )
                factor = xp.where(selector, pair_factor[..., None], factor)
                shrunk = shrunk | selector

        projected_ball = projected * factor

        # `shrunk` marks the coordinates a ball COULD move; `factor != 1.0` marks
        # the lanes it actually did. Both members of a moved pair are marked --
        # see the class docstring on the momentum reset.
        moved = shrunk & (factor != 1.0)

        return projected_ball, clipped_mask | moved
