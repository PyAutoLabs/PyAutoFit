"""
Per-parameter change of variables for the gradient searches.

Extends :mod:`autofit.non_linear.scaler` from a linear diagonal preconditioner
(``phi = theta / s``) to a per-coordinate **bijection**: an ``AbstractBijector``
may map a coordinate through ``log`` (multiplicative steps for a log-spread
parameter) or a bounded ``logit`` (a coordinate that is genuinely walled on both
sides), while every other coordinate stays the linear ``identity`` map --
optionally diagonally scaled, which is how a plain :class:`~autofit.non_linear.
scaler.AbstractScaler` is expressed here too (:class:`BijectorDiagonal`).

Read :mod:`autofit.non_linear.scaler` first. Everything below assumes its
"why not the unit cube" objections (a logit reparameterisation sends a
boundary optimum to infinity; an inverse-CDF transform has a face
singularity) and its invariance argument (a constant-Jacobian change of
variables cannot move the MAP) as background, and extends both.

Why per-coordinate bijections are safe where the unit cube was rejected
-------------------------------------------------------------------------

The scaler's objections to the unit cube are about *every coordinate at
once*, unconditionally. This module does not revisit that decision -- it
answers a narrower question: is there any coordinate for which a *specific*,
individually-chosen non-linear map is a strict improvement, applied only
there?

- **log**: only ever offered for :class:`~autofit.mapper.prior.log_uniform.
  LogUniformPrior` (positive support) and :class:`~autofit.mapper.prior.
  log_gaussian.LogGaussianPrior` (support ``(0, inf)``, already a lognormal --
  ``log`` recovers exactly the underlying Gaussian variable). Both have a
  *positive, open* domain with no face to send to infinity in the direction
  the search actually steps: the lower "wall" is the point at negative
  infinity in ``phi``, so a step that approaches it is receding, not
  accelerating toward a singularity. This is precisely the one case the
  scaler's log-ratio scale rule already targets (``ScalerPriorWidth``'s
  ``LogUniformPrior`` branch) -- the same coordinate, one step further: not
  just scaled multiplicatively, but *stepped* multiplicatively.

- **logit**: the scaler's specific objection stands and is not undone here --
  a genuinely boundary-pinned optimum is sent to infinity in ``phi``, exactly
  as it warned. ``BijectorLogit`` is offered as a **secondary, opt-in** arm
  for a model known *not* to have boundary optima (a mixed sweep of
  ``BijectorAuto`` vs ``BijectorLogit`` is how that would be established), not
  a default.

- **identity**: unconditionally safe -- it is the map the whole rest of the
  library already uses.

The equivalence argument (why no Jacobian for MAP)
----------------------------------------------------

Let ``g = inverse`` be one coordinate's bijection, a strictly monotone map of
that coordinate's prior support onto ``phi``-space (every kind here is
monotone increasing). The search minimises

    F(phi) = fom(g(phi))

where ``fom`` is the physical-space objective (``-2 * log_posterior``). Because
``g`` is a bijection, composing with it **relabels points without changing the
objective's value set** -- the same value of ``fom`` is attained at
``theta = g(phi)`` as at ``phi`` under ``F``. So

    argmin F = g^-1(argmin fom)

exactly, and **no Jacobian is added** to ``F``. Adding one would move the
answer: the pushforward density's mode is *not*, in general,
``g^-1(mode of theta)`` -- for ``log`` specifically it shifts by exactly the
``-log(lambda)`` term that ``LogUniformPrior.log_prior_from_value`` already
carries as its own (dropped) normalisation, i.e. folding a Jacobian into ``F``
here would silently double up a term the prior density already accounts for
under change of variables. This is the same "compose the objective through the
map, never re-express the density" discipline
:mod:`autofit.non_linear.scaler` uses, generalised from a constant Jacobian
(which cannot move the MAP either way) to a non-constant one (which can, and
must therefore never be added for a MAP search).

Why :meth:`AbstractBijector.log_det_jacobian` exists at all, then
--------------------------------------------------------------------

A *sampler* -- as opposed to a MAP optimizer -- is a different consumer with a
different contract: it wants **samples of the physical posterior**, not its
argmax. Drawing (or stepping) in ``phi`` and pushing the resulting samples
through ``g`` changes their density unless the Jacobian is folded in
(``p_theta(theta) = p_phi(phi) * |d phi/d theta|``, the ordinary change-of-
variables rule). ``log_det_jacobian`` is exposed for exactly that consumer --
the multi-chain gradient sampler tracked as PyAutoFit#1521/#1522 -- and is
**never called by MultiStart**, which is a MAP search and must not use it (see
above). Its absence from the ``MultiStartGradient`` wiring in
:mod:`autofit.non_linear.search.mle.multi_start_gradient.search` is therefore
not an oversight to "complete" later; it is the correctness condition.

Vectorisation and the double-where pattern
---------------------------------------------

Every method is vectorised over ``(n_starts, n_params)`` in
``model.priors_ordered_by_id`` order, matching
:meth:`~autofit.non_linear.scaler.AbstractScaler.scale_from_model` and
:meth:`~autofit.non_linear.clipper.AbstractClipper.bounds_from_model` so the
three compose without a reindex. The per-coordinate *kind* (identity / log /
logit) is resolved once, in :meth:`AbstractBijector.from_model`, into a
**static** ``numpy`` integer array baked onto the instance -- never re-derived
inside ``forward`` / ``inverse`` / ``log_det_jacobian`` themselves, so that
under ``jax.jit`` those methods trace to a single fixed program (one
``jnp.where`` selection tree per call) rather than branching in Python on
every trace.

Selecting a branch with ``jnp.where`` still *evaluates every branch* -- that is
how ``jnp.where`` computes its gradient. A coordinate outside a given branch's
domain (e.g. ``log`` of a coordinate that is not log-kind, which may be zero or
negative) would then feed an invalid value into ``log`` / division, and even
though the outer ``where`` discards the *result*, ``0 * NaN = NaN`` in IEEE 754
means the discarded branch's ``NaN`` can still poison the gradient through the
multiply-by-zero-mask that ``jnp.where``'s VJP performs. The fix used
throughout this module is the same "double where" safe-surrogate pattern
``LogUniformPrior.log_prior_from_value`` uses
(``autofit/mapper/prior/log_uniform.py``, ``in_bounds`` branch): substitute an
arbitrary in-domain value (e.g. ``1.0`` before a ``log``, ``0.0`` before an
``exp``) *inside* the risky call, via an inner ``where``, before the outer
``where`` picks the real branch. The discarded branch then evaluates to a
finite, safe number rather than ``NaN``/``-inf``, and the multiply-by-zero-mask
gradient trick is safe.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from autofit.mapper.prior.log_gaussian import LogGaussianPrior
from autofit.mapper.prior.log_uniform import LogUniformPrior
from autofit.non_linear.scaler import AbstractScaler

logger = logging.getLogger(__name__)

# Static per-coordinate kind codes. Plain ints (not an Enum) because these are
# baked into `numpy` arrays that get compared with `==` inside `jnp.where`
# chains -- an Enum would work too, but would force every comparison through
# `.value` for no benefit.
_IDENTITY = 0
_LOG = 1
_LOGIT = 2

_KIND_NAMES = {_IDENTITY: "identity", _LOG: "log", _LOGIT: "logit"}
_KIND_CODES = {name: code for code, name in _KIND_NAMES.items()}


def _stable_sigmoid(x, xp):
    """
    ``1 / (1 + exp(-x))``, computed so neither branch ever overflows ``exp``.

    For ``x >= 0`` uses ``1 / (1 + exp(-x))`` (``-x <= 0``, so ``exp`` cannot
    overflow); for ``x < 0`` uses the algebraically identical
    ``exp(x) / (1 + exp(x))`` (``x < 0``, so ``exp`` cannot overflow there
    either). Finite and safe for every finite ``x``, so no additional
    double-where guard is needed around it.
    """
    positive = x >= 0
    z = xp.exp(-xp.abs(x))
    return xp.where(positive, 1.0 / (1.0 + z), z / (1.0 + z))


def _softplus(x, xp):
    """``log(1 + exp(x))``, computed in the standard overflow-safe form."""
    return xp.maximum(x, 0.0) + xp.log1p(xp.exp(-xp.abs(x)))


class AbstractBijector(ABC):
    """
    A strategy supplying a per-parameter change of variables for a search that
    steps in physical parameter space.

    Subclasses are pluggable per-search, exactly like
    :class:`~autofit.non_linear.scaler.AbstractScaler` and
    :class:`~autofit.non_linear.clipper.AbstractClipper`, and the default is
    :class:`BijectorNone` so behaviour is unchanged until a user opts in.

    Unlike ``AbstractScaler`` (whose ``scale_from_model`` returns a fresh array
    every call and holds no state), a bijector's per-coordinate *kind* cannot
    be an argument of ``forward`` / ``inverse`` / ``log_det_jacobian`` without
    breaking ``jax.jit`` tracing (see the module docstring), so it is resolved
    **once**, against one model, and cached on the instance by
    :meth:`from_model`. Construct one bijector per search/model, the same
    discipline ``AbstractScaler`` follows in practice even though its own
    statelessness does not strictly require it.
    """

    def __init__(self):
        self._kind_code: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._lo: Optional[np.ndarray] = None
        self._hi: Optional[np.ndarray] = None
        self._logit_eps = 1.0e-12

    @abstractmethod
    def _kind_scale_bounds_from_model(
        self, model
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """
        Subclass hook: the per-coordinate ``(kind_code, scale, lower, upper)``
        in ``model.priors_ordered_by_id`` order.

        ``scale`` is used only by ``identity``-kind coordinates (``phi =
        theta / scale``); ``lower`` / ``upper`` only by ``logit``-kind ones
        (the box the logit is taken over). Both are still returned for every
        coordinate -- unused entries are never read, but ``from_model`` builds
        one dense array per field rather than a ragged one keyed by kind.
        """

    def from_model(self, model) -> "AbstractBijector":
        """
        Resolve and cache this bijector's per-coordinate kind/scale/bound
        arrays against ``model``, and return ``self``.

        Must be called once, before ``forward`` / ``inverse`` /
        ``log_det_jacobian`` / ``kinds``, and again for a different model --
        it overwrites the previous cache rather than accumulating state, the
        same "one strategy object per search" discipline
        :class:`~autofit.non_linear.scaler.AbstractScaler` follows.
        """
        kind_codes, scale, lower, upper = self._kind_scale_bounds_from_model(model)
        self._kind_code = np.asarray(kind_codes, dtype=np.int8)
        self._scale = np.asarray(scale, dtype=float)
        self._lo = np.asarray(lower, dtype=float)
        self._hi = np.asarray(upper, dtype=float)
        return self

    def _check_resolved(self):
        if self._kind_code is None:
            raise RuntimeError(
                f"{type(self).__name__}.from_model(model) must be called before "
                "forward / inverse / log_det_jacobian / bounds_forward / kinds "
                "-- the per-coordinate kind arrays are not yet resolved."
            )

    @property
    def kinds(self) -> List[str]:
        """The per-coordinate kind, in ``model.priors_ordered_by_id`` order."""
        self._check_resolved()
        return [_KIND_NAMES[int(code)] for code in self._kind_code]

    def forward(self, theta, xp=np):
        """
        The physical-to-transformed map, ``theta -> phi``, vectorised over the
        last axis (``(n_params,)`` or ``(n_starts, n_params)``).
        """
        self._check_resolved()

        is_id = self._kind_code == _IDENTITY
        is_log = self._kind_code == _LOG
        is_logit = self._kind_code == _LOGIT

        scale = self._scale
        lo = self._lo
        hi = self._hi

        phi_id = theta / scale

        # Safe surrogate `1.0` inside `log` for every non-log coordinate (which
        # may be zero, negative, or anything else); an extra floor guards a
        # genuinely log-kind coordinate that has drifted to exactly zero.
        theta_safe = xp.where(is_log, theta, 1.0)
        theta_safe = xp.maximum(theta_safe, 1.0e-300)
        phi_log = xp.log(theta_safe)

        # Safe surrogate `0.5` for `u` on every non-logit coordinate, clamped
        # away from the exact 0/1 edges where `logit` diverges.
        denom = xp.where(is_logit, hi - lo, 1.0)
        u = xp.where(is_logit, (theta - lo) / denom, 0.5)
        u = xp.clip(u, self._logit_eps, 1.0 - self._logit_eps)
        phi_logit = xp.log(u) - xp.log(1.0 - u)

        return xp.where(is_id, phi_id, xp.where(is_log, phi_log, phi_logit))

    def inverse(self, phi, xp=np):
        """
        The transformed-to-physical map, ``phi -> theta`` -- the search's
        ``g`` (see the module docstring's equivalence argument). Vectorised the
        same way as :meth:`forward`.
        """
        self._check_resolved()

        is_id = self._kind_code == _IDENTITY
        is_log = self._kind_code == _LOG
        is_logit = self._kind_code == _LOGIT

        scale = self._scale
        lo = self._lo
        hi = self._hi

        theta_id = phi * scale

        # Safe surrogate `0.0` inside `exp` for every non-log coordinate: `phi`
        # there is in an unrelated space (identity/logit) and may be large.
        phi_log_safe = xp.where(is_log, phi, 0.0)
        theta_log = xp.exp(phi_log_safe)

        phi_logit_safe = xp.where(is_logit, phi, 0.0)
        sigmoid = _stable_sigmoid(phi_logit_safe, xp)
        theta_logit = lo + (hi - lo) * sigmoid

        return xp.where(is_id, theta_id, xp.where(is_log, theta_log, theta_logit))

    def log_det_jacobian(self, phi, xp=np):
        """
        ``log |d theta / d phi|``, summed over parameters, per row -- shape
        ``()`` for a single ``(n_params,)`` vector or ``(n_starts,)`` for a
        batch. **Never used to form a MAP objective** (see the module
        docstring); this is for a sampler that must push ``phi``-space samples
        through :meth:`inverse` into physical-space ones via the ordinary
        change-of-variables rule.

        Closed forms, per kind (``theta = inverse(phi)``):

        - identity (scale ``s``): ``theta = s * phi``, so
          ``d theta / d phi = s`` and the contribution is ``log(s)``.
        - log: ``theta = exp(phi)``, so ``d theta / d phi = exp(phi) = theta``
          and the contribution is ``log(theta) = phi`` exactly.
        - logit (box ``[lo, hi]``): ``theta = lo + (hi - lo) * sigmoid(phi)``,
          so ``d theta / d phi = (hi - lo) * sigmoid(phi) * (1 - sigmoid(phi))``
          and the contribution is
          ``log(hi - lo) - softplus(-phi) - softplus(phi)`` (the standard
          stable log-sigmoid / log-one-minus-sigmoid pair).
        """
        self._check_resolved()

        is_id = self._kind_code == _IDENTITY
        is_log = self._kind_code == _LOG
        is_logit = self._kind_code == _LOGIT

        scale = self._scale
        width = self._hi - self._lo

        contrib_id = xp.log(scale)
        contrib_log = phi
        contrib_logit = (
            xp.log(xp.where(is_logit, width, 1.0))
            - _softplus(-phi, xp)
            - _softplus(phi, xp)
        )

        per_coord = xp.where(
            is_id, contrib_id, xp.where(is_log, contrib_log, contrib_logit)
        )
        return xp.sum(per_coord, axis=-1)

    def bounds_forward(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map a physical ``(lower, upper)`` box into transformed coordinates,
        elementwise, for :class:`~autofit.non_linear.clipper.AbstractClipper`
        to clip against.

        Every kind here is strictly monotone **increasing**, so this commutes
        with clipping: ``clip(forward(theta), forward(lo), forward(hi)) ==
        forward(clip(theta, lo, hi))``. Pure ``numpy`` -- the clipper's bounds
        are a constant vector, never a traced value.
        """
        self._check_resolved()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        return self.forward(lower, xp=np), self.forward(upper, xp=np)

    def info_from_model(self, model) -> str:
        """
        A human-readable summary of the resolved kinds, appended to the
        written ``model.info`` file. Resolves against ``model`` first if this
        bijector has not been used yet, matching
        :meth:`~autofit.non_linear.scaler.AbstractScaler.info_from_model`'s
        contract of being safe to call standalone.
        """
        if self._kind_code is None:
            self.from_model(model)

        priors = model.priors_ordered_by_id
        if len(priors) == 0:
            return ""

        rows = []
        for prior, kind in zip(priors, self.kinds):
            path = model.path_for_prior(prior)
            path_str = ".".join(str(p) for p in path) if path is not None else "?"
            rows.append((path_str, type(prior).__name__, kind))

        widths = [max(len(row[column]) for row in rows) for column in range(2)]

        info = f"Per-Parameter Bijector ({type(self).__name__})\n\n"
        for path_str, prior_name, kind in rows:
            info += f"{path_str:<{widths[0]}}   {prior_name:<{widths[1]}}   kind {kind}\n"

        return info


class BijectorNone(AbstractBijector):
    """
    The no-op bijector, and **the default**.

    Every coordinate is ``identity`` with ``scale = 1.0``, so ``phi == theta``
    exactly (``x / 1.0 == x`` and ``x * 1.0 == x`` bitwise in IEEE 754) and
    ``log_det_jacobian`` is exactly ``0.0``. Searches should test for
    ``isinstance(bijector, BijectorNone)`` and skip the change of variables
    entirely rather than applying it as a no-op, the same short-circuit
    :class:`~autofit.non_linear.scaler.ScalerNone` relies on, so the compiled
    step is unchanged too.
    """

    def _kind_scale_bounds_from_model(self, model):
        n = model.prior_count
        return (
            [_IDENTITY] * n,
            [1.0] * n,
            [0.0] * n,
            [1.0] * n,
        )

    def info_from_model(self, model) -> str:
        return ""


class BijectorAuto(AbstractBijector):
    """
    ``log`` for every eligible coordinate, ``identity`` (unscaled) elsewhere.

    Eligible means :class:`~autofit.mapper.prior.log_uniform.LogUniformPrior`
    with a strictly positive ``lower_limit`` (the only case the prior's own
    constructor allows in the first place, so the fallback below is a
    defence-in-depth guard rather than a reachable path through normal
    construction), or :class:`~autofit.mapper.prior.log_gaussian.
    LogGaussianPrior` (support ``(0, inf)``, unconditionally eligible). A
    malformed candidate falls back to ``identity`` with a logged warning,
    mirroring ``ScalerPriorWidth``'s fallback for the same prior type.
    """

    @staticmethod
    def _kind_for(prior) -> Tuple[int, float, float]:
        if isinstance(prior, LogUniformPrior):
            lower = float(prior.lower_limit)
            if lower > 0.0:
                return _LOG, lower, float(prior.upper_limit)
            logger.warning(
                f"BijectorAuto: LogUniformPrior with a non-positive lower_limit "
                f"({lower}); falling back to identity for this parameter."
            )
            return _IDENTITY, 0.0, 1.0

        if isinstance(prior, LogGaussianPrior):
            return _LOG, 0.0, np.inf

        return _IDENTITY, 0.0, 1.0

    def _kind_scale_bounds_from_model(self, model):
        kind_codes, lo, hi = [], [], []
        for prior in model.priors_ordered_by_id:
            code, low, high = self._kind_for(prior)
            kind_codes.append(code)
            lo.append(low)
            hi.append(high)
        return kind_codes, [1.0] * len(kind_codes), lo, hi


class BijectorLogit(AbstractBijector):
    """
    ``logit`` on the prior box for every two-sided finite coordinate,
    ``identity`` (unscaled) elsewhere.

    **Secondary arm.** :mod:`autofit.non_linear.scaler`'s objection to the
    unit cube stands unchanged: a genuinely boundary-pinned optimum is sent to
    infinity in ``phi``, and the reference cell has such optima. This class
    exists so that claim can be measured directly (a ``BijectorAuto`` vs
    ``BijectorLogit`` A/B), not as a recommended default.

    :class:`~autofit.mapper.prior.log_gaussian.LogGaussianPrior`'s support is
    ``(0, inf)`` -- half-open, not two-sided -- so it always falls back to
    ``identity`` here (with a logged warning), the same declared-support
    correction :class:`~autofit.non_linear.clipper.ClipperPriorBox` applies.
    """

    @staticmethod
    def _kind_for(prior) -> Tuple[int, float, float]:
        low = float(prior.lower_limit)
        high = float(prior.upper_limit)
        if isinstance(prior, LogGaussianPrior):
            low = 0.0

        if np.isfinite(low) and np.isfinite(high) and high > low:
            return _LOGIT, low, high

        logger.warning(
            f"BijectorLogit: {type(prior).__name__} has no two-sided finite "
            f"bound (lower={low}, upper={high}); falling back to identity for "
            "this parameter."
        )
        return _IDENTITY, 0.0, 1.0

    def _kind_scale_bounds_from_model(self, model):
        kind_codes, lo, hi = [], [], []
        for prior in model.priors_ordered_by_id:
            code, low, high = self._kind_for(prior)
            kind_codes.append(code)
            lo.append(low)
            hi.append(high)
        return kind_codes, [1.0] * len(kind_codes), lo, hi


class BijectorPerPath(AbstractBijector):
    """
    An explicit, user-declared ``{path: kind}`` mapping, resolved against a
    model's own path helper (:meth:`~autofit.mapper.prior_model.abstract.
    AbstractPriorModel.path_for_prior`).

    Parameters
    ----------
    kind_by_path
        Maps a dotted parameter path (as rendered by
        ``".".join(str(p) for p in model.path_for_prior(prior))`` -- the same
        format :meth:`~autofit.non_linear.scaler.ScalerPriorWidth.
        info_from_model` reports) to one of ``"identity"``, ``"log"``,
        ``"logit"``. A path not present in the mapping defaults to
        ``"identity"``. A requested kind that is not eligible for that
        prior (``"log"`` on anything but a positive-lower ``LogUniformPrior``
        or a ``LogGaussianPrior``; ``"logit"`` on a coordinate without a
        two-sided finite bound) falls back to ``"identity"`` with a logged
        warning rather than silently doing something else or raising --
        matching every other fallback in this module.
    """

    def __init__(self, kind_by_path: Dict[str, str]):
        super().__init__()
        self.kind_by_path = dict(kind_by_path)

    @staticmethod
    def _resolve_one(requested: str, prior, path_str: str) -> Tuple[int, float, float]:
        if requested == "log":
            if isinstance(prior, LogUniformPrior) and float(prior.lower_limit) > 0.0:
                return _LOG, float(prior.lower_limit), float(prior.upper_limit)
            if isinstance(prior, LogGaussianPrior):
                return _LOG, 0.0, np.inf
            logger.warning(
                f"BijectorPerPath: path {path_str!r} requested 'log' but its "
                f"prior ({type(prior).__name__}) is not eligible (needs a "
                "LogUniformPrior with a positive lower_limit, or a "
                "LogGaussianPrior); falling back to identity."
            )
            return _IDENTITY, 0.0, 1.0

        if requested == "logit":
            low = float(prior.lower_limit)
            high = float(prior.upper_limit)
            if isinstance(prior, LogGaussianPrior):
                low = 0.0
            if np.isfinite(low) and np.isfinite(high) and high > low:
                return _LOGIT, low, high
            logger.warning(
                f"BijectorPerPath: path {path_str!r} requested 'logit' but its "
                f"prior ({type(prior).__name__}) has no two-sided finite bound "
                f"(lower={low}, upper={high}); falling back to identity."
            )
            return _IDENTITY, 0.0, 1.0

        if requested != "identity":
            logger.warning(
                f"BijectorPerPath: path {path_str!r} requested unknown kind "
                f"{requested!r}; falling back to identity."
            )
        return _IDENTITY, 0.0, 1.0

    def _kind_scale_bounds_from_model(self, model):
        kind_codes, lo, hi = [], [], []
        for prior in model.priors_ordered_by_id:
            path = model.path_for_prior(prior)
            path_str = ".".join(str(p) for p in path) if path is not None else None
            requested = self.kind_by_path.get(path_str, "identity")
            code, low, high = self._resolve_one(requested, prior, path_str)
            kind_codes.append(code)
            lo.append(low)
            hi.append(high)
        return kind_codes, [1.0] * len(kind_codes), lo, hi


class BijectorDiagonal(AbstractBijector):
    """
    Adapter expressing an :class:`~autofit.non_linear.scaler.AbstractScaler`
    as a bijector: every coordinate is ``identity``, diagonally scaled by the
    wrapped scaler's ``scale_from_model``.

    This subsumes :class:`~autofit.non_linear.scaler.ScalerPriorWidth`
    exactly -- ``forward`` / ``inverse`` here reduce to precisely the
    ``phi = theta / scale`` / ``theta = scale * phi`` maps
    ``MultiStartGradient`` already applies under ``scaler=``. It exists so a
    diagonal preconditioner can be composed inside this framework (e.g.
    stacked with a per-path ``log`` on top, once a caller wants both) without
    duplicating the scale rule, and so ``scaler=`` keeps working completely
    unchanged: it is not wired into the ``scaler=`` code path itself, which
    stays exactly as it was.
    """

    def __init__(self, scaler: AbstractScaler):
        super().__init__()
        self.scaler = scaler

    def _kind_scale_bounds_from_model(self, model):
        scale = self.scaler.scale_from_model(model=model)
        n = len(scale)
        return [_IDENTITY] * n, list(scale), [0.0] * n, [1.0] * n

    def info_from_model(self, model) -> str:
        return self.scaler.info_from_model(model)
