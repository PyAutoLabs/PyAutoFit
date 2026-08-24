import numpy as np
import pytest

import autofit as af
from autofit import example
from autofit.non_linear.bijector import (
    BijectorAuto,
    BijectorDiagonal,
    BijectorLogit,
    BijectorNone,
    BijectorPerPath,
)
from autofit.non_linear.clipper import ClipperPriorBox
from autofit.non_linear.scaler import ScalerNone, ScalerPriorWidth

# Pure NumPy, deliberately -- the same discipline `test_scaler.py` uses. The
# bijector's contract is algebraic (round trip, monotonicity, the closed-form
# log-det, and that composing the objective through it cannot move the MAP)
# and none of that needs JAX.


def model_from(centre, normalization, sigma):
    model = af.Model(example.Gaussian)
    model.centre = centre
    model.normalization = normalization
    model.sigma = sigma
    return model


def log_uniform_model():
    return model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.LogUniformPrior(lower_limit=1.0e-6, upper_limit=1.0e6),
        sigma=af.GaussianPrior(mean=1.0, sigma=2.0),
    )


# ---------------------------------------------------------------------------
# BijectorNone
# ---------------------------------------------------------------------------


def test__bijector_none__is_identity_for_every_coordinate():
    model = log_uniform_model()
    bijector = BijectorNone().from_model(model)

    assert bijector.kinds == ["identity", "identity", "identity"]


def test__bijector_none__forward_and_inverse_are_bit_identical_to_the_input():
    """
    `x / 1.0 == x` and `x * 1.0 == x` exactly in IEEE 754, so this is not merely
    numerically close -- it is the byte-identical no-op the module docstring
    promises, matching `ScalerNone`'s guarantee.
    """
    model = log_uniform_model()
    bijector = BijectorNone().from_model(model)

    theta = np.array([3.0, 100.0, -1.5])

    assert (bijector.forward(theta) == theta).all()
    assert (bijector.inverse(theta) == theta).all()


def test__bijector_none__log_det_jacobian_is_exactly_zero():
    model = log_uniform_model()
    bijector = BijectorNone().from_model(model)

    phi = np.array([3.0, 100.0, -1.5])

    assert bijector.log_det_jacobian(phi) == 0.0


def test__bijector_none__writes_no_model_info_block():
    model = log_uniform_model()

    assert BijectorNone().info_from_model(model) == ""


def test__bijector_none__is_the_default_on_multi_start_gradient():
    assert isinstance(af.MultiStartAdam().bijector, BijectorNone)


# ---------------------------------------------------------------------------
# BijectorAuto: kind selection
# ---------------------------------------------------------------------------


def test__bijector_auto__picks_log_for_log_uniform_and_log_gaussian__identity_elsewhere():
    model = af.Model(example.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    model.normalization = af.LogUniformPrior(lower_limit=1.0e-6, upper_limit=1.0e6)
    model.sigma = af.LogGaussianPrior(mean=0.0, sigma=1.0)

    bijector = BijectorAuto().from_model(model)

    assert bijector.kinds == ["identity", "log", "log"]


def test__bijector_auto__log_uniform_with_a_non_positive_limit__falls_back_to_identity(
    caplog,
):
    """
    Regression / defence-in-depth: `LogUniformPrior.__init__` already raises on a
    non-positive `lower_limit`, so this path is not reachable through normal
    construction -- it is guarded anyway, the same "malformed prior, not merely
    awkward to scale" discipline `ScalerPriorWidth._term_for` uses.
    """

    class _MalformedLogUniform(af.LogUniformPrior):
        def __init__(self):
            # Bypass the constructor's own validation.
            super().__init__(lower_limit=1.0, upper_limit=2.0)
            self.lower_limit = 0.0

    model = af.Model(example.Gaussian)
    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    model.normalization = _MalformedLogUniform()
    model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    bijector = BijectorAuto().from_model(model)

    assert bijector.kinds[1] == "identity"
    assert "non-positive lower_limit" in caplog.text


# ---------------------------------------------------------------------------
# Round trip, monotonicity, closed-form log-det -- per kind.
# ---------------------------------------------------------------------------


class TestIdentityKind:
    def test__round_trip(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=-10.0, upper_limit=10.0),
            normalization=af.UniformPrior(lower_limit=-10.0, upper_limit=10.0),
            sigma=af.UniformPrior(lower_limit=-10.0, upper_limit=10.0),
        )
        bijector = BijectorAuto().from_model(model)

        theta = np.array([1.0, -2.5, 7.3])
        assert bijector.inverse(bijector.forward(theta)) == pytest.approx(theta)

    def test__log_det_jacobian_is_zero_for_unscaled_identity(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=-10.0, upper_limit=10.0),
            normalization=af.UniformPrior(lower_limit=-10.0, upper_limit=10.0),
            sigma=af.UniformPrior(lower_limit=-10.0, upper_limit=10.0),
        )
        bijector = BijectorAuto().from_model(model)

        assert bijector.log_det_jacobian(np.array([1.0, -2.5, 7.3])) == pytest.approx(
            0.0
        )


class TestLogKind:
    def test__round_trip(self):
        model = log_uniform_model()
        bijector = BijectorAuto().from_model(model)

        theta = np.array([3.0, 250.0, -1.5])
        assert bijector.inverse(bijector.forward(theta)) == pytest.approx(theta)

    def test__forward_is_strictly_monotone_increasing(self):
        model = log_uniform_model()
        bijector = BijectorAuto().from_model(model)

        thetas = np.geomspace(1.0e-6, 1.0e6, 25)
        phis = np.array(
            [bijector.forward(np.array([1.0, t, 0.0]))[1] for t in thetas]
        )

        assert (np.diff(phis) > 0.0).all()

    def test__log_det_jacobian_matches_the_closed_form(self):
        """
        `theta = exp(phi)` for a log-kind coordinate, so
        `d theta/d phi = exp(phi) = theta`, i.e. the per-coordinate contribution
        to `log_det_jacobian` is exactly `phi`, checked here against a finite
        difference of `inverse` directly (rather than re-deriving the same
        algebra `log_det_jacobian` uses, which would not be an independent
        check).
        """
        model = log_uniform_model()
        bijector = BijectorAuto().from_model(model)

        phi = np.array([1.0, 2.5, 0.5])  # centre/sigma identity, normalization log
        eps = 1.0e-6
        d_theta = (
            bijector.inverse(phi + np.array([0.0, eps, 0.0]))
            - bijector.inverse(phi - np.array([0.0, eps, 0.0]))
        ) / (2.0 * eps)
        numeric_log_det = np.log(np.abs(d_theta[1]))

        # Only one coordinate is log-kind here; isolate its own contribution by
        # zeroing the (known, `log(1.0) == 0`) identity contributions of the
        # unscaled centre/sigma coordinates.
        analytic = bijector.log_det_jacobian(phi)

        assert analytic == pytest.approx(numeric_log_det, abs=1.0e-4)


class TestLogitKind:
    def test__round_trip(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
            sigma=af.TruncatedGaussianPrior(
                mean=0.0, sigma=1.0, lower_limit=-1.0, upper_limit=1.0
            ),
        )
        bijector = BijectorLogit().from_model(model)

        assert bijector.kinds == ["logit", "logit", "logit"]

        theta = np.array([3.0, 0.1, -0.4])
        assert bijector.inverse(bijector.forward(theta)) == pytest.approx(theta)

    def test__forward_is_strictly_monotone_increasing(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
            sigma=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
        )
        bijector = BijectorLogit().from_model(model)

        thetas = np.linspace(0.0001, 7.9999, 25)
        phis = np.array(
            [bijector.forward(np.array([t, 0.0, 0.0]))[0] for t in thetas]
        )

        assert (np.diff(phis) > 0.0).all()

    def test__log_det_jacobian_matches_a_finite_difference(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
            sigma=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
        )
        bijector = BijectorLogit().from_model(model)

        phi = np.array([0.3, -0.7, 1.2])
        eps = 1.0e-6
        d_theta_0 = (
            bijector.inverse(phi + np.array([eps, 0.0, 0.0]))[0]
            - bijector.inverse(phi - np.array([eps, 0.0, 0.0]))[0]
        ) / (2.0 * eps)
        d_theta_1 = (
            bijector.inverse(phi + np.array([0.0, eps, 0.0]))[1]
            - bijector.inverse(phi - np.array([0.0, eps, 0.0]))[1]
        ) / (2.0 * eps)
        d_theta_2 = (
            bijector.inverse(phi + np.array([0.0, 0.0, eps]))[2]
            - bijector.inverse(phi - np.array([0.0, 0.0, eps]))[2]
        ) / (2.0 * eps)

        numeric_total = (
            np.log(np.abs(d_theta_0))
            + np.log(np.abs(d_theta_1))
            + np.log(np.abs(d_theta_2))
        )

        assert bijector.log_det_jacobian(phi) == pytest.approx(
            numeric_total, abs=1.0e-4
        )

    def test__log_gaussian_has_no_two_sided_bound__falls_back_to_identity(self, caplog):
        model = model_from(
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
            sigma=af.LogGaussianPrior(mean=0.0, sigma=1.0),
        )
        bijector = BijectorLogit().from_model(model)

        assert bijector.kinds[2] == "identity"
        assert "no two-sided finite bound" in caplog.text


# ---------------------------------------------------------------------------
# BijectorPerPath
# ---------------------------------------------------------------------------


class TestBijectorPerPath:
    def test__resolves_kinds_by_dotted_path(self):
        model = log_uniform_model()

        bijector = BijectorPerPath({"normalization": "log"}).from_model(model)

        assert bijector.kinds == ["identity", "log", "identity"]

    def test__unrequested_paths_default_to_identity(self):
        model = log_uniform_model()

        bijector = BijectorPerPath({}).from_model(model)

        assert bijector.kinds == ["identity", "identity", "identity"]

    def test__requesting_log_on_an_ineligible_prior_falls_back_to_identity(self, caplog):
        model = log_uniform_model()

        bijector = BijectorPerPath({"centre": "log"}).from_model(model)

        assert bijector.kinds[0] == "identity"
        assert "not eligible" in caplog.text

    def test__unknown_kind_falls_back_to_identity(self, caplog):
        model = log_uniform_model()

        bijector = BijectorPerPath({"centre": "banana"}).from_model(model)

        assert bijector.kinds[0] == "identity"
        assert "unknown kind" in caplog.text


# ---------------------------------------------------------------------------
# BijectorDiagonal: subsumes ScalerPriorWidth exactly.
# ---------------------------------------------------------------------------


class TestBijectorDiagonal:
    def test__reduces_to_the_scaler_map_exactly(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
            sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
        )
        scaler = ScalerPriorWidth()
        scale = scaler.scale_from_model(model)

        bijector = BijectorDiagonal(scaler).from_model(model)

        theta = np.array([3.0, 0.1, -0.05])

        assert bijector.forward(theta) == pytest.approx(theta / scale)
        assert bijector.inverse(theta / scale) == pytest.approx(theta)
        assert bijector.kinds == ["identity", "identity", "identity"]

    def test__info_from_model_delegates_to_the_wrapped_scaler(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
            sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
        )
        scaler = ScalerPriorWidth()

        assert BijectorDiagonal(scaler).info_from_model(model) == (
            scaler.info_from_model(model)
        )

    def test__scaler_none__is_the_identity_bijector_with_unit_scale(self):
        model = model_from(
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
            sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
        )
        bijector = BijectorDiagonal(ScalerNone()).from_model(model)

        theta = np.array([3.0, 0.1, -0.05])
        assert bijector.forward(theta) == pytest.approx(theta)


# ---------------------------------------------------------------------------
# The equivalence argument: composing the objective through the bijector
# cannot move the MAP. Cloned from `test_scaler.py`'s `test__objective_
# composed_through_the_scale_is_the_SAME_objective` / `test__scaling_cannot_
# move_the_MAP`, with one LOG coordinate added.
# ---------------------------------------------------------------------------


def _log_coordinate_objective(theta):
    """
    A three-parameter objective matching `log_uniform_model()`'s prior order:
    `theta[0]` (`UniformPrior`, identity-kind) is minimised at `3.0`,
    `theta[1]` (`LogUniformPrior`, log-kind) has a log-shaped optimum at
    `exp(2.0) ~ 7.39`, and `theta[2]` (`GaussianPrior`, identity-kind) is
    minimised at `0.05` -- so the identity and log branches are both
    exercised at once.
    """
    return (
        (3.0 - theta[0]) ** 2
        + (np.log(theta[1]) - 2.0) ** 2
        + 100.0 * (0.05 - theta[2]) ** 2
    )


def test__objective_composed_through_the_bijector_is_the_SAME_objective():
    """
    The algebraic pin. Composing as `f(bijector.inverse(phi))` evaluates the
    physical-space objective at the transformed point, so it is equal at every
    corresponding pair -- exactly, not approximately. Folding a Jacobian in
    instead would move the MAP and do so SILENTLY (see the module docstring's
    equivalence argument); this is the test that says which of the two was
    implemented.
    """
    model = log_uniform_model()
    bijector = BijectorAuto().from_model(model)

    rng = np.random.default_rng(0)
    for _ in range(50):
        theta = rng.uniform(
            np.array([-10.0, 0.5, -10.0]), np.array([10.0, 500.0, 10.0])
        )
        phi = bijector.forward(theta)

        assert _log_coordinate_objective(
            bijector.inverse(phi)
        ) == pytest.approx(_log_coordinate_objective(theta), rel=1.0e-9)


def test__bijector_cannot_move_the_MAP():
    """
    The end-to-end statement: minimise the raw objective, and minimise the
    bijector-composed one and map back, and the recovered argmin agrees --
    even though the third coordinate is stepping multiplicatively (in `log`)
    while the first two step linearly (`identity`).

    ``scipy.optimize.minimize`` rather than a hand-rolled fixed-step descent,
    for a reason that is itself part of the point: the raw-space objective is
    ``(log(theta) - 2)^2`` in the second coordinate, undefined for
    ``theta <= 0``, and unconstrained BFGS's own trial steps stray there and
    diverge -- exactly the failure a real physical-space search guards
    against with a ``Clipper`` (see :mod:`autofit.non_linear.clipper`), and
    exactly what stepping in ``phi = log(theta)`` (unconstrained, valid on
    the whole real line) sidesteps entirely. The raw-space run is therefore
    given the box a real search would also need (``L-BFGS-B``,
    ``theta[1] > 0``); the ``phi``-space run needs no such box and uses plain
    ``BFGS``.
    """
    from scipy import optimize

    model = log_uniform_model()
    bijector = BijectorAuto().from_model(model)

    start = np.array([8.0, 500.0, 0.5])
    expected = np.array([3.0, np.exp(2.0), 0.05])

    unscaled = optimize.minimize(
        _log_coordinate_objective,
        start,
        method="L-BFGS-B",
        bounds=[(None, None), (1.0e-8, None), (None, None)],
    ).x

    phi_start = bijector.forward(start)
    transformed_phi = optimize.minimize(
        lambda phi: _log_coordinate_objective(bijector.inverse(phi)),
        phi_start,
        method="BFGS",
    ).x
    transformed = bijector.inverse(transformed_phi)

    assert unscaled == pytest.approx(expected, abs=1.0e-3)
    assert transformed == pytest.approx(expected, abs=1.0e-3)
    assert transformed == pytest.approx(unscaled, abs=1.0e-3)


# ---------------------------------------------------------------------------
# Composition with the clipper: clipping commutes with `forward`.
# ---------------------------------------------------------------------------


def test__clip_commutes_with_forward():
    """
    Every bijector kind is monotone increasing, so clipping `theta` against
    physical bounds and then mapping through `forward` must give exactly the
    same result as mapping `theta` through `forward` first and then clipping
    against the forward-mapped bounds. This is the property
    `ClipperPriorBox.project(..., bijector=...)` relies on.
    """
    model = log_uniform_model()
    bijector = BijectorAuto().from_model(model)
    clipper = ClipperPriorBox()

    lower, upper = clipper._inset_from_model(model, kinds=bijector.kinds)

    theta = np.array([9.0, 1.0e-8, 100.0])  # centre and normalization out of box

    clip_then_forward = bijector.forward(np.clip(theta, lower, upper))

    lower_t, upper_t = bijector.bounds_forward(lower, upper)
    forward_then_clip = np.clip(bijector.forward(theta), lower_t, upper_t)

    assert clip_then_forward == pytest.approx(forward_then_clip)


def test__bijector_none__forward_and_bounds_forward_are_the_identity():
    model = log_uniform_model()
    bijector = BijectorNone().from_model(model)
    clipper = ClipperPriorBox()

    lower, upper = clipper.bounds_from_model(model)
    lower_t, upper_t = bijector.bounds_forward(lower, upper)

    assert lower_t == pytest.approx(lower)
    assert upper_t == pytest.approx(upper)
