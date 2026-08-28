from typing import Tuple

import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.mapper.identifier import Identifier
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.non_linear.bijector import (
    BijectorAuto,
    BijectorDiagonal,
    BijectorLogit,
    BijectorNone,
    BijectorPerPath,
)
from autofit.non_linear.clipper import (
    ClipperNone,
    ClipperPriorBox,
    ClipperPriorBoxJoint,
)


class Widget:
    def __init__(self, alpha=None, beta=None, gamma=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


def _model(**priors):
    model = af.Model(Widget)
    for name, prior in priors.items():
        setattr(model, name, prior)
    return model


class TestBoundsExtraction:
    def test__uniform_prior__finite_both_sides(self):
        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=2.0),
            beta=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
            gamma=af.UniformPrior(lower_limit=10.0, upper_limit=20.0),
        )

        lower, upper = ClipperPriorBox(margin=0.0).bounds_from_model(model=model)

        assert lower == pytest.approx(np.array([0.0, -1.0, 10.0]))
        assert upper == pytest.approx(np.array([2.0, 1.0, 20.0]))

    def test__truncated_gaussian_prior__reads_message_limits(self):
        model = _model(
            alpha=af.TruncatedGaussianPrior(
                mean=0.0, sigma=1.0, lower_limit=-1.0, upper_limit=1.0
            )
        )

        lower, upper = ClipperPriorBox(margin=0.0).bounds_from_model(model=model)

        assert lower[0] == pytest.approx(-1.0)
        assert upper[0] == pytest.approx(1.0)

    def test__log_uniform_prior__finite_both_sides(self):
        model = _model(alpha=af.LogUniformPrior(lower_limit=1.0e-6, upper_limit=1.0))

        lower, upper = ClipperPriorBox(margin=0.0).bounds_from_model(model=model)

        assert lower[0] == pytest.approx(1.0e-6)
        assert upper[0] == pytest.approx(1.0)

    def test__gaussian_prior__passes_through_as_infinite(self):
        """
        A GaussianPrior has unbounded support and must be reported as such -- and,
        critically, must not acquire a NaN bound from any inset arithmetic. See
        ``test__gaussian_coordinate_is_never_nan`` for the projection-level guard.
        """
        model = _model(alpha=af.GaussianPrior(mean=0.0, sigma=1.0))

        lower, upper = ClipperPriorBox().bounds_from_model(model=model)

        assert lower[0] == -np.inf
        assert upper[0] == np.inf
        assert not np.isnan(lower).any()
        assert not np.isnan(upper).any()

    def test__log_gaussian_prior__lower_bound_is_declared_by_the_prior(self):
        """
        LogGaussianPrior declares its own (0, inf) support and flags the lower bound
        strict, so the clipper reads it like any other prior and the projected value
        lands *above* zero rather than on it.

        This test used to assert ``prior.lower_limit == -np.inf`` and the clipper
        supplied the real bound itself via an ``isinstance`` switch. That was a
        workaround for PyAutoFit#1526: the prior was telling every consumer, not just
        this one, that a strictly positive parameter could go negative. The bounds
        assertions below are unchanged by the fix — that equivalence is the point.
        """
        prior = af.LogGaussianPrior(mean=0.0, sigma=1.0)

        assert prior.lower_limit == 0.0
        assert prior.upper_limit == np.inf
        assert prior.lower_limit_strict is True
        assert prior.upper_limit_strict is False

        model = _model(alpha=prior)
        lower, upper = ClipperPriorBox(strict_epsilon=1.0e-12).bounds_from_model(
            model=model
        )

        assert lower[0] > 0.0
        assert lower[0] == pytest.approx(1.0e-12)
        assert upper[0] == np.inf

    def test__mixed_model__every_prior_type_at_once(self):
        model = af.Model(Widget)
        model.alpha = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
        model.beta = af.GaussianPrior(mean=0.0, sigma=1.0)
        model.gamma = af.LogGaussianPrior(mean=0.0, sigma=1.0)

        lower, upper = ClipperPriorBox().bounds_from_model(model=model)

        assert not np.isnan(lower).any()
        assert not np.isnan(upper).any()
        assert len(lower) == model.prior_count


class TestOrdering:
    def test__bound_at_index_i_belongs_to_the_parameter_at_index_i(self):
        """
        The ordering is load-bearing and silent if wrong: a mismatch would clip the
        wrong parameter with no error.

        Asserting ``priors_ordered_by_id`` against ``prior_tuples_ordered_by_id``
        would be vacuous -- the former is derived from the latter. The real check is
        against ``instance_from_vector``, which is what actually consumes the vector,
        so the priors are given disjoint boxes and each attribute must land inside
        its own.
        """
        model = af.Model(Widget)
        # Assigned out of alphabetical order so prior id order and attribute order
        # cannot coincide by accident.
        model.gamma = af.UniformPrior(lower_limit=30.0, upper_limit=31.0)
        model.alpha = af.UniformPrior(lower_limit=10.0, upper_limit=11.0)
        model.beta = af.UniformPrior(lower_limit=20.0, upper_limit=21.0)

        lower, upper = ClipperPriorBox(margin=0.0).bounds_from_model(model=model)

        midpoints = (lower + upper) / 2.0
        instance = model.instance_from_vector(vector=list(midpoints))

        for value, low, high in [
            (instance.alpha, 10.0, 11.0),
            (instance.beta, 20.0, 21.0),
            (instance.gamma, 30.0, 31.0),
        ]:
            assert low <= value <= high


class TestProject:
    def test__overshoot_is_projected_back_inside_and_masked(self):
        """
        The boxes are placed near zero deliberately. At float32 an overshoot of
        ``2.0000001`` against an upper bound of ``2.0`` is not representable and the
        assertion would pass vacuously; near zero the overshoot survives the dtype.
        """
        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            gamma=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )

        clipper = ClipperPriorBox(margin=0.0)

        projected, mask = clipper.project(
            vector=np.array([1.5, 0.5, -0.5]), model=model
        )

        assert projected == pytest.approx(np.array([1.0, 0.5, 0.0]))
        assert list(mask) == [True, False, True]

    def test__in_bounds_vector_is_untouched_and_mask_is_all_false(self):
        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )

        vector = np.array([0.25, 0.75])
        projected, mask = ClipperPriorBox(margin=0.0).project(
            vector=vector, model=model
        )

        assert projected == pytest.approx(vector)
        assert not mask.any()

    def test__gaussian_coordinate_is_never_nan(self):
        """
        Regression guard.

        The obvious inset, ``lower + margin * (upper - lower)``, is ``-inf + inf`` for
        an unbounded prior, i.e. NaN -- and clipping against a NaN bound turns the
        coordinate into NaN, taking the whole objective with it. That failure looks
        exactly like the lane death the clipper exists to prevent, and it would fire
        on any model carrying a GaussianPrior. This test fails against that
        implementation and passes against the bound-kind one.
        """
        model = af.Model(Widget)
        model.alpha = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
        model.beta = af.GaussianPrior(mean=0.0, sigma=1.0)

        projected, mask = ClipperPriorBox().project(
            vector=np.array([2.5, 900.0]), model=model
        )

        assert not np.isnan(projected).any()
        assert projected[1] == 900.0
        assert list(mask) == [True, False]

    def test__log_gaussian_is_projected_to_finite_log_prior(self):
        """
        Regression guard.

        A relative margin is identically zero for a half-open bound, so it would clip
        onto exactly 0.0 -- where LogGaussianPrior's support is strict and
        ``log_prior`` is -inf. The projected value must therefore be strictly
        positive, and the resulting log prior finite.
        """
        prior = af.LogGaussianPrior(mean=0.0, sigma=1.0)
        model = _model(alpha=prior)

        for value in [0.0, -3.0]:
            projected, mask = ClipperPriorBox().project(
                vector=np.array([value]), model=model
            )

            assert projected[0] > 0.0
            assert mask[0]
            assert np.isfinite(prior.log_prior_from_value(value=projected[0], xp=np))

    def test__batched_vector_clips_per_lane(self):
        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )

        params = np.array([[1.5, 0.5], [0.25, 0.75], [-0.5, 2.0]])

        projected, mask = ClipperPriorBox(margin=0.0).project(
            vector=params, model=model
        )

        assert projected.shape == params.shape
        assert projected == pytest.approx(
            np.array([[1.0, 0.5], [0.25, 0.75], [0.0, 1.0]])
        )
        assert list(mask.any(axis=-1)) == [True, False, True]

    def test__margin_insets_inside_a_two_sided_box(self):
        model = _model(alpha=af.UniformPrior(lower_limit=0.0, upper_limit=2.0))

        projected, _ = ClipperPriorBox(margin=1.0e-6).project(
            vector=np.array([5.0]), model=model
        )

        assert projected[0] < 2.0
        assert projected[0] == pytest.approx(2.0 - 2.0e-6)


class TestClipperNone:
    def test__project_is_the_identity_and_mask_is_all_false(self):
        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )

        vector = np.array([5.0, -5.0])
        projected, mask = ClipperNone().project(vector=vector, model=model)

        assert projected is vector
        assert not mask.any()

    def test__bounds_are_infinite(self):
        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )

        lower, upper = ClipperNone().bounds_from_model(model=model)

        assert list(lower) == [-np.inf, -np.inf]
        assert list(upper) == [np.inf, np.inf]


class TestSearchWiring:
    def test__default_clipper_is_clipper_none(self):
        assert isinstance(af.LBFGS().clipper, ClipperNone)
        assert isinstance(af.MultiStartAdam().clipper, ClipperNone)

    def test__clipper_round_trips_through_the_constructor(self):
        clipper = ClipperPriorBox(margin=1.0e-4)

        assert af.MultiStartAdam(clipper=clipper).clipper is clipper
        assert af.LBFGS(clipper=clipper).clipper is clipper

    def test__lbfgs_builds_a_scipy_bounds_object(self):
        """
        Not merely "bounds are passed": ``bounds_from_model`` returns two arrays, and
        ``optimize.minimize`` reads a ``(lower, upper)`` tuple as a *sequence of
        (min, max) pairs*. Handing the tuple straight through silently mis-fits a
        two-parameter model and raises at any other dimensionality, so the search
        must build an explicit ``Bounds``.
        """
        from scipy import optimize

        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )

        bounds = af.LBFGS(clipper=ClipperPriorBox(margin=0.0))._bounds_from(model=model)

        assert isinstance(bounds, optimize.Bounds)
        assert bounds.lb == pytest.approx(np.array([0.0, 0.0]))
        assert bounds.ub == pytest.approx(np.array([1.0, 1.0]))

    def test__lbfgs_bounds_actually_constrain_a_two_parameter_fit(self):
        """
        Two parameters specifically: the dimensionality at which the wrong bounds
        shape fails *silently* rather than raising. With the optimum outside the box
        the correct answer is the corner; the tuple form would return ``[0, 1]``.
        """
        from scipy import optimize

        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )
        bounds = af.LBFGS(clipper=ClipperPriorBox(margin=0.0))._bounds_from(model=model)

        result = optimize.minimize(
            fun=lambda x: float(np.sum((x - 5.0) ** 2)),
            x0=np.array([0.5, 0.5]),
            method="L-BFGS-B",
            bounds=bounds,
        )

        assert result.x == pytest.approx(np.array([1.0, 1.0]))

    def test__no_clipper_means_no_bounds_are_passed(self):
        model = _model(alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0))

        assert af.LBFGS()._bounds_from(model=model) is None

    def test__lbfgs_fit_runs_end_to_end_with_and_without_a_clipper(self):
        """
        A real `search.fit()`, not just `_bounds_from` in isolation.

        The whole library suite passed while `_fit` had an undefined `optimize`,
        because nothing in it ever executed an LBFGS fit -- the bug only surfaced
        in a manual end-to-end run. This is the cheap smoke test that closes that
        gap: it fits a 1D Gaussian on the NumPy path and asserts the clipped run
        respects the box.
        """
        from autofit import example

        xvalues = np.arange(60)
        truth = example.Gaussian(centre=30.0, normalization=25.0, sigma=8.0)
        data = np.asarray(truth.model_data_from(xvalues=xvalues))
        noise_map = np.full(60, 1.0)

        def analysis():
            instance = example.Analysis(data=data, noise_map=noise_map)
            instance._use_jax = False
            return instance

        def model():
            built = af.Model(example.Gaussian)
            # The truth (centre=30) is deliberately outside this box, so an
            # enforced bound visibly changes the answer.
            built.centre = af.UniformPrior(lower_limit=5.0, upper_limit=20.0)
            built.normalization = af.UniformPrior(lower_limit=10.0, upper_limit=40.0)
            built.sigma = af.UniformPrior(lower_limit=1.0, upper_limit=20.0)
            return built

        unbounded = af.LBFGS(maxiter=15).fit(model=model(), analysis=analysis())
        clipped = af.LBFGS(maxiter=15, clipper=ClipperPriorBox()).fit(
            model=model(), analysis=analysis()
        )

        unbounded_centre = unbounded.samples.max_log_likelihood(as_instance=False)[0]
        clipped_centre = clipped.samples.max_log_likelihood(as_instance=False)[0]

        assert clipped_centre <= 20.0
        assert not np.isnan(clipped_centre)
        # Since PyAutoFit#1489 the strict NumPy-path priors make the objective
        # -inf outside the box, so even the clipper-less fit can no longer chase
        # the truth past the prior edge — it hits an infinite wall instead of
        # escaping. (This test previously asserted `unbounded_centre > 20.0`,
        # pinning the pre-#1489 escape as the foil for the clipped run.)
        assert unbounded_centre <= 20.0
        assert not np.isnan(unbounded_centre)

    def test__non_bound_supporting_method_raises_rather_than_being_ignored(self):
        """
        SciPy does not reject bounds that ``BFGS`` cannot use -- it warns and returns
        the unconstrained optimum. Silently handing back an unbounded fit to a user
        who asked for prior-support enforcement is the failure mode being prevented.
        """
        model = _model(alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0))

        search = af.BFGS(clipper=ClipperPriorBox())

        with pytest.raises(exc.SearchException):
            search._bounds_from(model=model)


class TestBijectorComposition:
    """
    ``AbstractClipper.project(..., bijector=...)`` -- see the module docstring's
    "Composition with a bijector" section and
    ``autofit.non_linear.bijector``'s equivalence argument.
    """

    def test__scale_and_bijector_together__raises(self):
        model = _model(alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0))
        bijector = BijectorAuto().from_model(model)

        with pytest.raises(ValueError):
            ClipperPriorBox().project(
                vector=np.array([0.5]),
                model=model,
                scale=np.array([1.0]),
                bijector=bijector,
            )

    def test__clipper_none__accepts_a_bijector_and_ignores_it(self):
        model = _model(alpha=af.UniformPrior(lower_limit=0.0, upper_limit=1.0))
        bijector = BijectorAuto().from_model(model)

        vector = np.array([5.0])
        projected, mask = ClipperNone().project(
            vector=vector, model=model, bijector=bijector
        )

        assert projected is vector
        assert not mask.any()

    def test__log_uniform_inset__is_taken_in_log_space_not_physical_space(self):
        """
        The bug found while wiring this up (see the ``clipper`` module
        docstring). The physical-relative inset (`margin * (upper - lower)`)
        is `~1e-6 + 1e-6 * (1e6 - 1e-6) ~ 1.0` for `LogUniform(1e-6, 1e6)` --
        fencing off virtually the entire support. The log-space inset must
        instead land close to the ORIGINAL bound, off by a factor of
        `exp(margin * log(upper / lower))`, not by an additive ~1.0.
        """
        model = _model(alpha=af.LogUniformPrior(lower_limit=1.0e-6, upper_limit=1.0e6))
        bijector = BijectorAuto().from_model(model)
        assert bijector.kinds == ["log"]

        clipper = ClipperPriorBox(margin=1.0e-6)

        physical_lower, physical_upper = clipper.bounds_from_model(model)
        log_aware_lower, log_aware_upper = clipper._inset_from_model(
            model, kinds=bijector.kinds
        )

        # The un-aware inset is the bug: it is nowhere near the original 1e-6.
        assert physical_lower[0] > 0.5

        # The kind-aware inset stays close (multiplicatively) to the original
        # bound, not displaced by an O(1) physical amount.
        assert log_aware_lower[0] == pytest.approx(1.0e-6, rel=1.0e-3)
        assert log_aware_upper[0] == pytest.approx(1.0e6, rel=1.0e-3)
        assert log_aware_lower[0] > 1.0e-6
        assert log_aware_upper[0] < 1.0e6

    def test__project_with_bijector__matches_a_direct_physical_clip_mapped_through(
        self,
    ):
        """
        ``clipper.project(..., bijector=...)`` clips in `phi`-space against
        `bijector.bounds_forward` of the (kind-aware) inset box. Because every
        bijector kind is monotone increasing this must recover EXACTLY the
        same physical point as clipping directly against that same inset box
        and never routing through `phi` at all.
        """
        model = _model(
            alpha=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
            beta=af.LogUniformPrior(lower_limit=1.0e-6, upper_limit=1.0e6),
            gamma=af.GaussianPrior(mean=1.0, sigma=2.0),
        )
        clipper = ClipperPriorBox()
        bijector = BijectorAuto().from_model(model)

        theta = np.array([9.0, 1.0e-8, 0.4])  # alpha and beta both out of box

        lower, upper = clipper._inset_from_model(model, kinds=bijector.kinds)
        direct_physical_clip = np.clip(theta, lower, upper)

        phi = bijector.forward(theta)
        projected_phi, mask = clipper.project(vector=phi, model=model, bijector=bijector)
        recovered = bijector.inverse(projected_phi)

        assert recovered == pytest.approx(direct_physical_clip)
        assert list(mask) == [True, True, False]

    def test__no_bijector__bounds_from_model_is_unaffected_by_the_log_kind_fix(self):
        """
        The plain (no-bijector) `bounds_from_model` path -- used directly by
        LBFGS's `_bounds_from` and by any existing caller -- must be
        byte-for-byte what it was before: only `project(..., bijector=...)`
        is kind-aware.
        """
        model = _model(alpha=af.LogUniformPrior(lower_limit=1.0e-6, upper_limit=1.0e6))
        clipper = ClipperPriorBox(margin=1.0e-6)

        lower, upper = clipper.bounds_from_model(model)

        assert lower[0] == pytest.approx(1.0e-6 + 1.0e-6 * (1.0e6 - 1.0e-6))


# The radius `Ellipse` below declares. Deliberately not 1.0: the class it stands
# in for (`autogalaxy.profiles.geometry_profiles.EllProfile`) declares the
# ellipticity CLAMP, past which its own conversion saturates, rather than the
# boundary of formal validity.
BALL_RADIUS = 0.999


class Ellipse:
    """A component declaring a ball on its `ell_comps` tuple prior.

    Stands in for an elliptical profile without importing one: the clipper knows
    nothing about ellipticity, only about `__model_ball_constraints__`.
    """

    __model_ball_constraints__ = ((("ell_comps",), BALL_RADIUS),)

    def __init__(
        self,
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
    ):
        self.ell_comps = ell_comps
        self.intensity = intensity


def _ball_model(lower_limit=-1.0, upper_limit=1.0):
    """An `Ellipse` model whose parameter vector is
    ``(ell_comps_0, ell_comps_1, intensity)``.

    Priors are set explicitly because the fixture class has no entry in the
    priors config, and the tuple prior is built by hand because a class with no
    config gets its tuple default as an instance rather than as priors.
    """
    model = af.Model(Ellipse)

    tuple_prior = TuplePrior()
    tuple_prior.ell_comps_0 = af.UniformPrior(
        lower_limit=lower_limit, upper_limit=upper_limit
    )
    tuple_prior.ell_comps_1 = af.UniformPrior(
        lower_limit=lower_limit, upper_limit=upper_limit
    )
    model.ell_comps = tuple_prior
    model.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

    return model


class TestJointBall:
    def test__a_corner_of_the_box_is_projected_onto_the_disk(self):
        """`(-1, -1)` is inside BOTH prior boxes and a factor `sqrt(2)` outside
        the disk -- the region no per-coordinate bound can see, and the whole
        reason this clipper exists."""
        model = _ball_model()

        projected, _ = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([-1.0, -1.0, 5.0]), model=model
        )

        assert np.hypot(projected[0], projected[1]) == pytest.approx(BALL_RADIUS)

    def test__the_projection_preserves_the_angle(self):
        """A radial shrink, not a per-coordinate clip: the lane keeps the
        ellipse orientation it had found and loses only the magnitude the
        geometry cannot support."""
        # A wider box than the disk, so the point under test is outside the ball
        # and inside the box -- otherwise the box clip lands first and there is
        # no angle left for the ball to preserve.
        model = _ball_model(lower_limit=-2.0, upper_limit=2.0)

        projected, _ = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([-1.2, 1.6, 5.0]), model=model
        )

        assert projected[0] / projected[1] == pytest.approx(-0.6 / 0.8)
        assert projected[0] == pytest.approx(-0.6 * BALL_RADIUS)
        assert projected[1] == pytest.approx(0.8 * BALL_RADIUS)

    def test__an_interior_point_is_bit_identical(self):
        model = _ball_model()
        vector = np.array([0.1, 0.2, 5.0])

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=vector, model=model
        )

        assert (projected == vector).all()
        assert not mask.any()

    def test__the_origin_is_not_nan(self):
        """`r = 0` is where a spherical profile's linked components sit and where
        many priors start. The `sqrt` and the division must both survive it."""
        model = _ball_model()

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([0.0, 0.0, 5.0]), model=model
        )

        assert np.isfinite(projected).all()
        assert projected[0] == 0.0
        assert projected[1] == 0.0
        assert not mask.any()

    def test__the_mask_is_set_on_both_members_and_only_them(self):
        """The mask's consumer is the momentum reset. A lane pushed out of the
        disk carries outward momentum in BOTH coordinates; zeroing one leaves the
        pair spiralling straight back out."""
        model = _ball_model()

        _, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([-1.0, -1.0, 5.0]), model=model
        )

        assert mask[0]
        assert mask[1]
        assert not mask[2]

    def test__batched_input_projects_per_lane(self):
        model = _ball_model()

        vector = np.array(
            [
                [-1.0, -1.0, 5.0],
                [0.1, 0.2, 5.0],
                [0.0, 0.0, 1.0],
            ]
        )

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=vector, model=model
        )

        assert np.hypot(projected[0, 0], projected[0, 1]) == pytest.approx(BALL_RADIUS)
        assert (projected[1] == vector[1]).all()
        assert (projected[2] == vector[2]).all()

        assert mask[0].tolist() == [True, True, False]
        assert not mask[1].any()
        assert not mask[2].any()

    def test__the_box_is_applied_before_the_ball(self):
        """A point outside BOTH is clipped onto the box and then shrunk onto the
        ball, landing inside both. Applying them the other way round would push a
        point off the ball it had just been placed on."""
        model = _ball_model()

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([5.0, 0.0, 5.0]), model=model
        )

        assert projected[0] == pytest.approx(BALL_RADIUS)
        assert projected[1] == 0.0
        assert mask[0]
        assert mask[1]

    def test__a_model_declaring_no_ball_is_the_plain_box_clipper(self):
        """The subclass is opt-in on BOTH sides: nothing happens to a model whose
        classes declare no geometry, so it can be configured globally."""
        model = _model(
            alpha=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
        )
        vector = np.array([5.0, -5.0])

        box, box_mask = ClipperPriorBox(margin=0.0).project(vector=vector, model=model)
        joint, joint_mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=vector, model=model
        )

        assert (box == joint).all()
        assert (box_mask == joint_mask).all()

    def test__the_prior_box_is_still_enforced(self):
        """Inheriting the box is not decoration: a coordinate outside its prior
        and inside the disk is still clipped."""
        model = _ball_model(lower_limit=-0.5, upper_limit=0.5)

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([0.9, 0.0, 5.0]), model=model
        )

        assert projected[0] == pytest.approx(0.5)
        assert mask[0]


class TestJointBallComposesWithAMap:
    """A ball is a statement about PHYSICAL parameters, but it survives a common
    linear rescale of both its members: a disk of radius `R` becomes a disk of
    radius `R / s`. Only a genuinely non-linear pair -- a `log`/`logit` kind, or
    two identity coordinates with DIFFERENT scales (an ellipse) -- has no
    projection, and only those are refused."""

    def test__the_no_op_bijector_composes_and_is_bit_identical(self):
        """`BijectorNone` is every coordinate identity-scaled by 1.0, so passing
        it must be indistinguishable from passing nothing -- not merely close."""
        model = _ball_model()
        clipper = ClipperPriorBoxJoint(margin=0.0)
        vector = np.array([-1.0, -1.0, 5.0])

        plain, plain_mask = clipper.project(vector=vector, model=model)
        composed, composed_mask = clipper.project(
            vector=vector,
            model=model,
            bijector=BijectorNone().from_model(model=model),
        )

        assert (composed == plain).all()
        assert (composed_mask == plain_mask).all()
        assert np.hypot(composed[0], composed[1]) == pytest.approx(BALL_RADIUS)

    def test__a_log_on_a_path_the_ball_does_not_touch_composes(self):
        """The case the blanket refusal cost: a model reparameterised through
        `log` on some OTHER parameter still has a linear `ell_comps` pair, so the
        disk is still a disk and the corner is still projected onto it."""
        model = _ball_model()
        model.intensity = af.LogUniformPrior(lower_limit=1.0e-3, upper_limit=1.0e3)

        bijector = BijectorPerPath({"intensity": "log"}).from_model(model=model)
        assert bijector.kinds == ["identity", "identity", "log"]

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([-1.0, -1.0, np.log(5.0)]),
            model=model,
            bijector=bijector,
        )

        # The ball pair is untouched by the map, so the physical answer stands:
        # the corner projected radially onto the disk, angle preserved.
        assert np.hypot(projected[0], projected[1]) == pytest.approx(BALL_RADIUS)
        assert projected[0] == pytest.approx(projected[1])
        assert mask[0]
        assert mask[1]

    def test__a_logit_on_the_ball_pair_itself_is_refused_and_names_the_pair(self):
        """`logit` maps the disk to a region with no closed form. The refusal has
        to say WHICH pair and WHAT kinds, or a user with a large model cannot act
        on it."""
        model = _ball_model()
        clipper = ClipperPriorBoxJoint(margin=0.0)
        bijector = BijectorLogit().from_model(model=model)

        assert bijector.kinds[:2] == ["logit", "logit"]

        with pytest.raises(ValueError) as exc_info:
            clipper.project(
                vector=np.array([-1.0, -1.0, 5.0]),
                model=model,
                bijector=bijector,
            )

        message = str(exc_info.value)
        assert "(0, 1)" in message
        assert "logit" in message

    def test__a_common_scale_projects_onto_the_divided_radius(self):
        """The composition rule itself: in coordinates scaled by `s`, the disk of
        radius `R` IS the disk of radius `R / s`."""
        model = _ball_model()

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            # The stepped vector: the physical corner `(-1, -1)` divided by 2.
            vector=np.array([-0.5, -0.5, 5.0]),
            model=model,
            scale=np.array([2.0, 2.0, 1.0]),
        )

        assert np.hypot(projected[0], projected[1]) == pytest.approx(BALL_RADIUS / 2.0)
        assert mask[0]
        assert mask[1]

    def test__an_unequal_scale_on_the_pair_is_refused(self):
        """Two different scales map the disk to an ELLIPSE, whose nearest-point
        projection is not a radial shrink at all."""
        model = _ball_model()

        with pytest.raises(ValueError) as exc_info:
            ClipperPriorBoxJoint(margin=0.0).project(
                vector=np.array([-0.5, -0.5, 5.0]),
                model=model,
                scale=np.array([2.0, 3.0, 1.0]),
            )

        assert "(0, 1)" in str(exc_info.value)

    def test__a_diagonal_bijector_with_equal_pair_widths_is_accepted(self):
        """`BijectorDiagonal(ScalerPriorWidth())` reports every coordinate as
        `identity`, and the two `ell_comps` priors have equal widths -- so the
        pair shares one scale and composes."""
        model = _ball_model()
        bijector = BijectorDiagonal(af.ScalerPriorWidth()).from_model(model=model)

        scales = bijector.identity_scales
        assert scales[0] == scales[1]
        assert scales[0] is not None

        projected, mask = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([-1.0, -1.0, 5.0]) / scales[0],
            model=model,
            bijector=bijector,
        )

        assert np.hypot(projected[0], projected[1]) == pytest.approx(
            BALL_RADIUS / scales[0]
        )
        assert mask[0]
        assert mask[1]

    def test__a_model_with_no_ball_is_unaffected_by_any_of_this(self):
        """The resolution is per DECLARED pair, so a model declaring none never
        reaches it -- a pipeline may configure the joint clipper globally and
        still use any bijector it likes."""
        model = _model(
            alpha=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
        )

        projected, _ = ClipperPriorBoxJoint(margin=0.0).project(
            vector=np.array([5.0, -5.0]),
            model=model,
            bijector=BijectorLogit().from_model(model=model),
        )

        assert np.isfinite(projected).all()


class TestJointBallIdentifier:
    def test__the_base_clippers_identifier_fields_are_pinned(self):
        """Pinned so that a subclass taking a further constructor argument cannot
        silently re-key every stored `ClipperPriorBox` result. The pinned tuple
        reproduces exactly what the identifier's argspec fallback inferred."""
        assert ClipperPriorBox.__identifier_fields__ == ("margin", "strict_epsilon")

    def test__the_joint_clipper_is_a_different_identifier(self):
        """It changes where a lane can sit and therefore the result, so two runs
        differing only in clipper must not share an output directory."""
        model = _ball_model()

        box = Identifier([af.MultiStartAdam(clipper=ClipperPriorBox()), model, None])
        joint = Identifier(
            [af.MultiStartAdam(clipper=ClipperPriorBoxJoint()), model, None]
        )

        assert str(box) != str(joint)

    def test__composing_with_a_bijector_leaves_the_identifier_unchanged(self):
        """The composition adds no constructor argument, deliberately: had it
        taken one, every stored `ClipperPriorBoxJoint` result would have been
        silently re-keyed by a change that does not alter what those runs did.

        Note what this does NOT say. The equality holds partly because the
        search identifier does not yet tag the `bijector` at all -- a separate,
        pre-existing gap that predates this change and is tracked on its own;
        this test pins that the clipper's own key is untouched, not that two
        runs differing only in bijector *should* share an output directory."""
        model = _ball_model()

        plain = Identifier(
            [af.MultiStartAdam(clipper=ClipperPriorBoxJoint()), model, None]
        )
        composed = Identifier(
            [
                af.MultiStartAdam(
                    clipper=ClipperPriorBoxJoint(),
                    bijector=BijectorPerPath({"intensity": "log"}),
                ),
                model,
                None,
            ]
        )

        assert str(plain) == str(composed)


# The `jit`/`grad`/`vmap` behaviour of the radial shrink -- in particular that the
# "double where" keeps `grad` finite at the origin, where `sqrt(0)` has an infinite
# derivative -- is deliberately NOT tested here: unit tests stay numpy-only because
# JAX is an optional dependency. It is exercised by the traced path in
# `autolens_workspace_test`, and the numpy tests above pin the same arithmetic.


class TestJointBallSearchWiring:
    def test__lbfgs_refuses_it_rather_than_dropping_the_ball(self):
        """`LBFGS` enforces bounds through scipy, which has no ball. Passing the
        box alone would give an unconstrained-in-the-corners fit from a clipper
        chosen precisely to constrain them."""
        search = af.LBFGS(clipper=ClipperPriorBoxJoint())

        with pytest.raises(exc.SearchException):
            search._bounds_from(model=_ball_model())

    def test__lbfgs_accepts_it_on_a_model_that_declares_no_ball(self):
        """The refusal is keyed on the model, not the clipper's type: on a model
        with no declared geometry the joint clipper is the plain box clipper, so a
        pipeline may configure it once without breaking its scipy searches."""
        model = _model(
            alpha=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
            beta=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
        )
        search = af.LBFGS(clipper=ClipperPriorBoxJoint())

        bounds = search._bounds_from(model=model)

        assert bounds.lb == pytest.approx(search.clipper.bounds_from_model(model)[0])

    def test__a_scaler_is_accepted_at_construction_because_no_model_exists_yet(self):
        """Whether a ball composes with a map is a question about the MODEL --
        which pairs carry a ball, and how the map treats each of them -- and no
        model exists at construction. The same triple is well defined on one
        model and undefinable on the next, so construction cannot answer it."""
        search = af.MultiStartAdam(
            clipper=ClipperPriorBoxJoint(), scaler=af.ScalerPriorWidth()
        )

        assert isinstance(search.clipper, ClipperPriorBoxJoint)
        assert isinstance(search.scaler, af.ScalerPriorWidth)

    def test__a_bijector_is_accepted_at_construction_too(self):
        search = af.MultiStartAdam(
            clipper=ClipperPriorBoxJoint(), bijector=BijectorAuto()
        )

        assert isinstance(search.clipper, ClipperPriorBoxJoint)
        assert isinstance(search.bijector, BijectorAuto)

    def test__an_unresolvable_map_is_refused_at_model_resolution(self):
        """Refused before the first STEP, not at construction: a multi-hour fit
        must still not die a minute in on a combination that was wrong before it
        started, so the check runs once against the model at the top of `_fit`,
        before anything is compiled or evaluated."""
        model = _ball_model()
        clipper = ClipperPriorBoxJoint()

        with pytest.raises(ValueError):
            clipper.pairs_in_stepped_coordinates(
                pairs=model.ball_constraint_index_pairs(),
                bijector=BijectorLogit().from_model(model=model),
            )

    def test__a_resolvable_map_passes_model_resolution(self):
        model = _ball_model()
        model.intensity = af.LogUniformPrior(lower_limit=1.0e-3, upper_limit=1.0e3)
        clipper = ClipperPriorBoxJoint()

        pairs = clipper.pairs_in_stepped_coordinates(
            pairs=model.ball_constraint_index_pairs(),
            bijector=BijectorPerPath({"intensity": "log"}).from_model(model=model),
        )

        assert pairs == [(0, 1, BALL_RADIUS)]

    def test__the_plain_box_clipper_still_composes_with_a_scaler(self):
        """The refusal is scoped to the joint clipper: nothing about the existing
        box-plus-scaler combination changes."""
        search = af.MultiStartAdam(
            clipper=ClipperPriorBox(), scaler=af.ScalerPriorWidth()
        )

        assert isinstance(search.clipper, ClipperPriorBox)
        assert isinstance(search.scaler, af.ScalerPriorWidth)
