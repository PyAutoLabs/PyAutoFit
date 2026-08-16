import numpy as np
import pytest

import autofit as af
from autofit import exc
from autofit.non_linear.clipper import ClipperNone, ClipperPriorBox


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

    def test__log_gaussian_prior__lower_bound_is_declared_by_the_clipper(self):
        """
        LogGaussianPrior reports (-inf, inf) because its TransformedMessage is never
        given limits, yet ``log_prior_from_value`` is -inf for value <= 0. The
        clipper declares the real (0, inf) support, strictly, so the projected value
        lands *above* zero rather than on it.
        """
        prior = af.LogGaussianPrior(mean=0.0, sigma=1.0)

        assert prior.lower_limit == -np.inf

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
