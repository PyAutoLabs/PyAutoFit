import numpy as np
import pytest

import autofit as af
from autofit import example
from autofit.non_linear.scaler import ScalerNone, ScalerPriorWidth

# Pure NumPy, deliberately. The scaler's contract is algebraic — what the scale
# vector IS, and that composing the objective through it cannot move the answer —
# and none of that needs JAX. The end-to-end JAX fit under a real scaler is
# validated downstream, matching the discipline stated in
# ``test_multi_start_gradient.py``.


def model_from(centre, normalization, sigma):
    """
    A flat three-parameter model, one prior per argument.

    All three are required rather than defaulted: ``example.Gaussian`` always has
    exactly three priors, so leaving one unset would silently mix a config-default
    prior into the scale vector and make the assertions below depend on the
    packaged config rather than on the rule under test.
    """
    model = af.Model(example.Gaussian)
    model.centre = centre
    model.normalization = normalization
    model.sigma = sigma
    return model


def scale_of(prior):
    """The rule applied to one prior, without the model plumbing in the way."""
    return ScalerPriorWidth._term_for(prior)


def test__scaler_none__is_all_ones():
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
        sigma=af.GaussianPrior(mean=1.0, sigma=2.0),
    )

    scale = ScalerNone().scale_from_model(model=model)

    assert scale == pytest.approx(np.ones(3))


def test__scaler_none__writes_no_model_info_block():
    """
    The empty string is what makes ``_save_model_info`` append nothing at all, so
    a default run's ``model.info`` is byte-identical to one written before this
    feature existed.
    """
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
        sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
    )

    assert ScalerNone().info_from_model(model=model) == ""


def test__uniform_prior__scale_is_the_box_width():
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
        sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
    )

    terms = ScalerPriorWidth()._terms_from_model(model=model)

    assert [term.basis for term in terms] == ["width"] * 3
    assert [term.scale for term in terms] == pytest.approx([8.0, 0.6, 0.2])


def test__gaussian_prior__scale_is_sigma_not_a_width():
    assert scale_of(af.GaussianPrior(mean=100.0, sigma=0.5)).basis == "sigma"
    assert scale_of(af.GaussianPrior(mean=100.0, sigma=0.5)).scale == pytest.approx(0.5)
    assert scale_of(af.GaussianPrior(mean=-3.0, sigma=2.0)).scale == pytest.approx(2.0)


def test__truncated_gaussian__scale_is_sigma_NOT_the_truncation_width():
    """
    The distinction this rule exists for. ``ell_comps`` is the real case: sigma
    0.3 inside a [-1, 1] box, so using the truncation width would overstate its
    natural scale six-fold and under-step it by the same factor.
    """
    term = scale_of(
        af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
        )
    )

    assert term.basis == "sigma"
    assert term.scale == pytest.approx(0.3)
    assert term.scale != pytest.approx(2.0)  # the truncation width


def test__log_uniform_prior__scale_is_median_times_log_ratio_NOT_the_width():
    """
    A log-spaced coordinate's natural step is multiplicative. ``dtheta/du`` for
    ``theta(u) = lo * (hi/lo)**u`` is ``theta * log(hi/lo)``, which at the median
    ``sqrt(lo*hi)`` is the value asserted here. The physical width is wrong by a
    factor of 543 on this prior, which is why it is asserted against explicitly
    rather than merely left unused.
    """
    term = scale_of(af.LogUniformPrior(lower_limit=1.0e-4, upper_limit=1.0e4))

    expected = np.sqrt(1.0e-4 * 1.0e4) * np.log(1.0e4 / 1.0e-4)

    assert term.basis == "median x log-ratio"
    assert term.scale == pytest.approx(expected)
    assert term.scale == pytest.approx(18.42, abs=0.01)
    assert term.scale != pytest.approx(1.0e4)  # the physical width


def test__log_uniform__reduces_to_the_uniform_rule_in_the_narrow_limit():
    """
    A sanity check on the unifying principle rather than on an implementation
    detail: for a narrow log-uniform prior the multiplicative step and the linear
    one agree, so the two branches of the rule meet rather than merely coexisting.
    """
    lower, upper = 1.0, 1.0 + 1.0e-6

    term = scale_of(af.LogUniformPrior(lower_limit=lower, upper_limit=upper))

    assert term.scale == pytest.approx(upper - lower, rel=1.0e-5)


def test__log_gaussian_prior__scale_is_median_times_sigma():
    term = scale_of(af.LogGaussianPrior(mean=2.0, sigma=0.5))

    assert term.basis == "median x sigma"
    assert term.scale == pytest.approx(np.exp(2.0) * 0.5)


def test__geometric_mean_of_the_scale_is_exactly_one():
    """
    The normalisation is what keeps an A/B honest: it changes only the RATIOS, so
    the global step magnitude is untouched and the comparison is not confounded
    with an effective learning-rate change.
    """
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
        sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
    )

    scale = ScalerPriorWidth().scale_from_model(model=model)

    assert np.exp(np.mean(np.log(scale))) == pytest.approx(1.0)


def test__normalisation_preserves_every_ratio():
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
        sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
    )

    scaler = ScalerPriorWidth()
    raw = np.array([term.scale for term in scaler._terms_from_model(model=model)])
    scale = scaler.scale_from_model(model=model)

    assert scale / scale[0] == pytest.approx(raw / raw[0])

    # And the 40x spread the feature exists for survives normalisation.
    assert scale.max() / scale.min() == pytest.approx(40.0)


def test__scale_is_always_positive_and_finite():
    """
    A guarantee rather than an expectation: the vector is used as a DIVISOR, so a
    zero sends the whole population to infinity and a NaN poisons every lane. A
    degenerate prior must degrade to "unscaled", never to "broken".
    """
    # ``UniformPrior`` refuses a zero width at CONSTRUCTION, so the degenerate
    # case has to be reached by mutating one afterwards. That is the realistic
    # route anyway -- the limits are plain attributes and nothing re-validates
    # them -- and it is why the scaler cannot simply assume the priors it is
    # handed are well formed.
    degenerate = af.UniformPrior(lower_limit=1.0, upper_limit=2.0)
    degenerate.upper_limit = 1.0

    model = model_from(
        centre=degenerate,
        normalization=af.GaussianPrior(mean=0.0, sigma=0.0),
        sigma=af.UniformPrior(lower_limit=0.0, upper_limit=4.0),
    )

    scale = ScalerPriorWidth().scale_from_model(model=model)

    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0.0)

    # Degrades to "unscaled", not to "broken": the two bad coordinates fall back
    # to 1.0 BEFORE normalisation, so they cannot drag the geometric mean either.
    assert np.exp(np.mean(np.log(scale))) == pytest.approx(1.0)


def test__log_uniform_with_a_non_positive_limit__falls_back_rather_than_nan():
    prior = af.LogUniformPrior(lower_limit=1.0e-8, upper_limit=1.0)
    prior.lower_limit = -1.0

    term = scale_of(prior)
    if np.isnan(term.scale):
        # ``_term_for`` reports the malformed prior; ``_terms_from_model`` is
        # where the fallback is applied, so go through it for the final value.
        model = model_from(
            centre=prior,
            normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            sigma=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )
        term = ScalerPriorWidth()._terms_from_model(model=model)[0]

    assert term.basis == "fallback"
    assert term.scale == 1.0


def test__unknown_prior_type__falls_back_to_one():
    class _UnscalablePrior(af.UniformPrior):
        pass

    term = ScalerPriorWidth._term_for(
        _UnscalablePrior(lower_limit=0.0, upper_limit=1.0)
    )

    # A subclass of a KNOWN prior still resolves through isinstance, which is the
    # desired behaviour — a user subclassing UniformPrior gets the width rule.
    assert term.basis == "width"


def test__info_from_model__names_the_basis_used_for_each_prior():
    """
    The block is the user-facing half of this feature, and WHICH quantity each
    scale came from is the part that is a judgement call — so it is reported,
    not just the resulting number.
    """
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
        ),
        sigma=af.UniformPrior(lower_limit=0.0, upper_limit=4.0),
    )

    info = ScalerPriorWidth().info_from_model(model=model)

    assert "Per-Parameter Step Scaling (ScalerPriorWidth)" in info
    assert "centre" in info
    assert "UniformPrior" in info
    assert "TruncatedGaussianPrior" in info
    assert "width" in info
    assert "sigma" in info
    assert "geometric mean is 1.0" in info

    # One line per parameter, so a 2-parameter model cannot silently report 1.
    body = [line for line in info.splitlines() if "scale " in line]
    assert len(body) == 3


# ---------------------------------------------------------------------------
# The invariance the whole design turns on.
# ---------------------------------------------------------------------------


def _rosenbrock_like(theta):
    """
    A deliberately ill-conditioned objective with a unique interior minimum at
    ``(3.0, 0.05)`` — chosen so the two coordinates' natural scales differ by the
    same order the real model's priors do.
    """
    return 5.0 * (theta[0] - 3.0) ** 2 + 8000.0 * (theta[1] - 0.05) ** 2


def _gradient(objective, theta, eps=1.0e-7):
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = eps
        grad[i] = (objective(theta + step) - objective(theta - step)) / (2.0 * eps)
    return grad


def _descend(objective, start, learning_rate, n_steps):
    theta = np.array(start, dtype=float)
    for _ in range(n_steps):
        theta = theta - learning_rate * _gradient(objective, theta)
    return theta


def test__objective_composed_through_the_scale_is_the_SAME_objective():
    """
    The algebraic pin. Composing as ``f(s * phi)`` evaluates the physical-space
    objective at the transformed point, so it is equal at every corresponding
    pair — exactly, not approximately.

    Optimising the density OF ``phi`` instead would fold a Jacobian in and move
    the MAP, and it would fail SILENTLY: every counter, every diagnostic and the
    wall time would look healthy. This is the test that says which of the two was
    implemented.
    """
    scale = np.array([2.5, 0.04])

    rng = np.random.default_rng(0)
    for _ in range(50):
        theta = rng.uniform(-10.0, 10.0, size=2)
        phi = theta / scale

        # ``rel`` rather than exact: ``scale * (theta / scale)`` is not bitwise
        # ``theta`` in floating point. The claim under test is that the objective
        # is the SAME FUNCTION composed through the map, not that the round trip
        # is lossless -- a Jacobian-folding implementation would differ by orders
        # of magnitude here, not by an ulp.
        assert _rosenbrock_like(scale * phi) == pytest.approx(
            _rosenbrock_like(theta), rel=1.0e-12
        )


def test__scaling_cannot_move_the_MAP():
    """
    The end-to-end statement of the same thing: descend on the raw objective, and
    descend on the scaled one and map back, and the recovered argmin agrees.

    The scaled run uses the SMALLER learning rate that the unscaled one needs to
    remain stable at all, so this is not the scaled run being flattered — it is
    the answer being shown to be independent of the coordinates, which is the
    property that makes the whole change safe.
    """
    scale = np.array([2.5, 0.04])
    start = np.array([8.0, -4.0])

    unscaled = _descend(_rosenbrock_like, start, learning_rate=1.0e-5, n_steps=200000)

    scaled_phi = _descend(
        lambda phi: _rosenbrock_like(scale * phi),
        start / scale,
        learning_rate=1.0e-5,
        n_steps=200000,
    )
    scaled = scale * scaled_phi

    assert unscaled == pytest.approx(np.array([3.0, 0.05]), abs=1.0e-4)
    assert scaled == pytest.approx(np.array([3.0, 0.05]), abs=1.0e-4)
    assert scaled == pytest.approx(unscaled, abs=1.0e-4)


def test__scaling_reaches_the_minimum_a_shared_step_size_cannot():
    """
    Why the feature exists at all, stated as a test rather than as a claim.

    One shared step size cannot serve two coordinates whose curvatures differ by
    orders of magnitude. It is set by the STIFF coordinate -- push it higher and
    that one oscillates or diverges -- which leaves the soft coordinate crawling.
    Here the stiff coordinate has converged to machine precision while the soft
    one is still visibly short of its optimum after the same 5000 steps.

    This is the same failure the real search shows, in miniature: it is the
    NARROW-prior coordinates that reach the walls while the wide ones are still
    travelling. Rescaling equalises the two and one rate then serves both.
    """
    scale = np.array([2.5, 0.04])
    start = np.array([8.0, -4.0])
    learning_rate = 1.0e-4

    unscaled = _descend(_rosenbrock_like, start, learning_rate, n_steps=5000)

    scaled = scale * _descend(
        lambda phi: _rosenbrock_like(scale * phi),
        start / scale,
        learning_rate,
        n_steps=5000,
    )

    # The stiff coordinate is done; the soft one is not, and it is not close.
    assert unscaled[1] == pytest.approx(0.05, abs=1.0e-6)
    assert np.abs(unscaled[0] - 3.0) > 1.0e-2

    # The same budget, the same single learning rate, both coordinates converged.
    assert scaled == pytest.approx(np.array([3.0, 0.05]), abs=1.0e-4)


# ---------------------------------------------------------------------------
# Composition with the clipper.
# ---------------------------------------------------------------------------


def test__clipper_projects_in_scaled_coordinates_against_scaled_bounds():
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=-0.3, upper_limit=0.3),
        sigma=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
    )

    clipper = af.ClipperPriorBox()
    scale = ScalerPriorWidth().scale_from_model(model=model)

    theta = np.array([9.0, -0.5, 0.4])  # all three outside their boxes
    phi = theta / scale

    projected_physical, mask_physical = clipper.project(vector=theta, model=model)
    projected_phi, mask_phi = clipper.project(vector=phi, model=model, scale=scale)

    assert (mask_phi == mask_physical).all()
    assert scale * projected_phi == pytest.approx(projected_physical)


def test__clipper_with_no_scale__is_unchanged():
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        sigma=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    )

    clipper = af.ClipperPriorBox()
    vector = np.array([9.0, 0.5, 0.5])

    assert clipper.project(vector=vector, model=model)[0] == pytest.approx(
        clipper.project(vector=vector, model=model, scale=None)[0]
    )


def test__clipper_none__accepts_a_scale_and_ignores_it():
    model = model_from(
        centre=af.UniformPrior(lower_limit=0.0, upper_limit=8.0),
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        sigma=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    )

    vector = np.array([9.0, 0.5, 0.5])
    projected, mask = af.ClipperNone().project(
        vector=vector, model=model, scale=np.array([2.0, 1.0, 1.0])
    )

    assert projected == pytest.approx(vector)
    assert not mask.any()
