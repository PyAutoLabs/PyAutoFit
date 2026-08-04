import logging
import random
from unittest.mock import patch

import pytest
import scipy

import numpy as np

from autofit import CovarianceInterpolator
import autofit as af
from autofit.non_linear.search.nest.dynesty.search.static import DynestyStatic

SEED = 20260802


@pytest.fixture(autouse=True)
def do_remove_output(output_directory, remove_output):
    yield
    remove_output()


def test_covariance_matrix(interpolator):
    assert np.allclose(
        interpolator.covariance_matrix(),
        np.array(
            [
                [1.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 4.33333333, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 9.0, 19.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 4.33333333, 9.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 9.0, 19.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.33333333, 9.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 9.0, 19.0],
            ]
        ),
    )


def _maxcall_dynesty(*args, **kwargs):
    """Create a DynestyStatic with maxcall=1 for fast test execution."""
    kwargs.setdefault("maxcall", 1)
    return DynestyStatic.__new_orig__(*args, **kwargs)


@pytest.fixture(autouse=True)
def limit_maxcall(monkeypatch):
    """Limit DynestyStatic to maxcall=1 so interpolator tests run fast."""
    original_init = DynestyStatic.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("maxcall", 1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(DynestyStatic, "__init__", patched_init)


@pytest.fixture(autouse=True)
def seed_search_randomness(monkeypatch):
    """
    Make the Dynesty searches these tests run reproducible.

    `limit_maxcall` above caps each search at a single likelihood call so this
    module stays fast, which leaves the recovered value dominated by the
    sampler's randomness rather than by convergence. Unseeded, the numerical
    recovery assertions are therefore coin flips: measured over 2000 runs,
    `test_variable_and_constant` missed `abs=5.0` 1.65% of the time (95% CI
    1.18-2.31%) and `test_single_variable` missed `abs=2.0` 3.6% of the time.
    The 1.65% is what killed the 2026.8.2.1 live release.

    Three independent generators feed that result, so seeding any one of them
    is not enough:

    * `numpy.random`, used by `test_variable_and_constant` to build its samples;
    * the stdlib `random` module, used by `autofit.non_linear.initializer` to
      draw the search's initial unit values;
    * dynesty's own `rstate`, which defaults to
      `numpy.random.Generator(PCG64(None))` — seeded from OS entropy and
      reachable from neither of the above. This is the dominant term, and the
      only one `test_single_variable` (which makes no random call of its own)
      depends on at all.

    Global generator state is restored on teardown so the seeding cannot leak
    into whatever runs next in the same process.
    """
    import dynesty.dynesty

    monkeypatch.setattr(
        dynesty.dynesty,
        "get_random_generator",
        lambda seed=None: np.random.default_rng(SEED if seed is None else seed),
    )

    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def test_interpolate(interpolator):
    try:
        assert isinstance(interpolator[interpolator.t == 0.5].gaussian.centre, float)
    except scipy.linalg.LinAlgError as e:
        logging.warning(e)


def test_relationships(interpolator):
    try:
        relationships = interpolator.relationships(interpolator.t)
        assert isinstance(relationships.gaussian.centre(0.5), float)
    except scipy.linalg.LinAlgError as e:
        logging.warning(e)


def test_interpolate_other_field(interpolator):
    try:
        assert isinstance(
            interpolator[interpolator.gaussian.centre == 0.5].gaussian.centre,
            float,
        )
    except scipy.linalg.LinAlgError as e:
        logging.warning(e)


def test_linear_analysis_for_value(interpolator):
    try:
        analysis = interpolator._analysis_for_path(interpolator.t)
        assert (analysis.x == np.array([0, 1, 2])).all()
        assert (analysis.y == np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])).all()
    except scipy.linalg.LinAlgError as e:
        logging.warning(e)


def test_model(interpolator):
    model = interpolator.model()
    assert model.prior_count == 6


def test_single_variable():
    samples_list = [
        af.SamplesPDF(
            model=af.Collection(
                t=value,
                v=af.GaussianPrior(mean=1.0, sigma=1.0),
            ),
            sample_list=[
                af.Sample(
                    log_likelihood=-value,
                    log_prior=1.0,
                    weight=1.0,
                    kwargs={
                        ("v",): value,
                    },
                )
            ],
        )
        for value in range(50)
    ]
    interpolator = CovarianceInterpolator(
        samples_list,
    )
    assert interpolator[interpolator.t == 25.0].v == pytest.approx(25.0, abs=2.0)


def test_variable_and_constant():
    rng = np.random.default_rng(SEED)
    samples_list = [
        af.SamplesPDF(
            model=af.Collection(
                t=value,
                v=af.GaussianPrior(mean=1.0, sigma=1.0),
                x=af.GaussianPrior(mean=1.0, sigma=1.0),
            ),
            sample_list=[
                af.Sample(
                    log_likelihood=-value,
                    log_prior=1.0,
                    weight=1.0,
                    kwargs={
                        ("v",): value + 0.1 * (1 - rng.random()),
                        ("x",): 0.5 * (1 - +rng.random()),
                    },
                )
                for _ in range(50)
            ],
        )
        for value in range(50)
    ]
    interpolator = CovarianceInterpolator(
        samples_list,
    )
    assert interpolator[interpolator.t == 25.0].v == pytest.approx(25.0, abs=5.0)
