import importlib.util

import numpy as np
import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__explicit_params():

    search = af.Zeus(
        nwalkers=51,
        nsteps=2001,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=False,
            check_size=101,
            required_length=51,
            change_threshold=0.02
        ),
        tune=False,
        number_of_cores=2,
    )

    assert search.nwalkers == 51
    assert search.nsteps == 2001
    assert search.tune is False
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.auto_correlation_settings.check_for_convergence is False
    assert search.auto_correlation_settings.check_size == 101
    assert search.auto_correlation_settings.required_length == 51
    assert search.auto_correlation_settings.change_threshold == 0.02
    assert search.number_of_cores == 2

    search = af.Zeus()

    assert search.nwalkers == 50
    assert search.nsteps == 2000
    assert search.tune is True
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.auto_correlation_settings.check_for_convergence is True
    assert search.auto_correlation_settings.check_size == 100
    assert search.auto_correlation_settings.required_length == 50
    assert search.auto_correlation_settings.change_threshold == 0.01
    assert search.number_of_cores == 1


requires_zeus = pytest.mark.skipif(
    importlib.util.find_spec("zeus") is None,
    reason="requires zeus-mcmc (installed via the [optional] extras)",
)


@requires_zeus
def test__log_posteriors_are_aligned_with_the_thinned_chain(monkeypatch):
    """
    The parameters and the log posteriors must come out of ``zeus`` under the
    *same* ``discard`` / ``thin``, so that sample ``i``'s log likelihood is the
    likelihood of sample ``i``'s parameters.

    Previously the chain was requested after burn-in and thinning while the log
    probabilities were the full, unthinned array, so the two were not in
    correspondence and the `zip` building the log likelihoods silently truncated
    to the shorter of the two (PyAutoFit#1628).

    The assertion below is the invariant rather than a pinned number: the log
    likelihood stored against the maximum likelihood sample must equal the log
    likelihood the `Analysis` computes for that sample's own parameters.
    """
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    np.random.seed(1)

    model = af.Model(af.ex.Gaussian)
    analysis = af.ex.Analysis(
        data=np.full(100, 5.0),
        noise_map=np.full(100, 1.0),
    )

    # Test mode 1 runs the real sampler for 10 steps with 20 walkers, so the
    # auto-correlation `check_size` must be smaller than the chain for the
    # auto-correlation calculation to have samples to work with.
    search = af.Zeus(
        name="zeus_log_prob_alignment",
        unique_tag="log_prob_alignment_test",
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=False,
            check_size=5,
            required_length=2,
        ),
        number_of_cores=1,
    )

    result = search.fit(model=model, analysis=analysis)

    samples = result.samples

    assert len(samples.parameter_lists) == len(samples.log_likelihood_list)
    assert len(samples.parameter_lists) > 0

    instance = samples.max_log_likelihood()

    assert samples.max_log_likelihood_sample.log_likelihood == pytest.approx(
        analysis.log_likelihood_function(instance=instance), rel=1.0e-8
    )
