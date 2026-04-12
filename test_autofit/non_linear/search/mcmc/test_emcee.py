import pytest

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__explicit_params():

    search = af.Emcee(
        nwalkers=51,
        nsteps=2001,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=False,
            check_size=101,
            required_length=51,
            change_threshold=0.02
        ),
        number_of_cores=2,
    )

    assert search.nwalkers == 51
    assert search.nsteps == 2001
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.auto_correlation_settings.check_for_convergence is False
    assert search.auto_correlation_settings.check_size == 101
    assert search.auto_correlation_settings.required_length == 51
    assert search.auto_correlation_settings.change_threshold == 0.02
    assert search.number_of_cores == 2

    search = af.Emcee()

    assert search.nwalkers == 50
    assert search.nsteps == 2000
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.auto_correlation_settings.check_for_convergence is True
    assert search.auto_correlation_settings.check_size == 100
    assert search.auto_correlation_settings.required_length == 50
    assert search.auto_correlation_settings.change_threshold == 0.01
    assert search.number_of_cores == 1
