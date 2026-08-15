import numpy as np
import pytest

from autonerves import conf

import autofit as af
from autofit import exc
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.summary import SamplesSummary


@pytest.fixture(name="check_likelihood_function_on")
def make_check_likelihood_function_on():
    """
    `test_autofit/config/general.yaml` disables the resume sanity check for the
    suite at large. These tests are about that check, so switch it on for their
    duration and restore it afterwards.
    """
    original = conf.instance["general"]["test"]["check_likelihood_function"]
    conf.instance["general"]["test"]["check_likelihood_function"] = True
    yield
    conf.instance["general"]["test"]["check_likelihood_function"] = original


@pytest.fixture(name="model")
def make_model():
    """
    A model whose priors give a *non-zero* summed log prior, so the log-prior leg
    of the figure-of-merit conversion is actually exercised. With the default
    uniform priors the log prior is 0.0 and a log-posterior FoM is numerically
    indistinguishable from a log-likelihood FoM.
    """
    model = af.Model(af.ex.Gaussian)
    model.centre = af.GaussianPrior(mean=50.0, sigma=20.0)
    model.normalization = af.GaussianPrior(mean=1.0, sigma=0.5)
    model.sigma = af.GaussianPrior(mean=5.0, sigma=2.0)
    return model


@pytest.fixture(name="analysis")
def make_analysis():
    return af.ex.Analysis(data=np.ones(20), noise_map=np.ones(20) * 0.1)


# Deliberately off the prior means: a `GaussianPrior` has log prior 0.0 at its mean,
# which would collapse the log-posterior conventions onto the log-likelihood one.
PARAMETERS = [10.0, 2.0, 3.0]


def _paths_with_stored_summary(model, analysis, tmp_path, name):
    """
    Write out the `samples_summary` a previous run would have checkpointed, holding
    the *true* log likelihood of `PARAMETERS` (which is what `Sample.log_likelihood`
    always stores, whatever figure-of-merit convention the search itself uses).
    """
    paths = af.DirectoryPaths(name=name, path_prefix=str(tmp_path))
    paths.model = model

    log_likelihood = float(Fitness(model=model, analysis=analysis).call(PARAMETERS))
    log_prior = float(
        np.sum(model.log_prior_list_from_vector(vector=PARAMETERS, xp=np))
    )
    assert not np.isclose(log_prior, 0.0), "fixture must exercise the log-prior leg"

    sample = Sample.from_lists(
        model=model,
        parameter_lists=[PARAMETERS],
        log_likelihood_list=[log_likelihood],
        log_prior_list=[log_prior],
        weight_list=[1.0],
    )[0]
    paths.save_samples_summary(
        samples_summary=SamplesSummary(max_log_likelihood_sample=sample, model=model)
    )

    return paths, log_likelihood


@pytest.mark.parametrize(
    "fom_is_log_likelihood, convert_to_chi_squared",
    [
        (True, False),  # nested samplers (Dynesty, Nautilus)
        (False, False),  # log-posterior searches (Emcee, Zeus, NUTS, Drawer)
        (False, True),  # chi-squared searches (MultiStartAdam/Prodigy, LBFGS)
    ],
)
def test_check_log_likelihood_passes_on_resume_for_every_fom_convention(
    model,
    analysis,
    tmp_path,
    check_likelihood_function_on,
    fom_is_log_likelihood,
    convert_to_chi_squared,
):
    """
    Resuming a search whose likelihood function has *not* changed must not raise,
    whatever figure-of-merit convention that search uses.

    Regression for the bug where `check_log_likelihood` compared the stored log
    likelihood against `fitness(...)`, i.e. against the search's own FoM. For a
    `MultiStartGradient` (`-2 * log_posterior`) that made the fresh value roughly
    `-2x` the stored one, so every resume of a killed mid-run search died with a
    `SearchException` on an unchanged likelihood function.
    """
    paths, _ = _paths_with_stored_summary(
        model=model,
        analysis=analysis,
        tmp_path=tmp_path,
        name=f"resume_{fom_is_log_likelihood}_{convert_to_chi_squared}",
    )

    # `check_log_likelihood` runs from `Fitness.__init__` whenever `paths` is set.
    Fitness(
        model=model,
        analysis=analysis,
        paths=paths,
        fom_is_log_likelihood=fom_is_log_likelihood,
        resample_figure_of_merit=-np.inf,
        convert_to_chi_squared=convert_to_chi_squared,
    )


@pytest.mark.parametrize(
    "fom_is_log_likelihood, convert_to_chi_squared",
    [(True, False), (False, False), (False, True)],
)
def test_check_log_likelihood_still_raises_when_likelihood_function_changes(
    model,
    analysis,
    tmp_path,
    check_likelihood_function_on,
    fom_is_log_likelihood,
    convert_to_chi_squared,
):
    """
    The guard exists to catch a genuinely changed likelihood function between runs.
    Converting the fresh value out of the FoM convention must not defeat that: a
    resume against a shifted likelihood still has to raise, in every convention.
    """
    paths, _ = _paths_with_stored_summary(
        model=model,
        analysis=analysis,
        tmp_path=tmp_path,
        name=f"changed_{fom_is_log_likelihood}_{convert_to_chi_squared}",
    )

    # Stand in for "the source changed underneath the resume" by shifting the data
    # the analysis fits, which moves the log likelihood and nothing else.
    changed_analysis = af.ex.Analysis(
        data=np.ones(20) * 2.0, noise_map=np.ones(20) * 0.1
    )

    with pytest.raises(exc.SearchException):
        Fitness(
            model=model,
            analysis=changed_analysis,
            paths=paths,
            fom_is_log_likelihood=fom_is_log_likelihood,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=convert_to_chi_squared,
        )


@pytest.mark.parametrize(
    "fom_is_log_likelihood, convert_to_chi_squared",
    [(True, False), (False, False), (False, True)],
)
def test_log_likelihood_from_inverts_the_figure_of_merit_convention(
    model, analysis, fom_is_log_likelihood, convert_to_chi_squared
):
    """
    `log_likelihood_from` is the exact inverse of the FoM mapping applied by `call`,
    and is the single conversion both `call_wrap` and `check_log_likelihood` use.
    """
    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=fom_is_log_likelihood,
        resample_figure_of_merit=-np.inf,
        convert_to_chi_squared=convert_to_chi_squared,
    )

    expected = float(Fitness(model=model, analysis=analysis).call(PARAMETERS))

    log_likelihood = fitness.log_likelihood_from(
        figure_of_merit=fitness.call(PARAMETERS), parameters=PARAMETERS
    )

    assert log_likelihood == pytest.approx(expected)


def test_call_wrap_returns_the_figure_of_merit_not_the_log_likelihood(model, analysis):
    """
    `call_wrap` derives the log likelihood for its quick-update / history bookkeeping
    but must still return the FoM the search consumes. Guards against the conversion
    leaking into the return value (the log-prior subtraction used to be an in-place
    `-=`, which mutated the returned array when the FoM was not chi-squared).
    """
    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=False,
        resample_figure_of_merit=-np.inf,
        convert_to_chi_squared=False,
    )

    assert float(fitness.call_wrap(np.array(PARAMETERS))) == pytest.approx(
        float(fitness.call(PARAMETERS))
    )
