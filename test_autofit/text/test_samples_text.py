from pathlib import Path

import pytest

import autofit as af

from autofit.non_linear.samples import Sample
from autofit import SamplesStored
from autofit.text import samples_text

text_path = Path(__file__).resolve().parent / "files" / "samples"


@pytest.fixture(name="model")
def make_model():
    return af.ModelMapper(mock_class=af.m.MockClassx2)


@pytest.fixture(name="samples")
def make_samples(model):
    parameters = [[1.0, 2.0], [1.2, 2.2]]

    log_likelihood_list = [1.0, 0.0]

    return SamplesStored(
        model=model,
        sample_list=Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model
        ),
    )


def test__summary(samples):
    results_at_sigma = samples_text.summary(samples=samples, sigma=3.0)

    assert "one       1.00 (1.00, 1.20)" in results_at_sigma
    assert "two       2.00 (2.00, 2.20)" in results_at_sigma


def test__latex(samples):
    latex_results_at_sigma = samples_text.latex(samples=samples, sigma=3.0)

    assert r"one_label^{\rm{o}} = 1.00^{+0.20}_{-0.00} & " in latex_results_at_sigma
    assert r"two_label^{\rm{o}} = 2.00^{+0.20}_{-0.00}" in latex_results_at_sigma

    latex_results_at_sigma = samples_text.latex(samples=samples, sigma=3.0, include_quickmath=True)

    assert r"$one_label^{\rm{o}} = 1.00^{+0.20}_{-0.00}$ & " in latex_results_at_sigma
    assert r"$two_label^{\rm{o}} = 2.00^{+0.20}_{-0.00}$" in latex_results_at_sigma

    model = af.ModelMapper(mock_class=af.m.MockClassx2FormatExp)

    parameters = [[1.0, 200.0], [1.2, 200.0]]

    log_likelihood_list = [1.0, 0.0]

    samples_exp = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model
        ),
    )

    latex_results_at_sigma = samples_text.latex(samples=samples_exp, sigma=3.0, include_quickmath=True)

    assert r"$one_label^{\rm{o}} = 1.00^{+0.20}_{-0.00}$ & " in latex_results_at_sigma
    assert r"$one_label^{\rm{o}} = 1.00^{+0.20}_{-0.00}$ & $2.00^{+0.00}_{-0.00} \times 10^{2}$" in latex_results_at_sigma