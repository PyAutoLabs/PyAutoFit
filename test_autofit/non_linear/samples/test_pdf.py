import os

import numpy as np
import pytest

from autoconf.conf import with_config

import autofit as af

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="samples_x5")
def make_samples_x5():
    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    return af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )


@pytest.fixture(autouse=True)
def remove_csv_output():
    yield
    try:
        os.remove("samples.csv")
    except FileNotFoundError:
        pass
    try:
        os.remove("covariance.csv")
    except FileNotFoundError:
        pass


def test_save_covariance_matrix(samples_x5):
    samples_x5.save_covariance_matrix("covariance.csv")
    with open("covariance.csv") as f:
        string = f.read()
        print(string)

    assert (
        string
        == """8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01
8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01
8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01
8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01,8.820000000000000284e+01
"""
    )


def test__from_csv_table(samples_x5):
    filename = "samples.csv"
    samples_x5.write_table(filename=filename)

    samples_x5 = af.SamplesPDF.from_table(filename=filename, model=samples_x5.model)

    assert samples_x5.parameter_lists == [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]
    assert samples_x5.log_likelihood_list == [1.0, 2.0, 3.0, 10.0, 5.0]
    assert samples_x5.log_prior_list == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert samples_x5.log_posterior_list == [1.0, 2.0, 3.0, 10.0, 5.0]
    assert samples_x5.weight_list == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_format(samples_x5):
    filename = "samples.csv"
    samples_x5.write_table(filename=filename)

    with open(filename) as f:
        text = f.read()

    assert (
        text
        == """mock_class_1.one,mock_class_1.two,mock_class_1.three,mock_class_1.four,log_likelihood,log_prior,log_posterior,weight
             0.0,             1.0,               2.0,              3.0,           1.0,      0.0,          1.0,   1.0
             0.0,             1.0,               2.0,              3.0,           2.0,      0.0,          2.0,   1.0
             0.0,             1.0,               2.0,              3.0,           3.0,      0.0,          3.0,   1.0
            21.0,            22.0,              23.0,             24.0,          10.0,      0.0,         10.0,   1.0
             0.0,             1.0,               2.0,              3.0,           5.0,      0.0,          5.0,   1.0
"""
    )


def test__median_pdf__converged():
    parameters = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [0.9, 1.9],
        [1.1, 2.1],
    ]

    log_likelihood_list = 10 * [0.1]
    weight_list = 10 * [0.1]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is True

    median_pdf = samples_x5.median_pdf(as_instance=False)

    assert median_pdf[0] == pytest.approx(1.0, 1.0e-4)
    assert median_pdf[1] == pytest.approx(2.0, 1.0e-4)

    median_pdf_instance = samples_x5.median_pdf(as_instance=True)

    assert median_pdf_instance.mock_class.one == pytest.approx(1.0, 1e-1)
    assert median_pdf_instance.mock_class.two == pytest.approx(2.0, 1e-1)


def test__median_pdf__unconverged():
    parameters = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.1, 2.1],
        [0.9, 1.9],
    ]

    log_likelihood_list = 9 * [0.0] + [1.0]
    weight_list = 9 * [0.0] + [1.0]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is False

    median_pdf = samples_x5.median_pdf(as_instance=False)

    assert median_pdf[0] == pytest.approx(0.9, 1.0e-4)
    assert median_pdf[1] == pytest.approx(1.9, 1.0e-4)


@with_config("general", "model", value=True)
def test__converged__vector_and_instance_at_upper_and_lower_sigma():
    parameters = [
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.0, 0.5],
        [0.2, 0.3],
    ]

    log_likelihood_list = list(range(10))

    weight_list = 10 * [0.1]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is True

    values = samples_x5.values_at_sigma(sigma=3.0, as_instance=False)

    assert values[0] == pytest.approx((0.00121, 0.19878), 1e-1)
    assert values[1] == pytest.approx((0.30121, 0.49878), 1e-1)

    values = samples_x5.values_at_upper_sigma(sigma=3.0, as_instance=False)

    assert values[0] == pytest.approx(0.19757, 1e-1)
    assert values[1] == pytest.approx(0.49757, 1e-1)

    values = samples_x5.values_at_lower_sigma(sigma=3.0, as_instance=False)

    assert values[0] == pytest.approx(0.00121, 1e-1)
    assert values[1] == pytest.approx(0.30121, 1e-1)

    values = samples_x5.values_at_sigma(sigma=1.0, as_instance=False)

    assert values[0] == pytest.approx((0.1, 0.1), 1e-1)
    assert values[1] == pytest.approx((0.4, 0.4), 1e-1)

    values = samples_x5.values_at_sigma(sigma=1.0)

    assert values.mock_class.one == pytest.approx((0.1, 0.1), 1e-1)
    assert values.mock_class.two == pytest.approx((0.4, 0.4), 1e-1)

    values = samples_x5.values_at_upper_sigma(sigma=3.0)

    assert values.mock_class.one == pytest.approx(0.19757, 1e-1)
    assert values.mock_class.two == pytest.approx(0.49757, 1e-1)

    values = samples_x5.values_at_lower_sigma(sigma=3.0)

    assert values.mock_class.one == pytest.approx(0.00121, 1e-1)
    assert values.mock_class.two == pytest.approx(0.30121, 1e-1)


def test__values_at_sigma__unconverged():
    parameters = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.1, 2.1],
        [0.9, 1.9],
    ]

    log_likelihood_list = 9 * [0.0] + [1.0]
    weight_list = 9 * [0.0] + [1.0]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is False

    values_at_sigma = samples_x5.values_at_sigma(sigma=1.0, as_instance=False)

    assert values_at_sigma[0] == pytest.approx(((0.9, 1.1)), 1e-2)
    assert values_at_sigma[1] == pytest.approx(((1.9, 2.1)), 1e-2)

    values_at_sigma = samples_x5.values_at_sigma(sigma=3.0, as_instance=False)

    assert values_at_sigma[0] == pytest.approx(((0.9, 1.1)), 1e-2)
    assert values_at_sigma[1] == pytest.approx(((1.9, 2.1)), 1e-2)


def test__errors_at__converged():
    parameters = [
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.1, 0.4],
        [0.0, 0.5],
        [0.2, 0.3],
    ]

    log_likelihood_list = list(range(10))

    weight_list = 10 * [0.1]

    model = af.ModelMapper(mock_class=af.m.MockClassx2)
    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    assert samples_x5.pdf_converged is True

    errors = samples_x5.error_magnitudes_at_sigma(sigma=3.0, as_instance=False)

    assert errors == pytest.approx([0.19514, 0.19514], 1e-1)

    errors = samples_x5.errors_at_upper_sigma(sigma=3.0, as_instance=False)

    assert errors == pytest.approx([0.09757, 0.09757], 1e-1)

    errors = samples_x5.errors_at_lower_sigma(sigma=3.0, as_instance=False)

    assert errors == pytest.approx([0.09757, 0.09757], 1e-1)

    errors = samples_x5.errors_at_sigma(sigma=3.0, as_instance=False)
    assert errors[0] == pytest.approx((0.09757, 0.09757), 1e-1)
    assert errors[1] == pytest.approx((0.09757, 0.09757), 1e-1)

    errors = samples_x5.error_magnitudes_at_sigma(sigma=1.0, as_instance=False)

    assert errors == pytest.approx([0.0, 0.0], 1e-1)

    errors_instance = samples_x5.errors_at_sigma(sigma=1.0, as_instance=True)

    assert errors_instance.mock_class.one[0] == pytest.approx(0.0, 1e-1)
    assert errors_instance.mock_class.two[0] == pytest.approx(0.0, 1e-1)

    errors_instance = samples_x5.errors_at_upper_sigma(sigma=3.0, as_instance=True)

    assert errors_instance.mock_class.one == pytest.approx(0.09757, 1e-1)
    assert errors_instance.mock_class.two == pytest.approx(0.09757, 1e-1)

    errors_instance = samples_x5.errors_at_lower_sigma(sigma=3.0, as_instance=True)

    assert errors_instance.mock_class.one == pytest.approx(0.09757, 1e-1)
    assert errors_instance.mock_class.two == pytest.approx(0.09757, 1e-1)


def test__unconverged_sample_size():
    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    log_likelihood_list = 4 * [0.0] + [1.0]
    weight_list = 4 * [0.0] + [1.0]

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=5 * [[]],
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            weight_list=weight_list,
        ),
        samples_info={"unconverged_sample_size": 2},
    )

    assert samples_x5.pdf_converged is False
    assert samples_x5.unconverged_sample_size == 2


def test__offset_values_via_input_values():
    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    parameters = [
        [1.1, 2.1, 3.1, 4.1],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.1],
    ]

    weight_list = [0.3, 0.2, 0.2, 0.2, 0.1]

    log_likelihood_list = list(map(lambda weight: 10.0 * weight, weight_list))

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=10 * [0.0],
            weight_list=weight_list,
        ),
    )

    offset_values = samples_x5.offset_values_via_input_values(
        input_vector=[1.0, 1.0, 2.0, 3.0], as_instance=False
    )

    assert offset_values == pytest.approx([0.0, 1.0, 1.0, 1.025], 1.0e-4)


def test__draw_randomly_via_pdf():
    parameters = [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [21.0, 22.0, 23.0, 24.0],
        [0.0, 1.0, 2.0, 3.0],
    ]

    model = af.ModelMapper(mock_class_1=af.m.MockClassx4)

    samples_x5 = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 4.0, 5.0],
            log_prior_list=5 * [0.0],
            weight_list=[0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    )

    vector = samples_x5.draw_randomly_via_pdf(as_instance=False)

    assert vector == [21.0, 22.0, 23.0, 24.0]

    instance = samples_x5.draw_randomly_via_pdf(as_instance=True)

    assert vector == [21.0, 22.0, 23.0, 24.0]

    assert instance.mock_class_1.one == 21.0
    assert instance.mock_class_1.two == 22.0
    assert instance.mock_class_1.three == 23.0
    assert instance.mock_class_1.four == 24.0


@pytest.fixture(name="make_samples")
def make_samples_fixture():
    def make_samples(parameters, weight_list=None):
        log_likelihood_list = list(range(len(parameters)))

        weight_list = weight_list or len(parameters) * [0.1]

        model = af.ModelMapper(mock_class=af.m.MockClassx2)
        return af.m.MockSamples(
            model=model,
            sample_list=af.Sample.from_lists(
                model=model,
                parameter_lists=parameters,
                log_likelihood_list=log_likelihood_list,
                log_prior_list=3 * [0.0],
                weight_list=weight_list,
            ),
        )

    return make_samples


def test__covariance_matrix(make_samples):
    samples_x5 = make_samples(
        parameters=[[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
    )

    assert samples_x5.covariance_matrix == pytest.approx(
        np.array([[1.0, 1.0], [1.0, 1.0]]), 1.0e-4
    )

    parameters = [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]

    samples_x5 = make_samples(parameters)

    assert samples_x5.covariance_matrix == pytest.approx(
        np.array([[1.0, -1.0], [-1.0, 1.0]]), 1.0e-4
    )

    samples_x5 = make_samples(
        parameters,
        weight_list=[0.1, 0.2, 0.3],
    )

    assert samples_x5.covariance_matrix == pytest.approx(
        np.array([[0.90909, -0.90909], [-0.90909, 0.90909]]), 1.0e-4
    )


def test__quantile_single_weighted_sample_does_not_crash():
    """
    A weighted `quantile` over a single sample must return that sample's value
    for every quantile rather than raising. Previously `np.cumsum(sw)[:-1]` was
    empty for one sample and `cdf[-1]` raised IndexError — surfaced when latent
    masking left exactly one finite sample whose weight was < 0.99.
    """
    from autofit.non_linear.samples.pdf import quantile

    assert quantile(x=[5.0], q=0.5, weights=[0.3]) == [5.0]
    assert quantile(x=[5.0], q=[0.16, 0.5, 0.84], weights=[0.3]) == [5.0, 5.0, 5.0]
    # n >= 2 weighted path is unchanged.
    assert quantile(x=[1.0, 2.0], q=0.5, weights=[0.5, 0.5]) == pytest.approx([1.5])
