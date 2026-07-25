import autofit as af

def test_constructor():
    prior_model = af.Model(af.m.MockOverload)

    assert prior_model.prior_count == 1

    instance = prior_model.instance_from_prior_medians()

    assert instance.one == 1.0
    assert instance.two == 2


def test_alternative():
    prior_model = af.Model(af.m.MockOverload.with_two)

    assert prior_model.prior_count == 1

    instance = prior_model.instance_from_prior_medians()

    assert instance.two == 1.0
    assert instance.one == 1.0 / 2
