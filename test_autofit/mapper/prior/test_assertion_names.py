import autofit as af
import pytest


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("name", ["", "sigma_floor"])
def test_added_assertion_name_and_violation(chained, name):
    model = af.Model(af.ex.Gaussian)
    assertion = (5.0 < model.sigma) < 10.0 if chained else model.sigma > 5.0
    model.add_assertion(assertion, name=name)
    assert model.gathered_assertions()[0].name == name
    with pytest.raises(af.exc.FitException) as error:
        model.check_assertions({model.sigma: 1.0})
    assert name in str(error.value)


def test_name_is_presentation_only():
    model = af.Model(af.ex.Gaussian)
    assertion = model.sigma > 5.0
    model.add_assertion(assertion)
    identifier = model.identifier
    assertion._name = "sigma_floor"
    assert assertion.name == "sigma_floor"
    assert model.identifier == identifier
    assert model.name == "Gaussian"
    assert af.Collection().name == "Collection"


def test_boolean_assertions_keep_their_behaviour():
    model = af.Model(af.ex.Gaussian)
    model.add_assertion(True, name="true")
    assert model.gathered_assertions() == []
    model.add_assertion(False, name="false")
    with pytest.raises(af.exc.FitException):
        model.check_assertions({})
