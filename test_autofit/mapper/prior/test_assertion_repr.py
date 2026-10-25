import autofit as af
import pytest


@pytest.mark.parametrize("kind", ["simple", "chained", "arithmetic"])
def test_print_gathered_assertions(kind, capsys):
    model = af.Model(af.ex.Gaussian)
    if kind == "simple":
        assertion = model.sigma > 5.0
    elif kind == "chained":
        assertion = (0.0 < model.sigma) < 5.0
    else:
        assertion = (model.sigma * 2.0) > 5.0
    model.add_assertion(assertion)
    identifier = model.identifier
    print(model.gathered_assertions())
    rendered = capsys.readouterr().out
    assert type(assertion).__name__ in rendered
    assert "5.0" in rendered
    assert rendered.count("\n") == 1
    assert model.identifier == identifier


def test_compound_prior_fallback_preserves_specialised_strings():
    prior = af.UniformPrior()
    assert "PowerPrior" in repr(prior ** 2.0)
    assert repr(prior + 2.0) == str(prior + 2.0)
    assert repr(prior * 2.0) == str(prior * 2.0)
