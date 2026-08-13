import pytest

import autofit as af


class _RejectsUnitMagnitude:
    """
    Mirrors an elliptical profile's ``ell_comps`` guard: the pair is only
    meaningful while its magnitude is below 1.
    """

    def __init__(self, ell_comps=(0.0, 0.0)):
        if ell_comps[0] ** 2 + ell_comps[1] ** 2 >= 1.0:
            raise ValueError(f"ell_comps magnitude is not below 1: {ell_comps}")

        self.ell_comps = ell_comps


def _guarded_model():
    model = af.Model(_RejectsUnitMagnitude)
    model.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    model.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
    return model


def test__mock_samples_summary__fills_with_prior_medians():
    summary = af.m.MockSamplesSummary(model=_guarded_model())

    assert list(summary.max_log_likelihood_sample.kwargs.values()) == [0.25, 0.25]
    assert list(summary.median_pdf_sample.kwargs.values()) == [0.25, 0.25]


def test__mock_samples__default_sample_list_fills_with_prior_medians():
    samples = af.m.MockSamples(model=_guarded_model())

    for sample in samples.sample_list:
        assert list(sample.kwargs.values()) == [0.25, 0.25]


def test__mock_result_without_samples_summary__builds_a_valid_instance():
    """
    A ``MockResult`` given a model but no ``samples_summary`` builds its own.
    That placeholder used to fill every parameter with ``1.0``, which is an
    invalid ``ell_comps`` — so reading ``.instance`` raised.
    """
    instance = af.m.MockResult(model=_guarded_model()).instance

    assert instance.ell_comps == (0.25, 0.25)


def test__all_ones_placeholder_would_be_rejected():
    """
    Guards the test above: confirms the model really does reject the old
    blanket ``1.0`` fill, so it cannot silently stop testing anything.
    """
    model = _guarded_model()

    assert model.instance_from_prior_medians().ell_comps == (0.25, 0.25)

    with pytest.raises(ValueError, match="magnitude is not below 1"):
        model.instance_for_arguments(
            {prior: 1.0 for prior in model.priors_ordered_by_id}
        )
