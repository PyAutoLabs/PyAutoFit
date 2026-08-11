import pytest

import autofit as af

from autofit.aggregator.base import AggBase


class _RejectsLowStoredValue:
    def __init__(self, value):
        if value < 0.5:
            raise af.exc.FitException("stored value is outside the current domain")
        self.value = value


class _Fit:
    def __init__(self, samples):
        self.samples = samples


class _Aggregator:
    def __init__(self, fit):
        self.fit = fit

    def map(self, func):
        return [func(self.fit)]


class _InstanceAgg(AggBase):
    def object_via_gen_from(self, fit, instance=None):
        return instance


def _samples_with_historical_invalid_point():
    model = af.Model(_RejectsLowStoredValue)
    model.value = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    return af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=[[0.1], [0.9]],
            log_likelihood_list=[2.0, 1.0],
            log_prior_list=[0.0, 0.0],
            weight_list=[0.6, 0.4],
        ),
    )


def test__weighted_aggregator_objects_skip_invalid_historical_samples():
    samples = _samples_with_historical_invalid_point()
    agg = _InstanceAgg(aggregator=_Aggregator(fit=_Fit(samples=samples)))

    objects = agg.all_above_weight_gen_from(minimum_weight=-1.0)
    weights = agg.weights_above_gen_from(minimum_weight=-1.0)

    assert [[instance.value for instance in result] for result in objects] == [[0.9]]
    assert weights == [[0.4]]


def test__weighted_aggregator_preserves_shared_factor_graph_children():
    shared_value = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    factor_graph = af.FactorGraphModel(
        *[
            af.AnalysisFactor(
                prior_model=af.Collection(
                    galaxies=af.Model(_RejectsLowStoredValue, value=shared_value)
                ),
                analysis=af.m.MockAnalysis(),
            )
            for _ in range(2)
        ]
    )
    model = factor_graph.global_prior_model
    samples = af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=[[0.9]],
            log_likelihood_list=[1.0],
            log_prior_list=[0.0],
            weight_list=[1.0],
        ),
    )
    agg = _InstanceAgg(aggregator=_Aggregator(fit=_Fit(samples=samples)))

    objects = agg.all_above_weight_gen_from(minimum_weight=-1.0)

    instance = objects[0][0]
    assert getattr(instance, "0").galaxies.value == 0.9
    assert getattr(instance, "1").galaxies.value == 0.9


def test__weighted_aggregator_does_not_hide_programming_errors():
    class _RaisesUnexpectedly:
        def __init__(self, value):
            raise ValueError("real bug")

    model = af.Model(_RaisesUnexpectedly)
    samples = af.SamplesPDF(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=[[1.0]],
            log_likelihood_list=[1.0],
            log_prior_list=[0.0],
            weight_list=[1.0],
        ),
    )
    agg = _InstanceAgg(aggregator=_Aggregator(fit=_Fit(samples=samples)))

    with pytest.raises(ValueError, match="real bug"):
        agg.all_above_weight_gen_from(minimum_weight=-1.0)
