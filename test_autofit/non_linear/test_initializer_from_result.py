import pytest

import autofit as af
from autofit import Sample
from autofit.non_linear.mock.mock_result import MockResult
from autofit.non_linear.mock.mock_samples import MockSamples
from autofit.non_linear.mock.mock_samples_summary import MockSamplesSummary

# Pure numpy: `InitializerParamStartPoints.from_result` never imports JAX/blackjax,
# so these tests run unconditionally (no `skipif`/`importorskip` needed).


@pytest.fixture(name="model")
def make_model():
    model = af.Model(af.m.MockClassx4)
    model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    model.two = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    model.three = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    model.four = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    return model


@pytest.fixture(name="result")
def make_result(model):
    sample_list = [
        Sample(
            log_likelihood=1.0,
            log_prior=0.0,
            weight=0.2,
            kwargs={"one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4},
        ),
        Sample(
            log_likelihood=2.0,
            log_prior=0.0,
            weight=0.5,
            kwargs={"one": 0.11, "two": 0.22, "three": 0.33, "four": 0.44},
        ),
        Sample(
            log_likelihood=1.5,
            log_prior=0.0,
            weight=0.3,
            kwargs={"one": 0.12, "two": 0.19, "three": 0.28, "four": 0.41},
        ),
    ]
    samples = MockSamples(sample_list=sample_list, model=model)
    return MockResult(
        samples=samples,
        samples_summary=MockSamplesSummary(
            model=model,
            max_log_likelihood_instance=[0.11, 0.22, 0.33, 0.44],
            median_pdf_sample=sample_list[1],
        ),
        model=model,
    )


def parameter_dict_by_path(model, initializer):
    return {
        ".".join(model.path_for_prior(prior)): value
        for prior, value in initializer.parameter_dict.items()
    }


class TestFromResult:
    def test__default_point_is_max_log_likelihood(self, result, model):
        vector = result.samples.max_log_likelihood(as_instance=False)
        assert vector == [0.11, 0.22, 0.33, 0.44]

        initializer = af.InitializerParamStartPoints.from_result(result)

        parameter_dict = parameter_dict_by_path(model, initializer)
        assert parameter_dict["one"] == pytest.approx((0.11 - 1e-8, 0.11 + 1e-8))
        assert parameter_dict["two"] == pytest.approx((0.22 - 1e-8, 0.22 + 1e-8))
        assert parameter_dict["three"] == pytest.approx((0.33 - 1e-8, 0.33 + 1e-8))
        assert parameter_dict["four"] == pytest.approx((0.44 - 1e-8, 0.44 + 1e-8))

    def test__median_pdf_point(self, result, model):
        expected = result.samples.median_pdf(as_instance=False)

        initializer = af.InitializerParamStartPoints.from_result(
            result, point="median_pdf"
        )

        parameter_dict = parameter_dict_by_path(model, initializer)
        for path, value in zip(["one", "two", "three", "four"], expected):
            assert parameter_dict[path] == pytest.approx((value - 1e-8, value + 1e-8))

    def test__unrecognised_point_raises(self, result):
        with pytest.raises(af.exc.InitializerException):
            af.InitializerParamStartPoints.from_result(result, point="not_a_point")

    def test__priors_are_never_mutated(self, result, model):
        before = [(p.id, p.lower_limit, p.upper_limit) for p in model.priors_ordered_by_id]

        af.InitializerParamStartPoints.from_result(result)
        af.InitializerParamStartPoints.from_result(result, point="median_pdf")
        af.InitializerParamStartPoints.from_result(result, n_points=3, jitter=0.05, seed=0)

        after = [(p.id, p.lower_limit, p.upper_limit) for p in model.priors_ordered_by_id]
        assert before == after

    def test__n_points_shapes_and_distinct_points(self, result, model):
        initializer = af.InitializerParamStartPoints.from_result(
            result, n_points=5, jitter=0.05, seed=1
        )

        assert len(initializer._point_dicts) == 5
        for point_dict in initializer._point_dicts:
            assert len(point_dict) == model.prior_count

        values_for_one = {
            point_dict[prior]
            for point_dict in initializer._point_dicts
            for prior in point_dict
            if model.path_for_prior(prior) == ("one",)
        }
        assert len(values_for_one) == 5  # every point jittered to a distinct value

    def test__n_points_generation_reproducible_with_seed(self, result):
        initializer_a = af.InitializerParamStartPoints.from_result(
            result, n_points=4, jitter=0.05, seed=7
        )
        initializer_b = af.InitializerParamStartPoints.from_result(
            result, n_points=4, jitter=0.05, seed=7
        )

        for point_a, point_b in zip(initializer_a._point_dicts, initializer_b._point_dicts):
            for prior in point_a:
                assert point_a[prior] == point_b[prior]

    def test__n_points_used_by_samples_from_model(self, result, model):
        initializer = af.InitializerParamStartPoints.from_result(
            result, n_points=4, jitter=0.05, seed=1
        )

        # Non-constant fitness so the identical-figure-of-merit guard in
        # `samples_from_model` does not trip (mirrors the pattern used
        # elsewhere in test_initializer.py for `MockFitness`).
        from random import random

        def fitness(parameters):
            return random()

        _, parameter_lists, _ = initializer.samples_from_model(
            total_points=4,
            model=model,
            fitness=fitness,
            paths=af.DirectoryPaths(),
        )

        assert len(parameter_lists) == 4
        # Each of the 4 generated points should be distinct (one per chain).
        assert len({tuple(p) for p in parameter_lists}) == 4

    def test__n_points_one_matches_legacy_behaviour(self, result, model):
        # n_points=1, jitter=0.0 (defaults) must reproduce the exact classic
        # `InitializerParamStartPoints` single-point behaviour: no `_point_dicts`.
        initializer = af.InitializerParamStartPoints.from_result(result)
        assert initializer._point_dicts is None

    def test__dimension_mismatch_raises(self):
        model_a = af.Model(af.m.MockClassx2)

        sample_list = [
            Sample(
                log_likelihood=1.0,
                log_prior=0.0,
                weight=0.0,
                kwargs={"one": 0.1, "two": 0.2},
            ),
        ]
        samples = MockSamples(sample_list=sample_list, model=model_a)
        result = MockResult(
            samples=samples,
            samples_summary=MockSamplesSummary(
                model=model_a,
                max_log_likelihood_instance=[0.1, 0.2],
                median_pdf_sample=sample_list[0],
            ),
            model=model_a,
        )

        model_b = af.Model(af.m.MockClassx4)

        with pytest.raises(af.exc.InitializerException):
            af.InitializerParamStartPoints.from_result(result, model=model_b)
