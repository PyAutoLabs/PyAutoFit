import numpy as np
import pytest

import autofit as af
from autofit.mapper.prior_model.constraint import (
    MODEL_CONSTRAINT,
    declares_model_constraint,
    violation_for_instance,
)
from autofit.non_linear.search.mle.multi_start_gradient.search import (
    AbstractMultiStartGradient,
)


class Unconstrained:
    def __init__(self, centre=0.0, normalization=0.1):
        self.centre = centre
        self.normalization = normalization


class Saturating:
    """Stands in for an elliptical profile: two components, a joint unit-disc
    constraint that no per-parameter prior limit can express."""

    def __init__(self, ell_comps_0=0.0, ell_comps_1=0.0):
        self.ell_comps_0 = ell_comps_0
        self.ell_comps_1 = ell_comps_1

    def __model_constraint__(self, xp=np):
        magnitude = xp.sqrt(self.ell_comps_0**2 + self.ell_comps_1**2)
        return xp.maximum(magnitude - 0.999, 0.0)


class TestDeclaration:
    def test_duck_typed_detection(self):
        assert declares_model_constraint(Saturating)
        assert not declares_model_constraint(Unconstrained)

    def test_model_exposes_the_declaration(self):
        assert af.Model(Saturating).has_model_constraint
        assert not af.Model(Unconstrained).has_model_constraint

    def test_malformed_declaration_fails_at_composition(self):
        class Malformed:
            def __init__(self, centre=0.0):
                self.centre = centre

        setattr(Malformed, MODEL_CONSTRAINT, "not callable")

        with pytest.raises(AssertionError):
            af.Model(Malformed)

    def test_violation_for_instance(self):
        assert violation_for_instance(Saturating(0.5, 0.0)) == 0.0
        assert violation_for_instance(Saturating(1.2, 0.0)) == pytest.approx(0.201)


def saturating_model():
    """Priors given explicitly: the fixture classes have no entry in the priors
    config, so `make_prior` would otherwise hand back a `ConfigException`."""
    return af.Model(
        Saturating,
        ell_comps_0=af.UniformPrior(lower_limit=-4.0, upper_limit=4.0),
        ell_comps_1=af.UniformPrior(lower_limit=-4.0, upper_limit=4.0),
    )


def unconstrained_model():
    return af.Model(
        Unconstrained,
        centre=af.UniformPrior(lower_limit=-1.0, upper_limit=1.0),
        normalization=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
    )


class TestConstrainedModelTuples:
    def test_finds_nested_components(self):
        model = af.Collection(
            a=saturating_model(),
            b=unconstrained_model(),
            c=af.Collection(d=saturating_model()),
        )
        assert len(model.constrained_model_tuples()) == 2

    def test_none_when_nothing_declares(self):
        model = af.Collection(b=unconstrained_model())
        assert model.constrained_model_tuples() == []


class TestConstraintFromVector:
    def test_zero_inside_the_valid_region(self):
        model = af.Collection(a=saturating_model())
        assert float(model.model_constraint_from_vector([0.5, 0.2])) == 0.0

    def test_positive_outside(self):
        model = af.Collection(a=saturating_model())
        assert float(model.model_constraint_from_vector([1.2, 0.0])) > 0.0

    def test_catches_the_corner_region(self):
        """Both components inside (-1, 1) yet magnitude above 1 — the case a
        per-parameter prior-limit test provably cannot see."""
        model = af.Collection(a=saturating_model())
        assert float(model.model_constraint_from_vector([0.8, 0.8])) > 0.0

    def test_zero_when_model_declares_nothing(self):
        model = af.Collection(b=unconstrained_model())
        assert float(model.model_constraint_from_vector([0.5, 0.2])) == 0.0

    def test_reduces_over_components_with_a_maximum(self):
        model = af.Collection(a=saturating_model(), c=saturating_model())
        vector = [0.0] * model.prior_count
        # set whichever pair belongs to `c`, without assuming prior ordering
        index = model.prior_count - 2
        for path, sub in model.constrained_model_tuples():
            if path and path[-1] == "c":
                index = min(
                    model.prior_tuples_ordered_by_id.index(t)
                    for t in sub.prior_tuples_ordered_by_id
                )
        vector[index] = 3.0
        assert float(model.model_constraint_from_vector(vector)) == pytest.approx(2.001)


class TestConstrainedLaneCount:
    """The predicate is pure NumPy and search-state-free, so it is driven
    directly — `_fit` needs jax + optax and cannot run in this suite."""

    def test_counts_only_violating_lanes(self):
        assert (
            AbstractMultiStartGradient._constrained_lane_count(
                alive=np.array([True, True, True]),
                grad_finite=np.array([True, True, True]),
                constraint_violation=np.array([0.0, 0.5, 0.0]),
            )
            == 1
        )

    def test_disjoint_from_value_nan(self):
        """A dead lane is not also counted as constrained."""
        assert (
            AbstractMultiStartGradient._constrained_lane_count(
                alive=np.array([False, True]),
                grad_finite=np.array([True, True]),
                constraint_violation=np.array([9.0, 0.0]),
            )
            == 0
        )

    def test_disjoint_from_gradient_nan(self):
        assert (
            AbstractMultiStartGradient._constrained_lane_count(
                alive=np.array([True, True]),
                grad_finite=np.array([False, True]),
                constraint_violation=np.array([9.0, 0.0]),
            )
            == 0
        )

    def test_buckets_sum_to_at_most_the_lane_count(self):
        foms = np.array([np.nan, 1.0, 2.0, 3.0])
        grad_finite = np.array([False, False, True, True])
        violation = np.array([5.0, 5.0, 5.0, 0.0])

        alive, n_value_nan, n_grad_nan = AbstractMultiStartGradient._nan_lane_counts(
            foms=foms, grad_finite=grad_finite
        )
        n_constrained = AbstractMultiStartGradient._constrained_lane_count(
            alive=alive, grad_finite=grad_finite, constraint_violation=violation
        )
        assert (n_value_nan, n_grad_nan, n_constrained) == (1, 1, 1)
        assert n_value_nan + n_grad_nan + n_constrained <= len(foms)
