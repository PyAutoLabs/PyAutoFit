import autofit as af


def test_fixed_scalar_is_enumerated_once_and_free_parameters_are_excluded():
    model = af.Model(af.ex.Gaussian, centre=0.0, normalization=2.0)
    identifier = model.identifier
    pairs = model.direct_instance_tuples
    assert [(pair.name, float(pair.value)) for pair in pairs] == [("centre", 0.0), ("normalization", 2.0)]
    assert model.prior_count == 1
    assert model.identifier == identifier


def test_raw_float_is_retained():
    model = af.Collection()
    object.__setattr__(model, "raw", 1.5)
    assert [(pair.name, float(pair.value)) for pair in model.direct_instance_tuples] == [("raw", 1.5)]


def test_fixed_tuple_slots_are_unique():
    model = af.Collection(centre=(0.0, 0.0))
    assert model.direct_instance_tuples == []
    assert model.instance_from_vector([]).centre == (0.0, 0.0)
    tuple_model = af.Model(af.m.MockClassx2Tuple)
    tuple_model.one_tuple.one_tuple_0 = 0.0
    tuple_model.one_tuple.one_tuple_1 = 0.0
    assert len(tuple_model.one_tuple.instance_tuples) == 2
    assert tuple_model.instance_from_vector([]).one_tuple == (0.0, 0.0)
