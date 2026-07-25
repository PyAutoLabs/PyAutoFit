import pytest

import autofit as af


@pytest.fixture(
    name="target_gaussian"
)
def make_target_gaussian():
    return af.Model(
        af.ex.Gaussian
    )


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.UniformPrior()


@pytest.fixture(
    name="source_gaussian"
)
def make_source_gaussian(prior):
    return af.Model(
        af.ex.Gaussian,
        centre=prior
    )


def test_simple(
        source_gaussian,
        target_gaussian,
        prior
):
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == prior


def test_assertions(
        source_gaussian,
        target_gaussian
):
    target_gaussian.add_assertion(
        target_gaussian.centre <= target_gaussian.normalization
    )

    with pytest.raises(AssertionError):
        target_gaussian.take_attributes(
            source_gaussian
        )


def test_assertions_collection(
        source_gaussian,
        target_gaussian
):
    target_gaussian.add_assertion(
        target_gaussian.centre <= target_gaussian.normalization
    )

    target_collection = af.Collection(
        gaussian=target_gaussian
    )
    source_collection = af.Collection(
        gaussian=source_gaussian
    )

    with pytest.raises(AssertionError):
        target_collection.take_attributes(
            source_collection
        )


def test_in_collection(
        source_gaussian,
        target_gaussian,
        prior
):
    target = af.Collection(
        gaussian=target_gaussian
    )
    source = af.Collection(
        gaussian=source_gaussian
    )
    target.take_attributes(
        source
    )

    assert target.gaussian.centre == prior


def test_tuple(
        source_gaussian,
        target_gaussian,
        prior
):
    source_gaussian.centre = (prior, 1.0)
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == (prior, 1.0)


def test_tuple_prior(
        source_gaussian,
        target_gaussian,
        prior
):
    source_gaussian.centre = (prior, 1.0)
    target_gaussian.centre = af.TuplePrior()
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == (prior, 1.0)


def test_tuple_in_instance(
        target_gaussian,
        prior
):
    # noinspection PyTypeChecker
    source_gaussian = af.ex.Gaussian(
        centre=(prior, 1.0)
    )
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == (prior, 1.0)


def test_tuple_in_collection(
        source_gaussian,
        target_gaussian,
        prior
):
    source_gaussian.centre = (prior, 1.0)

    source = af.Collection(
        gaussian=source_gaussian
    )
    target = af.Collection(
        gaussian=target_gaussian
    )

    target.take_attributes(source)
    assert target.gaussian.centre == (prior, 1.0)


def test_tuple_in_instance_in_collection(
        target_gaussian,
        prior
):
    # noinspection PyTypeChecker
    source_gaussian = af.ex.Gaussian(
        centre=(prior, 1.0)
    )

    source = af.Collection(
        gaussian=source_gaussian
    )
    target = af.Collection(
        gaussian=target_gaussian
    )

    target.take_attributes(source)
    assert target.gaussian.centre == (prior, 1.0)


def test_source_is_dict(
        source_gaussian,
        target_gaussian,
        prior
):
    source = dict(
        gaussian=source_gaussian
    )
    target = af.Collection(
        gaussian=target_gaussian
    )
    target.take_attributes(source)

    assert target.gaussian.centre == prior


def test_target_is_dict(
        source_gaussian,
        target_gaussian,
        prior
):
    source = af.Collection(
        collection=af.Collection(
            gaussian=source_gaussian
        )
    )
    target = af.Collection(
        collection=dict(
            gaussian=target_gaussian
        )
    )
    target.take_attributes(source)

    assert target.collection.gaussian.centre == prior


def test_missing_from_source(
        target_gaussian,
        prior
):
    target_gaussian.centre = prior

    target_gaussian.take_attributes(
        af.Collection()
    )
    assert target_gaussian.centre == prior


def test_unlabelled_in_collection(
        source_gaussian,
        target_gaussian,
        prior
):
    target = af.Collection(
        [target_gaussian]
    )
    source = af.Collection(
        [source_gaussian]
    )
    target.take_attributes(
        source
    )

    assert target[0].centre == prior


def test_passing_float(
        source_gaussian,
        target_gaussian
):
    source_gaussian.centre = 2.0
    target_gaussian.take_attributes(
        source_gaussian
    )

    assert target_gaussian.centre == 2.0


def test_missing_from_origin(
        target_gaussian
):
    target_gaussian.take_attributes(
        af.Collection()
    )


def test_limits(
        source_gaussian,
        target_gaussian
):
    source_gaussian.centre = af.TruncatedGaussianPrior(
        mean=0,
        sigma=1,
        lower_limit=-1,
        upper_limit=1
    )
    target_gaussian.take_attributes(
        source_gaussian
    )
    assert target_gaussian.centre.lower_limit == -1
    assert target_gaussian.centre.upper_limit == 1


def test_tuples():
    centre = (0.0, 1.0)
    source = af.Model(
        af.ex.Gaussian,
        centre=centre
    )
    target = af.Model(
        af.ex.Gaussian
    )
    target.take_attributes(source)
    assert target.centre == centre
