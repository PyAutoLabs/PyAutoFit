import pytest

import autofit as af


class Analysis(af.Analysis):

    def log_likelihood_function(self, instance):
        return -1


@pytest.fixture(
    name="search"
)
def make_search(session):
    return af.DynestyStatic(
        session=session
    )


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
        af.ex.Gaussian
    )


def test_is_database_paths(search):
    assert isinstance(
        search.paths,
        af.DatabasePaths
    )


# def test_is_complete(search, session, model):
#     search.fit(
#         model,
#         Analysis()
#     )
#
#     fit, = m.Fit.all(session)
#
#     assert fit.id == search.paths.identifier
#     assert fit.is_complete
#
#     search = af.DynestyStatic(
#         session=session
#     )
#
#     search.paths.model = model
#
#     assert search.paths.identifier == fit.id
#     assert search.paths.is_complete
#
#     search.fit(
#         model,
#         Analysis()
#     )
#
#
# def test_remove_after(search, model):
#     search.paths.model = model
#     output_path = search.paths.output_path
#
#     search.fit(
#         model,
#         Analysis()
#     )
#
#     assert not Path(output_path).exists()
