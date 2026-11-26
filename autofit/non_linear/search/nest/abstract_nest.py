from __future__ import annotations

from abc import ABC
from typing import Optional, TYPE_CHECKING, Union

from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.search.deprecated_family import warn_family_subclass
from autofit.non_linear.search.posterior_kind import WEIGHTED
from autofit.non_linear.samples.nest import SamplesNest

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa


class AbstractNest(NonLinearSearch, ABC):
    """
    **Deprecated** (search-extensibility A4c): no public search inherits it any more,
    and it is removed one release later. Subclass ``NonLinearSearch`` with
    ``posterior_kind = PosteriorKind.WEIGHTED`` and apply
    ``autofit.non_linear.search.posterior_kind.WEIGHTED`` in ``__init__`` instead.

    Kept as a distinct class with the former defaults (``InitializerPrior``, the
    rejection of ``InitializerParamBounds``, ``SamplesNest``, the anesthetic corner
    plot), so an existing subclass behaves exactly as before; subclassing it warns.
    """

    samples_cls = SamplesNest

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warn_family_subclass(cls, "AbstractNest", "WEIGHTED")

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        iterations_per_quick_update: Optional[int] = None,
        iterations_per_full_update: Optional[int] = None,
        number_of_cores: Union[int, str] = "auto",
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        initializer: Optional[AbstractInitializer] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=WEIGHTED.nested_initializer_for(initializer),
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs
        )

    def plot_results(self, samples):
        WEIGHTED.plot_results(self, samples)
