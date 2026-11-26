from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union

from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.search.deprecated_family import warn_family_subclass
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.posterior_kind import CHAIN
from autofit.non_linear.samples.mcmc import SamplesMCMC

if TYPE_CHECKING:
    from autofit.database.sqlalchemy_ import sa


class AbstractMCMC(NonLinearSearch):
    """
    **Deprecated** (search-extensibility A4c): no public search inherits it any more,
    and it is removed one release later. Subclass ``NonLinearSearch`` with
    ``posterior_kind = PosteriorKind.CHAIN`` and apply
    ``autofit.non_linear.search.posterior_kind.CHAIN`` in ``__init__`` instead.

    Kept as a distinct class with the former defaults (``InitializerBall(0.49, 0.51)``,
    ``auto_correlation_settings``, ``SamplesMCMC``, the cornerpy corner plot), so an
    existing subclass behaves exactly as before; subclassing it warns.
    """

    samples_cls = SamplesMCMC

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warn_family_subclass(cls, "AbstractMCMC", "CHAIN")

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            initializer: Optional[Initializer] = None,
            auto_correlation_settings=AutoCorrelationsSettings(),
            iterations_per_full_update: Optional[int] = None,
            iterations_per_quick_update: int = None,
            number_of_cores: Union[int, str] = "auto",
            silence: bool = False,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        self.auto_correlation_settings = auto_correlation_settings

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=CHAIN.initializer_for(initializer),
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs
        )

    def plot_results(self, samples):
        CHAIN.plot_results(self, samples)
