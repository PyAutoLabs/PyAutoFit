from typing import Optional

from autoconf import conf
from autofit.database.sqlalchemy_ import sa
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.samples import SamplesMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.plot import corner_cornerpy

class AbstractMCMC(NonLinearSearch):

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            initializer: Optional[Initializer] = None,
            auto_correlation_settings=AutoCorrelationsSettings(),
            iterations_per_full_update: Optional[int] = None,
            iterations_per_quick_update: int = None,
            number_of_cores: Optional[int] = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        
        self.auto_correlation_settings = auto_correlation_settings
        self.auto_correlation_settings.update_via_config(
            config=self.config_type[self.__class__.__name__]["auto_correlations"]
        )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mcmc"]

    @property
    def samples_cls(self):
        return SamplesMCMC

    def plot_results(self, samples):

        if not samples.pdf_converged:
            return

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["mcmc"][name]

        if should_plot("corner_cornerpy"):
            corner_cornerpy(
                samples=samples,
                path=self.paths.image_path / "search",
                format="png",
            )
