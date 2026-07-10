import logging
import os
from pathlib import Path
import warnings

from autofit.graphical.expectation_propagation.history import EPHistory

logger = logging.getLogger(__name__)


class Visualise:
    def __init__(self, ep_history: EPHistory, output_path: Path):
        """
        Handles visualisation of expectation propagation optimisation.

        This includes plotting key metrics such as Evidence and KL Divergence
        which are expected to converge.

        Parameters
        ----------
        ep_history
            A history describing previous optimisations by factor
        output_path
            The path that plots are written to
        """
        self.ep_history = ep_history
        self.output_path = output_path

        os.makedirs(output_path, exist_ok=True)

    def __call__(self):
        """
        Save a plot of Evidence and KL Divergence for the ep_history
        """
        import matplotlib.pyplot as plt

        fig, (evidence_plot, kl_plot) = plt.subplots(2)
        fig.suptitle("Evidence and KL Divergence")
        evidence_plot.plot(self.ep_history.evidences(), label="evidence")
        kl_plot.semilogy(self.ep_history.kl_divergences(), label="KL divergence")
        evidence_plot.legend()
        kl_plot.legend()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.savefig(str(self.output_path / "graph.png"))
        plt.close()

        self.plot_factors()

    def plot_factors(self):
        """
        Save `graph_factors.png`: each factor's evidence and KL-divergence
        history on its own curve, so a single misbehaving factor (failing
        fits, oscillating KL) is visible instead of being averaged into the
        global curves of `graph.png`.
        """
        import matplotlib.pyplot as plt

        fig, (evidence_plot, kl_plot) = plt.subplots(2)
        fig.suptitle("Per-factor Evidence and KL Divergence")
        for factor, factor_history in self.ep_history.items():
            evidence_plot.plot(
                factor_history.evidences, label=f"{factor.name}"
            )
            kl_plot.plot(
                factor_history.kl_divergences, label=f"{factor.name}"
            )
        kl_plot.set_yscale("log")
        evidence_plot.legend(fontsize="small")
        kl_plot.legend(fontsize="small")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.savefig(str(self.output_path / "graph_factors.png"))
        plt.close()
