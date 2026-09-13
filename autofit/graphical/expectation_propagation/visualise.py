import logging
import os
from pathlib import Path
import warnings

from autofit.graphical.expectation_propagation.history import EPHistory

logger = logging.getLogger(__name__)


class Visualise:
    def __init__(
        self,
        ep_history: EPHistory,
        output_path: Path,
        factor_graph=None,
    ):
        """
        Handles visualisation of expectation propagation optimisation.

        This includes plotting key metrics such as Evidence and KL Divergence
        which are expected to converge, and -- when the `model_figure` key is
        switched on -- the factor graph itself with the run painted on it.

        Parameters
        ----------
        ep_history
            A history describing previous optimisations by factor
        output_path
            The path that plots are written to
        factor_graph
            The `EPOptimiser`'s own factor graph, which `graph_model.png` and
            `graph_state.png` are drawn from. `None` -- the default, and what
            every caller that does not pass one gets -- writes neither.
        """
        self.ep_history = ep_history
        self.output_path = output_path
        self.factor_graph = factor_graph
        self._model_written = False

        os.makedirs(output_path, exist_ok=True)

    def __call__(self, model_approx=None):
        """
        Save a plot of Evidence and KL Divergence for the ep_history.

        Parameters
        ----------
        model_approx
            The latest `EPMeanField`, passed through to the factor graph
            figure. Optional: callers which do not have one (`stochastic.py`)
            still get the status, age and reversion overlay, which is read from
            the history rather than the mean field.
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
        self._model_figures(model_approx)

    def _model_figures(self, model_approx=None):
        """
        Save `graph_model.png` and `graph_state.png`: the factor graph the
        optimiser sweeps, and the same graph with what the run has done to each
        factor drawn on it.

        `graph_model.png` is structure alone and is therefore written once, on
        the first call, beside `graph.info`. `graph_state.png` is rewritten on
        every call, so the file always shows the sweep just finished.

        Both are behind `output.yaml`'s `model_figure` key -- the same key the
        per-search `model.png` is behind, read the same strict way, so a config
        which has never heard of it writes nothing. Each render costs roughly a
        third of a second, which is why it rides `visualise_interval` rather
        than every sweep.

        A figure must never kill a fit, so the whole body is guarded exactly as
        `DirectoryPaths._save_model_info`'s is: anything that goes wrong is
        logged and the run carries on.
        """
        if self.factor_graph is None:
            return

        try:
            from autofit.model_figure.config import model_figure_enabled
            from autofit.non_linear.test_mode import skip_visualization

            if not model_figure_enabled() or skip_visualization():
                return

            from autofit.model_figure.ep.plotter import EPPlotter, FILENAMES

            if not self._model_written:
                EPPlotter(self.factor_graph).figure(
                    path=self.output_path,
                    filename=FILENAMES["model"],
                    format="png",
                    kind="model",
                )
                self._model_written = True

            EPPlotter(
                self.factor_graph,
                ep_history=self.ep_history,
                model_approx=model_approx,
            ).figure(
                path=self.output_path,
                filename=FILENAMES["state"],
                format="png",
                kind="state",
            )
        except Exception as e:
            logger.info(f"graph_model.png / graph_state.png not written: {e!r}")

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
            evidence_plot.plot(factor_history.evidences, label=f"{factor.name}")
            kl_plot.plot(factor_history.kl_divergences, label=f"{factor.name}")
        kl_plot.set_yscale("log")
        evidence_plot.legend(fontsize="small")
        kl_plot.legend(fontsize="small")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.savefig(str(self.output_path / "graph_factors.png"))
        plt.close()
