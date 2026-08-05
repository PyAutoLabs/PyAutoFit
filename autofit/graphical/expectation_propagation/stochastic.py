import logging
from typing import Dict, List, Generator

from autofit import exc
from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.mean_field import Status
from autofit.graphical.utils import StatusFlag, LogWarnings
from autofit.mapper.variable import Plate
from autofit.tools.util import IntervalCounter
from .optimiser import EPOptimiser

logger = logging.getLogger(__name__)


class StochasticEPOptimiser(EPOptimiser):
    def factor_step(self, factor, subset_approx, optimiser):
        factor_logger = logging.getLogger(factor.name)
        factor_logger.debug("Optimising...")

        subset_factor = subset_approx._factor_subset_factor[factor]
        try:
            with LogWarnings(
                logger=factor_logger.debug, action="always"
            ) as caught_warnings:
                factor_approx = subset_approx.factor_approximation(factor)
                new_model_dist, status = optimiser.optimise(factor_approx)
                subset_approx, status = self.updater.update_model_approx(
                    new_model_dist, factor_approx, subset_approx, status
                )

            messages = status.messages + tuple(caught_warnings.messages)
            # Keyword arguments: `Status`'s third positional parameter is
            # `updated`, not `flag` — see the same fix in `optimiser.factor_step`.
            status = Status(
                success=status.success,
                messages=messages,
                updated=status.updated,
                flag=status.flag,
            )
        except (
            ValueError,
            ArithmeticError,
            RuntimeError,
            exc.InitializerException,
        ) as e:
            logger.exception(e)
            status = Status(
                success=False,
                messages=status.messages
                + (f"Factor: {factor} experienced error {e}",),
                updated=False,
                flag=StatusFlag.EXCEPTION,
            )

        factor_logger.debug(status)
        return subset_approx, status

    def run(
        self,
        model_approx: EPMeanField,
        batches: Generator[Dict[Plate, List[int]], None, None],
        log_interval: int = 10,
        visualise_interval: int = 100,
        output_interval: int = 10,
        inplace=False,
    ) -> EPMeanField:
        """
        Run the optimisation on an approximation of the model.

        Parameters
        ----------
        model_approx
            A collection of messages describing priors on the model's variables.
        max_steps
            The maximum number of steps prior to termination. Termination may also
            occur when difference in log evidence or KL Divergence drop below a given
            threshold for two consecutive optimisations of a given factor.
        log_interval
            How steps should we wait before logging information?
        visualise_interval
            How steps should we wait before visualising information?
            This includes plots of KL Divergence and Evidence.
        output_interval
            How steps should we wait before outputting information?
            This includes the model.results file which describes the current mean values
            of each message.

        Returns
        -------
        An updated approximation of the model
        """
        should_log = IntervalCounter(log_interval)
        should_visualise = IntervalCounter(visualise_interval)
        should_output = IntervalCounter(output_interval)

        for batch in batches:
            subset_approx = model_approx[batch]

            for factor, optimiser in self.factor_optimisers.items():
                subset_approx, status = self.factor_step(
                    factor, subset_approx, optimiser
                )

                if status and should_log():
                    self._log_factor(factor)

                if self.ep_history(factor, subset_approx, status):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

            else:  # If no break do next iteration

                if inplace:
                    model_approx[batch] = subset_approx
                else:
                    model_approx = model_approx.merge(batch, subset_approx)

                if self.ep_history(model_approx.factor_graph, model_approx):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

                if should_visualise():
                    self.visualiser()
                if should_output():
                    self._output_results(model_approx)
                continue
            break  # stop iterations

        self.visualiser()
        self._output_results(model_approx)

        return model_approx
