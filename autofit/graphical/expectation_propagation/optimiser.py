import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple

from autofit import exc
from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.graphical.expectation_propagation.history import EPHistory
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.graphical.mean_field import Status, MeanField, FactorApproximation
from autofit.graphical.utils import StatusFlag, LogWarnings
from autofit.mapper.identifier import Identifier
from autofit.non_linear.parallel import fork_context
from autofit.non_linear.paths import DirectoryPaths
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.tools.util import IntervalCounter
from .diagnostics import EPDiagnostics, check_sigma_collapse, mean_field_summary
from .factor_optimiser import AbstractFactorOptimiser, ExactFactorFit
from .visualise import Visualise

logger = logging.getLogger(__name__)


class ApproxUpdater(ABC):
    """
    Handles updating of the EPMeanField using new model distributions from
    individual factors
    """

    @abstractmethod
    def delta(self, factor: Factor, model_approx: EPMeanField):
        """
        Compute a delta for a given factor. This dictates how much the message
        for that factor is updated. A delta of 0 is no update and a delta of 1
        is maximum update.
        """

    def update_model_approx(
        self,
        new_model_dist: MeanField,
        factor_approx: FactorApproximation,
        model_approx: EPMeanField,
        status: Status = Status(),
    ) -> Tuple[EPMeanField, Status]:
        delta = self.delta(factor=factor_approx.factor, model_approx=model_approx,)

        return model_approx.project_mean_field(
            new_model_dist, factor_approx, delta=delta, status=status,
        )


class SimplerUpdater(ApproxUpdater):
    def __init__(self, delta: float = 1.0):
        """
        Simply set delta to a fixed value

        Parameters
        ----------
        delta
            A fixed rate at which all factors update
        """
        self._delta = delta

    def delta(self, factor, model_approx):
        return self._delta


class FactorUpdater(SimplerUpdater):
    def __init__(self, factor_deltas: Dict[Factor, float], default=1.0):
        """
        Determine how fast factors update on a factor by factor basis.

        Parameters
        ----------
        factor_deltas
            A dictionary dictating how fast each factor should update the mean field.
        default
            A default value for when no value is provided.
        """
        super().__init__(delta=default)
        self.factor_deltas = factor_deltas

    def delta(self, factor, model_approx):
        return self.factor_deltas.get(factor, self._delta)


class DynamicUpdater(SimplerUpdater):
    def delta(self, factor, model_approx):
        """
        Variables are updated dynamically. The more factors that share a variable
        the slower that variable is updated. This accounts for cases when a variable
        is shared by many more factors than another variable and so shrinks too
        quickly.
        """

        variable_message_count = model_approx.variable_message_count
        min_value = min(variable_message_count.values())
        return MeanField(
            {
                variable: self._delta * (min_value / message_count)
                for variable, message_count in variable_message_count.items()
            }
        )


def factor_step(factor_approx, optimiser, model_approx=None):
    factor = factor_approx.factor
    factor_logger = logging.getLogger(factor.name)
    factor_logger.debug("Optimising...")
    # Mean-field opt-in: factors that implement ``set_model_approx``
    # (e.g. ``EPAnalysisFactor``) get the *full* ``EPMeanField`` first.
    # This is needed by hierarchical factors that have to inspect
    # sibling factors' messages on variables the current factor's
    # cavity no longer contains — e.g. variables that were dropped from
    # this factor's prior model via the constant-element pattern of
    # ``Array.__setitem__``. Default factors lack the method, so the
    # call is a no-op for them. ``model_approx`` is None when called
    # from a context that does not propagate it (older call sites).
    if hasattr(factor, "set_model_approx") and model_approx is not None:
        factor.set_model_approx(model_approx)
    # Cavity-message opt-in: factors that implement ``set_cavity_dist``
    # (e.g. ``EPAnalysisFactor``) receive the current cavity distribution
    # before optimisation so their Analysis can read per-variable cavity
    # messages inside ``log_likelihood_function``. Default factors lack
    # the method, so this is a no-op for them.
    if hasattr(factor, "set_cavity_dist"):
        factor.set_cavity_dist(factor_approx.cavity_dist)
    try:
        with LogWarnings(
            logger=factor_logger.debug, action="always"
        ) as caught_warnings:
            new_model_dist, status = optimiser.optimise(factor_approx)

        messages = status.messages + tuple(caught_warnings.messages)

        # Keyword arguments matter here: `Status`'s third positional parameter is
        # `updated`, not `flag`. Passing the flag positionally silently dropped it
        # and left `flag` at its `SUCCESS` default, so a failed factor step was
        # recorded in `ep_history.csv` as a success.
        status = Status(
            success=status.success,
            messages=messages,
            updated=status.updated,
            flag=status.flag,
            result=status.result,
        )

    except (
        ValueError,
        ArithmeticError,
        RuntimeError,
        exc.InitializerException,
    ) as e:
        # `InitializerException` is raised when a factor's own optimiser cannot
        # find a start point — most commonly because EP has driven the factor to
        # a state where every drawn point has the same figure of merit. That is a
        # failure of this sweep's update for this factor, not of the graph fit:
        # degrade to the factor's previous message and let the sweep continue,
        # with the failure recorded. `EPOptimiser` aborts if one factor keeps
        # failing (see `max_consecutive_failures`).
        logger.exception(e)
        status = Status(
            success=False,
            messages=(f"Factor: {factor} experienced error {e}",),
            updated=False,
            flag=StatusFlag.EXCEPTION,
        )
        new_model_dist = factor_approx.model_dist

    factor_logger.debug(status)
    return new_model_dist, status


class EPOptimiser:
    def __init__(
        self,
        factor_graph: FactorGraph,
        default_optimiser: Optional[AbstractFactorOptimiser] = None,
        factor_optimisers: Optional[Dict[Factor, AbstractFactorOptimiser]] = None,
        ep_history: Optional[EPHistory] = None,
        factor_order: Optional[List[Factor]] = None,
        paths: AbstractPaths = None,
        updater: Optional[ApproxUpdater] = None,
    ):
        """
        Optimise a factor graph.

        Cycles through factors optimising them individually using priors
        created through expectation propagation; in effect the prior for each
        variable for a given factor is the product of the posteriors of that
        variable for all other factors.

        Parameters
        ----------
        factor_graph
            A graph describing the relationships between multiple factors
        default_optimiser
            An optimiser that is used if no specific optimiser is provided for a factor
        factor_optimisers
            Specific optimisers used to optimise each factor
        ep_history
            Optionally specify an alternate history
        factor_order
            The factors in the graph but placed in the order in which they should
            be optimised
        paths
            Optionally define how data should be output
        updater
            An optional policy controlling the strength of factor-message
            updates. If omitted, full updates are used via
            ``SimplerUpdater(delta=1.0)``.
        """
        factor_optimisers = factor_optimisers or {}
        self.factor_graph = factor_graph
        self.factors = factor_order or self.factor_graph.factors
        self.default_optimiser = default_optimiser
        self.updater = updater or SimplerUpdater(delta=1.0)

        if default_optimiser is None:
            self.factor_optimisers = factor_optimisers
            missing = set(self.factors) - self.factor_optimisers.keys()
            if missing:
                raise (
                    ValueError(
                        f"missing optimisers for {missing}, "
                        "pass a default_optimiser or add missing optimsers"
                    )
                )
        else:
            self.factor_optimisers = {
                factor: factor_optimisers.get(factor, default_optimiser)
                for factor in self.factors
            }

            
        self.ep_history = ep_history or EPHistory()
        self.diagnostics = EPDiagnostics()
        self.diagnostics.register_hierarchical_scales(factor_graph)

        # Per-factor count of consecutive failed updates; see
        # `_check_consecutive_failures`. Reset at the start of every `run`.
        self._consecutive_failures: Dict[Factor, int] = {}
        # Factors that raised at least once, and factors that landed at least
        # one successful update; together these identify a factor whose message
        # is still the one it started with. See `_stale_factor_warnings`.
        self._factors_raised: Set[Factor] = set()
        self._factors_skipped: Set[Factor] = set()
        self._factors_updated: Set[Factor] = set()

        self.visualiser = None
        if paths is None:
            try:
                paths = DirectoryPaths(identifier=str(Identifier(factor_graph)))
            except KeyError: 
                pass 

        self.paths = paths
        if self.paths:
            for optimiser in self.factor_optimisers.values():
                # optimiser.paths = optimiser_paths or self.paths
                optimiser.paths = self.paths

            with open(self.output_path / "graph.info", "w+") as f:
                f.write(self.factor_graph.info)

            self.visualiser = Visualise(self.ep_history, self.output_path)

    @classmethod
    def from_meanfield(
        cls,
        model_approx: EPMeanField,
        default_optimiser: Optional[AbstractFactorOptimiser] = None,
        factor_optimisers: Optional[Dict[Factor, AbstractFactorOptimiser]] = None,
        ep_history: Optional[EPHistory] = None,
        factor_order: Optional[List[Factor]] = None,
        paths: AbstractPaths = None,
        exact_fit_kws=None,
    ):
        factor_graph = model_approx.factor_graph
        factor_order = factor_order or factor_graph.factors

        factor_optimisers = factor_optimisers or {}
        factor_mean_field = model_approx.factor_mean_field
        for factor in factor_graph.factors:
            if factor not in factor_optimisers:
                factor_dist = factor_mean_field[factor]
                if factor.has_exact_projection(factor_dist):
                    factor_optimisers[factor] = ExactFactorFit(**(exact_fit_kws or {}))
                else:
                    factor_optimisers[factor] = default_optimiser

        return cls(
            factor_graph=factor_graph,
            default_optimiser=default_optimiser,
            factor_optimisers=factor_optimisers,
            ep_history=ep_history,
            factor_order=factor_order,
            paths=paths,
        )

    @property
    def output_path(self) -> Optional[Path]:
        """
        The path at which data will be output. Uses the name of the optimiser.

        If the path does not exist it is created.
        """
        if self.paths:
            path = Path(self.paths.output_path)
            os.makedirs(path, exist_ok=True)
            return path
        
        return None

    def _log_factor(self, factor: Factor):
        """
        Log information for the factor and its history.
        """
        factor_logger = logging.getLogger(factor.name)
        try:
            factor_history = self.ep_history[factor]
            if factor_history.history:
                log_evidence = factor_history.latest_update.log_evidence
                divergence = factor_history.kl_divergence()

                factor_logger.info(f"Log Evidence = {log_evidence}")
                factor_logger.info(f"KL Divergence = {divergence}")
        except exc.HistoryException as e:
            factor_logger.exception(e)

    def factor_step(self, factor_approx, optimiser, model_approx=None):
        return factor_step(factor_approx, optimiser, model_approx=model_approx)

    def _check_consecutive_failures(
        self,
        factor: Factor,
        status: Status,
        max_consecutive_failures: int,
        raised: bool,
    ) -> bool:
        """
        Track how many sweeps in a row a given factor's optimiser has *raised*.

        A single raise is survivable — the sweep continues on that factor's
        previous message. A factor that raises on *every* sweep is not going to
        start working, so once it has raised `max_consecutive_failures` times in
        a row there is nothing to gain by sweeping further: stop, warn, and let
        `run` return what it has. The result is still reported, but loudly
        qualified — see `_stale_factor_warnings`.

        Only raises are counted. A returned `StatusFlag.FAILURE` is an ordinary,
        recoverable outcome that EP absorbs by design — the Laplace optimiser
        returns one whenever its line search fails — and counting those would
        cut healthy fits short.

        Counting is per-factor and resets on any sweep that does not raise, so
        an intermittent failure (the observed case) never trips it.

        Parameters
        ----------
        raised
            Whether this factor's optimiser raised on this sweep. Read from the
            status `factor_step` returned, *before* the mean-field projection,
            which may legitimately overwrite the flag with `BAD_PROJECTION`.

        Returns
        -------
        True if this factor has now failed enough consecutive sweeps that the
        run should stop early.
        """
        if raised:
            self._factors_raised.add(factor)
            count = self._consecutive_failures.get(factor, 0) + 1
            self._consecutive_failures[factor] = count

            logger.warning(
                "Factor %s raised on %d consecutive step(s) "
                "(giving up on it at %d); continuing with its previous message. "
                "Latest messages: %s",
                factor.name,
                count,
                max_consecutive_failures,
                "; ".join(status.messages) or "(none)",
            )

            if max_consecutive_failures and count >= max_consecutive_failures:
                logger.warning(
                    "Factor %s has raised on %d consecutive steps; abandoning "
                    "further sweeps. Its message is whatever it last held, so "
                    "the returned mean field is not a posterior for this factor.",
                    factor.name,
                    count,
                )
                return True
        else:
            if status.updated:
                self._factors_updated.add(factor)
            else:
                # A skipped update (failed line search, bad projection): the
                # message is unchanged, so this does not count as an update.
                self._factors_skipped.add(factor)
            self._consecutive_failures.pop(factor, None)

        return False

    def _stale_factor_warnings(self) -> List[str]:
        """
        Warn about any factor whose message is still the one it started with.

        A per-factor failure count is not enough to detect this. When several
        factors raise, *nothing* in the mean field changes, so the KL step
        between sweeps is zero and `EPHistory` declares convergence — often
        within two sweeps, before any count reaches its threshold. The run then
        terminates "successfully" and the returned mean field holds the starting
        priors for those factors, dressed up as a posterior (PyAutoFit#1405).

        The result is still returned — callers with a partly-failed graph may
        well want the factors that did converge — but never quietly: these
        strings are logged as warnings and written into `ep_diagnostics.results`
        alongside the sigma-collapse warnings.

        The condition is deliberately narrow: a factor that raised, or whose
        update was skipped (failed line search, bad projection), at least once
        and *never once* updated. A factor that failed intermittently but landed
        at least one update has a real message and is not reported.
        """
        stale = (self._factors_raised | self._factors_skipped) - self._factors_updated
        if not stale:
            return []

        names = ", ".join(sorted(factor.name for factor in stale))
        return [
            f"STALE FACTORS: {names} never completed a single update — their "
            f"optimisers raised or their updates were skipped on every sweep. "
            f"The mean field returned for "
            f"them is the prior the fit started with, not a posterior. Do not "
            f"read those values as a result. Note that EP may also report "
            f"convergence in this state: with no factor updating, the KL step "
            f"between sweeps is zero, which is indistinguishable from having "
            f"converged."
        ]

    def _warn_stale_factors(self):
        """
        Log the stale-factor warnings, whether or not output paths are enabled.
        """
        for warning in self._stale_factor_warnings():
            logger.warning(warning)

    def run(
        self,
        model_approx: EPMeanField,
        max_steps: int = 100,
        log_interval: int = 10,
        visualise_interval: int = 100,
        output_interval: int = 10,
        max_consecutive_failures: int = 3,
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
        max_consecutive_failures
            How many consecutive sweeps a single factor's optimiser may *raise*
            on before the fit is aborted. One raise is not fatal — the sweep
            continues on that factor's previous message — but a factor that
            raises every sweep would leave EP converging on a stale message and
            reporting success. A returned failure status (e.g. a failed line
            search) is not counted. Set to 0 to never abort.

        Returns
        -------
        An updated approximation of the model
        """
        should_log = IntervalCounter(log_interval)
        should_visualise = IntervalCounter(visualise_interval)
        should_output = IntervalCounter(output_interval)

        self._consecutive_failures = {}
        self._factors_raised = set()
        self._factors_skipped = set()
        self._factors_updated = set()

        for _ in range(max_steps):
            _should_log = should_log()
            _should_visualise = should_visualise()
            _should_output = should_output()
            for factor, optimiser in self.factor_optimisers.items():
                factor_approx = model_approx.factor_approximation(factor)
                new_model_dist, status = self.factor_step(
                    factor_approx, optimiser, model_approx=model_approx,
                )
                raised = status.flag is StatusFlag.EXCEPTION
                model_approx, status = self.updater.update_model_approx(
                    new_model_dist, factor_approx, model_approx, status
                )
                self.diagnostics.snapshot(factor, model_approx, status)
                if self._check_consecutive_failures(
                    factor, status, max_consecutive_failures, raised=raised
                ):
                    break
                if status and _should_log:
                    self._log_factor(factor)

                if self.ep_history(factor, model_approx, status):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

            else:  # If no break do next iteration
                if self.visualiser and _should_visualise:
                    self.visualiser()
                if self.output_path and _should_output:
                    self._output_results(model_approx)
                    self._output_diagnostics()
                continue
            break  # stop iterations

        if self.paths:
            self.visualiser()
            self._output_results(model_approx)
            self._output_diagnostics(final=True, model_approx=model_approx)
        self._warn_sigma_collapse()
        self._warn_stale_factors()

        return model_approx

    def _output_results(self, model_approx: EPMeanField):
        """
        Save the graph.results text
        """
        if self.paths:
            with open(self.output_path / "graph.results", "w+") as f:
                f.write(self.factor_graph.make_results_text(model_approx))

    def _output_diagnostics(
        self, final: bool = False, model_approx: Optional[EPMeanField] = None
    ):
        """
        Write the diagnostics CSVs (`ep_history.csv`, `mean_field_history.csv`)
        and the mean-field evolution plot to the output folder. At the end of
        a run (``final=True``) also write ``ep_diagnostics.results``: the
        mean-field summary table plus any sigma-collapse warnings.
        """
        if not self.output_path:
            return
        self.diagnostics.write(self.output_path)
        self.diagnostics.plot(self.output_path)

        if final and model_approx is not None:
            warnings_list = (
                self._stale_factor_warnings() + check_sigma_collapse(self.diagnostics)
            )
            with open(self.output_path / "ep_diagnostics.results", "w+") as f:
                f.write(mean_field_summary(model_approx.mean_field))
                f.write("\n")
                if warnings_list:
                    f.write("\nWARNINGS\n\n")
                    f.write("\n".join(warnings_list))
                    f.write("\n")

    def _warn_sigma_collapse(self):
        """
        Log any sigma-collapse warnings (PyAutoFit #1332 F10) at the end of a
        run — emitted regardless of whether output paths are enabled.
        """
        for warning in check_sigma_collapse(self.diagnostics):
            logger.warning(warning)


class ParallelEPOptimiser(EPOptimiser):
    def __init__(
        self,
        factor_graph: FactorGraph,
        n_cores: int,
        default_optimiser: Optional[AbstractFactorOptimiser] = None,
        factor_optimisers: Optional[Dict[Factor, AbstractFactorOptimiser]] = None,
        ep_history: Optional[EPHistory] = None,
        factor_order: Optional[List[Factor]] = None,
        paths: AbstractPaths = None,
        updater: Optional[ApproxUpdater] = None,
    ):
        """
        Optimise a factor graph.

        Optimises all factors simultaneously on parallel processes, combines the results
        and repeats.

        Parameters
        ----------
        factor_graph
            A graph describing the relationships between multiple factors
        n_cores
            How many cores are available? The main process takes one core so the
            multiprocessing pool has n_cores - 1 processes. Should be at least 3
        default_optimiser
            An optimiser that is used if no specific optimiser is provided for a factor
        factor_optimisers
            Specific optimisers used to optimise each factor
        ep_history
            Optionally specify an alternate history
        factor_order
            The factors in the graph but placed in the order in which they should
            be optimised
        paths
            Optionally define how data should be output
        """

        if n_cores < 3:
            raise AssertionError(
                "With less than three cores it is better to use EPOptimiser"
            )

        super().__init__(
            factor_graph=factor_graph,
            default_optimiser=default_optimiser,
            factor_optimisers=factor_optimisers,
            ep_history=ep_history,
            factor_order=factor_order,
            paths=paths,
            updater=updater,
        )
        self.pool = fork_context().Pool(n_cores - 1)

    def run(
        self,
        model_approx: EPMeanField,
        max_steps: int = 100,
        log_interval: int = 10,
        visualise_interval: int = 100,
        output_interval: int = 10,
        max_consecutive_failures: int = 3,
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
        max_consecutive_failures
            How many consecutive sweeps a single factor's optimiser may raise on
            before the fit is aborted. See `EPOptimiser.run`.

        Returns
        -------
        An updated approximation of the model
        """
        should_log = IntervalCounter(log_interval)
        should_visualise = IntervalCounter(visualise_interval)
        should_output = IntervalCounter(output_interval)

        self._consecutive_failures = {}
        self._factors_raised = set()
        self._factors_skipped = set()
        self._factors_updated = set()

        for _ in range(max_steps):
            _should_log = should_log()
            _should_visualise = should_visualise()
            _should_output = should_output()

            factor_approx_optimisers = [
                (model_approx.factor_approximation(factor), optimiser)
                for factor, optimiser in self.factor_optimisers.items()
            ]

            new_dist_statuses = self.pool.starmap(factor_step, factor_approx_optimisers)

            for (factor_approx, _), (new_model_dist, status) in zip(
                factor_approx_optimisers, new_dist_statuses
            ):
                raised = status.flag is StatusFlag.EXCEPTION
                model_approx, status = self.updater.update_model_approx(
                    new_model_dist, factor_approx, model_approx, status
                )
                factor = factor_approx.factor
                self.diagnostics.snapshot(factor, model_approx, status)
                if self._check_consecutive_failures(
                    factor, status, max_consecutive_failures, raised=raised
                ):
                    break
                if status and _should_log:
                    self._log_factor(factor)

                if self.ep_history(factor, model_approx, status):
                    logger.info("Terminating optimisation")
                    break  # callback controls convergence

            else:  # If no break do next iteration
                if self.visualiser and _should_visualise:
                    self.visualiser()
                if self.output_path and _should_output:
                    self._output_results(model_approx)
                    self._output_diagnostics()
                continue
            break  # stop iterations

        if self.visualiser:
            self.visualiser()
        if self.output_path:
            self._output_results(model_approx)
            self._output_diagnostics(final=True, model_approx=model_approx)
        self._warn_sigma_collapse()
        self._warn_stale_factors()

        return model_approx
