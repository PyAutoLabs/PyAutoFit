"""
Diagnostics and monitoring for expectation-propagation fits.

An EP fit is a sequence of per-factor updates to a global mean-field
approximation. This module records that sequence so it can be inspected
while the fit runs and analysed after it finishes:

- ``EPDiagnostics`` — collects one snapshot per factor update (status,
  factor log-evidence, KL step size, and every variable's mean/std) and
  writes them as machine-readable CSVs and a matplotlib evolution plot.
- ``mean_field_summary`` — a human-readable table of a mean field,
  suitable for printing at the end of any example or script.
- ``check_sigma_collapse`` — guards against the known pathology where
  repeated undamped EP updates over-count shared information and every
  sigma collapses towards zero around the starting point (rather than
  the data); see PyAutoFit issue #1332 (F10).

Outputs written to the EP output folder by ``EPOptimiser`` when paths
are enabled:

- ``ep_history.csv`` — one row per factor update:
  ``step, factor, success, updated, flag, log_evidence, kl_divergence``
- ``mean_field_history.csv`` — one row per (factor update, variable):
  ``step, factor, variable, mean, std``
- ``mean_field_evolution.png`` — per-variable mean ± std vs update.
"""
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _scalar_mean_std(message) -> Tuple[float, float]:
    """
    Reduce a message's mean/std to scalars for logging.

    Scalar variables pass through; array (plated) variables are
    summarised as the mean of their means and the maximum of their
    stds — the max is chosen so a single collapsing element is not
    hidden by averaging.

    The std is derived from ``variance``, which every message family
    exposes (``TransformedMessage`` — e.g. any ``UniformPrior``-backed
    variable — has no ``std`` attribute).
    """
    mean = np.asarray(message.mean, dtype=float)
    std = np.sqrt(np.asarray(message.variance, dtype=float))
    return float(mean.mean()), float(std.max())


class EPDiagnostics:
    def __init__(self):
        """
        Collects a per-factor-update record of an EP fit.

        ``EPOptimiser`` calls ``snapshot`` after every factor update.
        Each snapshot stores the update's status, the current global
        log-evidence, the KL divergence of the global mean field from
        the previous snapshot (the "step size" EP convergence watches),
        and every variable's (mean, std).
        """
        self.factor_rows: List[dict] = []
        self.variable_rows: List[dict] = []
        self._previous_mean_field = None
        self._step = 0

    def snapshot(self, factor, model_approx, status) -> None:
        """
        Record the state of the approximation after one factor update.

        Parameters
        ----------
        factor
            The factor that was just updated.
        model_approx
            The ``EPMeanField`` after the update.
        status
            The update's ``Status``.
        """
        mean_field = model_approx.mean_field

        try:
            log_evidence = float(model_approx.log_evidence)
        except Exception as e:  # diagnostics must never kill the fit
            logger.warning(f"EPDiagnostics: log_evidence failed: {e}")
            log_evidence = float("nan")

        if self._previous_mean_field is not None:
            try:
                kl = float(np.sum(mean_field.kl(self._previous_mean_field)))
            except Exception as e:
                logger.warning(f"EPDiagnostics: kl computation failed: {e}")
                kl = float("nan")
        else:
            kl = float("nan")

        self.factor_rows.append(
            {
                "step": self._step,
                "factor": factor.name,
                "success": status.success,
                "updated": status.updated,
                "flag": status.flag.name,
                "log_evidence": log_evidence,
                "kl_divergence": kl,
            }
        )

        for variable, message in mean_field.items():
            mean, std = _scalar_mean_std(message)
            self.variable_rows.append(
                {
                    "step": self._step,
                    "factor": factor.name,
                    "variable": variable.name,
                    "mean": mean,
                    "std": std,
                }
            )

        self._previous_mean_field = mean_field
        self._step += 1

    @property
    def variable_history(self) -> Dict[str, List[Tuple[int, float, float]]]:
        """
        The recorded history keyed by variable name: a list of
        (step, mean, std) tuples in update order.
        """
        history: Dict[str, List[Tuple[int, float, float]]] = {}
        for row in self.variable_rows:
            history.setdefault(row["variable"], []).append(
                (row["step"], row["mean"], row["std"])
            )
        return history

    def write(self, output_path: Path) -> None:
        """
        Write ``ep_history.csv`` and ``mean_field_history.csv`` to the
        output folder (overwriting — the CSVs always reflect the full
        history so far, so they can be watched during a run).
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for filename, rows in (
            ("ep_history.csv", self.factor_rows),
            ("mean_field_history.csv", self.variable_rows),
        ):
            if not rows:
                continue
            with open(output_path / filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    def plot(self, output_path: Path) -> None:
        """
        Write ``mean_field_evolution.png``: each variable's mean with a
        ± std band against factor-update number.
        """
        if not self.variable_rows:
            return

        import matplotlib.pyplot as plt

        history = self.variable_history

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, rows in history.items():
            steps, means, stds = map(np.array, zip(*rows))
            (line,) = ax.plot(steps, means, label=name)
            ax.fill_between(
                steps, means - stds, means + stds, alpha=0.2, color=line.get_color()
            )
        ax.set_xlabel("factor update")
        ax.set_ylabel("posterior mean ± std")
        ax.set_title("Mean-field evolution")
        ax.legend(fontsize="small")
        fig.savefig(str(Path(output_path) / "mean_field_evolution.png"))
        plt.close(fig)


def mean_field_summary(mean_field) -> str:
    """
    A human-readable summary table of a mean field.

    One row per variable: name, posterior mean and std (arrays
    summarised as mean-of-means / max-std). Call this at the end of an
    EP example to display the approximate posterior:

        print(mean_field_summary(result.updated_ep_mean_field.mean_field))
    """
    rows = []
    for variable, message in mean_field.items():
        mean, std = _scalar_mean_std(message)
        rows.append((variable.name, mean, std))

    name_width = max([len(name) for name, _, _ in rows] + [len("variable")])
    lines = [
        f"{'variable':<{name_width}}  {'mean':>14}  {'std':>14}",
        "-" * (name_width + 32),
    ]
    for name, mean, std in rows:
        lines.append(f"{name:<{name_width}}  {mean:>14.6g}  {std:>14.6g}")
    return "\n".join(lines)


def check_sigma_collapse(
    diagnostics: EPDiagnostics,
    std_floor: float = 1e-8,
    monotone_steps: int = 5,
    shrink_factor: float = 1e-3,
) -> List[str]:
    """
    Detect the EP sigma-collapse pathology (PyAutoFit #1332, F10).

    Repeated undamped EP updates can over-count shared-variable
    information: every std shrinks monotonically towards zero around
    the *starting* means, while the KL convergence criterion never
    triggers. This check flags a variable when either:

    - its latest std is below ``std_floor``, or
    - its std has shrunk monotonically for the last ``monotone_steps``
      updates *and* by more than a factor ``1 / shrink_factor`` overall.

    Returns
    -------
    A list of warning strings, one per flagged variable (empty when
    healthy). ``EPOptimiser`` logs these and appends them to the
    results text at the end of a run.
    """
    warnings_list = []

    for name, rows in diagnostics.variable_history.items():
        stds = np.array([std for _, _, std in rows], dtype=float)
        if len(stds) == 0:
            continue

        if stds[-1] < std_floor:
            warnings_list.append(
                f"sigma-collapse: variable '{name}' has std {stds[-1]:.3g} "
                f"below the floor {std_floor:.1g} — the fit has likely "
                f"collapsed to a point (see PyAutoFit #1332 F10; consider "
                f"damping, e.g. delta < 1, or per-factor sampler optimisers)."
            )
            continue

        if len(stds) > monotone_steps:
            tail = stds[-(monotone_steps + 1):]
            if np.all(np.diff(tail) < 0) and stds[-1] < shrink_factor * stds[0]:
                warnings_list.append(
                    f"sigma-collapse: variable '{name}' std has shrunk "
                    f"monotonically over the last {monotone_steps} updates to "
                    f"{stds[-1]:.3g} ({stds[-1] / stds[0]:.1e} of its initial "
                    f"value) — possible information over-counting (PyAutoFit "
                    f"#1332 F10)."
                )

    return warnings_list
