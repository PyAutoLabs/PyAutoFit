"""
The one conversion from a backend's raw samples to a ``Samples`` object.

Every search turns its backend's internal state into the same five things: the
parameter vectors, a log likelihood or a log posterior per vector, a weight per
vector, and the backend-specific ``samples_info`` entries. A search expresses that
as a ``RawSamples`` (``NonLinearSearch.raw_samples_from``) and ``samples_from_raw``
does everything else, once, for every search:

- the log prior of every vector, through one helper (``log_prior_list_from``);
- the log likelihood from a log posterior (``log_likelihood = log_posterior -
  log_prior``) when the backend reports a posterior;
- the length checks that keep parameters, log values and weights in correspondence
  (PyAutoFit#1628);
- recognition of the backends' invalid-model sentinels (``-1e99``, ``-1e30``,
  ``-inf``) through one predicate (``is_invalid``);
- the injection of ``samples_info["time"]`` and, through the ``Samples`` class,
  ``samples_info["class_path"]``.

Representations are preserved (decision D13 of the search-extensibility epic).
Parameter vectors, log likelihoods and weights reach ``Sample.from_lists`` exactly
as the backend produced them, in the backend's order: equally weighted draws keep
weight 1, weighted samples keep the weights their search computed (normalised or
not), nothing is renormalised and no sentinel value is rewritten. The ordering of
the vectors is the search's to specify (for example ``ChainPosterior`` flattens
draw-major for emcee and zeus and chain-major for BlackJAX NUTS), and diagnostics
belong to the search's ``info`` entries, which are passed through untouched.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from autofit import exc

logger = logging.getLogger(__name__)

INVALID_SENTINELS = (-1.0e99, -1.0e30, -float("inf"))
"""
The values backends use for a model whose likelihood is invalid: ``-1e99``
(dynesty and the post-fit ``Fitness`` resample value), ``-1e30`` (NSS sampling) and
``-inf`` (emcee, zeus, BlackJAX, the MLE searches).
"""

INVALID_THRESHOLD = -1.0e30
"""
Every log likelihood at or below this value is one of ``INVALID_SENTINELS`` (or a
value derived from one, such as ``-1e99 - log_prior``), never a likelihood a real
model evaluates to.
"""


def is_invalid(value) -> bool:
    """
    Whether a log likelihood or log posterior is a backend's invalid-model sentinel.

    This is the single recognition rule for all three sentinels; the value itself is
    never rewritten, so stored samples keep the representation their backend gave
    them. NaN (the MultiStart diagnostic lanes, which carry no likelihood) is not a
    sentinel.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(value):
        return False
    return value <= INVALID_THRESHOLD


def invalid_count(values: Sequence) -> int:
    """
    The number of entries of ``values`` that are invalid-model sentinels.
    """
    return sum(1 for value in values if is_invalid(value))


def log_prior_list_from(model, parameter_lists: Sequence[Sequence[float]]) -> List:
    """
    The log prior of every parameter vector: the one log-prior helper of sample
    conversion (``model.log_prior_list_from``, the sum of each vector's per-prior log
    densities).
    """
    return model.log_prior_list_from(parameter_lists=parameter_lists)


def _as_list(values):
    """
    ``values`` as a list, without changing the type of its elements unless it is a
    NumPy array (whose ``tolist`` gives plain Python scalars, as every search did
    before the adapter).
    """
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


@dataclass(frozen=True)
class RawSamples:
    """
    What a backend produced, before any conversion.

    Parameters
    ----------
    parameters
        The parameter vectors, ``(N, D)``, in physical coordinates and in the
        search's sample order.
    log_likelihood
        The log likelihood of every vector, ``(N,)``. Give either this or
        ``log_posterior``.
    log_posterior
        The log posterior of every vector, ``(N,)``; the log likelihood is then
        ``log_posterior - log_prior``.
    weights
        The weight of every vector, ``(N,)``, exactly as stored. ``None`` means
        equally weighted draws (weight 1 each). Never renormalised here.
    info
        The search-specific ``samples_info`` entries (diagnostics, settings,
        counters). ``time`` is injected by ``samples_from_raw`` when absent.
    log_prior
        The log prior of every vector when the backend already has it; ``None``
        (the usual case) computes it with ``log_prior_list_from``.
    samples_kwargs
        Extra keyword arguments of the ``Samples`` class (for example
        ``SamplesMCMC``'s ``auto_correlations``).
    label
        The backend's name, used in error messages.
    """

    parameters: Any
    log_likelihood: Optional[Any] = None
    log_posterior: Optional[Any] = None
    weights: Optional[Any] = None
    info: Dict[str, Any] = field(default_factory=dict)
    log_prior: Optional[Any] = None
    samples_kwargs: Dict[str, Any] = field(default_factory=dict)
    label: str = "search"

    def __post_init__(self):
        if (self.log_likelihood is None) == (self.log_posterior is None):
            raise exc.SamplesException(
                f"The raw samples of {self.label} must give exactly one of "
                "log_likelihood and log_posterior."
            )


def _check_length(label: str, n_parameters: int, values, name: str):
    """
    Raise when a per-sample column does not have one entry per parameter vector.
    """
    if values is not None and len(values) != n_parameters:
        raise exc.SamplesException(
            f"The number of {label} parameter samples does not match the number of "
            f"{name} values returned by the sampler: "
            f"{n_parameters} parameter samples versus "
            f"{len(values)} {name} values. "
            f"The parameters and {name}s are therefore not in correspondence "
            "and the samples cannot be built."
        )


def samples_from_raw(
    model,
    raw: RawSamples,
    samples_cls: Type,
    time: Optional[float] = None,
    **samples_kwargs,
):
    """
    Build the ``Samples`` object of ``raw`` (see the module docstring for what is
    done here, once, for every search).

    Parameters
    ----------
    model
        The model the parameter vectors belong to.
    raw
        The backend's raw samples.
    samples_cls
        The ``Samples`` class to build (``Samples``, ``SamplesMCMC``,
        ``SamplesNest``, ``SamplesSMC``, ``NSSamples``, ...). Its constructor stamps
        ``samples_info["class_path"]``, which the aggregator and
        ``DirectoryPaths.samples`` use to rebuild it.
    time
        The search's run time, injected as ``samples_info["time"]`` unless the
        search's own ``info`` already gives one.
    samples_kwargs
        Extra keyword arguments of ``samples_cls``, merged over
        ``raw.samples_kwargs``.
    """
    from autofit.non_linear.samples.sample import Sample

    parameter_lists = _as_list(raw.parameters)
    n = len(parameter_lists)

    if raw.log_posterior is not None:
        log_posterior_list = _as_list(raw.log_posterior)
        _check_length(raw.label, n, log_posterior_list, "log posterior")
    else:
        log_posterior_list = None

    log_likelihood_list = _as_list(raw.log_likelihood)
    _check_length(raw.label, n, log_likelihood_list, "log likelihood")

    if raw.log_prior is not None:
        log_prior_list = _as_list(raw.log_prior)
        _check_length(raw.label, n, log_prior_list, "log prior")
    else:
        log_prior_list = log_prior_list_from(
            model=model, parameter_lists=parameter_lists
        )

    if log_likelihood_list is None:
        log_likelihood_list = [
            log_posterior - log_prior
            for log_posterior, log_prior in zip(log_posterior_list, log_prior_list)
        ]

    if raw.weights is None:
        weight_list = n * [1.0]
    else:
        weight_list = _as_list(raw.weights)
        _check_length(raw.label, n, weight_list, "weight")

    invalid = invalid_count(log_likelihood_list)
    if invalid:
        logger.debug(
            f"{invalid} of the {n} {raw.label} samples carry an invalid-model "
            "sentinel log likelihood; they are kept as stored."
        )

    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameter_lists,
        log_likelihood_list=log_likelihood_list,
        log_prior_list=log_prior_list,
        weight_list=weight_list,
    )

    info = dict(raw.info)
    if "time" not in info:
        info["time"] = time

    return samples_cls(
        model=model,
        sample_list=sample_list,
        samples_info=info,
        **{**raw.samples_kwargs, **samples_kwargs},
    )


class ChainPosterior:
    """
    The equally weighted draws of an ensemble or multi-chain MCMC backend, thinned and
    with burn-in removed: the one implementation shared by Emcee, Zeus and
    BlackJAX NUTS.

    ``get_chain(discard, thin)`` and ``get_log_prob(discard, thin)`` return the flat
    ``(N, D)`` draws and ``(N,)`` log values for a burn-in and thinning, so emcee's and
    zeus's own ``get_chain(discard=..., thin=..., flat=True)`` slicing is used
    unchanged. ``from_arrays`` builds the getters for a backend that only stores
    arrays.
    """

    def __init__(
        self,
        get_chain: Callable[[int, int], np.ndarray],
        get_log_prob: Callable[[int, int], np.ndarray],
        label: str = "MCMC",
    ):
        self.get_chain = get_chain
        self.get_log_prob = get_log_prob
        self.label = label

    @classmethod
    def from_sampler(cls, sampler, label: str = "MCMC") -> "ChainPosterior":
        """
        For an emcee or zeus sampler (or backend), via their ``get_chain`` and
        ``get_log_prob`` with ``flat=True``.
        """
        return cls(
            get_chain=lambda discard, thin: sampler.get_chain(
                discard=discard, thin=thin, flat=True
            ),
            get_log_prob=lambda discard, thin: sampler.get_log_prob(
                discard=discard, thin=thin, flat=True
            ),
            label=label,
        )

    @classmethod
    def from_arrays(
        cls,
        chain: np.ndarray,
        log_prob: np.ndarray,
        chain_major: bool = False,
        label: str = "MCMC",
    ) -> "ChainPosterior":
        """
        For draws stored as arrays of shape ``(n_steps, n_chains, n_dim)`` and
        ``(n_steps, n_chains)``.

        Burn-in and thinning slice the step axis as emcee does
        (``[discard + thin - 1 :: thin]``, the whole chain for ``discard=0,
        thin=1``). ``chain_major=True`` flattens chain by chain (all of chain 0's
        draws in order, then chain 1's, ...); otherwise draw by draw, as emcee's
        ``flat=True`` does.
        """
        chain = np.asarray(chain)
        log_prob = np.asarray(log_prob)

        def _flat(values, discard, thin):
            values = values[discard + thin - 1 :: thin]
            if chain_major:
                values = np.moveaxis(values, 0, 1)
            return values.reshape((-1,) + values.shape[2:])

        return cls(
            get_chain=lambda discard, thin: _flat(chain, discard, thin),
            get_log_prob=lambda discard, thin: _flat(log_prob, discard, thin),
            label=label,
        )

    def thin(self, discard: int, thin: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The draws and log values after removing ``discard`` burn-in steps and keeping
        every ``thin``-th step.

        When that leaves no draws (an unconverged chain whose auto-correlation time
        is comparable to its length), the whole chain is used instead, with a log
        message, so the samples can still be built and inspected. The log values are
        always requested with the same ``discard`` and ``thin`` as the draws, so the
        two stay in correspondence (PyAutoFit#1628).
        """
        draws = self.get_chain(discard, thin)

        if len(draws) == 0:
            logger.info(
                f"""
                After thinning the {self.label} samples in order to remove burn-in, no samples were left.

                To create a samples object containing samples, so that the code can continue and results
                can be inspected, the full list of samples before removing burn-in has been used. This may
                indicate that the sampler has not converged and therefore your results may not be reliable.

                To fix this, run {self.label} with more steps to ensure convergence is achieved or change the
                auto correlation settings to be less aggressive in thinning samples.
                """
            )
            discard = 0
            thin = 1
            draws = self.get_chain(discard, thin)

        return draws, self.get_log_prob(discard, thin)
