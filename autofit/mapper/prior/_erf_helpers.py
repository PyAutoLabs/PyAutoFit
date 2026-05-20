"""Direct-`ndtr` primitives for hot prior-transform paths.

Replaces `scipy.stats.norm.cdf` / `norm.ppf` (and their `jax.scipy.stats`
counterparts) with direct calls to `scipy.special.ndtr` / `ndtri` — the
Cephes routines that scipy.stats wraps. Bit-exact equivalent on both
NumPy and JAX backends, but skips the
`scipy.stats._distn_infrastructure` wrapper overhead — which the
graphical-ep-scale-up cProfile baseline showed was the #1 hotspot in
`TruncatedGaussianPrior.value_for` (~33% of total wall time at N=10).

See PyAutoFit issue #1284 for the motivating measurements.
"""

import numpy as np


def _norm_cdf(z, xp):
    """Standard-normal CDF (== ``scipy.stats.norm.cdf(z)`` to ULPs)."""
    if xp is np:
        from scipy.special import ndtr
    else:
        from jax.scipy.special import ndtr
    return ndtr(z)


def _norm_ppf(p, xp):
    """Standard-normal PPF (== ``scipy.stats.norm.ppf(p)`` to ULPs)."""
    if xp is np:
        from scipy.special import ndtri
    else:
        from jax.scipy.special import ndtri
    return ndtri(p)


def truncated_normal_value_for(unit, mean, sigma, lower_limit, upper_limit, xp=np):
    """Inverse-CDF mapping for a truncated normal distribution.

    Returns ``mean + sigma * Phi^{-1}(Phi(a) + unit * (Phi(b) - Phi(a)))``
    where ``a = (lower_limit - mean) / sigma`` and
    ``b = (upper_limit - mean) / sigma``.

    Used by ``TruncatedGaussianPrior.value_for`` and
    ``TruncatedNormalMessage.value_for`` to share a single
    `scipy.special.erf`-based code path on both NumPy and JAX backends.

    Parameters
    ----------
    unit
        Unit-cube draw(s) in ``[0, 1]``. Scalar or array.
    mean, sigma
        Underlying-Gaussian mean and standard deviation.
    lower_limit, upper_limit
        Truncation bounds. ``-inf`` / ``+inf`` are supported.
    xp
        Array module: ``numpy`` (default) or ``jax.numpy``. Determines
        whether ``scipy.special`` or ``jax.scipy.special`` is used.

    Returns
    -------
    Physical sample(s) drawn from the truncated normal.
    """
    a = (lower_limit - mean) / sigma
    b = (upper_limit - mean) / sigma

    lower_cdf = _norm_cdf(a, xp)
    upper_cdf = _norm_cdf(b, xp)
    truncated_cdf = lower_cdf + unit * (upper_cdf - lower_cdf)

    x_standard = _norm_ppf(truncated_cdf, xp)
    return mean + sigma * x_standard
