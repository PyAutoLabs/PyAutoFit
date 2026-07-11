import importlib.util
import os
import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search.mcmc.blackjax.nuts.search import (
    _times_from_positions,
)

# `_times_from_positions` runs blackjax (which pulls jax); both ship via the
# `[optional]` extras. The Python-version matrix installs autofit without those
# extras, so skip there rather than fail with `No module named 'blackjax'`.
requires_blackjax = pytest.mark.skipif(
    importlib.util.find_spec("blackjax") is None,
    reason="requires blackjax (installed via the [optional] extras; absent on the matrix env)",
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__explicit_params():
    search = af.BlackJAXNUTS(
        num_warmup=321,
        num_samples=987,
        num_chains=1,
        target_accept=0.9,
        max_num_doublings=8,
        seed=2024,
        initializer=af.InitializerBall(lower_limit=0.2, upper_limit=0.8),
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=True,
            check_size=42,
            required_length=21,
            change_threshold=0.05,
        ),
        number_of_cores=1,
    )

    assert search.num_warmup == 321
    assert search.num_samples == 987
    assert search.num_chains == 1
    assert search.target_accept == 0.9
    assert search.max_num_doublings == 8
    assert search.seed == 2024
    assert isinstance(search.initializer, af.InitializerBall)
    assert search.initializer.lower_limit == 0.2
    assert search.initializer.upper_limit == 0.8
    assert search.auto_correlation_settings.check_for_convergence is True
    assert search.auto_correlation_settings.check_size == 42


def test__defaults():
    search = af.BlackJAXNUTS()

    assert search.num_warmup == 500
    assert search.num_samples == 1000
    assert search.num_chains == 1
    assert search.target_accept == 0.8
    assert search.max_num_doublings == 10
    assert isinstance(search.initializer, af.InitializerBall)
    # Convergence checking is intentionally OFF by default for NUTS — the
    # sampler runs to a fixed budget after warmup tunes the kernel.
    assert search.auto_correlation_settings.check_for_convergence is False


def test__multi_chain_not_implemented():
    with pytest.raises(NotImplementedError):
        af.BlackJAXNUTS(num_chains=2)


def test__test_mode_reduces_iterations(monkeypatch):
    # autofit.non_linear.test_mode reads PYAUTO_TEST_MODE; ``apply_test_mode``
    # is only triggered inside ``__init__`` when that's set, so we must patch
    # the env var (not after-the-fact mutate the search instance).
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.BlackJAXNUTS(num_warmup=10000, num_samples=10000)

    assert search.num_warmup == 20
    assert search.num_samples == 20


def test__identifier_fields_distinguish_run_shape():
    a = af.BlackJAXNUTS(num_warmup=100, num_samples=500, num_chains=1)
    b = af.BlackJAXNUTS(num_warmup=200, num_samples=500, num_chains=1)
    c = af.BlackJAXNUTS(num_warmup=100, num_samples=999, num_chains=1)

    # __identifier_fields__ is a class attribute; make sure it covers all
    # three knobs. The autofit identifier hash uses these to keep distinct
    # runs from colliding on the same output path.
    assert af.BlackJAXNUTS.__identifier_fields__ == (
        "num_warmup",
        "num_samples",
        "num_chains",
    )

    # Instance attributes should round-trip exactly.
    assert (a.num_warmup, a.num_samples, a.num_chains) == (100, 500, 1)
    assert (b.num_warmup, b.num_samples, b.num_chains) == (200, 500, 1)
    assert (c.num_warmup, c.num_samples, c.num_chains) == (100, 999, 1)


@requires_blackjax
def test__times_from_positions_clamps_low_ess():
    # Construct a degenerate "chain" — every sample is identical → ESS
    # collapses; the clamp must keep ``times`` finite and equal to N.
    rng = np.random.default_rng(0)
    n_dim = 3
    n_samples = 200
    positions = np.tile(rng.normal(size=(1, n_dim)), (n_samples, 1))

    times = _times_from_positions(positions)

    assert times.shape == (n_dim,)
    assert np.all(np.isfinite(times))
    # With ESS clamped to >= 1, the synthetic ``times`` cannot exceed N.
    assert np.all(times <= n_samples + 1e-9)


@requires_blackjax
def test__times_from_positions_independent_chain():
    # Independent draws → ESS ≈ N, so τ ≈ 1.
    rng = np.random.default_rng(1)
    n_samples = 2000
    n_dim = 3
    positions = rng.normal(size=(n_samples, n_dim))

    times = _times_from_positions(positions)

    # Expect each per-param ``τ`` to be O(1) — well under N. We use a loose
    # bound (10) so the test isn't flaky on small-sample variance.
    assert np.all(times < 10.0)
