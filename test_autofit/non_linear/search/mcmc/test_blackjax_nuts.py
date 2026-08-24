import importlib.util
import os
import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search.mcmc.blackjax.nuts.search import (
    _build_search_internal,
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


class _FakeSamples:
    """A minimal `Samples`-like object -- just enough for
    `inverse_mass_matrix_kind_from` (construction-time classification only;
    no resolution) to recognise it as a "result" kind."""

    def __init__(self, covariance_matrix):
        self.covariance_matrix = covariance_matrix


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
    assert search.inverse_mass_matrix == "none"
    assert search.mass_matrix_shrinkage == 0.0
    assert search.share_adaptation is False


def test__multi_chain_construction():
    search = af.BlackJAXNUTS(num_chains=4)

    assert search.num_chains == 4


class TestInverseMassMatrix:
    def test__none(self):
        search = af.BlackJAXNUTS(inverse_mass_matrix=None)
        assert search.inverse_mass_matrix == "none"

    def test__diagonal_string(self):
        search = af.BlackJAXNUTS(inverse_mass_matrix="diagonal")
        assert search.inverse_mass_matrix == "diagonal"

    def test__dense_string(self):
        search = af.BlackJAXNUTS(inverse_mass_matrix="dense")
        assert search.inverse_mass_matrix == "dense"

    def test__1d_array(self):
        search = af.BlackJAXNUTS(inverse_mass_matrix=np.array([1.0, 2.0, 3.0]))
        assert search.inverse_mass_matrix == "array"

    def test__2d_array(self):
        search = af.BlackJAXNUTS(inverse_mass_matrix=np.eye(3))
        assert search.inverse_mass_matrix == "array"

    def test__result_like(self):
        fake_samples = _FakeSamples(covariance_matrix=np.eye(3))
        search = af.BlackJAXNUTS(inverse_mass_matrix=fake_samples)
        assert search.inverse_mass_matrix == "result"

    def test__bad_string_raises(self):
        with pytest.raises(ValueError):
            af.BlackJAXNUTS(inverse_mass_matrix="not_a_valid_kind")

    def test__bad_type_raises(self):
        with pytest.raises(ValueError):
            af.BlackJAXNUTS(inverse_mass_matrix=object())


def test__test_mode_reduces_iterations(monkeypatch):
    # autofit.non_linear.test_mode reads PYAUTO_TEST_MODE; ``apply_test_mode``
    # is only triggered inside ``__init__`` when that's set, so we must patch
    # the env var (not after-the-fact mutate the search instance).
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.BlackJAXNUTS(num_warmup=10000, num_samples=10000, num_chains=8)

    assert search.num_warmup == 20
    assert search.num_samples == 20
    # `apply_test_mode` clamps `num_chains` to at most 2 so test-mode fits
    # stay fast even when a warm-started multi-chain config is requested.
    assert search.num_chains == 2


def test__test_mode_leaves_single_chain_alone(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    search = af.BlackJAXNUTS(num_chains=1)

    assert search.num_chains == 1


def test__identifier_fields_distinguish_run_shape():
    a = af.BlackJAXNUTS(num_warmup=100, num_samples=500, num_chains=1)
    b = af.BlackJAXNUTS(num_warmup=200, num_samples=500, num_chains=1)
    c = af.BlackJAXNUTS(num_warmup=100, num_samples=999, num_chains=1)
    d = af.BlackJAXNUTS(
        num_warmup=100, num_samples=500, num_chains=1, inverse_mass_matrix="dense"
    )

    # __identifier_fields__ is a class attribute; make sure it covers every
    # knob that changes the run's shape. The autofit identifier hash uses
    # these to keep distinct runs from colliding on the same output path.
    assert af.BlackJAXNUTS.__identifier_fields__ == (
        "num_warmup",
        "num_samples",
        "num_chains",
        "inverse_mass_matrix",
    )

    # Instance attributes should round-trip exactly.
    assert (a.num_warmup, a.num_samples, a.num_chains) == (100, 500, 1)
    assert (b.num_warmup, b.num_samples, b.num_chains) == (200, 500, 1)
    assert (c.num_warmup, c.num_samples, c.num_chains) == (100, 999, 1)
    assert a.inverse_mass_matrix == "none"
    assert d.inverse_mass_matrix == "dense"

    from autofit.mapper.identifier import Identifier

    assert str(Identifier(a)) != str(Identifier(d))


@requires_blackjax
def test__times_from_positions_clamps_low_ess():
    # Construct a degenerate "chain" — every sample is identical → ESS
    # collapses; the clamp must keep ``times`` finite and equal to N.
    # Positions carry the `(n_samples, n_chains, n_dim)` search_internal
    # layout, so a single chain still gets an explicit size-1 chain axis.
    rng = np.random.default_rng(0)
    n_dim = 3
    n_samples = 200
    positions = np.tile(rng.normal(size=(1, 1, n_dim)), (n_samples, 1, 1))

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
    positions = rng.normal(size=(n_samples, 1, n_dim))

    times = _times_from_positions(positions)

    # Expect each per-param ``τ`` to be O(1) — well under N. We use a loose
    # bound (10) so the test isn't flaky on small-sample variance.
    assert np.all(times < 10.0)


@requires_blackjax
def test__times_from_positions_multi_chain_pools_samples():
    # `N` in the tau = N / ESS identity should be the *total* sample count
    # across all chains, not just one chain's.
    rng = np.random.default_rng(2)
    n_samples = 500
    n_chains = 4
    n_dim = 2
    positions = rng.normal(size=(n_samples, n_chains, n_dim))

    times = _times_from_positions(positions)

    assert times.shape == (n_dim,)
    assert np.all(np.isfinite(times))


def test__build_search_internal_shape_contract():
    # Pure numpy: exercises the persistence-dict glue directly, without
    # touching blackjax/jax.
    n_samples, n_chains, n_dim = 5, 3, 2

    rng = np.random.default_rng(0)
    positions_chunks = [rng.normal(size=(n_samples, n_chains, n_dim))]
    log_likelihood_chunks = [rng.normal(size=(n_samples, n_chains))]
    info_chunks = {
        "acceptance_rate": [np.full((n_samples, n_chains), 0.85)],
        "num_integration_steps": [np.full((n_samples, n_chains), 7, dtype=int)],
        "is_divergent": [np.zeros((n_samples, n_chains), dtype=bool)],
        "num_trajectory_expansions": [np.full((n_samples, n_chains), 3, dtype=int)],
    }

    search_internal = _build_search_internal(
        positions_chunks=positions_chunks,
        log_likelihood_chunks=log_likelihood_chunks,
        info_chunks=info_chunks,
        tuned_params={"step_size": np.ones(n_chains), "inverse_mass_matrix": np.ones((n_chains, n_dim))},
        last_state_position=positions_chunks[0][-1],
        num_warmup=50,
        num_samples_completed=n_samples,
        num_samples_total=n_samples,
        num_chains=n_chains,
        warm_start_source="InitializerParamStartPoints",
        inverse_mass_matrix_kind="dense",
    )

    assert search_internal["positions"].shape == (n_samples, n_chains, n_dim)
    assert search_internal["log_likelihood_history"].shape == (n_samples, n_chains)
    assert search_internal["infos"]["is_divergent"].shape == (n_samples, n_chains)
    assert search_internal["infos"]["num_trajectory_expansions"].shape == (
        n_samples,
        n_chains,
    )
    assert search_internal["num_chains"] == n_chains
    assert search_internal["warm_start_source"] == "InitializerParamStartPoints"
    assert search_internal["inverse_mass_matrix_kind"] == "dense"


@requires_blackjax
def test__samples_via_internal_from_shape_contract():
    n_samples, n_chains = 6, 3

    model = af.Model(af.m.MockClassx2)
    model.one = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    model.two = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)
    n_dim = model.prior_count

    rng = np.random.default_rng(3)
    positions = rng.uniform(-1.0, 1.0, size=(n_samples, n_chains, n_dim))
    log_likelihood_history = rng.normal(size=(n_samples, n_chains))

    search_internal = {
        "positions": positions,
        "log_likelihood_history": log_likelihood_history,
        "infos": {
            "acceptance_rate": np.full((n_samples, n_chains), 0.8),
            "num_integration_steps": np.full((n_samples, n_chains), 5, dtype=int),
            "is_divergent": np.zeros((n_samples, n_chains), dtype=bool),
            "num_trajectory_expansions": np.full((n_samples, n_chains), 2, dtype=int),
        },
        "tuned_params": {
            "step_size": np.ones(n_chains),
            "inverse_mass_matrix": np.ones((n_chains, n_dim)),
        },
        "last_state_position": positions[-1],
        "num_warmup": 10,
        "num_samples_completed": n_samples,
        "num_samples": n_samples,
        "num_chains": n_chains,
        "warm_start_source": "InitializerBall",
        "inverse_mass_matrix_kind": "none",
    }

    search = af.BlackJAXNUTS(num_chains=n_chains)
    samples = search.samples_via_internal_from(model=model, search_internal=search_internal)

    assert len(samples.sample_list) == n_samples * n_chains
    assert samples.total_walkers == n_chains
    assert samples.samples_info["num_chains"] == n_chains
    assert samples.samples_info["warm_start_source"] == "InitializerBall"
    assert samples.samples_info["inverse_mass_matrix_kind"] == "none"
    assert len(samples.samples_info["ess_bulk_per_param"]) == n_dim
    assert len(samples.samples_info["ess_tail_per_param"]) == n_dim
    assert len(samples.samples_info["rhat_per_param"]) == n_dim
    assert samples.samples_info["n_divergent"] == 0
    assert samples.samples_info["divergent_indices"] == []
    assert samples.samples_info["tree_depth_histogram"] == {2: n_samples * n_chains}

    # `SamplesMCMC` read-only convenience properties (autofit/non_linear/samples/mcmc.py).
    assert samples.ess_bulk == samples.samples_info["ess_bulk_per_param"]
    assert samples.ess_tail == samples.samples_info["ess_tail_per_param"]
    assert samples.rhat == samples.samples_info["rhat_per_param"]
    assert samples.n_divergent == 0
    assert samples.tree_depths == {2: n_samples * n_chains}

    # chain-major flatten: the first n_samples rows are chain 0.
    chain_0_first_row = samples.sample_list[0].parameter_lists_for_paths(
        samples.paths
    )
    assert chain_0_first_row == pytest.approx(positions[0, 0].tolist())
