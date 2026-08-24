import numpy as np
import pytest

# `autofit.non_linear.search.mcmc.blackjax.chains` is pure numpy -- no JAX or
# blackjax import at module scope (following the repo convention of no JAX in
# unit tests) -- so these tests run unconditionally, unlike
# `test_blackjax_nuts.py`'s `@requires_blackjax`-guarded diagnostics tests.
from autofit.non_linear.search.mcmc.blackjax.chains import (
    inverse_mass_matrix_kind_from,
    resolve_inverse_mass_matrix,
    split_chain_diagnostics,
    stack_initial_positions,
)


class _FakeSamples:
    def __init__(self, covariance_matrix, n_samples):
        self.covariance_matrix = covariance_matrix
        self._n_samples = n_samples

    def __len__(self):
        return self._n_samples


class _FakeResult:
    def __init__(self, samples):
        self.samples = samples


class TestInverseMassMatrixKindFrom:
    def test__none(self):
        assert inverse_mass_matrix_kind_from(None) == "none"

    def test__diagonal_and_dense_strings(self):
        assert inverse_mass_matrix_kind_from("diagonal") == "diagonal"
        assert inverse_mass_matrix_kind_from("dense") == "dense"

    def test__bad_string_raises(self):
        with pytest.raises(ValueError):
            inverse_mass_matrix_kind_from("not_a_kind")

    def test__array_1d_and_2d(self):
        assert inverse_mass_matrix_kind_from(np.ones(3)) == "array"
        assert inverse_mass_matrix_kind_from(np.eye(3)) == "array"

    def test__result_like(self):
        samples = _FakeSamples(covariance_matrix=np.eye(3), n_samples=20)
        assert inverse_mass_matrix_kind_from(samples) == "result"
        assert inverse_mass_matrix_kind_from(_FakeResult(samples)) == "result"

    def test__unsupported_type_raises(self):
        with pytest.raises(ValueError):
            inverse_mass_matrix_kind_from(object())


class TestResolveInverseMassMatrix:
    def test__none_gives_diagonal_no_seed(self):
        is_diagonal, seed = resolve_inverse_mass_matrix(None, n_dim=3)
        assert is_diagonal is True
        assert seed is None

    def test__diagonal_string(self):
        is_diagonal, seed = resolve_inverse_mass_matrix("diagonal", n_dim=3)
        assert is_diagonal is True
        assert seed is None

    def test__dense_string(self):
        is_diagonal, seed = resolve_inverse_mass_matrix("dense", n_dim=3)
        assert is_diagonal is False
        assert seed is None

    def test__bad_string_raises(self):
        with pytest.raises(ValueError):
            resolve_inverse_mass_matrix("not_a_kind", n_dim=3)

    def test__1d_array_seeds_diagonal(self):
        diag = np.array([1.0, 2.0, 3.0])
        is_diagonal, seed = resolve_inverse_mass_matrix(diag, n_dim=3)
        assert is_diagonal is True
        assert seed is diag or np.array_equal(seed, diag)

    def test__1d_array_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            resolve_inverse_mass_matrix(np.ones(2), n_dim=3)

    def test__2d_array_seeds_dense(self):
        dense = np.eye(3) * 2.0
        is_diagonal, seed = resolve_inverse_mass_matrix(dense, n_dim=3)
        assert is_diagonal is False
        assert np.array_equal(seed, dense)

    def test__2d_array_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            resolve_inverse_mass_matrix(np.eye(2), n_dim=3)

    def test__3d_array_raises(self):
        with pytest.raises(ValueError):
            resolve_inverse_mass_matrix(np.ones((2, 2, 2)), n_dim=3)

    def test__result_covariance_used_as_dense_seed(self):
        cov = np.array([[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 1.5]])
        samples = _FakeSamples(covariance_matrix=cov, n_samples=20)

        is_diagonal, seed = resolve_inverse_mass_matrix(_FakeResult(samples), n_dim=3)

        assert is_diagonal is False
        assert np.array_equal(seed, cov)

        # Also works passed directly as a `Samples`-like object (not wrapped
        # in a `Result`-like object).
        is_diagonal, seed = resolve_inverse_mass_matrix(samples, n_dim=3)
        assert is_diagonal is False
        assert np.array_equal(seed, cov)

    def test__mle_only_too_few_samples_raises(self):
        # < 2 * n_dim samples -- an MLE-style optimizer's "samples", not a
        # posterior-exploring sampler's.
        samples = _FakeSamples(covariance_matrix=np.eye(3), n_samples=5)

        with pytest.raises(ValueError, match="MLE-style"):
            resolve_inverse_mass_matrix(_FakeResult(samples), n_dim=3)

    def test__degenerate_identity_covariance_raises(self):
        # `Samples.covariance_matrix` itself falls back to `np.eye(1)` (or
        # similar) when it can't compute a real covariance -- treat an
        # identity covariance from enough samples as still-unusable.
        samples = _FakeSamples(covariance_matrix=np.eye(3), n_samples=20)

        with pytest.raises(ValueError, match="identity"):
            resolve_inverse_mass_matrix(_FakeResult(samples), n_dim=3)

    def test__non_finite_covariance_raises(self):
        cov = np.array([[1.0, np.nan, 0.0], [np.nan, 1.0, 0.0], [0.0, 0.0, 1.0]])
        samples = _FakeSamples(covariance_matrix=cov, n_samples=20)

        with pytest.raises(ValueError, match="non-finite"):
            resolve_inverse_mass_matrix(_FakeResult(samples), n_dim=3)

    def test__result_with_no_samples_raises(self):
        with pytest.raises(ValueError):
            resolve_inverse_mass_matrix(_FakeResult(samples=None), n_dim=3)


class TestStackInitialPositions:
    def test__list_of_lists_stacked(self):
        stacked = stack_initial_positions([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        assert stacked.shape == (3, 2)
        assert np.array_equal(stacked, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


class TestSplitChainDiagnostics:
    def test__moves_sample_axis_behind_chain_axis(self):
        n_samples, n_chains, n_dim = 200, 2, 3
        rng = np.random.default_rng(0)
        positions = rng.normal(size=(n_samples, n_chains, n_dim))

        chain_major = split_chain_diagnostics(positions)

        assert chain_major.shape == (n_chains, n_samples, n_dim)
        assert np.array_equal(chain_major[0], positions[:, 0, :])
        assert np.array_equal(chain_major[1], positions[:, 1, :])

    def test__single_chain(self):
        n_samples, n_dim = 50, 4
        positions = np.zeros((n_samples, 1, n_dim))

        chain_major = split_chain_diagnostics(positions)

        assert chain_major.shape == (1, n_samples, n_dim)
