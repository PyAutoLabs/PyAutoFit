"""Native schema, trajectory, atomicity and normal-return witnesses."""
import copy

import numpy as np
import pytest

import autofit as af

nautilus = pytest.importorskip("nautilus")
from autofit.non_linear.search.nest.nautilus import checkpoint_policy as policy
from autofit.non_linear.search.nest.nautilus.search import _large_attribute_hdf5


def prior(x):
    return x


def likelihood(x):
    return -np.sum((x - .5)**2) / .02


def blob_likelihood(x):
    return likelihood(x), x[0], x[1]


def sampler(path, *, batch=True, blobs=False):
    cls = policy.BatchedCheckpointSampler if batch else nautilus.Sampler
    return cls(prior, blob_likelihood if blobs else likelihood, n_dim=2,
               n_live=30, n_points_min=10, n_batch=20, n_networks=4,
               neural_network_kwargs={"hidden_layer_sizes": (8,), "max_iter": 50},
               seed=3, filepath=path)


def assert_native_equal(a, b):
    assert a.n_like == b.n_like
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert a.explored == b.explored
    assert a.discard_exploration == b.discard_exploration
    for key in ("shell_n", "shell_end_exp", "shell_n_sample_exp", "points_t", "log_l_t", "shell_t", "blobs_t"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    for key in ("points", "log_l", "blobs"):
        left, right = getattr(a, key), getattr(b, key)
        if left is None:
            assert right is None
        else:
            for x, y in zip(left, right):
                np.testing.assert_array_equal(x, y)
    for x, y in zip(a.posterior(), b.posterior()):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5", None])
def test_invalid_intervals(bad):
    with pytest.raises(ValueError, match="positive integer"):
        af.Nautilus(checkpoint_interval_batches=bad)
    with pytest.raises(ValueError, match="positive integer"):
        policy.validate_interval(bad)


def test_default_native_path_and_disabled_checkpoints():
    assert af.Nautilus().sampler_cls is nautilus.Sampler
    assert af.Nautilus(checkpoint=False).sampler_cls is nautilus.Sampler
    assert af.Nautilus(checkpoint=False).checkpoint_file is None
    with pytest.raises(ValueError, match="requires checkpoint=True"):
        af.Nautilus(checkpoint=False, checkpoint_interval_batches=5)
    assert "checkpoint_interval_batches" not in af.Nautilus.__identifier_fields__
    assert af.Nautilus(checkpoint_interval_batches=5).sampler_cls.func is policy.BatchedCheckpointSampler


def test_version_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "version", lambda name: "1.0.6")
    with pytest.raises(ValueError, match="1.0.5"):
        sampler(tmp_path / "checkpoint.hdf5")


@pytest.mark.parametrize("blobs", [False, True])
def test_exact_trajectory_all_shells_caches_and_native_reopen(tmp_path, blobs):
    with _large_attribute_hdf5():
        original = sampler(tmp_path / "original.hdf5", batch=False, blobs=blobs)
        candidate = sampler(tmp_path / "candidate.hdf5", blobs=blobs)
        boundaries = []
        for obj in (original, candidate):
            events = []
            write = obj.write
            def observed(*args, **kwargs):
                events.append((obj.n_like, obj.explored, len(obj.bounds)))
                return write(*args, **kwargs)
            obj.write = observed
            assert obj.run(n_eff=50, n_shell=2, f_live=.2, n_like_max=3000,
                           timeout=15, discard_exploration=True, verbose=False)
            obj.write = write
            boundaries.append(set(events))
        assert boundaries[0] <= boundaries[1]
        assert len(candidate.bounds) > 1
        assert_native_equal(original, candidate)
        reopened = sampler(tmp_path / "candidate.hdf5", batch=False, blobs=blobs)
        assert_native_equal(candidate, reopened)
        assert candidate._pending_checkpoint_batches == 0
        state = copy.deepcopy(candidate.rng.bit_generator.state)
        candidate.write(tmp_path / "extra.hdf5")
        assert candidate.rng.bit_generator.state == state
        assert_native_equal(candidate, sampler(tmp_path / "extra.hdf5", batch=False, blobs=blobs))


@pytest.mark.parametrize("limit", [60, 140])
def test_normal_budget_return_flushes_pending_batches(tmp_path, limit):
    with _large_attribute_hdf5():
        obj = sampler(tmp_path / "checkpoint.hdf5")
        assert not obj.run(n_like_max=limit, verbose=False)
        restored = sampler(tmp_path / "checkpoint.hdf5", batch=False)
        assert_native_equal(obj, restored)
        assert obj._pending_checkpoint_batches == 0


def test_timeout_normal_return_flushes_without_new_sampling(tmp_path, monkeypatch):
    obj = sampler(tmp_path / "checkpoint.hdf5")
    obj.run(n_like_max=40, verbose=False)
    obj.add_samples(-1)
    obj.write_shell_update(obj.filepath, -1)
    assert obj._pending_checkpoint_batches == 1
    obj.run(timeout=0, verbose=False)
    assert_native_equal(obj, sampler(obj.filepath, batch=False))


@pytest.mark.parametrize("failure", ["native_write", "replace"])
def test_failed_publication_keeps_previous_file_and_cleans_temp(tmp_path, monkeypatch, failure):
    path = tmp_path / "checkpoint.hdf5"
    obj = sampler(path)
    obj.run(n_like_max=40, verbose=False)
    before, rng = path.read_bytes(), copy.deepcopy(obj.rng.bit_generator.state)
    obj._pending_checkpoint_batches = 2
    def fail(*args, **kwargs):
        raise OSError("injected failure")
    if failure == "native_write":
        def partial(self, filepath, **kwargs):
            filepath.write_bytes(b"incomplete")
            raise OSError("injected failure")
        monkeypatch.setattr(nautilus.Sampler, "write", partial)
    else:
        monkeypatch.setattr(policy.os, "replace", fail)
    with pytest.raises(OSError, match="injected failure"):
        obj.flush_checkpoint()
    assert path.read_bytes() == before
    assert obj.rng.bit_generator.state == rng
    assert obj._pending_checkpoint_batches == 2
    assert not list(tmp_path.glob(".*.hdf5"))
    assert sampler(path, batch=False).n_like == 40


def test_exception_does_not_flush_half_mutated_state(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.hdf5"
    obj = sampler(path)
    obj.run(n_like_max=40, verbose=False)
    before = path.read_bytes()
    def broken_run(self, **kwargs):
        self._pending_checkpoint_batches = 2
        self.n_like += 99
        raise RuntimeError("broken batch")
    monkeypatch.setattr(nautilus.Sampler, "run", broken_run)
    with pytest.raises(RuntimeError, match="broken batch"):
        obj.run()
    assert path.read_bytes() == before
    assert sampler(path, batch=False).n_like == 40


def test_interval_publishes_all_dirty_shells_together(tmp_path):
    path = tmp_path / "checkpoint.hdf5"
    with _large_attribute_hdf5():
        obj = sampler(path, blobs=True)
        assert obj.run(n_eff=50, n_shell=2, f_live=.2, n_like_max=3000,
                       timeout=15, discard_exploration=True, verbose=False)
        assert len(obj.bounds) > 1
        before = path.read_bytes()
        for index in range(5):
            shell = index % len(obj.bounds)
            obj.add_samples(shell)
            obj.write_shell_update(path, shell)
            if index < 4:
                assert path.read_bytes() == before
        assert obj._pending_checkpoint_batches == 0
        assert_native_equal(obj, sampler(path, batch=False, blobs=True))


def test_export_does_not_clear_unpublished_primary_updates(tmp_path):
    path = tmp_path / "checkpoint.hdf5"
    obj = sampler(path)
    obj.run(n_like_max=40, verbose=False)
    obj.add_samples(-1)
    obj.write_shell_update(path, -1)
    obj.write(tmp_path / "export.hdf5")
    assert obj._pending_checkpoint_batches == 1
    assert sampler(path, batch=False).n_like == 40
    obj.flush_checkpoint()
    assert sampler(path, batch=False).n_like == obj.n_like
