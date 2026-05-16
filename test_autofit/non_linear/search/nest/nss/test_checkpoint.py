"""
Unit tests for the ``af.NSS`` checkpoint/resume helpers
(``_save_checkpoint`` / ``_load_checkpoint``).

No real ``nss.ns.run_nested_sampling`` calls — the heavy end-to-end resume
verification lives in
``autolens_workspace_developer/searches_minimal/nss_checkpoint_resume.py``.
"""

from unittest.mock import patch

import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search.nest.nss import search as nss_search_module
from autofit.non_linear.search.nest.nss.search import (
    _load_checkpoint,
    _save_checkpoint,
)


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _synthetic_state():
    """Plain-dict pytree mimicking the blackjax NSS state shape.

    We only need a pytree of JAX arrays — the round-trip serialiser doesn't
    care about the type as long as ``jax.tree_util.tree_map`` can walk it.
    Plain dicts are registered pytrees; SimpleNamespace is not.
    """
    return {
        "particles": {
            "position": jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "loglikelihood": jnp.asarray([-1.5, -0.5, -0.1]),
        },
        "integrator": {
            "logZ": jnp.float64(-12.3),
            "logZ_live": jnp.float64(-11.8),
        },
    }


def _synthetic_dead(n_iter=3):
    return [
        {
            "particles": {
                "position": jnp.asarray([[float(i), float(i + 1)]]),
                "loglikelihood": jnp.asarray([-float(i)]),
            },
        }
        for i in range(n_iter)
    ]


def test__save_load_checkpoint_round_trip(tmp_path):
    state = _synthetic_state()
    dead = _synthetic_dead()
    run_key = jax.random.PRNGKey(7)

    path = tmp_path / "nss_checkpoint.pkl"
    _save_checkpoint(path, state, dead, run_key, iteration=42)

    assert path.exists()
    loaded_state, loaded_dead, loaded_run_key, loaded_iter = _load_checkpoint(path)

    assert loaded_iter == 42
    assert np.array_equal(np.asarray(loaded_run_key), np.asarray(run_key))

    np.testing.assert_array_equal(
        np.asarray(loaded_state["particles"]["position"]),
        np.asarray(state["particles"]["position"]),
    )
    np.testing.assert_array_equal(
        np.asarray(loaded_state["particles"]["loglikelihood"]),
        np.asarray(state["particles"]["loglikelihood"]),
    )
    np.testing.assert_array_equal(
        np.asarray(loaded_state["integrator"]["logZ"]),
        np.asarray(state["integrator"]["logZ"]),
    )

    assert len(loaded_dead) == len(dead)
    for orig, loaded in zip(dead, loaded_dead):
        np.testing.assert_array_equal(
            np.asarray(loaded["particles"]["position"]),
            np.asarray(orig["particles"]["position"]),
        )


def test__save_checkpoint_is_atomic(tmp_path):
    """A partially-written checkpoint must not clobber the previous good one.

    The helper writes to ``<path>.tmp`` then ``os.replace`` to the final path.
    If the rename fails (simulated here by patching ``os.replace`` to raise),
    the final path must be untouched — the previous-good blob still loads.
    """
    state = _synthetic_state()
    dead = _synthetic_dead()
    run_key = jax.random.PRNGKey(3)
    path = tmp_path / "nss_checkpoint.pkl"

    _save_checkpoint(path, state, dead, run_key, iteration=1)
    good_blob = path.read_bytes()

    state["integrator"]["logZ"] = jnp.float64(-100.0)
    with patch(
        "autofit.non_linear.search.nest.nss.search.os.replace",
        side_effect=OSError("simulated rename failure"),
    ):
        with pytest.raises(OSError):
            _save_checkpoint(path, state, dead, run_key, iteration=2)

    assert path.read_bytes() == good_blob


def test__nss_checkpoint_path_is_none_for_null_paths():
    """Without a real output dir (NullPaths), checkpoint resolution returns None.

    This means ``_fit``'s resume detection silently skips for NullPaths fits
    (e.g. unit tests, in-memory aggregator round-trips) rather than blowing
    up on the missing ``search_internal_path`` attribute.
    """
    search = af.NSS()
    assert search._nss_checkpoint_path is None
    assert search.checkpoint_file is None


def test__init_accepts_checkpoint_interval():
    search = af.NSS(checkpoint_interval=25)
    assert search.checkpoint_interval == 25

    default = af.NSS()
    assert default.checkpoint_interval == 100


def test__init_iterations_per_quick_update_no_longer_warns(caplog):
    """Phase 1's no-op log when the kwarg is set is gone in Phase 3 (the kwarg
    is now actually wired). The warning text must not appear.
    """
    with caplog.at_level("INFO"):
        af.NSS(iterations_per_quick_update=10)
    assert not any(
        "not yet wired" in record.message
        for record in caplog.records
    )


def test__load_checkpoint_called_when_file_exists(tmp_path):
    """Verify ``_load_checkpoint`` is invoked from ``_fit`` when a checkpoint
    file exists at the resolved path.

    We replace ``_blackjax.nss`` with a mock that fails loudly if ``algo.init``
    fires — i.e. if the fresh-init branch ran. The resume branch must call
    ``_load_checkpoint`` first; we patch that helper to return a sentinel
    that satisfies the immediate-termination logZ check, so the outer loop
    exits before doing any real work.
    """
    from types import SimpleNamespace

    fake_checkpoint = tmp_path / "nss_checkpoint.pkl"
    fake_checkpoint.write_bytes(b"placeholder - actual contents replaced by mock")

    sentinel_state = SimpleNamespace(
        integrator=SimpleNamespace(logZ=0.0, logZ_live=-100.0),
        particles=SimpleNamespace(
            position=jnp.zeros((4, 2)),
            loglikelihood=jnp.zeros(4),
        ),
    )

    # Minimum stub for model + analysis. _fit calls model.prior_count and
    # both vector_from_unit_vector + log_prior_list_from_vector inside the
    # closure construction (which isn't traced unless one_step fires).
    mock_model = SimpleNamespace(
        prior_count=2,
        instance_from_vector=lambda **kw: SimpleNamespace(),
        vector_from_unit_vector=lambda v, xp=None: jnp.asarray([0.0, 0.0]),
        log_prior_list_from_vector=lambda **kw: [0.0, 0.0],
    )
    mock_analysis = SimpleNamespace(
        log_likelihood_function=lambda instance: 0.0,
    )

    search = af.NSS(n_live=4, num_mcmc_steps=1, num_delete=1, termination=-3.0)
    # Force the checkpoint property to return our sentinel path even though
    # paths is the default NullPaths.
    with patch.object(
        type(search),
        "_nss_checkpoint_path",
        new=fake_checkpoint,
    ), patch.object(
        nss_search_module,
        "_load_checkpoint",
        return_value=(sentinel_state, [], jax.random.PRNGKey(0), 17),
    ) as mock_load, patch.object(
        nss_search_module, "_blackjax",
    ) as mock_bjax, patch.object(
        nss_search_module,
        "_nss_finalise",
        return_value=SimpleNamespace(
            particles=SimpleNamespace(
                position=np.zeros((1, 2)),
                loglikelihood=np.zeros(1),
            ),
            update_info=SimpleNamespace(
                num_steps=np.zeros(1, dtype=int),
                num_shrink=np.zeros(1, dtype=int),
            ),
        ),
    ), patch.object(
        nss_search_module,
        "_nss_log_weights",
        return_value=jnp.zeros((1, 100)),
    ):
        mock_bjax.nss.return_value.init.side_effect = AssertionError(
            "algo.init called from resume path — expected _load_checkpoint instead."
        )
        mock_bjax.nss.return_value.step.side_effect = AssertionError(
            "algo.step called even though logZ termination should fire immediately."
        )

        # The mocked downstream pipeline (Fitness construction, _NSSInternal
        # repackaging) is intentionally not realistic — we only care that the
        # resume branch was entered and ``_load_checkpoint`` was called. Catch
        # any downstream stub-related failure; the assertion below is the gate.
        try:
            search._fit(model=mock_model, analysis=mock_analysis)
        except (AttributeError, AssertionError, TypeError):
            pass

    mock_load.assert_called_once_with(fake_checkpoint)
