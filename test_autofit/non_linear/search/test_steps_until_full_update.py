import inspect

import pytest

import autofit as af

# ``_steps_until_full_update`` is the one place the float
# ``iterations_per_full_update`` becomes the ``int`` chunk size that chunked
# searches feed to ``range`` (or to a JAX scan / key split). It lives on
# ``AbstractSearch`` because every chunked search needs it and forgetting it is
# a real, shipped bug class: it crashed MultiStart for anyone setting a real
# cadence (PyAutoFit#1420) and stayed latent in Emcee and BlackJAX NUTS
# (PyAutoFit#1422).
#
# NumPy-only, like the rest of the library suite — the searches' ``_fit`` bodies
# need jax/optax/emcee and are exercised in the workspace test repos.

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__real_cadence_is_an_int_range_can_consume():
    """A cadence *below* the remaining budget is the case that reaches
    ``range``. With the ``1e99`` default ``min`` returns the int
    ``iterations_remaining`` instead, which is why this stayed latent for every
    user of the default until someone set a real checkpoint cadence."""
    search = af.MultiStartAdam(n_steps=3000, iterations_per_full_update=50)

    # The knob really is stored as a float — this is what made the crash latent.
    assert isinstance(search.iterations_per_full_update, float)

    iterations = search._steps_until_full_update(iterations_remaining=3000)

    assert iterations == 50
    assert isinstance(iterations, int)
    # The operation that used to raise TypeError.
    assert len(list(range(iterations))) == 50


def test__clamps_to_the_remaining_budget():
    # Near the end of the budget the chunk shrinks to what is left, so a search
    # never overshoots its ceiling.
    search = af.MultiStartAdam(n_steps=3000, iterations_per_full_update=50)

    assert search._steps_until_full_update(iterations_remaining=20) == 20
    assert isinstance(search._steps_until_full_update(iterations_remaining=20), int)


def test__default_cadence_is_a_single_chunk():
    # The packaged default is the inf-like 1e99: one chunk covering the whole
    # budget, i.e. checkpoint only at the end.
    search = af.MultiStartAdam(n_steps=300)

    assert search.iterations_per_full_update == pytest.approx(1e99)

    iterations = search._steps_until_full_update(iterations_remaining=300)

    assert iterations == 300
    assert isinstance(iterations, int)


def test__zero_cadence_raises_rather_than_meaning_never_checkpoint():
    """``1e99`` is already the "no intermediate checkpoint" sentinel and flows
    through the ``min`` without a special case, so a stored ``0`` is a
    misconfiguration, not a second sentinel.

    ``__init__`` cannot produce it from a public argument (``x or config``
    replaces a falsy one), but the HPC branch assigns the config value with no
    such fallback — so an HPC cadence of ``0`` would otherwise silently disable
    checkpointing for the whole run instead of failing.
    """
    search = af.MultiStartAdam(n_steps=120)
    search.iterations_per_full_update = 0.0

    with pytest.raises(ValueError, match="iterations_per_full_update"):
        search._steps_until_full_update(iterations_remaining=120)


@pytest.mark.parametrize("cadence", [0.5, 50.9, -5])
def test__unusable_cadence_raises_rather_than_being_clamped(cadence):
    """A cadence below 1 (or a fractional one) truncates to a zero-length
    chunk: nothing runs, the counter never advances, and the enclosing ``while``
    spins forever re-running ``perform_update``.

    Clamping to 1 would hide the mistake and silently run a schedule the user
    never asked for — a silent hang on a cluster is far more expensive than an
    error at the first chunk boundary.
    """
    search = af.MultiStartAdam(n_steps=300, iterations_per_full_update=cadence)

    with pytest.raises(ValueError, match="iterations_per_full_update"):
        search._steps_until_full_update(iterations_remaining=300)


@pytest.mark.parametrize("remaining", [2.5, 0, -10])
def test__unusable_remaining_budget_raises(remaining):
    """The chunk can only be a usable positive whole number if the remaining
    budget is one. Searches keep that budget under different names (``n_steps``,
    ``nsteps``, ``num_samples``, ``maxiter``) and none of them validates it, so
    it is validated here at the point of use — a fractional budget leaves a
    fractional remainder that truncates to a zero-length chunk."""
    search = af.MultiStartAdam(n_steps=300, iterations_per_full_update=50)

    with pytest.raises(ValueError, match="iterations_remaining"):
        search._steps_until_full_update(iterations_remaining=remaining)


@pytest.mark.parametrize(
    "search_cls",
    [af.MultiStartAdam, af.Emcee, af.Zeus, af.BFGS, af.BlackJAXNUTS],
)
def test__chunked_searches_take_their_chunk_size_from_the_helper(search_cls):
    """Wiring guard. Every test above drives ``_steps_until_full_update``
    directly, so all of them would still pass if a search went back to computing
    its chunk inline — which is precisely the regression this helper exists to
    prevent, and precisely what shipped in PyAutoFit#1420.

    These ``_fit`` bodies cannot be executed from this suite (jax/optax/emcee
    plus a real ``Analysis``), so the call site is asserted at the source level.
    """
    source = inspect.getsource(search_cls._fit)

    assert "self._steps_until_full_update(" in source
    # the un-cast inline forms the crashes came from must not come back
    assert "self.iterations_per_full_update or self.n_steps" not in source
    assert "min(self.iterations_per_full_update" not in source
    assert "iterations = self.iterations_per_full_update" not in source
