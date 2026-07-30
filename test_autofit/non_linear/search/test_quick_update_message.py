import pytest

import autofit as af

# ``quick_update_message`` is logged once at the start of every search. It is the
# only thing that explains the terminal's behaviour during a long fit: either
# on-the-fly maximum-likelihood updates appear every N iterations, or nothing
# appears until the search finishes.
#
# It exists because the workspace example scripts used to print the *name* of the
# knob rather than its value ("On-the-fly updates every
# iterations_per_quick_update are printed to the notebook"). Interpolating the
# value alone would not have fixed it: the packaged default is the inf-like
# ``1e99`` sentinel, so the honest message for most users is not a number at all.
#
# NumPy-only, like the rest of the library suite.

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test__real_cadence_renders_as_a_plain_integer():
    # The knob is stored as a float (so ``search.json`` keeps a readable
    # ``1e99`` rather than a 99-digit integer), so the message has to cast --
    # otherwise a user configuring 250000 is told "every 250000.0 iterations".
    search = af.MultiStartAdam(n_steps=3000, iterations_per_quick_update=250000)

    assert isinstance(search.iterations_per_quick_update, float)

    message = search.quick_update_message

    assert "every 250000 iterations" in message
    assert "250000.0" not in message
    assert "disabled" not in message


def test__default_sentinel_says_disabled_and_names_the_config_key():
    # The packaged default is 1e99 — updates never fire. Reporting that as
    # "every 1e+99 iterations" would be true and useless, so the disabled branch
    # says so plainly and names the key that turns them on, because that is the
    # branch a user is most likely to want to act on.
    search = af.MultiStartAdam(n_steps=300)

    assert search.iterations_per_quick_update == pytest.approx(1e99)

    message = search.quick_update_message

    assert "disabled" in message
    assert "iterations_per_quick_update" in message
    assert "config/general.yaml" in message
    assert "1e+99" not in message


@pytest.mark.parametrize("never", [1e99, 1e100, float("inf")])
def test__any_never_sized_cadence_reads_as_disabled(never):
    # The threshold is compared against, not equality-tested on 1e99, so a
    # hand-set 1e100 in a workspace config reads as "never" too. ``inf`` is
    # covered separately because it is not comparable-then-castable: ``int(inf)``
    # raises, so the finite check has to come first.
    search = af.MultiStartAdam(n_steps=300)
    search.iterations_per_quick_update = never

    assert "disabled" in search.quick_update_message


def test__a_cadence_of_one_still_reads_as_a_number():
    # Guards the boundary from the other side: the disabled branch must not
    # swallow small real cadences.
    search = af.MultiStartAdam(n_steps=300, iterations_per_quick_update=1)

    assert "every 1 iterations" in search.quick_update_message
