import pytest

from autofit.non_linear.plot.plot_util import (
    PlotKwargsError,
    accepted_kwarg_names,
    checked_kwargs,
    log_plot_exception,
)


def target(alpha=None, beta=None, *args, **kwargs):
    pass


def test__accepted_kwarg_names__ignores_the_var_keyword_sink():
    # `corner.corner` funnels what it does not name into `hist2d`, which reads
    # only `extent` — so a `**kwargs` must never be read as "accepts anything".
    assert accepted_kwarg_names(target) == {"alpha", "beta"}


def test__accepted_kwarg_names__unions_across_targets():
    def other(gamma=None):
        pass

    assert accepted_kwarg_names(target, other) == {"alpha", "beta", "gamma"}


def test__checked_kwargs__accepted_names_pass_through():
    assert checked_kwargs(
        {"alpha": 1}, accepts={"alpha", "beta"}, target="target"
    ) == {"alpha": 1}


def test__checked_kwargs__unknown_name_is_rejected_and_named():
    with pytest.raises(PlotKwargsError) as error:
        checked_kwargs({"gamma": 1}, accepts={"alpha", "beta"}, target="target")

    assert "gamma" in str(error.value)
    assert "target" in str(error.value)


def test__checked_kwargs__close_match_is_offered_as_a_hint():
    with pytest.raises(PlotKwargsError) as error:
        checked_kwargs({"alpah": 1}, accepts={"alpha"}, target="target")

    assert "did you mean 'alpha'?" in str(error.value)


def test__checked_kwargs__reserved_name_is_rejected():
    with pytest.raises(PlotKwargsError) as error:
        checked_kwargs(
            {"alpha": 1}, accepts={"alpha"}, reserved=("alpha",), target="target"
        )

    assert "cannot be overridden" in str(error.value)


def test__checked_kwargs__open_pass_through_enforces_reserved_only():
    # `accepts=None` is for a target that raises on what it cannot use, so only
    # the arguments PyAutoFit owns are guarded.
    assert checked_kwargs({"anything": 1}, target="target") == {"anything": 1}

    with pytest.raises(PlotKwargsError):
        checked_kwargs({"data": 1}, reserved=("data",), target="target")


def test__log_plot_exception__swallows_an_unconverged_posterior_error():
    @log_plot_exception
    def plot():
        raise ValueError("not enough samples")

    assert plot() is None


def test__log_plot_exception__never_swallows_a_rejected_kwarg():
    # A `PlotKwargsError` is a `TypeError`, which this decorator catches. Left
    # unguarded it would surface as "posterior estimate not yet sufficient",
    # hiding the caller's mistake behind a misleading info log.
    @log_plot_exception
    def plot():
        raise PlotKwargsError("bad kwarg")

    with pytest.raises(PlotKwargsError):
        plot()
