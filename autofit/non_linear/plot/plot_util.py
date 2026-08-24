import difflib
import inspect
import logging
import os
from functools import wraps
from pathlib import Path

import numpy as np

from autofit.non_linear.test_mode import skip_visualization

logger = logging.getLogger(__name__)


class PlotKwargsError(TypeError):
    """
    Raised when a plot function is handed a ``**kwargs`` entry its underlying
    plotting library cannot honour.

    A ``TypeError`` subclass so it reads like the error Python raises for an
    unexpected keyword argument, while staying distinguishable from the
    ``TypeError``s ``log_plot_exception`` swallows.
    """


def accepted_kwarg_names(*funcs):
    """
    The keyword-argument names ``funcs`` genuinely honour.

    Only *named* parameters count. A function's own ``**kwargs`` is deliberately
    not read as "accepts anything": ``corner.corner`` funnels everything it does
    not name into ``corner.core.hist2d``, whose body reads only ``extent`` and
    silently discards the rest, so treating that sink as permissive would leave
    the very failure this guard exists to catch.
    """
    names = set()
    for func in funcs:
        for name, parameter in inspect.signature(func).parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            names.add(name)
    return names


def checked_kwargs(kwargs, *, accepts=None, reserved=(), target):
    """
    Validate caller ``kwargs`` against what ``target`` accepts, before forwarding.

    The plot functions pass their ``**kwargs`` on to ``corner`` / ``anesthetic``
    / ``matplotlib``. Anything those libraries would drop on the floor is raised
    here instead, so a mis-typed or wrong-library argument fails loudly rather
    than producing a figure that quietly ignores it.

    Parameters
    ----------
    kwargs
        The caller's keyword arguments.
    accepts
        Names ``target`` honours — see ``accepted_kwarg_names``. ``None`` when
        ``target`` has a genuine open pass-through that raises on what it cannot
        use (``anesthetic`` hands unknown names to matplotlib, which rejects
        them), so only ``reserved`` is enforced.
    reserved
        Names this wrapper sets itself and the caller may not override (e.g. the
        sample array ``corner`` is being asked to plot).
    target
        Human-readable name of the receiving function, for the error message.

    Raises
    ------
    PlotKwargsError
        If any name is reserved or is not accepted by ``target``. Unknown names
        get a "did you mean" hint where a close match exists.
    """
    reserved = set(reserved)

    overridden = sorted(name for name in kwargs if name in reserved)
    if overridden:
        raise PlotKwargsError(
            f"{', '.join(overridden)} is set by PyAutoFit and cannot be "
            f"overridden when forwarding to {target}."
            if len(overridden) == 1
            else f"{', '.join(overridden)} are set by PyAutoFit and cannot be "
            f"overridden when forwarding to {target}."
        )

    if accepts is None:
        return dict(kwargs)

    unknown = sorted(name for name in kwargs if name not in accepts)
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, sorted(accepts), n=1)
            hints.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise PlotKwargsError(
            f"{target} does not accept {', '.join(hints)}. These would be "
            f"silently ignored, so they are rejected instead."
        )

    return dict(kwargs)


def skip_in_test_mode(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if skip_visualization():
            return
        return func(*args, **kwargs)

    return wrapper


def log_plot_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PlotKwargsError:
            # A rejected kwarg is the caller's mistake, not an unconverged
            # posterior — never downgrade it to the info log below.
            raise
        except (
            ValueError,
            KeyError,
            AssertionError,
            IndexError,
            TypeError,
            RuntimeError,
            np.linalg.LinAlgError,
        ):
            logger.info(
                f"Unable to produce {func.__name__} visual: posterior estimate "
                f"not yet sufficient. Should succeed in a later update."
            )

    return wrapper


def output_figure(path=None, filename="figure", format="show"):
    import matplotlib.pyplot as plt

    if format == "show":
        plt.show()
    elif format in ("png", "pdf"):
        if path is not None:
            os.makedirs(path, exist_ok=True)
            plt.savefig(Path(path) / f"{filename}.{format}")
    plt.close()
