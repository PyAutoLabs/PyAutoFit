import os
import logging
from functools import wraps
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def skip_in_test_mode(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return
        return func(*args, **kwargs)

    return wrapper


def log_plot_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
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
