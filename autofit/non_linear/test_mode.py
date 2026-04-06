import os


def test_mode_level():
    """
    Return the current test mode level.

    0 = off (normal operation)
    1 = reduce sampler iterations to minimum (existing behavior)
    2 = bypass sampler entirely, call likelihood once
    3 = bypass sampler entirely, skip likelihood call
    """
    return int(os.environ.get("PYAUTOFIT_TEST_MODE", "0"))


def is_test_mode():
    """
    Return True if any test mode is active.
    """
    return test_mode_level() > 0
