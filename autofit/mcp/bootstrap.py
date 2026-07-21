"""
Launch-time hardening for the MCP stdio server.

This module is deliberately kept free of any ``autofit`` import so a launcher can
call it *before* importing the autofit-backed tool modules — autofit reads its
configuration at import time, and jax's backend probe logs to stdout during that
import, both of which must be handled first. It imports only the standard library
and ``autonerves``.
"""

import logging
import sys
from pathlib import Path
from typing import Union


def pin_config(config_path: Union[str, Path]) -> None:
    """
    Pin ``autonerves`` config to ``config_path`` instead of letting it default to
    ``os.getcwd()/config``.

    autofit reads config at import time; if the server is launched from a foreign
    working directory — e.g. a chat client spawning ``wsl.exe`` in a Windows
    folder — the default would scan unrelated files and crash on import (a
    ``desktop.ini`` trips configparser interpolation). Call this before importing
    ``autofit.mcp.server`` (or anything else that pulls in autofit).
    """
    from autonerves import conf

    config_path = Path(config_path)
    conf.instance = conf.Config(
        str(config_path), output_path=str(config_path.parent / "output")
    )


def route_logging_to_stderr() -> None:
    """
    Rebind every stdout-bound logging handler to stderr.

    stdout carries the JSON-RPC channel, but autofit's logging config (loaded on
    import) attaches stdout stream handlers — one stray log line corrupts the
    protocol. Call this after autofit has been imported.
    """
    loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]
    for logger in loggers:
        for handler in getattr(logger, "handlers", []):
            if (
                isinstance(handler, logging.StreamHandler)
                and getattr(handler, "stream", None) is sys.stdout
            ):
                handler.setStream(sys.stderr)
