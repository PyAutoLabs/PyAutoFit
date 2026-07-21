"""
Standalone runner for the ``autofit[mcp]`` extra: ``python -m autofit.mcp``.

Serves the seven read-only core tools over stdio. Downstream servers that add
their own tools (e.g. the PyAutoLens lens layer) provide their own launcher and
call :func:`autofit.mcp.server.core_server` directly.

The ordering here is load-bearing: force JAX onto CPU and pin config to autofit's
own bundled ``config/`` (so the server does not depend on the launch directory)
*before* importing the autofit-backed server, and guard that import so no
import-time stdout reaches the JSON-RPC channel.
"""

import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from autofit.mcp.bootstrap import pin_config

pin_config(Path(__file__).resolve().parents[1] / "config")

with contextlib.redirect_stdout(sys.stderr):
    from autofit.mcp.server import core_server

core_server().run()
