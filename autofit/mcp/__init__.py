"""
The read-only results-inspector MCP core, distributed as the optional
``autofit[mcp]`` extra.

It exposes PyAutoFit output directories to chat harnesses that cannot execute
code (Claude Desktop, ChatGPT) as a small set of read-only tools — list fits
ranked by evidence, read model/posterior/result summaries, view result images
inline. It is deliberately *glue, not code*: every tool is one existing public
aggregator call plus serialization; composing models and running searches stay
Python-first.

Layout:

- ``tools`` — the plain aggregator-wrapper functions (no ``mcp`` dependency; the
  test suite exercises them directly).
- ``server.core_server`` — registers those functions on a FastMCP stdio server.
- ``bootstrap`` — the launch-time config-pin + stdout protections, kept free of
  any autofit import so a launcher can call them *before* importing the
  autofit-backed modules.

Run standalone with ``python -m autofit.mcp`` (needs the extra:
``pip install autofit[mcp]``). Downstream servers — e.g. the PyAutoLens
results-inspector, which adds a lens image/FITS layer — call ``core_server()``
and register their own tools on top.
"""


def __getattr__(name):
    # Lazy re-exports (PEP 562): ``autofit.mcp.core_server`` / ``autofit.mcp.png``
    # resolve without importing ``autofit.mcp.server`` (and therefore autofit and
    # the ``mcp`` package) at ``import autofit.mcp`` time — keeping this package
    # import-light so a launcher can pin config before the heavy import.
    if name in ("core_server", "png", "_png"):
        from autofit.mcp import server

        return server.core_server if name == "core_server" else server._png
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

