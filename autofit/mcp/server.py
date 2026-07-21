"""
The read-only results-inspector MCP stdio server core.

``core_server()`` returns a FastMCP with the seven read-only tools registered
over ``autofit.mcp.tools``. Callers — the assistants, or ``python -m autofit.mcp``
— register any extra tools on the returned object and call ``.run()``.

Importing this module imports autofit. A launcher must therefore, *before*
importing it: pin config with ``autofit.mcp.bootstrap.pin_config`` (so config
does not depend on the working directory), set ``JAX_PLATFORMS=cpu`` (skips the
jax backend probe, which logs to stdout), and guard the import with
``contextlib.redirect_stdout(sys.stderr)`` — stdout carries the JSON-RPC channel.
``__main__`` does exactly this for the standalone case.
"""

import io

from mcp.server.fastmcp import FastMCP, Image

from autofit.mcp import tools
from autofit.mcp.bootstrap import route_logging_to_stderr


def _png(image) -> Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Image(data=buffer.getvalue(), format="png")


def core_server(name: str = "pyauto-results-inspector") -> FastMCP:
    """
    Build a FastMCP server with the seven read-only results-inspector tools
    registered. Downstream servers add their own tools on the returned object.
    """
    route_logging_to_stderr()

    mcp = FastMCP(name)

    @mcp.tool()
    def list_searches(
        directory: str,
        sort_by: str = "log_evidence",
        limit: int = 20,
        completed_only: bool = False,
    ) -> list:
        """
        List every model-fit found under `directory` (searched recursively): one
        row per fit with its name, unique tag, output directory, completion state,
        log evidence, maximum log likelihood and free-parameter count.

        Rows are sorted by `sort_by` (descending; fits without that value last) —
        use "log_evidence" for nested samplers or "max_log_likelihood" generally —
        and truncated to `limit` (pass 0 for all). The returned `directory` of a
        row is what the other tools take as their `directory` argument.
        """
        return tools.list_searches(
            directory, sort_by=sort_by, limit=limit, completed_only=completed_only
        )

    @mcp.tool()
    def get_model(directory: str) -> dict:
        """
        The model that was fitted in one search-output directory: a human-readable
        `info` block (component classes, priors) and the full model as a dict.
        """
        return tools.get_model(directory)

    @mcp.tool()
    def get_result_summary(directory: str) -> str:
        """
        The `model.results` text for one search-output directory: the fit's own
        summary of the maximum-likelihood model and (when the search produces
        them) parameter estimates with errors.
        """
        return tools.get_result_summary(directory)

    @mcp.tool()
    def get_samples_summary(directory: str) -> dict:
        """
        Posterior summary for one search-output directory: log evidence (None for
        MLE/MCMC searches without one), maximum log likelihood, the model's
        parameter paths, and the maximum-likelihood and median-PDF parameter
        vectors (`median_pdf_parameters` is None for MLE searches, which have no
        PDF).
        """
        return tools.get_samples_summary(directory)

    @mcp.tool()
    def get_search_info(directory: str) -> dict:
        """
        The non-linear search used in one search-output directory: name, unique
        tag, completion state, and the search's serialized settings.
        """
        return tools.get_search_info(directory)

    @mcp.tool()
    def list_images(directory: str) -> list:
        """
        Names of the visualization images (`image/*.png`) available in one
        search-output directory — pass a name (without `.png`) to `fetch_image`.
        """
        return tools.list_images(directory)

    @mcp.tool()
    def fetch_image(directory: str, name: str = "subplot_fit") -> Image:
        """
        One visualization image from a search-output directory (e.g.
        "subplot_fit"), returned inline so it renders directly in chat. Use
        `list_images` to see what is available.
        """
        return _png(tools.fetch_image(directory, name=name))

    return mcp
