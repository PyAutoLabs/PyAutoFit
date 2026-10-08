import datetime
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use Path(...).resolve() to make it absolute, like shown here.
#

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import autofit

# -- Project information -----------------------------------------------------

year = datetime.date.today().year
project = "PyAutoFit"
copyright = "2025, James Nightingale, Richard Hayes"
author = "James Nightingale, Richard Hayes"

# The full version, including alpha/beta/rc tags
release = autofit.__version__
master_doc = "index"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_autodoc_typehints",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]
templates_path = ["_templates"]

set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

# -- Options for extlinks ----------------------------------------------------

extlinks = {
    "pypi": ("https://pypi.org/project/%s/", "%s"),
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Options for TODOs -------------------------------------------------------

todo_include_todos = True

# -- Options for copybutton --------------------------------------------------

# The default selector is "div.highlight pre" (code blocks only). The second
# entry adds the natural-language prompt blocks: any blockquote wrapped in a
# `:::{container} nl-prompt` aside gets its own copy button, so a reader can
# lift the prompt straight into an AI coding agent. The button is inserted as a
# sibling of the blockquote, inside the container, and copies the blockquote's
# innerText (the plain prompt, without the Markdown "> " markers).
copybutton_selector = "div.highlight pre, .nl-prompt blockquote"

# -- Options for Markdown files ----------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_class_signature = "separated"

# Optional sampler backends are mocked when they are not installed, so the API pages
# (autosummary of every search, including NSS / SMC / Zeus / Nautilus, whose modules
# import their backend) build in the minimal `pip install autofit[docs]` environment
# that ReadTheDocs uses. In the full-extras environment nothing is mocked. The list
# mirrors the `requires` of autofit/non_linear/search/registry.py.
import importlib.util

_OPTIONAL_BACKENDS = [
    "jax",
    "jaxlib",
    "blackjax",
    "optax",
    "prodigyopt",
    "nautilus",
    "dynesty",
    "emcee",
    "zeus",
    "getdist",
]
autodoc_mock_imports = [
    module for module in _OPTIONAL_BACKENDS if importlib.util.find_spec(module) is None
]
autoclass_content = "init"

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "PyAutoFit"
html_short_title = "PyAutoFit"
html_permalinks_icon = '<span>#</span>'
html_last_updated_fmt = "%b %d, %Y"

html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# pygments_style = "sphinx"
# pygments_dark_style = "monokai"
add_function_parentheses = False

language = "en"

html_static_path = ["_static"]
html_css_files = ["pyauto.css", "pyautofit.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#c2410c",
        "color-brand-content": "#c2410c",
    },
    "dark_css_variables": {
        "color-brand-primary": "#f28c38",
        "color-brand-content": "#f28c38",
    },
}

from sphinx.builders.html import StandaloneHTMLBuilder

StandaloneHTMLBuilder.supported_image_types = ["image/gif", "image/png", "image/jpeg"]

typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = False
