# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import os

import pkg_resources

__version__ = pkg_resources.get_distribution("nnsvs").version

ON_RTD = os.environ.get("READTHEDOCS", None) == "True"

project = "nnsvs"
copyright = "2020, Ryuichi Yamamoto"
author = "Ryuichi Yamamoto"

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.youtube",
]

if ON_RTD:
    # Remove extensions not currently supported on RTD
    extensions.remove("matplotlib.sphinxext.plot_directive")

bibtex_bibfiles = ["refs.bib"]

autodoc_member_order = "bysource"

autosummary_generate = True
numpydoc_show_class_members = False

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------
doctest_global_setup = """
import numpy as np
import scipy
import librosa
np.random.seed(123)
np.set_printoptions(precision=3, linewidth=64, edgeitems=2, threshold=200)
"""

plot_pre_code = (
    doctest_global_setup
    + """
import matplotlib
import librosa
import librosa.display
matplotlib.rcParams['figure.constrained_layout.use'] = librosa.__version__ >= '0.8'
"""
)
plot_include_source = True
plot_html_show_source_link = False
plot_formats = [("png", 100), ("pdf", 100)]
plot_html_show_formats = False

plot_rcparams = {}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = []

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/dev", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
