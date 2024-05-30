# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "ReLife2"
copyright = "2024, RTE"
author = "RTE"
release = "2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "myst_parser",  # to enable markdown files
    "sphinx.ext.autodoc",  # to enable autodoc from docstrings
    "sphinx.ext.napoleon",  # to configure docstring style (use google style by default)
    "sphinx_copybutton",  # copy button in code block
    "sphinx.ext.viewcode",  # to insert source code link next to objects documentation
    "sphinx_design",  # to enable grids and other cool stuff
]
# myst_enable_extensions = [
#     "colon_fence"
# ]  # to enable when using both myst_parser and sphinx-design

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_book_theme"
html_theme_options = {"navigation_with_keys": True}
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# must add "from __future__ import annotations" in the targetted module
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "FloatArray": "FloatArray",
}
