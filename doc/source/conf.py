# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# theme used : https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html

import os
from datetime import datetime
from importlib.metadata import version as get_version

project = "relife"
author = "RTE-SAGA"
copyright = f"2007 - {datetime.now().year}, {author} (Apache 2.0 License)"
version = get_version("relife")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # to enable autodoc from docstrings
    "sphinx.ext.autosummary",
    "numpydoc",  # if used, get rid of napoleon
    "sphinx.ext.viewcode",  # to insert source code link next to objects documentation
    "sphinx.ext.githubpages",  # necessary to publish to as github pages
    # (see : https://www.sphinx-doc.org/en/master/usage/extensions/githubpages.html
    # and https://stackoverflow.com/questions/62626125/github-pages-with-sphinx-generated-documentation-not-displaying-html-correctly)
    "sphinx_copybutton",  # copy button in code block
    "nbsphinx",  # to insert notebook
    "sphinx_design",  # advanced design tools
    "myst_parser",  # to use markdown pages
]

myst_enable_extensions = ["colon_fence"]  # required by sphinx design with myst-parser

autodoc_typehints = "none"
# autosummary_generate = True

# napoleon_use_param = False
# napoleon_use_rtype = False
# autodoc_class_signature = "separated" # remove Class(*args, **kwargs)

# copied from sklearn
#########################################################################################
# We do not need the table of class members because `sphinxext/override_pst_pagetoc.py`
# will show them in the secondary sidebar
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
# We want in-page toc of class members instead of a separate page for each entry
numpydoc_class_members_toctree = False

# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
#########################################################################################

templates_path = ["_templates"]
exclude_patterns = [
    "_*",
    "user_guide/_*",
    "user_guide/.ipynb_checkpoints/*",
    "learn/_*",
    "api/_*",
]  # note to be parsed by compiler

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "ReLife documentation"  # sidebar title
html_logo = "_static/small_relife.gif"
html_favicon = "_static/relife_favicon.png"
html_js_files = ["custom-icon.js"]
html_css_files = ["css/custom.css"]  # custom css to change some colors

html_sidebars = {
    "installation": [],  # removes navigation bar for installation.rst
    "basic/index": [],  # idem
    "developper/index": [],  #  idem
}

html_theme_options = {
    "navigation_with_keys": False,
    "navbar_align": "left",  # align to the left header bar sections
    "header_links_before_dropdown": 5,  # control the number of section displayed in the header bar
    "show_prev_next": False,  # hide previous and next button
    "show_nav_level": 2,  # unfold nav section by 2 levels
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/rte-france/relife/tree/refactoring",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "PyPI",
            # URL where the link will redirect
            "url": "https://pypi.org/project/relife/",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-custom fa-pypi",
        },
    ],
    "navbar_start": ["navbar-logo", "version"],
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {"index": "index.html"}
