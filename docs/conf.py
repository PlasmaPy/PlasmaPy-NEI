#!/usr/bin/env python3
#
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

from datetime import datetime
from pkg_resources import parse_version
from sphinx.application import Sphinx

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

from plasmapy import __version__ as release  # noqa

# -- General configuration ------------------------------------------------

autosummary_generate = True

# automodapi_group_order = (
#    "modules",
#    "classes",
#    "exceptions",
#    "warnings",
#    "functions",
#    "aliases",
#    "lite-functions",
#    "variables",
# )

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones. When extensions are removed or added, please update the section
# in docs/doc_guide.rst on Sphinx extensions.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_changelog",
    "sphinx_automodapi.automodapi",
    # "hoverxref.extension",
]

# bibtex_bibfiles = ["bibliography.bib"]
# bibtex_default_style = "plain"
# bibtex_reference_style = "author_year"

# Intersphinx generates automatic links to the documentation of objects
# in other packages. When mappings are removed or added, please update
# the section in docs/doc_guide.rst on references to other packages.
intersphinx_mapping = {
    "readthedocs": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "plasmapy": ("https://docs.plasmapy.org/en/stable/", None),
    "sphinx_automodapi": (
        "https://sphinx-automodapi.readthedocs.io/en/latest/",
        None,
    ),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}
hoverxref_intersphinx = [
    "readthedocs",
    "python",
    "numpy",
    "scipy",
    "pandas",
    "astropy",
    "pytest",
    "sphinx_automodapi",
    "sphinx",
    "numba",
]

numpydoc_show_class_members = False

autoclass_content = "both"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The root toctree document.
root_doc = "index"

# General information about the project.
project = "PlasmaPy-NEI"
author = "PlasmaPy-NEI Developers"
copyright = f"2020–{datetime.utcnow().year}, {author}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
#  Note: If plasmapy.__version__ can not be defined then it is set to 'unknown'.
#        However, release needs to be a semantic style version number, so set
#        the 'unknown' case to ''.
release = "" if release == "unknown" else release
pv = parse_version(release)
release = pv.public
version = ".".join(release.split(".")[:2])  # short X.Y version
if pv.local is not None:
    revision = pv.local[1:]  # revision number w/o the leading g
else:
    revision = ""

# This is added to the end of RST files — a good place to put substitutions to
# be used globally.
rst_epilog = ""
with open("common_links.rst") as cl:
    rst_epilog += cl.read()

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "notebooks/langmuir_samples",
    "**.ipynb_checkpoints",
    "plasmapy_sphinx",
    "common_links.rst",
]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

default_role = "obj"

# Customizations for make linkcheck using regular expressions
linkcheck_allowed_redirects = {
    r"https://doi\.org/.+": r"https://.+",  # DOI links are more persistent
    r"https://docs.+\.org": r"https://docs.+\.org/en/.+",
    r"https://docs.+\.io": r"https://docs.+\.io/en/.+",
    r"https://docs.+\.com": r"https://docs.+\.com/en/.+",
    r"https://.+\.readthedocs\.io": r"https://.+\.readthedocs\.io/en/.+",
    r"https://www\.sphinx-doc\.org": r"https://www\.sphinx-doc\.org/en/.+",
    r"https://.+/github\.io": r"https://.+/github\.io/en/.+",
    r"https://.+": r".+(google|github).+[lL]ogin.+",  # some links require logins
    r"https://jinja\.palletsprojects\.com": r"https://jinja\.palletsprojects\.com/.+",
    r"https://pip\.pypa\.io": r"https://pip\.pypa\.io/en/.+",
}

linkcheck_anchors = True
linkcheck_anchors_ignore = [
    "/room",
    r".+openastronomy.+",
    "L[0-9].+",
    "!forum/plasmapy",
]

# Use a code highlighting style that meets the WCAG AA contrast standard
pygments_style = "default"

# adapted from sphinx-hoverxref conf.py
if os.environ.get("READTHEDOCS"):
    # Building on Read the Docs
    hoverxref_api_host = "https://readthedocs.org"

    if os.environ.get("PROXIED_API_ENDPOINT"):
        # Use the proxied API endpoint
        # - A RTD thing to avoid a CSRF block when docs are using a
        #   custom domain
        hoverxref_api_host = "/_"

hoverxref_tooltip_maxwidth = 600  # RTD main window is 696px
hoverxref_auto_ref = True
hoverxref_mathjax = True
hoverxref_sphinxtabs = True

# hoverxref has to be applied to these
hoverxref_domains = ["py", "cite"]
hoverxref_roles = ["confval", "term"]

hoverxref_role_types = {
    # roles with cite domain
    "p": "tooltip",
    "t": "tooltip",
    #
    # roles with py domain
    "attr": "tooltip",
    "class": "tooltip",
    "const": "tooltip",
    "data": "tooltip",
    "exc": "tooltip",
    "func": "tooltip",
    "meth": "tooltip",
    "mod": "tooltip",
    "obj": "tooltip",
    #
    # roles with std domain
    "confval": "tooltip",
    "hoverxref": "tooltip",
    "ref": "tooltip",
    "term": "tooltip",
}

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'traditional'
# html_theme = 'agogo'
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_logo = "./_static/with-text-light-190px.png"
html_theme_options = {
    "logo_only": True,
    #
    # TOC options
    #   https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html#theme-options
    "includehidden": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# A list of prefixes that are ignored for sorting the Python module
# index (e.g., if this is set to ['foo.'], then foo.bar is shown under
# B, not F).
modindex_common_prefix = ["plasmapy."]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PlasmaPydoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    #
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    #
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    #
    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        root_doc,
        "PlasmaPy.tex",
        "PlasmaPy Documentation",
        "PlasmaPy Community",
        "manual",
    )
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "plasmapy", "PlasmaPy Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        "PlasmaPy",
        "PlasmaPy Documentation",
        author,
        "PlasmaPy",
        "Python package for plasma physics",
        "Miscellaneous",
    )
]

html_favicon = "./_static/icon.ico"

# -- NBSphinx options

nbsphinx_thumbnails = {
    "notebooks/*": "_static/graphic-circular.png",
    "notebooks/*/*": "_static/graphic-circular.png",
    "notebooks/diagnostics/langmuir_analysis": (
        "_static/notebook_images/langmuir_analysis.png"
    ),
    "notebooks/plasma/grids_cartesian": (
        "_static/notebook_images/uniform_grid_thumbnail.png"
    ),
    "notebooks/plasma/grids_nonuniform": (
        "_static/notebook_images/nonuniform_grid_thumbnail.png"
    ),
    "notebooks/getting_started/units": "_static/notebook_images/astropy_logo_notext.png",  # CC BY-SA
}

# adapted from
# https://github.com/spatialaudio/nbsphinx/blob/58b8034dd9d7349c1b4ac3e7a7d6baa87ab2a6a9/doc/conf.py

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
{% set nb_base = 'tree' if env.config.revision else 'blob' %}
{% set nb_where = env.config.revision if env.config.revision else 'main' %}

.. raw:: html

    <div class="admonition note">
      <p style="margin-bottom:0px">
        This page was generated by
        <a href="https://nbsphinx.readthedocs.io/">nbsphinx</a> from
        <a class="reference external" href="https://github.com/PlasmaPy/PlasmaPy/{{ nb_base|e }}/{{ nb_where|e }}/{{ docname|e }}">{{ docname|e }}</a>.
        <br>
        Interactive online version:
        <a href="https://mybinder.org/v2/gh/PlasmaPy/PlasmaPy/{{ nb_where|e }}/?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.
      </p>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""


def setup(app: Sphinx) -> None:
    app.add_config_value("revision", "", True)
    app.add_css_file("css/admonition_color_contrast.css")
    app.add_css_file("css/plasmapy.css", priority=600)
