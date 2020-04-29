# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../agatha'))


# -- Project information -----------------------------------------------------

project = 'Agatha'
copyright = '2020, Justin Sybrandt, Ilya  Tyagin'
author = 'Justin Sybrandt, Ilya  Tyagin'

# The full version, including alpha/beta/rc tags
release = '2020-04-29'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",  # Allow markdown
    "sphinx.ext.autodoc",  # Make documentation from source
    "sphinx.ext.napoleon",  # Read docs in Google format
    "sphinx_autodoc_typehints",  # Allow automatic documentation to see hints
    "sphinx_rtd_theme",  # Provides theme
    "sphinxcontrib.apidoc",  # Automatically run apidoc on each build
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '_api/test*',
    '_api/*_pb2.rst',
]

source_suffix = {
  '.rst': 'restructuredtext',
  '.txt': 'markdown',
  '.md': 'markdown',
}

# Needed to ensure that ReadTheDocs finds the index
master_doc = "index"

# -- Options for apidoc

apidoc_module_dir = "../agatha"
apidoc_output_dir = "_api"
apidoc_separate_modules = True
apidoc_extra_args = [
    "--force",
    "--implicit-namespaces",
    "-H", "Agatha",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/sidebar_logo.png"
