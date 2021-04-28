# import pytorch_sphinx_theme
import os
import sys
from pprint import pformat

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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Glasses'
copyright = '2020, Francesco Saverio Zuppichini & Francesco Cicala'
author = 'Francesco Saverio Zuppichini & Francesco Cicala'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    "sphinx.ext.napoleon",
    'sphinx.ext.mathjax',
    # "pytorch_sphinx_theme",
    'sphinx.ext.viewcode',
    "recommonmark"
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# autodoc_default_options = {
#     'undoc-members': 'forward',
# }
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme_options = {
#     'pytorch_project': 'docs',
#     'canonical_url': 'https://pytorch.org/docs/stable/',
#     'collapse_navigation': False,
#     'display_version': True,
#     'logo_only': True,
# }
napoleon_include_special_with_doc = True

source_parsers = {
    '.md': 'markdown',
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = ['.rst', '.md']

latex_engine = 'xelatex'


# def setup(app):
#     def skip(app, what, name, obj, skip, options):
#         members = [
#             '__init__',
#             '__repr__',
#             '__weakref__',
#             '__dict__',
#             '__module__',
#         ]
#         return True if name in members else skip

#     app.connect('autodoc-skip-member', skip)
autoclass_content = 'both'
