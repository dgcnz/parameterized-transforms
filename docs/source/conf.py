# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Parameterized Transforms'
copyright = '2025, Apple Inc'
author = 'Eeshan Gunesh Dhekane'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
]

myst_enable_extensions = ['html_image']
source_suffix = ['.rst', '.md']
autoclass_content = 'both'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'Parameterized Transforms'
html_theme_options = {
    'repository_url': 'https://github.com/apple/parameterized-transforms',
    'use_repository_button': True,
    'navigation_with_keys': False,
}
html_static_path = ['_static']
