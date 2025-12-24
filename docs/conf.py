"""
Sphinx Configuration for SAGA Documentation
============================================

Generates comprehensive API documentation from docstrings.

Author: ARC SAGA Development Team
Date: December 14, 2025
"""

import sys
from pathlib import Path

# Add saga module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------

project = 'SAGA - Systematized Autonomous Generative Assistant'
copyright = '2025, ARC SAGA Development Team'
author = 'ARC SAGA Development Team'

# The full version, including alpha/beta/rc tags
release = '2.0.0-alpha'
version = '2.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',       # Support Google/NumPy docstring styles
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.intersphinx',    # Link to other project docs
    'sphinx.ext.todo',           # Support for TODO items
    'sphinx.ext.coverage',       # Check documentation coverage
    'sphinx.ext.mathjax',        # Math support
    'sphinx.ext.githubpages',    # GitHub Pages support
]

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'  # ReadTheDocs theme
html_static_path = ['_static']

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_type_aliases = {
    'AsyncSession': 'sqlalchemy.ext.asyncio.AsyncSession',
}

# Intersphinx mapping (link to other project docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'fastapi': ('https://fastapi.tiangolo.com/', None),
    'sqlalchemy': ('https://docs.sqlalchemy.org/en/14/', None),
    'pydantic': ('https://docs.pydantic.dev/', None),
}

# Todo extension
todo_include_todos = True
