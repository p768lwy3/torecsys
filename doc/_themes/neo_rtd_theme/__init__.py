"""
Neo-RTD theme for Sphinx documentation generator. 

Based on the color combination of the original sphinx_rtd_theme, but updated with better readability. 
"""
import os

__version__ = '1.0'
__version_full__ = __version__


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    return cur_dir
