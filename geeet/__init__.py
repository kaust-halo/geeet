"""Top-level package for geeet."""

__author__ = """Oliver Lopez"""
__email__ = 'lopezv.oliver@gmail.com'
__version__ = '0.2.0'

import geeet.ptjpl
import geeet.tseb

# Optional gee features (requires earthengine-api to be installed)
try:
    import geeet.eepredefined
except ImportError:
    pass