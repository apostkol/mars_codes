from __future__ import annotations

"""Tools for mapping magnetic fields and current densities around Mars.

The package is intentionally lightweight at import time. Modules such as
``mag_files`` and ``swia_files`` depend on scientific file readers, so import
them directly when they are needed rather than as a side effect of
``import mars_currents``.
"""

__author__ = "Apostolos Kolokotronis"
__email__ = "apostolos.kolokotronis@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "1.0.0"
__status__ = "Prototype"

__all__ = ["__version__"]
