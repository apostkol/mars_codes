"""Public grid API.

The implementation is split across focused modules, while this facade keeps
backwards-compatible imports such as ``from mars_currents.grid import Grid``.
"""

from __future__ import annotations

from .geometry import (
    arc_on_xy,
    cart_to_sph,
    rot_axis_angle,
    rot_mat,
    rotate_to_xy,
    sph_to_cart,
)
from .grid_calculus import curl, div, radial_profile, sigma_gradient
from .grid_cell import Cell
from .grid_io import load_grid, save_grid
from .grid_model import Grid
from .grid_plotting import plotly_trisurf, tri_indices
from .synthetic import inject_fake_data

__author__ = "Apostolos Kolokotronis, David Andrews"
__email__ = "apostolos.kolokotronis@irf.se, david.andrews@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "1.0.0"
__status__ = "Prototype"

__all__ = [
    "Cell",
    "Grid",
    "arc_on_xy",
    "cart_to_sph",
    "curl",
    "div",
    "load_grid",
    "plotly_trisurf",
    "radial_profile",
    "rot_axis_angle",
    "rot_mat",
    "rotate_to_xy",
    "save_grid",
    "sigma_gradient",
    "sph_to_cart",
    "tri_indices",
    "inject_fake_data",
]
