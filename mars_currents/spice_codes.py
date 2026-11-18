from __future__ import annotations

import os

import spiceypy as spice

__author__ = "Apostolos Kolokotronis, David Andrews"
__email__ = "apostolos.kolokotronis@irf.se, david.andrews@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "1.0.0"
__status__ = "Prototype"

__all__ = ["load_spice_kernels"]


def load_spice_kernels(kernel_path):
    if not os.path.exists(f"{kernel_path}"):
        raise OSError(f"spice kernel: {kernel_path} was not found.")
    else:
        spice.furnsh(str(kernel_path))
    return
