#!/usr/bin/env python
from __future__ import annotations

from .create_paths import create_paths
from .load_swia_data import load_swia_data
from .swia_files_list import swia_files_list

__author__ = 'Apostolos Kolokotronis'
__email__ = 'apostolos.kolokotronis@irf.se'
__copyright__ = 'Copyright 2020-2024'
__license__ = 'MIT'
__version__ = '1.0.0'
__status__ = 'Prototype'

__all__ = [
    'create_paths',
    'load_swia_data',
    'swia_files_list',
]
