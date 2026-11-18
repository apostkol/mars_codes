from __future__ import annotations

import os.path

import numpy as np
import scipy
import spiceypy as spice

from .grid import load_grid

# import pickle
# import sys
# import os


def load_pck_kernel(pck_file):
    if os.path.isfile(pck_file):
        return spice.furnsh(str(pck_file))
    else:
        raise FileNotFoundError(f"File not found: {pck_file}")


def smoothing_3d(B, kernel, mode="nearest"):

    B_smoothed = scipy.ndimage.convolve(B, kernel, mode=mode)
    nans_positions = np.where(np.isnan(B_smoothed))

    if nans_positions[0].size > 0:
        B_smoothed[nans_positions] = np.where(
            np.isfinite(B[nans_positions]),
            B[nans_positions],
            B_smoothed[nans_positions],
        )
    return B_smoothed


def init_grid(
    data_file_name=None,
    load_from_file=False,
    file_path=None,
    dr=None,
    dth=None,
    dphi=None,
    r_lim=None,
    th_lim=None,
    phi_lim=None,
):

    if data_file_name is not None and not load_from_file:
        grid_dict = dict(
            dr=dr,
            dth_deg=dth,
            dphi_deg=dphi,
            r_lim=r_lim,
            th_lim_deg=th_lim,
            phi_lim_deg=phi_lim,
        )

        grid = load_grid(grid_dict, load_from_file=load_from_file, data=data_file_name)

    elif file_path is not None and load_from_file:
        grid = load_grid(load_from_file=load_from_file, file_path=file_path)

    return grid


def binning(grid, data_file_name=None):  # , pck_file=None):
    """[Deprecated] Backward-compatible wrapper. Use grid.bin_data() instead.

    Bins magnetic field data into the grid and populates cell values.
    This function is maintained for backward compatibility; new code should use
    grid.bin_data(data_file_name) directly.

    Parameters
    ----------
    grid : Grid
        The grid to populate with binned data.
    data_file_name : str, optional
        Path to the CSV data file.

    Returns
    -------
    grid : Grid
        The populated grid instance.
    """
    return grid.bin_data(data_file_name)
