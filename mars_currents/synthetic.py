"""Synthetic grid data helpers for tests, examples, and debugging."""

from __future__ import annotations

import numpy as np


def inject_fake_data(grid, B0=10.0, r_flip=1.5, smooth=False):
    """Inject a simple fake field into a Grid for plotting/debugging.

    Field definition:
        Br   = B0 in MSE NORTH, -B0 in MSE SOUTH
        Bth  = 0
        Bphi = B0 / r      on dayside
             = 0.1 B0/r    Nightside
             + with sign flip: B0 -> -B0 for r > r_flip

    """

    nphi = len(grid.phi)
    nth = len(grid.th)
    nr = len(grid.r)

    shape = (nphi, nth, nr)

    Br = np.zeros(shape)
    Bth = np.zeros(shape)
    Bphi = np.zeros(shape)
    counts = np.ones(shape)

    for i, phi in enumerate(grid.phi):
        dayside = np.cos(phi) > 0.0
        # dayside = True
        # dayside = (phi > 0.) and (phi < np.pi/2)

        for j, th in enumerate(grid.th):
            # if th < np.pi/2.:
            #     continue
            for k, r in enumerate(grid.r):
                sign = -1.0 if r > r_flip else 1.0

                if dayside:
                    Bphi[i, j, k] = sign * B0 / r
                else:
                    Bphi[i, j, k] = 0.1 * sign * B0 / r

                Br[i, j, k] = B0 / r
                if th > np.pi / 2.0:
                    Br[i, j, k] *= -1

    grid.Br = Br
    grid.Bth = Bth
    grid.Bphi = Bphi

    grid.Bsmr = Br.copy()
    grid.Bsmth = Bth.copy()
    grid.Bsmphi = Bphi.copy()

    grid.sigma_Br = np.zeros(shape)
    grid.sigma_Bth = np.zeros(shape)
    grid.sigma_Bphi = np.zeros(shape)

    grid.measurements_counts = counts
    grid.measurement_counts = counts

    # Coordinates in the same shape/order as the field arrays.
    phi_3d, th_3d, r_3d = np.meshgrid(
        grid.phi,
        grid.th,
        grid.r,
        indexing="ij",
    )

    grid.phi_2d = phi_3d
    grid.th_2d = th_3d
    grid.r_2d = r_3d

    # Current-density calculations read Cell-level values.
    grid.sync_cell_fields_from_grid()

    for cell in grid.cells:
        i = cell.phi_idx
        j = cell.th_idx
        k = cell.r_idx

        cell.Bsmr = Br[i, j, k]
        cell.Bsmth = Bth[i, j, k]
        cell.Bsmphi = Bphi[i, j, k]

        cell.sigma_Br = 0.0
        cell.sigma_Bth = 0.0
        cell.sigma_Bphi = 0.0
        cell.n_measurements = 1

    grid.current_densities(smoothed=False)
    grid.calculate_JdotE()
    grid.calculate_JxB()

    return grid
