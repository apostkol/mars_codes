import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

import mars_currents
import mars_currents.config as cfg
from mars_currents import binning
from mars_currents.grid import load_grid
from mars_currents.maps import cartesian_plane_map, mollweide_maps, single_mollweide_map

grid_name = "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle"

grid = load_grid(load_from_file=True, file_path=cfg.RESULTS / grid_name)

localtimes = [-6, -3, 0, 3, 6]
localtimes = [-6, -4, -2, 0, 2, 4]

fig, axs = plt.subplots(
    1, len(localtimes), figsize=(5 * len(localtimes), 5), sharex=False, sharey=True
)

norm = plt.Normalize(-7, 7)
parameter = "Bphi"
for lt, ax in zip(localtimes, axs):
    az = lt / 12 * np.pi
    plane_normal = (np.sin(az), np.cos(az), 0.0)
    _out = cartesian_plane_map(
        grid,
        parameter,
        ax=ax,
        plane_normal=plane_normal,
        cmap="RdBu_r",
        norm=norm,
        colorbar=False,
    )
    mesh = _out[0]
    ax.set_title(f"{(lt + 24) % 24} - {(lt + 36) % 24} h LT")
    # xl = plt.xlim()
    # plt.xlim(-3, 0)
cb = fig.colorbar(mesh, ax=axs)
cb.set_label(parameter)
plt.savefig(f"Azimuth{parameter}.pdf")
plt.show()
