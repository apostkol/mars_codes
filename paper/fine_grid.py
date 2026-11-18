DEBUG = globals().get("DEBUG", False)

import matplotlib.pyplot as plt
import spiceypy as spice

import mars_currents
import mars_currents.config as cfg
from mars_currents import binning
from mars_currents.grid import Grid, load_grid
from mars_currents.maps import mollweide_maps

# MS only?
data_file_name = cfg.RESULTS / "mag_mse_2014-10-20_2020-08-13_ms_n20_all.csv"


data_file_name = cfg.RESULTS / "mag_mse_2014-10-20_2020-08-13_n20_all.csv"

params = dict(
    dr=0.01,
    dth_deg=18,
    dphi_deg=30,
    r_lim=[1.03, 2.9],
    # th_lim=[0, 180],
    # phi_lim=[-180, 180],
    name="FineGrid2",
    file_name="finegrid2",
)
restore = False
if not restore:
    grid = Grid(**params)

    grid.bin_data(data_file_name=data_file_name)
    grid.current_densities()
    grid.currents_uncertainties()
    grid.calculate_JxB()
    grid.calculate_JdotE()
    grid.save_grid()
else:
    pass
