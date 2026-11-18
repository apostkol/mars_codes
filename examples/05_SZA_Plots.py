# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: mars-currents (3.13.6)
#     language: python
#     name: python3
# ---

# %%
# Jupyter notebook test flag (set to False for full production runs)
DEBUG = globals().get("DEBUG", False)
if DEBUG:
    msg = "Not finished writing this example, no debuggery allowed."
    # raise NotImplementedError(msg)
    print(msg)
    exit
# %%
# %matplotlib widget

import mars_currents.config as cfg
from mars_currents.grid import load_grid

# %%
loaded_grids = {}
grid_colors = {}

location = cfg.RESULTS
grids_to_load = [
    (
        "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle",
        "All_",
        "black",
    ),
    # ("grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_NH.pickle", 'NH_', "C0"),
    # ("grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_high_pdyn.pickle", 'High_', "C3"),
    # ("grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_low_pdyn.pickle", 'Low_', "C1"),
]

for grid_pickle, global_prefix, color in grids_to_load:
    loaded_grids[global_prefix] = load_grid(
        load_from_file=True, file_path=location / grid_pickle
    )
    grid_colors[global_prefix] = color

grid = loaded_grids["All_"]

# %%
