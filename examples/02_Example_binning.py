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

import matplotlib.pyplot as plt
import spiceypy as spice

import mars_currents
import mars_currents.config as cfg
from mars_currents import binning
from mars_currents.grid import Grid, load_grid
from mars_currents.maps import mollweide_maps

# %%

data_file_name = cfg.RESULTS / "mag_mse_2014-10-20_2020-08-13_ms_n20_all.csv"
if DEBUG:
    file_name = cfg.RESULTS / "mag_mse_2015-10-03_2015-10-31_ms_n20_all.csv"
    print("Debuggering")
# %%
# # %%time

if not DEBUG:
    params = dict(
        dr=0.04,
        dth_deg=9,
        dphi_deg=15,
        r_lim=[1.055, 2.9],
        # th_lim=[0, 180],
        # phi_lim=[-180, 180],
        name="02ExampleBinning",
        file_name="test",
    )
else:
    params = dict(
        dr=0.3,
        dth_deg=30,
        dphi_deg=30,
        r_lim=[1.04, 1.5],
        # th_lim=[0, 180],
        # phi_lim=[-180, 180],
        name="02ExampleBinning",
        file_name="test",
    )

# %%
# %%time
restore = False
if not restore:
    # grid = binning.init_grid(
    #     data_file_name=file_name.as_posix(),
    #     load_from_file=False,
    #     file_path=cfg.RESULTS / "test.pickle",
    #     # dr=0.3,
    #     # dth=30,
    #     # dphi=18,
    #     **params,
    # )
    grid = Grid(**params)

    grid.bin_data(data_file_name=data_file_name)
    grid.current_densities()
    grid.currents_uncertainties()
    grid.calculate_JxB()
    grid.calculate_JdotE()
    grid.save_grid()

else:
    grid = load_grid(
        file_name=cfg.RESULTS
        / ("grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle"),
    )
    # grid.current_densities()
    # grid.currents_uncertainties()
    # grid.calculate_JxB()
    # grid.calculate_JdotE()
print(grid)

# %%
mollweide_maps(
    grid,
    maps=["Bsmr", "Bsmth", "Bsmphi", "measurement_counts"],
    maps_names=[r"$B_{r}$", r"$B_{\theta}$", r"$B_{\phi}$", "counts/bin"],
    cb_names=[r"$B \, [nT]$", ""],
    fts=14,
    hw=3,
    hl=4,
    hal=5,
    plots_path=cfg.PLOTS,
    map_type_name="magnetic_field",
    save=True,
    cb_lim=None,
    r_values="All",
)
plt.savefig("SmoothedBTest.pdf")

# %%
mollweide_maps(
    grid,
    maps=["Jr", "Jth", "Jphi", "divB_curlB"],
    maps_names=[
        r"$J_{r}$",
        r"$J_{\theta}$",
        r"$J_{\phi}$",
        r"$|\nabla \cdot B |/ |\nabla \times B|$",
    ],
    cb_names=[r"$J \, [nA/m^{2}]$", ""],
    fts=14,
    hw=3,
    hl=4,
    hal=5,
    plots_path=cfg.PLOTS,
    map_type_name="current_density",
    save=True,
    cb_lim=None,
    r_values="All",
)
plt.savefig("CurrentsTest.pdf")
