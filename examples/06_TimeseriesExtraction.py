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
    raise NotImplementedError(
        "Not finished writing this example, no debuggery allowed yet"
    )

# We're going to compare magnetic fields measured during a particular interval of
# time to those from the binned averages


# %%
# %matplotlib widget
import matplotlib.pyplot as plt

from mars_currents.grid import load_grid
import mars_currents.config as cfg
import pandas as pd
import spiceypy as sp
import numpy as np

import irfuplanets as irfpl

# %%
sp.furnsh(str(cfg.KERNELS / "maven_time_frames.tm"))
grid_pickle = "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle"

grid = load_grid(load_from_file=True, file_path=cfg.RESULTS / grid_pickle)

file_name = cfg.RESULTS / "mag_mse_2015-10-03_2015-10-31_ms_n20_all.csv"
file_name = cfg.RESULTS / "mag_mse_2014-10-20_2020-08-13_n20_all.csv"

df = pd.read_csv(file_name)


# %%
start = sp.utc2et("2015-03-16T00:00:00")
finish = sp.utc2et("2015-03-19T00:00:00")
start = sp.utc2et("2015-03-16T23:55:00")
finish = sp.utc2et("2015-03-17T00:35:00")

# start = sp.utc2et("2023-02-23T17:25:00")
# finish = sp.utc2et("2023-02-23T17:50:00")

mask = (df["time"] >= start) & (df["time"] <= finish)
print(start, finish, mask.sum())

dfm = df[mask]


# %%
# Now do the same thing but in spherical polar coordinates
from mars_currents.generic import cart_to_sph_vf, cartesian_vector_to_spherical

B_mse_cart = dfm[["Bx_mse", "By_mse", "Bz_mse"]].values
print(B_mse_cart.shape, dfm["th_mse"].values.shape, dfm["phi_mse"].values.shape)
B_mse_sph = cart_to_sph_vf(
    B_mse_cart,
    dfm["th_mse"].values,
    dfm["phi_mse"].values,
)

print(B_mse_sph[0].shape)

dfm["Br_mse"] = B_mse_sph[0]
dfm["Bth_mse"] = B_mse_sph[1]
dfm["Bphi_mse"] = B_mse_sph[2]

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

plt.sca(axs[0])
plt.plot(
    (dfm["time"].iloc[0], dfm["time"].iloc[-1]),
    [0, 0],
    label="_Zero Line",
    color="k",
    ls="--",
)
plt.plot(dfm["time"], dfm["Bx_mse"], label="Bx_MSE", color="C0")
plt.plot(dfm["time"], dfm["By_mse"], label="By_MSE", color="C1")
plt.plot(dfm["time"], dfm["Bz_mse"], label="Bz_MSE", color="C2")

bx = grid.sample_fields(dfm["R_mse"], dfm["th_mse"], dfm["phi_mse"], field_name="Bx")
by = grid.sample_fields(dfm["R_mse"], dfm["th_mse"], dfm["phi_mse"], field_name="By")
bz = grid.sample_fields(dfm["R_mse"], dfm["th_mse"], dfm["phi_mse"], field_name="Bz")

plt.plot(dfm["time"], bx, label="Bx_grid", color="C0", ls="--")
plt.plot(dfm["time"], by, label="By_grid", color="C1", ls="--")
plt.plot(dfm["time"], bz, label="Bz_grid", color="C2", ls="--")
# plt.xlabel("Time")
plt.ylabel("B_CART_MSE / nT")
plt.legend(loc="upper right")

plt.sca(axs[1])
plt.plot(
    (dfm["time"].iloc[0], dfm["time"].iloc[-1]),
    [0, 0],
    label="_Zero Line",
    color="k",
    ls="--",
)
plt.plot(dfm["time"], dfm["Br_mse"], label="Br_MSE", color="C0", alpha=0.5)
plt.plot(dfm["time"], dfm["Bth_mse"], label="Bth_MSE", color="C1", alpha=0.5)
plt.plot(dfm["time"], dfm["Bphi_mse"], label="Bphi_MSE", color="C2", alpha=0.5)

br = grid.sample_fields(dfm["R_mse"], dfm["th_mse"], dfm["phi_mse"], field_name="Br")
bth = grid.sample_fields(dfm["R_mse"], dfm["th_mse"], dfm["phi_mse"], field_name="Bth")
bphi = grid.sample_fields(
    dfm["R_mse"], dfm["th_mse"], dfm["phi_mse"], field_name="Bphi"
)

plt.plot(dfm["time"], br, label="Br_grid", color="C0", ls="--")
plt.plot(dfm["time"], bth, label="Bth_grid", color="C1", ls="--")
plt.plot(dfm["time"], bphi, label="Bphi_grid", color="C2", ls="--")

# As a test, rotating from cartesian to spherical and back again should give the same result.  RESULT: No problem
# br_test, bt_test, bp_test = cartesian_vector_to_spherical(
#     dfm["Bx_mse"].values, dfm["By_mse"].values, dfm["Bz_mse"].values,
#     dfm["th_mse"].values,
#     dfm["phi_mse"].values,
# )

# plt.plot(dfm["time"], br_test, label="Br_test", color="C0", ls=":")
# plt.plot(dfm["time"], bt_test, label="Bth_test", color="C1", ls=":")
# plt.plot(dfm["time"], bp_test, label="Bphi_test", color="C2", ls=":")


plt.legend(loc="upper right")
plt.ylabel("B_SPH_MSE / nT")

plt.sca(axs[2])
plt.plot(
    (dfm["time"].iloc[0], dfm["time"].iloc[-1]),
    [0, 0],
    label="_Zero Line",
    color="k",
    ls="--",
)

plt.plot(dfm["time"], dfm["R_mse"], label="R_MSE", color="C3")
plt.plot(dfm["time"], dfm["th_mse"], label="th_MSE", color="C4")
plt.plot(dfm["time"], dfm["phi_mse"], label="phi_MSE", color="C5")
plt.plot(dfm["time"], dfm["Z"], label="Z_MSO", color="C6")


plt.legend(loc="upper right")
plt.xlabel("Time")

import irfuplanets.time

print(irfpl.time.spiceet_to_utcstr(dfm["time"].iloc[0]))
print(irfpl.time.spiceet_to_utcstr(dfm["time"].iloc[-1]))

irfpl.time.setup_time_axis(axs[0])

plt.show()
