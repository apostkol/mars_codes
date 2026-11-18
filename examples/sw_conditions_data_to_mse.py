# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import pandas as pd

from mars_currents import mse_processing as data_mse

# %%
df_weak = pd.read_csv("mag_mso_2014-10-11_2020-08-13_n20_weak_sw.csv", index_col=0)

# %%
bow_shock_pars = {"x0": 0.74, "L": 1.82, "eps": 1.01}
df_weak

# %%
df_orbits = pd.read_csv(
    "../maven_orbits.txt", sep=", ", header=2, engine="python"
)  # , names=["a", "b", "c"])
df_orbits = df_orbits[
    (df_orbits["START_ET"] >= df_weak.loc[df_weak.index[0], "time"] - 86400 * 2)
    & (df_weak.loc[df_weak.index[-1], "time"] + 86400 >= df_orbits["START_ET"])
]
df_orbits = df_orbits.drop(
    columns=[
        "ORBIT_NUMBER",
        "START_ET",
        "FINISH_ET",
        "START_UTC",
        "FINISH_UTC",
        "PERIAPSIS_UTC",
    ],
    axis=1,
)

# %%
smooth_factor = 15
aberration = 4

data_dict = dict(
    frame="ss",
    sampl="1s",
    start=None,
    end=None,
    freq_sampl=20,
    folder_path="../mag/pc1s/l2",
    kernel_path="../maven_spice/maven_time_frames.tm",
    orbits_file_path="../maven_orbits.txt",
    spice_pck="pck00010.tpc",
)

# %%
df_weak = data_mse.rotated_to_mse(
    **data_dict,
    aberration=aberration,
    smooth_factor=smooth_factor,
    bow_shock_pars=bow_shock_pars,
    save=[True, "_weak_sw"],
    out=True,
    df=df_weak,
    df_orbits=df_orbits,
)

# %%
fig, ax = plt.subplots(4, 1)
# df_weak[0].R_mse.plot.line('.', markersize=0.4)
ax[0].plot(df_weak[0].time, df_weak[0].R_mse)
ax[1].plot(df_weak[0].time, df_weak[0].Z_mse)
ax[2].plot(df_weak[0].time, df_weak[0].clock_angle)
ax[3].plot(df_weak[0].time, df_weak[0]["loc"], ".")
