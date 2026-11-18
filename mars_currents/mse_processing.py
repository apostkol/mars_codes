from __future__ import annotations

import os
from pathlib import Path

import matplotlib.patches as patches
import numpy as np
import pandas as pd
import spiceypy as spice
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

from .config import RESULTS
from .generic import Rot, bow_shock
from .mag_files import load_mag_data
from .spice_codes import load_spice_kernels

BOW_SHOCK_PARAMETERS = {"x0": 0.74, "L": 1.82, "eps": 1.01}
PLOT_MSE_DEFAULTS = {
    "fig_width": 12,
    "markersize": 1,
    "plot_bs_distance": False,
    "plot_location": False,
    "plot_crossings": {"plot": False},
}


def _validate_required(values, context):
    """Raise a clear error when required processing arguments are missing."""

    missing = [name for name, value in values.items() if value is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required arguments for {context}: {joined}")


def load_data(
    *,
    start,
    end,
    frame,
    sampl,
    freq_sampl,
    folder_path,
    kernel_path,
    orbits_file_path,
    spice_pck,
    aberration=None,
    aberation=None,
):
    """Load MAG data and orbit metadata for later MSO-to-MSE processing.

    Parameters
    ----------
    start, end : str
        UTC timestamps bounding the interval to load.
    frame : str
        MAG coordinate-frame suffix used in MAVEN file names.
    sampl : str
        MAG sampling-rate suffix used in MAVEN file names.
    freq_sampl : int or None
        Optional row-thinning interval passed to :func:`load_mag_data`.
    folder_path : str
        Root directory containing MAVEN MAG data files.
    kernel_path : str
        SPICE meta-kernel path used by :func:`load_mag_data`.
    orbits_file_path : str
        CSV file containing orbit/periapsis metadata.
    spice_pck : str
        SPICE planetary constants kernel used to read Mars' radius.
    aberration : float, optional
        Rotation angle in degrees applied around the MSO z-axis. The misspelled
        ``aberation`` keyword is still accepted for older notebooks.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        MAG samples and orbit metadata filtered to the requested interval.
    """

    if aberration is None:
        aberration = aberation

    if not os.path.isfile(spice_pck):
        raise FileNotFoundError(f"SPICE PCK file not found: {spice_pck}")
    spice.furnsh(str(spice_pck))

    # mars_radius = spice.bodvcd(499, "RADII", 3)[1][0]
    # For internal consistency with everything after, we just take:
    mars_radius = 3396.0

    output = load_mag_data(
        start,
        end,
        frame,
        sampl,
        folder_path,
        kernel_path,
        re_sampling=freq_sampl,
    )
    pre_df = {
        "time": output["time"],
        "Bx": output["b"][0],
        "By": output["b"][1],
        "Bz": output["b"][2],
        "X": output["position"][0] / mars_radius,
        "Y": output["position"][1] / mars_radius,
        "Z": output["position"][2] / mars_radius,
    }

    output.clear()
    t_end = spice.utc2et(end)
    t_start = spice.utc2et(start)
    df = pd.DataFrame(data=pre_df)
    pre_df.clear()
    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]

    df_orbits = pd.read_csv(
        orbits_file_path,
        sep=", ",
        header=2,
        engine="python",
    )  # , names=["a", "b", "c"])
    df_orbits = df_orbits[
        (df_orbits["START_ET"] >= t_start - 86400 * 2)
        & (t_end + 86400 >= df_orbits["START_ET"])
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
        # axis=1,
    )

    if aberration is not None:
        df["X_aber"], df["Y_aber"], df["Z_aber"] = np.array(
            np.dot(
                Rot(np.deg2rad(aberration), axis="z"),
                np.array([df["X"], df["Y"], df["Z"]]),
            ),
        )
        df["Bx_aber"], df["By_aber"], df["Bz_aber"] = np.array(
            np.dot(
                Rot(np.deg2rad(aberration), axis="z"),
                np.array([df["Bx"], df["By"], df["Bz"]]),
            ),
        )

    return df, df_orbits


def find_cross_idx(df):
    """Return indices where the trajectory crosses the bow-shock boundary."""

    idx_ms2sw = df[df["loc"].diff() == 1].index
    idx_sw2ms = df[df["loc"].diff() == -1].index

    return list(idx_ms2sw), list(idx_sw2ms)


def crossings(df, bow_shock_pars=None):
    """Classify samples as solar wind or induced magnetosphere.

    The bow-shock model is evaluated in a cylindrical coordinate system whose
    origin is shifted by ``x0`` along the x-axis. ``loc`` is set to ``1`` for
    samples outside the bow shock and ``0`` for samples inside it.
    """

    bow_shock_pars = BOW_SHOCK_PARAMETERS if bow_shock_pars is None else bow_shock_pars

    if "Bx_aber" in df.columns:
        subsc = "_aber"
    else:
        subsc = ""

    x0 = bow_shock_pars["x0"]
    L = bow_shock_pars["L"]
    eps = bow_shock_pars["eps"]

    positions_new = np.array([df["X" + subsc] - x0, df["Y" + subsc], df["Z" + subsc]])

    df["R"] = np.sqrt(
        df["X" + subsc] ** 2 + df["Y" + subsc] ** 2 + df["Z" + subsc] ** 2,
    )
    # df['R'+subsc] = np.sqrt(df['X'+subsc]**2+df['Y'+subsc]**2+df['Z'+subsc]**2)
    df["r_sp_cyl"] = np.sqrt(np.nansum(positions_new**2, axis=0))
    df["th_sp_cyl"] = np.arctan2(
        np.sqrt(df["Y" + subsc] ** 2 + df["Z" + subsc] ** 2),
        df["X" + subsc] - x0,
    )
    df["r_bs"] = abs(bow_shock(x0, L, eps, df["th_sp_cyl"]))
    df["loc"] = np.where(df["r_sp_cyl"] > df["r_bs"], 1, 0)

    idx_ms2sw, idx_sw2ms = find_cross_idx(df)

    return df, idx_ms2sw, idx_sw2ms


def plot_crossings(df, bow_shock_pars=None, markerstyle="-"):
    """Plot a 2D bow-shock crossing diagnostic."""

    bow_shock_pars = BOW_SHOCK_PARAMETERS if bow_shock_pars is None else bow_shock_pars

    if "Bx_aber" in df.columns:
        subsc = "_aber"
    else:
        subsc = ""

    fig, ax = plt.subplots(figsize=(10, 10))
    x0 = bow_shock_pars["x0"]
    L = bow_shock_pars["L"]
    eps = bow_shock_pars["eps"]

    yz_proj_sw = np.sqrt(
        df.loc[df["loc"] == 1, "Y" + subsc].values ** 2
        + df.loc[df["loc"] == 1, "Z" + subsc].values ** 2,
    )
    yz_proj_ms = np.sqrt(
        df.loc[df["loc"] == 0, "Y" + subsc].values ** 2
        + df.loc[df["loc"] == 0, "Z" + subsc].values ** 2,
    )
    ax.plot(
        df.loc[df["loc"] == 1, "X" + subsc].values,
        yz_proj_sw,
        f"b{markerstyle}",
        markersize=1,
        label="Solar wind",
    )

    ax.plot(
        df.loc[df["loc"] == 0, "X" + subsc].values,
        yz_proj_ms,
        f"r{markerstyle}",
        markersize=1,
        label="Induced magnetosphere",
    )
    ax.plot(x0, 0, "r*", markersize=13)

    circle_yz_proj = patches.Circle(
        (0, 0),
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
    )
    # ax.axhline(y=0, color ='k', ls = '--')
    ax.set_ylim([-0.1, 3])
    ax.set_xlim([-3, 3])
    ax.set_ylabel(
        "$\\sqrt{Z_{\\rm MSO} ^2 + Y_{\\rm MSO}^2}$ $[R_{\\rm M}]$",
        fontsize=14,
    )
    ax.set_xlabel("$X_{\\rm MSO}$ $[R_{\\rm M}]$", fontsize=14)
    ax.add_patch(circle_yz_proj)

    th = np.linspace(np.deg2rad(-140), np.deg2rad(140), 60)
    r_con = bow_shock(x0, L, eps, th)
    x_con = x0 + np.cos(th) * r_con
    y_con = np.sin(th) * r_con

    ax.plot(
        x_con,
        y_con,
        color="black",
        linestyle="--",
        label="Bow shock conic section: " + r"$L(1+\epsilon \cos(\theta))^{-1}$",
    )
    ax.legend(loc="upper left", fontsize=14)
    plt.gca().set_aspect("equal")
    # plt.show()
    return fig, ax


def clock_angles(
    *,
    smooth_factor,
    start=None,
    end=None,
    frame=None,
    sampl=None,
    freq_sampl=None,
    folder_path=None,
    kernel_path=None,
    orbits_file_path=None,
    spice_pck=None,
    aberration=None,
    aberation=None,
    bow_shock_pars=None,
    df=None,
    df_orbits=None,
):
    """Estimate IMF clock angles for solar-wind intervals.

    Pass ``df`` and ``df_orbits`` to work with already-loaded data, or provide
    the explicit MAG/orbit file arguments accepted by :func:`load_data`.
    ``smooth_factor`` is the median-filter window used on the solar-wind
    magnetic-field components before taking ``atan2(Bz, By)``.
    """

    bow_shock_pars = BOW_SHOCK_PARAMETERS if bow_shock_pars is None else bow_shock_pars
    if aberration is None:
        aberration = aberation

    if df is None or df_orbits is None:
        _validate_required(
            {
                "start": start,
                "end": end,
                "frame": frame,
                "sampl": sampl,
                "folder_path": folder_path,
                "kernel_path": kernel_path,
                "orbits_file_path": orbits_file_path,
                "spice_pck": spice_pck,
            },
            context="loading MAG data",
        )

        df, df_orbits = load_data(
            start=start,
            end=end,
            frame=frame,
            sampl=sampl,
            freq_sampl=freq_sampl,
            folder_path=folder_path,
            kernel_path=kernel_path,
            orbits_file_path=orbits_file_path,
            spice_pck=spice_pck,
            aberration=aberration,
        )

    else:
        _validate_required(
            {"kernel_path": kernel_path, "spice_pck": spice_pck},
            context="using pre-loaded data",
        )

        found_pck = False
        for i in range(spice.ktotal("ALL")):
            if spice.kdata(i, "ALL")[0][:3] == "pck":
                found_pck = True
                break
            else:
                found_pck = False

        if not found_pck:
            spice.furnsh(str(spice_pck))

        load_spice_kernels(os.path.normpath(kernel_path))

    df, idx_ms2sw, idx_sw2ms = crossings(df=df, bow_shock_pars=bow_shock_pars)

    remove = []

    if not bool(idx_sw2ms) and not bool(idx_ms2sw):
        if df["loc"].values[0] == 0:
            print("No data outside of the magnetosheath.")
        elif df["loc"].values[0] == 1:
            print("MAVEN has only been in the solar wind during the selected timespan.")
            idx_ms2sw.append(df.index[0])
            idx_sw2ms.append(df.index[-1])
    elif (not bool(idx_sw2ms)) and (bool(idx_ms2sw)):
        print("not sw to ms cross")
        idx_sw2ms.append(df.index[-1])
    elif (bool(idx_sw2ms)) and (not bool(idx_ms2sw)):
        print("not ms to sw cross")
        idx_ms2sw.append(df.index[0])
    elif idx_sw2ms[0] < idx_ms2sw[0]:
        idx_ms2sw.insert(0, df.index[0])

    if len(idx_sw2ms) > len(idx_ms2sw):
        idx_ms2sw.append(df.index[-1])
    elif len(idx_sw2ms) < len(idx_ms2sw):
        idx_sw2ms.append(df.index[-1])

    # orbit_id = (df_orbits['PERIAPSIS_ET'] < df.loc[idx_ms2sw[0], 'time']).idxmax()

    df["clock_angle"] = np.nan

    if "Bx_aber" in df.columns:
        subsc = "_aber"
    else:
        subsc = ""

    for i, j in zip(range(len(idx_sw2ms)), range(len(idx_ms2sw))):
        inx_I = idx_sw2ms[i]
        inx_J = idx_ms2sw[j]

        orbit_id = (
            min(
                df_orbits.loc[(df_orbits["PERIAPSIS_ET"] > df.loc[inx_J, "time"])].index
            )
            - 1
        )

        orbit_start = df_orbits.loc[orbit_id, "PERIAPSIS_ET"]
        orbit_end = df_orbits.loc[orbit_id + 1, "PERIAPSIS_ET"]

        if (df.loc[inx_J, "time"] >= orbit_start) and (
            df.loc[inx_I, "time"] <= orbit_end
        ):
            Bz_sw = median_filter(
                df.loc[inx_J:inx_I, "Bz" + subsc].values, smooth_factor
            )
            By_sw = median_filter(
                df.loc[inx_J:inx_I, "By" + subsc].values, smooth_factor
            )

            if inx_J != idx_ms2sw[-1]:
                df.loc[inx_J:inx_I, "clock_angle"] = np.arctan2(Bz_sw, By_sw)
                K = idx_ms2sw[j + 1]
                if abs(df.loc[K, "time"] - df.loc[inx_I, "time"]) <= (
                    orbit_end - orbit_start
                ):
                    df.loc[inx_I:K, "clock_angle"] = np.median(np.arctan2(Bz_sw, By_sw))

            else:
                df.loc[inx_J:inx_I, "clock_angle"] = np.arctan2(Bz_sw, By_sw)

                break

        else:
            remove.append([inx_I, inx_J])

    return df, remove


def rotation_to_mse(df, remove=None):
    """Rotate magnetic-field and position vectors from MSO-like axes into MSE."""

    if "Bx_aber" in df.columns:
        subsc = "_aber"
    else:
        subsc = ""

    Rot_t = np.array([Rot(-th, axis="x") for th in df["clock_angle"].values])
    B_mso = np.array(
        [df["Bx" + subsc].values, df["By" + subsc].values, df["Bz" + subsc].values],
    )
    B_mse = np.zeros(shape=B_mso.shape) * np.nan

    pos_mso = np.array(
        [df["X" + subsc].values, df["Y" + subsc].values, df["Z" + subsc].values],
    )
    pos_mse = np.zeros(shape=pos_mso.shape) * np.nan

    if remove is None:
        B_mse = np.array(np.einsum("kji,ik->jk", Rot_t, B_mso))
        pos_mse = np.array(np.einsum("kji,ik->jk", Rot_t, pos_mso))

    else:
        B_mse = np.array(np.einsum("kji,ik->jk", Rot_t, B_mso))
        pos_mse = np.array(np.einsum("kji,ik->jk", Rot_t, pos_mso))

        for pair_rem in remove:
            i_rem = pair_rem[0]
            j_rem = pair_rem[1]
            B_mse[:, j_rem:i_rem] = np.nan
            pos_mse[:, j_rem:i_rem] = np.nan

    bx_mse = B_mse[0]
    by_mse = B_mse[1]
    bz_mse = B_mse[2]
    x_mse = pos_mse[0]
    y_mse = pos_mse[1]
    z_mse = pos_mse[2]

    return bx_mse, by_mse, bz_mse, x_mse, y_mse, z_mse


def rotated_to_mse(
    *,
    smooth_factor,
    start=None,
    end=None,
    frame=None,
    sampl=None,
    freq_sampl=None,
    folder_path=None,
    kernel_path=None,
    orbits_file_path=None,
    spice_pck=None,
    aberration=None,
    aberation=None,
    bow_shock_pars=None,
    save=None,
    out=True,
    df=None,
    df_orbits=None,
    remove=None,
    output_folder=None,
):
    """Add MSE coordinates and magnetic-field components to MAG samples.

    This is the main processing step in the module. If ``clock_angle`` is
    already present in ``df``, it is reused; otherwise clock angles are computed
    from the supplied data/orbit inputs. Set ``save`` to ``(True, suffix)`` to
    write all-data and magnetosphere-only CSV files.
    """

    bow_shock_pars = BOW_SHOCK_PARAMETERS if bow_shock_pars is None else bow_shock_pars
    save = (True, "") if save is None else save
    if aberration is None:
        aberration = aberation

    if output_folder is None:
        output_folder = RESULTS
    else:
        output_folder = Path(output_folder)

    if df is None or "clock_angle" not in df.columns:
        df, remove = clock_angles(
            start=start,
            end=end,
            frame=frame,
            sampl=sampl,
            freq_sampl=freq_sampl,
            folder_path=folder_path,
            kernel_path=kernel_path,
            orbits_file_path=orbits_file_path,
            spice_pck=spice_pck,
            aberration=aberration,
            smooth_factor=smooth_factor,
            bow_shock_pars=bow_shock_pars,
            df=df,
            df_orbits=df_orbits,
        )

    df["Bx_mse"], df["By_mse"], df["Bz_mse"], df["X_mse"], df["Y_mse"], df["Z_mse"] = (
        rotation_to_mse(df, remove)
    )
    df["R_mse"] = np.sqrt(df["X_mse"] ** 2 + df["Y_mse"] ** 2 + df["Z_mse"] ** 2)
    df["th_mse"] = np.arccos(df["Z_mse"] / df["R_mse"])
    df["phi_mse"] = np.arctan2(df["Y_mse"], df["X_mse"])
    df_ms = df[(df["loc"] == 0)]  # Only inside the magnetsphere measurements

    t_start = df.loc[df.index[0], "time"]
    t_end = df.loc[df.index[-1], "time"]

    if save[0]:
        file_all = output_folder / (
            "mag_mse_"
            + spice.timout((t_start), pictur="YYYY-MM-DD")
            + "_"
            + spice.timout((t_end), pictur="YYYY-MM-DD")
            + "_n"
            + str(freq_sampl)
            + save[1]
            + ".csv"
        )
        file_ms = output_folder / (
            "mag_mse_"
            + spice.timout((t_start), pictur="YYYY-MM-DD")
            + "_"
            + spice.timout((t_end), pictur="YYYY-MM-DD")
            + "_ms_n"
            + str(freq_sampl)
            + save[1]
            + ".csv"
        )

        df.to_csv(file_all)
        print(f"\nSaved file: {file_all}, of all data.")
        df_ms.to_csv(file_ms)
        print(f"\nSaved file: {file_ms}, of data inside the bow shock.")

    if out:
        return df, df_ms

    return


def plot_mse(df, plot_dict=None):
    """Plot processed MSE magnetic-field components and diagnostics."""

    plot_dict = {**PLOT_MSE_DEFAULTS, **({} if plot_dict is None else plot_dict)}
    plot_n = 6
    if plot_dict["plot_bs_distance"]:
        plot_n += 1
    if plot_dict["plot_location"]:
        plot_n += 1

    if plot_dict["plot_crossings"]["plot"]:
        idx_ms2sw = plot_dict["plot_crossings"]["idx_ms2sw"]
        idx_sw2ms = plot_dict["plot_crossings"]["idx_sw2ms"]

    fig_width = plot_dict["fig_width"]
    fig_height = plot_n * 2

    fig, ax = plt.subplots(
        nrows=plot_n,
        ncols=1,
        figsize=(fig_width, fig_height),
        sharex=True,
    )
    t_plot = df["time"].values

    if "x_lim" in plot_dict.keys():
        x_lim = [
            spice.utc2et(plot_dict["x_lim"][0]),
            spice.utc2et(plot_dict["x_lim"][1]),
        ]

    else:
        x_lim = [t_plot[0], t_plot[-1]]

    if "dt" in plot_dict.keys():
        dt = plot_dict["dt"]

    else:
        dt = (t_plot[-1] - t_plot[0]) / 10

    times_ticks = np.arange(x_lim[0], x_lim[-1], step=dt)
    times_ticks_txt = []
    times_ticks_txt.append(
        spice.timout(times_ticks[0], pictur="HR:MN")
        + "\n"
        + spice.timout(times_ticks[0], pictur="YYYY-MM-DD"),
    )
    times_ticks_txt += [
        spice.timout(times_ticks[i], "YYYY-MM") for i in range(1, len(times_ticks))
    ]

    # y_lim = plot_dict['y_lim']
    markersize = plot_dict["markersize"]

    ax[0].plot(t_plot, df["Bx_mse"], ".k", markersize=markersize, label="Bx_mse")
    ax[0].plot(t_plot, df["By_mse"], ".g", markersize=markersize, label="By_mse")
    ax[0].plot(t_plot, df["Bz_mse"], ".r", markersize=markersize, label="Bz_mse")
    ax[0].set_xticks(ticks=times_ticks, labels=[], fontsize=14)
    ax[0].set_ylabel("$B_{{\\rm MSE}} \\, [nT]$", fontsize=14)
    ax[0].axhline(y=0, color="k", linestyle="--")
    ax[0].legend(loc="upper left", fontsize=14)

    ax[1].plot(t_plot, df["Bx_aber"], ".k", markersize=markersize)
    ax[1].set_xticks(ticks=times_ticks, labels=[], fontsize=14)
    ax[1].set_ylabel("$B_{{\\rm X,MSO}} \\, [nT]$", fontsize=14)
    ax[1].axhline(y=0, color="k", linestyle="--")

    ax[2].plot(t_plot, df["By_aber"], ".g", markersize=markersize, label="By")
    ax[2].set_xticks(ticks=times_ticks, labels=[], fontsize=14)
    ax[2].set_ylabel("$B_{{\\rm Y,MSO}} \\, [nT]$", fontsize=14)
    ax[2].axhline(y=0, color="k", linestyle="--")

    ax[3].plot(t_plot, df["Bz_aber"], ".r", markersize=markersize, label="Bz")
    ax[3].set_xticks(ticks=times_ticks, labels=[], fontsize=14)
    ax[3].set_ylabel("$B_{{\\rm Z,MSO}} \\, [nT]$", fontsize=14)
    ax[3].axhline(y=0, color="k", linestyle="--")

    ax[4].plot(
        t_plot,
        np.sqrt(df["Bx"] ** 2 + df["By"] ** 2 + df["Bz"] ** 2),
        ".k",
        markersize=markersize,
        label="Bz",
    )
    ax[4].set_ylabel(r"$|B| \, [nT]$", fontsize=14)
    ax[4].set_xticks(ticks=times_ticks, labels=[], fontsize=14)

    ax[5].plot(t_plot, np.rad2deg(df["clock_angle"]), ".b", markersize=markersize)
    # ax[5].plot(t_plot, df['Z_mse'], '.b', markersize = markersize)
    ax[5].set_xticks(ticks=times_ticks, labels=[], fontsize=14)
    ax[5].set_ylabel("$\\theta_{{\\rm cl}} \\, [\\degree]$", fontsize=14)

    ax[5].set_xticks(ticks=times_ticks, labels=[], fontsize=14)
    ax[5].axhline(y=0, color="k", linestyle="--")

    if plot_dict["plot_bs_distance"]:
        goes_to_ms = df["r_bs"].values - df["r_sp_cyl"].values
        mask = goes_to_ms < 0
        ax[6].plot(t_plot[mask], goes_to_ms[mask], ".", markersize=markersize)
        ax[6].plot(t_plot[~mask], goes_to_ms[~mask], ".", markersize=markersize)
        ax[6].set_ylabel("Distance to bowshock [R_M]", fontsize=14)
        ax[6].set_xticks(ticks=times_ticks, labels=[], fontsize=14)
        ax[6].axhline(y=0)
        ax[6].set_ylim([-3, 5])

    if plot_dict["plot_location"]:
        ax[7].plot(t_plot, df["loc"])
    # [ax[5].axvline(x=df.loc[remove[i][0], 'time'], color = 'y') for i in range(len(remove))];
    # [ax[5].axvline(x=df.loc[remove[i][1], 'time'], color = 'g') for i in range(len(remove))];

    if plot_dict["plot_crossings"]["plot"]:
        [
            ax[i].vlines(
                x=df.loc[idx_ms2sw, "time"],
                label="outbound",
                ymin=ax[i].get_ylim()[0],
                ymax=ax[i].get_ylim()[1],
                color="y",
                linestyle="--",
                linewidths=2,
            )
            for i in range(len(ax))
        ]
        [
            ax[i].vlines(
                x=df.loc[idx_sw2ms, "time"],
                label="inbound",
                ymin=ax[i].get_ylim()[0],
                ymax=ax[i].get_ylim()[1],
                color="m",
                linestyle="--",
                linewidths=2,
            )
            for i in range(len(ax))
        ]

    [ax[i].set_xlim(x_lim) for i in range(len(ax)) if bool(x_lim)]
    ax[-1].set_xticks(ticks=times_ticks, labels=times_ticks_txt)
    ax[-1].set_xlabel("$UTC$", fontsize=14)

    return fig, ax


def plot_space_coverage(
    df,
    bow_shock_pars=None,
    markersize=1,
):
    """Plot the MSE-space trajectory against the bow-shock model."""

    bow_shock_pars = BOW_SHOCK_PARAMETERS if bow_shock_pars is None else bow_shock_pars

    circle_xz = patches.Circle(
        (0, 0),
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
    )
    circle_yz = patches.Circle(
        (0, 0),
        1,
        edgecolor="black",
        facecolor="none",
        linewidth=2,
    )

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    th = np.linspace(np.deg2rad(-140), np.deg2rad(140), 60)

    x0 = bow_shock_pars["x0"]
    L = bow_shock_pars["L"]
    eps = bow_shock_pars["eps"]
    r_con = bow_shock(x0, L, eps, th)
    x_con = x0 + np.cos(th) * r_con
    y_con = np.sin(th) * r_con

    ax[0].plot(x_con, y_con, "k--", label="Bow shock")
    ax[0].plot(
        df["X_mse"],
        df["Z_mse"],
        "r.",
        markersize=markersize,
        label="trajectory in MSE",
    )
    ax[0].set_xlabel(r"$X_{\rm MSE}$ $[R_{\rm M}]$", fontsize=14)
    ax[0].set_ylabel(r"$Z_{\rm MSE}$ $[R_{\rm M}]$", fontsize=14)
    # ax[0].set_title(r"Projection on $X_{\rm MSE}$ - $Z_{\rm MSE}$ Plane", fontsize = 14)
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])
    ax[0].legend(loc="upper right", fontsize=14)
    ax[0].add_patch(circle_xz)

    # Plot on Y_MSO - Z_MSO plane
    ax[1].plot(
        df["Y_mse"],
        df["Z_mse"],
        "r.",
        markersize=markersize,
        label="trajectory in MSE",
    )
    ax[1].set_xlabel(r"$Y_{\rm MSE}$ $[R_{\rm M}]$", fontsize=14)
    ax[1].set_xlim([-3, 3])
    ax[1].set_ylim([-3, 3])
    ax[1].invert_xaxis()
    # ax[1].set_title(r"Projection on $Y_{\rm MSE}$ - $Z_{\rm MSE}$ Plane", fontsize = 14)
    ax[1].legend(loc="upper right", fontsize=14)
    ax[1].add_patch(circle_yz)

    # fig.savefig(f'{plots_path}/mse_coverage.png', bbox_inches = 'tight', dpi = 600)

    return fig, ax
