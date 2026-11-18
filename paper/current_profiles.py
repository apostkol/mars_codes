DEBUG = globals().get("DEBUG", False)

# %matplotlib widget
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mars_currents.config as cfg
from mars_currents.generic import angle_difference
from mars_currents.grid import load_grid

savefig_kwargs = dict(bbox_inches="tight", pad_inches=0)


class FigLabeler:
    def __init__(self, xy=(0.02, 0.98), **kwargs):
        self.iter = iter("abcdefghijklmnopqrstuvwxyz")
        self.xy = xy
        defaults = dict(ha="left", va="top", xycoords="axes fraction", size="large")
        defaults.update(kwargs)
        self.kwargs = defaults

    def label(self, ax=None):
        if ax is None:
            ax = plt.gca()
        try:
            v = next(self.iter)
        except StopIteration:
            v = ""
        ax.annotate(v, self.xy, **self.kwargs)

    def turn_off(self):
        self.iter = iter("")


def radial_current_profile2(
    dimension="r",
    grid=None,
    phi=(np.pi, np.pi / 2),
    theta=(0.0, np.pi / 2),
    plot=False,
    ax=None,
    color="C0",
    prefix="",
):
    """Calculate radial current profile in specified sector.

    dimensions = 'r', 'theta', or 'phi'
    phi = (center, width) in radians
    theta = (lower, upper) in radians

    # Average current density multiplied by area.

    returns:
        dict of arrays with keys 'dimension', 'positive', 'negative', 'total'
    """

    if ax is not None:
        plot = True

    if ax is None and plot:
        ax = plt.gca()

    currents = dict(r=[], avg=[], avg_std=[], dimension=dimension, phi=phi, theta=theta)

    labels = dict(r="$_r$", theta=r"$_\theta$", phi=r"$_\phi$")
    if dimension not in ["r", "theta", "phi"]:
        raise ValueError("dimension must be 'r', 'theta', or 'phi'")

    for r_min, r_max in zip(grid.r_bins[:-1], grid.r_bins[1:]):
        vals = []
        weights = []
        for c in grid.cells:
            if (
                (c.r_c > r_min)
                and (c.r_c <= r_max)
                and (abs(angle_difference(c.phi_c, phi[0])) < phi[1])
                and (c.th_c >= theta[0])
                and (c.th_c <= theta[1])
            ):
                if dimension == "r":
                    v = c.Jr
                    w = c.Ir
                elif dimension == "theta":
                    v = c.Jth
                    w = c.Ith
                elif dimension == "phi":
                    v = c.Jphi
                    w = c.Iphi
                v = v / 1e9  # in A [from nA]
                w = w / 1e9  # in A [from nA]
                if not np.isfinite(v):
                    continue
                vals.append(v)
                weights.append(w)

        r = (r_min + r_max) / 2
        currents["r"].append(r)
        if not vals:
            v = np.nan
            std = np.nan
        else:
            # w2 = np.array(weights)
            # Weighting by absolute current; alternative approaches could be used.
            weights = np.abs(np.array(weights) / np.sum(weights))
            v = np.average(vals, weights=weights)
            std = np.std(vals)

        rm = 3.3895e6  # in meters
        dtheta = theta[1] - theta[0]
        dphi = phi[1] * 2
        dr = (r_max - r_min) * rm  # in meters
        this_r = r * rm
        sint = np.sin((theta[0] + theta[1]) / 2)

        per_rm = True
        if dimension == "r":
            # To calculate total current through spherical shell segment:
            area = this_r**2 * dtheta * dphi * sint
            per_rm = False

        elif dimension == "theta":
            # To calculate current per unit radial distance:
            area = dr * this_r * sint * dphi
            area = area / (r_max - r_min)

        elif dimension == "phi":
            # To calculate current per unit radial distance:
            area = dr * this_r * dtheta
            area = area / (r_max - r_min)

        # in kA:
        v = v * area / 1e3
        std = std * area / 1e3

        currents["avg"].append(v)
        currents["avg_std"].append(std)

        # if dimension == 'r':
        #     print(r, v, np.sum(w2)/1e3)

    for k in currents.keys():
        currents[k] = np.array(currents[k])

    if plot:
        plt.sca(ax)

        plt.plot(currents["r"], currents["avg"], label=prefix, color=color, ls="-")
        plt.fill_between(
            currents["r"],
            currents["avg"] - currents["avg_std"],
            currents["avg"] + currents["avg_std"],
            color=color,
            alpha=0.3,
        )

        unit = "[kA]"
        if per_rm:
            unit = "[kA/R$_M$]"
        plt.ylabel("I" + labels[dimension] + " " + unit)
        plt.xlabel(r"r [R$_M$]")

    return currents


def plot_profile(
    loaded_grids=None,
    names=None,
    grid_colors=None,
    place="Dayside",
    display_names=None,
):

    places = {
        "Dayside": 0.0,
        "Nightside": np.pi,
        "Dawn": -np.pi / 2,
        "Dusk": np.pi / 2,
    }
    width = np.pi / 4
    val = places[place]

    # width = np.pi/8 + 0.0001
    theta_edges = (0.0, np.pi / 3, 2 * np.pi / 3, np.pi)
    # theta_edges = (0., 4*np.pi/9, 5*np.pi/9, np.pi)

    fig, ax = plt.subplots(
        2, 3, figsize=(10, 10), constrained_layout=True, sharey="row", sharex=True
    )

    f = FigLabeler()

    for grid, global_prefix, color in zip(
        [loaded_grids[n] for n in names], names, [grid_colors[n] for n in names]
    ):
        radial_current_profile2(
            "r",
            grid=grid,
            phi=(val, width),
            theta=(theta_edges[0], theta_edges[1]),
            ax=ax[0, 0],
            prefix=display_names[global_prefix] if display_names else global_prefix,
            color=color,
        )
        ax[0, 0].set_title(f"{place} North sector")
        plt.legend()
        plt.grid(True)
        f.label(ax[0, 0])

        radial_current_profile2(
            "r",
            grid=grid,
            phi=(val, width),
            theta=(theta_edges[1], theta_edges[2]),
            ax=ax[0, 1],
            prefix=display_names[global_prefix] if display_names else global_prefix,
            color=color,
        )
        ax[0, 1].set_title(f"{place}  Equatorial sector")
        plt.grid(True)
        f.label(ax[0, 1])

        radial_current_profile2(
            "r",
            grid=grid,
            phi=(val, width),
            theta=(theta_edges[2], theta_edges[3]),
            ax=ax[0, 2],
            prefix=display_names[global_prefix] if display_names else global_prefix,
            color=color,
        )
        ax[0, 2].set_title(f"{place} South sector")
        plt.grid(True)
        f.label(ax[0, 2])

        # =======
        radial_current_profile2(
            "theta",
            grid=grid,
            phi=(val, width),
            theta=(theta_edges[0], theta_edges[1]),
            ax=ax[1, 0],
            prefix=display_names[global_prefix] if display_names else global_prefix,
            color=color,
        )
        ax[1, 0].set_title(f"{place} North sector")
        plt.grid(True)
        f.label(ax[1, 0])

        radial_current_profile2(
            "theta",
            grid=grid,
            phi=(val, width),
            theta=(theta_edges[1], theta_edges[2]),
            ax=ax[1, 1],
            prefix=display_names[global_prefix] if display_names else global_prefix,
            color=color,
        )
        ax[1, 1].set_title(f"{place} Equatorial sector")
        plt.grid(True)
        f.label(ax[1, 1])

        radial_current_profile2(
            "theta",
            grid=grid,
            phi=(val, width),
            theta=(theta_edges[2], theta_edges[3]),
            ax=ax[1, 2],
            prefix=display_names[global_prefix] if display_names else global_prefix,
            color=color,
        )
        ax[1, 2].set_title(f"{place} South sector")
        plt.grid(True)
        f.label(ax[1, 2])
        f.turn_off()

    for a in ax.flat:
        a.set_xlim(1.0, 2.5)

    plt.savefig(
        cfg.PLOTS / (global_prefix + f"tail_currents_{place.lower()}.pdf"),
        **savefig_kwargs,
    )
    # plt.show()


def dayside_power_profiles(
    loaded_grids=None, names=None, display_names=None, grid_colors=None, place=None
):
    output_dir = cfg.PLOTS

    fig, axs = plt.subplots(
        len(names),
        1,
        figsize=(4, len(names) * 3),
        constrained_layout=True,
        sharey=True,
        sharex=True,
    )

    f = FigLabeler()

    for grid, global_prefix, color, ax in zip(
        [loaded_grids[n] for n in names], names, [grid_colors[n] for n in names], axs
    ):
        plt.sca(ax)
        grid.calculate_JdotE()
        grid.calculate_JxB()

        pos = []
        neg = []
        total = []

        pos_jdote = []
        neg_jdote = []
        total_jdote = []

        dayside = (grid.phi_bins > -np.pi / 2) & (grid.phi_bins < np.pi / 2)
        # dayside = ~dayside

        for i, r in enumerate(grid.r_bins):
            jdote_vals = grid.JdotE[dayside, :, i]
            power_vals = grid.power[dayside, :, i] / 1e6

            inx = power_vals > 0.0
            pos.append(np.nansum(power_vals[inx]))
            neg.append(np.nansum(power_vals[~inx]))
            total.append(np.nansum(power_vals))

            pos_jdote.append(np.nansum(jdote_vals[inx]))
            neg_jdote.append(np.nansum(jdote_vals[~inx]))
            total_jdote.append(np.nansum(jdote_vals))

        pos = np.array(pos)
        neg = np.array(neg)
        total = np.array(total)

        pos_jdote = np.array(pos_jdote)
        neg_jdote = np.array(neg_jdote)
        total_jdote = np.array(total_jdote)

        plt.axhline(0.0, ls="--", c="k")
        # plt.plot(grid.r_bins, pos_jdote, "r-")
        # plt.plot(grid.r_bins, neg_jdote, "b-")
        # plt.plot(grid.r_bins, total_jdote, "k-")

        dr = grid.r_bins[1] - grid.r_bins[0]
        dr = dr * 3396.0
        print("Scaling by dr = {:.3f}".format(dr))

        # in km:
        plt.plot(grid.r_bins, pos / dr, "r-")
        plt.plot(grid.r_bins, neg / dr, "b-")
        plt.plot(grid.r_bins, total / dr, "k-")

        f.label(ax)
        plt.annotate(
            display_names[global_prefix] if display_names else global_prefix,
            (0.1, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )

        # plt.ylabel(r"$\int$J$\cdot$E [MW$\;\mathrm{R_M}^{-1}$]")
        plt.ylabel(r"$\int$J$\cdot$E [MW$\;km^{-1}$]")
        plt.ylabel(r"$\iint$J$\cdot$E d$\theta$ d$\varphi$ [MW$\;km^{-1}$]")

        msg = rf" $\Sigma$ +ve: {np.nansum(pos):.0f} [MW], $\Sigma$ -ve: {np.nansum(neg):.0f} [MW], $\Sigma$ all : {np.nansum(total):.0f} [MW]"
        plt.annotate(
            "\n".join(msg.split(",")),
            (0.95, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
        )
    plt.xlim(1.0, 2.25)
    plt.xlabel(r"r [R$_\mathrm{M}$]")
    plt.savefig(output_dir / f"PowerProfile_{'_'.join(names)}.pdf", **savefig_kwargs)
    # plt.show()


if __name__ == "__main__":
    loaded_grids = {}
    grid_colors = {}

    location = cfg.RESULTS
    grids_to_load = [
        (
            "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle",
            "All_",
            "black",
        ),
        (
            "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_NH.pickle",
            "NH_",
            "C0",
        ),
        (
            "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_high_pdyn.pickle",
            "High_",
            "C3",
        ),
        (
            "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_low_pdyn.pickle",
            "Low_",
            "C1",
        ),
        (
            "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_intermediate_pdyn.pickle",
            "Intermediate_",
            "C2",
        ),
    ]

    for grid_pickle, global_prefix, color in grids_to_load:
        try:
            loaded_grids[global_prefix] = load_grid(
                load_from_file=True, file_path=location / grid_pickle
            )
            grid_colors[global_prefix] = color
        except Exception as e:
            print(f"Error loading {grid_pickle}: {e}")
            raise

    plot_profile(
        loaded_grids=loaded_grids,
        names=["All_", "NH_"],
        display_names=dict(All_="All data", NH_="Northern geographic\nhemisphere only"),
        grid_colors=grid_colors,
        place="Dayside",
    )

    plot_profile(
        loaded_grids=loaded_grids,
        names=["High_", "Low_"],
        display_names=dict(High_="High Pdyn", Low_="Low Pdyn"),
        grid_colors=grid_colors,
        place="Dayside",
    )

    dayside_power_profiles(
        loaded_grids=loaded_grids,
        names=[
            "All_",
            "NH_",
            "High_",
            "Intermediate_",
            "Low_",
        ],
        display_names=dict(
            All_="All data",
            NH_="Northern geographic\nhemisphere only",
            High_="High Pdyn",
            Low_="Low Pdyn",
            Intermediate_="Intermediate Pdyn",
        ),
        grid_colors=grid_colors,
        place="Dayside",
    )
