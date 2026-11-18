# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %matplotlib widget
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mars_currents.config as cfg
from mars_currents.generic import angle_difference
from mars_currents.grid import load_grid


# %%
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
        ax.annotate(next(self.iter), self.xy, **self.kwargs)


# %%
loaded_grids = {}
grid_colors = {}

location = cfg.PLOTS
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
]

if not DEBUG:
    for grid_pickle, global_prefix, color in grids_to_load:
        try:
            loaded_grids[global_prefix] = load_grid(
                load_from_file=True, file_path=location / grid_pickle
            )
            grid_colors[global_prefix] = color
        except Exception as e:
            print(f"Error loading {grid_pickle}: {e}")
            raise
    else:
        global_prefix = "All_"
        color = grid_colors[global_prefix]
        grid = loaded_grids[global_prefix]

else:
    # If we're debugging the whole thing, we're just going to grab the smallest file
    # size grid
    files = cfg.RESULTS.glob("*.pickle")
    smallest_file = min(files, key=lambda f: f.stat().st_size)
    grid = load_grid(load_from_file=True, file_path=smallest_file)
    loaded_grids["All_"] = grid
    grid_colors["All_"] = "black"
    global_prefix = "All_"

# %%
# Simple extraction (scalar input)
Jr_value = grid.sample_fields(r=1.5, theta=np.pi / 4, phi=0.0, field_name="Jr")

# Vector extraction with options
Jr_values, (r_idx, th_idx, phi_idx), (r_c, th_c, phi_c) = grid.sample_fields(
    r=[1.5, 1.6, 1.7],
    theta=[np.pi / 4, np.pi / 3, np.pi / 2],
    phi=[0.0, 0.1, 0.2],
    field_name="Jr",
    return_indices=True,
    return_centers=True,
)


# Extract magnetic field magnitude
def vector_processor(B_vector=None):
    shape = (3,)
    if B_vector is None:
        return np.zeros(shape) + np.nan
    out = np.array(B_vector)
    assert out.shape == shape, f"Expected shape {shape} but got {out.shape}"
    return out


B_magnitude = grid.sample_fields(
    r=1.5, theta=np.pi / 4, phi=0.0, field_name="B_mse_mean", processor=vector_processor
)
# print(B_magnitude)

r_pos = np.linspace(1.0, 3.0, 30)
theta_pos = np.zeros_like(r_pos) + np.pi / 2
phi_pos = np.zeros_like(r_pos)

b_values, (r_idx, th_idx, phi_idx), (r_c, th_c, phi_c), cells = grid.sample_fields(
    r=r_pos,
    theta=theta_pos,
    phi=phi_pos,
    field_name="B_mse_mean",
    processor=vector_processor,
    return_indices=True,
    return_centers=True,
    return_cells=True,
)


b_r_values, (r_idx, th_idx, phi_idx), (r_c, th_c, phi_c), cells = grid.sample_fields(
    r=r_pos,
    theta=theta_pos,
    phi=phi_pos,
    field_name="Br",
    return_indices=True,
    return_centers=True,
    return_cells=True,
)
plt.figure()
plt.plot(r_pos, b_values[:, 0], label="Br")
plt.plot(r_pos, b_values[:, 1], label="Btheta")
plt.plot(r_pos, b_values[:, 2], label="Bphi")
plt.plot(r_pos, b_r_values, label="Br2")
plt.xlabel("r (R_M)")
plt.ylabel("B (nT)")
plt.title("Radial Magnetic Field Along the Subsolar Line")
plt.legend()
plt.grid()
# plt.show()


# %%
def radial_current_plot(center=np.pi, width=np.pi / 2, ax=None, grid_name=None):
    ns_currents = dict(r=[], positive=[], negative=[])

    grid = loaded_grids[grid_name]
    # color = grid_colors[grid_name]

    for r_min, r_max in zip(grid.r_bins[:-1], grid.r_bins[1:]):
        positive_current, negative_current = [], []

        for c in grid.cells:
            # if (c.r_c > r_min) and (c.r_c <= r_max) and ((c.phi_c < -pio2) or (c.phi_c > pio2)):
            if (
                (c.r_c > r_min)
                and (c.r_c <= r_max)
                and (abs(angle_difference(c.phi_c, center)) < width)
            ):
                v = c.Ir / 1e3 / 1e9  # in kA [from nA]

                if not np.isfinite(v):
                    continue

                if v >= 0:
                    positive_current.append(v)
                else:
                    negative_current.append(v)

            # print(c)
            # raise StopIteration
        if not positive_current and not negative_current:
            # raise StopIteration("No cells found")
            continue
        else:
            # print(positive_current)
            pass

        r = (r_min + r_max) / 2
        # print(f"r={r:.2f} RM: +I = {np.sum(positive_current):.2f} kA, -I = {np.sum(negative_current):.2f} kA")
        # print(f"r={r:.2f} RM: +I STD = {np.std(positive_current):.2f} kA, -I STD = {np.std(negative_current):.2f} kA")
        # print("----")
        ns_currents["r"].append(r)
        ns_currents["positive"].append(np.sum(positive_current))
        ns_currents["negative"].append(np.sum(negative_current))

    ns_currents["r"] = np.array(ns_currents["r"])
    ns_currents["positive"] = np.array(ns_currents["positive"])
    ns_currents["negative"] = np.array(ns_currents["negative"])

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    plt.plot(ns_currents["r"], ns_currents["positive"], label=r"I$^+$", color="r")
    plt.plot(ns_currents["r"], ns_currents["negative"], label=r"I$^-$", color="b")
    plt.plot(
        ns_currents["r"],
        ns_currents["positive"] + ns_currents["negative"],
        label=r"$\Delta$ I = I$^+$+I$^-$",
        color="k",
        ls="-",
    )
    plt.plot(
        ns_currents["r"],
        (ns_currents["positive"] - ns_currents["negative"]) / 2,
        label=r"I$^*$ = (I$^+$-I$^-$)/2",
        color="darkgrey",
        ls="--",
    )

    if abs(angle_difference(center, np.pi)) < 1e-3:
        ax.set_title("Currents in the nightside sector")
    elif abs(angle_difference(center, 0.0)) < 1e-3:
        ax.set_title("Currents in the dayside sector")
    else:
        ax.set_title(
            f"Currents in the sector centered at {center * 180 / np.pi:.0f} deg"
        )
    plt.ylabel(r"I$_R$ [kA]")
    plt.xlabel(r"r [R$_M$]")
    plt.legend()
    plt.grid()

    return ns_currents


f = FigLabeler()
fig, ax = plt.subplots(
    1, 2, figsize=(10, 5), constrained_layout=True, sharey=True, sharex=True
)
radial_current_plot(0.0, ax=ax[0], grid_name="All_")
f.label()

radial_current_plot(np.pi, ax=ax[1], grid_name="All_")
f.label()
plt.savefig(location / f"{global_prefix}tail_currents.pdf")
#  plt.show()

fig, ax = plt.subplots(
    1, 1, figsize=(10, 5), constrained_layout=True, sharey=True, sharex=True
)
radial_current_plot(0.0, ax=ax, width=np.pi, grid_name="All_")
ax.set_title("Everythingeeeeeeeeeeeeee")
# plt.show()


# %%
def radial_current_profile(
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

    returns:
        dict of arrays with keys 'dimension', 'positive', 'negative', 'total'
    """

    if ax is not None:
        plot = True

    if ax is None and plot:
        ax = plt.gca()

    currents = dict(
        r=[],
        positive=[],
        negative=[],
        positive_std=[],
        negative_std=[],
        total=[],
        total_std=[],
        dimension=dimension,
        phi=phi,
        theta=theta,
    )

    labels = dict(r="r", theta=r"$_\theta$", phi=r"$_\phi$")
    if dimension not in ["r", "theta", "phi"]:
        raise ValueError("dimension must be 'r', 'theta', or 'phi'")

    if not dimension == "r":
        raise ValueError("Currently only dimension='r' is valid")

    for r_min, r_max in zip(grid.r_bins[:-1], grid.r_bins[1:]):
        positive_current, negative_current = [], []

        for c in grid.cells:
            if (
                (c.r_c > r_min)
                and (c.r_c <= r_max)
                and (abs(angle_difference(c.phi_c, phi[0])) < phi[1])
                and (c.th_c >= theta[0])
                and (c.th_c <= theta[1])
            ):
                if dimension == "r":
                    v = c.Ir
                elif dimension == "theta":
                    v = c.Ith
                elif dimension == "phi":
                    v = c.Iphi
                v = v / 1e3 / 1e9  # in kA [from nA]

                if not np.isfinite(v):
                    continue

                if v >= 0:
                    positive_current.append(v)
                else:
                    negative_current.append(v)

        if not positive_current and not negative_current:
            # raise StopIteration("No cells found")
            continue
        else:
            # print(positive_current)
            pass

        r = (r_min + r_max) / 2
        currents["r"].append(r)
        currents["positive"].append(np.sum(positive_current))
        currents["negative"].append(np.sum(negative_current))

        currents["positive_std"].append(
            np.std(positive_current) * np.sqrt(len(positive_current))
        )
        currents["negative_std"].append(
            np.std(negative_current) * np.sqrt(len(negative_current))
        )

        currents["total"].append(np.sum(positive_current + negative_current))
        currents["total_std"].append(
            np.std(positive_current + negative_current)
            * np.sqrt(len(positive_current + negative_current))
        )

    for k in currents.keys():
        currents[k] = np.array(currents[k])

    if plot:
        plt.sca(ax)

        plt.plot(
            currents["r"],
            currents["positive"],
            label=prefix + r"I$^+$",
            color=color,
            ls="--",
        )
        plt.fill_between(
            currents["r"],
            currents["positive"] - currents["positive_std"],
            currents["positive"] + currents["positive_std"],
            color=color,
            alpha=0.3,
        )

        plt.plot(
            currents["r"],
            currents["negative"],
            label=prefix + r"I$^-$",
            color=color,
            ls=":",
        )
        plt.fill_between(
            currents["r"],
            currents["negative"] - currents["negative_std"],
            currents["negative"] + currents["negative_std"],
            color=color,
            alpha=0.3,
        )

        plt.plot(
            currents["r"],
            currents["total"],
            label=prefix + r"$\Sigma$ I = I$^+$+I$^-$",
            color="k",
            ls="-",
        )
        plt.fill_between(
            currents["r"],
            currents["total"] - currents["total_std"],
            currents["total"] + currents["total_std"],
            color="k",
            alpha=0.3,
        )

        # plt.plot(currents['r'],  (currents['positive'] - currents['negative'])/2,
        #             label=prefix + r'$\Delta$ I = (I$^+$-I$^-$)/2', color='k',ls='-.')

        plt.ylabel("I" + labels[dimension] + "[kA]")
        plt.xlabel(r"r [R$_M$]")

    return currents


# %%
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

        plt.plot(
            currents["r"], currents["avg"], label=prefix + r"I", color=color, ls="-"
        )
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


# %%
def integrate_current(
    dimension="r",
    grid=None,
    r=(0.0, 10.0),
    phi=(np.pi, np.pi / 2),
    theta=(0.0, np.pi / 2),
    plot=False,
    ax=None,
    color="C0",
    prefix="",
):
    """Integrate current profile in specified sector.

    dimensions = 'r', 'theta', or 'phi'
    phi = (center, width) in radians
    theta = (center, width) in radians
    r = (center, width) in R_M
    # Average current density multiplied by area.

    returns:
        current, std
    """

    if ax is not None:
        plot = True

    if ax is None and plot:
        ax = plt.gca()

    # labels = dict(r="r", theta=r"$_\theta$", phi=r"$_\phi$")
    if dimension not in ["r", "theta", "phi"]:
        raise ValueError("dimension must be 'r', 'theta', or 'phi'")

    vals = []
    weights = []
    last_valid = None
    for c in grid.cells:
        if (
            (c.r_c > (r[0] - r[1]))
            and (c.r_c < (r[0] + r[1]))
            and (abs(angle_difference(c.phi_c, phi[0])) < phi[1])
            and (c.th_c > (theta[0] - theta[1]))
            and (c.th_c < (theta[0] + theta[1]))
        ):
            if dimension == "r":
                v = c.Jr
                w = c.Ir
                this_c = c.r_c

            elif dimension == "theta":
                v = c.Jth
                w = c.Ith
                this_c = c.th_c

            elif dimension == "phi":
                v = c.Jphi
                w = c.Iphi
                this_c = c.phi_c

            if last_valid is not None:
                if (this_c - last_valid) / (this_c + last_valid) > 0.001:
                    print(c.r_c, c.th_c, c.phi_c, v, w)
                    raise ValueError(f"Multiple cells {this_c} {last_valid}")
            last_valid = this_c

            v = v / 1e9 / 1e3  # in kA [from nA]
            w = w / 1e9 / 1e3  # in kA [from nA]

            if not np.isfinite(v):
                continue
            vals.append(v)
            weights.append(w)

    if not vals:
        print("No cells found")

    return np.sum(weights, dtype=np.float64), np.std(weights) * np.sqrt(len(weights))


print(grid.r_bins)
print(grid.th_bins)

dr = 0.04 / 2.0 - 0.0001
dt = np.pi / 6
dt2 = 10.0 / 180.0 * np.pi / 2.0
dp = np.pi / 2
for i in range(3):
    a, b = integrate_current(
        "r", grid, r=(1.250, dr), phi=(np.pi, dp), theta=((1 + 2 * i) * dt, dt)
    )
    print(f"Ir: {a:.1f} ± {b:.1f} kA")

    print(((1 + 2 * i) * dt, dt2))
    a, b = integrate_current(
        "theta", grid, r=(1.75, 0.25), phi=(np.pi, dp), theta=((1 + 2 * i) * dt, dt2)
    )
    print(f"It: {a:.1f} ± {b:.1f} kA")

# %%
fig, ax = plt.subplots(
    1, 2, figsize=(10, 5), constrained_layout=True, sharey=True, sharex=True
)

width = np.pi / 4

radial_current_profile2(
    "r", grid=grid, phi=(0.0, width), theta=(0.0, np.pi), ax=ax[0], prefix="Day "
)
ax[0].set_title("Dayside sector")
plt.legend()
plt.grid()

ns = radial_current_profile2(
    "r", grid=grid, phi=(np.pi, width), theta=(0.0, np.pi), ax=ax[1], prefix="Night "
)
ax[1].set_title("Nightside sector")
plt.grid()
plt.savefig(location / f"{global_prefix}radial_r_current_profile_dayside_nightside.pdf")
# plt.show()


fig, ax = plt.subplots(
    1, 2, figsize=(10, 5), constrained_layout=True, sharey=True, sharex=True
)

radial_current_profile2(
    "phi",
    grid=grid,
    phi=(-np.pi / 2, width),
    theta=(0.0, np.pi),
    ax=ax[0],
    prefix="Dawn ",
)
ax[0].set_title("Dawn sector")
plt.grid()

ns = radial_current_profile2(
    "phi",
    grid=grid,
    phi=(np.pi / 2, width),
    theta=(0.0, np.pi),
    ax=ax[1],
    prefix="Dusk ",
)
ax[1].set_title("Dusk sector")
plt.grid()

plt.savefig(location / f"{global_prefix}radial_phi_current_profile_dawn_dusk.pdf")
# plt.show()

# %%
for place, val in zip(
    ("Dayside", "Nightside", "Dawn", "Dusk"), (0.0, np.pi, -np.pi / 2, np.pi / 2)
):
    # Generate profiles for all four local-time sectors.

    width = np.pi / 4

    # # 260319?
    # width = np.pi/8 + 0.0001
    theta_edges = (0.0, np.pi / 3, 2 * np.pi / 3, np.pi)
    # theta_edges = (0., 4*np.pi/9, 5*np.pi/9, np.pi)

    fig, ax = plt.subplots(
        2, 3, figsize=(10, 10), constrained_layout=True, sharey="row", sharex=True
    )

    f = FigLabeler()

    radial_current_profile2(
        "r",
        grid=grid,
        phi=(val, width),
        theta=(theta_edges[0], theta_edges[1]),
        ax=ax[0, 0],
        prefix="N ",
    )
    ax[0, 0].set_title(f"{place} North sector")
    plt.legend()
    plt.grid()
    f.label(ax[0, 0])

    radial_current_profile2(
        "r",
        grid=grid,
        phi=(val, width),
        theta=(theta_edges[1], theta_edges[2]),
        ax=ax[0, 1],
        prefix="Eq ",
    )
    ax[0, 1].set_title(f"{place}  Equatorial sector")
    plt.grid()
    f.label(ax[0, 1])

    radial_current_profile2(
        "r",
        grid=grid,
        phi=(val, width),
        theta=(theta_edges[2], theta_edges[3]),
        ax=ax[0, 2],
        prefix="S ",
    )
    ax[0, 2].set_title(f"{place} South sector")
    plt.grid()
    f.label(ax[0, 2])

    # =======
    radial_current_profile2(
        "theta",
        grid=grid,
        phi=(val, width),
        theta=(theta_edges[0], theta_edges[1]),
        ax=ax[1, 0],
        prefix="N ",
    )
    ax[1, 0].set_title(f"{place} North sector")
    plt.grid()
    f.label(ax[1, 0])

    radial_current_profile2(
        "theta",
        grid=grid,
        phi=(val, width),
        theta=(theta_edges[1], theta_edges[2]),
        ax=ax[1, 1],
        prefix="Eq ",
    )
    ax[1, 1].set_title(f"{place} Equatorial sector")
    plt.grid()
    f.label(ax[1, 1])

    radial_current_profile2(
        "theta",
        grid=grid,
        phi=(val, width),
        theta=(theta_edges[2], theta_edges[3]),
        ax=ax[1, 2],
        prefix="S ",
    )
    ax[1, 2].set_title(f"{place} South sector")
    plt.grid()
    f.label(ax[1, 2])

    plt.savefig(location / f"{global_prefix}tail_currents_{place.lower()}.pdf")
# plt.show()


# %%
# Approximate current crossing successive MSE X planes. Each spherical cell is
# assigned to a slab by its center; its volume divided by the slab width is
# the area used for the X-current integral. This is not an exact geometric
# intersection of each spherical cell with an X plane.
def plot_x_current_profile(grid, slab_width_km=None):
    mars_radius_km = grid.radius_km
    if slab_width_km is None:
        slab_width_km = grid.dr * mars_radius_km
    if slab_width_km <= 0:
        raise ValueError("slab_width_km must be positive")

    slab_width_m = slab_width_km * 1e3
    x_positions_km = [cell.X * mars_radius_km for cell in grid.cells]
    x_edges_km = np.arange(
        np.floor(min(x_positions_km) / slab_width_km) * slab_width_km,
        np.ceil(max(x_positions_km) / slab_width_km) * slab_width_km + slab_width_km,
        slab_width_km,
    )
    x_centers_km = (x_edges_km[:-1] + x_edges_km[1:]) / 2
    positive = np.zeros(len(x_centers_km))
    negative = np.zeros(len(x_centers_km))
    plus_z = np.zeros(len(x_centers_km))
    minus_z = np.zeros(len(x_centers_km))
    counts = np.zeros(len(x_centers_km), dtype=int)

    for cell, x_km in zip(grid.cells, x_positions_km):
        # Grid arrays are indexed (phi, theta, radius); Jx is in nA/m^2.
        jx = grid.Jx[cell.phi_idx, cell.th_idx, cell.r_idx]
        if not np.isfinite(jx):
            continue

        radius_m = cell.r_c * mars_radius_km * 1e3
        dr_m = cell.dr * mars_radius_km * 1e3
        cell_volume_m3 = radius_m**2 * np.sin(cell.th_c) * dr_m * cell.dth * cell.dphi
        current_kA = jx * cell_volume_m3 / slab_width_m / 1e12
        index = min(
            np.searchsorted(x_edges_km, x_km, side="right") - 1,
            len(x_centers_km) - 1,
        )
        counts[index] += 1

        if current_kA >= 0:
            positive[index] += current_kA
        else:
            negative[index] += current_kA
        if cell.Z > 0:  # MSE Z > 0
            plus_z[index] += current_kA
        elif cell.Z < 0:  # MSE Z < 0
            minus_z[index] += current_kA

    total = positive + negative
    valid = counts > 0
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(x_centers_km[valid], total[valid], label="Total (net)", color="Blue")
    # ax.plot(
    #     x_centers_km[valid], positive[valid], label="Positive $J_x$", color="Purple"
    # )
    # ax.plot(x_centers_km[valid], negative[valid], label="Negative $J_x$", color="Grey")
    ax.plot(
        x_centers_km[valid], plus_z[valid], label="MSE $Z>0$", color="Green", ls="-"
    )
    ax.plot(
        x_centers_km[valid], minus_z[valid], label="MSE $Z<0$", color="Orange", ls="-"
    )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set(
        xlabel="MSE $x$ [km]",
        ylabel="Integrated $I_x$ [kA]",
        title="X-directed current through MSE X slabs",
    )
    ax.grid(True)
    ax.legend()
    return fig, ax


fig, ax = plot_x_current_profile(grid)
plt.show()
