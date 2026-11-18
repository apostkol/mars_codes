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
# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

import mars_currents
import mars_currents.config as cfg
from mars_currents import binning
from mars_currents.grid import load_grid
from mars_currents.maps import cartesian_plane_map, single_mollweide_map

np.seterr(divide="ignore")

# %%
file_name = cfg.RESULTS / "mag_mse_2014-10-20_2020-08-13_ms_n20_all.csv"
output_dir = cfg.PLOTS
params = dict(
    dr=0.03,
    dth=15,
    dphi=15,
    r_lim=[1.04, 2.5],
    th_lim=[0, 180],
    phi_lim=[-180, 180],
)


grid = binning.init_grid(
    data_file_name=file_name.as_posix(),
    load_from_file=False,
    file_path=cfg.RESULTS / "test.pickle",
    # dr=0.3,
    # dth=30,
    # dphi=18,
    **params,
)


restore = True
if not restore:
    grid = binning.binning(grid, data_file_name=file_name)
    grid.currents_uncertainties()
    grid.save_grid()

else:
    grid = load_grid(
        load_from_file=True,
        file_path=cfg.RESULTS
        / ("grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle"),
        # / ("grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_NH.pickle"),
        # file_path = "../grid_dr0.03_dth30_dphi30__unnamed.pickle"
    )


# %%
def inject_fake_dayside_Bphi(grid, B0=10.0, r_flip=1.5, smooth=False):
    """Inject a simple fake field into a Grid for plotting/debugging.

    Field definition:
        Br   = 0
        Bth  = 0
        Bphi = B0 / r      on dayside
             = 0           elsewhere

    with sign flip:
        B0 -> -B0 for r > r_flip

    Dayside here means x > 0, equivalent to:
        cos(phi) > 0
    i.e. phi in [-pi/2, pi/2] modulo 2*pi.

    Grid array order is assumed to be:
        (phi, theta, r)
    """

    import numpy as np

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

    print(np.mean(np.abs(Bphi) > 0.000001))

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

    return grid


# grid = inject_fake_dayside_Bphi(grid, B0=10.0, r_flip=1.5)
grid.current_densities()
grid.currents_uncertainties()
grid.calculate_JdotE()
grid.calculate_JxB()

# %%
if False:
    params = ["Bphi", "Jth", "JxB_x"]
    plane_normal = (1.0, 0.0, 0.0)
    plane_point = (0.01, 0.0, 0.0)

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))

    for i, parameter in enumerate(params):
        cartesian_plane_map(
            grid,
            parameter,
            plane_normal=plane_normal,
            plane_point=plane_point,
            arrows="B",
            cmap="RdBu_r",
            # norm=plt.Normalize(-5.,5),
            ax=axs[i],
        )
    plt.show()
    raise RuntimeError()

# %%
plt.figure()
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


plt.xlabel(r"r [R$_\mathrm{M}$]")
# plt.ylabel(r"$\int$J$\cdot$E [MW$\;\mathrm{R_M}^{-1}$]")
plt.ylabel(r"$\int$J$\cdot$E [MW$\;km^{-1}$]")
plt.ylabel(r"$\iint$J$\cdot$E d$\theta$ d$\varphi$ [MW$\;km^{-1}$]")

msg = rf" $\Sigma$ +ve: {np.nansum(pos):.0f} [MW], $\Sigma$ -ve: {np.nansum(neg):.0f} [MW], $\Sigma$ all : {np.nansum(total):.0f} [MW]"
plt.annotate(
    "\n".join(msg.split(",")),
    (0.95, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
)


plt.savefig(output_dir / "All_Integrated_Dayside.pdf")
plt.show()

# %%
vals = grid.power[:, 1:-1, :] / 1e6
v = vals.flatten()
print("Total MW:", np.nansum(v))
print("+ve MW:", np.nansum(v[v > 0.0]))
print("-ve MW:", np.nansum(v[v < 0.0]))

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "mollweide"})
for r, ax in zip((1.1, 1.5, 2.0), axs):
    single_mollweide_map(grid, "JdotE", r_value=r, ax=ax)
plt.show()

# %%
r_values = 10.0 ** np.linspace(np.log10(1.1), np.log10(2.5), 6)

fig, axs = plt.subplots(
    len(r_values),
    3,
    figsize=(15, 5 * len(r_values)),
    subplot_kw={"projection": "mollweide"},
)

axlabels = iter("abcdefghijklmnopqrstuvwxyz1234567890xxxxx")

titles = dict(Br=r"B$_r$", Jth=r"J$_{\theta}$", JdotE=r"J$\cdot$E")

for i, r_value in enumerate(r_values):
    single_mollweide_map(
        grid,
        "Br",
        arrows="B",
        ax=axs[i, 0],
        r_value=r_value,
        title_prefix=next(axlabels) + ") ",
        title=titles["Br"],
    )
    single_mollweide_map(
        grid,
        "Jth",
        arrows="J",
        ax=axs[i, 1],
        r_value=r_value,
        title_prefix=next(axlabels) + ") ",
        title=titles["Jth"],
    )
    single_mollweide_map(
        grid,
        "JdotE",
        arrows="B",
        ax=axs[i, 2],
        r_value=r_value,
        title_prefix=next(axlabels) + ") ",
        title=titles["JdotE"],
    )
plt.savefig(output_dir / "BrJthetaJdotE.pdf")
plt.show()

# %%
depths = (-1, 0, 1)

combo = 501
match combo:
    case 0:
        params = ["Br", "Bth", "Bphi"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0, 0.0, 0.0)

    case 1:
        params = ["Bphi", "Jth", "JdotE"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0.0, 0.0, 0.0)

    case 2:
        params = ["Bphi", "Jth", "JdotE"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (1.5, 0.0, 0.0)

    case 3:
        params = ["Br", "Bth", "Bphi"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0, 0.0, 0.1)

    case 4:
        params = ["JxB_x", "JxB_y", "JxB_z"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0, 0.0, 0.0)

    case 5:
        params = ["JxB_x", "JxB_y", "JxB_z"]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0, 0.0, 0.0)

    case 6:
        # Very interesting?
        params = ["Bphi", "Jth", "JdotE"]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0, 0.0, 0.0)

    case 7:
        params = ["JxB_r", "JxB_th", "JxB_phi"]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0, 0.0, 0.0)

    case 8:
        params = ["Bphi", "JxB_x", "JxB_phi"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0, 0.0, -1.5)

    case 9:
        params = ["Br", "Jr", "JxB_x"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (-1.3, 0.0, 0)

    case 33:
        params = ["Jr", "Jth", "Jphi"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0, 0.0, 0.1)

    case 10:
        params = ["Bphi", "Jth", "JxB_r"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0, 0.0, 0.0)

    case 11:
        params = [
            "Bphi",
            "Br",
            "Jr",
        ]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (0.5, -0.0, -0.5, -1.0, -1.5)

    case 12:
        params = [
            "Bsmphi",
            "Bsmr",
            "JdotE",
        ]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-1.5, -0.5, -0.0, 0.5, 1.5)

    case 111:
        params = [
            "Bx",
            "By",
            "Bz",
        ]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (1.0, -0.0, 1.0)
        depths = np.arange(-1.5, 1.6, 0.5)

    case 112:
        params = [
            "Jr",
            "Jth",
            "Jphi",
        ]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (1.0, -0.0, 1.0)
        depths = np.arange(-1.5, 1.6, 0.5)

    case 113:
        params = ["Bx", "By", "Bz", "Jx", "Jy", "Jz"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (1.0, -0.0, 1.0)
        depths = np.arange(-2.0, 1.1, 0.5)

    case 114:
        # To compare with Azari?
        params = [
            "Bx",
            "By",
            "Bz",
            "Jx",
            "Jy",
            "Jz",
        ]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-1.5, -0.75, 0.0, 0.75, 1.5)

    case 115:
        # To compare with Azari?
        params = [
            "Bx",
            "By",
            "Bz",
            "JxB_x",
            "JxB_y",
            "JxB_z",
        ]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-1.5, -0.75, 0.0, 0.75, 1.5)

    case 201:
        # To compare with Zhang05 F3 Top row?
        params = ["B_tot", "J_tot"]
        plane_normal = (0.0, 0.0, 1.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-1.5, 0.0, 1.5)

    case 202:
        # To compare with Zhang05 F3 Lower row?
        params = ["Bx", "J_tot", "Jr", "Jx", "Jth"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = np.arange(-2.0, 1.1, 0.5)

    case 203:
        params = ["Jr", "Jth", "Jphi", "Jx", "Jy", "Jz"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = np.arange(-2.0, 1.1, 0.5)

    case 31:
        # MatsH
        params = ["Bx", "Jx", "JdotE"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-1.5, -0.5, 0.0)

    case 301:
        # Zhang 2026 Fig1
        params = ["By", "Jy", "Jth"]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-1.0, 0.0, 1.0)

    case 401:
        params = ["Bx", "Jx", "JxB_x"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-0.25, 0.0, 0.25)

    case 501:
        params = ["Bx", "J_parallel_tot", "J_perp_tot"]
        plane_normal = (1.0, 0.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-2.0, -1.0, -0.5, 0.0, 0.5)

    case 502:
        params = ["By", "J_parallel_tot", "J_perp_tot"]
        plane_normal = (0.0, 1.0, 0.0)
        plane_point = (0.0, 0.0, 0.0)
        depths = (-0.5, 0.0, 0.5)

# depths = (-0.0, -1., -1.5)
fig, axs = plt.subplots(
    len(depths),
    len(params),
    sharex=True,
    sharey=True,
    figsize=(5 * len(params), len(depths) * 5),
)

for j, d in enumerate(depths):
    for i, parameter in enumerate(params):
        p = plane_point
        # offset along normal by depth
        p = (
            plane_point[0] + d * plane_normal[0],
            plane_point[1] + d * plane_normal[1],
            plane_point[2] + d * plane_normal[2],
        )

        cartesian_plane_map(
            grid,
            parameter,
            plane_normal=plane_normal,
            plane_point=p,
            arrows="J" if parameter[0] == "J" else "B",
            arrow_length_transform="sqrt",
            cmap="RdBu_r",
            scale=15.0,
            scale_units="xy",
            # norm=plt.Normalize(-5.,5),
            # cmap='Spectral_r',
            # norm=plt.Normalize(0,30),
            ax=axs[j, i],
        )
        axs[j, i].annotate(
            f"Depth {d} R$_M$",
            (0.0, 1.04),
            va="bottom",
            ha="left",
            xycoords="axes fraction",
            size=8,
        )

plt.savefig(output_dir / f"All_Depths_{''.join(params)}-{combo}.png", dpi=200)
plt.show()


# %%
# These are XY planes at successive Z offsets. Define the plane explicitly
# instead of inheriting the normal and origin from the selected combo above.
xy_plane_normal = (0.0, 0.0, 1.0)
xy_plane_point = (0.0, 0.0, 0.0)
for p in ("B", "J", "JxB_"):
    params = [p + r for r in ("r", "th", "phi")]
    depths = (-1, 0, 1)
    fig, axs = plt.subplots(
        len(depths), 3, sharex=True, sharey=True, figsize=(15, 3 * 5)
    )

    for j, d in enumerate(depths):
        for i, parameter in enumerate(params):
            cartesian_plane_map(
                grid,
                parameter,
                plane_normal=xy_plane_normal,
                plane_point=(xy_plane_point[0], xy_plane_point[1], d),
                arrows="J" if parameter[0] == "J" else "B",
                arrow_length_transform="sqrt",
                cmap="RdBu_r",
                scale=15.0,
                scale_units="xy",
                # norm=plt.Normalize(-5.,5),
                ax=axs[j, i],
            )
        axs[j, i - 2].annotate(
            f"z = {d} R$_M$",
            (0.0, 1.04),
            va="bottom",
            ha="left",
            xycoords="axes fraction",
            size=8,
        )

    plt.savefig(output_dir / f"All_XY_offsets_{p}.png", dpi=200)
    plt.show()


# %%
# if True:
#     raise RuntimeError()


# def make_slices(parameter, plane=(0.0, 0.0, 1.0)):
#     params = ["Bphi", "Jth", "JxB_r"]
#     plane_normal = (0.0, 1.0, 0.0)
#     plane_point = (0, -1.5, 0.0)


# for i, parameter in enumerate(params):
#     cartesian_plane_map(
#         grid,
#         parameter,
#         plane_normal=plane_normal,
#         plane_point=plane_point,
#         arrows="B",
#         cmap="RdBu_r",
#         # norm=plt.Normalize(-5.,5),
#         ax=axs[i],
#     )
# plt.show()

# %%
depths = np.arange(-2.0, 2.1, 0.2)
param = "JxB_z"
plane_normal = (0, 1, 0)
plane_point = (0, 0, 0)
for j, d in enumerate(depths):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    cartesian_plane_map(
        grid,
        param,
        plane_normal=plane_normal,
        plane_point=(plane_point[0], plane_point[1] + d, plane_point[2]),
        arrows="J" if param[0] == "J" else "B",
        arrow_length_transform="sqrt",
        cmap="RdBu_r",
        norm=plt.Normalize(-2e-17, 2e-17),
        scale=15.0,
        scale_units="xy",
        # norm=plt.Normalize(-5.,5),
        ax=axs,
    )
    axs.annotate(
        f"y = {d:.1f} R$_M$",
        (0.0, 1.04),
        va="bottom",
        ha="left",
        xycoords="axes fraction",
        size=8,
    )

    plt.savefig(output_dir / f"Stack_{param}_{j:03d}.png", dpi=200)
plt.close("all")


# %%
