import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

import mars_currents
import mars_currents.config as cfg
from mars_currents import binning
from mars_currents.grid import load_grid
from mars_currents.maps import mollweide_maps

radial_centers = [1.07, 1.11, 1.3, 1.5, 2]
radial_centers = radial_centers[::-1]

print(radial_centers)

savefig_kwargs = dict(bbox_inches="tight", pad_inches=0)

restore = True
if not restore:
    file_name = cfg.RESULTS / "mag_mse_2014-10-20_2020-08-13_ms_n20_all.csv"

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

    grid = binning.binning(grid, data_file_name=file_name)
    grid.current_densities()
    grid.currents_uncertainties()
    grid.save_grid()
    grids = [
        grid,
    ]
else:
    grids = []
    for fname in [
        "finegrid.pickle",
        # "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle",
        # "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_NH.pickle",
        # # "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_low_pdyn.pickle",
        # # "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_intermediate_pdyn.pickle",
        # # "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_high_pdyn.pickle",
        # "mag_mse_2014-10-20_2020-08-13_n20_all_NEW_low_pdyn.pickle",
        # "mag_mse_2014-10-20_2020-08-13_n20_all_NEW_intermediate_pdyn.pickle",
        # "mag_mse_2014-10-20_2020-08-13_n20_all_NEW_high_pdyn.pickle",
    ]:
        grids.append(
            load_grid(
                load_from_file=True,
                file_path=cfg.RESULTS / fname,
            )
        )


for grid in grids:
    grid.calculate_JdotE()
    plt.figure()
    plt.imshow(grid.JdotE[:, :, 5], cmap="RdBu_r")
    plt.colorbar()
    # plt.show()
    # exit()
    print(grid)

    smoothed = True
    for sm in ("sm", ""):
        mollweide_maps(
            grid,
            maps=[
                "B" + sm + "r",
                "B" + sm + "th",
                "B" + sm + "phi",
                "measurement_counts",
            ],
            maps_names=[r"$B_{r}$", r"$B_{\theta}$", r"$B_{\phi}$", "counts/bin"],
            cb_names=[r"$B \, [nT]$", ""],
            fts=14,
            hw=3,
            hl=4,
            hal=5,
            plots_path=cfg.PLOTS,
            map_type_name="magnetic_field",
            save=False,
            cb_lim=None,
            r_values=radial_centers,
            # fig=fig,
            # ax=_ax,
        )
        if sm == "sm":
            plt.savefig(cfg.PLOTS / f"{grid.file_name}-SmoothedB.pdf", **savefig_kwargs)
        else:
            plt.savefig(cfg.PLOTS / f"{grid.file_name}-RawB.pdf", **savefig_kwargs)

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
        save=False,
        cb_lim=None,
        r_values=radial_centers,
    )
    plt.savefig(cfg.PLOTS / f"{grid.file_name}-Currents.pdf", **savefig_kwargs)
