# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: maven_projects
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

# TODO: Delete this whole file? No longer relevant.

# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from mars_currents.grid import load_grid

mars_radius = 3396.0

# %%
base_dir = "../data/"
grid_hp = load_grid(
    load_from_file=True,
    file_path=base_dir
    + "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_high_pdyn.pickle",
)
grid_lp = load_grid(
    load_from_file=True,
    file_path=base_dir
    + "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_low_pdyn.pickle",
)
grid_all = load_grid(
    load_from_file=True,
    file_path=base_dir
    + "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle",
)


# %%
# def calculate_total_current_uncertainty(grid):

#     for cell in grid.cells:

#         cell.dI_I = np.nan

#         dIr = 0

#         if cell.neighbours['r'][+1] is not None and cell.neighbours['r'][-1] is not None:

#             if not np.isnan(cell.neighbours['r'][+1]['Jr']) and not np.isnan(cell.neighbours['r'][-1]['Jr']):

#                 dIr += cell.neighbours['r'][+1]['Jr'] * (cell.neighbours['r'][+1]['r_c'])**2 * np.sin(cell.th_c) * cell.dth * cell.dphi * (mars_radius* 1e3)**2
#             # else:
#             #     dIr += cell.Ir
#                 dIr = dIr - cell.neighbours['r'][-1]['Jr'] * (cell.neighbours['r'][-1]['r_c'])**2 * np.sin(cell.th_c) * cell.dth * cell.dphi * (mars_radius* 1e3)**2
#             # else:
#             #     dIr = dIr - cell.Ir

#         dIth = 0

#         if cell.neighbours['th'][+1] is not None and cell.neighbours['th'][-1] is not None:

#             if not np.isnan(cell.neighbours['th'][+1]['Jth']) and not np.isnan(cell.neighbours['th'][-1]['Jth']):

#                 dIth += cell.neighbours['th'][+1]['Jth'] * cell.r_c*np.sin(cell.neighbours['th'][+1]['th_c'])*cell.dr*cell.dphi *(mars_radius* 1e3)**2
#             # else:
#             #     dIth += cell.Ith
#                 dIth = dIth - cell.neighbours['th'][-1]['Jth'] * cell.r_c*np.sin(cell.neighbours['th'][-1]['th_c'])*cell.dr*cell.dphi *(mars_radius* 1e3)**2
#             # else:
#         #     dIth = dIth - cell.Ith

#         dIphi = 0

#         if cell.neighbours['phi'][+1] is not None and cell.neighbours['phi'][-1] is not None:

#             if not np.isnan(cell.neighbours['phi'][+1]['Jphi']) and not np.isnan(cell.neighbours['phi'][-1]['Jphi']):
#                 dIphi += cell.neighbours['phi'][+1]['Jphi'] * cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2
#             # else:
#             #     dIphi += cell.Iphi
#                 dIphi = dIphi - cell.neighbours['phi'][-1]['Jphi'] * cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2
#         # else:
#         #     dIphi = dIphi - cell.Iphi
#         cell.dI_I = abs(dIr + dIth + dIphi)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2)

# calculate_total_current_uncertainty(grid_hp)
# calculate_total_current_uncertainty(grid_lp)
# calculate_total_current_uncertainty(grid_all)


# %%
def calculate_total_current_uncertainty(grid):
    count = 0
    for cell in grid.cells:
        cell.dI_I = np.nan

        dIr = 0
        dIr_in = 0
        dIr_out = 0

        if (
            cell.neighbours["r"][+1] is not None
            and cell.neighbours["r"][-1] is not None
        ):
            if not np.isnan(cell.neighbours["r"][+1]["Jr"]) and not np.isnan(
                cell.neighbours["r"][-1]["Jr"]
            ):
                if cell.neighbours["r"][+1]["Jr"] <= 0:
                    dIr += (
                        cell.neighbours["r"][+1]["Jr"]
                        * (cell.neighbours["r"][+1]["r_c"]) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                    dIr_in += abs(
                        cell.neighbours["r"][+1]["Jr"]
                        * (cell.neighbours["r"][+1]["r_c"]) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                else:
                    dIr_out += abs(
                        cell.neighbours["r"][+1]["Jr"]
                        * (cell.neighbours["r"][+1]["r_c"]) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )

                if cell.neighbours["r"][-1]["Jr"] >= 0:
                    dIr = (
                        dIr
                        + cell.neighbours["r"][-1]["Jr"]
                        * (cell.neighbours["r"][-1]["r_c"]) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                    dIr_in += abs(
                        cell.neighbours["r"][-1]["Jr"]
                        * (cell.neighbours["r"][-1]["r_c"]) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                else:
                    dIr_out += abs(
                        cell.neighbours["r"][-1]["Jr"]
                        * (cell.neighbours["r"][-1]["r_c"]) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
            else:
                dIr_in = np.nan
                dIr_out = np.nan
                dIr = 0
        else:
            dIr_in = np.nan
            dIr_out = np.nan
            dIr = 0

        dIth = 0
        dIth_in = 0
        dIth_out = 0

        if (
            cell.neighbours["th"][+1] is not None
            and cell.neighbours["th"][-1] is not None
        ):
            if not np.isnan(cell.neighbours["th"][+1]["Jth"]) and not np.isnan(
                cell.neighbours["th"][-1]["Jth"]
            ):
                if cell.neighbours["th"][+1]["Jth"] <= 0:
                    dIth += (
                        cell.neighbours["th"][+1]["Jth"]
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][+1]["th_c"])
                        * cell.dr
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                    dIth_in += abs(
                        cell.neighbours["th"][+1]["Jth"]
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][+1]["th_c"])
                        * cell.dr
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                else:
                    dIth_out += abs(
                        cell.neighbours["th"][+1]["Jth"]
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][+1]["th_c"])
                        * cell.dr
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                if cell.neighbours["th"][-1]["Jth"] >= 0:
                    dIth = (
                        dIth
                        + cell.neighbours["th"][-1]["Jth"]
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][-1]["th_c"])
                        * cell.dr
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                    dIth_in += abs(
                        cell.neighbours["th"][-1]["Jth"]
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][-1]["th_c"])
                        * cell.dr
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )
                else:
                    dIth_out += abs(
                        cell.neighbours["th"][-1]["Jth"]
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][-1]["th_c"])
                        * cell.dr
                        * cell.dphi
                        * (mars_radius * 1e3) ** 2
                    )

            else:
                dIth_in = np.nan
                dIth_out = np.nan
                dIth = 0
        else:
            dIth_in = np.nan
            dIth_out = np.nan
            dIth = 0

        dIphi = 0
        dIphi_in = 0
        dIphi_out = 0

        if (
            cell.neighbours["phi"][+1] is not None
            and cell.neighbours["phi"][-1] is not None
        ):
            if not np.isnan(cell.neighbours["phi"][+1]["Jphi"]) and not np.isnan(
                cell.neighbours["phi"][-1]["Jphi"]
            ):
                if cell.neighbours["phi"][+1]["Jphi"] <= 0:
                    dIphi += (
                        cell.neighbours["phi"][+1]["Jphi"]
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (mars_radius * 1e3) ** 2
                    )
                    dIphi_in += abs(
                        cell.neighbours["phi"][+1]["Jphi"]
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (mars_radius * 1e3) ** 2
                    )
                else:
                    dIphi_out += abs(
                        cell.neighbours["phi"][+1]["Jphi"]
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (mars_radius * 1e3) ** 2
                    )
                if cell.neighbours["phi"][-1]["Jphi"] >= 0:
                    dIphi = (
                        dIphi
                        + cell.neighbours["phi"][-1]["Jphi"]
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (mars_radius * 1e3) ** 2
                    )
                    dIphi_in += abs(
                        cell.neighbours["phi"][-1]["Jphi"]
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (mars_radius * 1e3) ** 2
                    )
                else:
                    dIphi_out += abs(
                        cell.neighbours["phi"][-1]["Jphi"]
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (mars_radius * 1e3) ** 2
                    )
            else:
                dIphi_in = np.nan
                dIphi_out = np.nan
                dIphi = 0
        else:
            dIphi_in = np.nan
            dIphi_out = np.nan
            dIphi = 0

        # if dIr != 0 and dIth != 0 and dIphi != 0:
        # cell.dI_I = abs(dIr + dIth + dIphi)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2)
        dI_in = dIr_in + dIth_in + dIphi_in
        dI_out = dIr_out + dIth_out + dIphi_out
        cell.dI_I = abs(dI_in - dI_out) / np.sqrt(
            cell.Ir**2 + cell.Ith**2 + cell.Iphi**2
        )
        # else:
        # cell.dI_I = 0
    print(f"Total cells without neighbours for r: {count}")


calculate_total_current_uncertainty(grid_hp)
calculate_total_current_uncertainty(grid_lp)
calculate_total_current_uncertainty(grid_all)

# %%
# def calculate_total_current_uncertainty(grid):
#     for cell in grid.cells:

#         cell.Ir = cell.Jr*cell.r_c**2*np.sin(cell.th_c)*cell.dth*cell.dphi *(mars_radius* 1e3)**2 if not np.isnan(cell.Jr) else np.nan
#         cell.Ith = cell.Jth*cell.r_c*np.sin(cell.th_c)*cell.dr*cell.dphi *(mars_radius* 1e3)**2 if not np.isnan(cell.Jth) else np.nan
#         cell.Iphi = cell.Jphi*cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2 if not np.isnan(cell.Jphi) else np.nan

#         cell.dI_I = np.nan
#         # for coordinate in cell.neighbours.keys():
#         #     for direction in cell.neighbours[coordinate].keys():
#         #         if cell.neighbours[coordinate][direction] == None:
#         #             good = False
#         # dIri = 0
#         # dIro = 0
#         # # dIro = 0
#         # if cell.neighbours['r'][+1] is not None:
#         #     if cell.neighbours['r'][+1]['Jr'] <= 0:
#         #         if not np.isnan(cell.neighbours['r'][+1]['Jr']):
#         #             dIri += abs(cell.neighbours['r'][+1]['Jr'] * (cell.neighbours['r'][+1]['r_c'])**2 * np.sin(cell.th_c) * cell.dth * cell.dphi * (mars_radius* 1e3))

#         # if cell.neighbours['r'][-1] is not None:
#         #     if cell.neighbours['r'][-1]['Jr'] >= 0:
#         #         if not np.isnan(cell.neighbours['r'][-1]['Jr']):
#         #             dIri += abs(cell.neighbours['r'][-1]['Jr'] * (cell.neighbours['r'][-1]['r_c'])**2 * np.sin(cell.th_c) * cell.dth * cell.dphi * (mars_radius* 1e3))

#         dIr = 0

#         if cell.neighbours['r'][+1] is not None:
#             if not np.isnan(cell.neighbours['r'][+1]['Jr']):
#                 dIr += cell.neighbours['r'][+1]['Jr'] * (cell.neighbours['r'][+1]['r_c'])**2 * np.sin(cell.th_c) * cell.dth * cell.dphi * (mars_radius* 1e3)**2
#         else:
#             dIr += cell.Ir
#         if cell.neighbours['r'][-1] is not None:
#             if not np.isnan(cell.neighbours['r'][-1]['Jr']):
#                 dIr = dIr - cell.neighbours['r'][-1]['Jr'] * (cell.neighbours['r'][-1]['r_c'])**2 * np.sin(cell.th_c) * cell.dth * cell.dphi * (mars_radius* 1e3)**2
#         else:
#             dIr = dIr - cell.Ir

#         dIth = 0

#         if cell.neighbours['th'][+1] is not None:
#             if not np.isnan(cell.neighbours['th'][+1]['Jth']):
#                 dIth += cell.neighbours['th'][+1]['Jth'] * cell.r_c*np.sin(cell.neighbours['th'][+1]['th_c'])*cell.dr*cell.dphi *(mars_radius* 1e3)**2
#         else:
#             dIth += cell.Ith
#         if cell.neighbours['th'][-1] is not None:
#             if not np.isnan(cell.neighbours['th'][-1]['Jth']):
#                 dIth = dIth - cell.neighbours['th'][-1]['Jth'] * cell.r_c*np.sin(cell.neighbours['th'][-1]['th_c'])*cell.dr*cell.dphi *(mars_radius* 1e3)**2
#         else:
#             dIth = dIth - cell.Ith

#         dIphi = 0

#         if cell.neighbours['phi'][+1] is not None:
#             if not np.isnan(cell.neighbours['phi'][+1]['Jphi']):
#                 dIphi += cell.neighbours['phi'][+1]['Jphi'] * cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2
#         else:
#             dIphi += cell.Iphi
#         if cell.neighbours['phi'][-1] is not None:
#             if not np.isnan(cell.neighbours['phi'][-1]['Jphi']):
#                 dIphi = dIphi - cell.neighbours['phi'][-1]['Jphi'] * cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2
#         else:
#             dIphi = dIphi - cell.Iphi
#         # if cell.Jr >= 0:
#         #     if not np.isnan(cell.Jr):
#         #         dIro = abs(cell.Jr*(cell.r_c + cell.dr/2)**2*np.sin(cell.th_c)*cell.dth*cell.dphi *(mars_radius* 1e3)**2)
#         # elif cell.Jr <= 0:
#         #     if not np.isnan(cell.Jr):
#         #         dIro = abs(cell.Jr*(cell.r_c - cell.dr/2)**2*np.sin(cell.th_c)*cell.dth*cell.dphi *(mars_radius* 1e3)**2)

#         # dIr = abs(dIri - dIro)
#         # dIr = dIri - cell.Ir

#         # dIthi = 0
#         # dItho = 0
#         # # dItho = 0
#         # if cell.neighbours['th'][+1] is not None:
#         #     if cell.neighbours['th'][+1]['Jth'] <= 0:
#         #         if not np.isnan(cell.neighbours['th'][+1]['Jth']):
#         #             dIthi += abs(cell.neighbours['th'][+1]['Jth'] * cell.r_c*np.sin(cell.neighbours['th'][+1]['th_c'])*cell.dr*cell.dphi *(mars_radius* 1e3)**2)

#         # if cell.neighbours['th'][-1] is not None:
#         #     if cell.neighbours['th'][-1]['Jth'] >= 0:
#         #         if not np.isnan(cell.neighbours['th'][-1]['Jth']):
#         #             dIthi += abs(cell.neighbours['th'][-1]['Jth'] * cell.r_c*np.sin(cell.neighbours['th'][-1]['th_c'])*cell.dr*cell.dphi *(mars_radius* 1e3)**2)


#         # if cell.Jth >= 0:
#         #     if not np.isnan(cell.Jth):
#         #         dItho = abs(cell.Jth*cell.r_c*np.sin(cell.th_c+cell.dth)*cell.dr*cell.dphi *(mars_radius* 1e3)**2)
#         # elif cell.Jth <= 0:
#         #     if not np.isnan(cell.Jth):
#         #         dItho = abs(cell.Jth*cell.r_c*np.sin(cell.th_c-cell.dth)*cell.dr*cell.dphi *(mars_radius* 1e3)**2)

#         # dIth = abs(dIthi - dItho)
#         # dIth = dIthi - cell.Ith

#         # dIphii = 0
#         # dIphio = 0
#         # # dIphio = 0
#         # if cell.neighbours['phi'][+1] is not None:
#         #     if cell.neighbours['phi'][+1]['Jphi'] <= 0:
#         #         if not np.isnan(cell.neighbours['phi'][+1]['Jphi']):
#         #             dIphii += abs(cell.neighbours['phi'][+1]['Jphi'] * cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2)

#         # if cell.neighbours['phi'][-1] is not None:
#         #     if cell.neighbours['phi'][-1]['Jphi'] >= 0:
#         #         if not np.isnan(cell.neighbours['phi'][-1]['Jphi']):
#         #             dIphii += abs(cell.neighbours['phi'][-1]['Jphi'] * cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2)


#         # if cell.Jphi >= 0:
#         #     if not np.isnan(cell.Jphi):
#         #         dIphio = abs(cell.Jphi*cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2)
#         # elif cell.Jphi <= 0:
#         #     if not np.isnan(cell.Jphi):
#         #         dIphio = abs(cell.Jphi*cell.r_c*cell.dr*cell.dth *(mars_radius* 1e3)**2)

#         # dIphi = abs(dIphii - dIphio)

#         # dIphi =  dIphii - cell.Iphi

#         # print(dIphi)
#         # ##cell.dI_I = ((abs(dIr + dIth + dIphi)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2)))
#         # cell.dI_I =abs(1- np.sqrt((dIri)**2 + (dIthi)**2 + (dIphii)**2)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2))
#         # cell.dI_I =abs(dIr + dIth + dIphi)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2)
#         # cell.dI_I = ((np.sqrt(dIr**2 + dIth**2 + dIphi**2)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2)))
#         cell.dI_I = abs(dIr + dIth + dIphi)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2)
#         # cell.dI_I = abs(dIr + dIth + dIphi)/(np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2))
#         # cell.dI_I =  np.sqrt(1-2*(cell.Ir*dIri + cell.Ith*dIthi + cell.Iphi*dIphii)/np.sqrt(cell.Ir**2+cell.Ith**2+cell.Iphi**2))
# calculate_total_current_uncertainty(grid_hp)
# calculate_total_current_uncertainty(grid_lp)
# calculate_total_current_uncertainty(grid_all)


# %%
def total_current_uncertainty_arr(grid):

    dI_I = np.empty((len(grid.phi), len(grid.th), len(grid.r)))
    dI_I[:] = np.nan

    for i in range(dI_I.shape[0]):
        for j in range(dI_I.shape[1]):
            for k in range(dI_I.shape[2]):
                matching_cell = []
                for cell in grid.cells:
                    if (
                        cell.r_bin == k + 1
                        and cell.th_bin == j + 1
                        and cell.phi_bin == i + 1
                    ):
                        matching_cell.append(cell)
                        break

                if not bool(matching_cell):
                    dI_I[i, j, k] = np.nan

                elif len(matching_cell) == 1:
                    dI_I[i, j, k] = matching_cell[0].dI_I

                else:
                    print("Grid error")

    grid.dI_I = dI_I


total_current_uncertainty_arr(grid_hp)
total_current_uncertainty_arr(grid_lp)
total_current_uncertainty_arr(grid_all)

# %%
# def get_total_currents_sides(grid, th_ranges = [[0, 90], [90, 180]]):

#     th_ranges = np.deg2rad(th_ranges)
#     dayside = np.deg2rad([-90, 90])
#     # nightside = np.deg2rad([[-180, -90], [90, 180]])

#     Ir_sides = np.zeros(shape=(len(th_ranges), 2, len(grid.r)))
#     Ith_sides = np.zeros(shape=(len(th_ranges), 2, len(grid.r)))
#     Iphi_sides = np.zeros(shape=(len(th_ranges), 2, len(grid.r)))
#     Itot_sides = np.zeros(shape=(len(th_ranges), 2, len(grid.r)))
#     dIt_sides = np.zeros(shape=(len(th_ranges), 2, len(grid.r)))
#     counts = np.zeros(shape=(len(th_ranges), 2, len(grid.r)))

#     for cell in grid.cells:

#         ir = np.digitize(cell.r_c, grid.r_bins) - 1#np.where(cell.r_c == r)[0][0]

#         for ith, th_range in enumerate(th_ranges):

#             if cell.th_c >= th_range[0] and cell.th_c < th_range[1]:

#                 if cell.phi_c >= dayside[0] and cell.phi_c < dayside[1]:

#                     Ir_sides[ith, 0, ir] += cell.Ir if not np.isnan(cell.Ir) else 0
#                     Ith_sides[ith, 0, ir] += cell.Ith if not np.isnan(cell.Ith) else 0
#                     Iphi_sides[ith, 0, ir] += cell.Iphi if not np.isnan(cell.Iphi) else 0

#                     if Ir_sides[ith, 0, ir] != 0 and Ith_sides[ith, 0, ir] != 0 and Iphi_sides[ith, 0, ir] != 0:
#                         Itot_sides[ith, 0, ir] = np.sqrt(Ir_sides[ith, 0, ir]**2 + Ith_sides[ith, 0, ir]**2 + Iphi_sides[ith, 0, ir]**2)
#                         dIt_sides[ith, 0, ir] += (cell.dI_I)**2
#                         counts[ith, 0, ir] += 1
#                 else:

#                     Ir_sides[ith, 1, ir] += cell.Ir if not np.isnan(cell.Ir) else 0
#                     Ith_sides[ith, 1, ir] += cell.Ith if not np.isnan(cell.Ith) else 0
#                     Iphi_sides[ith, 1, ir] += cell.Iphi if not np.isnan(cell.Iphi) else 0

#                     if Ir_sides[ith, 1, ir] != 0 and Ith_sides[ith, 1, ir] != 0 and Iphi_sides[ith, 1, ir] != 0:
#                         Itot_sides[ith, 1, ir] = np.sqrt(Ir_sides[ith, 1, ir]**2 + Ith_sides[ith, 1, ir]**2 + Iphi_sides[ith, 1, ir]**2)
#                         dIt_sides[ith, 1, ir] += (cell.dI_I)**2
#                         counts[ith, 1, ir] += 1

#     dIt_sides = np.sqrt(dIt_sides)
#     Ir_sides= Ir_sides*1e-12; Ith_sides= Ith_sides*1e-12; Iphi_sides= Iphi_sides*1e-12; Itot_sides= Itot_sides*1e-12

#     return Ir_sides, Ith_sides, Iphi_sides, Itot_sides, dIt_sides


# %%
def get_total_currents_sides(grid, th_ranges=np.deg2rad([[0, 90], [90, 180]])):
    r = grid.r
    dayside = np.deg2rad([-90, 90])

    Ir_sides = np.zeros(shape=(len(th_ranges), 2, len(r)))
    Ith_sides = np.zeros(shape=(len(th_ranges), 2, len(r)))
    Iphi_sides = np.zeros(shape=(len(th_ranges), 2, len(r)))
    dIt = np.zeros(shape=(len(th_ranges), 2, len(r)))
    Ir_counts = np.zeros(shape=(len(th_ranges), 2, len(r)))
    Ith_counts = np.zeros(shape=(len(th_ranges), 2, len(r)))
    Iphi_counts = np.zeros(shape=(len(th_ranges), 2, len(r)))
    dIt_counts = np.zeros(shape=(len(th_ranges), 2, len(r)))

    for cell in grid.cells:
        ir = np.digitize(cell.r_c, grid.r_bins) - 1

        for ith, th_range in enumerate(th_ranges):
            if cell.th_c >= th_range[0] and cell.th_c < th_range[1]:
                if cell.phi_c >= dayside[0] and cell.phi_c < dayside[1]:
                    if (
                        not np.isnan(cell.Ir)
                        and not np.isnan(cell.Ith)
                        and not np.isnan(cell.Iphi)
                    ):
                        Ir_sides[ith, 0, ir] += cell.Ir if not np.isnan(cell.Ir) else 0
                        Ir_counts[ith, 0, ir] += 1 if not np.isnan(cell.Ir) else 0

                        Ith_sides[ith, 0, ir] += (
                            cell.Ith if not np.isnan(cell.Ith) else 0
                        )
                        Ith_counts[ith, 0, ir] += 1 if not np.isnan(cell.Ith) else 0

                        Iphi_sides[ith, 0, ir] += (
                            cell.Iphi if not np.isnan(cell.Iphi) else 0
                        )
                        Iphi_counts[ith, 0, ir] += 1 if not np.isnan(cell.Iphi) else 0

                        dIt[ith, 0, ir] += cell.dI_I if not np.isnan(cell.dI_I) else 0
                        dIt_counts[ith, 0, ir] += 1 if not np.isnan(cell.dI_I) else 0

                else:
                    if (
                        not np.isnan(cell.Ir)
                        and not np.isnan(cell.Ith)
                        and not np.isnan(cell.Iphi)
                    ):
                        Ir_sides[ith, 1, ir] += cell.Ir if not np.isnan(cell.Ir) else 0
                        Ir_counts[ith, 1, ir] += 1 if not np.isnan(cell.Ir) else 0

                        Ith_sides[ith, 1, ir] += (
                            cell.Ith if not np.isnan(cell.Ith) else 0
                        )
                        Ith_counts[ith, 1, ir] += 1 if not np.isnan(cell.Ith) else 0

                        Iphi_sides[ith, 1, ir] += (
                            cell.Iphi if not np.isnan(cell.Iphi) else 0
                        )
                        Iphi_counts[ith, 1, ir] += 1 if not np.isnan(cell.Iphi) else 0

                        dIt[ith, 1, ir] += cell.dI_I if not np.isnan(cell.dI_I) else 0
                        dIt_counts[ith, 1, ir] += 1 if not np.isnan(cell.dI_I) else 0

    # dIt = np.sqrt(dIt)
    dIt = dIt / dIt_counts
    Ir_sides = Ir_sides * 1e-12 / Ir_counts
    Ith_sides = Ith_sides * 1e-12 / Ith_counts
    Iphi_sides = Iphi_sides * 1e-12 / Iphi_counts
    Itot_sides = np.sqrt(Ir_sides**2 + Ith_sides**2 + Iphi_sides**2)
    return Ir_sides, Ith_sides, Iphi_sides, Itot_sides, dIt


# %%
th_ranges = np.deg2rad([[0, 90], [90, 180]])
Ir_sides_hp, Ith_sides_hp, Iphi_sides_hp, Itot_sides_hp, dIt_sides_hp = (
    get_total_currents_sides(grid_hp)
)
Ir_sides_lp, Ith_sides_lp, Iphi_sides_lp, Itot_sides_lp, dIt_sides_lp = (
    get_total_currents_sides(grid_lp)
)
Ir_sides_all, Ith_sides_all, Iphi_sides_all, Itot_sides_all, dIt_sides_all = (
    get_total_currents_sides(grid_all)
)

# %%
from matplotlib.lines import Line2D

plt.ioff()

fig, ax = plt.subplots(4, figsize=(10, 7), layout="tight", sharex="all")
fig.subplots_adjust(hspace=0.0, wspace=0.0)
u = ["r", "\\theta", "\\phi"]
# labels_phi_ranges = [rf'${np.rad2deg(phi_range[0])} \degree \leq \phi \leq {np.rad2deg(phi_range[1])} \degree$' for phi_range in phi_ranges]
labels_th_ranges = [
    rf"${np.around(np.rad2deg(th_range[0]))} \degree \leq \theta \leq {np.around(np.rad2deg(th_range[1]))} \degree$"
    for th_range in th_ranges
]
labels_sides = ["dayside", "nightside"]
linestyles = ["-", "--"]
colors = ["k", "b", "r", "g", "m", "y"]
custom_lines = []
# custom_labels = []
for ith in range(len(th_ranges)):
    # custom_labels.append([labels_th_ranges[ith]])
    custom_lines.extend(
        [Line2D([0], [0], color=colors[ith], lw=4, label=labels_th_ranges[ith])]
    )

custom_lines_pdyn = [
    Line2D([0], [0], color="k", linestyle=linestyles[0], label="low $P_{dyn}$"),
    Line2D([0], [0], color="k", linestyle=linestyles[1], label="high $P_{dyn}$"),
]
# [[ax[0].plot(r, Ir[ith, iphi, :], label = labels_th_ranges[ith] + labels_phi_ranges[iphi]) for ith in range(len(th_ranges))] for iphi in range(len(phi_ranges))]
for iside in range(2):
    r_plot = -grid_lp.r if iside == 1 else grid_lp.r
    [
        ax[0].plot(
            r_plot,
            Ir_sides_lp[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[0],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    [
        ax[1].plot(
            r_plot,
            Ith_sides_lp[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[0],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    [
        ax[2].plot(
            r_plot,
            Iphi_sides_lp[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[0],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    # [ax[3].plot(r_plot, Itot_sides_lp[ith, iside, :], label = labels_th_ranges[ith] , linestyle = linestyles[0], color = colors[ith]) for ith in range(len(th_ranges))]
    [
        ax[3].errorbar(
            r_plot,
            Itot_sides_lp[ith, iside, :],
            yerr=dIt_sides_lp[ith, iside, :] * Itot_sides_lp[ith, iside, :],
            color=colors[ith],
            linestyle=linestyles[0],
            capsize=1.5,
        )
        for ith in range(len(th_ranges))
    ]

for iside in range(2):
    r_plot = -grid_lp.r if iside == 1 else grid_lp.r
    [
        ax[0].plot(
            r_plot,
            Ir_sides_hp[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[1],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    [
        ax[1].plot(
            r_plot,
            Ith_sides_hp[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[1],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    [
        ax[2].plot(
            r_plot,
            Iphi_sides_hp[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[1],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    # [ax[3].plot(r_plot, Itot_sides_hp[ith, iside, :], label = labels_th_ranges[ith] , linestyle = linestyles[1], color = colors[ith]) for ith in range(len(th_ranges))]
    [
        ax[3].errorbar(
            r_plot,
            Itot_sides_hp[ith, iside, :],
            yerr=dIt_sides_hp[ith, iside, :] * Itot_sides_hp[ith, iside, :],
            color=colors[ith],
            linestyle=linestyles[1],
            capsize=1.5,
        )
        for ith in range(len(th_ranges))
    ]

ticks = np.around(np.arange(-3, 3.5, 0.5), 1)
ticks_labels = abs(ticks)
[
    axx.set_xticks(
        ticks=ticks,
        labels=ticks_labels,
    )
    for axx in ax
]
[ax[i].set_ylabel(rf"$ I_{u[i]} \, [kA]$", fontsize=14) for i in range(3)]
[ax[3].set_ylabel(r"$ I_{tot} \, [kA]$", fontsize=14)]
ax[-1].set_xlabel(r"$r \, [R_{\rm M}]$", fontsize=14)
ax[0].legend(handles=custom_lines_pdyn, fontsize=14, loc="center")
ax[2].legend(handles=custom_lines, fontsize=14, loc="center")
[axx.grid() for axx in ax]
ax[0].text(
    x=0.1,
    y=1.1,
    transform=ax[0].transAxes,
    s=r"$nightside$",
    color="black",
    fontsize=16,
)
ax[0].text(
    x=0.8, y=1.1, transform=ax[0].transAxes, s=r"$dayside$", color="black", fontsize=16
)
# [axx.set_ylim([-140,140]) for axx in ax]
# ax[0].set_ylim([-280, 280])
# ax[1].set_ylim([-150, 150])
# ax[2].set_ylim([-150, 150])
[axx.set_xlim([-3, 3]) for axx in ax]

# fig.suptitle(r'all data', fontsize = 16);
# fig.suptitle(r'$ P_{dyn} \leq 0.72 \, nPa$', fontsize = 16);

# fig.savefig(f'{plots_path}/total_currents_theta_ranges_{fig._suptitle.get_text()}.png', bbox_inches = 'tight')
# fig.savefig(f'{plots_path}/total_currents_north_south.png', bbox_inches = 'tight')

# fig.savefig(f'../pres/total_currents_dayside_nightside_intermediate_P_dave.png', bbox_inches = 'tight')
# fig.savefig(f"plots/total_currents_dayside_nightside_low_high_Pdyn.png", bbox_inches = 'tight', dpi = 500)
plt.show()

# %%
from matplotlib.lines import Line2D

plt.ioff()

fig, ax = plt.subplots(4, figsize=(10, 7), layout="tight", sharex="all")
fig.subplots_adjust(hspace=0.0, wspace=0.0)
u = ["r", "\\theta", "\\phi"]
# labels_phi_ranges = [rf'${np.rad2deg(phi_range[0])} \degree \leq \phi \leq {np.rad2deg(phi_range[1])} \degree$' for phi_range in phi_ranges]
labels_th_ranges = [
    rf"${np.around(np.rad2deg(th_range[0]))} \degree \leq \theta \leq {np.around(np.rad2deg(th_range[1]))} \degree$"
    for th_range in th_ranges
]
labels_sides = ["dayside", "nightside"]
linestyles = ["-", "--"]
colors = ["k", "b", "r", "g", "m", "y"]
custom_lines = []
# custom_labels = []
for ith in range(len(th_ranges)):
    # custom_labels.append([labels_th_ranges[ith]])
    custom_lines.extend(
        [Line2D([0], [0], color=colors[ith], lw=4, label=labels_th_ranges[ith])]
    )

# custom_lines_pdyn = [Line2D([0], [0], color = 'k', linestyle=linestyles[0],  label = 'low $P_{dyn}$'),
#                      Line2D([0], [0], color = 'k', linestyle=linestyles[1],  label = 'high $P_{dyn}$')]
# [[ax[0].plot(r, Ir[ith, iphi, :], label = labels_th_ranges[ith] + labels_phi_ranges[iphi]) for ith in range(len(th_ranges))] for iphi in range(len(phi_ranges))]
for iside in range(2):
    r_plot = -grid_all.r if iside == 1 else grid_all.r
    [
        ax[0].plot(
            r_plot,
            Ir_sides_all[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[0],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    [
        ax[1].plot(
            r_plot,
            Ith_sides_all[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[0],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    [
        ax[2].plot(
            r_plot,
            Iphi_sides_all[ith, iside, :],
            label=labels_th_ranges[ith],
            linestyle=linestyles[0],
            color=colors[ith],
        )
        for ith in range(len(th_ranges))
    ]
    # [ax[3].plot(r_plot, Itot_sides_lp[ith, iside, :], label = labels_th_ranges[ith] , linestyle = linestyles[0], color = colors[ith]) for ith in range(len(th_ranges))]
    [
        ax[3].errorbar(
            r_plot,
            Itot_sides_all[ith, iside, :],
            yerr=dIt_sides_all[ith, iside, :] * Itot_sides_all[ith, iside, :],
            color=colors[ith],
            linestyle=linestyles[0],
            capsize=3,
        )
        for ith in range(len(th_ranges))
    ]
ticks = np.around(np.arange(-3, 3.5, 0.5), 1)
ticks_labels = abs(ticks)
[axx.set_xticks(ticks=ticks, labels=ticks_labels) for axx in ax]
[ax[i].set_ylabel(rf"$ I_{u[i]} \, [kA]$", fontsize=14) for i in range(3)]
[ax[3].set_ylabel(r"$ I_{tot} \, [kA]$", fontsize=14)]

ax[-1].set_xlabel(r"$r \, [R_{\rm M}]$", fontsize=14)
# ax[0].legend(handles = custom_lines_pdyn, fontsize = 10, loc='center')
ax[1].legend(handles=custom_lines, fontsize=14, loc="center")
[axx.grid() for axx in ax]
ax[0].text(
    x=0.1,
    y=1.1,
    transform=ax[0].transAxes,
    s=r"$nightside$",
    color="black",
    fontsize=16,
)
ax[0].text(
    x=0.8, y=1.1, transform=ax[0].transAxes, s=r"$dayside$", color="black", fontsize=16
)
# [axx.set_ylim([-140,140]) for axx in ax]
# ax[0].set_ylim([-300, 300])
# ax[1].set_ylim([-100, 200])
# ax[2].set_ylim([-80, 120])
[axx.set_xlim([-3, 3]) for axx in ax]

# fig.suptitle(r'all data', fontsize = 16);
# fig.suptitle(r'$ P_{dyn} \leq 0.72 \, nPa$', fontsize = 16);

# fig.savefig(f'{plots_path}/total_currents_theta_ranges_{fig._suptitle.get_text()}.png', bbox_inches = 'tight')
# fig.savefig(f'{plots_path}/total_currents_north_south.png', bbox_inches = 'tight')

# fig.savefig(f'../pres/total_currents_dayside_nightside_intermediate_P_dave.png', bbox_inches = 'tight')
# fig.savefig(f"plots/total_currents_dayside_nightside_all.png", bbox_inches = 'tight', dpi = 500)
plt.show()
