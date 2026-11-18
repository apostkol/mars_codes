#!/usr/bin/env python3
"""Visualize analytic current-density sanity checks on Cartesian plane cuts.

The left column uses a uniform Cartesian magnetic field, whose exact current is
zero. The right column uses ``B = 0.5 C x r``, whose exact current is uniform
and points along +MSE Z. Rows show the X-Z and X-Y planes respectively.

Run with, for example::

    uv run python scripts/visualize_analytic_currents.py
    uv run python scripts/visualize_analytic_currents.py --output currents.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from mars_currents.generic import cart_to_sph_vf
from mars_currents.grid import Grid
from mars_currents.maps import cartesian_plane_map


def make_grid(dtheta_deg, dphi_deg):
    return Grid(
        dr=0.1,
        dth_deg=dtheta_deg,
        dphi_deg=dphi_deg,
        r_lim=(0.8, 2.0),
        th_lim_deg=(0.0, 180.0),
        phi_lim_deg=(-180.0, 180.0),
    )


def calculate_currents(field_function, dtheta_deg, dphi_deg):
    grid = make_grid(dtheta_deg, dphi_deg)
    phi, theta, radius = np.meshgrid(grid.phi, grid.th, grid.r, indexing="ij")
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    bx, by, bz = field_function(x, y, z)
    cartesian = np.stack(np.broadcast_arrays(bx, by, bz), axis=-1)
    grid.Br, grid.Bth, grid.Bphi = cart_to_sph_vf(cartesian, theta, phi)
    for name in ("sigma_Br", "sigma_Bth", "sigma_Bphi"):
        setattr(grid, name, np.zeros(grid.shape))
    grid.sync_cell_fields_from_grid(
        ("Br", "Bth", "Bphi", "sigma_Br", "sigma_Bth", "sigma_Bphi")
    )
    grid.current_densities(smoothed=False)
    return grid


def uniform_field(x, y, z):
    del x, y, z
    return 1.0, 0.3, -0.2


def constant_curl_field(x, y, z):
    del z
    return -y, x, 0.0 * x


def plot(args):
    null_grid = calculate_currents(uniform_field, args.dtheta, args.dphi)
    curl_grid = calculate_currents(constant_curl_field, args.dtheta, args.dphi)

    scale = max(
        np.nanpercentile(np.abs(null_grid.Jz), 99),
        np.nanpercentile(np.abs(curl_grid.Jz), 99),
    )
    linear_threshold = max(scale * 1e-3, np.finfo(float).eps)
    norm = SymLogNorm(linthresh=linear_threshold, vmin=-scale, vmax=scale)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    planes = (
        ((0.0, 1.0, 0.0), "X-Z cut"),
        ((0.0, 0.0, 1.0), "X-Y cut"),
    )
    cases = (
        (null_grid, "Uniform B: exact J = 0"),
        (curl_grid, "B = (-y, x, 0): exact J along +Z"),
    )

    for row, (normal, plane_title) in enumerate(planes):
        for column, (grid, case_title) in enumerate(cases):
            cartesian_plane_map(
                grid,
                "Jz",
                plane_normal=normal,
                extent=2.05,
                n=args.samples,
                arrows="J",
                arrow_step=max(1, args.samples // 20),
                cmap="RdBu_r",
                norm=norm,
                ax=axes[row, column],
                plot_boundaries=False,
                pole_theta_bins=args.pole_theta_bins,
                colorbar=False,
                title=f"{case_title}\n{plane_title}",
                scale_units="xy",
                angles="xy",
            )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    colorbar = fig.colorbar(scalar_mappable, ax=axes, shrink=0.88)
    colorbar.set_label(r"Computed $J_z$ [A m$^{-2}$]")
    fig.suptitle(
        rf"Analytic current check: $\Delta\theta={args.dtheta:g}^\circ$, "
        rf"$\Delta\phi={args.dphi:g}^\circ$, pole mask={args.pole_theta_bins} ring(s)"
    )
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtheta", type=float, default=10.0)
    parser.add_argument("--dphi", type=float, default=15.0)
    parser.add_argument("--samples", type=int, default=220)
    parser.add_argument("--pole-theta-bins", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    figure = plot(arguments)
    if arguments.output is None:
        plt.show()
    else:
        figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
        print(f"Saved {arguments.output}")
