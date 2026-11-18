# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: mars-currents
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visual checks of `Grid` operations with analytic fields
#
# Each test geometry below is independent and produces its own figure. Every
# figure uses the same map functions as the scientific analysis: one Mollweide
# map and two Cartesian plane projections. The lower row shows numerical error
# by latitude, longitude, and radius.
#
# To add a geometry, define its Cartesian field and diagnostic functions, then
# append one `AnalyticCase` to `ANALYTIC_CASES`. The generic sampling and
# plotting code does not need to change.
#
# These figures are visual diagnostics, not image-regression tests. The scalar
# regression tests for the current-density operator live in
# `tests/test_analytic_current_density.py`.

# %%
DEBUG = globals().get("DEBUG", False)

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from mars_currents.generic import cart_to_sph_vf
from mars_currents.grid import Grid
from mars_currents.maps import cartesian_plane_map, single_mollweide_map


MU0 = 1.25663706212e-6
ArrayFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple]
DiagnosticFunction = Callable[[Grid], np.ndarray]
ExpectedFunction = Callable[[Grid], float]


@dataclass(frozen=True)
class AnalyticCase:
    """One known field geometry and the scalar result used to check it."""

    name: str
    description: str
    field_function: ArrayFunction
    diagnostic_function: DiagnosticFunction
    expected_function: ExpectedFunction
    diagnostic_label: str
    color_limits: tuple[float, float]
    cmap: str = "RdBu_r"


# %% [markdown]
# ## Grid construction and analytic-field sampling


# %%
def make_grid(dtheta_deg=10.0, dphi_deg=15.0):
    """Return the small, data-free grid used by every analytic case."""
    return Grid(
        dr=0.2,
        dth_deg=dtheta_deg,
        dphi_deg=dphi_deg,
        r_lim=(1.0, 1.8),
        th_lim_deg=(0.0, 180.0),
        phi_lim_deg=(-180.0, 180.0),
        name="analytic_visual_check",
    )


def evaluate_cartesian_field(grid, field_function):
    """Sample a Cartesian field and run the production current calculation."""
    phi, theta, radius = np.meshgrid(
        grid.phi,
        grid.th,
        grid.r,
        indexing="ij",
    )
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


def interior_rms(values, excluded_theta_rings=2):
    """Return RMS away from the spherical-coordinate polar singularities."""
    interior = values[:, excluded_theta_rings:-excluded_theta_rings, :]
    return float(np.sqrt(np.mean(interior**2)))


# %% [markdown]
# ## Known field geometries


# %%
def uniform_field(x, y, z):
    """A constant Cartesian field, for which curl(B) and J are zero."""
    del x, y, z
    return 1.0, 0.3, -0.2


def uniform_curl_diagnostic(grid):
    """Dimensionless curl magnitude inferred from the calculated current."""
    return grid.J_tot * MU0 * grid.radius_km * 1e3


def zero_expected(grid):
    del grid
    return 0.0


CURL_AMPLITUDE = 2.0


def constant_z_curl_field(x, y, z):
    """B = 0.5 C z-hat x r, for which curl(B) = C z-hat."""
    del z
    return (
        -0.5 * CURL_AMPLITUDE * y,
        0.5 * CURL_AMPLITUDE * x,
        0.0 * x,
    )


def expected_constant_jz(grid):
    """Analytic current for `constant_z_curl_field`."""
    return CURL_AMPLITUDE / (MU0 * grid.radius_km * 1e3)


def normalized_jz_diagnostic(grid):
    """Calculated Jz divided by its analytic value."""
    return grid.Jz / expected_constant_jz(grid)


# Adding a test geometry only requires one more entry here (plus its field and
# diagnostic functions above).
ANALYTIC_CASES = [
    AnalyticCase(
        name="Uniform Cartesian magnetic field",
        description=r"Expected: $\nabla\times B=0$ and $J=0$",
        field_function=uniform_field,
        diagnostic_function=uniform_curl_diagnostic,
        expected_function=zero_expected,
        diagnostic_label=r"dimensionless $|\nabla\times B|$ (expected 0)",
        color_limits=(0.0, 0.05),
        cmap="magma",
    ),
    AnalyticCase(
        name="Solid-body field with constant z-directed curl",
        description=r"$B=\frac{1}{2}C\hat{z}\times r$; expected: $J_z/J_{z,true}=1$",
        field_function=constant_z_curl_field,
        diagnostic_function=normalized_jz_diagnostic,
        expected_function=lambda grid: 1.0,
        diagnostic_label=r"calculated $J_z$ / expected $J_z$",
        color_limits=(0.98, 1.02),
    ),
]


# %% [markdown]
# ## Standard figure for one geometry
#
# The upper row contains the production projections:
#
# - Mollweide map at the middle radial shell,
# - MSE XY plane,
# - MSE XZ plane.
#
# Magnetic-field arrows are overlaid so that the artificial input geometry is
# visible along with the calculated scalar diagnostic. The lower row reduces
# the error along each grid coordinate, making polar, periodic-seam, and radial
# boundary behavior easy to distinguish.


# %%
def plot_case(case, grid, plane_samples=180):
    """Create one self-contained projection and error figure for a case."""
    diagnostic = np.asarray(case.diagnostic_function(grid))
    if diagnostic.shape != grid.shape:
        raise ValueError(
            f"{case.name!r} diagnostic has shape {diagnostic.shape}; "
            f"expected {grid.shape}."
        )

    expected = float(case.expected_function(grid))
    error = diagnostic - expected
    rms = interior_rms(error)
    radial_index = len(grid.r) // 2
    norm = plt.Normalize(*case.color_limits)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    ax_mollweide = fig.add_subplot(2, 3, 1, projection="mollweide")
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xz = fig.add_subplot(2, 3, 3)
    ax_latitude = fig.add_subplot(2, 3, 4)
    ax_longitude = fig.add_subplot(2, 3, 5)
    ax_radius = fig.add_subplot(2, 3, 6)

    single_mollweide_map(
        grid,
        diagnostic,
        arrows="B",
        ir=radial_index,
        ax=ax_mollweide,
        fig=fig,
        title="Mollweide projection",
        cb_label=case.diagnostic_label,
        cmap=case.cmap,
        norm=norm,
        note_medians=None,
        quiver_kwargs={"scale": 35},
    )

    plane_kwargs = dict(
        parameter=diagnostic,
        arrows="B",
        cmap=case.cmap,
        norm=norm,
        n=plane_samples,
        plot_boundaries=False,
        pole_theta_bins=2,
        arrow_step=max(1, plane_samples // 22),
        colorbar=True,
        cb_label=case.diagnostic_label,
        show_xyz_triad=True,
    )
    cartesian_plane_map(
        grid,
        plane_normal=(0, 0, 1),
        u_axis=(1, 0, 0),
        ax=ax_xy,
        title="MSE XY plane",
        **plane_kwargs,
    )
    cartesian_plane_map(
        grid,
        plane_normal=(0, 1, 0),
        u_axis=(1, 0, 0),
        ax=ax_xz,
        title="MSE XZ plane",
        **plane_kwargs,
    )

    latitude_deg = 90.0 - np.rad2deg(grid.th)
    latitude_rms = np.sqrt(np.mean(error**2, axis=(0, 2)))
    ax_latitude.semilogy(latitude_deg, latitude_rms, "o-")
    ax_latitude.set(
        title="Error by latitude",
        xlabel="MSE latitude [deg]",
        ylabel="RMS error",
    )

    longitude_deg = np.rad2deg(grid.phi)
    longitude_rms = np.sqrt(np.mean(error**2, axis=(1, 2)))
    ax_longitude.semilogy(longitude_deg, longitude_rms, "o-")
    ax_longitude.set(
        title="Periodic-seam check",
        xlabel="MSE longitude [deg]",
        ylabel="RMS error",
    )

    radial_rms = np.sqrt(np.mean(error**2, axis=(0, 1)))
    ax_radius.semilogy(grid.r, radial_rms, "o-")
    ax_radius.set(
        title="Radial-boundary check",
        xlabel=r"radius [$R_M$]",
        ylabel="RMS error",
    )

    for axis in (ax_latitude, ax_longitude, ax_radius):
        axis.grid(which="both", alpha=0.25)

    fig.suptitle(
        f"{case.name}\n{case.description}; interior RMS error = {rms:.3e}",
        fontsize=15,
    )
    return fig, grid, diagnostic, error


# %% [markdown]
# ## Run every registered geometry

# %%
dtheta = 10.0 if DEBUG else 5.0
plane_samples = 70 if DEBUG else 180

# Results remain available by case name when this file is opened as a notebook.
case_results = {}
for analytic_case in ANALYTIC_CASES:
    analytic_grid = evaluate_cartesian_field(
        make_grid(dtheta_deg=dtheta),
        analytic_case.field_function,
    )
    case_figure, analytic_grid, diagnostic, error = plot_case(
        analytic_case,
        analytic_grid,
        plane_samples=plane_samples,
    )
    case_results[analytic_case.name] = {
        "figure": case_figure,
        "grid": analytic_grid,
        "diagnostic": diagnostic,
        "error": error,
    }
    print(f"{analytic_case.name}: interior RMS error = {interior_rms(error):.3e}")

    if DEBUG:
        plt.close(case_figure)
    else:
        plt.show()
