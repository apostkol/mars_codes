"""Analytic sanity checks for the spherical-grid current calculation.

These tests exercise the complete path used for science products: construct a
Cartesian magnetic field at grid-cell centres, rotate it into the local
spherical basis, copy it to Cells, and evaluate ``curl(B) / mu0``.
"""

from __future__ import annotations

import numpy as np

from mars_currents.generic import cart_to_sph_vf
from mars_currents.grid import Grid


MU0 = 1.25663706212e-6


def _grid(dtheta_deg=10.0, dphi_deg=15.0):
    return Grid(
        dr=0.2,
        dth_deg=dtheta_deg,
        dphi_deg=dphi_deg,
        r_lim=(1.0, 1.8),
        th_lim_deg=(0.0, 180.0),
        phi_lim_deg=(-180.0, 180.0),
    )


def _evaluate_cartesian_field(grid, field_function):
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


def _constant_field(x, y, z):
    del x, y, z
    return 1.0, 0.3, -0.2


def _constant_z_curl_field(curl_amplitude=1.0):
    """Return B = 0.5 C x r, for which curl(B) = C z-hat exactly."""

    def field(x, y, z):
        del z
        return -0.5 * curl_amplitude * y, 0.5 * curl_amplitude * x, 0.0 * x

    return field


def _interior_theta_slice(grid, excluded_rings=2):
    del grid
    return np.s_[:, excluded_rings:-excluded_rings, :]


def test_uniform_cartesian_field_has_small_interior_spurious_current():
    """A uniform Cartesian B is curl-free despite varying spherical components."""
    grid = _evaluate_cartesian_field(_grid(), _constant_field)
    interior = _interior_theta_slice(grid, excluded_rings=2)

    # Normalize out the physical conversion applied by current_densities().
    dimensionless_curl_error = grid.J_tot[interior] * MU0 * grid.radius_km * 1e3

    assert np.sqrt(np.mean(dimensionless_curl_error**2)) < 0.02


def test_uniform_field_exposes_polar_error_amplification():
    """The first visible rings remain contaminated by the 1/sin(theta) terms."""
    grid = _evaluate_cartesian_field(_grid(), _constant_field)
    jr_ring_rms = np.sqrt(np.mean(grid.Jr**2, axis=(0, 2)))

    polar_rms = np.mean(jr_ring_rms[[0, 1, -2, -1]])
    equatorial_rms = np.mean(jr_ring_rms[len(grid.th) // 2 - 1 : len(grid.th) // 2 + 1])

    # This intentionally records the known numerical vulnerability. If the
    # operator is replaced by a pole-safe implementation, update this test to
    # require the polar error to be comparable to the equatorial error.
    assert polar_rms > 20.0 * equatorial_rms


def test_uniform_field_has_no_periodic_azimuth_seam_error():
    """The periodic stencil must use the local wrapped phi separation."""
    grid = _evaluate_cartesian_field(_grid(), _constant_field)
    j_ring_rms = np.sqrt(np.mean(grid.J_tot**2, axis=(1, 2)))

    seam_rms = max(j_ring_rms[0], j_ring_rms[-1])
    nonseam_rms = np.median(j_ring_rms[1:-1])

    assert seam_rms < 1.1 * nonseam_rms


def test_uniform_field_error_converges_when_azimuth_is_refined():
    """Away from the poles, finite-difference error should decrease with dphi."""
    coarse = _evaluate_cartesian_field(_grid(dphi_deg=30.0), _constant_field)
    fine = _evaluate_cartesian_field(_grid(dphi_deg=15.0), _constant_field)

    coarse_error = np.sqrt(np.mean(coarse.J_tot[_interior_theta_slice(coarse)] ** 2))
    fine_error = np.sqrt(np.mean(fine.J_tot[_interior_theta_slice(fine)] ** 2))

    assert fine_error < 0.4 * coarse_error


def test_solid_body_field_recovers_known_constant_cartesian_curl():
    """B = 0.5 C x r should produce J = C z-hat / (mu0 R_Mars)."""
    curl_amplitude = 2.0
    grid = _evaluate_cartesian_field(
        _grid(dtheta_deg=5.0, dphi_deg=15.0),
        _constant_z_curl_field(curl_amplitude),
    )
    interior = _interior_theta_slice(grid, excluded_rings=2)
    expected_jz = curl_amplitude / (MU0 * grid.radius_km * 1e3)

    assert np.allclose(grid.Jz[interior], expected_jz, rtol=4e-3, atol=0.0)
    assert np.sqrt(np.mean(grid.Jx[interior] ** 2)) < 4e-3 * expected_jz
    assert np.sqrt(np.mean(grid.Jy[interior] ** 2)) < 4e-3 * expected_jz
