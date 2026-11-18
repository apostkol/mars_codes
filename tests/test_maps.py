from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from mars_currents.maps import (
    cartesian_plane_map,
    mollweide_comparison_maps,
    mollweide_maps,
    single_mollweide_map,
)


def _minimal_grid():
    phi = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    th = np.linspace(0.2, np.pi - 0.2, 4)
    r = np.array([1.0, 2.0])
    return SimpleNamespace(
        r=r,
        th=th,
        phi=phi,
        value=np.zeros((len(phi), len(th), len(r))),
    )


def _minimal_mollweide_grid():
    phi = np.linspace(0.0, 2 * np.pi, 5, endpoint=False)
    th = np.linspace(0.2, np.pi - 0.2, 4)
    r = np.array([1.0, 2.0])
    phi_2d, th_2d, _ = np.meshgrid(phi, th, r, indexing="ij")
    shape = phi_2d.shape
    return SimpleNamespace(
        r=r,
        phi_2d=phi_2d,
        th_2d=th_2d,
        Bsmr=np.ones(shape),
        Bsmth=np.full(shape, 2.0),
        Bsmphi=np.full(shape, 3.0),
        measurements_counts=np.full(shape, 4.0),
    )


def test_mollweide_maps_delegates_each_panel_to_single_map(monkeypatch):
    import mars_currents.maps as maps_module

    calls = []
    original = maps_module.single_mollweide_map

    def recording_single_map(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(maps_module, "single_mollweide_map", recording_single_map)
    fig, axes = mollweide_maps(
        _minimal_mollweide_grid(),
        r_values=[1.0, 2.0],
    )

    try:
        assert axes.shape == (2, 4)
        assert len(calls) == 8
        assert all(call["add_colorbar"] is False for call in calls)
        assert all(hasattr(axis, "_mars_currents_cmesh") for axis in axes.flat)
        assert [axis.get_title() for axis in axes[0]] == [
            r"$B_{r}$",
            r"$B_{\theta}$",
            r"$B_{\phi}$",
            "counts/bin",
        ]
        assert all(axis.get_title() == "" for axis in axes[1])
        assert all(axis.get_xlabel() == "" for axis in axes[0])
        assert all(axis.get_xlabel() for axis in axes[1])
        assert all(axis.get_ylabel() for axis in axes[:, 0])
        assert all(axis.get_ylabel() == "" for axis in axes[:, 1:].flat)
        assert np.allclose(
            [axis.get_position().height for axis in axes[:, 0]],
            axes[0, 0].get_position().height,
        )
        panel_labels = [
            [text.get_text() for text in axis.texts if text.get_text()[:1].isalpha()][0]
            for axis in axes.flat
        ]
        assert panel_labels == ["a1", "b1", "c1", "d1", "a2", "b2", "c2", "d2"]
        assert sum(
            text.get_text() == r"$\vec{B}_{\rm IMF}$"
            for text in fig.findobj(match=lambda artist: hasattr(artist, "get_text"))
        ) == 1
    finally:
        plt.close(fig)


def test_mollweide_maps_supports_column_wise_component_scales():
    fig, axes = mollweide_maps(
        _minimal_mollweide_grid(),
        r_values=[1.0, 2.0],
        color_scale="column",
        percentile=100,
    )

    try:
        expected = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]
        for row in axes:
            assert [axis._mars_currents_cmesh.get_clim() for axis in row[:3]] == expected
    finally:
        plt.close(fig)


def test_mollweide_maps_scales_counts_from_zero_separately():
    grid = _minimal_mollweide_grid()
    grid.measurements_counts[:, :, 1] = 8.0
    fig, axes = mollweide_maps(
        grid,
        r_values=[1.0, 2.0],
        color_scale="all",
        percentile=100,
    )

    try:
        assert axes[0, 3]._mars_currents_cmesh.get_clim() == (0.0, 8.0)
        assert axes[1, 3]._mars_currents_cmesh.get_clim() == (0.0, 8.0)
        assert axes[0, 0]._mars_currents_cmesh.get_clim() == (-3.0, 3.0)
    finally:
        plt.close(fig)


def test_single_mollweide_map_notes_dayside_sign_statistics():
    grid = _minimal_mollweide_grid()
    values = np.zeros_like(grid.Bsmr)
    values[0, :, 0] = 1.0
    values[1, :2, 0] = 1.0
    values[1, 2:, 0] = -2.0
    values[4, 0, 0] = -2.0

    fig, ax = single_mollweide_map(
        grid,
        values,
        ir=0,
        note_medians="signed",
        add_colorbar=False,
    )

    try:
        annotation_text = [text.get_text() for text in ax.texts]
        assert "+1.0 (50%)\n-2.0 (25%)" in annotation_text
    finally:
        plt.close(fig)


def test_single_mollweide_map_uses_scientific_notation_for_small_statistics():
    grid = _minimal_mollweide_grid()
    values = np.zeros_like(grid.Bsmr)
    values[0, :2, 0] = 1.2e-11
    values[0, 2:, 0] = -8.4e-12

    fig, ax = single_mollweide_map(
        grid,
        values,
        ir=0,
        note_medians="signed",
        add_colorbar=False,
    )

    try:
        annotation_text = [text.get_text() for text in ax.texts]
        assert "+1.2e-11 (17%)\n-8.4e-12 (17%)" in annotation_text
    finally:
        plt.close(fig)


def test_mollweide_comparison_maps_uses_one_grid_per_column():
    low = _minimal_mollweide_grid()
    high = _minimal_mollweide_grid()
    low.dr = high.dr = 1.0
    low.Bsmr = np.full_like(low.Bsmr, 1.0)
    high.Bsmr = np.full_like(high.Bsmr, 4.0)
    norm = matplotlib.colors.Normalize(-5, 5)

    fig, axes = mollweide_comparison_maps(
        [low, high],
        "Bsmr",
        [1.0, 2.0],
        column_names=["Low", "High"],
        norm=norm,
        note_medians=None,
        show_IMF=False,
    )

    try:
        assert axes.shape == (2, 2)
        assert [axis.get_title() for axis in axes[0]] == ["Low", "High"]
        assert all(axis.get_title() == "" for axis in axes[1])
        assert axes[0, 0]._mars_currents_cmesh.norm is norm
        assert axes[1, 1]._mars_currents_cmesh.norm is norm
        assert np.allclose(
            [axis.get_position().height for axis in axes[:, 0]],
            axes[0, 0].get_position().height,
        )
        assert np.all(axes[0, 0]._mars_currents_cmesh.get_array() == 1.0)
        assert np.all(axes[0, 1]._mars_currents_cmesh.get_array() == 4.0)
        radial_note = next(
            text.get_text()
            for text in axes[0, 0].texts
            if text.get_text().startswith("$h_")
        )
        assert "\n" in radial_note
        assert "$r_{\\rm min}=" in radial_note
        assert "R_{M}$" in radial_note
        panel_labels = [
            [text.get_text() for text in axis.texts if text.get_text()[:1].isalpha()][0]
            for axis in axes.flat
        ]
        assert panel_labels == ["a1", "b1", "a2", "b2"]
    finally:
        plt.close(fig)


def _grid_with_cartesian_x_field(magnitude):
    grid = _minimal_grid()
    shape = (len(grid.phi), len(grid.th), len(grid.r))
    phi = grid.phi[:, None, None]
    grid.Br = np.broadcast_to(magnitude * np.cos(phi), shape)
    grid.Bth = np.zeros(shape)
    grid.Bphi = np.broadcast_to(-magnitude * np.sin(phi), shape)
    grid.Bx = np.full(shape, magnitude)
    grid.By = np.zeros(shape)
    grid.Bz = np.zeros(shape)
    return grid


def _grid_with_explicit_cartesian_field(vector):
    grid = _minimal_grid()
    shape = grid.value.shape
    for name, value in zip(("Bx", "By", "Bz"), vector):
        setattr(grid, name, np.full(shape, value, dtype=float))
    # Deliberately contradictory spherical values: the plane mapper must prefer
    # the unambiguous Cartesian vector products when they are present.
    grid.Br = np.full(shape, 99.0)
    grid.Bth = np.full(shape, 99.0)
    grid.Bphi = np.full(shape, 99.0)
    return grid


def _plane_map(*args, **kwargs):
    _, fig, ax = cartesian_plane_map(*args, **kwargs)
    return fig, ax


def test_cartesian_plane_map_orients_z_up_when_plane_contains_z():
    fig, ax = _plane_map(
        _minimal_grid(),
        "value",
        plane_normal=(0.0, 1.0, 0.0),
        n=8,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        basis = ax._mars_currents_plane_basis
        assert np.allclose(basis["v_hat"], [0.0, 0.0, 1.0])
        assert np.dot(basis["u_hat"], [1.0, 0.0, 0.0]) > 0.0
    finally:
        plt.close(fig)


def test_cartesian_plane_map_passes_standard_quiver_kwargs():
    fig, ax = _plane_map(
        _grid_with_cartesian_x_field(4.0),
        "value",
        plane_normal=(0.0, 0.0, 1.0),
        n=8,
        arrows="B",
        scale=25,
        scale_units="xy",
        angles="xy",
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        quiver = ax._mars_currents_quiver
        metadata = ax._mars_currents_arrow_components
        assert quiver.scale == 25
        assert quiver.scale_units == "xy"
        assert quiver.angles == "xy"
        assert metadata["quiver_kwargs"]["scale"] == 25
    finally:
        plt.close(fig)


def test_cartesian_plane_map_prefers_cartesian_arrow_components():
    fig, ax = _plane_map(
        _grid_with_explicit_cartesian_field((2.0, 3.0, 4.0)),
        "value",
        plane_normal=(0.0, 1.0, 0.0),
        n=8,
        arrows="B",
        arrow_step=1,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        arrows = ax._mars_currents_arrow_components
        finite = np.isfinite(arrows["u"]) & np.isfinite(arrows["v"])

        assert arrows["source"] == "cartesian"
        assert np.allclose(arrows["u"][finite], 2.0)
        assert np.allclose(arrows["v"][finite], 4.0)
    finally:
        plt.close(fig)


def test_cartesian_plane_map_requires_cartesian_arrow_components():
    grid = _minimal_grid()

    with np.testing.assert_raises_regex(
        AttributeError,
        "Cartesian arrow components are missing: Bx, By, Bz",
    ):
        cartesian_plane_map(
            grid,
            "value",
            arrows="B",
            n=8,
            colorbar=False,
            pole_theta_bins=0,
            show_xyz_triad=False,
        )


def test_cartesian_plane_map_can_sqrt_transform_arrow_lengths_before_quiver():
    fig_linear, ax_linear = _plane_map(
        _grid_with_cartesian_x_field(4.0),
        "value",
        plane_normal=(0.0, 0.0, 1.0),
        n=8,
        arrows="B",
        scale=25,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )
    fig_sqrt, ax_sqrt = _plane_map(
        _grid_with_cartesian_x_field(4.0),
        "value",
        plane_normal=(0.0, 0.0, 1.0),
        n=8,
        arrows="B",
        arrow_length_transform="sqrt",
        scale=25,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        linear = ax_linear._mars_currents_arrow_components
        sqrt = ax_sqrt._mars_currents_arrow_components
        linear_magnitude = np.hypot(linear["u"], linear["v"])
        sqrt_magnitude = np.hypot(sqrt["u"], sqrt["v"])
        finite = np.isfinite(linear_magnitude) & (linear_magnitude > 0)

        assert sqrt["arrow_length_transform"] == "sqrt"
        assert sqrt["quiver_kwargs"]["scale"] == 25
        assert np.allclose(sqrt_magnitude[finite], np.sqrt(linear_magnitude[finite]))
        assert np.allclose(
            sqrt["u"][finite] / sqrt_magnitude[finite],
            linear["u"][finite] / linear_magnitude[finite],
        )
        assert np.allclose(
            sqrt["v"][finite] / sqrt_magnitude[finite],
            linear["v"][finite] / linear_magnitude[finite],
        )
    finally:
        plt.close(fig_linear)
        plt.close(fig_sqrt)


def test_cartesian_plane_map_labels_cardinal_plane_axes():
    fig, ax = _plane_map(
        _minimal_grid(),
        "value",
        plane_normal=(0.0, 1.0, 0.0),
        n=8,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        assert ax.get_xlabel() == "MSE X"
        assert ax.get_ylabel() == "MSE Z"
    finally:
        plt.close(fig)


def test_cartesian_plane_map_uses_specificity_for_near_z_containing_plane():
    fig, ax = _plane_map(
        _minimal_grid(),
        "value",
        plane_normal=(0.0, 1.0, 0.2),
        n=8,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        v_hat = ax._mars_currents_plane_basis["v_hat"]
        assert np.dot(v_hat, [0.0, 0.0, 1.0]) > 0.95
    finally:
        plt.close(fig)


def test_cartesian_plane_map_allows_explicit_axis_labels():
    fig, ax = _plane_map(
        _minimal_grid(),
        "value",
        plane_normal=(0.0, 1.0, 0.0),
        n=8,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
        u_label="Tailward distance",
        v_label="Northward distance",
    )

    try:
        assert ax.get_xlabel() == "Tailward distance"
        assert ax.get_ylabel() == "Northward distance"
    finally:
        plt.close(fig)


def test_cartesian_plane_map_plots_conic_boundary_intersections():
    fig, ax = _plane_map(
        _minimal_grid(),
        "value",
        plane_normal=(0.0, 1.0, 0.0),
        extent=(-3.0, 3.0, -3.0, 3.0),
        n=101,
        plot_boundaries=True,
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        boundaries = ax._mars_currents_boundaries
        assert boundaries["bow_shock"]["contour"] is not None
        assert boundaries["pileup_boundary"]["contour"] is not None
        assert boundaries["bow_shock"]["parameters"] == {
            "x0": 0.74,
            "L": 1.82,
            "eps": 1.01,
        }
    finally:
        plt.close(fig)


def test_cartesian_plane_map_accepts_custom_boundary_parameters_and_style():
    custom = {"x0": 0.0, "L": 1.5, "eps": 0.0}
    fig, ax = _plane_map(
        _minimal_grid(),
        "value",
        plane_normal=(0.0, 0.0, 1.0),
        extent=2.0,
        n=51,
        plot_boundaries=True,
        bow_shock_pars=custom,
        bow_shock_kwargs={"color": "magenta"},
        colorbar=False,
        pole_theta_bins=0,
        show_xyz_triad=False,
    )

    try:
        boundary = ax._mars_currents_boundaries["bow_shock"]
        assert boundary["parameters"] == custom
        assert boundary["contour"] is not None
        assert np.allclose(
            boundary["contour"].get_edgecolor()[0][:3],
            matplotlib.colors.to_rgb("magenta"),
        )
    finally:
        plt.close(fig)
