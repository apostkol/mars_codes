import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

SCRIPT = Path(__file__).parent.parent / "paper" / "sza_plots.py"
SPEC = importlib.util.spec_from_file_location("sza_plots", SCRIPT)
sza_plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sza_plots)


def test_regional_statistics_use_centers_and_disjoint_altitude_bands():
    rows = sza_plots.regional_map_statistics(
        np.array([0, 40, 50, 180]),
        np.array([225, 275, 325, 375, 425]),
        np.array([[999, 1, 3], [999, 2, np.nan], [999, 4, 6], [999, 8, 10]]),
    )
    assert rows[0]["count"] == 2
    assert rows[0]["mean"] == pytest.approx(2)
    assert rows[0]["std"] == pytest.approx(1)
    assert rows[1]["count"] == 5
    assert rows[1]["mean"] == pytest.approx(6)
    assert rows[1]["std"] == pytest.approx(np.sqrt(8))


def test_regional_statistics_report_empty_regions():
    rows = sza_plots.regional_map_statistics(
        np.array([0, 180]), np.array([500, 600]), np.array([[1.0]])
    )
    assert all(row["count"] == 0 and np.isnan(row["mean"]) for row in rows)


_resolve_selections = sza_plots._resolve_selections
sza_altitude_map = sza_plots.sza_altitude_map
plot_sza_panel = sza_plots.plot_sza_panel
_all_panel_specification = sza_plots._all_panel_specification
_expand_grid_arguments = sza_plots._expand_grid_arguments
optical_shadow_profile = sza_plots.optical_shadow_profile


def _controlled_grid():
    """Two equal-area values in each of two radial levels."""
    cells = []
    for radius in (1.1, 1.2):
        cells.extend(
            [
                SimpleNamespace(
                    r_c=radius,
                    th_c=np.pi / 2,
                    phi_c=0.0,
                    Jr=-1.0,
                    Br=5.0,
                ),
                SimpleNamespace(
                    r_c=radius,
                    th_c=np.pi / 2,
                    phi_c=0.0,
                    Jr=3.0,
                    Br=7.0,
                ),
            ]
        )
    return SimpleNamespace(cells=cells, radius_km=3396.19)


def test_sza_panel_data_rectangles_have_requested_aspect():
    fig, axes, _ = sza_plots.plot_sza_panels(
        _controlled_grid(), ("Jr", "Br"), plot_boundaries=False
    )
    fig.canvas.draw()
    for ax in axes.flat:
        bounds = ax.get_window_extent()
        assert bounds.width / bounds.height == pytest.approx(1.62, rel=0.01)
    plt.close(fig)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("all", 1.0), ("positive", 3.0), ("negative", -1.0)],
)
def test_sign_selection_happens_before_averaging(selection, expected):
    _, _, values = sza_altitude_map(
        _controlled_grid(), "Jr", selection=selection, sza_step=10
    )
    assert values[0, 0] == pytest.approx(expected)


def test_arbitrary_scalar_cell_parameter():
    _, _, values = sza_altitude_map(_controlled_grid(), "Br", sza_step=10)
    assert values[0, 0] == pytest.approx(6.0)


def test_virtual_horizontal_current_from_cell_components():
    grid = _controlled_grid()
    for cell in grid.cells:
        cell.Jth = 3.0
        cell.Jphi = 4.0

    _, _, values = sza_altitude_map(grid, "Jh", sza_step=10)
    assert values[0, 0] == pytest.approx(5.0)


def test_grid_level_array_parameter_uses_cell_indices():
    cells = []
    for phi_index, phi in enumerate((-np.pi / 4, np.pi / 4)):
        for radial_index, radius in enumerate((1.1, 1.2)):
            cells.append(
                SimpleNamespace(
                    r_c=radius,
                    th_c=np.pi / 2,
                    phi_c=phi,
                    phi_idx=phi_index,
                    th_idx=0,
                    r_idx=radial_index,
                )
            )
    grid = SimpleNamespace(
        cells=cells,
        phi=np.array([-np.pi / 4, np.pi / 4]),
        th=np.array([np.pi / 2]),
        r=np.array([1.1, 1.2]),
        JdotE=np.array([[[-2.0, -4.0]], [[6.0, 8.0]]]),
        Jth=np.full((2, 1, 2), 5.0),
        Jphi=np.full((2, 1, 2), 12.0),
        radius_km=3396.19,
    )

    _, _, all_values = sza_altitude_map(grid, "JdotE", sza_step=10)
    _, _, positive_values = sza_altitude_map(
        grid, "JdotE", selection="positive", sza_step=10
    )
    assert all_values[0, 4] == pytest.approx(2.0)
    assert positive_values[0, 4] == pytest.approx(6.0)
    _, _, horizontal_values = sza_altitude_map(grid, "Jh", sza_step=10)
    assert horizontal_values[0, 4] == pytest.approx(13.0)


def test_selection_input_forms():
    parameters = ["Jr", "Jth", "Br"]
    assert _resolve_selections(parameters, "positive") == ["positive"] * 3
    assert _resolve_selections(parameters, {"Jr": "negative"}) == [
        "negative",
        "all",
        "all",
    ]
    assert _resolve_selections(parameters, ["all", "positive", "negative"]) == [
        "all",
        "positive",
        "negative",
    ]


def test_all_panel_specification_is_parameter_row_triads():
    parameters, selections = _all_panel_specification()
    assert len(parameters) == 8 * 3
    assert parameters[:6] == ["Br", "Br", "Br", "Bth", "Bth", "Bth"]
    assert selections[:6] == [
        "all",
        "positive",
        "negative",
        "all",
        "positive",
        "negative",
    ]
    assert parameters[-3:] == ["JdotE"] * 3


def test_optical_shadow_starts_at_limb_and_follows_mars_radius_cylinder():
    sza, altitude = optical_shadow_profile(radius_km=3396.19)
    planet_radius = 1 + altitude / 3396.19
    cylindrical_radius = planet_radius * np.sin(np.radians(sza))

    assert sza[0] == pytest.approx(90.0)
    assert altitude[0] == pytest.approx(0.0)
    assert cylindrical_radius == pytest.approx(np.ones_like(cylindrical_radius))


def test_grid_arguments_expand_sorted_wildcards_and_remove_duplicates(tmp_path):
    second = tmp_path / "grid_b.pickle"
    first = tmp_path / "grid_a.pickle"
    first.touch()
    second.touch()

    paths = _expand_grid_arguments([str(tmp_path / "grid_*.pickle"), str(first)])
    assert paths == [first, second]


def test_unmatched_grid_wildcard_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="matched no files"):
        _expand_grid_arguments([str(tmp_path / "missing_*.pickle")])


def test_mse_e_hemisphere_filter_is_applied_before_averaging():
    cells = []
    for radius in (1.1, 1.2):
        cells.extend(
            [
                SimpleNamespace(
                    r_c=radius,
                    th_c=np.radians(60),
                    phi_c=np.radians(10),
                    Jr=2.0,
                ),
                SimpleNamespace(
                    r_c=radius,
                    th_c=np.radians(120),
                    phi_c=np.radians(10),
                    Jr=8.0,
                ),
            ]
        )
    grid = SimpleNamespace(cells=cells, radius_km=3396.19)

    _, _, both = sza_altitude_map(grid, "Jr", sza_step=10, hemisphere="both")
    _, _, plus_e = sza_altitude_map(grid, "Jr", sza_step=10, hemisphere="+E")
    _, _, minus_e = sza_altitude_map(grid, "Jr", sza_step=10, hemisphere="-E")
    _, _, difference = sza_altitude_map(grid, "Jr", sza_step=10, hemisphere="diff")

    assert both[0, 3] == pytest.approx(5.0)
    assert plus_e[0, 3] == pytest.approx(2.0)
    assert minus_e[0, 3] == pytest.approx(8.0)
    assert difference[0, 3] == pytest.approx(-6.0)


def test_hemisphere_panels_retain_both_hemisphere_color_range():
    cells = []
    for radius in (1.1, 1.2):
        cells.extend(
            [
                SimpleNamespace(
                    r_c=radius,
                    th_c=np.radians(60),
                    phi_c=np.radians(10),
                    Jr=2.0,
                ),
                SimpleNamespace(
                    r_c=radius,
                    th_c=np.radians(120),
                    phi_c=np.radians(10),
                    Jr=8.0,
                ),
            ]
        )
    grid = SimpleNamespace(cells=cells, radius_km=3396.19)
    fig, axes = plt.subplots(1, 2)
    both = plot_sza_panel(grid, "Jr", axes[0], hemisphere="both", plot_boundaries=False)
    plus_e = plot_sza_panel(grid, "Jr", axes[1], hemisphere="+E", plot_boundaries=False)

    assert plus_e["mesh"].norm.vmin == pytest.approx(both["mesh"].norm.vmin)
    assert plus_e["mesh"].norm.vmax == pytest.approx(both["mesh"].norm.vmax)
    assert np.nanmax(plus_e["values"]) == pytest.approx(2.0)
    assert np.nanmax(plus_e["color_values"]) == pytest.approx(5.0)
    plt.close(fig)


def test_hemisphere_difference_uses_brbg_colormap():
    grid = _controlled_grid()
    for index, cell in enumerate(grid.cells):
        cell.th_c = np.radians(60 if index % 2 == 0 else 120)

    fig, ax = plt.subplots()
    result = plot_sza_panel(grid, "Jr", ax, hemisphere="diff", plot_boundaries=False)
    assert result["mesh"].cmap.name == "BrBG"
    plt.close(fig)
