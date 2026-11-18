"""Plot gridded quantities as a function of SZA and altitude.

``plot_sza_panel`` plots one parameter on an existing Matplotlib axis;
``plot_sza_panels`` arranges and optionally saves a list of panels.

The angular dimensions of a spherical ``Grid`` are reduced to solar-zenith-
angle bins. Values are averaged using angular cell area (proportional to
``sin(theta)``) as a weight. Positive or negative values may be selected before
that average is calculated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import NullFormatter

import mars_currents.config as cfg
from mars_currents.grid import load_grid
from mars_currents.maps import DEFAULT_BOUNDARY_PARAMETERS, MARS_RADIUS_KM

GRID_FILES = (
    "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle",
    "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_NH.pickle",
    "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_low_pdyn.pickle",
    "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_intermediate_pdyn.pickle",
    "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_high_pdyn.pickle",
)

DEFAULT_PARAMETERS = ("Jr", "Jth", "Jphi")
VALID_SELECTIONS = ("all", "positive", "negative")
VALID_HEMISPHERES = ("both", "+E", "-E", "diff")
PANEL_WIDTH_HEIGHT_RATIO = 1.62
ALL_PARAMETER_ROWS = (
    "Br",
    "Bth",
    "Bphi",
    "Jr",
    "Jth",
    "Jphi",
    "Jh",
    "JdotE",
)

PARAMETER_LABELS = {
    "Jr": r"$J_r$",
    "Jth": r"$J_\theta$",
    "Jphi": r"$J_\phi$",
    "Jh": r"$J_h$",
    "Br": r"$B_r$",
    "Bth": r"$B_\theta$",
    "Bphi": r"$B_\phi$",
    "Bsmr": r"$B_{\mathrm{sm},r}$",
    "Bsmth": r"$B_{\mathrm{sm},\theta}$",
    "Bsmphi": r"$B_{\mathrm{sm},\phi}$",
    "JdotE": r"$\mathbf{J}\cdot\mathbf{E}$",
    "power": "Power",
}

PARAMETER_UNITS = {
    "JdotE": r"W m$^{-3}$",
    "power": "W",
    "JxB_mag": r"N m$^{-3}$",
}


def _center_edges(centers):
    """Return plotting edges for monotonically increasing bin centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("At least two radial grid levels are required.")
    midpoints = (centers[1:] + centers[:-1]) / 2
    return np.r_[
        centers[0] - (midpoints[0] - centers[0]),
        midpoints,
        centers[-1] + (centers[-1] - midpoints[-1]),
    ]


def _sza_edges(sza_step):
    if not 0 < sza_step <= 180:
        raise ValueError("sza_step must be in the interval (0, 180].")
    edges = np.arange(0.0, 180.0, sza_step)
    if edges.size == 0 or not np.isclose(edges[-1], 180.0):
        edges = np.r_[edges, 180.0]
    return edges


def _include_value(value, selection):
    """State explicitly which values enter a selected weighted average."""
    if not np.isfinite(value):
        return False
    if selection == "positive":
        return value > 0
    if selection == "negative":
        return value < 0
    return True


def sza_altitude_map(grid, parameter, selection="all", sza_step=5.0, hemisphere="both"):
    """Calculate one area-weighted SZA/altitude map.

    ``parameter`` names a scalar cell attribute such as ``"Jr"`` or ``"Br"``.
    ``selection`` is ``"all"``, ``"positive"``, or ``"negative"`` and is
    applied before averaging. Zero is excluded from both sign selections.
    ``hemisphere`` optionally retains only MSE Z>0 (``"+E"``) or Z<0
    (``"-E"``); cells on Z=0 are excluded from either one-sided selection.
    ``"diff"`` returns the difference of the independently averaged maps,
    ``<+E> - <-E>``.
    """
    if selection not in VALID_SELECTIONS:
        raise ValueError(f"selection must be one of {VALID_SELECTIONS}.")
    if hemisphere not in VALID_HEMISPHERES:
        raise ValueError(f"hemisphere must be one of {VALID_HEMISPHERES}.")
    if not grid.cells:
        raise ValueError("The grid has no cells.")

    if hemisphere == "diff":
        sza_edges, altitude_edges, plus_e = sza_altitude_map(
            grid,
            parameter,
            selection=selection,
            sza_step=sza_step,
            hemisphere="+E",
        )
        _, _, minus_e = sza_altitude_map(
            grid,
            parameter,
            selection=selection,
            sza_step=sza_step,
            hemisphere="-E",
        )
        return sza_edges, altitude_edges, plus_e - minus_e

    # Jh is a virtual parameter: calculate it for plotting without changing the
    # Grid or its Cells. Prefer the Grid arrays when they are available.
    grid_values = None
    cell_virtual_parameter = False
    if parameter == "Jh" and hasattr(grid, "Jth") and hasattr(grid, "Jphi"):
        grid_values = np.hypot(np.asarray(grid.Jth), np.asarray(grid.Jphi))
    elif parameter == "Jh" and all(
        hasattr(grid.cells[0], component) for component in ("Jth", "Jphi")
    ):
        cell_virtual_parameter = True

    # Grid arrays use (phi, theta, radius) order. Prefer such an array when it
    # exists; derived fields such as JdotE and power are stored only this way.
    # Otherwise fall back to a scalar attribute carried by every Cell.
    elif hasattr(grid, parameter):
        possible_grid_values = np.asarray(getattr(grid, parameter))
        expected_shape = (len(grid.phi), len(grid.th), len(grid.r))
        if possible_grid_values.shape == expected_shape:
            grid_values = possible_grid_values

    if grid_values is not None:
        expected_shape = (len(grid.phi), len(grid.th), len(grid.r))
        if grid_values.shape != expected_shape:
            raise ValueError(
                f"Grid values used for {parameter!r} have shape "
                f"{grid_values.shape}; expected {expected_shape}."
            )

    if (
        grid_values is None
        and not cell_virtual_parameter
        and not hasattr(grid.cells[0], parameter)
    ):
        raise AttributeError(
            f"{parameter!r} is neither a scalar Cell attribute nor a Grid array "
            "with shape (phi, theta, radius). Derived fields may need to be "
            "calculated on the Grid before plotting."
        )

    radii = np.unique([cell.r_c for cell in grid.cells]).astype(float)
    radii.sort()
    radial_index = {radius: index for index, radius in enumerate(radii)}
    sza_edges = _sza_edges(sza_step)

    weighted_sum = np.zeros((len(radii), len(sza_edges) - 1), dtype=float)
    weight_sum = np.zeros_like(weighted_sum)

    # Keep this loop explicit: it records every coordinate, selection, and
    # weighting decision involved in constructing a plotted pixel.
    for cell in grid.cells:
        if hemisphere == "+E" and not cell.th_c < np.pi / 2:
            continue
        if hemisphere == "-E" and not cell.th_c > np.pi / 2:
            continue

        if grid_values is not None:
            value = grid_values[cell.phi_idx, cell.th_idx, cell.r_idx]
        elif cell_virtual_parameter:
            value = np.hypot(cell.Jth, cell.Jphi)
        else:
            value = getattr(cell, parameter, np.nan)
        if not _include_value(value, selection):
            continue

        cos_sza = np.sin(cell.th_c) * np.cos(cell.phi_c)
        sza = np.degrees(np.arccos(np.clip(cos_sza, -1.0, 1.0)))
        sza_index = np.searchsorted(sza_edges, sza, side="right") - 1
        sza_index = min(sza_index, len(sza_edges) - 2)
        if sza_index < 0:
            continue

        r_index = radial_index[cell.r_c]
        area_weight = np.sin(cell.th_c)
        if area_weight <= 0:
            continue

        weighted_sum[r_index, sza_index] += value * area_weight
        weight_sum[r_index, sza_index] += area_weight

    values = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full_like(weighted_sum, np.nan),
        where=weight_sum > 0,
    )

    radius_km = getattr(grid, "radius_km", MARS_RADIUS_KM)
    altitude_edges = (_center_edges(radii) - 1.0) * radius_km
    return sza_edges, altitude_edges, values


def boundary_profile(parameters, radius_km=MARS_RADIUS_KM, sample_count=2000):
    """Return SZA and altitude for a displaced-focus conic boundary."""
    x0 = float(parameters["x0"])
    semi_latus_rectum = float(parameters["L"])
    eccentricity = float(parameters["eps"])

    theta_max = np.pi
    if eccentricity >= 1:
        theta_max = np.arccos(-1 / eccentricity) - 1e-5
    theta = np.linspace(0, theta_max, sample_count)
    radius_from_focus = semi_latus_rectum / (1 + eccentricity * np.cos(theta))
    x = x0 + radius_from_focus * np.cos(theta)
    cylindrical_radius = radius_from_focus * np.sin(theta)
    planet_radius = np.hypot(x, cylindrical_radius)
    sza = np.degrees(np.arctan2(cylindrical_radius, x))
    altitude = (planet_radius - 1) * radius_km

    valid = np.isfinite(sza) & np.isfinite(altitude)
    order = np.argsort(sza[valid])
    return sza[valid][order], altitude[valid][order]


def optical_shadow_profile(radius_km=MARS_RADIUS_KM, sample_count=1000):
    """Return the nightside optical-shadow boundary cast by solid Mars.

    The shadow is a cylinder of radius one Mars radius about the MSE X axis.
    Therefore ``r sin(SZA) = 1`` outside the planet, from the optical limb at
    90 degrees toward the antisolar direction.
    """
    sza = np.linspace(90.0, 180.0 - 1e-5, sample_count)
    planet_radius = 1 / np.sin(np.radians(sza))
    altitude = (planet_radius - 1) * radius_km
    return sza, altitude


def _parameter_metadata(parameter):
    """Supply a readable title and physical unit for common Grid fields."""
    title = PARAMETER_LABELS.get(parameter, parameter.replace("_", " "))
    if parameter in PARAMETER_UNITS:
        unit = PARAMETER_UNITS[parameter]
    elif parameter.startswith("B"):
        unit = "nT"
    elif parameter.startswith("J") and parameter not in {"JdotE", "JxB"}:
        unit = r"nA m$^{-2}$"
    else:
        unit = None
    return title, unit


def _color_settings(values, selection, percentile):
    """Choose a color scale from the selected finite values."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return "RdBu_r", Normalize(vmin=-1, vmax=1)
    if selection == "positive":
        upper = np.nanpercentile(finite, percentile)
        return "Reds", Normalize(vmin=0, vmax=upper if upper > 0 else 1.0)
    if selection == "negative":
        lower = np.nanpercentile(finite, 100 - percentile)
        return "Blues_r", Normalize(vmin=lower if lower < 0 else -1.0, vmax=0)
    limit = np.nanpercentile(np.abs(finite), percentile)
    limit = limit if limit > 0 else 1.0
    return "RdBu_r", TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)


def _plot_boundaries(ax, radius_km, add_labels):
    """Overlay nominal plasma boundaries and the solid-Mars optical shadow."""
    specifications = (
        ("bow_shock", "Bow shock", "--"),
        ("pileup_boundary", "Magnetic pile-up boundary", ":"),
    )
    for name, label, linestyle in specifications:
        sza, altitude = boundary_profile(
            DEFAULT_BOUNDARY_PARAMETERS[name], radius_km=radius_km
        )
        ax.plot(
            sza,
            altitude,
            color="black",
            linestyle=linestyle,
            linewidth=1.8,
            label=label if add_labels else None,
            zorder=3,
        )

    shadow_sza, shadow_altitude = optical_shadow_profile(radius_km=radius_km)
    ax.plot(
        shadow_sza,
        shadow_altitude,
        color="black",
        linestyle="-.",
        linewidth=1.8,
        label="Optical shadow" if add_labels else None,
        zorder=3,
    )


def _format_altitude_axis(ax, altitude_edges):
    """Apply a logarithmic altitude scale with plain-number ticks."""
    positive_edges = altitude_edges[altitude_edges > 0]
    if positive_edges.size == 0:
        raise ValueError("A logarithmic altitude axis requires positive altitudes.")
    limits = (positive_edges[0], positive_edges[-1])
    candidates = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)
    ticks = candidates[(candidates >= limits[0]) & (candidates <= limits[1])]
    ax.set_yscale("log")
    ax.set_ylim(*limits)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_sza_panel(
    grid,
    parameter,
    ax=None,
    *,
    selection="all",
    hemisphere="both",
    sza_step=5.0,
    percentile=99,
    plot_boundaries=True,
    boundary_labels=False,
):
    """Plot one Grid parameter against SZA and logarithmic altitude."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    sza_edges, altitude_edges, values = sza_altitude_map(
        grid,
        parameter,
        selection=selection,
        sza_step=sza_step,
        hemisphere=hemisphere,
    )

    # The displayed values may be hemisphere-restricted, but the color scale
    # must remain comparable between +E, -E, and both plots. Calculate its
    # limits from the same parameter/sign selection over both hemispheres.
    color_values = values
    if hemisphere != "both":
        _, _, color_values = sza_altitude_map(
            grid,
            parameter,
            selection=selection,
            sza_step=sza_step,
            hemisphere="both",
        )
    cmap, norm = _color_settings(color_values, selection, percentile)
    if hemisphere == "diff":
        cmap = "BrBG"
    mesh = ax.pcolormesh(
        sza_edges,
        altitude_edges,
        np.ma.masked_invalid(values),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )

    if plot_boundaries:
        radius_km = getattr(grid, "radius_km", MARS_RADIUS_KM)
        _plot_boundaries(ax, radius_km, add_labels=boundary_labels)

    title, unit = _parameter_metadata(parameter)
    if selection != "all":
        relation = "> 0" if selection == "positive" else "< 0"
        title = f"{title} {relation}"
    if hemisphere == "diff":
        title = rf"$\Delta_E$ {title}"
    ax.set_title(title)
    ax.set_xlabel("SZA [deg]")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    _format_altitude_axis(ax, altitude_edges)
    # Fix the visible data rectangle, independently of colorbar/layout space.
    ax.set_box_aspect(1 / PANEL_WIDTH_HEIGHT_RATIO)

    return {
        "ax": ax,
        "mesh": mesh,
        "values": values,
        "color_values": color_values,
        "sza_edges": sza_edges,
        "altitude_edges": altitude_edges,
        "unit": unit,
        "selection": selection,
        "hemisphere": hemisphere,
    }


def regional_map_statistics(sza_edges, altitude_edges, values):
    """Summarize finite map pixels by bin center, without further weighting.

    SZA centers are included from 45 through 180 degrees. Altitude bands are
    [200, 300) and [300, 400] km, so a center at 300 km is counted only once.
    Standard deviations are population standard deviations (ddof=0).
    These describe spatial variation of plotted means, not measurement error.
    """
    sza_centers = (sza_edges[:-1] + sza_edges[1:]) / 2
    altitude_centers = (altitude_edges[:-1] + altitude_edges[1:]) / 2
    sza_mask = (sza_centers >= 45) & (sza_centers <= 180)
    rows = []
    for lower, upper in ((200, 300), (300, 400)):
        altitude_mask = (altitude_centers >= lower) & (
            (altitude_centers < upper) if upper == 300 else (altitude_centers <= upper)
        )
        selected = values[np.ix_(altitude_mask, sza_mask)]
        finite = selected[np.isfinite(selected)]
        rows.append(
            {
                "altitude_band": (lower, upper),
                "count": finite.size,
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "std": float(np.std(finite, ddof=0)) if finite.size else np.nan,
            }
        )
    return rows


def print_current_statistics(grid, parameters, *, hemisphere="both", sza_step=5.0):
    """Print regional all/positive/negative map statistics for requested currents.

    Sign filtering and hemisphere handling use exactly the plotting pipeline.
    In diff mode these are statistics of the differences of hemisphere means.
    """
    currents = list(
        dict.fromkeys(
            parameter for parameter in parameters if parameter in ("Jr", "Jth", "Jphi")
        )
    )
    if not currents:
        return
    print(f"\nCurrent map statistics: SZA 45–180 deg; hemisphere={hemisphere}")
    print("Equal-weight finite map pixels; mean/std in nA m^-2; population std.")
    print(
        f"{'Field':<7} {'Selection':<10} {'h [km]':<12} {'N':>6} {'Mean':>13} {'Std':>13}"
    )
    for parameter in currents:
        for selection in VALID_SELECTIONS:
            edges, heights, values = sza_altitude_map(
                grid,
                parameter,
                selection=selection,
                sza_step=sza_step,
                hemisphere=hemisphere,
            )
            for row in regional_map_statistics(edges, heights, values):
                lower, upper = row["altitude_band"]
                band = f"[{lower},{upper}{')' if upper == 300 else ']'}"
                print(
                    f"{parameter:<7} {selection:<10} {band:<12} "
                    f"{row['count']:>6} {row['mean']:>13.6g} {row['std']:>13.6g}"
                )


def _resolve_selections(parameters, selections):
    """Expand global, keyed, or positional selection input per parameter."""
    if selections is None:
        return ["all"] * len(parameters)
    if isinstance(selections, str):
        resolved = [selections] * len(parameters)
    elif isinstance(selections, Mapping):
        resolved = [selections.get(parameter, "all") for parameter in parameters]
    elif isinstance(selections, Sequence):
        if len(selections) != len(parameters):
            raise ValueError("selections must have the same length as parameters.")
        resolved = list(selections)
    else:
        raise TypeError("selections must be a string, mapping, sequence, or None.")

    invalid = set(resolved).difference(VALID_SELECTIONS)
    if invalid:
        raise ValueError(f"Unknown selections: {sorted(invalid)}")
    return resolved


def plot_sza_panels(
    grid,
    parameters,
    *,
    selections=None,
    hemisphere="both",
    label=None,
    output_path=None,
    columns=3,
    sza_step=5.0,
    percentile=99,
    plot_boundaries=True,
):
    """Arrange SZA/altitude panels for an arbitrary parameter list.

    ``selections`` may be one mode for every panel, a mapping such as
    ``{"Jr": "positive"}``, or a sequence parallel to ``parameters``.
    ``hemisphere`` applies the same MSE Z selection to every panel.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("At least one parameter is required.")
    if columns < 1:
        raise ValueError("columns must be at least 1.")
    panel_selections = _resolve_selections(parameters, selections)

    columns = min(columns, len(parameters))
    rows = int(np.ceil(len(parameters) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.5 * columns, 4.0 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    panel_results = []

    for index, (parameter, selection) in enumerate(zip(parameters, panel_selections)):
        result = plot_sza_panel(
            grid,
            parameter,
            flat_axes[index],
            selection=selection,
            hemisphere=hemisphere,
            sza_step=sza_step,
            percentile=percentile,
            plot_boundaries=plot_boundaries,
            boundary_labels=plot_boundaries and index == 0,
        )
        colorbar_label = parameter if result["unit"] is None else result["unit"]
        fig.colorbar(result["mesh"], ax=flat_axes[index], label=colorbar_label)
        panel_results.append(result)

    for ax in flat_axes[len(parameters) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Altitude [km]")
    if plot_boundaries:
        flat_axes[0].legend(loc="upper left", fontsize="small")
    if label:
        fig.suptitle(label)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes, panel_results


def plot_grid(grid, label, output_path, sza_step=5.0):
    """Backward-compatible wrapper for the original three-current figure."""
    fig, _, _ = plot_sza_panels(
        grid,
        DEFAULT_PARAMETERS,
        label=label,
        output_path=output_path,
        sza_step=sza_step,
    )
    plt.close(fig)


def _grid_label(path):
    stem = path.stem
    for suffix, label in (
        ("_intermediate_pdyn", "Intermediate dynamic pressure"),
        ("_high_pdyn", "High dynamic pressure"),
        ("_low_pdyn", "Low dynamic pressure"),
        ("_NH", "Northern geographic hemisphere"),
        ("_all", "All data"),
    ):
        if stem.endswith(suffix):
            return label
    return stem


def _parse_cli_selections(items):
    selections = {}
    for item in items or ():
        try:
            parameter, selection = item.split("=", maxsplit=1)
        except ValueError as error:
            raise ValueError("Selections must have the form PARAMETER=MODE.") from error
        selections[parameter] = selection
    return selections


def _all_panel_specification():
    """Return row-major parameters and selections used by CLI ``--all``."""
    parameters = []
    selections = []
    for parameter in ALL_PARAMETER_ROWS:
        parameters.extend([parameter] * len(VALID_SELECTIONS))
        selections.extend(VALID_SELECTIONS)
    return parameters, selections


def _expand_grid_arguments(arguments):
    """Expand CLI grid paths and wildcard patterns in a predictable order."""
    expanded_paths = []
    seen_paths = set()

    for argument in arguments:
        pattern = str(Path(argument).expanduser())
        if glob.has_magic(pattern):
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"Grid pattern matched no files: {argument}")
        else:
            matches = [pattern]

        for match in matches:
            path = Path(match)
            identity = path.resolve()
            if identity not in seen_paths:
                expanded_paths.append(path)
                seen_paths.add(identity)

    return expanded_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        action="append",
        help=(
            "Grid pickle or quoted wildcard pattern; repeat as needed "
            "(defaults to paper grids)."
        ),
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        help="Grid or Cell fields to plot.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Plot Br/Bth/Bphi, Jr/Jth/Jphi, Jh, and JdotE as rows, with "
            "all/positive/negative columns."
        ),
    )
    parser.add_argument(
        "--selection",
        action="append",
        help="Pre-average sign selection, e.g. Jr=positive or Jr=negative.",
    )
    parser.add_argument(
        "--hemisphere",
        choices=("both", "plusE", "minusE", "diff"),
        default="both",
        help=(
            "Use both hemispheres, MSE plusE (Z>0), minusE (Z<0), or their "
            "weighted-mean difference plusE-minusE."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=cfg.PLOTS, help="Figure output directory."
    )
    parser.add_argument(
        "--sza-step", type=float, default=5.0, help="SZA-bin width in degrees."
    )
    parser.add_argument(
        "--format", choices=("pdf", "png"), default="pdf", help="Output format."
    )
    args = parser.parse_args(argv)
    if args.all and args.parameters is not None:
        parser.error("--all cannot be combined with --parameters.")
    if args.all and args.selection:
        parser.error("--all cannot be combined with --selection.")

    if args.all:
        parameters, selections = _all_panel_specification()
    else:
        parameters = args.parameters or list(DEFAULT_PARAMETERS)
        selections = _parse_cli_selections(args.selection)

    grid_paths = (
        _expand_grid_arguments(args.grid)
        if args.grid
        else [cfg.RESULTS / name for name in GRID_FILES]
    )
    parameter_stem = "all_parameters" if args.all else "_".join(parameters)
    hemisphere = {
        "both": "both",
        "plusE": "+E",
        "minusE": "-E",
        "diff": "diff",
    }[args.hemisphere]
    hemisphere_stem = {
        "both": "",
        "+E": "_plusE",
        "-E": "_minusE",
        "diff": "_hemisphere_diff",
    }[hemisphere]
    for path in grid_paths:
        path = Path(path)
        print(f"Loading {path}")
        grid = load_grid(load_from_file=True, file_path=path)

        if "JdotE" in parameters:
            print("Calculating JdotE with Grid.calculate_JdotE() defaults")
            grid.calculate_JdotE()
        if any(parameter.startswith("JxB") for parameter in parameters):
            print("Calculating JxB with Grid.calculate_JxB() defaults")
            grid.calculate_JxB()

        output_path = args.output_dir / (
            f"sza_{parameter_stem}_{_grid_label(path).lower().replace(' ', '_')}"
            f"{hemisphere_stem}."
            f"{args.format}"
        )
        figure_label = _grid_label(path)
        if hemisphere == "diff":
            figure_label += (
                r" — MSE hemisphere difference ($\langle+E\rangle-\langle-E\rangle$)"
            )
        elif hemisphere != "both":
            figure_label += f" — MSE {hemisphere} hemisphere"
        fig, _, _ = plot_sza_panels(
            grid,
            parameters,
            selections=selections,
            hemisphere=hemisphere,
            label=figure_label,
            output_path=output_path,
            columns=3,
            sza_step=args.sza_step,
        )
        plt.close(fig)
        print(f"Saved {output_path}")
        print_current_statistics(
            grid, parameters, hemisphere=hemisphere, sza_step=args.sza_step
        )


if __name__ == "__main__":
    main()
