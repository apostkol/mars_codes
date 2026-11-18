from __future__ import annotations

"""Plotting helpers for Mars current-density maps."""

import os

import numpy as np
from matplotlib import pyplot as plt

plt.ioff()

__author__ = "Apostolos Kolokotronis, David Andrews"
__email__ = "apostolos.kolokotronis@irf.se, david.andrews@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "1.0.0"
__status__ = "Prototype"

__all__ = [
    "mollweide_maps",
    "mollweide_comparison_maps",
    "single_mollweide_map",
    "cartesian_plane_map",
    "FigLabeler",
]

DEFAULT_CB_LABELS = {
    "JdotE": r"J$\cdot$E [W m$^{-3}$]",
    "JxB": r"J$\times$B [N m$^{-3}$]",
    "power": "Power [W]",
    "measurements_counts": "counts/bin",
    "divB_curlB": r"$\nabla B / \nabla\timesB$",
    "B*": "B [nT]",
    "J*": r"J [$nA\;m^{-2}$]",
}

# Conic surfaces of revolution about the MSE X axis.  The bow-shock values are
# from Hall et al. (2016); the magnetic pile-up boundary values are from
# Edberg et al. (2008).
DEFAULT_BOUNDARY_PARAMETERS = {
    "bow_shock": {"x0": 0.74, "L": 1.82, "eps": 1.01},
    "pileup_boundary": {"x0": 0.86, "L": 0.90, "eps": 0.92},
}

MARS_RADIUS_KM = 3396.19


def _radial_bin_lower_radius(grid, ir):
    """Return the lower radial-bin boundary in Mars radii."""
    radius = float(grid.r[ir])
    if hasattr(grid, "dr"):
        half_width = float(grid.dr) / 2
    elif len(grid.r) > 1:
        neighbours = np.delete(np.asarray(grid.r, dtype=float), ir)
        half_width = np.min(np.abs(neighbours - radius)) / 2
    else:
        half_width = 0
    return radius - half_width


def _radial_bin_annotation(grid, ir):
    """Format the lower radial-bin boundary as altitude and Mars radii."""
    lower_radius = _radial_bin_lower_radius(grid, ir)
    lower_altitude = int(np.rint((lower_radius - 1) * MARS_RADIUS_KM))
    return (
        rf"$h_{{\rm min}}={lower_altitude}\,\mathrm{{km}}$"
        "\n"
        rf"$r_{{\rm min}}={lower_radius:.2f}\,R_{{M}}$"
    )


def _format_signed_statistic(value):
    """Keep ordinary values readable without rounding small values to zero."""
    if not np.isfinite(value):
        return "nan"
    if value != 0 and (abs(value) < 0.05 or abs(value) >= 1e4):
        return f"{value:+.1e}"
    return f"{value:+.1f}"


def _add_horizontal_colorbar(
    fig,
    mappable,
    axes,
    label,
    fts,
    width_fraction=0.6,
    pad_inches=0.18,
    height_inches=0.1,
):
    """Add a colorbar without resizing its associated map axes."""
    axes = np.asarray(axes, dtype=object).ravel()
    positions = [axis.get_position() for axis in axes]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    bottom = min(position.y0 for position in positions)
    width = (right - left) * width_fraction
    colorbar_left = (left + right - width) / 2
    colorbar_height = height_inches / fig.get_figheight()
    colorbar_bottom = bottom - pad_inches / fig.get_figheight() - colorbar_height
    colorbar_axis = fig.add_axes(
        [colorbar_left, colorbar_bottom, width, colorbar_height]
    )
    colorbar = fig.colorbar(
        mappable=mappable, cax=colorbar_axis, orientation="horizontal"
    )
    if label:
        colorbar.set_label(label, fontsize=fts)
    return colorbar


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
        try:
            v = next(self.iter)
        except StopIteration:
            v = ""
        ax.annotate(v, self.xy, **self.kwargs)

    def turn_off(self):
        self.iter = iter("")


def mollweide_maps(
    grid,
    maps=["Bsmr", "Bsmth", "Bsmphi", "measurements_counts"],
    maps_names=[r"$B_{r}$", r"$B_{\theta}$", r"$B_{\phi}$", "counts/bin"],
    cb_names=[r"$B \, [nT]$", ""],
    fts=14,
    hw=3,
    hl=4,
    hal=5,
    plots_path="../plots",
    map_type_name="magnetic_field",
    save=False,
    cb_lim=None,
    r_values=None,
    color_scale="row",
    percentile=98,
    fig=None,
    ax=None,
):
    """Plot several quantities at one or more radii on Mollweide axes.

    ``color_scale`` controls whether the first three (component) maps share
    symmetric color limits by ``"panel"``, ``"row"``, ``"column"``, or
    ``"all"`` panels. Limits are based on ``percentile`` of their absolute
    values. Count maps are normalized separately from zero to their upper
    percentile; ``divB_curlB`` retains its fixed 0--1 range. Explicit
    ``cb_lim`` values override the component-map percentile calculation.
    """

    grid_maps = [getattr(grid, map_name) for map_name in maps]

    def label_panel(axis, row, column):
        column_label = ""
        while True:
            column, remainder = divmod(column, 26)
            column_label = chr(ord("a") + remainder) + column_label
            if column == 0:
                break
            column -= 1
        axis.annotate(
            f"{column_label}{row + 1}",
            xy=(0.02, 0.98),
            ha="left",
            va="top",
            xycoords="axes fraction",
            size="large",
        )

    if save:
        plots_path_date = os.path.abspath(
            f"{plots_path}/{grid.name}_dr_{grid.dr}_dth_{grid.dth_deg}"
            f"_dphi_{grid.dphi_deg}_{map_type_name}",
        )
        if not os.path.isdir(plots_path_date):
            os.makedirs(plots_path_date)

    # Indices of radial ranges to plot
    if isinstance(r_values, str):
        if r_values.upper() == "ALL":
            r_values = [r for r in grid.r]
            inx = list(range(len(grid.r)))
        else:
            raise ValueError(f"Don't know what to do with {r_values=}")

    if r_values is None:
        r_values = np.linspace(grid.r[0], grid.r[-1], 5)

    inx = [np.argmin(np.abs(grid.r - r)) for r in r_values]
    # Sorted not available in 2.2 numpy
    # inx = np.unique(inx, sorted=False)
    new_inx = []
    for i in inx:
        if i not in new_inx:
            new_inx.append(i)
    inx = new_inx

    valid_color_scales = {"panel", "row", "column", "all"}
    if color_scale not in valid_color_scales:
        raise ValueError(
            f"`color_scale` must be one of {sorted(valid_color_scales)}; "
            f"got {color_scale!r}"
        )

    component_limits = {}
    for e_ir, ir in enumerate(inx):
        for i in range(3):
            if cb_lim is not None:
                component_limits[e_ir, i] = cb_lim
                continue
            if color_scale == "panel":
                values = grid_maps[i][:, :, ir]
            elif color_scale == "row":
                values = np.stack([grid_map[:, :, ir] for grid_map in grid_maps[:3]])
            elif color_scale == "column":
                values = grid_maps[i][:, :, inx]
            else:
                values = np.stack([grid_map[:, :, inx] for grid_map in grid_maps[:3]])
            limit = np.nanpercentile(np.abs(values), percentile)
            component_limits[e_ir, i] = (-limit, limit)

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(
                nrows=len(inx),
                ncols=len(maps),
                figsize=(5 * len(maps), len(inx) * 3.2),
                subplot_kw={"projection": "mollweide"},
                gridspec_kw={"wspace": 0.08, "hspace": 0.18},
            )
        else:
            ax = fig.subplots(
                nrows=len(inx),
                ncols=len(maps),
                subplot_kw={"projection": "mollweide"},
                gridspec_kw={"wspace": 0.08, "hspace": 0.18},
            )
    else:
        if ax is not None:
            ax = np.asarray(ax, dtype=object)

            if ax.ndim == 0:
                ax = ax.reshape(1, 1)
            elif ax.ndim == 1:
                ax = ax.reshape(len(inx), len(maps))

            if ax.ndim != 2:
                raise ValueError(
                    f"`ax` must be a 2D array of Axes; got shape {ax.shape}"
                )

            fig = ax.flat[0].figure

    for e_ir, ir in enumerate(inx):
        sliced_maps = [grid_map[:, :, ir] for grid_map in grid_maps]
        colorbar_pad = 0.32

        for i in range(3):
            vmin, vmax = component_limits[e_ir, i]
            single_mollweide_map(
                grid,
                sliced_maps[i],
                arrows=(sliced_maps[2], sliced_maps[1]),
                ax=ax[e_ir, i],
                ir=ir,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                alpha=1,
                title=maps_names[i] if e_ir == 0 else "",
                fts=fts,
                hw=hw,
                hl=hl,
                hal=hal,
                add_colorbar=False,
                show_radius=False,
            )
            if e_ir < len(inx) - 1:
                ax[e_ir, i].set_xlabel("")
                ax[e_ir, i].tick_params(axis="x", labelbottom=False)
            if i > 0:
                ax[e_ir, i].set_ylabel("")
                ax[e_ir, i].tick_params(axis="y", labelleft=False)
            label_panel(ax[e_ir, i], e_ir, i)

        if color_scale in {"row", "all"} or cb_lim is not None:
            _add_horizontal_colorbar(
                fig,
                ax[e_ir, 2]._mars_currents_cmesh,
                ax[e_ir, :3],
                cb_names[0] if e_ir == len(inx) - 1 else None,
                fts,
                width_fraction=0.4,
                pad_inches=colorbar_pad,
            )
        else:
            for i in range(3):
                _add_horizontal_colorbar(
                    fig,
                    ax[e_ir, i]._mars_currents_cmesh,
                    ax[e_ir, i],
                    cb_names[0] if e_ir == len(inx) - 1 else None,
                    fts,
                    width_fraction=0.8,
                    pad_inches=colorbar_pad,
                )

        if len(maps) == 4:
            if maps[3] in ["divB_curlB"]:
                vmin = 0
                vmax = 1
            elif "count" in maps[3].lower():
                vmin = 0
                if color_scale in {"column", "all"}:
                    count_values = grid_maps[3][:, :, inx]
                else:
                    count_values = sliced_maps[3]
                vmax = np.nanpercentile(count_values, percentile)
            else:
                vmin = None
                vmax = None

            single_mollweide_map(
                grid,
                sliced_maps[3],
                ax=ax[e_ir, 3],
                ir=ir,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                title=maps_names[3] if e_ir == 0 else "",
                fts=fts,
                add_colorbar=False,
                show_radius=False,
                note_medians=None,
            )
            if e_ir < len(inx) - 1:
                ax[e_ir, 3].set_xlabel("")
                ax[e_ir, 3].tick_params(axis="x", labelbottom=False)
            ax[e_ir, 3].set_ylabel("")
            ax[e_ir, 3].tick_params(axis="y", labelleft=False)
            label_panel(ax[e_ir, 3], e_ir, 3)

            _add_horizontal_colorbar(
                fig,
                ax[e_ir, 3]._mars_currents_cmesh,
                ax[e_ir, 3],
                cb_names[1] if e_ir == len(inx) - 1 else None,
                fts,
                width_fraction=1.0,
                pad_inches=colorbar_pad,
            )

        ax[e_ir, 0].annotate(
            _radial_bin_annotation(grid, ir),
            xy=(0.0, 0.00),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=fts * 0.8,
        )
        dist = format(grid.r[ir], ".2f")

    ax[0, 1].annotate(
        r"$\vec{B}_{\rm IMF}$",
        xy=(0.6, 1.2),
        xytext=(0.3, 1.2),
        xycoords="axes fraction",
        arrowprops=dict(facecolor="red", width=3),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fts,
    )

    if save:
        fig.savefig(
            f"{plots_path_date}/{map_type_name}_r_{dist}_Rm.png",
            bbox_inches="tight",
        )

    return fig, ax


def mollweide_comparison_maps(
    grids,
    parameter,
    r_values,
    column_names=None,
    arrows=None,
    norm=None,
    cb_lim=None,
    cb_label=None,
    color_scale="all",
    percentile=98,
    note_medians="signed",
    cmap="RdBu_r",
    fts=14,
    show_IMF=True,
    fig=None,
    ax=None,
):
    """Compare one quantity across grids, with radii as rows and grids as columns.

    This is the multi-grid counterpart to :func:`mollweide_maps`. The existing
    single-grid function retains its component-map and statistics-column
    semantics, while this function gives every column its own grid.
    """
    grids = list(grids)
    r_values = list(r_values)
    if not grids:
        raise ValueError("`grids` must contain at least one Grid object")
    if column_names is None:
        column_names = [getattr(grid, "name", "") for grid in grids]
    if len(column_names) != len(grids):
        raise ValueError("`column_names` must have one entry per grid")
    valid_color_scales = {"panel", "row", "column", "all"}
    if color_scale not in valid_color_scales:
        raise ValueError(
            f"`color_scale` must be one of {sorted(valid_color_scales)}; "
            f"got {color_scale!r}"
        )

    radial_indices = [
        [int(np.argmin(np.abs(grid.r - radius))) for grid in grids]
        for radius in r_values
    ]
    panel_data = np.empty((len(r_values), len(grids)), dtype=object)
    for row, indices in enumerate(radial_indices):
        for column, (grid, ir) in enumerate(zip(grids, indices)):
            panel_data[row, column] = getattr(grid, parameter)[:, :, ir]

    panel_limits = {}
    for row in range(len(r_values)):
        for column in range(len(grids)):
            if norm is not None:
                continue
            if cb_lim is not None:
                panel_limits[row, column] = cb_lim
            elif color_scale == "panel":
                values = panel_data[row, column]
            elif color_scale == "row":
                values = np.stack(panel_data[row, :].tolist())
            elif color_scale == "column":
                values = np.stack(panel_data[:, column].tolist())
            else:
                values = np.stack(panel_data.ravel().tolist())
            if cb_lim is None:
                limit = np.nanpercentile(np.abs(values), percentile)
                panel_limits[row, column] = (-limit, limit)

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(
                len(r_values),
                len(grids),
                figsize=(5 * len(grids), 3.7 * len(r_values)),
                subplot_kw={"projection": "mollweide"},
                gridspec_kw={"wspace": 0.08, "hspace": 0.22},
            )
        else:
            ax = fig.subplots(
                len(r_values),
                len(grids),
                subplot_kw={"projection": "mollweide"},
                gridspec_kw={"wspace": 0.08, "hspace": 0.22},
            )
    else:
        ax = np.asarray(ax, dtype=object)
        fig = ax.flat[0].figure
    ax = np.asarray(ax, dtype=object).reshape(len(r_values), len(grids))

    for row, indices in enumerate(radial_indices):
        for column, (grid, ir) in enumerate(zip(grids, indices)):
            plot_kwargs = {}
            if norm is None:
                plot_kwargs["vmin"], plot_kwargs["vmax"] = panel_limits[row, column]
            single_mollweide_map(
                grid,
                parameter,
                ir=ir,
                arrows=arrows,
                norm=norm,
                cmap=cmap,
                ax=ax[row, column],
                title=column_names[column] if row == 0 else "",
                fts=fts,
                add_colorbar=False,
                show_radius=False,
                note_medians=note_medians,
                **plot_kwargs,
            )
            if row < len(r_values) - 1:
                ax[row, column].set_xlabel("")
                ax[row, column].tick_params(axis="x", labelbottom=False)
            if column > 0:
                ax[row, column].set_ylabel("")
                ax[row, column].tick_params(axis="y", labelleft=False)
            label = f"{chr(ord('a') + column)}{row + 1}"
            ax[row, column].annotate(
                label,
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                ha="left",
                va="top",
                size="large",
            )
        first_grid = grids[0]
        first_ir = indices[0]
        ax[row, 0].annotate(
            _radial_bin_annotation(first_grid, first_ir),
            xy=(0.0, 0.0),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=fts * 0.8,
        )

    if cb_label is None:
        cb_label = next(
            (label for key, label in DEFAULT_CB_LABELS.items() if key in parameter),
            parameter,
        )

    def add_colorbar(axes, mappable):
        _add_horizontal_colorbar(
            fig,
            mappable,
            axes,
            cb_label,
            fts,
            width_fraction=0.6,
            pad_inches=0.32,
        )

    if norm is not None or cb_lim is not None or color_scale == "all":
        add_colorbar(ax, ax[-1, -1]._mars_currents_cmesh)
    elif color_scale == "row":
        for row in range(len(r_values)):
            add_colorbar(ax[row, :], ax[row, -1]._mars_currents_cmesh)
    elif color_scale == "column":
        for column in range(len(grids)):
            add_colorbar(ax[:, column], ax[-1, column]._mars_currents_cmesh)
    else:
        for row in range(len(r_values)):
            for column in range(len(grids)):
                add_colorbar(
                    ax[row, column],
                    ax[row, column]._mars_currents_cmesh,
                )

    if show_IMF:
        ax[0, min(1, len(grids) - 1)].annotate(
            r"$\vec{B}_{\rm IMF}$",
            xy=(0.6, 1.2),
            xytext=(0.3, 1.2),
            xycoords="axes fraction",
            arrowprops=dict(facecolor="red", width=3),
            ha="center",
            va="center",
            fontsize=fts,
        )

    return fig, ax


def single_mollweide_map(
    grid,
    parameter,
    arrows=None,
    norm=None,
    cmap=None,
    ax=None,
    fig=None,
    show_IMF=False,
    title_prefix=None,
    note_medians="signed",
    **kwargs,
):
    """Plot one grid quantity on a Mollweide map.

    Parameters
    ----------
    grid
        Grid object.
    parameter : str or array-like
        Name of a grid attribute, or an already-loaded 2D/3D array to plot. If
        3D, a single radial slice is selected using ``ir`` or ``r_value`` from
        ``kwargs``.
    arrows : {None, "Bsm", "B", "J"} or pair of array-like, optional
        Overlay horizontal vector arrows. ``"B"`` uses ``Bphi``/``Bth`` and
        ``"J"`` uses ``Jphi``/``Jth``. A pair supplies the longitude and theta
        components directly.
    norm, cmap, ax, fig
        Passed to Matplotlib plotting helpers. If ``ax``/``fig`` are not given,
        a new Mollweide axis is created.
    show_IMF : bool, optional
        If true, add the IMF annotation used by :func:`mollweide_maps`.
    title_prefix:
        Prepended to the title (e.g, an axis label a,b,...)
    note_medians:
        One of ``"signed"``, ``"unsigned"``, or None. ``"signed"`` reports
        the dayside means of the positive and negative values together with
        the percentage of finite dayside bins having each sign. ``"unsigned"``
        reports the median absolute value over the full map.
    **kwargs
        Recognized keys include ``ir``, ``r_value``, ``title``, ``cb_label``,
        ``fts``, ``hw``, ``hl``, ``hal``, ``vmin``, ``vmax``, ``alpha``,
        ``add_colorbar``, and ``quiver_kwargs``. Remaining keys are passed to
        ``pcolormesh``.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    fts = kwargs.pop("fts", 14)
    hw = kwargs.pop("hw", 3)
    hl = kwargs.pop("hl", 4)
    hal = kwargs.pop("hal", 5)
    ir = kwargs.pop("ir", None)
    r_value = kwargs.pop("r_value", None)
    title = kwargs.pop("title", None)
    cb_label = kwargs.pop("cb_label", None)
    add_colorbar = kwargs.pop("add_colorbar", True)
    show_radius = kwargs.pop("show_radius", True)
    alpha = kwargs.pop("alpha", 1)
    quiver_kwargs = kwargs.pop("quiver_kwargs", {})

    y_ticks_deg = [r"$45 \degree $ S", "0", r"$45 \degree $ N"]
    y_ticks = np.deg2rad([-45, 0, 45])

    x_ticks_deg = [-135, -90, -45, 0, 45, 90, 135]
    x_ticks = np.deg2rad(x_ticks_deg)

    if add_colorbar and cb_label is None and isinstance(parameter, str):
        for c, v in DEFAULT_CB_LABELS.items():
            if c in parameter:
                cb_label = v
                break
            if "*" in c:
                if parameter[0] == c[0]:
                    cb_label = v
                    break
        else:
            cb_label = "???"

    th_mol = -(grid.th_2d - np.pi / 2)
    phi_mol = grid.phi_2d + (np.sign(np.pi - grid.phi_2d) - 1) * np.pi

    if isinstance(parameter, str):
        parameter_name = parameter
        grid_map = getattr(grid, parameter)
    else:
        parameter_name = None
        grid_map = np.asarray(parameter)

    def _radial_index():
        if ir is not None:
            return int(ir)
        if r_value is not None:
            return int(np.argmin(np.abs(grid.r - r_value)))
        if np.ndim(grid_map) == 3:
            return 0
        return None

    selected_ir = _radial_index()

    def _slice(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            if selected_ir is None:
                raise ValueError("A radial index is needed for 3D arrays.")
            return arr[:, :, selected_ir]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Expected a 2D or 3D array; got shape {arr.shape}")

    map_2d = _slice(grid_map)
    phi_2d = _slice(phi_mol)
    th_2d = _slice(th_mol)

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(
                figsize=kwargs.pop("figsize", (6, 5)),
                subplot_kw={"projection": "mollweide"},
            )
        else:
            ax = fig.add_subplot(111, projection="mollweide")
    else:
        fig = ax.figure

    if cmap is None:
        cmap = kwargs.pop("default_cmap", "RdBu_r")

    pcolormesh_kwargs = dict(kwargs)
    if norm is not None:
        pcolormesh_kwargs["norm"] = norm
    else:
        if "vmin" not in pcolormesh_kwargs and "vmax" not in pcolormesh_kwargs:
            finite_values = map_2d[np.isfinite(map_2d)]
            if finite_values.size:
                limit = np.nanmax(np.abs(finite_values))
                pcolormesh_kwargs.setdefault("vmin", -limit)
                pcolormesh_kwargs.setdefault("vmax", limit)
    pcolormesh_kwargs.setdefault("alpha", alpha)

    cmesh = ax.pcolormesh(phi_2d, th_2d, map_2d, cmap=cmap, **pcolormesh_kwargs)
    ax._mars_currents_cmesh = cmesh

    if note_medians is not None:
        note_kwargs = dict(
            xy=(0.99, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=fts * 0.6,
            style="italic",
        )
        if note_medians == "signed":
            dayside = np.cos(phi_2d) > 0
            finite_dayside = dayside & np.isfinite(map_2d)
            dayside_values = map_2d[finite_dayside]
            positive = dayside_values[dayside_values > 0]
            negative = dayside_values[dayside_values < 0]
            positive_mean = np.mean(positive) if positive.size else np.nan
            negative_mean = np.mean(negative) if negative.size else np.nan
            if dayside_values.size:
                positive_fraction = 100 * positive.size / dayside_values.size
                negative_fraction = 100 * negative.size / dayside_values.size
            else:
                positive_fraction = np.nan
                negative_fraction = np.nan
            ax.annotate(
                f"{_format_signed_statistic(positive_mean)} "
                f"({positive_fraction:.0f}%)\n"
                f"{_format_signed_statistic(negative_mean)} "
                f"({negative_fraction:.0f}%)",
                **note_kwargs,
            )
        elif note_medians == "unsigned":
            median = np.nanmedian(np.abs(map_2d))
            ax.annotate(
                f"{median:.3g}",
                **note_kwargs,
            )
        else:
            raise ValueError(
                f"note_medians must be one of 'signed', 'unsigned', or None; got {note_medians}"
            )

    if arrows is not None:
        if isinstance(arrows, (tuple, list)) and len(arrows) == 2:
            arrows_key = "arrays"
            u = _slice(arrows[0])
            v = -_slice(arrows[1])
        else:
            arrows_key = str(arrows)
            arrow_components = {
                "B": ("Bphi", "Bth"),
                "Bsm": ("Bsmphi", "Bsmth"),
                "J": ("Jphi", "Jth"),
            }
            if arrows_key not in arrow_components:
                raise ValueError(
                    '`arrows` must be None, "B", "Bsm", "J", or an array pair'
                )
            phi_name, th_name = arrow_components[arrows_key]
            if not hasattr(grid, phi_name) or not hasattr(grid, th_name):
                raise AttributeError(
                    f"Could not find horizontal components for {arrows_key}: "
                    f"{phi_name}/{th_name}"
                )
            u = _slice(getattr(grid, phi_name))
            v = -_slice(getattr(grid, th_name))

        q_kwargs = {
            "headwidth": hw,
            "headaxislength": hal,
            "headlength": hl,
            "color": "purple" if arrows_key == "J" else "k",
        }
        q_kwargs.update(quiver_kwargs)
        ax.quiver(phi_2d, th_2d, u, v, **q_kwargs)

    if title is None:
        title = parameter_name if parameter_name is not None else ""
    if title_prefix is None:
        title_prefix = ""
    ax.set_title(title_prefix + title, fontsize=fts)
    ax.set_xlabel(r"$Lon_{MSE}\,  [\degree] $", fontsize=fts)
    ax.set_ylabel(r"$Lat_{MSE} \, [\degree]$", fontsize=fts)
    ax.set_xticks(ticks=x_ticks, labels=x_ticks_deg)
    ax.set_yticks(ticks=y_ticks, labels=y_ticks_deg)

    ax.grid(color="black", linestyle="--", alpha=0.5)
    gridlines = ax.xaxis.get_gridlines()
    if len(gridlines) > 5:
        gridlines[1].set_linestyle("--")
        gridlines[1].set_color("black")
        gridlines[1].set_linewidth(3.5)
        gridlines[5].set_linestyle("--")
        gridlines[5].set_color("black")
        gridlines[5].set_linewidth(3.5)

    ax.set_facecolor("0.6")

    if add_colorbar:
        cb = fig.colorbar(mappable=cmesh, ax=ax, orientation="horizontal", shrink=0.8)
        if cb_label is not None:
            cb.set_label(label=cb_label, fontsize=fts)

    if show_IMF:
        ax.annotate(
            r"$\vec{B}_{\rm IMF}$",
            xy=(0.58, 1.08),
            xytext=(0.42, 1.08),
            xycoords="axes fraction",
            arrowprops=dict(facecolor="red", width=3),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fts,
        )

    if show_radius and selected_ir is not None and hasattr(grid, "r"):
        ax.annotate(
            _radial_bin_annotation(grid, selected_ir),
            xy=(0.99, 0.00),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=fts * 0.8,
        )

    return fig, ax


def cartesian_plane_map(
    grid,
    parameter,
    plane_normal=(0, 0, 1),
    plane_point=(0, 0, 0),
    u_axis=None,
    extent=None,
    n=200,
    arrows=None,
    cmap=None,
    norm=None,
    ax=None,
    fig=None,
    plot_boundaries=True,
    **kwargs,
):
    """Plot a nearest-neighbour slice through a spherical grid on a Cartesian plane.

    The plane is defined by:

        dot((x, y, z) - plane_point, plane_normal) = 0

    A regular 2D Cartesian mesh is built inside that plane, converted to
    spherical coordinates, then nearest-neighbour sampled from grid.parameter.

    Parameters
    ----------
    grid
        Object with spherical coordinates and data arrays. Expected:
        grid.r, grid.th, grid.phi, or 3D-compatible data with shape
        (nth, nphi, nr), matching the existing plotting helpers.
    parameter : str or array-like
        Grid attribute name or array to sample.
    plane_normal : 3-tuple
        Normal vector of the slicing plane.
    plane_point : 3-tuple
        One point through which the plane passes, in the same units as grid.r.
    u_axis : 3-tuple, optional
        Preferred horizontal axis inside the plane. If omitted, one is chosen.
    extent : float or (umin, umax, vmin, vmax), optional
        Plane plotting limits. If omitted, uses +/- max(grid.r).
    n : int
        Number of samples along each plane direction.
    arrows : {None, "B", "J"}, optional
        Overlay projected vector arrows from the grid's Cartesian ``Bx/By/Bz``
        or ``Jx/Jy/Jz`` products.
    quiver_kwargs : dict, optional
        Additional keyword arguments passed to ``matplotlib.axes.Axes.quiver``.
        Standard quiver keyword arguments such as ``scale``, ``scale_units``,
        ``angles``, ``width``, ``pivot``, and ``color`` may also be passed
        directly to this function.
    arrow_length_transform : {None, "sqrt"}, optional
        Optional transform applied to projected arrow magnitudes before passing
        them to Matplotlib. ``"sqrt"`` preserves direction while replacing each
        nonzero projected magnitude with its square root. Matplotlib ``quiver``
        scaling is still controlled separately through standard quiver kwargs.
    pole_theta_bins : int, optional
        Number of theta bins to mask at each spherical-polar coordinate pole.
        Defaults to 1. Pass 0 to keep all theta bins.
    z_up_specificity : float, optional
        Minimum length of the original +Z axis projection into the slice plane
        for orienting +Z upward in the page. Defaults to 0.95.
    axis_label_specificity : float, optional
        Minimum alignment with an original Cartesian axis for replacing generic
        u/v axis labels with cardinal-axis labels. Defaults to 0.95.
    u_label, v_label : str, optional
        Explicit labels for the plotted u and v axes. If omitted, labels are
        inferred from the selected plane basis.
    plot_boundaries : bool, optional
        Plot the intersections of the slice plane with conic bow-shock and
        magnetic pile-up boundary surfaces. Defaults to ``False``.
        ``bow_shock_pars`` and ``pileup_boundary_pars`` may be supplied as
        dictionaries containing ``x0``, ``L``, and ``eps`` to override the
        default Hall et al. (2016) and Edberg et al. (2008) models.
        Styling can be overridden with ``bow_shock_kwargs`` and
        ``pileup_boundary_kwargs``.
    """

    fts = kwargs.get("fts", 14)
    alpha = kwargs.get("alpha", 1)
    shading = kwargs.get("shading", "auto")
    colorbar = kwargs.get("colorbar", True)
    cb_label = kwargs.get("cb_label", parameter if isinstance(parameter, str) else "")
    title = kwargs.get("title", parameter if isinstance(parameter, str) else "")
    pole_theta_bins = kwargs.get("pole_theta_bins", 1)
    arrow_step = kwargs.get("arrow_step", max(1, n // 35))
    arrow_length_transform = kwargs.get("arrow_length_transform", None)
    hw = kwargs.get("hw", 3)
    hl = kwargs.get("hl", 4)
    hal = kwargs.get("hal", 5)
    z_up_specificity = kwargs.get("z_up_specificity", 0.95)
    axis_label_specificity = kwargs.get("axis_label_specificity", 0.95)

    # ---------- small local helpers ----------

    def _as_array(name_or_array):
        if isinstance(name_or_array, str):
            return getattr(grid, name_or_array)
        return np.asarray(name_or_array)

    def _nearest_index(values, query):
        """Nearest index using ordinary linear distance."""
        values = np.asarray(values)
        return np.abs(values[:, None, None] - query[None, :, :]).argmin(axis=0)

    def _nearest_phi_index(phi_values, query_phi):
        """Nearest phi index using circular distance.

        Works whether phi_values are stored as:
            [-pi, pi]
        or:
            [0, 2*pi]
        """

        phi_values = np.asarray(phi_values)

        delta = (phi_values[:, None, None] - query_phi[None, :, :] + np.pi) % (
            2 * np.pi
        ) - np.pi

        return np.abs(delta).argmin(axis=0)

    def _normalise(vec):
        vec = np.asarray(vec, dtype=float)
        length = np.linalg.norm(vec)
        if length == 0:
            raise ValueError("Zero-length vector is not allowed.")
        return vec / length

    def _axis_label(axis_name, basis_vec):
        label = f"Plane coordinate {axis_name}"
        cardinal_axes = {
            "X": np.array([1.0, 0.0, 0.0]),
            "Y": np.array([0.0, 1.0, 0.0]),
            "Z": np.array([0.0, 0.0, 1.0]),
        }

        for cardinal_label, cardinal_vec in cardinal_axes.items():
            alignment = np.dot(basis_vec, cardinal_vec)
            if alignment >= axis_label_specificity:
                return f"MSE {cardinal_label}"
            if alignment <= -axis_label_specificity:
                return f"MSE -{cardinal_label}"

        return label

    def _transform_arrow_lengths(u_comp, v_comp):
        if arrow_length_transform is None:
            return u_comp, v_comp

        transform = str(arrow_length_transform).lower()
        if transform in ("none", "linear"):
            return u_comp, v_comp
        if transform != "sqrt":
            raise ValueError('`arrow_length_transform` must be None or "sqrt".')

        magnitude = np.hypot(u_comp, v_comp)
        transformed_magnitude = np.sqrt(magnitude)
        factor = np.divide(
            transformed_magnitude,
            magnitude,
            out=np.zeros_like(magnitude),
            where=magnitude > 0,
        )
        return u_comp * factor, v_comp * factor

    # ---------- define the plane basis ----------

    normal = _normalise(plane_normal)
    point = np.asarray(plane_point, dtype=float)

    if u_axis is None:
        # Pick a vector that is unlikely to be parallel to the normal.
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, normal)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        u_axis = trial

    u_axis = np.asarray(u_axis, dtype=float)
    z_axis = np.array([0.0, 0.0, 1.0])
    z_projection = z_axis - np.dot(z_axis, normal) * normal
    z_projection_length = np.linalg.norm(z_projection)

    if z_projection_length >= z_up_specificity:
        v_hat = z_projection / z_projection_length
        u_hat = _normalise(np.cross(v_hat, normal))

        if np.dot(u_hat, u_axis) < 0:
            u_hat = -u_hat
    else:
        u_hat = u_axis - np.dot(u_axis, normal) * normal
        u_hat = _normalise(u_hat)
        v_hat = _normalise(np.cross(normal, u_hat))

    # ---------- make a regular mesh in the plane ----------

    if extent is None:
        extent = float(np.nanmax(grid.r))

    if np.isscalar(extent):
        umin, umax, vmin, vmax = -extent, extent, -extent, extent
    else:
        umin, umax, vmin, vmax = extent

    u = np.linspace(umin, umax, n)
    v = np.linspace(vmin, vmax, n)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    xyz = (
        point[None, None, :]
        + uu[:, :, None] * u_hat[None, None, :]
        + vv[:, :, None] * v_hat[None, None, :]
    )

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos(np.divide(z, r, out=np.full_like(r, np.nan), where=r != 0))

    # No clamping here, since _nearest_phi_index will be used.
    # phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    phi = np.arctan2(y, x)

    inside = (r >= np.nanmin(grid.r)) & (r <= np.nanmax(grid.r))

    # ---------- nearest-neighbour spherical lookup ----------

    data = _as_array(parameter)

    ir = _nearest_index(grid.r, r)

    if hasattr(grid, "th"):
        th_values = np.asarray(grid.th)
        ith = _nearest_index(grid.th, th)
    else:
        # Fall back to the first column/radial slice of th_2d.
        th_values = np.asarray(grid.th_2d[:, 0, 0])
        ith = _nearest_index(th_values, th)

    if hasattr(grid, "phi"):
        iphi = _nearest_phi_index(grid.phi, phi)
    else:
        phi_values = np.asarray(grid.phi_2d[:, 0, 0])
        iphi = _nearest_phi_index(phi_values, phi)

    if pole_theta_bins:
        nth = len(th_values)
        away_from_poles = (ith >= pole_theta_bins) & (ith < nth - pole_theta_bins)
    else:
        away_from_poles = np.ones_like(inside, dtype=bool)

    valid = inside & away_from_poles

    sampled = data[iphi, ith, ir]
    sampled = np.where(valid, sampled, np.nan)

    # ---------- figure / axes ----------

    if norm is None:
        finite_sampled = sampled[np.isfinite(sampled)]
        if finite_sampled.size:
            mv = np.nanpercentile(np.abs(finite_sampled), 95)
        else:
            mv = 1
        norm = plt.Normalize(-mv, mv)

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 6)))
        else:
            ax = fig.subplots()
    else:
        fig = ax.figure

    ax._mars_currents_plane_basis = {
        "u_hat": u_hat.copy(),
        "v_hat": v_hat.copy(),
        "normal": normal.copy(),
    }
    u_label = kwargs.get("u_label", _axis_label("u", u_hat))
    v_label = kwargs.get("v_label", _axis_label("v", v_hat))

    mesh = ax.pcolormesh(
        uu,
        vv,
        sampled,
        cmap=cmap,
        norm=norm,
        shading=shading,
        alpha=alpha,
        vmin=kwargs.get("vmin", None),
        vmax=kwargs.get("vmax", None),
    )

    # ---------- bow shock and magnetic pile-up boundary ----------
    #
    # A conic r = L / (1 + eps*cos(theta)), with its focus displaced
    # to (x0, 0, 0), has the implicit form
    #
    #     sqrt((x-x0)^2 + y^2 + z^2) + eps*(x-x0) - L = 0.
    #
    # Evaluating that exact function on the existing plane mesh and drawing
    # only its zero contour gives the plane/surface intersection for arbitrary
    # plane orientations without special-case analytic geometry.

    if plot_boundaries:
        boundary_specs = (
            (
                "bow_shock",
                kwargs.get("bow_shock_pars", DEFAULT_BOUNDARY_PARAMETERS["bow_shock"]),
                {"colors": "black", "linestyles": "--", "linewidths": 1.8},
                "bow_shock_kwargs",
            ),
            (
                "pileup_boundary",
                kwargs.get(
                    "pileup_boundary_pars",
                    DEFAULT_BOUNDARY_PARAMETERS["pileup_boundary"],
                ),
                {"colors": "black", "linestyles": ":", "linewidths": 1.8},
                "pileup_boundary_kwargs",
            ),
        )
        ax._mars_currents_boundaries = {}

        for boundary_name, pars, style, style_kwarg in boundary_specs:
            missing = {"x0", "L", "eps"}.difference(pars)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    f"`{boundary_name}` parameters are missing: {missing_names}"
                )

            x_from_focus = x - float(pars["x0"])
            implicit_conic = (
                np.sqrt(x_from_focus**2 + y**2 + z**2)
                + float(pars["eps"]) * x_from_focus
                - float(pars["L"])
            )
            style.update(kwargs.get(style_kwarg, {}))
            # Accept the familiar Line2D spellings as well as contour's
            # plural keyword names.
            for singular, plural in (
                ("color", "colors"),
                ("linestyle", "linestyles"),
                ("linewidth", "linewidths"),
            ):
                if singular in style:
                    style[plural] = style.pop(singular)

            contour = None
            finite_values = implicit_conic[np.isfinite(implicit_conic)]
            if (
                finite_values.size
                and np.nanmin(finite_values) <= 0
                and np.nanmax(finite_values) >= 0
            ):
                contour = ax.contour(uu, vv, implicit_conic, levels=[0.0], **style)

            ax._mars_currents_boundaries[boundary_name] = {
                "contour": contour,
                "parameters": dict(pars),
                "implicit_values": implicit_conic,
            }

    # ---------- optional vector arrows ----------
    #
    # Sample Cartesian components directly, then project them onto the two
    # in-plane basis vectors u_hat and v_hat. Cartesian products are regenerated
    # when a Grid is loaded, so using one required representation avoids parallel
    # projection paths and keeps a cell's vector fixed across its raster pixels.

    if arrows is not None:
        arrows = arrows.upper()

        if arrows == "B":
            cartesian_names = ("Bx", "By", "Bz")
        elif arrows == "J":
            cartesian_names = ("Jx", "Jy", "Jz")
        else:
            raise ValueError('`arrows` must be None, "B", or "J".')

        missing = [name for name in cartesian_names if not hasattr(grid, name)]
        if missing:
            missing_names = ", ".join(missing)
            raise AttributeError(
                f"Cartesian arrow components are missing: {missing_names}. "
                "Regenerate the grid's Cartesian vector products before plotting."
            )

        cartesian_arrays = [np.asarray(getattr(grid, name)) for name in cartesian_names]
        bad_shapes = {
            name: values.shape
            for name, values in zip(cartesian_names, cartesian_arrays)
            if values.shape != data.shape
        }
        if bad_shapes:
            details = ", ".join(f"{name}={shape}" for name, shape in bad_shapes.items())
            raise ValueError(
                f"Cartesian arrow components must match data shape {data.shape}; {details}"
            )

        vx, vy, vz = (values[iphi, ith, ir] for values in cartesian_arrays)

        vec = np.stack([vx, vy, vz], axis=-1)

        u_comp = np.sum(vec * u_hat[None, None, :], axis=-1)
        v_comp = np.sum(vec * v_hat[None, None, :], axis=-1)

        u_comp = np.where(valid, u_comp, np.nan)
        v_comp = np.where(valid, v_comp, np.nan)
        u_comp, v_comp = _transform_arrow_lengths(u_comp, v_comp)

        sl = (slice(None, None, arrow_step), slice(None, None, arrow_step))
        quiver_kwargs = {
            "headwidth": hw,
            "headaxislength": hal,
            "headlength": hl,
        }
        quiver_passthrough = (
            "angles",
            "scale",
            "scale_units",
            "units",
            "width",
            "pivot",
            "color",
            "headwidth",
            "headlength",
            "headaxislength",
            "cmap",
            "norm",
            "minshaft",
            "minlength",
        )
        quiver_kwargs.update(
            {name: kwargs[name] for name in quiver_passthrough if name in kwargs}
        )
        quiver_kwargs.update(kwargs.get("quiver_kwargs", {}))
        ax._mars_currents_arrow_components = {
            "u": u_comp[sl].copy(),
            "v": v_comp[sl].copy(),
            "source": "cartesian",
            "arrow_length_transform": arrow_length_transform,
            "quiver_kwargs": quiver_kwargs.copy(),
        }
        ax._mars_currents_quiver = ax.quiver(
            uu[sl],
            vv[sl],
            u_comp[sl],
            v_comp[sl],
            **quiver_kwargs,
        )

    # ---------- projected Cartesian basis triad ----------
    #
    # Shows where the original Cartesian +X, +Y, +Z directions point
    # after projection into this slice plane.
    #
    # Since the plot axes are the plane coordinates (u, v), we project each
    # Cartesian unit vector onto u_hat and v_hat.

    if kwargs.get("show_xyz_triad", True):
        triad_origin = kwargs.get("triad_origin", (0.12, 0.88))
        triad_scale = kwargs.get("triad_scale", 0.10)
        triad_fontsize = kwargs.get("triad_fontsize", fts - 2)

        basis_vectors = {
            "X": np.array([1.0, 0.0, 0.0]),
            "Y": np.array([0.0, 1.0, 0.0]),
            "Z": np.array([0.0, 0.0, 1.0]),
        }

        for label, basis_vec in basis_vectors.items():
            # Components of the original Cartesian vector in the slice-plane basis.
            du = np.dot(basis_vec, u_hat)
            dv = np.dot(basis_vec, v_hat)

            # If a basis vector is normal to the plane, its projection is zero.
            # Draw it as a dot/cross instead of an invisible arrow.
            projected_length = np.hypot(du, dv)

            if projected_length < 1e-12:
                ax.plot(
                    triad_origin[0],
                    triad_origin[1],
                    marker="x",
                    markersize=6,
                    transform=ax.transAxes,
                    clip_on=False,
                )
                ax.text(
                    triad_origin[0] + 0.015,
                    triad_origin[1],
                    label,
                    transform=ax.transAxes,
                    fontsize=triad_fontsize,
                    ha="left",
                    va="center",
                    clip_on=False,
                )
                continue

            du /= projected_length
            dv /= projected_length

            ax.annotate(
                "",
                xy=(
                    triad_origin[0] + triad_scale * du,
                    triad_origin[1] + triad_scale * dv,
                ),
                xytext=triad_origin,
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=1.8,
                ),
                clip_on=False,
            )

            ax.text(
                triad_origin[0] + 1.18 * triad_scale * du,
                triad_origin[1] + 1.18 * triad_scale * dv,
                label,
                transform=ax.transAxes,
                fontsize=triad_fontsize,
                ha="center",
                va="center",
                clip_on=False,
            )

    # ---------- formatting ----------

    ax.set_title(title, fontsize=fts)
    ax.set_xlabel(u_label, fontsize=fts)
    ax.set_ylabel(v_label, fontsize=fts)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="black", linestyle="--", alpha=0.5)
    ax.set_facecolor("0.6")

    if colorbar:
        cb = fig.colorbar(
            mesh, ax=ax, orientation=kwargs.get("cb_orientation", "vertical")
        )
        cb.set_label(cb_label, fontsize=fts)

    return mesh, fig, ax
