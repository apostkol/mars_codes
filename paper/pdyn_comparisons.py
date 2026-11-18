import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import page_trend_test

import mars_currents.config as cfg
from mars_currents.grid import load_grid
from mars_currents.maps import mollweide_comparison_maps
from mars_currents.solar_wind import (
    incident_power_statistics,
    required_incident_radius_rm,
)

try:
    from paper.build_pdyn_grids import (
        SOLAR_WIND_COLUMNS,
        SOLAR_WIND_FILE,
        prepare_solar_wind,
        pressure_groups,
    )
except ModuleNotFoundError:  # Direct execution: python paper/pdyn_comparisons.py
    from build_pdyn_grids import (
        SOLAR_WIND_COLUMNS,
        SOLAR_WIND_FILE,
        prepare_solar_wind,
        pressure_groups,
    )

PRESSURE_LABELS = ("Low", "Intermediate", "High")
savefig_kwargs = dict(bbox_inches="tight", pad_inches=0)

# E = -Vx X (BY+BZ)
# 18/08/2024:
#            low: 0.6342
#   intermediate: 0.9591
#   high: 1.6193

# V/m
CONVECTION_ELECTRIC_FIELD = {
    "Low_": 0.6342 * 1e-3,
    "Intermediate_": 0.9591 * 1e-3,
    "High_": 1.6193 * 1e-3,
}

INCIDENT_POWER_RADII_RM = {
    "low": 2.3,
    "intermediate": 2.0,
    "high": 1.4,
}
OBSERVED_POWER_MW = {
    "low": 787.0,
    "intermediate": 1187.0,
    "high": 3590.0,
}
SOLAR_WIND_INTERVAL_UTC = ("2014-10-11", "2020-08-15")


def report_incident_solar_wind_power():
    """Calculate and save paper-ready incident solar-wind power statistics."""
    solar_wind = pd.read_csv(
        SOLAR_WIND_FILE,
        sep=r"\s+",
        names=SOLAR_WIND_COLUMNS,
        comment="#",
    )
    timestamps = pd.to_datetime(
        solar_wind["UT"].str.replace("/", "T", regex=False),
        utc=True,
        errors="coerce",
    )
    start, stop = (pd.Timestamp(value, tz="UTC") for value in SOLAR_WIND_INTERVAL_UTC)
    solar_wind = solar_wind.loc[timestamps.ge(start) & timestamps.lt(stop)].reset_index(
        drop=True
    )
    solar_wind = prepare_solar_wind(solar_wind)

    groups, quartiles = pressure_groups(solar_wind["p_dyn"])
    statistics = incident_power_statistics(
        solar_wind,
        groups,
        INCIDENT_POWER_RADII_RM,
    )
    statistics["observed_power_mw"] = statistics["population"].map(OBSERVED_POWER_MW)
    statistics["required_radius_rm"] = required_incident_radius_rm(
        statistics["observed_power_mw"],
        statistics["mean_n_proton_cm-3"],
        statistics["mean_n_alpha_cm-3"],
        statistics["mean_speed_km_s"],
    )
    q1, median, q3 = quartiles
    print(
        "Incident solar-wind kinetic power, "
        r"0.5 rho v^3 pi R^2 "
        f"(Pdyn Q1/median/Q3 = {q1:.3f}/{median:.3f}/{q3:.3f} nPa):"
    )
    print(statistics.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    inferred = statistics.loc[
        statistics["population"].isin(OBSERVED_POWER_MW),
        [
            "population",
            "observed_power_mw",
            "mean_n_proton_cm-3",
            "mean_n_alpha_cm-3",
            "mean_speed_km_s",
            "required_radius_rm",
        ],
    ]
    print("Required circular-area radii from the mean solar-wind inputs:")
    print(inferred.to_string(index=False, float_format=lambda value: f"{value:.4g}"))

    output = cfg.PLOTS / "Pdyn_incident_solar_wind_power.csv"
    statistics.to_csv(output, index=False)
    print(f"Wrote incident-power statistics to {output}")
    return statistics


def _benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values, preserving NaNs and input order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return adjusted

    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * finite.size / np.arange(1, finite.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _signed_samples(values, sign):
    """Return per-pressure signed populations and persistent-sign cell values."""
    values = np.asarray(values, dtype=float)
    if sign == "positive":
        sign_mask = values > 0
    elif sign == "negative":
        sign_mask = values < 0
    else:
        raise ValueError("sign must be 'positive' or 'negative'")

    populations = [
        values[np.isfinite(values[:, i]) & sign_mask[:, i], i] for i in range(3)
    ]
    persistent = values[np.all(np.isfinite(values) & sign_mask, axis=1)]
    return populations, persistent


def _paired_trend(paired, reference_magnitude):
    """Summarise an ordered trend in matched low/intermediate/high grid cells.

    Page's test uses the repeated values from all three pressure levels and is
    designed specifically for an ordered treatment alternative.  We test both
    orders and double the smaller one-sided p-value.  Only cells finite in all
    three pressure grids and retain their sign throughout are used. Values are
    expressed relative to the intermediate-population median magnitude.
    """
    paired = np.asarray(paired, dtype=float)
    if (
        len(paired) < 2
        or not np.isfinite(reference_magnitude)
        or reference_magnitude <= 0
    ):
        return np.empty((0, 3)), np.array([]), np.nan

    magnitudes = np.abs(paired)
    percent_relative_to_intermediate = (magnitudes / reference_magnitude - 1) * 100
    cell_changes_per_step = (
        percent_relative_to_intermediate[:, 2] - percent_relative_to_intermediate[:, 0]
    ) / 2
    if np.all(cell_changes_per_step == 0):
        p_value = 1.0
    else:
        increasing_p = page_trend_test(percent_relative_to_intermediate).pvalue
        decreasing_p = page_trend_test(percent_relative_to_intermediate[:, ::-1]).pvalue
        p_value = min(1.0, 2 * min(increasing_p, decreasing_p))
    return percent_relative_to_intermediate, cell_changes_per_step, p_value


def _altitude_label(grid, radial_index):
    half_width = grid.dr / 2
    lower = (grid.r[radial_index] - half_width - 1) * grid.radius_km
    upper = (grid.r[radial_index] + half_width - 1) * grid.radius_km
    return f"{lower:.0f}–{upper:.0f} km", lower, upper


def plot_pressure_distributions(
    grids,
    radii,
    parameter,
    ylabel,
    output_stem,
    break_percentile=95,
    upper_height_ratio=1,
    lower_height_ratio=3,
):
    """Plot signed populations using a compact broken axis for the upper tail."""
    reference = grids[0]
    radial_indices = [int(np.argmin(np.abs(reference.r - r))) for r in radii]
    radial_selections = [(index, False) for index in radial_indices] + [(None, True)]
    n_columns = len(radial_selections)
    fig = plt.figure(figsize=(2.0 * n_columns, 6))
    gridspec = fig.add_gridspec(
        4,
        n_columns,
        height_ratios=(
            upper_height_ratio,
            lower_height_ratio,
            upper_height_ratio,
            lower_height_ratio,
        ),
        hspace=0.06,
        wspace=0.12,
        left=0.05,
    )
    upper_axes = np.empty((2, n_columns), dtype=object)
    axes = np.empty((2, n_columns), dtype=object)
    for row_index in range(2):
        for column in range(n_columns):
            share_upper = upper_axes[row_index, 0] if column else None
            share_lower = axes[row_index, 0] if column else None
            upper_axes[row_index, column] = fig.add_subplot(
                gridspec[2 * row_index, column], sharey=share_upper
            )
            axes[row_index, column] = fig.add_subplot(
                gridspec[2 * row_index + 1, column],
                sharex=upper_axes[row_index, column],
                sharey=share_lower,
            )
    colors = [grid_colors[prefix] for _, prefix, _ in grids_to_load]
    rows = []
    row_values = [[], []]

    for column, (radial_index, pooled) in enumerate(radial_selections):
        if pooled:
            radial_mask = (reference.r >= 1.0) & (reference.r <= 2.0)
            samples = np.column_stack(
                [
                    np.asarray(getattr(grid, parameter))[:, :, radial_mask].ravel()
                    for grid in grids
                ]
            )
            altitude = r"All: 1.0–2.0 $R_M$"
            altitude_min = 0.0
            altitude_max = reference.radius_km
            altitude = rf"All h < {int(altitude_max)} km"

        else:
            samples = np.column_stack(
                [
                    np.asarray(getattr(grid, parameter))[:, :, radial_index].ravel()
                    for grid in grids
                ]
            )
            altitude, altitude_min, altitude_max = _altitude_label(
                reference, radial_index
            )
        finite_counts = np.sum(np.isfinite(samples), axis=0)

        for row_index, sign in enumerate(("positive", "negative")):
            ax = axes[row_index, column]
            upper_ax = upper_axes[row_index, column]
            populations, paired = _signed_samples(samples, sign)
            reference_magnitude = (
                np.median(np.abs(populations[1])) if len(populations[1]) else np.nan
            )
            plot_values = [
                (np.abs(population) / reference_magnitude - 1) * 100
                if len(population) and np.isfinite(reference_magnitude)
                else np.array([np.nan])
                for population in populations
            ]
            row_values[row_index].extend(plot_values)
            if pooled:
                relative_percent = np.empty((0, 3))
                changes = np.array([])
                p_value = np.nan
            else:
                relative_percent, changes, p_value = _paired_trend(
                    paired, reference_magnitude
                )
            for box_axis in (upper_ax, ax):
                boxes = box_axis.boxplot(
                    plot_values,
                    tick_labels=PRESSURE_LABELS,
                    patch_artist=True,
                    widths=0.65,
                    medianprops={"color": "black", "linewidth": 1.4},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
                )
                for patch, color in zip(boxes["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.65)

            for pressure_index, population in enumerate(populations, start=1):
                count_label = (
                    f"N={len(population)}"
                    if pressure_index == 1
                    else str(len(population))
                )
                upper_ax.text(
                    pressure_index,
                    0.95,
                    count_label,
                    transform=upper_ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9,
                )

            if row_index == 0:
                upper_ax.set_title(altitude)
            ax.tick_params(axis="x", labelrotation=25)
            upper_ax.tick_params(axis="x", bottom=False, labelbottom=False)
            for box_axis in (upper_ax, ax):
                box_axis.axhline(0, color="0.75", linewidth=0.8, zorder=0)
            if False:
                upper_ax.text(
                    0.5,
                    0.98,
                    "box n = "
                    + "/".join(str(len(population)) for population in populations)
                    + ("; unpaired" if pooled else f"; paired n = {len(paired)}"),
                    transform=upper_ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                )

            median_percent = np.array([np.nanmedian(values) for values in plot_values])
            for pressure_index, median in enumerate(median_percent, start=1):
                if np.isfinite(median):
                    ax.annotate(
                        f"{median:+.0f}%" if not np.isclose(median, 0) else "0%",
                        xy=(pressure_index, median),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round,pad=0.15",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.75,
                        },
                    )
            sign_counts = np.sum(
                samples > 0 if sign == "positive" else samples < 0, axis=0
            )
            sign_fractions = np.divide(
                sign_counts,
                finite_counts,
                out=np.full(3, np.nan),
                where=finite_counts > 0,
            )
            monotonic = (
                np.mean(
                    (np.abs(paired[:, 0]) <= np.abs(paired[:, 1]))
                    & (np.abs(paired[:, 1]) <= np.abs(paired[:, 2]))
                )
                if len(paired)
                else np.nan
            )
            rows.append(
                {
                    "quantity": parameter,
                    "sign": sign,
                    "distribution_scope": "1.0--2.0_RM" if pooled else "altitude_shell",
                    "altitude_min_km": altitude_min,
                    "altitude_max_km": altitude_max,
                    "n_population_low": len(populations[0]),
                    "n_population_intermediate": len(populations[1]),
                    "n_population_high": len(populations[2]),
                    "n_persistent_sign_cells": np.nan if pooled else len(paired),
                    "fraction_sign_low": sign_fractions[0],
                    "fraction_sign_intermediate": sign_fractions[1],
                    "fraction_sign_high": sign_fractions[2],
                    "median_intermediate_reference_magnitude": (reference_magnitude),
                    "median_percent_low_vs_intermediate": median_percent[0],
                    "median_percent_intermediate_vs_intermediate": median_percent[1],
                    "median_percent_high_vs_intermediate": median_percent[2],
                    "median_percent_high_minus_low": (
                        median_percent[2] - median_percent[0]
                        if pooled
                        else np.median(relative_percent[:, 2] - relative_percent[:, 0])
                        if len(relative_percent)
                        else np.nan
                    ),
                    "median_percentage_point_trend_per_pressure_step": (
                        (median_percent[2] - median_percent[0]) / 2
                        if pooled
                        else np.median(changes)
                        if len(changes)
                        else np.nan
                    ),
                    "fraction_monotonic_magnitude_increase": (
                        np.nan if pooled else monotonic
                    ),
                    "trend_p_value": p_value,
                }
            )

    for row_index in range(2):
        finite_values = np.concatenate(
            [values[np.isfinite(values)] for values in row_values[row_index]]
        )
        lower = np.min(finite_values)
        upper = np.max(finite_values)
        split = np.percentile(finite_values, break_percentile)
        if np.isclose(split, upper):
            split = np.percentile(finite_values, 90)
        lower_span = max(split - lower, 1.0)
        upper_span = max(upper - split, 1.0)
        lower_limits = (lower - 0.05 * lower_span, split)
        upper_limits = (split, upper + 0.08 * upper_span)

        for column in range(n_columns):
            upper_ax = upper_axes[row_index, column]
            lower_ax = axes[row_index, column]
            panel_index = row_index * n_columns + column
            upper_ax.text(
                0.03,
                0.76,
                f"{chr(ord('a') + panel_index)})",
                transform=upper_ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            upper_ax.set_ylim(*upper_limits)
            lower_ax.set_ylim(*lower_limits)
            upper_ax.spines.bottom.set_visible(False)
            lower_ax.spines.top.set_visible(False)
            upper_ax.tick_params(axis="x", bottom=False, labelbottom=False)
            lower_ax.xaxis.tick_bottom()

            break_style = dict(
                marker=[(-1, -0.5), (1, 0.5)],
                markersize=8,
                linestyle="none",
                color="black",
                mec="black",
                mew=1,
                clip_on=False,
            )
            upper_ax.plot([0, 1], [0, 0], transform=upper_ax.transAxes, **break_style)
            lower_ax.plot([0, 1], [1, 1], transform=lower_ax.transAxes, **break_style)
            if column > 0:
                upper_ax.tick_params(axis="y", labelleft=False)
                lower_ax.tick_params(axis="y", labelleft=False)

    adjusted = _benjamini_hochberg([row["trend_p_value"] for row in rows])
    for result_index, (row, q_value) in enumerate(zip(rows, adjusted)):
        row["trend_q_value_bh"] = q_value
        trend = row["median_percentage_point_trend_per_pressure_step"]
        direction = "strengthens" if trend > 0 else "weakens"
        if trend == 0:
            direction = "no change"
        elif not np.isfinite(trend):
            direction = "insufficient data"
        row_index = 0 if row["sign"] == "positive" else 1
        column = result_index // 2
        ax = axes[row_index, column]
        annotation = (
            "whole distributions; no paired test"
            if row["distribution_scope"] == "1.0--2.0_RM"
            else f"{direction}; BH q = {q_value:.2g}"
        )
        if False:
            ax.text(
                0.5,
                0.02,
                annotation,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[0, 0].set_ylabel(f"Positive {ylabel} Change [%]")
    axes[1, 0].set_ylabel(f"Negative {ylabel} Change [%]")
    # plt.annotate(
    #     "Magnitude change",
    #     (0.00, 0.5),
    #     xycoords="figure fraction",
    #     rotation=90,
    #     ha="right",
    #     va="center",
    # )
    fig.suptitle(f"{ylabel}: matched-cell magnitude relative to intermediate pressure")
    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.1, top=0.9)
    fig.savefig(cfg.PLOTS / f"{output_stem}.pdf", dpi=300, **savefig_kwargs)
    return rows


def write_statistics_table(rows, file_path):
    """Write the values behind the figure annotations to a reusable CSV."""
    with Path(file_path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# %%
incident_power_statistics_table = report_incident_solar_wind_power()

loaded_grids = {}
grid_colors = {}

location = cfg.RESULTS
grids_to_load = [
    # (
    #     "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle",
    #     "All_",
    #     "black",
    # ),
    # (
    #     "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_NH.pickle",
    #     "NH_",
    #     "C0",
    # ),
    (
        "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_low_pdyn.pickle",
        "Low_",
        "C1",
    ),
    (
        "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_intermediate_pdyn.pickle",
        "Intermediate_",
        "C2",
    ),
    (
        "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_high_pdyn.pickle",
        "High_",
        "C3",
    ),
]

for grid_pickle, global_prefix, color in grids_to_load:
    try:
        loaded_grids[global_prefix] = load_grid(
            load_from_file=True, file_path=location / grid_pickle
        )
        grid_colors[global_prefix] = color
    except Exception as e:
        print(f"Error loading {grid_pickle}: {e}")
        raise

radii = (1.05, 1.1, 1.2, 1.3, 1.6)
radii = radii[::-1]

pressure_grids = [loaded_grids[prefix] for _, prefix, _ in grids_to_load]
pressure_names = [prefix.rstrip("_") for _, prefix, _ in grids_to_load]

if True:
    fig, axs = mollweide_comparison_maps(
        pressure_grids,
        "Bsmr",
        radii,
        column_names=pressure_names,
        arrows="Bsm",
        color_scale="row",
        cb_label=r"$B_r\,[\mathrm{nT}]$",
    )
    plt.savefig(cfg.PLOTS / "BsmrPdyn.pdf", dpi=300, **savefig_kwargs)

if True:
    for param in ("Jr", "Jth", "Jphi"):
        fig, axs = mollweide_comparison_maps(
            pressure_grids,
            param,
            radii,
            column_names=pressure_names,
            arrows="J",
            color_scale="row",
            cb_label=rf"${param}\,[\mathrm{{nA\,m^{{-2}}}}]$",
        )
        plt.savefig(cfg.PLOTS / f"{param}Pdyn.pdf", dpi=300, **savefig_kwargs)


if True:
    for _, global_prefix, _ in grids_to_load:
        grid = loaded_grids[global_prefix]
        e_vec = np.array((0.0, 0.0, CONVECTION_ELECTRIC_FIELD[global_prefix]))
        print(f"Using E = {e_vec} for {global_prefix}")
        grid.calculate_JdotE(e_vec)

    fig, axs = mollweide_comparison_maps(
        pressure_grids,
        "JdotE",
        radii,
        column_names=pressure_names,
        arrows="J",
        color_scale="row",
        cb_label=r"$J\cdot E\,[\mathrm{W\,m^{-3}}]$",
    )

    plt.savefig(cfg.PLOTS / "JdotEPdyn.pdf", dpi=300, **savefig_kwargs)
    # plt.show()

    # %%


# Distributional and paired statistical comparisons.  Values in a given row
# of the input to the trend test are the same spatial cell at the three dynamic
# pressure levels; this avoids treating matched grid locations as independent.
if True:
    for grid in pressure_grids:
        if not hasattr(grid, "JdotE"):
            grid.calculate_JdotE()

    statistical_rows = []
    statistical_rows.extend(
        plot_pressure_distributions(
            pressure_grids,
            radii,
            "Bsmr",
            r"$B_r$",
            "BsmrPdyn_distributions",
        )
    )
    statistical_rows.extend(
        plot_pressure_distributions(
            pressure_grids,
            radii,
            "Jth",
            r"$J_\theta$",
            "JthPdyn_distributions",
        )
    )
    statistical_rows.extend(
        plot_pressure_distributions(
            pressure_grids,
            radii,
            "JdotE",
            r"$J\cdot E$",
            "JdotEPdyn_distributions",
        )
    )
    write_statistics_table(
        statistical_rows, cfg.PLOTS / "Pdyn_distribution_statistics.csv"
    )
