"""Make the dynamic-pressure grids and solar-wind figures used in the paper.

This is intentionally a paper script, not a general command-line tool. Edit the
settings below when the input data or analysis choices change, then run

    python paper/build_pdyn_grids.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spice
from matplotlib.ticker import NullFormatter

import mars_currents.config as cfg
from mars_currents.grid import Grid
from mars_currents.solar_wind import ALPHA_MASS_KG, PROTON_MASS_KG
from mars_currents.spice_codes import load_spice_kernels

# Turn off scalar divide warnings
np.seterr(divide="ignore")

savefig_kwargs = dict(bbox_inches="tight", pad_inches=0)

# -----------------------------------------------------------------------------
# Inputs and analysis choices

# This file must have been produced by ``mse_processing`` with
# ``aberration=4``. The correction is applied to the MSO vectors before their
# transformation to MSE; ``Grid.bin_data()`` does not apply it, and it should
# not be applied again here.
MAG_FILE = cfg.DATA / "mag_mse_2014-10-11_2020-08-14_n15_all.csv"
# MAG_FILE = cfg.DATA / "mag_mse_2014-10-20_2020-08-13_n20_all.csv"
SOLAR_WIND_FILE = cfg.DATA / "drivers_merge_l2_hires_recent.txt"
SPICE_KERNEL = cfg.KERNELS / "maven_time_frames.tm"
OUTPUT_STEM = "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15"
# OUTPUT_STEM = "mag_mse_2014-10-20_2020-08-13_n20_all_NEW"

SOLAR_WIND_COLUMNS = [
    "UT",
    "n_proton",
    "n_alpha",
    "|v_proton|",
    "vx",
    "vy",
    "vz",
    "T_proton_eV",
    "Bx",
    "By",
    "Bz",
]

GRID_PARAMETERS = dict(
    dr=0.04,
    dth_deg=10,
    dphi_deg=18,
    r_lim=(1.04, 2.5),
)

PRESSURE_GROUPS = {
    "low": "C0",
    "intermediate": "C2",
    "high": "C3",
}

# Clock-angle waiting-time choices. A gap ends a continuous observing interval.
CLOCK_ROTATIONS_DEG = (15, 45, 90, 180)
CLOCK_MAX_GAP_S = 3600 * 2
# N.b., 96 h? To be checked.
CLOCK_MAX_WAIT_HOURS = 96
CLOCK_ORIGIN_STRIDE = 10

# -----------------------------------------------------------------------------
# Solar-wind preparation


def dynamic_pressure_npa(n_proton, n_alpha, speed_km_s):
    """Solar-wind dynamic pressure rho v² in nPa."""
    mass_density = 1e6 * (
        np.asarray(n_proton) * PROTON_MASS_KG + np.asarray(n_alpha) * ALPHA_MASS_KG
    )
    return mass_density * (np.asarray(speed_km_s) * 1e3) ** 2 * 1e9


def read_solar_wind():
    """Read the authoritative, headerless driver file and add SPICE ET."""
    print(f"Reading solar-wind drivers from {SOLAR_WIND_FILE}")
    solar_wind = pd.read_csv(
        SOLAR_WIND_FILE,
        sep=r"\s+",
        names=SOLAR_WIND_COLUMNS,
        comment="#",
    )
    # SPICE accepts the ISO form after replacing the date/time separator.
    solar_wind["et"] = [spice.utc2et(utc.replace("/", "T")) for utc in solar_wind["UT"]]
    print(f"Converted {len(solar_wind):,} solar-wind timestamps to SPICE ET")
    return solar_wind


def prepare_solar_wind(solar_wind):
    """Add the quantities plotted and used to divide the observations."""
    solar_wind = solar_wind.copy()
    # The original text file already gives proton speed in km/s.
    solar_wind["speed_km_s"] = solar_wind["|v_proton|"]
    solar_wind["p_dyn"] = dynamic_pressure_npa(
        solar_wind["n_proton"],
        solar_wind["n_alpha"],
        solar_wind["speed_km_s"],
    )
    solar_wind["clock_angle"] = np.degrees(
        np.arctan2(solar_wind["By"], solar_wind["Bz"])
    )
    solar_wind["B"] = np.sqrt(
        solar_wind["Bx"] ** 2 + solar_wind["By"] ** 2 + solar_wind["Bz"] ** 2
    )
    # Upstream convection field with V = (Vx, 0, 0):
    # |E| = |Vx| |B_perp|. The factor converts (km/s) nT to mV/m.
    solar_wind["E_convection_mV_m"] = (
        np.abs(solar_wind["vx"])
        * np.sqrt(solar_wind["By"] ** 2 + solar_wind["Bz"] ** 2)
        * 1e-3
    )
    return solar_wind


def pressure_groups(pressure):
    """Split pressure below Q1, from Q1 through Q3, and above Q3."""
    q1, median, q3 = np.nanpercentile(pressure, (25, 50, 75), method="median_unbiased")
    groups = {
        "low": pressure < q1,
        "intermediate": (pressure >= q1) & (pressure <= q3),
        "high": pressure > q3,
    }
    return groups, (q1, median, q3)


def pressure_at_mag_times(mag_times, solar_wind):
    """Linearly interpolate pressure onto MAG times without extrapolation."""
    ordered = solar_wind[["et", "p_dyn"]].dropna().sort_values("et")
    ordered = ordered.groupby("et", as_index=False)["p_dyn"].mean()
    return np.interp(
        mag_times,
        ordered["et"],
        ordered["p_dyn"],
        left=np.nan,
        right=np.nan,
    )


# -----------------------------------------------------------------------------
# Paper figures


def label_solar_wind_panels(axes):
    """Add panel letters in reading order to a six-panel figure."""
    for letter, ax in zip("abcdef", axes.flat):
        ax.text(
            0.03,
            0.97,
            f"({letter})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
            zorder=10,
        )


def plot_solar_wind_distributions(solar_wind, groups, quartiles):
    """Compare distributions and report statistics in pressure quartiles."""
    quantities = {
        r"$P_\mathrm{dyn}$ [nPa]": (solar_wind["p_dyn"], True, (0.1, 10)),
        "IMF clock angle [deg]": (solar_wind["clock_angle"], False, (-180, 180)),
        r"$|V|$ [km s$^{-1}$]": (solar_wind["speed_km_s"], True, (200, 1000)),
        r"$n_p$ [cm$^{-3}$]": (solar_wind["n_proton"], True, (0.1, 40)),
        r"$|B|$ [nT]": (solar_wind["B"], True, None),
        r"$|E_\mathrm{conv}|$ [mV m$^{-1}$]": (
            solar_wind["E_convection_mV_m"],
            True,
            (0.01, 10),
        ),
    }

    q1, median, q3 = quartiles
    pressure = solar_wind["p_dyn"]
    quartile_groups = {
        "Q1": pressure < q1,
        "IQR": (pressure >= q1) & (pressure < q3),
        "Q4": pressure > q3,
        "All": np.ones(len(solar_wind), dtype=bool),
    }
    statistics = []
    for label, (values, _, _) in quantities.items():
        for quartile, selected in quartile_groups.items():
            sample = values[selected & np.isfinite(values)]
            statistics.append(
                {
                    "parameter": label,
                    "pressure_quartile": quartile,
                    "count": len(sample),
                    "mean": sample.mean(),
                    "stddev": sample.std(ddof=1),
                }
            )

    statistics = pd.DataFrame(statistics)
    report = statistics.to_string(
        index=False, float_format=lambda value: f"{value:.6g}"
    )
    report_file = cfg.DATA / f"{OUTPUT_STEM}_solar_wind_distribution_statistics.txt"
    report_file.write_text(report + "\n", encoding="utf-8")
    print("Solar-wind mean and standard deviation by dynamic-pressure quartile:")
    print(report)
    print(f"Wrote solar-wind statistics to {report_file}")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), constrained_layout=True)

    for ax, (label, (values, use_log_scale, x_limits)) in zip(
        axes.flat, quantities.items()
    ):
        finite = np.isfinite(values)
        if use_log_scale:
            finite &= values > 0
            bins = np.geomspace(values[finite].min(), values[finite].max(), 51)
            ax.set_xscale("log")
            bin_widths = np.diff(np.log10(bins))
        else:
            bins = np.linspace(-180, 180, 51)
            ax.set_xticks([-180, -90, 0, 90, 180])

        def plot_density(sample, color, linewidth, name, fill=False):
            if use_log_scale:
                # Normalize in log10(x), so area on the log axis is one.
                counts, _ = np.histogram(sample, bins=bins)
                density = counts / (len(sample) * bin_widths)
                if fill:
                    ax.stairs(
                        density,
                        bins,
                        fill=True,
                        facecolor=color,
                        alpha=0.3,
                        linewidth=0,
                        zorder=1,
                    )
                ax.stairs(
                    density,
                    bins,
                    color=color,
                    linewidth=linewidth,
                    label=name,
                    zorder=2,
                )
            else:
                if fill:
                    ax.hist(
                        sample,
                        bins=bins,
                        density=True,
                        histtype="stepfilled",
                        color=color,
                        alpha=0.3,
                        zorder=1,
                    )
                ax.hist(
                    sample,
                    bins=bins,
                    density=True,
                    histtype="step",
                    color=color,
                    linewidth=linewidth,
                    label=name,
                    zorder=2,
                )

        # Remove this trace if only the pressure-separated populations are wanted.
        plot_density(
            values[finite],
            "0.25",
            1.4,
            "All",
            fill=True,
        )
        for name, color in PRESSURE_GROUPS.items():
            selected = finite & groups[name]
            plot_density(
                values[selected],
                color,
                1.5,
                name.capitalize(),
            )
        ax.set(
            xlabel=label,
            ylabel="Probability density per decade"
            if use_log_scale
            else "Probability density",
        )
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if label == r"$|V|$ [km s$^{-1}$]":
            ax.set_xticks([200, 500, 1000], labels=["200", "500", "1000"])
            ax.set_xticks([300, 400, 600, 700, 800, 900], minor=True)
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)

    axes.flat[0].axvline(q1, color="0.4", linestyle="--")
    axes.flat[0].axvline(q3, color="0.4", linestyle="--")
    label_solar_wind_panels(axes)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
    )
    fig.savefig(
        cfg.PLOTS / f"{OUTPUT_STEM}_solar_wind_distributions.pdf",
        **savefig_kwargs,
    )
    plt.close(fig)


def plot_solar_wind_ecdfs(solar_wind, quartiles):
    """Compare solar-wind distributions using empirical CDFs.

    At a given x value, an ECDF gives the fraction of observations less than
    or equal to x. Unlike a histogram, it requires no choice of bin edges or
    normalization. The four groups here are the four dynamic-pressure
    quartiles, rather than the three populations used in the histogram figure.
    """
    quantities = {
        r"$P_\mathrm{dyn}$ [nPa]": (solar_wind["p_dyn"], True),
        "IMF clock angle [deg]": (solar_wind["clock_angle"], False),
        r"$|V|$ [km s$^{-1}$]": (solar_wind["speed_km_s"], True),
        r"$n_p$ [cm$^{-3}$]": (solar_wind["n_proton"], True),
        r"$|B|$ [nT]": (solar_wind["B"], True),
        r"$|E_\mathrm{conv}|$ [mV m$^{-1}$]": (
            solar_wind["E_convection_mV_m"],
            True,
        ),
    }

    q1, median, q3 = quartiles
    pressure = solar_wind["p_dyn"]
    quartile_groups = {
        "Q1": pressure < q1,
        "Q2": (pressure >= q1) & (pressure < median),
        "Q3": (pressure >= median) & (pressure <= q3),
        "Q4": pressure > q3,
    }

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), constrained_layout=True)
    for ax, (label, (values, use_log_scale)) in zip(axes.flat, quantities.items()):
        finite = np.isfinite(values)
        if use_log_scale:
            finite &= values > 0
            ax.set_xscale("log")
        else:
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, -90, 0, 90, 180])

        all_values = np.sort(np.asarray(values[finite]))
        all_fraction = np.arange(1, len(all_values) + 1) / len(all_values)
        ax.step(
            all_values,
            all_fraction,
            where="post",
            color="0.25",
            linewidth=1.4,
            label=f"All (n={len(all_values):,})",
        )

        for index, (name, group) in enumerate(quartile_groups.items()):
            sample = np.sort(np.asarray(values[finite & group]))
            fraction = np.arange(1, len(sample) + 1) / len(sample)
            ax.step(
                sample,
                fraction,
                where="post",
                color=f"C{index}",
                linewidth=1.5,
                label=f"{name} (n={len(sample):,})",
            )

        ax.set(xlabel=label, ylabel=r"Fraction $\leq x$", ylim=(0, 1))
        ax.grid(alpha=0.2)

    label_solar_wind_panels(axes)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
    )
    fig.savefig(
        cfg.PLOTS / f"{OUTPUT_STEM}_solar_wind_ecdfs.pdf",
        **savefig_kwargs,
    )
    plt.close(fig)


def clock_angle_waits(times, angles, threshold):
    """Return observed and censored first-passage waits for one rotation.

    Clock angle is unwrapped separately between data gaps. Every tenth sample is
    used as a starting point, and waits that reach 48 hours are right-censored.
    These two choices are set explicitly at the top of the script.
    """
    times = np.asarray(times, dtype=float)
    angles = np.asarray(angles, dtype=float)
    finite = np.isfinite(times) & np.isfinite(angles)
    times, angles = times[finite], angles[finite]
    order = np.argsort(times)
    times, angles = times[order], angles[order]

    gaps = np.flatnonzero(np.diff(times) > CLOCK_MAX_GAP_S) + 1
    starts = np.r_[0, gaps]
    stops = np.r_[gaps, len(times)]
    max_wait = CLOCK_MAX_WAIT_HOURS * 3600
    durations = []
    events = []

    for start, stop in zip(starts, stops):
        segment_times = times[start:stop]
        segment_angles = np.degrees(np.unwrap(np.radians(angles[start:stop])))
        for origin in range(0, len(segment_times) - 1, CLOCK_ORIGIN_STRIDE):
            end = np.searchsorted(
                segment_times, segment_times[origin] + max_wait, side="right"
            )
            change = np.abs(segment_angles[origin + 1 : end] - segment_angles[origin])
            crossings = np.flatnonzero(change >= threshold)
            if len(crossings):
                finish = origin + 1 + crossings[0]
                durations.append(segment_times[finish] - segment_times[origin])
                events.append(True)
            elif end > origin + 1:
                durations.append(segment_times[end - 1] - segment_times[origin])
                events.append(False)

    return np.asarray(durations), np.asarray(events, dtype=bool)


def kaplan_meier(durations, events):
    """Kaplan–Meier estimate of the probability a rotation has occurred."""
    event_times = np.unique(durations[events])
    survival = 1.0
    probability = []
    for time in event_times:
        at_risk = np.sum(durations >= time)
        events_now = np.sum((durations == time) & events)
        survival *= 1 - events_now / at_risk
        probability.append(1 - survival)
    return event_times, np.asarray(probability)


def wait_at_probability(times, probability, target):
    reached = np.flatnonzero(probability >= target)
    return times[reached[0]] if len(reached) else np.nan


def plot_clock_angle_waits(solar_wind):
    """Plot first-passage curves and save the numerical answers as CSV."""
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    summary = []

    for index, threshold in enumerate(CLOCK_ROTATIONS_DEG):
        durations, events = clock_angle_waits(
            solar_wind["et"], solar_wind["clock_angle"], threshold
        )
        times, probability = kaplan_meier(durations, events)
        times_hours = times / 3600
        median_hours = wait_at_probability(times_hours, probability, 0.5)
        ax.step(
            times_hours,
            probability,
            where="post",
            color=f"C{index}",
            label=f"{threshold}° (50%: {median_hours:.2g} h)",
        )
        summary.append(
            {
                "rotation_deg": threshold,
                "wait_hours_25_percent": wait_at_probability(
                    times_hours, probability, 0.25
                ),
                "wait_hours_50_percent": median_hours,
                "wait_hours_75_percent": wait_at_probability(
                    times_hours, probability, 0.75
                ),
                "starting_points": len(durations),
                "observed_rotations": events.sum(),
                "censored": (~events).sum(),
            }
        )

    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Wait time [hours]",
        ylabel="Estimated probability rotation has occurred",
        xlim=(0.1, CLOCK_MAX_WAIT_HOURS),
        ylim=(0, 1),
    )
    plt.xscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(
        cfg.PLOTS / f"{OUTPUT_STEM}_clock_angle_waiting_times.pdf", **savefig_kwargs
    )
    plt.close(fig)

    summary = pd.DataFrame(summary)
    summary.to_csv(
        cfg.DATA / f"{OUTPUT_STEM}_clock_angle_waiting_times.csv", index=False
    )
    print(summary.to_string(index=False))


# -----------------------------------------------------------------------------
# Grid construction


def build_pressure_grids(mag, pressure_at_mag, quartiles):
    """Write the three selected data sets and pass each one to Grid."""
    q1, _, q3 = quartiles
    finite = np.isfinite(pressure_at_mag)
    groups = {
        "low": finite & (pressure_at_mag < q1),
        "intermediate": finite & (pressure_at_mag >= q1) & (pressure_at_mag <= q3),
        "high": finite & (pressure_at_mag > q3),
    }
    grid_columns = [
        "Bx_mse",
        "By_mse",
        "Bz_mse",
        "X_mse",
        "Y_mse",
        "Z_mse",
        "R_mse",
        "th_mse",
        "phi_mse",
    ]

    for name in PRESSURE_GROUPS:
        selected_file = cfg.DATA / f"{OUTPUT_STEM}_{name}_pdyn.csv"
        mag.loc[groups[name], grid_columns].to_csv(selected_file, index=False)

        grid = Grid(
            **GRID_PARAMETERS,
            name=f"{OUTPUT_STEM}_{name}_pdyn",
            file_name=f"{OUTPUT_STEM}_{name}_pdyn",
        )
        grid.bin_data(selected_file)
        grid.current_densities()
        grid.currents_uncertainties()
        grid.save(cfg.DATA)


def main(recompute=False):
    cfg.DATA.mkdir(parents=True, exist_ok=True)
    cfg.PLOTS.mkdir(parents=True, exist_ok=True)

    mag = pd.read_csv(MAG_FILE)
    load_spice_kernels(SPICE_KERNEL)
    solar_wind = prepare_solar_wind(read_solar_wind())

    # Use only solar-wind observations overlapping the MAG record.
    solar_wind = solar_wind[
        solar_wind["et"].between(mag["time"].min(), mag["time"].max())
    ].reset_index(drop=True)
    groups, quartiles = pressure_groups(solar_wind["p_dyn"])
    q1, median, q3 = quartiles
    print(f"Pdyn quartiles: Q1={q1:.3f}, median={median:.3f}, Q3={q3:.3f} nPa")
    plot_solar_wind_distributions(solar_wind, groups, quartiles)
    plot_solar_wind_ecdfs(solar_wind, quartiles)
    plot_clock_angle_waits(solar_wind)

    pressure = pressure_at_mag_times(mag["time"], solar_wind)

    if recompute:
        print("Building low, intermediate, and high dynamic-pressure grids")
        build_pressure_grids(mag, pressure, quartiles)


if __name__ == "__main__":
    main()
