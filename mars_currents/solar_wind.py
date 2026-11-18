"""Solar-wind pressure and incident-power calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

PROTON_MASS_KG = 1.67262192595e-27
ALPHA_MASS_KG = 6.6446573450e-27
MARS_RADIUS_KM = 3396.0


def incident_kinetic_power_gw(n_proton, n_alpha, speed_km_s, radius_rm):
    """Return incident solar-wind kinetic power through a circular area.

    The calculation uses ``0.5 rho v**3 A`` with proton and alpha-particle mass
    density, and ``A = pi (radius_rm R_M)**2``. Number densities are in cm^-3,
    speed is in km/s, and the result is in GW.
    """
    mass_density = 1e6 * (
        np.asarray(n_proton) * PROTON_MASS_KG + np.asarray(n_alpha) * ALPHA_MASS_KG
    )
    speed_m_s = np.asarray(speed_km_s) * 1e3
    radius_m = np.asarray(radius_rm) * MARS_RADIUS_KM * 1e3
    area_m2 = np.pi * radius_m**2
    return 0.5 * mass_density * speed_m_s**3 * area_m2 / 1e9


def required_incident_radius_rm(power_mw, n_proton, n_alpha, speed_km_s):
    """Return the circular-area radius needed to intercept ``power_mw``.

    Density and speed have the same units as :func:`incident_kinetic_power_gw`.
    The result follows by inverting ``P = 0.5 rho v**3 pi R**2`` and is returned
    in Mars radii.
    """
    power_gw_at_one_rm = incident_kinetic_power_gw(
        n_proton,
        n_alpha,
        speed_km_s,
        radius_rm=1.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        radius_rm = np.sqrt(np.asarray(power_mw) / 1e3 / power_gw_at_one_rm)
    return np.where(
        (np.asarray(power_mw) >= 0) & (power_gw_at_one_rm > 0),
        radius_rm,
        np.nan,
    )


def incident_power_statistics(solar_wind, groups, radii_rm):
    """Summarise incident kinetic power in pressure-selected populations.

    The overall population uses the radius assigned to each observation's
    pressure group, so its power distribution is the union of the three
    pressure-selected distributions rather than an estimate at a fourth
    obstacle radius.
    """
    required_columns = ("n_proton", "n_alpha", "speed_km_s", "p_dyn")
    missing = [name for name in required_columns if name not in solar_wind]
    if missing:
        raise KeyError(f"Missing solar-wind columns: {', '.join(missing)}")

    sample_radii = pd.Series(np.nan, index=solar_wind.index, dtype=float)
    for name, radius_rm in radii_rm.items():
        sample_radii.loc[np.asarray(groups[name], dtype=bool)] = radius_rm

    populations = [
        (name, np.asarray(groups[name], dtype=bool), float(radii_rm[name]))
        for name in ("low", "intermediate", "high")
    ]
    populations.append(("overall", sample_radii.notna().to_numpy(), np.nan))

    rows = []
    for name, selected, radius_rm in populations:
        density_and_speed = solar_wind.loc[
            selected,
            ["n_proton", "n_alpha", "speed_km_s", "p_dyn"],
        ]
        finite = np.all(np.isfinite(density_and_speed), axis=1)
        finite &= density_and_speed["speed_km_s"] > 0
        finite &= density_and_speed["n_proton"] >= 0
        finite &= density_and_speed["n_alpha"] >= 0
        sample = density_and_speed.loc[finite]
        selected_radii = (
            sample_radii.loc[sample.index].to_numpy()
            if name == "overall"
            else np.full(len(sample), radius_rm)
        )
        power_gw = incident_kinetic_power_gw(
            sample["n_proton"],
            sample["n_alpha"],
            sample["speed_km_s"],
            selected_radii,
        )
        if len(power_gw):
            mean_power = np.mean(power_gw)
            median_power = np.median(power_gw)
            power_q25, power_q75 = np.percentile(power_gw, (25, 75))
        else:
            mean_power = median_power = power_q25 = power_q75 = np.nan

        rows.append(
            {
                "population": name,
                "radius_rm": radius_rm,
                "count": len(sample),
                "mean_n_proton_cm-3": sample["n_proton"].mean(),
                "mean_n_alpha_cm-3": sample["n_alpha"].mean(),
                "mean_speed_km_s": sample["speed_km_s"].mean(),
                "mean_p_dyn_npa": sample["p_dyn"].mean(),
                "mean_power_gw": mean_power,
                "std_power_gw": np.std(power_gw, ddof=1)
                if len(power_gw) > 1
                else np.nan,
                "median_power_gw": median_power,
                "power_q25_gw": power_q25,
                "power_q75_gw": power_q75,
            }
        )

    return pd.DataFrame(rows)
