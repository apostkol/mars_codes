from __future__ import annotations

import numpy as np
import pandas as pd

from mars_currents.solar_wind import (
    ALPHA_MASS_KG,
    MARS_RADIUS_KM,
    PROTON_MASS_KG,
    incident_kinetic_power_gw,
    incident_power_statistics,
    required_incident_radius_rm,
)


def test_incident_kinetic_power_matches_si_definition():
    n_proton = 2.0
    n_alpha = 0.1
    speed_km_s = 400.0
    radius_rm = 2.0

    expected_mass_density = 1e6 * (n_proton * PROTON_MASS_KG + n_alpha * ALPHA_MASS_KG)
    expected_area = np.pi * (radius_rm * MARS_RADIUS_KM * 1e3) ** 2
    expected_gw = (
        0.5 * expected_mass_density * (speed_km_s * 1e3) ** 3 * expected_area / 1e9
    )

    assert np.isclose(
        incident_kinetic_power_gw(
            n_proton,
            n_alpha,
            speed_km_s,
            radius_rm,
        ),
        expected_gw,
    )


def test_required_incident_radius_inverts_power_calculation():
    expected_radius_rm = np.array([2.3, 2.0, 1.4])
    power_mw = 1e3 * incident_kinetic_power_gw(
        n_proton=[1.5, 3.0, 7.0],
        n_alpha=[0.05, 0.1, 0.2],
        speed_km_s=[350.0, 425.0, 550.0],
        radius_rm=expected_radius_rm,
    )

    actual_radius_rm = required_incident_radius_rm(
        power_mw,
        n_proton=[1.5, 3.0, 7.0],
        n_alpha=[0.05, 0.1, 0.2],
        speed_km_s=[350.0, 425.0, 550.0],
    )

    assert np.allclose(actual_radius_rm, expected_radius_rm)


def test_incident_power_statistics_uses_group_radii_for_overall_population():
    solar_wind = pd.DataFrame(
        {
            "n_proton": [1.0, 2.0, 3.0],
            "n_alpha": [0.0, 0.0, 0.0],
            "speed_km_s": [300.0, 400.0, 500.0],
            "p_dyn": [0.1, 1.0, 5.0],
        }
    )
    groups = {
        "low": np.array([True, False, False]),
        "intermediate": np.array([False, True, False]),
        "high": np.array([False, False, True]),
    }
    radii = {"low": 2.3, "intermediate": 2.0, "high": 1.4}

    statistics = incident_power_statistics(solar_wind, groups, radii)

    expected = incident_kinetic_power_gw(
        solar_wind["n_proton"],
        solar_wind["n_alpha"],
        solar_wind["speed_km_s"],
        [2.3, 2.0, 1.4],
    )
    overall = statistics.set_index("population").loc["overall"]
    assert overall["count"] == 3
    assert np.isnan(overall["radius_rm"])
    assert np.isclose(overall["mean_power_gw"], np.mean(expected))
    assert np.isclose(overall["median_power_gw"], np.median(expected))
