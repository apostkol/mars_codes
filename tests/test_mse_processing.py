from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mars_currents import mse_processing


def test_clock_angles_reports_missing_named_arguments():
    with pytest.raises(ValueError, match="start, end, frame"):
        mse_processing.clock_angles(smooth_factor=3)


def test_crossings_adds_bow_shock_columns():
    df = pd.DataFrame(
        {
            "X": [0.0, 3.0],
            "Y": [0.0, 0.0],
            "Z": [0.0, 0.0],
        },
    )

    processed, idx_ms2sw, idx_sw2ms = mse_processing.crossings(df)

    assert list(processed["loc"]) == [0, 1]
    assert idx_ms2sw == [1]
    assert idx_sw2ms == []
    assert {"R", "r_sp_cyl", "th_sp_cyl", "r_bs"}.issubset(processed.columns)


def test_rotation_to_mse_uses_clock_angle():
    df = pd.DataFrame(
        {
            "clock_angle": [0.0, -np.pi / 2],
            "Bx": [1.0, 0.0],
            "By": [2.0, 1.0],
            "Bz": [3.0, 0.0],
            "X": [1.0, 0.0],
            "Y": [2.0, 1.0],
            "Z": [3.0, 0.0],
        },
    )

    bx, by, bz, x, y, z = mse_processing.rotation_to_mse(df, remove=None)

    assert np.allclose(bx, [1.0, 0.0])
    assert np.allclose(by, [2.0, 0.0])
    assert np.allclose(bz, [3.0, 1.0])
    assert np.allclose(x, [1.0, 0.0])
    assert np.allclose(y, [2.0, 0.0])
    assert np.allclose(z, [3.0, 1.0])


def test_rotated_to_mse_accepts_precomputed_clock_angles():
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "loc": [0, 1],
            "clock_angle": [0.0, 0.0],
            "Bx": [1.0, 2.0],
            "By": [0.0, 0.0],
            "Bz": [0.0, 0.0],
            "X": [1.0, 2.0],
            "Y": [0.0, 0.0],
            "Z": [0.0, 0.0],
        },
    )

    processed, magnetosphere = mse_processing.rotated_to_mse(
        smooth_factor=3,
        save=(False, ""),
        df=df,
    )

    assert np.allclose(processed["Bx_mse"], [1.0, 2.0])
    assert np.allclose(processed["R_mse"], [1.0, 2.0])
    assert list(magnetosphere.index) == [0]
