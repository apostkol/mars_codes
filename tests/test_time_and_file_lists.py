from __future__ import annotations

from mars_currents import mag_files, swia_files, time_operations


def test_datetime_to_day_of_year_string_and_float():
    date_time = '2020-02-01T12:00:00.000'

    assert time_operations.datetime2doy(date_time) == '032'
    assert time_operations.datetime2doy(date_time, out='float') == 32.5


def test_day_of_year_to_datetime_string():
    assert time_operations.year_doy2date_str(2020, 32.5) == '2020-02-01T12:00:00.000'


def test_mag_file_names_span_multiple_days():
    names = mag_files.mag_files_list(
        '2020-01-01T00:00:00.000',
        '2020-01-02T00:00:00.000',
        frame='_mso',
        sampl='1sec',
    )

    assert names == [
        'mvn_mag_l2_2020001_mso1sec_20200101_v**_r**',
        'mvn_mag_l2_2020002_mso1sec_20200102_v**_r**',
    ]


def test_swia_file_names_span_multiple_days():
    names = swia_files.swia_files_list(
        '2020-01-01T00:00:00.000',
        '2020-01-02T00:00:00.000',
        kind='onboardsvymom',
    )

    assert names == [
        'mvn_swi_l2_onboardsvymom_20200101_v**_r**',
        'mvn_swi_l2_onboardsvymom_20200102_v**_r**',
    ]
