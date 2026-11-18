from __future__ import annotations

"""Helpers for locating and loading MAVEN MAG files."""

import glob
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import requests
import spiceypy as spice

from .spice_codes import load_spice_kernels
from .time_operations import (
    datetime2doy,
    reduce_date_time,
    time,
    year_doy2date_str,
    yearisLeap,
)

__author__ = "Apostolos Kolokotronis, David Andrews"
__email__ = "apostolos.kolokotronis@irf.se, david.andrews@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "1.0.0"
__status__ = "Prototype"

__all__ = ["create_paths", "load_mag_data", "mag_files_list"]


def _mag_file_name(date_time, frame, sampl=""):
    """Outputs the mag l2_v01_r01 file name, given the date, time, frame of reference and sampling rate."""

    day_of_year = datetime2doy(date_time, out="str")
    year, month, day = reduce_date_time(date_time)[0:3]

    name = f"mvn_mag_l2_{year}{day_of_year}{frame}{sampl}_{year}{month}{day}_v**_r**"

    return name


def mag_files_list(start, end, frame, sampl):

    t = time(start)
    finish = time(end)
    list_file_names = []

    while int(t.doy) <= int(finish.doy) or t.year < finish.year:
        date_time = year_doy2date_str(t.year, t.doy)
        date_str = f"{date_time}"
        list_file_names.append(_mag_file_name(date_str, frame, sampl))

        if t.doy >= int(yearisLeap(t.year)) + 365:
            t.year += 1
            t.doy = 0

        t.doy = int(t.doy) + 1

    return list_file_names


def create_paths(start, end, frame, sampl, data_path, verbose=True):
    """Returns a list of paths for the mag files."""

    paths = []
    not_exist = []
    for name in mag_files_list(start, end, frame, sampl):
        year = name[11:15]
        month = name[-12:-10]

        file_path = f"{data_path}/{year}/{month}/{name}.sts"
        file_in_path = glob.glob(file_path)

        if not bool(file_in_path):
            not_exist.append(file_path)
        elif len(file_in_path) > 1:
            paths.append(file_in_path[-1])
        else:
            paths.append(file_in_path[0])

    if verbose:
        print("Paths of files: \n", paths)

    return [paths, not_exist]


def load_mag_data(
    start,
    end,
    frame,
    sampl,
    data_path,
    kernel_path,
    verbose=True,
    re_sampling=None,
):

    load_spice_kernels(os.path.normpath(kernel_path))
    paths, not_exist = create_paths(start, end, frame, sampl, data_path, verbose=False)

    if verbose:
        print("These files do not exist: \n")
        for no_file in not_exist:
            print(f"{no_file}")
        print("--------------------------------------------------------------------")
    output = {"time": None}

    if verbose:
        print("Loading files: \n")

    for name in paths:
        if verbose:
            print("\r", name, end=" ")

        # header size is variable. Find four END_OBJECT in a row
        count = 0
        skip = 1

        with open(name) as f:
            for line in f.readlines():
                skip += 1

                if line.strip() == "END_OBJECT":
                    count += 1

                else:
                    count = 0

                if count == 3:
                    break
            else:
                raise ValueError("Could not parse the header in file %s" % f)

        year = name.split("_")[-3][:4] + "-001T00:00:00.000"

        if re_sampling is not None:
            c = []

            with open(name) as f:
                for line_count, line in enumerate(f.readlines()[skip:]):
                    if line_count % re_sampling == 0:
                        line_array = np.fromstring(line, sep=" ").tolist()

                        c.append(line_array[6:10] + line_array[11:14])

            c = np.stack(c, axis=1)

        else:
            c = np.loadtxt(name, skiprows=skip, usecols=[6, 7, 8, 9, 11, 12, 13]).T

        c[0] = (c[0] - 1.0) * 86400.0 + spice.utc2et(year)

        if name == paths[0]:
            output["b"] = np.array(c[1:4])
            output["time"] = np.array(c[0])
            output["position"] = np.array(c[4:])

        else:
            output["b"] = np.hstack((output["b"], np.array(c[1:4])))
            output["time"] = np.hstack((output["time"], np.array(c[0])))
            output["position"] = np.hstack((output["position"], np.array(c[4:])))

    return output


BASE_URL = "https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2"


def get_regexp(coords="ss1s"):

    assert coords in [
        "ss1s",
        "pc1s",
    ], "coords must be one of 'ss1s', 'pc1s'"

    return re.compile(
        r"(mvn_mag_l2_(\d{7})" + coords + r"_(\d{8})_v(\d+)_r(\d+))\.(sts|xml)"
    )


def best_revision(files, coords="ss1s"):
    """
    Return highest version/release pair.
    """

    FILE_RE = get_regexp(coords)

    return max(
        files,
        key=lambda f: (
            int(FILE_RE.match(f).group(4)),  # version
            int(FILE_RE.match(f).group(5)),  # release
        ),
    )


def download_day(day, rootdir, coords="ss1s", dry_run=False):

    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    yyyymmdd = day.strftime("%Y%m%d")
    doy = day.strftime("%j")

    url = f"{BASE_URL}/{yyyy}/{mm}/"

    html = requests.get(url, timeout=30).text

    candidates = []

    FILE_RE = get_regexp(coords)

    for m in FILE_RE.finditer(html):
        fname = m.group(0)

        if (
            m.group(2) == f"{yyyy}{doy}"
            and m.group(3) == yyyymmdd
            and fname.endswith(".sts")
        ):
            candidates.append(fname)

    if not candidates:
        print(f"No file found for {yyyymmdd}")
        return

    sts_name = best_revision(candidates)

    xml_name = sts_name[:-4] + ".xml"

    if dry_run:
        print(f"Would download {sts_name} and {xml_name} to {rootdir}/{yyyy}/{mm}")
        return

    outdir = Path(rootdir) / yyyy / mm
    outdir.mkdir(parents=True, exist_ok=True)

    for fname in (sts_name, xml_name):
        outfile = outdir / fname

        if os.path.exists(outfile):
            print(f"{outfile} already exists, skipping download.")
            continue

        r = requests.get(url + fname, stream=True, timeout=60)
        r.raise_for_status()

        with open(outfile, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

        print("Downloaded", outfile)


def download_range(start_date, end_date, rootdir, coords="ss1s", dry_run=False):
    assert isinstance(start_date, datetime) and isinstance(end_date, datetime), (
        "start_date and end_date must be datetime objects"
    )

    day = start_date

    while day <= end_date:
        download_day(day, rootdir, coords, dry_run)
        day += timedelta(days=1)


def sync_files():
    start = datetime(2020, 8, 1)
    end = datetime(2025, 12, 30)

    for coords in ["ss1s", "pc1s"]:
        rootdir = f"../data/mag_transfer/mag_{coords}/l2/"

        if not os.path.exists(rootdir):
            raise IOError(f"Directory {rootdir} does not exist - probably it should??!")
        download_range(start, end, rootdir, coords)
