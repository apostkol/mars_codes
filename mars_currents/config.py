import os
from pathlib import Path

DATA_ROOT = Path(
    os.getenv(
        "MARS_CURRENTS_DATA_ROOT",
        Path.cwd() / "data",
    )
).expanduser()
DATA = DATA_ROOT
MAG_DATA = DATA_ROOT / "mag_transfer"
RESULTS = DATA_ROOT
KERNELS = DATA_ROOT
PLOTS = DATA_ROOT
