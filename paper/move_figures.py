import shutil
from pathlib import Path

figurenames = {
    "Figure1": "<Banana sketch>",
    "Figure2": "<Grid sketch>",
    "All_B.pdf": "data/dr0.04_dth10_dphi18_mag_mse_2014-10-11_2020-08-14_n15_all.csv-SmoothedB.pdf",
    "All_J.pdf": "data/dr0.04_dth10_dphi18_mag_mse_2014-10-11_2020-08-14_n15_all.csv-Currents.pdf",
    "NH_B.pdf": "data/dr0.04_dth10_dphi18_mag_mse_2014-10-11_2020-08-14_n15_NH.csv-SmoothedB.pdf",
    "NH_J.pdf": "data/dr0.04_dth10_dphi18_mag_mse_2014-10-11_2020-08-14_n15_NH.csv-Currents.pdf",
    "SWDist.pdf": "data/mag_mse_2014-10-20_2020-08-13_n20_all_NEW_solar_wind_distributions.pdf",
    "Pdyn_B.pdf": "data/BsmrPdyn.pdf",
    "Pdyn_J.pdf": "data/JthPdyn.pdf",
    "Pdyn_JdotE.pdf": "data/JdotEPdyn.pdf",
    "DaysideI.pdf": "data/NH_tail_currents_dayside.pdf",
    "DaysideJdotE.pdf": "data/PowerProfile_All__NH__High__Intermediate__Low_.pdf",
    "DistJdotE.pdf": "data/JdotEPdyn_distributions.pdf",
    "DistJth.pdf": "data/JthPdyn_distributions.pdf",
}

outdir = Path("../data/PaperFigures/")
outdir.mkdir(parents=True, exist_ok=True)

for figure, source in figurenames.items():
    source = Path("../" + source)

    if source.is_file():
        destination = outdir / figure
        shutil.copy2(source, destination)
        print(f"{source} -> {destination}")
    else:
        print(f"Skipping {figure}: {source} not found")
