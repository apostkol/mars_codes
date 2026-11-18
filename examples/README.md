# Examples

The examples are intended to be read in this order:

1. `01_Example_MSO_to_MSE.py`
   Converts MAG data into MSE coordinates with `mars_currents.mse_processing`.
   The current calling pattern is `load_data(**data_dict, ...)` and
   `rotated_to_mse(**data_dict, ...)`.

2. `02_Example_binning.py`
   Builds a spherical grid, bins MSE MAG data into cells, calculates currents,
   and produces map-style plots.

3. `03_Tail_Currents.py`
   Extracts currents in the tail region

4. `04_Total_Currents.py`
   Extracts total currents.   

5. SWIA notebooks
   `swia_diff_energy_flux.py`, `swia_moments_sw_conditions.py`, and
   `sw_conditions_data_to_mse.py` are exploratory notebooks for solar-wind
   conditions and SWIA products. They are useful references, but they are not
   currently the cleanest first path through the package.

## Working with Examples

The examples are stored as Python files using [jupytext](https://jupytext.readthedocs.io/),
which allows them to be version-controlled cleanly in Git while still being opened and run as
Jupyter notebooks.

### Opening as Notebooks
Simply open any `.py` file in JupyterLab or VS Code—they will automatically display as
interactive notebooks thanks to jupytext.  Get the VS Code extension.

### Running from Command Line
```bash
uv run python 04_Total_Currents.py
```

### Creating New Examples
When creating new examples, save them as `.py` files in percent format using jupytext.
The pre-commit hook will enforce this and prevent `.ipynb` files from being committed.