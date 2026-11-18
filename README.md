# Mars Currents

Turn MAVEN magnetic-field observations into maps of magnetic fields and electric
current density around Mars. The workflow is: prepare observations in Mars
Solar Electric (MSE) coordinates, bin them on a spherical grid, then calculate
and plot currents.

You can work in Python scripts or notebooks. Start with the small example below;
no MAVEN data or SPICE kernels are needed for this first step.

## Get started

With Python 3.10 or newer and uv installed, open a terminal in your local copy
of this repository and run:

```bash
uv sync
uv run python
```

At the Python prompt, create a small grid:

```python
from mars_currents.grid import Grid

grid = Grid(r_lim=(1.0, 2.0), dr=0.2, dth_deg=30, dphi_deg=30)
print(grid)
```

This prints a grid summary and checks that the package is ready to use. Radial
distances are in Mars radii; angular spacing is in degrees. The grid is empty
until you supply observations.

## Analyse observations

Once you have a prepared MSE CSV, the main calculation is:

```python
grid.bin_data("path/to/mag_mse.csv")  # Bin observations and smooth the field
grid.current_densities()             # Calculate currents from the smoothed field
j_magnitude = grid.J_tot             # Current-density magnitude in nA/m²
```

The filename is a placeholder, not bundled example data. Follow the
[example guide](examples/README.md) for data preparation (example 01), binning
and maps (02), and current profiles (03–04). Those workflows require external
data; raw-data preparation also requires SPICE kernels. Set
`MARS_CURRENTS_DATA_ROOT` to your data directory before running them, and check
the filenames in your chosen example.

For paper-specific workflows, see [paper/README.md](paper/README.md).
For field-aligned currents, tests, and notebook setup, see the
[appendix below](#appendix).

## Credits and attribution

This project was originally developed by Apostolos Kolokotronis, with later
development by David Andrews including substantial refactoring, additional
capabilities, tests, and documentation. The codebase is collaborative, and no
attempt is made to assign every file or line to an individual contributor.

Later development was heavily assisted by OpenAI Codex—affectionately known
during this work as *Silicon haunted by monsters from the Id*—used interactively
under human direction and review. Codex is acknowledged as a development tool,
not as a legal author or copyright holder.

## Appendix

### Field-aligned currents

After observations have been binned, calculating currents also creates their
magnetic-field-aligned decomposition:

```python
grid.current_densities(smoothed=True)

# Signed field-aligned density and non-negative perpendicular magnitude
j_parallel = grid.J_parallel
j_perpendicular = grid.J_perp

# Full vectors are available in spherical and Cartesian components
j_perpendicular_spherical = (grid.J_perp_r, grid.J_perp_th, grid.J_perp_phi)
j_parallel_cartesian = (grid.J_parallel_x, grid.J_parallel_y, grid.J_parallel_z)
```

Cells where the selected magnetic field is zero or non-finite receive `NaN`
projection values because the field-aligned direction is undefined. The
`grid.current_density_projection_field` attribute records whether the raw `B`
or smoothed `Bsm` field was used. Use `smoothed=False` to calculate currents
and their projections from the raw binned field.

### Tests

Run the data-independent analytic current-density checks with:

```bash
uv run pytest tests/test_analytic_current_density.py
```

These check known fields, refinement behaviour, azimuth seams, and polar errors.
Other tests cover geometry, binning, derived quantities, and serialization.
The full suite also executes examples that require external data and kernels:

```bash
uv run pytest
```

### Notebooks and scripts

Examples use Jupytext percent-format Python files (`# %%` cells). They can be
read as scripts or opened interactively with Jupytext's notebook integration
or VS Code's Python/Jupyter support.

Start JupyterLab from the repository root with:

```bash
uv run jupyter lab examples/
```

Use Jupytext's notebook integration to open the Python examples as notebooks.
Alternatively, run an example as a script after configuring its inputs:

```bash
uv run python examples/04_Total_Currents.py
```
