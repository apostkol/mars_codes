"""Compatibility helpers for loading and saving grids."""

from __future__ import annotations

from .grid_model import Grid


def load_grid(grid_dict=None, load_from_file=False, file_path=None, data=None):
    """Create a :class:`Grid` from a configuration dictionary or pickle file.

    Parameters
    ----------
    grid_dict : dict, optional
        Grid configuration. Required when ``load_from_file`` is false.
    load_from_file : bool, default=False
        When true, ``file_path`` is read as a pickle created by ``save_grid``.
    file_path : str or pathlib.Path, optional
        Path to a saved grid pickle.
    data : str, optional
        Data label or source path associated with a newly created grid.

    Returns
    -------
    Grid
        A new Grid instance.
    """

    if (load_from_file and not file_path) or (file_path and not load_from_file):
        raise ValueError(f"I don't understand: {load_from_file=} and {file_path=}")

    if load_from_file and grid_dict:
        raise ValueError(
            "Shouldn't use load_grid with both a `file_path` and `grid_dict`"
            "simultaneously: Do you you want to restore from a dict, or from a file? "
            "Not both at the same time"
        )

    if load_from_file:
        if file_path is None:
            raise ValueError("file_path is required when load_from_file is True")

        print(f"Restoring a grid from {file_path}")
        grid = Grid.from_file(file_path)
        return grid

    else:
        if grid_dict is None:
            raise ValueError("grid_dict is required when load_from_file is False")
        print(f"Restoring a grid from a dict with keys: {list(grid_dict.keys())}")

        return Grid.from_dict(grid_dict)


def save_grid(grid, directory="."):
    """Serialize a grid to a pickle file and return the output path.

    Parameters
    ----------
    grid : mars_currents.grid.Grid
        Grid instance to save.
    directory : str or pathlib.Path, default='.'
        Directory where the pickle file should be written.

    Returns
    -------
    Path
        Path to the saved pickle file.
    """

    return grid.save(directory=directory)
