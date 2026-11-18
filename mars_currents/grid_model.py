"""Grid model for binning magnetic fields and deriving currents."""

from __future__ import annotations

import pickle
from pathlib import Path

import astropy.convolution
import numpy as np
import pandas as pd

from .generic import cart_to_sph_vf, sph_to_cart_vf, spherical_unit_vectors
from .grid_calculus import curl, div, sigma_gradient
from .grid_cell import Cell, _nan_field_components


class Grid:
    """Spherical grid used to bin magnetic-field data and derive currents.

    A ``Grid`` is built from a dictionary containing radial, colatitude, and
    azimuth limits plus spacing values. Angles in the public ``grid_dict`` are
    supplied in degrees for compatibility with the original notebooks, then
    converted to radians internally.
    """

    def __init__(
        self,
        *,
        dr: float,
        dth_deg: float,
        dphi_deg: float,
        r_lim: tuple,
        th_lim_deg: tuple = (0, 180),
        phi_lim_deg: tuple = (-180, 180),
        file_name: str | None = None,
        name: str | None = None,
        data_source: str | None = None,
        radius_km: float = 3396.0,
    ):
        """Create a grid from configuration parameters.

        Parameters
        ----------
        dr : float
            Radial spacing.
        dth_deg : float
            Colatitude spacing in degrees.
        dphi_deg : float
            Azimuth spacing in degrees.
        r_lim : tuple of float
            Radial limits (min, max).
        th_lim_deg : tuple of float
            Colatitude limits in degrees (min, max).
            Default (0, 180) deg
        phi_lim_deg : tuple of float
            Azimuth limits in degrees (min, max).
            Default (-180, 180)
        data_source : str, optional
            Data label or source identifier.
        name : str, optional
            Grid name identifier. If not provided, auto-generated from spacing.
        radius_km : float
            Scale between km and grid radial bins.  Default = 3396. for Mars.
        """
        self.dr = dr
        self.dth_deg = dth_deg
        self.dphi_deg = dphi_deg
        self.r_lim = r_lim
        self.th_lim_deg = th_lim_deg
        self.phi_lim_deg = phi_lim_deg

        self.radius_km = radius_km

        self.data_source = data_source

        if name is None:
            name = f"dr{dr}_dth{dth_deg}_dphi{dphi_deg}"
        self.name = name

        if file_name is None:
            file_name = name
            if data_source:
                file_name += "_" + str(data_source)
        self.file_name = file_name

        th_lim = np.deg2rad(self.th_lim_deg)
        phi_lim = np.deg2rad(self.phi_lim_deg)

        self.dth = np.deg2rad(self.dth_deg)
        self.dphi = np.deg2rad(self.dphi_deg)

        (
            self.cells,
            self.r_bins,
            self.th_bins,
            self.phi_bins,
            self.r,
            self.th,
            self.phi,
        ) = self._set_grid_cells(
            self.r_lim,
            th_lim,
            phi_lim,
            self.dr,
            self.dth,
            self.dphi,
        )
        self._initialize_binned_products()

        # Book keeping
        self._never_calculated_currents = True

    def __repr__(self):
        """Return a concise representation of the grid configuration."""
        return (
            f"Grid(name={self.name!r}, file_name={self.file_name!r}, "
            f"shape={self.shape}, dr={self.dr!r}, dth_deg={self.dth_deg!r}, "
            f"dphi_deg={self.dphi_deg!r}, r_lim={self.r_lim!r}, "
            f"th_lim_deg={self.th_lim_deg!r}, phi_lim_deg={self.phi_lim_deg!r})"
        )

    def __str__(self):
        """Return a human-readable summary of the grid and loaded products."""
        name = self.name if self.name is not None else "<unnamed>"
        file_name = self.file_name if self.file_name is not None else "<none>"
        shape = self.shape
        fields = self._available_grid_products()
        fields_text = ", ".join(fields) if fields else "none"

        return (
            f"Grid {name}\n"
            f"  file_name: {file_name}\n"
            f"  shape: phi={shape[0]}, theta={shape[1]}, r={shape[2]} "
            f"({len(self.cells)} cells)\n"
            f"  spacing: dr={self.dr}, dtheta={self.dth_deg} deg, "
            f"dphi={self.dphi_deg} deg\n"
            f"  limits: r={self.r_lim}, theta={self.th_lim_deg} deg, "
            f"phi={self.phi_lim_deg} deg\n"
            f"  products: {fields_text}"
        )

    @property
    def shape(self):
        """Return the grid-array shape used by map products."""
        return (len(self.phi), len(self.th), len(self.r))

    def _available_grid_products(self):
        """Return computed grid products present as full-size arrays."""
        products = []
        shape = self.shape
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(value, np.ndarray) and value.shape == shape:
                products.append(name)
        return sorted(products)

    def _cell_scalar_array(self, attr, fill_value=np.nan):
        """Build a grid-shaped array from a scalar attribute on each cell."""
        values = np.full(self.shape, fill_value, dtype=float)
        for cell in self.cells:
            values[cell.phi_idx, cell.th_idx, cell.r_idx] = getattr(
                cell,
                attr,
                fill_value,
            )
        return values

    def _nan_grid_array(self):
        """Return a grid-shaped floating array filled with NaNs."""
        return np.full(self.shape, np.nan, dtype=float)

    def _initialize_binned_products(self):
        """Define empty grid-level products populated by :meth:`bin_data`."""
        for attr in (
            "Br",
            "Bth",
            "Bphi",
            "Bx",
            "By",
            "Bz",
            "B_tot",
            "sigma_Br",
            "sigma_Bth",
            "sigma_Bphi",
            "measurement_counts",
        ):
            setattr(self, attr, self._nan_grid_array())

        phi_2d, th_2d, _ = np.meshgrid(self.phi, self.th, self.r, indexing="ij")
        self.th_2d = th_2d
        self.phi_2d = phi_2d

        self.X_cells = []
        self.Y_cells = []
        self.Z_cells = []
        self.Bx_cells = []
        self.By_cells = []
        self.Bz_cells = []

    def _clear_binned_products(self):
        """Reset binned data products without changing grid geometry."""
        for attr in (
            "Br",
            "Bth",
            "Bphi",
            "Bx",
            "By",
            "Bz",
            "B_tot",
            "sigma_Br",
            "sigma_Bth",
            "sigma_Bphi",
            "measurement_counts",
        ):
            getattr(self, attr)[:] = np.nan

        self.X_cells.clear()
        self.Y_cells.clear()
        self.Z_cells.clear()
        self.Bx_cells.clear()
        self.By_cells.clear()
        self.Bz_cells.clear()

    def _populate_cartesian_vector(self, field, spherical_components=None):
        """Store Cartesian components and magnitude for a spherical vector.

        Parameters
        ----------
        field : str
            Component prefix, such as ``"B"``, ``"Bsm"``, or ``"J"``.
        spherical_components : array-like, optional
            Explicit ``(..., 3)`` array ordered as radial, colatitudinal, and
            azimuthal components. By default, the arrays are read from the
            corresponding ``<field>r``, ``<field>th``, and ``<field>phi``
            attributes.
        """
        spherical_names = (f"{field}r", f"{field}th", f"{field}phi")
        cartesian_names = (f"{field}x", f"{field}y", f"{field}z")
        magnitude_name = f"{field.rstrip('_')}_tot"

        if spherical_components is None:
            components = []
            for name in spherical_names:
                if not hasattr(self, name):
                    raise AttributeError(f"Grid has no {name!r}.")
                values = np.asarray(getattr(self, name))
                if values.shape != self.shape:
                    raise ValueError(
                        f"{name!r} has shape {values.shape}; expected "
                        f"{self.shape} for grid array order (phi, theta, r).",
                    )
                components.append(values)
            spherical_components = np.stack(components, axis=-1)
        else:
            spherical_components = np.asarray(spherical_components)
            expected_shape = self.shape + (3,)
            if spherical_components.shape != expected_shape:
                raise ValueError(
                    f"Spherical components have shape {spherical_components.shape}; "
                    f"expected {expected_shape}.",
                )

        cartesian_components = sph_to_cart_vf(
            spherical_components,
            self.th_2d,
            self.phi_2d,
        )

        for name, values in zip(cartesian_names, cartesian_components):
            setattr(self, name, values)
        magnitude = np.linalg.norm(spherical_components, axis=-1)
        setattr(self, magnitude_name, magnitude)
        if field == "JxB_":
            self.JxB_mag = magnitude

        for cell in self.cells:
            idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
            for name, values in zip(cartesian_names, cartesian_components):
                setattr(cell, name, values[idx])
            setattr(cell, magnitude_name, magnitude[idx])

        if field == "B":
            self.Bx_cells = [cell.Bx for cell in self.cells]
            self.By_cells = [cell.By for cell in self.cells]
            self.Bz_cells = [cell.Bz for cell in self.cells]

        return self

    def _refresh_cartesian_components(self):
        """Regenerate Cartesian components for spherical vectors already present.

        This keeps older serialized grids compatible without recalculating
        expensive derived products. A vector is converted only when all three
        of its spherical component arrays are available.
        """
        fields = {
            "B": ("Br", "Bth", "Bphi"),
            "Bsm": ("Bsmr", "Bsmth", "Bsmphi"),
            "J": ("Jr", "Jth", "Jphi"),
            "JxB_": ("JxB_r", "JxB_th", "JxB_phi"),
        }

        for field, spherical_names in fields.items():
            if all(hasattr(self, name) for name in spherical_names):
                self._populate_cartesian_vector(field)

        return self

    def _populate_binned_products_from_cells(self):
        """Copy per-cell binned values into grid-level arrays."""
        for cell in self.cells:
            idx = (cell.phi_idx, cell.th_idx, cell.r_idx)

            cell.Br = cell.B_mse_mean[0]
            cell.Bth = cell.B_mse_mean[1]
            cell.Bphi = cell.B_mse_mean[2]

            self.Br[idx] = cell.Br
            self.Bth[idx] = cell.Bth
            self.Bphi[idx] = cell.Bphi
            self.sigma_Br[idx] = cell.sigma_Br
            self.sigma_Bth[idx] = cell.sigma_Bth
            self.sigma_Bphi[idx] = cell.sigma_Bphi
            self.measurement_counts[idx] = cell.n_measurements

    def sync_cell_fields_from_grid(
        self,
        field_attrs=("Br", "Bth", "Bphi"),
        update_mean=True,
    ):
        """Copy grid-level magnetic-field arrays onto their matching cells.

        Current-density calculations use ``Cell`` attributes as their source
        field values. This helper is useful after synthetic or externally
        calculated field arrays are assigned directly to the grid.
        """
        for attr in field_attrs:
            values = np.asarray(getattr(self, attr))
            if values.shape != self.shape:
                raise ValueError(
                    f"{attr!r} has shape {values.shape}; expected {self.shape} "
                    "for grid array order (phi, theta, r).",
                )

            for cell in self.cells:
                idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
                setattr(cell, attr, values[idx])

        if update_mean and all(hasattr(self, attr) for attr in ("Br", "Bth", "Bphi")):
            for cell in self.cells:
                cell.B_mse_mean = [cell.Br, cell.Bth, cell.Bphi]

        if all(attr in field_attrs for attr in ("Br", "Bth", "Bphi")):
            self._populate_cartesian_vector("B")

        self._ensure_cell_neighbour_references()

        return self

    def _refresh_current_density_arrays(self):
        """Copy current-density cell values to grid-level arrays."""
        for attr in (
            "Jr",
            "Jth",
            "Jphi",
            "sigma_Jr",
            "sigma_Jth",
            "sigma_Jphi",
            "divB_curlB",
            "divJ_magJ",
        ):
            setattr(self, attr, self._cell_scalar_array(attr))

    def project_current_densities(self, field="B"):
        """Decompose current density parallel and perpendicular to a field.

        The signed field-aligned current density is ``J dot b_hat``. The
        perpendicular current is stored as a vector and as its non-negative
        magnitude. All quantities retain the units of the current-density
        components.

        Parameters
        ----------
        field : str, default="B"
            Cell attribute prefix for the magnetic field. Use ``"B"`` for the
            raw field or ``"Bsm"`` for the smoothed field.

        Returns
        -------
        Grid
            The grid with parallel and perpendicular current products added.
        """
        field_names = tuple(f"{field}{component}" for component in ("r", "th", "phi"))
        current_names = ("Jr", "Jth", "Jphi")

        missing = [
            name
            for name in (*field_names, *current_names)
            if not all(hasattr(cell, name) for cell in self.cells)
        ]
        if missing:
            missing_text = ", ".join(repr(name) for name in missing)
            raise AttributeError(f"Grid cells have no attributes {missing_text}.")

        magnetic_field = np.stack(
            [self._cell_scalar_array(name) for name in field_names],
            axis=-1,
        )
        current_density = np.stack(
            [self._cell_scalar_array(name) for name in current_names],
            axis=-1,
        )
        for name, values in zip(current_names, np.moveaxis(current_density, -1, 0)):
            setattr(self, name, values)
        self._populate_cartesian_vector("J", current_density)

        field_magnitude = np.linalg.norm(magnetic_field, axis=-1)
        valid = (
            np.all(np.isfinite(magnetic_field), axis=-1)
            & np.all(np.isfinite(current_density), axis=-1)
            & (field_magnitude > 0.0)
        )

        field_direction = np.full_like(magnetic_field, np.nan, dtype=float)
        np.divide(
            magnetic_field,
            field_magnitude[..., np.newaxis],
            out=field_direction,
            where=valid[..., np.newaxis],
        )

        parallel_signed = np.sum(current_density * field_direction, axis=-1)
        parallel_signed[~valid] = np.nan
        parallel = parallel_signed[..., np.newaxis] * field_direction
        perpendicular = current_density - parallel
        perpendicular[~valid] = np.nan
        perpendicular_magnitude = np.linalg.norm(perpendicular, axis=-1)

        self.J_parallel = parallel_signed
        self.J_perp = perpendicular_magnitude
        self.J_parallel_r = parallel[..., 0]
        self.J_parallel_th = parallel[..., 1]
        self.J_parallel_phi = parallel[..., 2]
        self.J_perp_r = perpendicular[..., 0]
        self.J_perp_th = perpendicular[..., 1]
        self.J_perp_phi = perpendicular[..., 2]
        self.current_density_projection_field = field

        self._populate_cartesian_vector("J_parallel_", parallel)
        self._populate_cartesian_vector("J_perp_", perpendicular)

        for cell in self.cells:
            idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
            for name in (
                "J_parallel",
                "J_parallel_r",
                "J_parallel_th",
                "J_parallel_phi",
                "J_perp",
                "J_perp_r",
                "J_perp_th",
                "J_perp_phi",
            ):
                setattr(cell, name, getattr(self, name)[idx])

        return self

    def _ensure_smoothed_field_aliases(self):
        """Expose smoothed magnetic fields with the ``Bsm`` prefix convention."""
        aliases = {
            "Bsmr": "Br_sm",
            "Bsmth": "Bth_sm",
            "Bsmphi": "Bphi_sm",
        }

        for canonical, legacy in aliases.items():
            if not hasattr(self, canonical) and hasattr(self, legacy):
                setattr(self, canonical, getattr(self, legacy))

        for cell in self.cells:
            for canonical, legacy in aliases.items():
                if not hasattr(cell, canonical) and hasattr(self, canonical):
                    idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
                    setattr(cell, canonical, getattr(self, canonical)[idx])
                if not hasattr(cell, canonical) and hasattr(cell, legacy):
                    setattr(cell, canonical, getattr(cell, legacy))

            cell.sigma_Bsmr = cell.sigma_Br
            cell.sigma_Bsmth = cell.sigma_Bth
            cell.sigma_Bsmphi = cell.sigma_Bphi

        return self

    def _ensure_cell_neighbour_references(self):
        """Replace serialized neighbour dictionaries with live Cell references."""
        cells_by_indices = {
            (cell.r_idx, cell.th_idx, cell.phi_idx): cell for cell in self.cells
        }

        for cell in self.cells:
            for coordinate in ("r", "th", "phi"):
                for direction in (-1, +1):
                    neighbour = cell.neighbours[coordinate][direction]
                    if not isinstance(neighbour, dict):
                        continue

                    key = (
                        neighbour.get("r_idx"),
                        neighbour.get("th_idx"),
                        neighbour.get("phi_idx"),
                    )
                    cell.neighbours[coordinate][direction] = cells_by_indices.get(key)

        return self

    @classmethod
    def from_dict(cls, grid_dict: dict) -> Grid:
        """Reconstruct a Grid from a dictionary representation.

        Parameters
        ----------
        grid_dict : dict
            Dictionary containing grid parameters. Should contain keys like
            'dr', 'dth_deg', 'dphi_deg', 'r_lim', 'th_lim_deg', 'phi_lim_deg',
            'data', 'name', and optionally 'r_bins', 'th_bins', 'phi_bins',
            'r', 'th', 'phi', 'cells_dict' for loaded grids.

        Returns
        -------
        Grid
            A new Grid instance reconstructed from the dictionary.
        """
        # Extract parameters needed for Grid construction
        grid_params = {
            "dr": grid_dict["dr"],
            "dth_deg": grid_dict["dth_deg"],
            "dphi_deg": grid_dict["dphi_deg"],
            "r_lim": grid_dict["r_lim"],
            "th_lim_deg": grid_dict["th_lim_deg"],
            "phi_lim_deg": grid_dict["phi_lim_deg"],
        }

        # Add optional parameters if present
        # Backwards compatible with old pickles
        if "data" in grid_dict:
            grid_params["data_source"] = grid_dict["data"]
        elif "data_source" in grid_dict:
            grid_params["data_source"] = grid_dict["data_source"]

        if "grid_name" in grid_dict:
            grid_params["name"] = grid_dict["grid_name"]
        elif "name" in grid_dict:
            grid_params["name"] = grid_dict["name"]

        # Only used for Mars so far, so I feel OK about hard-coding this:
        if "radius_km" not in grid_dict:
            grid_params["radius_km"] = 3396.0
            # warnings.warn(
            #     f"Loading grid from dict {id(grid_dict)} with {len(grid_dict)} keys: "
            #     f"Assuming {grid_params['radius_km']=} km."
            # )
        else:
            grid_params["radius_km"] = grid_dict["radius_km"]

        # Create grid using standard constructor
        grid = cls(**grid_params)

        # Restore all additional attributes from dict (cells_dict, field arrays, etc.)
        grid._restore_from_dict(grid_dict)
        grid._ensure_smoothed_field_aliases()
        grid._refresh_cartesian_components()

        return grid

    def to_dict(self) -> dict:
        """Convert the grid and its cells into a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing all grid parameters and cell data,
            suitable for serialization or reconstruction via from_dict().
        """
        grid_dict = self.__dict__.copy()

        # Remove cells as they'll be serialized in cells_dict
        save_dict = {
            k: grid_dict[k]
            for k in grid_dict.keys() - {"cells"}
            if not k.startswith("_")
        }

        save_dict["cells_dict"] = {}
        c = 0

        for cell in self.cells:
            cell_dict = cell.__dict__
            cell_save_dict = {
                k: cell_dict[k]
                for k in cell_dict.keys()
                - {
                    "r",
                    "th",
                    "phi",
                    "neighbours",
                    "r_bins",
                    "th_bins",
                    "phi_bins",
                    "neighbours_bins",
                }
            }

            save_dict["cells_dict"].update({c: cell_save_dict})

            neighbours_indices = {}
            for coordinate in cell_dict["neighbours"].keys():
                neighbours_indices[coordinate] = {}
                for direction in cell_dict["neighbours"][coordinate].keys():
                    neighbour_cell = cell_dict["neighbours"][coordinate][direction]
                    if neighbour_cell is not None:
                        if isinstance(neighbour_cell, dict):
                            neighbours_indices[coordinate][direction] = {
                                "r_idx": neighbour_cell.get("r_idx"),
                                "th_idx": neighbour_cell.get("th_idx"),
                                "phi_idx": neighbour_cell.get("phi_idx"),
                            }
                        else:
                            neighbours_indices[coordinate][direction] = {
                                "r_idx": neighbour_cell.r_idx,
                                "th_idx": neighbour_cell.th_idx,
                                "phi_idx": neighbour_cell.phi_idx,
                            }
                    else:
                        neighbours_indices[coordinate][direction] = None

            save_dict["cells_dict"][c]["neighbours_indices"] = neighbours_indices
            c += 1

        return save_dict

    def _restore_from_dict(self, grid_dict: dict) -> None:
        """Restore grid cells and related state from a dictionary representation.

        This is called internally by from_dict() to restore cells and neighbours
        from a previously serialized grid.

        Parameters
        ----------
        grid_dict : dict
            Dictionary containing 'cells_dict' and grid state from to_dict().
        """
        # Restore cells if cells_dict is present
        if "cells_dict" in grid_dict:
            self.cells_dict = grid_dict["cells_dict"]
            # Ensure r, th, phi arrays are set before calling _set_grid_cells
            if "r" in grid_dict:
                self.r = grid_dict["r"]
            if "th" in grid_dict:
                self.th = grid_dict["th"]
            if "phi" in grid_dict:
                self.phi = grid_dict["phi"]
            self.cells = self._set_grid_cells(cells_dict=self.cells_dict)
            # Rebuild neighbor connections from indices
            self._rebuild_neighbours_from_indices()

        # Restore all other grid attributes (field arrays, etc.)
        # Skip constructor-related attributes and internal attributes
        skip_attrs = {
            "dr",
            "dth_deg",
            "dphi_deg",
            "dth",
            "dphi",
            "r_lim",
            "th_lim_deg",
            "phi_lim_deg",
            "data",
            "name",
            "cells",
            "cells_dict",
            "r",
            "th",
            "phi",
            "r_bins",
            "th_bins",
            "phi_bins",
        }

        for key, value in grid_dict.items():
            if key not in skip_attrs and not key.startswith("_"):
                setattr(self, key, value)

    def _rebuild_neighbours_from_indices(self) -> None:
        """Rebuild neighbor Cell references from stored indices.

        After deserialization, neighbors are stored as index dicts in cells_dict.
        This method reconstructs the actual Cell object references using index lookups.
        """
        # Build a quick lookup dict for cells by their indices
        cells_by_indices = {}
        for cell in self.cells:
            cells_by_indices[(cell.r_idx, cell.th_idx, cell.phi_idx)] = cell

        # Reconnect neighbors using the stored indices
        for cell in self.cells:
            neighbours_data = cell.__dict__.get("neighbours_indices", {})
            if neighbours_data:
                for coordinate in ["r", "th", "phi"]:
                    for direction in [-1, +1]:
                        indices_info = neighbours_data.get(coordinate, {}).get(
                            direction
                        )
                        if indices_info is not None:
                            key = (
                                indices_info["r_idx"],
                                indices_info["th_idx"],
                                indices_info["phi_idx"],
                            )
                            neighbour = cells_by_indices.get(key)
                            cell.set_neighbour(coordinate, direction, neighbour)
                # Clean up the temporary indices dict to avoid confusion
                del cell.neighbours_indices

    @classmethod
    def from_file(cls, file_path: str | Path) -> Grid:
        """Load a Grid from a pickle file.

        Parameters
        ----------
        file_path : str or Path
            Path to a saved grid pickle file.

        Returns
        -------
        Grid
            A new Grid instance reconstructed from the pickle file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("rb") as f:
            grid_dict = pickle.load(f)

        return cls.from_dict(grid_dict)

    def save(self, directory: str | Path = ".") -> Path:
        """Serialize this grid to a pickle file.

        Parameters
        ----------
        directory : str or Path, optional
            Directory where the pickle file should be written. Default is '.'.

        Returns
        -------
        Path
            Path to the saved pickle file.
        """
        grid_dict = self.to_dict()
        filename = self.file_name
        if not filename.endswith(".pickle"):
            filename = filename + ".pickle"
        path = Path(directory) / filename

        with path.open("wb") as f:
            pickle.dump(grid_dict, f)

        return path

    def sample_fields(
        self,
        r,
        theta,
        phi,
        field_name="Jr",
        processor=None,
        return_indices=False,
        return_centers=False,
        return_cells=False,
    ):
        """Extract field/current values at specified positions using nearest-neighbor
        lookup.

        Samples the grid at specified spherical coordinates by finding the
        nearest cell
        center and returning the requested field value from that cell.

        Parameters
        ----------
        r : float or array-like
            Radial coordinate(s) in km
        theta : float or array-like
            Colatitude coordinate(s) in radians
        phi : float or array-like
            Azimuth coordinate(s) in radians
        field_name : str, optional
            Name of the field/current to extract (e.g., "Jr", "Jth", "Jphi",
            "Br", "Bth", "Bphi", or "B_mse_mean"). Default is "Jr".
        processor: callable, optional
            Optional function to apply to the extracted field values (e.g.,
            np.linalg.norm for vector fields). Default is None (no processing).
            Only applied if the field value is an array or list.  Processor should
            always return the same shape regardless of input, so that the output
            shape can be determined from a single sample value.
        return_indices : bool, optional
            If True, also return the (r_idx, th_idx, phi_idx) of nearest cells.
            Default is False.
        return_centers : bool, optional
            If True, also return the center coordinates (r_c, th_c, phi_c) of nearest
            cells. Default is False.
        return_cells: bool, optional
            If True, also return the Cell objects of nearest cells. Default is False.

        Returns
        -------
        values : float or ndarray
            Extracted field values at each position
        indices : tuple of ndarrays, optional
            If return_indices=True: (r_indices, th_indices, phi_indices)
        centers : tuple of ndarrays, optional
            If return_centers=True: (r_centers, th_centers, phi_centers)
        cells : tuple of Cell objects, optional
            If return_cells=True: (cell_objects)
        """

        assert hasattr(self.cells[0], field_name), (
            f"Cells do not have attribute {field_name!r}"
        )
        assert hasattr(processor, "__call__") or processor is None, (
            "Processor must be a callable function or None"
        )

        # Check that processor exists if field value is an array or list
        if processor is None and isinstance(
            getattr(self.cells[0], field_name), (list, np.ndarray)
        ):
            raise ValueError(
                f"Field '{field_name}' is an array or list. A processor function"
                f"ust be provided to extract a single value."
            )

        # Convert inputs to arrays for uniform handling
        r_arr = np.atleast_1d(np.asarray(r))
        theta_arr = np.atleast_1d(np.asarray(theta))
        phi_arr = np.atleast_1d(np.asarray(phi))

        # Ensure all have the same shape
        if not (r_arr.shape == theta_arr.shape == phi_arr.shape):
            raise ValueError(
                f"Position arrays must have the same shape: "
                f"r.shape={r_arr.shape}, theta.shape={theta_arr.shape}, "
                f"phi.shape={phi_arr.shape}"
            )

        # Flatten for easier processing
        original_shape = r_arr.shape
        r_flat = r_arr.ravel()
        theta_flat = theta_arr.ravel()
        phi_flat = phi_arr.ravel()

        # Initialize output arrays
        # For values, we need to determine the output shape based on the field value
        # and processor if processor is needed.
        if processor is not None:
            # We require the processor to return the correct shape even with no input
            shape = processor(self.cells[0].__getattribute__(field_name))
            values = np.empty((len(r_flat),) + np.shape(shape))
        else:
            values = np.empty(r_flat.shape)

        r_indices = np.empty(r_flat.shape, dtype=int)
        th_indices = np.empty(r_flat.shape, dtype=int)
        phi_indices = np.empty(r_flat.shape, dtype=int)
        r_centers = np.empty(r_flat.shape)
        th_centers = np.empty(r_flat.shape)
        phi_centers = np.empty(r_flat.shape)
        cells = []

        # For each query position, find the nearest cell
        for idx, (r_pos, theta_pos, phi_pos) in enumerate(
            zip(r_flat, theta_flat, phi_flat)
        ):
            # Create 3D coordinate grids from 1D arrays
            r_grid, th_grid, phi_grid = np.meshgrid(
                self.r, self.th, self.phi, indexing="ij"
            )

            # Compute distances to all cell centers
            distances = np.sqrt(
                (r_grid - r_pos) ** 2
                + (th_grid - theta_pos) ** 2
                + (phi_grid - phi_pos) ** 2
            )

            # Find nearest cell (linear index)
            nearest_linear_idx = np.argmin(distances)

            # Convert linear index to 3D indices using the mesh shape
            r_idx, th_idx, phi_idx = np.unravel_index(
                nearest_linear_idx, distances.shape
            )

            # Get the cell
            cell = self.cells[nearest_linear_idx]
            # print("DBG [Pos]:" + str((idx, r_pos, theta_pos, phi_pos)))
            # print("DBG [Idx]:" + str((idx, r_idx, th_idx, phi_idx)))
            # print(cell)

            # Extract the requested field value
            if hasattr(cell, field_name):
                field_value = getattr(cell, field_name)
                # Handle array fields like B_mse_mean
                # (extract magnitude or first component)
                if isinstance(field_value, (list, np.ndarray)):
                    # print("DBG [Field Value]:" + str(field_value),
                    #       field_value.shape)
                    values[idx] = processor(field_value)
                else:
                    values[idx] = field_value
            else:
                raise AttributeError(
                    f"Field '{field_name}' not found in cell at index"
                    f"{nearest_linear_idx}"
                )

            # Store indices and centers if requested
            r_indices[idx] = r_idx
            th_indices[idx] = th_idx
            phi_indices[idx] = phi_idx
            r_centers[idx] = cell.r_c
            th_centers[idx] = cell.th_c
            phi_centers[idx] = cell.phi_c
            cells.append(cell)

        # Reshape output to match input shape
        # Redundant.
        # values = values.reshape(original_shape)

        # Build return tuple based on optional parameters
        result = (values,)
        if return_indices:
            result += (
                (
                    r_indices.reshape(original_shape),
                    th_indices.reshape(original_shape),
                    phi_indices.reshape(original_shape),
                ),
            )
        if return_centers:
            result += (
                (
                    r_centers.reshape(original_shape),
                    th_centers.reshape(original_shape),
                    phi_centers.reshape(original_shape),
                ),
            )
        if return_cells:
            result += (tuple(cells),)

        # Return single value or tuple based on what was requested
        return result[0] if len(result) == 1 else result

    def _set_grid_cells(
        self,
        r_lim=None,
        th_lim=None,
        phi_lim=None,
        dr=None,
        dth=None,
        dphi=None,
        cells_dict=None,
    ):
        """Create cells and connect their direct neighbours."""

        # If cells_dict is provided, restore from serialized state
        if cells_dict is not None:
            cells = []

            for _, cell_dict in cells_dict.items():
                cell = Cell(
                    cell_dict["dr"],
                    cell_dict["dth"],
                    cell_dict["dphi"],
                    cell_dict["r_idx"],
                    cell_dict["th_idx"],
                    cell_dict["phi_idx"],
                    self.r,
                    self.th,
                    self.phi,
                )

                for attr in cell_dict.keys():
                    cell.__setattr__(attr, cell_dict[attr])

                cells.append(cell)

            return cells

        else:
            r = np.arange(r_lim[0], r_lim[1] + dr, dr)
            th = np.arange(th_lim[0] + dth / 2, th_lim[1] + dth / 2, dth)
            phi = np.arange(phi_lim[0] + dphi / 2, phi_lim[1] + dphi / 2, dphi)

            r_bins = np.array(
                [
                    r[0] + (2 * n - 1) * dr / 2
                    for n in range(len(r))
                    if (r[0] + (n - 1) * dr) <= r[-1]
                ],
            )
            th_bins = np.array(
                [
                    th[0] + (2 * n - 1) * dth / 2
                    for n in range(len(th))
                    if (th[0] + (n - 1) * dth) <= th[-1]
                ],
            )
            phi_bins = np.array(
                [
                    phi[0] + (2 * n - 1) * dphi / 2
                    for n in range(len(phi))
                    if (phi[0] + (n - 1) * dphi) <= phi[-1]
                ],
            )

            cells = []

            for i in range(len(r)):
                r_shell = np.digitize(r[i], r_bins, right=True)

                for j in range(len(th)):
                    th_shell = np.digitize(th[j], th_bins, right=True)

                    for k in range(len(phi)):
                        phi_shell = np.digitize(phi[k], phi_bins, right=True)
                        r_bins_digitized = np.digitize(r, r_bins, right=True)
                        th_bins_digitized = np.digitize(th, th_bins, right=True)
                        phi_bins_digitized = np.digitize(phi, phi_bins, right=True)

                        cell = Cell(
                            dr,
                            dth,
                            dphi,
                            i,
                            j,
                            k,
                            r,
                            th,
                            phi,
                            r_bin=r_shell,
                            th_bin=th_shell,
                            phi_bin=phi_shell,
                            r_bins=r_bins_digitized,
                            th_bins=th_bins_digitized,
                            phi_bins=phi_bins_digitized,
                        )
                        cells.append(cell)
                        cell.set_neighbours_bins()

            # Build a fast lookup dict for cells by (r_bin, th_bin, phi_bin) tuple
            cells_by_bins = {(c.r_bin, c.th_bin, c.phi_bin): c for c in cells}

            # Connect neighbors using bin index lookup (much faster than O(n²) search)
            for cell in cells:
                for coordinate in ["r", "th", "phi"]:
                    for direction in [-1, +1]:
                        neighbor_bins = cell.neighbours_bins[coordinate][direction]
                        lookup_key = (
                            neighbor_bins["r_bin"],
                            neighbor_bins["th_bin"],
                            neighbor_bins["phi_bin"],
                        )
                        # Handle NaN boundaries (which can't be found in lookup dict)
                        if not any(
                            np.isnan(v)
                            for v in lookup_key
                            if isinstance(v, (int, float))
                        ):
                            neighbour = cells_by_bins.get(lookup_key)
                            cell.set_neighbour(coordinate, direction, neighbour)

            return cells, r_bins, th_bins, phi_bins, r, th, phi

    def current_densities(self, smoothed=True):
        """Calculate current-density components and uncertainties for each cell.

        Parameters
        ----------
        smoothed : bool, default=True
            If true, calculate currents from ``Bsmr``, ``Bsmth``, and
            ``Bsmphi`` instead of ``Br``, ``Bth``, and ``Bphi``.
        """

        mu0 = 1.25663706212e-6
        field = "Bsm" if smoothed else "B"
        if smoothed:
            self._ensure_smoothed_field_aliases()

        self._ensure_cell_neighbour_references()

        for cell in self.cells:
            curlB = np.array(curl(cell, field=field))
            J = 1 / (self.radius_km * 1e3) * 1 / (mu0) * curlB

            cell.divB_curlB = abs(div(cell, field=field)) / np.linalg.norm(curlB)

            cell.Jr, cell.Jth, cell.Jphi = J[0], J[1], J[2]

        for cell in self.cells:
            cell.divJ_magJ = abs(div(cell, field="J")) / np.sqrt(
                cell.Jr**2 + cell.Jth**2 + cell.Jphi**2,
            )

            cell.sigma_Jr = (
                1
                / mu0
                * np.sqrt(
                    (
                        1
                        / (cell.r_c * self.radius_km * 1e3)
                        * sigma_gradient(
                            cell,
                            field,
                            "phi",
                            "th",
                        )
                    )
                    ** 2
                    + (
                        np.cos(cell.th_c)
                        / (cell.r_c * self.radius_km * 1e3 * np.sin(cell.th_c))
                        * getattr(cell, "sigma_" + field + "phi")
                    )
                    ** 2
                    + (
                        1
                        / (cell.r_c * self.radius_km * 1e3 * np.sin(cell.th_c))
                        * sigma_gradient(
                            cell,
                            field,
                            "th",
                            "phi",
                        )
                    )
                    ** 2,
                )
            )

            cell.sigma_Jth = (
                1
                / mu0
                * np.sqrt(
                    (
                        1
                        / (cell.r_c * self.radius_km * 1e3 * np.sin(cell.th_c))
                        * sigma_gradient(
                            cell,
                            field,
                            "r",
                            "phi",
                        )
                    )
                    ** 2
                    + (
                        sigma_gradient(
                            cell,
                            field,
                            "phi",
                            "r",
                        )
                        / (self.radius_km * 1e3)
                    )
                    ** 2
                    + (
                        1
                        / (cell.r_c * self.radius_km * 1e3)
                        * getattr(cell, "sigma_" + field + "phi")
                    )
                    ** 2,
                )
            )

            cell.sigma_Jphi = (
                1
                / mu0
                * np.sqrt(
                    (
                        sigma_gradient(
                            cell,
                            field,
                            "th",
                            "r",
                        )
                        / (self.radius_km * 1e3)
                    )
                    ** 2
                    + (
                        1
                        / (cell.r_c * self.radius_km * 1e3)
                        * getattr(cell, "sigma_" + field + "th")
                    )
                    ** 2
                    + (
                        1
                        / (cell.r_c * self.radius_km * 1e3)
                        * sigma_gradient(
                            cell,
                            field,
                            "r",
                            "th",
                        )
                    )
                    ** 2,
                )
            )  # / abs(cell.Jphi)

        self._refresh_current_density_arrays()
        self.project_current_densities(field=field)

    def grid_to_dict(self):
        """Convert the grid and its cells into a pickle-friendly dictionary.

        .. deprecated::
            Use :meth:`to_dict` instead. This method is kept for backward
            compatibility and stores result in ``self.save_dict``.
        """
        self.save_dict = self.to_dict()

    def currents_uncertainties(self):
        """Populate grid arrays for current density, uncertainty, and quality checks."""

        # self.current_densities()

        if not any([np.isfinite(c.Jr) for c in self.cells]):
            msg = (
                "No finite current densities exist - probably not called "
                "grid.current_densities() first?"
            )
            raise ValueError(msg)

        for cell in self.cells:
            cell.Ir = (
                cell.Jr
                * cell.r_c**2
                * np.sin(cell.th_c)
                * cell.dth
                * cell.dphi
                * (self.radius_km * 1e3) ** 2
                if not np.isnan(cell.Jr)
                else np.nan
            )
            cell.Ith = (
                cell.Jth
                * cell.r_c
                * np.sin(cell.th_c)
                * cell.dr
                * cell.dphi
                * (self.radius_km * 1e3) ** 2
                if not np.isnan(cell.Jth)
                else np.nan
            )
            cell.Iphi = (
                cell.Jphi * cell.r_c * cell.dr * cell.dth * (self.radius_km * 1e3) ** 2
                if not np.isnan(cell.Jphi)
                else np.nan
            )

        for cell in self.cells:
            cell.dI_I = np.nan

            dIr = 0

            if cell.neighbours["r"][+1] is not None:
                if not np.isnan(cell.neighbours["r"][+1].Jr):
                    dIr += (
                        cell.neighbours["r"][+1].Jr
                        * (cell.neighbours["r"][+1].r_c) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (self.radius_km * 1e3) ** 2
                    )
            else:
                dIr += cell.Ir
            if cell.neighbours["r"][-1] is not None:
                if not np.isnan(cell.neighbours["r"][-1].Jr):
                    dIr = (
                        dIr
                        - cell.neighbours["r"][-1].Jr
                        * (cell.neighbours["r"][-1].r_c) ** 2
                        * np.sin(cell.th_c)
                        * cell.dth
                        * cell.dphi
                        * (self.radius_km * 1e3) ** 2
                    )
            else:
                dIr = dIr - cell.Ir

            dIth = 0

            if cell.neighbours["th"][+1] is not None:
                if not np.isnan(cell.neighbours["th"][+1].Jth):
                    dIth += (
                        cell.neighbours["th"][+1].Jth
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][+1].th_c)
                        * cell.dr
                        * cell.dphi
                        * (self.radius_km * 1e3) ** 2
                    )
            else:
                dIth += cell.Ith
            if cell.neighbours["th"][-1] is not None:
                if not np.isnan(cell.neighbours["th"][-1].Jth):
                    dIth = (
                        dIth
                        - cell.neighbours["th"][-1].Jth
                        * cell.r_c
                        * np.sin(cell.neighbours["th"][-1].th_c)
                        * cell.dr
                        * cell.dphi
                        * (self.radius_km * 1e3) ** 2
                    )
            else:
                dIth = dIth - cell.Ith

            dIphi = 0

            if cell.neighbours["phi"][+1] is not None:
                if not np.isnan(cell.neighbours["phi"][+1].Jphi):
                    dIphi += (
                        cell.neighbours["phi"][+1].Jphi
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (self.radius_km * 1e3) ** 2
                    )
            else:
                dIphi += cell.Iphi
            if cell.neighbours["phi"][-1] is not None:
                if not np.isnan(cell.neighbours["phi"][-1].Jphi):
                    dIphi = (
                        dIphi
                        - cell.neighbours["phi"][-1].Jphi
                        * cell.r_c
                        * cell.dr
                        * cell.dth
                        * (self.radius_km * 1e3) ** 2
                    )
            else:
                dIphi = dIphi - cell.Iphi
            cell.dI_I = abs(dIr + dIth + dIphi) / np.sqrt(
                cell.Ir**2 + cell.Ith**2 + cell.Iphi**2,
            )

        self.dI_I = self._cell_scalar_array("dI_I")

        self._refresh_current_density_arrays()

    def smooth_magnetic_field(self):
        """Populate smoothed magnetic-field arrays and matching Cell aliases."""
        for attr in ("Br", "Bth", "Bphi"):
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Cannot smooth magnetic field before grid.{attr} is populated",
                )

        # Average over the central cell and the 6 cells that share faces with it.
        kernel = (
            1
            / 7
            * np.array(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ],
            )
        )

        convolve_kwargs = {
            "boundary": "extend",
            "nan_treatment": "interpolate",
            "normalize_kernel": True,
            "preserve_nan": True,
        }
        self.Bsmr = astropy.convolution.convolve(
            self.Br,
            kernel,
            **convolve_kwargs,
        )
        self.Bsmth = astropy.convolution.convolve(
            self.Bth,
            kernel,
            **convolve_kwargs,
        )
        self.Bsmphi = astropy.convolution.convolve(
            self.Bphi,
            kernel,
            **convolve_kwargs,
        )

        for cell in self.cells:
            idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
            cell.Bsmr = self.Bsmr[idx]
            cell.Bsmth = self.Bsmth[idx]
            cell.Bsmphi = self.Bsmphi[idx]
            cell.sigma_Bsmr = cell.sigma_Br
            cell.sigma_Bsmth = cell.sigma_Bth
            cell.sigma_Bsmphi = cell.sigma_Bphi

        self._populate_cartesian_vector("Bsm")

        return self

    def bin_data(self, data_file_name=None):
        """Bin magnetic field data and populate grid cell values.

        Reads magnetic field data from a CSV file and bins it into the grid.
        Computes averages and smoothed field components for each grid cell.

        Parameters
        ----------
        data_file_name : str, optional
            Path to the CSV data file. Must contain columns: Bx_mse, By_mse, Bz_mse,
            X_mse, Y_mse, Z_mse, R_mse, th_mse, phi_mse.

        Returns
        -------
        self : Grid
            Returns the Grid instance for method chaining.
        """
        if data_file_name is None:
            raise FileNotFoundError("No data file given \n")
        self._clear_binned_products()

        print(f"Loading data from {data_file_name}")

        # Read the mag_mse file into a pandas dataframe
        df = pd.read_csv(data_file_name)
        df = df.drop(df[np.isnan(df["R_mse"])].index)
        df = df[
            [
                "Bx_mse",
                "By_mse",
                "Bz_mse",
                "X_mse",
                "Y_mse",
                "Z_mse",
                "R_mse",
                "th_mse",
                "phi_mse",
            ]
        ]

        r_binned = np.digitize(df["R_mse"].values, self.r_bins, right=True)
        th_binned = np.digitize(df["th_mse"].values, self.th_bins, right=True)
        phi_binned = np.digitize(df["phi_mse"].values, self.phi_bins, right=True)

        # B_mse_cart = [
        #     [Bx, By, Bz] for Bx, By, Bz in
        #               zip(df["Bx_mse"], df["By_mse"], df["Bz_mse"])
        # ]

        B_mse_cart = df[["Bx_mse", "By_mse", "Bz_mse"]].values

        B_mse_sph = cart_to_sph_vf(
            B_mse_cart,
            df["th_mse"].values,
            df["phi_mse"].values,
        )

        # df["Br_mse"] = [B_sph.A[0][0] for B_sph in B_mse_sph]
        # df["Bth_mse"] = [B_sph.A[0][1] for B_sph in B_mse_sph]
        # df["Bphi_mse"] = [B_sph.A[0][2] for B_sph in B_mse_sph]

        df["Br_mse"], df["Bth_mse"], df["Bphi_mse"] = B_mse_sph

        df["r_bin"] = r_binned
        df["th_bin"] = th_binned
        df["phi_bin"] = phi_binned

        df = df.drop(columns=["Bx_mse", "By_mse", "Bz_mse", "X_mse", "Y_mse", "Z_mse"])

        # Average measurements in each cell and keep cartesian values for cone plots.
        grouped_measurements = df.groupby(["r_bin", "th_bin", "phi_bin"], sort=False)

        for cell in self.cells:
            cell.B_mse_sph = _nan_field_components()
            bin_key = (cell.r_bin, cell.th_bin, cell.phi_bin)

            if bin_key in grouped_measurements.groups:
                df_data_in_bin = grouped_measurements.get_group(bin_key)
                cell.B_mse_sph[0] = df_data_in_bin["Br_mse"].values
                cell.B_mse_sph[1] = df_data_in_bin["Bth_mse"].values
                cell.B_mse_sph[2] = df_data_in_bin["Bphi_mse"].values

                cell.n_measurements = len(cell.B_mse_sph[0])
            else:
                cell.n_measurements = np.nan

            cell.B_average()
            cell.X, cell.Y, cell.Z = cell.sph_to_cart(cell.r_c, cell.th_c, cell.phi_c)
            self.X_cells.append(cell.X)
            self.Y_cells.append(cell.Y)
            self.Z_cells.append(cell.Z)

        del df

        self._populate_binned_products_from_cells()
        self._populate_cartesian_vector("B")

        self.smooth_magnetic_field()

        return self

    def save_grid(self, directory="."):
        """Serialize this grid to a pickle file and return the output path.

        .. deprecated::
            Use :meth:`save` instead. This method is kept for backward compatibility.
        """
        return self.save(directory=directory)

    def calculate_JdotE(self, E_cart=(0.0, 0.0, 400e3 * 3e-9)):
        """
        Compute J·E [W/m^3].

        Parameters
        ----------
        E_cart : array-like
            Constant electric field [Ex, Ey, Ez] in MSE Cartesian coordinates (V/m).

        Creates
        -------
        self.JdotE : ndarray
            Same shape as self.Jr, self.Jth, self.Jphi
        """

        for attr in ("Jr", "Jth", "Jphi", "Br", "Bth", "Bphi"):
            if not hasattr(self, attr):
                raise AttributeError(f"Grid has no {attr!r}.")

        Ex, Ey, Ez = E_cart

        # Build angular coordinate grids
        phi3d, th3d, r3d = np.meshgrid(
            self.phi,
            self.th,
            self.r,
            indexing="ij",
        )

        e_r, e_th, e_phi = spherical_unit_vectors(th3d, phi3d)

        # Project constant Cartesian E into local spherical basis
        E_r = Ex * e_r[..., 0] + Ey * e_r[..., 1] + Ez * e_r[..., 2]
        E_th = Ex * e_th[..., 0] + Ey * e_th[..., 1] + Ez * e_th[..., 2]
        E_phi = Ex * e_phi[..., 0] + Ey * e_phi[..., 1] + Ez * e_phi[..., 2]

        # nA/m2 -> A/m2
        self.JdotE = (self.Jr * E_r + self.Jth * E_th + self.Jphi * E_phi) / 1e9

        dV = (
            r3d**2 * np.sin(th3d) * self.dr * self.dth * self.dphi * (3396.0 * 1e3) ** 3
        )

        self.power = self.JdotE * dV

        return self

    def calculate_JxB(self, store_spherical=True, nAnT=True):
        """
        Compute J x B on the grid.

        Assumes:
            J components are self.Jr, self.Jth, self.Jphi
            B components are self.Br, self.Bth, self.Bphi

        Creates:
            self.JxB_x, self.JxB_y, self.JxB_z

        Optionally also creates:
            self.JxB_r, self.JxB_th, self.JxB_phi

        Units:
            If J is A/m^2 and B is T, JxB is N/m^3.
        """

        for attr in ("Jr", "Jth", "Jphi", "Br", "Bth", "Bphi"):
            if not hasattr(self, attr):
                raise AttributeError(f"Grid has no {attr!r}.")

        JxB_r = self.Jth * self.Bphi - self.Jphi * self.Bth
        JxB_th = self.Jphi * self.Br - self.Jr * self.Bphi
        JxB_phi = self.Jr * self.Bth - self.Jth * self.Br

        JxB_sph = np.stack([JxB_r, JxB_th, JxB_phi], axis=-1)

        if nAnT:
            JxB_sph *= 1e-9 * 1e-9

        self._populate_cartesian_vector("JxB_", JxB_sph)

        if store_spherical:
            self.JxB_r = JxB_sph[..., 0]
            self.JxB_th = JxB_sph[..., 1]
            self.JxB_phi = JxB_sph[..., 2]

        return self
