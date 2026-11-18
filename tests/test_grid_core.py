from __future__ import annotations

import numpy as np
import pytest

from mars_currents.grid import (
    Cell,
    Grid,
    cart_to_sph,
    load_grid,
    radial_profile,
    save_grid,
    sph_to_cart,
)


@pytest.fixture
def small_grid_dict():
    return {
        "r_lim": [1.0, 2.0],
        "dr": 1.0,
        "th_lim_deg": [0.0, 90.0],
        "dth_deg": 45.0,
        "phi_lim_deg": [0.0, 180.0],
        "dphi_deg": 90.0,
        "radius_km": 100.0,
    }


def test_spherical_cartesian_round_trip():
    point = [1.5, np.deg2rad(60.0), np.deg2rad(120.0)]

    cartesian = sph_to_cart(*point)
    spherical = cart_to_sph(*cartesian)

    assert np.allclose(spherical, point)


def test_cell_phi_neighbour_bins_wrap_around():
    r = np.array([1.0])
    th = np.array([np.pi / 2])
    phi = np.array([0.0, np.pi])

    cell = Cell(
        1.0,
        np.pi / 2,
        np.pi,
        0,
        0,
        0,
        r,
        th,
        phi,
        r_bin=1,
        th_bin=1,
        phi_bin=1,
        r_bins=np.array([1]),
        th_bins=np.array([1]),
        phi_bins=np.array([1, 2]),
    )

    cell.set_neighbours_bins()

    assert cell.neighbours_bins["phi"][-1]["phi_bin"] == 2
    assert cell.neighbours_bins["phi"][+1]["phi_bin"] == 2


def test_grid_builds_expected_number_of_cells(small_grid_dict):
    grid = Grid(**small_grid_dict)

    assert len(grid.r) == 2
    assert len(grid.th) == 2
    assert len(grid.phi) == 2
    assert len(grid.cells) == 8


def test_grid_initializes_empty_binned_products(small_grid_dict):
    grid = Grid(**small_grid_dict)

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
        values = getattr(grid, attr)
        assert values.shape == grid.shape
        assert np.isnan(values).all()

    for cell in grid.cells:
        idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
        assert grid.th_2d[idx] == cell.th_c
        assert grid.phi_2d[idx] == cell.phi_c

    assert grid.X_cells == []
    assert grid.Bx_cells == []


def test_current_densities_refresh_grid_arrays():
    grid = Grid(
        dr=1.0,
        dth_deg=45.0,
        dphi_deg=90.0,
        r_lim=[1.0, 3.0],
        th_lim_deg=[0.0, 180.0],
        phi_lim_deg=[0.0, 360.0],
    )

    for cell in grid.cells:
        cell.Br = cell.r_idx + 2 * cell.th_idx + 3 * cell.phi_idx
        cell.Bth = cell.r_idx - cell.th_idx + 0.5 * cell.phi_idx
        cell.Bphi = 0.25 * cell.r_idx + cell.th_idx - cell.phi_idx
        cell.sigma_Br = 0.1
        cell.sigma_Bth = 0.1
        cell.sigma_Bphi = 0.1

    grid.current_densities(smoothed=False)

    assert grid.current_density_projection_field == "B"
    assert grid.Jr.shape == grid.shape
    assert grid.Jth.shape == grid.shape
    assert grid.Jphi.shape == grid.shape
    assert grid.Jx.shape == grid.shape
    assert grid.Jy.shape == grid.shape
    assert grid.Jz.shape == grid.shape
    assert grid.J_tot.shape == grid.shape

    for cell in grid.cells:
        idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
        assert grid.Jr[idx] == cell.Jr
        assert grid.Jth[idx] == cell.Jth
        assert grid.Jphi[idx] == cell.Jphi
        assert grid.Jx[idx] == cell.Jx
        assert grid.Jy[idx] == cell.Jy
        assert grid.Jz[idx] == cell.Jz
        assert grid.J_tot[idx] == cell.J_tot


def test_sync_cell_fields_from_grid_feeds_current_densities():
    grid = Grid(
        dr=0.5,
        dth_deg=45.0,
        dphi_deg=90.0,
        r_lim=[1.0, 2.0],
        th_lim_deg=[0.0, 180.0],
        phi_lim_deg=[0.0, 360.0],
    )

    for cell in grid.cells:
        cell.Br = 100 + cell.r_idx + 10 * cell.th_idx + 20 * cell.phi_idx
        cell.Bth = -50 + cell.r_idx
        cell.Bphi = 25 - cell.phi_idx

    for cell in grid.cells:
        for coordinate in ("r", "th", "phi"):
            for direction in (-1, +1):
                neighbour = cell.neighbours[coordinate][direction]
                if neighbour is not None:
                    cell.neighbours[coordinate][direction] = neighbour.attr_dict()

    _, _, r3d = np.meshgrid(grid.phi, grid.th, grid.r, indexing="ij")
    grid.Br = np.zeros(grid.shape)
    grid.Bth = np.zeros(grid.shape)
    grid.Bphi = r3d

    grid.sync_cell_fields_from_grid()

    for cell in grid.cells:
        idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
        for coordinate in ("r", "th", "phi"):
            for direction in (-1, +1):
                neighbour = cell.neighbours[coordinate][direction]
                assert neighbour is None or isinstance(neighbour, Cell)

        assert cell.Br == grid.Br[idx]
        assert cell.Bth == grid.Bth[idx]
        assert cell.Bphi == grid.Bphi[idx]
        assert cell.B_mse_mean == [grid.Br[idx], grid.Bth[idx], grid.Bphi[idx]]

    grid.smooth_magnetic_field()
    grid.current_densities()

    assert np.nanmax(np.abs(grid.Jth)) > 0.0


def test_current_densities_can_use_smoothed_bsm_fields():
    grid = Grid(
        dr=0.5,
        dth_deg=45.0,
        dphi_deg=90.0,
        r_lim=[1.0, 2.0],
        th_lim_deg=[0.0, 180.0],
        phi_lim_deg=[0.0, 360.0],
    )

    _, _, r3d = np.meshgrid(grid.phi, grid.th, grid.r, indexing="ij")
    grid.Bsmr = np.zeros(grid.shape)
    grid.Bsmth = np.zeros(grid.shape)
    grid.Bsmphi = r3d

    for cell in grid.cells:
        idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
        cell.Br = 0.0
        cell.Bth = 0.0
        cell.Bphi = 0.0
        cell.Bsmr = grid.Bsmr[idx]
        cell.Bsmth = grid.Bsmth[idx]
        cell.Bsmphi = grid.Bsmphi[idx]
        cell.sigma_Br = 0.1
        cell.sigma_Bth = 0.1
        cell.sigma_Bphi = 0.1

    # We don't run this step because it overwrites!
    # grid.smooth_magnetic_field()

    grid.current_densities(smoothed=True)

    assert grid.current_density_projection_field == "Bsm"
    assert np.nanmax(np.abs(grid.Jth)) > 0.0


def _set_cell_vectors(grid, current, magnetic_field, smoothed_field=None):
    """Assign constant spherical J and B vectors to every cell in a grid."""
    for cell in grid.cells:
        cell.Jr, cell.Jth, cell.Jphi = current
        cell.Br, cell.Bth, cell.Bphi = magnetic_field
        if smoothed_field is not None:
            cell.Bsmr, cell.Bsmth, cell.Bsmphi = smoothed_field


_PROJECTED_CURRENT_FIELDS = (
    "J_parallel",
    "J_parallel_r",
    "J_parallel_th",
    "J_parallel_phi",
    "J_parallel_x",
    "J_parallel_y",
    "J_parallel_z",
    "J_parallel_tot",
    "J_perp",
    "J_perp_r",
    "J_perp_th",
    "J_perp_phi",
    "J_perp_x",
    "J_perp_y",
    "J_perp_z",
    "J_perp_tot",
)


@pytest.mark.parametrize(
    ("current", "magnetic_field", "expected_parallel", "expected_perp"),
    [
        ((3.0, 0.0, 0.0), (2.0, 0.0, 0.0), 3.0, 0.0),
        ((-3.0, 0.0, 0.0), (2.0, 0.0, 0.0), -3.0, 0.0),
        ((0.0, 4.0, 0.0), (2.0, 0.0, 0.0), 0.0, 4.0),
    ],
)
def test_project_current_densities_handles_parallel_and_perpendicular_vectors(
    small_grid_dict,
    current,
    magnetic_field,
    expected_parallel,
    expected_perp,
):
    grid = Grid(**small_grid_dict)
    _set_cell_vectors(grid, current, magnetic_field)

    grid.project_current_densities()

    assert np.allclose(grid.J_parallel, expected_parallel)
    assert np.allclose(grid.J_perp, expected_perp)
    assert np.allclose(grid.J_parallel_tot, abs(expected_parallel))
    assert np.allclose(grid.J_perp_tot, expected_perp)


def test_project_current_densities_reconstructs_oblique_current(small_grid_dict):
    grid = Grid(**small_grid_dict)
    _set_cell_vectors(
        grid,
        current=(2.0, -1.0, 4.0),
        magnetic_field=(1.0, 2.0, 2.0),
    )

    grid.project_current_densities()

    current = np.stack((grid.Jr, grid.Jth, grid.Jphi), axis=-1)
    parallel = np.stack(
        (grid.J_parallel_r, grid.J_parallel_th, grid.J_parallel_phi), axis=-1
    )
    perpendicular = np.stack((grid.J_perp_r, grid.J_perp_th, grid.J_perp_phi), axis=-1)
    magnetic_field = np.stack(
        [grid._cell_scalar_array(name) for name in ("Br", "Bth", "Bphi")],
        axis=-1,
    )

    assert np.allclose(parallel + perpendicular, current)
    assert np.allclose(np.sum(perpendicular * magnetic_field, axis=-1), 0.0)
    assert np.allclose(
        grid.J_tot**2,
        grid.J_parallel**2 + grid.J_perp**2,
    )

    for cell in grid.cells:
        idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
        for name in _PROJECTED_CURRENT_FIELDS:
            assert np.isclose(getattr(cell, name), getattr(grid, name)[idx])


@pytest.mark.parametrize(
    "magnetic_field",
    [(0.0, 0.0, 0.0), (np.nan, 1.0, 0.0)],
)
def test_project_current_densities_marks_undefined_field_direction_nan(
    small_grid_dict,
    magnetic_field,
):
    grid = Grid(**small_grid_dict)
    _set_cell_vectors(grid, current=(1.0, 2.0, 3.0), magnetic_field=magnetic_field)

    grid.project_current_densities()

    for name in (
        "J_parallel",
        "J_parallel_r",
        "J_parallel_tot",
        "J_perp",
        "J_perp_r",
        "J_perp_tot",
    ):
        assert np.isnan(getattr(grid, name)).all()


def test_project_current_densities_uses_requested_field(small_grid_dict):
    grid = Grid(**small_grid_dict)
    _set_cell_vectors(
        grid,
        current=(3.0, 0.0, 0.0),
        magnetic_field=(1.0, 0.0, 0.0),
        smoothed_field=(0.0, 1.0, 0.0),
    )

    grid.project_current_densities(field="Bsm")

    assert np.allclose(grid.J_parallel, 0.0)
    assert np.allclose(grid.J_perp, 3.0)
    assert grid.current_density_projection_field == "Bsm"


def test_projected_current_densities_survive_serialization(small_grid_dict):
    grid = Grid(**small_grid_dict)
    _set_cell_vectors(
        grid,
        current=(2.0, -1.0, 4.0),
        magnetic_field=(1.0, 2.0, 2.0),
    )
    grid.project_current_densities()

    restored = Grid.from_dict(grid.to_dict())

    for name in _PROJECTED_CURRENT_FIELDS:
        assert np.allclose(getattr(restored, name), getattr(grid, name))

        for cell in restored.cells:
            idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
            assert np.isclose(getattr(cell, name), getattr(restored, name)[idx])

    assert restored.current_density_projection_field == "B"


def test_calculate_JxB_stores_spherical_components_by_default():
    grid = Grid(
        dr=1.0,
        dth_deg=90.0,
        dphi_deg=90.0,
        r_lim=[1.0, 1.0],
        th_lim_deg=[45.0, 135.0],
        phi_lim_deg=[-45.0, 45.0],
    )

    grid.Jr = np.full(grid.shape, 2.0)
    grid.Jth = np.full(grid.shape, 3.0)
    grid.Jphi = np.full(grid.shape, 5.0)
    grid.Br = np.full(grid.shape, 7.0)
    grid.Bth = np.full(grid.shape, 11.0)
    grid.Bphi = np.full(grid.shape, 13.0)

    grid.calculate_JxB()

    unit_scale = 1e-18
    expected_r = (3.0 * 13.0 - 5.0 * 11.0) * unit_scale
    expected_th = (5.0 * 7.0 - 2.0 * 13.0) * unit_scale
    expected_phi = (2.0 * 11.0 - 3.0 * 7.0) * unit_scale

    assert np.allclose(grid.JxB_r, expected_r)
    assert np.allclose(grid.JxB_th, expected_th)
    assert np.allclose(grid.JxB_phi, expected_phi)
    assert np.allclose(grid.JxB_x, expected_r)
    assert np.allclose(grid.JxB_y, expected_phi)
    assert np.allclose(grid.JxB_z, -expected_th)
    assert np.allclose(grid.JxB_tot, grid.JxB_mag)
    assert np.allclose(
        grid.JxB_mag,
        np.sqrt(expected_r**2 + expected_th**2 + expected_phi**2),
    )


def test_cartesian_components_are_populated_for_raw_and_smoothed_fields():
    grid = Grid(
        dr=1.0,
        dth_deg=90.0,
        dphi_deg=90.0,
        r_lim=[1.0, 1.0],
        th_lim_deg=[45.0, 135.0],
        phi_lim_deg=[45.0, 135.0],
    )
    grid.Br = np.full(grid.shape, 2.0)
    grid.Bth = np.full(grid.shape, 3.0)
    grid.Bphi = np.full(grid.shape, 5.0)

    grid.sync_cell_fields_from_grid()

    assert np.allclose(grid.Bx, -5.0)
    assert np.allclose(grid.By, 2.0)
    assert np.allclose(grid.Bz, -3.0)
    assert np.allclose(grid.B_tot, np.sqrt(38.0))
    assert np.allclose(grid.Bx_cells, -5.0)
    assert np.allclose(grid.By_cells, 2.0)
    assert np.allclose(grid.Bz_cells, -3.0)

    for cell in grid.cells:
        assert np.isclose(cell.Bx, -5.0)
        assert np.isclose(cell.By, 2.0)
        assert np.isclose(cell.Bz, -3.0)
        assert np.isclose(cell.B_tot, np.sqrt(38.0))

    grid.smooth_magnetic_field()

    assert np.allclose(grid.Bsmx, -5.0)
    assert np.allclose(grid.Bsmy, 2.0)
    assert np.allclose(grid.Bsmz, -3.0)
    assert np.allclose(grid.Bsm_tot, np.sqrt(38.0))
    for cell in grid.cells:
        assert np.isclose(cell.Bsmx, -5.0)
        assert np.isclose(cell.Bsmy, 2.0)
        assert np.isclose(cell.Bsmz, -3.0)
        assert np.isclose(cell.Bsm_tot, np.sqrt(38.0))


def test_smooth_magnetic_field_populates_bsm_fields():
    grid = Grid(
        dr=1.0,
        dth_deg=45.0,
        dphi_deg=90.0,
        r_lim=[1.0, 3.0],
        th_lim_deg=[0.0, 180.0],
        phi_lim_deg=[0.0, 360.0],
    )
    grid.Br = np.zeros(grid.shape)
    grid.Bth = np.zeros(grid.shape)
    grid.Bphi = np.zeros(grid.shape)
    grid.Br[0, 0, 0] = 7.0

    for cell in grid.cells:
        idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
        cell.Br = grid.Br[idx]
        cell.Bth = grid.Bth[idx]
        cell.Bphi = grid.Bphi[idx]
        cell.sigma_Br = 0.1
        cell.sigma_Bth = 0.2
        cell.sigma_Bphi = 0.3

    grid.smooth_magnetic_field()

    cell = next(
        c for c in grid.cells if c.r_idx == 0 and c.th_idx == 0 and c.phi_idx == 0
    )
    assert grid.Bsmr.shape == grid.shape
    assert cell.Br == grid.Br[cell.phi_idx, cell.th_idx, cell.r_idx]
    assert cell.Bsmr == grid.Bsmr[cell.phi_idx, cell.th_idx, cell.r_idx]
    assert cell.Bsmr != cell.Br
    assert cell.sigma_Bsmphi == cell.sigma_Bphi


def test_bin_data_keeps_raw_and_smoothed_cell_fields_separate(tmp_path):
    grid = Grid(
        dr=1.0,
        dth_deg=45.0,
        dphi_deg=90.0,
        r_lim=[1.0, 2.0],
        th_lim_deg=[0.0, 90.0],
        phi_lim_deg=[0.0, 180.0],
    )
    rows = ["Bx_mse,By_mse,Bz_mse,X_mse,Y_mse,Z_mse,R_mse,th_mse,phi_mse"]
    for cell in grid.cells:
        br = 10.0 if cell.r_idx == 0 else 100.0
        bx, by, bz = (
            br * np.sin(cell.th_c) * np.cos(cell.phi_c),
            br * np.sin(cell.th_c) * np.sin(cell.phi_c),
            br * np.cos(cell.th_c),
        )
        x, y, z = (
            cell.r_c * np.sin(cell.th_c) * np.cos(cell.phi_c),
            cell.r_c * np.sin(cell.th_c) * np.sin(cell.phi_c),
            cell.r_c * np.cos(cell.th_c),
        )
        rows.append(f"{bx},{by},{bz},{x},{y},{z},{cell.r_c},{cell.th_c},{cell.phi_c}")
    data_file = tmp_path / "mag.csv"
    data_file.write_text("\n".join(rows))

    grid.bin_data(data_file)

    cell = next(
        c for c in grid.cells if c.r_idx == 0 and c.th_idx == 0 and c.phi_idx == 0
    )
    assert cell.Br == grid.Br[cell.phi_idx, cell.th_idx, cell.r_idx]
    assert cell.Bsmr == grid.Bsmr[cell.phi_idx, cell.th_idx, cell.r_idx]
    assert cell.Br != cell.Bsmr


def test_load_grid_from_dict_sets_generated_name(small_grid_dict):
    grid = load_grid(grid_dict=small_grid_dict)

    assert grid.name == "dr1.0_dth45.0_dphi90.0"
    assert len(grid.cells) == 8


def test_save_and_reload_grid_round_trip(tmp_path, small_grid_dict):
    grid = load_grid(grid_dict=small_grid_dict)

    path = save_grid(grid, directory=tmp_path)
    loaded = load_grid(load_from_file=True, file_path=path)

    assert path.exists()
    assert loaded.name == grid.name
    assert len(loaded.cells) == len(grid.cells)
    assert np.allclose(loaded.r, grid.r)


def test_loading_legacy_grid_regenerates_missing_cartesian_components(
    small_grid_dict,
):
    grid = Grid(**small_grid_dict)
    grid.Br = np.full(grid.shape, 2.0)
    grid.Bth = np.full(grid.shape, 3.0)
    grid.Bphi = np.full(grid.shape, 5.0)
    grid.sync_cell_fields_from_grid()
    grid.smooth_magnetic_field()

    grid.Jr = np.full(grid.shape, 7.0)
    grid.Jth = np.full(grid.shape, 11.0)
    grid.Jphi = np.full(grid.shape, 13.0)
    grid._populate_cartesian_vector("J")

    expected = {
        name: np.asarray(getattr(grid, name)).copy()
        for name in (
            "Bx",
            "By",
            "Bz",
            "B_tot",
            "Bsmx",
            "Bsmy",
            "Bsmz",
            "Bsm_tot",
            "Jx",
            "Jy",
            "Jz",
            "J_tot",
        )
    }

    legacy_dict = grid.to_dict()
    cartesian_names = tuple(expected)
    for name in cartesian_names:
        legacy_dict.pop(name, None)
    for cell_dict in legacy_dict["cells_dict"].values():
        for name in cartesian_names:
            cell_dict.pop(name, None)

    loaded = Grid.from_dict(legacy_dict)

    for name, values in expected.items():
        assert np.allclose(getattr(loaded, name), values, equal_nan=True)

    for cell in loaded.cells:
        idx = (cell.phi_idx, cell.th_idx, cell.r_idx)
        for name in cartesian_names:
            assert np.isclose(
                getattr(cell, name),
                getattr(loaded, name)[idx],
                equal_nan=True,
            )


def test_load_grid_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_grid(load_from_file=True, file_path=tmp_path / "missing.pickle")


def test_neighbor_serialization_preserves_cell_references(small_grid_dict):
    """Test that neighbors are Cell objects, not dicts, after deserialization."""
    grid = Grid(**small_grid_dict)

    # Verify original neighbors are Cell objects
    original_cell = grid.cells[0]
    original_neighbor = original_cell.neighbours["r"][+1]
    assert hasattr(original_neighbor, "r_idx")
    assert isinstance(original_neighbor, Cell)

    # Serialize and deserialize
    grid_dict = grid.to_dict()
    grid2 = Grid.from_dict(grid_dict)

    # Verify restored neighbors are also Cell objects (not dicts)
    restored_cell = grid2.cells[0]
    restored_neighbor = restored_cell.neighbours["r"][+1]
    assert hasattr(restored_neighbor, "r_idx")
    assert isinstance(restored_neighbor, Cell)
    assert not isinstance(restored_neighbor, dict)


def test_neighbor_serialization_round_trip(small_grid_dict):
    """Test that all neighbor relationships are preserved across serialization."""
    grid = Grid(**small_grid_dict)

    # Record original neighbor indices
    original_neighbors = {}
    for i, cell in enumerate(grid.cells):
        original_neighbors[i] = {}
        for coord in ["r", "th", "phi"]:
            for direction in [-1, +1]:
                n = cell.neighbours[coord][direction]
                original_neighbors[i][(coord, direction)] = (
                    (n.r_idx, n.th_idx, n.phi_idx) if n else None
                )

    # Serialize and deserialize
    grid_dict = grid.to_dict()
    grid2 = Grid.from_dict(grid_dict)

    # Verify all neighbor relationships match
    for i, cell in enumerate(grid2.cells):
        for coord in ["r", "th", "phi"]:
            for direction in [-1, +1]:
                n = cell.neighbours[coord][direction]
                restored_indices = (n.r_idx, n.th_idx, n.phi_idx) if n else None
                expected = original_neighbors[i][(coord, direction)]
                assert restored_indices == expected, (
                    f"Neighbor mismatch at cell {i}, {coord}:{direction}"
                )


def test_neighbor_serialization_with_none_boundaries(small_grid_dict):
    """Test that None boundary neighbors are preserved during serialization."""
    # Create grid with limited colatitude range to have None neighbors
    grid_dict = {
        "r_lim": [1.0, 2.0],
        "dr": 1.0,
        "th_lim_deg": [0.0, 90.0],
        "dth_deg": 90.0,  # Only one colatitude bin → None at boundaries
        "phi_lim_deg": [0.0, 180.0],
        "dphi_deg": 90.0,
    }
    grid = Grid(**grid_dict)

    # Find a cell with None neighbor
    cell_with_none = None
    for cell in grid.cells:
        if any(
            cell.neighbours[coord][direction] is None
            for coord in ["r", "th", "phi"]
            for direction in [-1, +1]
        ):
            cell_with_none = cell
            break

    assert cell_with_none is not None, "Test requires a cell with None neighbor"

    # Serialize and deserialize
    grid_dict_ser = grid.to_dict()
    grid2 = Grid.from_dict(grid_dict_ser)

    # Verify None neighbors are preserved
    restored_cell = grid2.cells[grid.cells.index(cell_with_none)]
    for coord in ["r", "th", "phi"]:
        for direction in [-1, +1]:
            original = cell_with_none.neighbours[coord][direction]
            restored = restored_cell.neighbours[coord][direction]
            if original is None:
                assert restored is None, (
                    f"None neighbor not preserved at {coord}:{direction}"
                )
            else:
                assert restored is not None


def test_neighbor_serialization_indices_not_stored_in_dict(small_grid_dict):
    """Test that neighbors_indices is only temporary and cleaned up after deserialization."""
    grid = Grid(**small_grid_dict)
    grid_dict = grid.to_dict()
    grid2 = Grid.from_dict(grid_dict)

    # Verify temporary neighbours_indices is cleaned up
    for cell in grid2.cells:
        assert not hasattr(cell, "neighbours_indices"), (
            "neighbours_indices should be deleted after reconstruction"
        )


def test_performance_grid_creation():
    """Test that grid creation with large grid is about as fast as expected.

    Catch any degradation in performance when fiddling with the algorithms.
    """
    import time

    # Large grid that would be very slow with O(n^2) algorithm
    start = time.time()
    grid = Grid(
        dr=0.5,
        dth_deg=10.0,
        dphi_deg=18.0,
        r_lim=[1.0, 3.0],
        th_lim_deg=[0.0, 180.0],
        phi_lim_deg=[0.0, 360.0],
    )
    elapsed = time.time() - start

    # With O(n^2), this would take >1 second for 1800 cells
    # With O(1) optimization, should be <0.2s [MBA M2, 24GiB].  Relax to 0.5 for now.
    assert elapsed < 0.5, (
        f"Grid creation too slow ({elapsed:.2f}s), possible O(n^2) behavior"
    )
    assert len(grid.cells) > 1000, "Test requires large grid"


def test_radial_boundary_neighbors_are_none():
    """Test that radial boundaries have None neighbors at domain limits."""
    grid = Grid(
        dr=1.0,
        dth_deg=45.0,
        dphi_deg=90.0,
        r_lim=[1.0, 2.0],
        th_lim_deg=[0.0, 90.0],
        phi_lim_deg=[0.0, 180.0],
    )

    # Find cells at radial boundaries
    r_min_cells = [c for c in grid.cells if c.r_idx == 0]
    r_max_cells = [
        c for c in grid.cells if c.r_idx == max(cl.r_idx for cl in grid.cells)
    ]

    # Cells at min radius should have no r=-1 neighbor
    assert len(r_min_cells) > 0
    for cell in r_min_cells:
        assert cell.neighbours["r"][-1] is None, (
            "Min radial boundary should have r=-1 as None"
        )

    # Cells at max radius should have no r=+1 neighbor
    assert len(r_max_cells) > 0
    for cell in r_max_cells:
        assert cell.neighbours["r"][+1] is None, (
            "Max radial boundary should have r=+1 as None"
        )


def test_phi_wraparound_connectivity():
    """Test that phi dimension wraps correctly at 360 degrees."""
    grid = Grid(
        dr=1.0,
        dth_deg=45.0,
        dphi_deg=90.0,
        r_lim=[1.0, 2.0],
        th_lim_deg=[0.0, 90.0],
        phi_lim_deg=[0.0, 360.0],  # Full 360 degrees for wraparound
    )

    # Find cells at max phi indices (min phi would have wrapping into it)
    max_phi_idx = max(c.phi_idx for c in grid.cells)
    max_phi_cells = [c for c in grid.cells if c.phi_idx == max_phi_idx]

    # Cell at max phi should wrap to min phi for phi=+1 direction
    for cell_max in max_phi_cells:
        neighbor_phi_plus = cell_max.neighbours["phi"][+1]
        # Should connect to a cell at min phi with same r and th indices
        if neighbor_phi_plus is not None:
            assert neighbor_phi_plus.phi_idx == 0, "Phi should wrap to 0"
            assert (
                neighbor_phi_plus.r_idx == cell_max.r_idx
                and neighbor_phi_plus.th_idx == cell_max.th_idx
            ), "Wrapped phi neighbor should have same r and th indices"


def test_neighbor_bin_indices_match():
    """Test that neighbor cell bin indices match the expected neighbor_bins values."""
    grid = Grid(
        dr=0.5,
        dth_deg=15.0,
        dphi_deg=45.0,
        r_lim=[1.0, 2.0],
        th_lim_deg=[45.0, 135.0],
        phi_lim_deg=[90.0, 270.0],
    )

    # For each cell, verify that actual neighbors have matching bin indices
    for cell in grid.cells:
        for coordinate in ["r", "th", "phi"]:
            for direction in [-1, +1]:
                neighbor = cell.neighbours[coordinate][direction]
                if neighbor is not None:
                    # Get expected neighbor bins
                    expected_bins = cell.neighbours_bins[coordinate][direction]
                    # Verify neighbor actually has those bin values
                    assert neighbor.r_bin == expected_bins["r_bin"], (
                        f"Neighbor r_bin mismatch: expected {expected_bins['r_bin']}, got {neighbor.r_bin}"
                    )
                    assert neighbor.th_bin == expected_bins["th_bin"], (
                        f"Neighbor th_bin mismatch: expected {expected_bins['th_bin']}, got {neighbor.th_bin}"
                    )
                    assert neighbor.phi_bin == expected_bins["phi_bin"], (
                        f"Neighbor phi_bin mismatch: expected {expected_bins['phi_bin']}, got {neighbor.phi_bin}"
                    )


def test_multiple_serialization_cycles_consistency():
    """Test that grid structure is preserved across multiple serialize/deserialize cycles."""
    original_grid = Grid(
        dr=0.5,
        dth_deg=20.0,
        dphi_deg=60.0,
        r_lim=[1.0, 2.0],
        th_lim_deg=[0.0, 180.0],
        phi_lim_deg=[0.0, 360.0],
    )

    # Record original neighbor structure
    original_structure = {}
    for i, cell in enumerate(original_grid.cells):
        original_structure[i] = {}
        for coord in ["r", "th", "phi"]:
            for direction in [-1, +1]:
                n = cell.neighbours[coord][direction]
                original_structure[i][(coord, direction)] = (
                    (n.r_idx, n.th_idx, n.phi_idx) if n else None
                )

    # Perform 3 consecutive serialize/deserialize cycles
    current_dict = original_grid.to_dict()
    for cycle in range(3):
        grid = Grid.from_dict(current_dict)
        current_dict = grid.to_dict()

        # Verify structure hasn't changed
        for i, cell in enumerate(grid.cells):
            for coord in ["r", "th", "phi"]:
                for direction in [-1, +1]:
                    n = cell.neighbours[coord][direction]
                    current_indices = (n.r_idx, n.th_idx, n.phi_idx) if n else None
                    expected_indices = original_structure[i][(coord, direction)]
                    assert current_indices == expected_indices, (
                        f"Cycle {cycle + 1}: Structure changed for cell {i}"
                    )


def test_large_grid_stress():
    """Stress test with large grid to catch any algorithmic issues."""
    # Create a large grid with 5000+ cells
    grid = Grid(
        dr=0.25,
        dth_deg=10.0,
        dphi_deg=15.0,
        r_lim=[1.0, 4.0],
        th_lim_deg=[0.0, 180.0],
        phi_lim_deg=[0.0, 360.0],
    )

    assert len(grid.cells) > 5000, "Test grid should have 5000+ cells"

    # Verify all neighbors are either Cell objects or None
    for cell in grid.cells:
        for coord in ["r", "th", "phi"]:
            for direction in [-1, +1]:
                neighbor = cell.neighbours[coord][direction]
                assert neighbor is None or hasattr(neighbor, "r_idx"), (
                    f"Invalid neighbor type: {type(neighbor)}"
                )

    # Serialize and deserialize
    grid_dict = grid.to_dict()
    grid2 = Grid.from_dict(grid_dict)

    # Verify structure is preserved
    assert len(grid2.cells) == len(grid.cells), (
        "Cell count changed after deserialization"
    )

    # Spot-check that neighbors are still Cell objects
    for cell in grid2.cells[:100]:  # Check first 100 cells
        for coord in ["r", "th", "phi"]:
            for direction in [-1, +1]:
                neighbor = cell.neighbours[coord][direction]
                assert neighbor is None or hasattr(neighbor, "r_idx"), (
                    f"Invalid neighbor type after deserialization: {type(neighbor)}"
                )


def test_radial_profile_averages_angles(small_grid_dict):
    grid = Grid(**small_grid_dict)
    grid.Jr = np.arange(8, dtype=float).reshape((2, 2, 2))

    radii, profile = radial_profile(grid, "Jr")

    assert np.allclose(radii, grid.r)
    assert np.allclose(profile, [3.0, 4.0])


def test_radial_profile_selects_nearest_latitude_longitude(small_grid_dict):
    grid = Grid(**small_grid_dict)
    grid.Jr = np.arange(8, dtype=float).reshape((2, 2, 2))

    _, profile = radial_profile(
        grid,
        "Jr",
        latitude_deg=22.5,
        longitude_deg=135.0,
    )

    assert np.allclose(profile, [6.0, 7.0])
