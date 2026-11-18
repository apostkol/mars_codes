"""
pyvista_adapter.py

This code is currently unused, retained for a possible future use case.

Utilities for converting Mars spherical Grid objects
to PyVista meshes suitable for streamlines, glyphs,
slices, and volume visualization.
"""

import numpy as np
import pyvista as pv


def sph_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def sph_vector_to_cart(vr, vtheta, vphi, theta, phi):
    """
    Convert vector components expressed in the spherical basis
    (e_r, e_theta, e_phi) into Cartesian coordinates.
    """

    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    vx = vr * st * cp + vtheta * ct * cp - vphi * sp
    vy = vr * st * sp + vtheta * ct * sp + vphi * cp
    vz = vr * ct - vtheta * st

    return vx, vy, vz


def grid_cell_centers(grid):
    """
    Return Cartesian coordinates of all cell centers.
    """

    pts = np.empty((len(grid.cells), 3))

    for i, cell in enumerate(grid.cells):
        pts[i] = sph_to_cart(
            cell.r_c,
            cell.th_c,
            cell.phi_c,
        )

    return pts


def cell_vector_array(grid, field="B"):
    """
    Return Cartesian vector array for either:

        field='B'
        field='J'
    """

    vectors = np.empty((len(grid.cells), 3))

    for i, cell in enumerate(grid.cells):
        if field.upper() == "B":
            vr, vt, vp = cell.B_mse_mean

        elif field.upper() == "J":
            vr = cell.Jr
            vt = cell.Jth
            vp = cell.Jphi

        else:
            raise ValueError("field must be 'B' or 'J'")

        vectors[i] = sph_vector_to_cart(
            vr,
            vt,
            vp,
            cell.th_c,
            cell.phi_c,
        )

    return vectors


def cell_scalar_array(grid, field="sJdE"):
    """
    Return Cartesian vector array for either:

        field='sJdE'
    """

    scalars = np.empty((len(grid.cells)))

    for i, cell in enumerate(grid.cells):
        if field.upper() == "SJDE":
            scalars[i] = np.cos(cell.th_c) * cell.Jr - np.sin(cell.th_c) * cell.Jth
            scalars[i] = np.sign(scalars[i])
        else:
            raise ValueError("field must be 'B' or 'J'")

    return scalars


# def polydata_from_cells(grid):
#     """
#     Simplest possible representation.

#     Produces a PolyData object containing cell-center
#     sample points and attached B/J vectors.
#     """

#     mesh = pv.PolyData(grid_cell_centers(grid))

#     mesh["B"] = cell_vector_array(grid, "B")
#     mesh["J"] = cell_vector_array(grid, "J")

#     return mesh


def polydata_from_cells(grid):

    mesh = pv.PolyData(grid_cell_centers(grid))

    B = cell_vector_array(grid, "B")
    J = cell_vector_array(grid, "J")
    S = cell_scalar_array(grid, "sJdE")
    print(B.shape)

    mesh["B"] = B
    mesh["J"] = J
    mesh["S"] = S

    mesh["Bmag"] = np.sqrt(np.linalg.norm(B, axis=1))
    mesh["Jmag"] = np.sqrt(np.linalg.norm(J, axis=1))
    mesh["Jmag2"] = np.sqrt(np.linalg.norm(J, axis=1)) * S

    # # mesh["JdotE"] = np.array([getattr(cell, "JdotE", np.nan) for cell in grid.cells])
    # print("KKKKK")
    # # Bust
    # print(grid.JdotE.shape)

    # mesh["JdotE"] = np.array(grid.JdotE.flatten())

    # i = 0
    # while i < 10:
    #     for c in grid.cells:
    #         print(c, c.JdotE)
    #         i += 1

    # mesh["JdotE_sign"] = np.sign(mesh["JdotE"])

    # print(np.nanmean(mesh["JdotE"]))
    # print(np.nanmean(np.sign(mesh["JdotE"])))

    return mesh


def structured_grid_from_centers(grid):
    """
    Construct a PyVista StructuredGrid using the
    logical spherical topology.

    Dimensions are:

        (Nr, Ntheta, Nphi)

    and vectors are attached as point data.
    """

    Nr = len(grid.r)
    Nt = len(grid.th)
    Np = len(grid.phi)

    R, TH, PH = np.meshgrid(
        grid.r,
        grid.th,
        grid.phi,
        indexing="ij",
    )

    X, Y, Z = sph_to_cart(R, TH, PH)

    sg = pv.StructuredGrid(
        X,
        Y,
        Z,
    )

    B = np.empty((Nr * Nt * Np, 3))
    J = np.empty((Nr * Nt * Np, 3))

    lookup = {}

    for cell in grid.cells:
        lookup[(cell.r_idx, cell.th_idx, cell.phi_idx)] = cell

    n = 0

    for i in range(Nr):
        for j in range(Nt):
            for k in range(Np):
                cell = lookup[(i, j, k)]

                B[n] = sph_vector_to_cart(
                    *cell.B_mse_mean,
                    cell.th_c,
                    cell.phi_c,
                )

                J[n] = sph_vector_to_cart(
                    cell.Jr,
                    cell.Jth,
                    cell.Jphi,
                    cell.th_c,
                    cell.phi_c,
                )

                n += 1

    sg["B"] = B
    sg["J"] = J
    sg["SJdE"] = J

    return sg


def make_streamlines(grid, seed_radius=None, n_seed_points=500, current=False):
    """
    Convenience wrapper.

    Returns streamline PolyData.
    """

    mesh = structured_grid_from_centers(grid)

    if seed_radius is None:
        seed_radius = np.min(grid.r)

    seed = pv.Sphere(
        radius=seed_radius,
        center=(0, 0, 0),
        theta_resolution=30,
        phi_resolution=30,
    )

    vectors = "B"
    if current:
        vectors = "J"

    streams = mesh.streamlines_from_source(
        seed,
        vectors=vectors,
        max_length=1000.0,
    )

    return streams


def plot_streamlines(grid, current=False, plotter=None):
    streams = make_streamlines(grid, current=current)

    if plotter is None:
        p = pv.Plotter()
    else:
        p = plotter

    p.add_mesh(
        streams,
        line_width=2,
    )

    # p.show()
    return p


def plot_B_glyphs(grid, factor=0.05, plotter=None):

    mesh = polydata_from_cells(grid)

    glyphs = mesh.glyph(
        orient="B",
        scale="Bmag",
        factor=factor,
    )

    if plotter is None:
        p = pv.Plotter()
    else:
        p = plotter

    p.add_mesh(
        glyphs,
        scalars="Bmag",
        show_scalar_bar=True,
    )

    # p.show()
    return p


def plot_J_glyphs(grid, factor=0.05, plotter=None):

    mesh = polydata_from_cells(grid)

    glyphs = mesh.glyph(
        orient="J",
        scale="Jmag",
        factor=factor,
    )

    if plotter is None:
        p = pv.Plotter()
    else:
        p = plotter

    p.add_mesh(glyphs, scalars="Jmag")
    # p.show()

    return p


def plot_J_glyphs_by_JdotE_sign(
    grid,
    factor=0.05,
    plotter=None,
):
    """
    Plot current-density glyphs.

    Arrow direction: J
    Arrow length: |J|
    Arrow color:
        red  = JdotE > 0
        blue = JdotE < 0
    """

    if plotter is None:
        p = pv.Plotter()
    else:
        p = plotter

    mesh = polydata_from_cells(grid)

    glyphs = mesh.glyph(
        orient="J",
        scale="Jmag2",
        factor=factor,
    )

    p.add_mesh(
        glyphs,
        scalars="Jmag2",
        # cmap=["blue", "gray", "red"],
        cmap=["blue", "red"],
        clim=(-1, 1),
        show_scalar_bar=True,
    )

    return p


def random_seed_points_in_spherical_shell(
    grid,
    n_seeds=1000,
    r_bounds=None,
    theta_bounds=None,
    phi_bounds=None,
    rng=None,
):
    """
    Random seed points inside the spherical grid domain.

    Points are sampled uniformly in spherical volume, not uniformly in r.
    theta is colatitude.
    """

    rng = np.random.default_rng(rng)

    if r_bounds is None:
        r_bounds = (np.nanmin(grid.r), np.nanmax(grid.r))

    if theta_bounds is None:
        theta_bounds = (np.nanmin(grid.th), np.nanmax(grid.th))

    if phi_bounds is None:
        phi_bounds = (np.nanmin(grid.phi), np.nanmax(grid.phi))

    r_min, r_max = r_bounds
    th_min, th_max = theta_bounds
    ph_min, ph_max = phi_bounds

    # Uniform in volume: r^3 is uniform
    u = rng.random(n_seeds)
    r = (r_min**3 + u * (r_max**3 - r_min**3)) ** (1.0 / 3.0)

    # Uniform in solid angle: cos(theta) is uniform
    cos_th_min = np.cos(th_min)
    cos_th_max = np.cos(th_max)
    cos_th = rng.uniform(cos_th_max, cos_th_min, n_seeds)
    theta = np.arccos(cos_th)

    phi = rng.uniform(ph_min, ph_max, n_seeds)

    x, y, z = sph_to_cart(r, theta, phi)

    return pv.PolyData(np.column_stack([x, y, z]))


def random_streamlines(
    grid,
    vectors="B",
    n_seeds=1000,
    r_bounds=None,
    theta_bounds=None,
    phi_bounds=None,
    max_length=1000.0,
    rng=None,
):
    """
    Streamlines from random seed points in the spherical domain.

    vectors can be "B" or "J".
    """

    mesh = structured_grid_from_centers(grid)

    seeds = random_seed_points_in_spherical_shell(
        grid,
        n_seeds=n_seeds,
        r_bounds=r_bounds,
        theta_bounds=theta_bounds,
        phi_bounds=phi_bounds,
        rng=rng,
    )

    streams = mesh.streamlines_from_source(
        seeds,
        vectors=vectors,
        integration_direction="both",
        max_length=max_length,
        # interpolator_type="cell",
        min_step_length=0.001,
    )

    return streams, seeds, mesh


def make_random_streamlines(grid, current=False, plotter=None):
    vectors = "B"
    if current:
        vectors = "J"
    streams, seeds, mesh = random_streamlines(
        grid,
        vectors=vectors,
        n_seeds=2000,
        rng=42,
    )

    if plotter is None:
        p = pv.Plotter()
    else:
        plotter = p
    p.add_mesh(streams, line_width=1.5)
    p.add_mesh(
        seeds,
        color="white",
        point_size=4,
        render_points_as_spheres=True,
    )
    # p.show()
    return p


def add_mse_frame(plotter, origin=(0, 0, 0), scale=0.5):

    V = np.array([-1, 0, 0])
    B = np.array([0, 1, 0])
    E = -np.cross(V, B)

    v_arrow = pv.Arrow(
        start=origin,
        direction=V,
        scale=scale,
    )

    b_arrow = pv.Arrow(
        start=origin,
        direction=B,
        scale=scale,
    )

    e_arrow = pv.Arrow(
        start=origin,
        direction=E,
        scale=scale,
    )

    plotter.add_mesh(v_arrow, color="white")
    plotter.add_mesh(b_arrow, color="cyan")
    plotter.add_mesh(e_arrow, color="orange")

    plotter.add_point_labels(
        [origin + scale * V],
        ["Vsw"],
        font_size=18,
    )

    plotter.add_point_labels(
        [origin + scale * B],
        ["IMF B"],
        font_size=18,
    )

    plotter.add_point_labels(
        [origin + scale * E],
        ["E=-V×B"],
        font_size=18,
    )


if __name__ == "__main__":
    import numpy as np
    import pyvista as pv

    import mars_currents.config as cfg
    from mars_currents.grid import load_grid

    location = cfg.RESULTS
    pickled_grid = (
        "grid_dr0.04_dth10_dphi18__mag_mse_2014-10-11_2020-08-14_n15_all.pickle"
    )
    grid = load_grid(load_from_file=True, file_path=location / pickled_grid)
    grid.calculate_JdotE()
    grid.calculate_JxB()

    # p = plot_B_glyphs(grid)
    # p = plot_J_glyphs(grid)
    p = plot_J_glyphs_by_JdotE_sign(grid)

    # p = plot_streamlines(grid, current=True)
    # plot_streamlines(grid, plotter=p)
    # p = make_random_streamlines(grid, current=True)
    add_mse_frame(p)
    p.show()
